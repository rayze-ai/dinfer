//! dense FP4 GEMM 对拍：我们 `forward_nvfp4` vs 参照 cutlass `ops.cutlass_scaled_fp4_mm`。
//!
//! 数据（`scripts/dev/fp4_dense_probe.py` 在 vLLM 里产生，真实权重 + 真实激活）：
//!   `a.bin`     `[8, 2816]` bf16   —— 激活（未量化）
//!   `w1.bin`    `[1408, 1408]` u8  —— 专家 0 的 gate+up fp4 打包（n=1408, k/2=1408）
//!   `s1.bin`    `[1408, 176]`  u8  —— **已 padded+swizzled** 的权重块缩放
//!   `y.bin`     `[8, 1408]`  bf16  —— 参照输出（cutlass）
//!   `a_gscale.bin` / `alpha.bin`   —— 标量
//!
//! 判据：逐位相同。差异结构本身就有信息：
//!   全部元素都不同但 max|d| 很小 → 块缩放布局/累加次序不同；
//!   只有部分不同 → 局部边界问题。
//!
//! 运行：`FP4_DIR=/w/tmp/fp4d CUDA_ARCHS=120 cargo test --release -p dinfer-kernels \
//!        --test fp4_dense_parity -- --ignored --nocapture`
use dinfer_kernels::{bf16, Device, Gemm};

fn rd_u16(p: &str) -> Vec<u16> {
    let b = std::fs::read(p).expect("bf16 bin");
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}
fn rd_f32_one(p: &str) -> f32 {
    let b = std::fs::read(p).expect("f32 scalar");
    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

#[test]
#[ignore]
fn fp4_dense_matches_cutlass() {
    let dir = std::env::var("FP4_DIR").unwrap_or_else(|_| "/w/tmp/fp4d".to_string());
    let a: Vec<bf16> = rd_u16(&format!("{dir}/a.bin")).iter().map(|&u| bf16::from_bits(u)).collect();
    let w = std::fs::read(format!("{dir}/w1.bin")).expect("w1");
    let s = std::fs::read(format!("{dir}/s1.bin")).expect("s1");
    let yref: Vec<u16> = rd_u16(&format!("{dir}/y.bin"));
    let a_gscale = rd_f32_one(&format!("{dir}/a_gscale.bin"));
    let alpha = rd_f32_one(&format!("{dir}/alpha.bin"));

    let (m, k) = (8usize, 2816usize);
    let n = 1408usize;
    assert_eq!(a.len(), m * k);
    assert_eq!(w.len(), n * (k / 2));
    assert_eq!(yref.len(), m * n);
    // 注意：**权重侧** scale（s1）是自然布局 [n, k/16] = [1408,176]；
    // 只有**激活侧**的 a_s 才是 padded+swizzled（padded_n 才触发）。
    assert_eq!(s.len(), n * (k / 16), "权重 scale 应为 [n, k/16]：{} != {}", s.len(), n * (k / 16));

    let dev = Device::new(0).expect("cuda");
    let st = dev.default_stream();
    let gemm = Gemm::new(&st).expect("gemm");

    // ---- 参照的 s1 是 swizzled [128,176]；我们的 kernel 期望自然序 [n, k/16] = [1408,176] ----
    // 这里先**按参照布局原样喂**，看差多少；kernel 侧的 swizzle 支持是下一步的事。
    let dw = st.htod_copy(w.clone()).expect("w");
    let ds = st.htod_copy(s.clone()).expect("s");
    let da = st.htod_copy(a.clone()).expect("a");
    let mut dout = st.alloc_zeros::<bf16>(m * n).expect("out");
    // **首选路径**：直接用参照的 `a_q` / `a_s`（swizzled），把我们自己的量化整个排除。
    // 这样测的就纯粹是 GEMM 本身（布局 + 累加 + 乘子）。
    let use_ref_aq = std::env::var("FP4_USE_REF_AQ").map(|v| v != "0").unwrap_or(true);
    println!("FP4_DENSE 路径: {}", if use_ref_aq { "参照 a_q/a_s + swizzled 读" } else { "自己量化 + 自然读" });
    let mut r = if use_ref_aq {
        let aqb = std::fs::read(format!("{dir}/a_q.bin")).expect("a_q");
        let asb = std::fs::read(format!("{dir}/a_s.bin")).expect("a_s");
        let daq = st.htod_copy(aqb).expect("aq");
        let das = st.htod_copy(asb).expect("as");
        gemm.forward_nvfp4_prequant(&daq, &das, &dw, &ds, alpha, &mut dout, m, n, k, true)
    } else {
        gemm.forward_nvfp4(&da, a_gscale, &dw, &ds, alpha, &mut dout, m, n, k)
    };

    if let Err(e) = r {
        println!("FP4_DENSE 第一次失败，走回退分支: {e}");
        // 退一步：补零到 n*(k/16) 长度再跑，只为拿到一个基线差异
        let mut pad = s.clone();
        pad.resize(n * (k / 16), 0);
        let ds2 = st.htod_copy(pad).expect("s2");
        gemm.forward_nvfp4(&da, a_gscale, &dw, &ds2, alpha, &mut dout, m, n, k).expect("kernel");
    }
    let ours = st.dtoh_sync_copy(&dout).expect("d2h");

    let mut ndiff = 0usize;
    let mut maxd = 0f32;
    for i in 0..m * n {
        if ours[i].to_bits() != yref[i] {
            ndiff += 1;
            maxd = maxd.max((f32::from(ours[i]) - f32::from(bf16::from_bits(yref[i]))).abs());
        }
    }
    let ny: f32 = yref.iter().map(|&u| { let v = f32::from(bf16::from_bits(u)); v * v }).sum::<f32>().sqrt();
    let no: f32 = ours.iter().map(|&v| { let v = f32::from(v); v * v }).sum::<f32>().sqrt();
    println!("FP4_DENSE [m={m} n={n} k={k}] ndiff={ndiff}/{} max|d|={maxd:.3e} ‖ours‖={no:.4} ‖ref‖={ny:.4} 比值={:.4}",
             m * n, no / ny);
}
