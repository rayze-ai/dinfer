//! dense FP4 GEMM 对拍：我们 `forward_nvfp4_prequant` vs **自洽合成参照**。
//!
//! 为什么不用上一轮那份 `/w/tmp/fp4d`：那份的 `y.bin` 与参照自己复算的 corr 只有 0.25，
//! 判据本身是坏的（见 `test/kaon3/2026-10-24/kaon3-fp4dense.md` 第 0 轮）。
//!
//! 本轮数据（`scripts/dev/fp4_dense_synth.py` 产生，vLLM CI 同款配方）：
//!   `a_q.bin`   `[m, k/2]` u8   —— `scaled_fp4_quant` 输出（低半字节 = 偶数列）
//!   `a_s.bin`   `[128, k/16]` u8 —— **padded+swizzled**
//!   `b_q.bin`   `[n, k/2]` u8
//!   `b_s.bin`   `[n, k/16]`  u8   —— **padded+swizzled**（`scaled_fp4_quant` 的第二个返回值）
//!   `b_s_nat.bin` 同上但已反 swizzle 成线性布局（给我们的自然读路径）
//!   `y.bin`     `[m, n]` bf16   —— `cutlass_scaled_fp4_mm` 输出
//!   `a_gs.bin` / `b_gs.bin` / `alpha.bin` 标量
//!
//! 参照侧已自证：cutlass 与参照 dequant+matmul 的 corr = 0.99999869、bf16 只 1/11264 不同；
//! numpy 按其公式独立复算 relerr = 0。所以 `y.bin` 是可信判据。
//!
//! 运行：`FP4_DIR=/w/tmp/fp4s CUDA_ARCHS=120 cargo test --release -p dinfer-kernels \
//!        --test fp4_dense_synth_parity -- --ignored --nocapture`
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
fn fp4_dense_synth_matches_cutlass() {
    let dir = std::env::var("FP4_DIR").unwrap_or_else(|_| "/w/tmp/fp4s".to_string());
    let (m, n, k) = (8usize, 1408usize, 2816usize);

    let aq = std::fs::read(format!("{dir}/a_q.bin")).expect("a_q");
    let asw = std::fs::read(format!("{dir}/a_s.bin")).expect("a_s");
    let bq = std::fs::read(format!("{dir}/b_q.bin")).expect("b_q");
    let bsw = std::fs::read(format!("{dir}/b_s.bin")).expect("b_s");
    let bnat = std::fs::read(format!("{dir}/b_s_nat.bin")).expect("b_s_nat");
    let anat = std::fs::read(format!("{dir}/a_s_nat.bin")).expect("a_s_nat");
    let yref = rd_u16(&format!("{dir}/y.bin"));
    let alpha = rd_f32_one(&format!("{dir}/alpha.bin"));
    let a_gs = rd_f32_one(&format!("{dir}/a_gs.bin"));
    let b_gs = rd_f32_one(&format!("{dir}/b_gs.bin"));

    assert_eq!(aq.len(), m * (k / 2), "a_q 形状");
    assert_eq!(bq.len(), n * (k / 2), "b_q 形状");
    assert_eq!(asw.len(), (m.div_ceil(128)) * 128 * (k / 16), "a_s 应为 padded+swizzled");
    assert_eq!(bsw.len(), n * (k / 16), "b_s 字节数（两种布局相等，不能据此判布局）");
    assert_eq!(bnat.len(), bsw.len(), "b_s_nat 字节数");
    assert_eq!(yref.len(), m * n, "y 形状");

    println!("FP4_SYNTH m={m} n={n} k={k} alpha={alpha:.8e} a_gs={a_gs:.6} b_gs={b_gs:.6}");

    let dev = Device::new(0).expect("cuda");
    let st = dev.default_stream();
    let gemm = Gemm::new(&st).expect("gemm");

    // `forward_nvfp4_prequant` 的 `global` 是**单一**乘子；两侧 gs 已分别折进各自的块缩放
    // （ref 的反量化是 q·(sf/gs)，所以这里把 q 的反量化整体折进乘子不可能 —— 我们的 kernel
    //  只接受一个 global，故按驱动方式传 alpha，两侧 gs 靠 `a_s`/`b_s` 自身携带）。
    let daq = st.htod_copy(aq.clone()).expect("aq");
    let daw = st.htod_copy(asw.clone()).expect("asw");
    let dbq = st.htod_copy(bq.clone()).expect("bq");

    // 注意：`forward_nvfp4_prequant` 里 `swz` 只作用于 **A 侧**；B 侧始终按自然 `[n, k/16]` 读。
    // 所以 B 侧一律喂 `*_nat`。A 侧两种布局各跑一次。
    let run = |ascl: &[u8], swz: bool, tag: &str| {
        let das2 = st.htod_copy(ascl.to_vec()).expect("as");
        let ds = st.htod_copy(bnat.clone()).expect("bs");
        let mut dout = st.alloc_zeros::<bf16>(m * n).expect("out");
        if let Err(e) = gemm.forward_nvfp4_prequant(&daq, &das2, &dbq, &ds, alpha, &mut dout, m, n, k, swz) {
            println!("FP4_SYNTH {tag} 失败: {e}");
            return;
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
        let corr = {
            let (mut sa, mut sb, mut sab) = (0f64, 0f64, 0f64);
            let nn = (m * n) as f64;
            let ma: f64 = ours.iter().map(|&v| f32::from(v) as f64).sum::<f64>() / nn;
            let mb: f64 = yref.iter().map(|&u| f32::from(bf16::from_bits(u)) as f64).sum::<f64>() / nn;
            for i in 0..m * n {
                let x = f32::from(ours[i]) as f64 - ma;
                let y = f32::from(bf16::from_bits(yref[i])) as f64 - mb;
                sa += x * x; sb += y * y; sab += x * y;
            }
            sab / (sa.sqrt() * sb.sqrt())
        };
        println!("FP4_SYNTH {:<24} ndiff={}/{} max|d|={:.3e} ‖ours‖/‖ref‖={:.6} corr={:.8}",
                  tag, ndiff, m * n, maxd, no / ny, corr);
    };

    run(&asw, true, "A=swz  B=nat");
    run(&anat, false, "A=nat  B=nat");
    let _ = (&bsw, &anat);

    // ---- 关键：走我们**自己的量化器**（`forward_nvfp4` 吃 bf16 输入）----
    // 上面两组喂的是参照的 a_q/a_s（prequant 路径），完全绕开了 `quantize_a_kernel`。
    // 端到端走的是 `forward_nvfp4`，差异很可能在这一段。
    {
        let a: Vec<bf16> = rd_u16(&format!("{dir}/a.bin")).iter().map(|&u| bf16::from_bits(u)).collect();
        assert_eq!(a.len(), m * k, "a 形状");
        let da = st.htod_copy(a).expect("a");
        let ds = st.htod_copy(bnat.clone()).expect("bs");
        let mut dout = st.alloc_zeros::<bf16>(m * n).expect("out");
        match gemm.forward_nvfp4(&da, a_gs, &dbq, &ds, alpha, &mut dout, m, n, k) {
            Err(e) => println!("FP4_SYNTH {:<24} 失败: {e}", "self-quant"),
            Ok(()) => {
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
                println!("FP4_SYNTH {:<24} ndiff={}/{} max|d|={:.3e} ‖ours‖/‖ref‖={:.6}",
                         "self-quant", ndiff, m * n, maxd, no / ny);
            }
        }
    }
}
