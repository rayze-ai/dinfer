//! bf16 GEMM 的 algo 标定：找出与 vLLM（PyTorch `F.linear` → cuBLAS）逐位一致的 cuBLASLt 候选。
//!
//! 背景（`test/kaon3/2026-10-23/D-16-REBASE.md` §46）：L2 的 `router_in` 逐位相同，
//! `router_logits` 却差 4 个 1-ulp。router 走 `QLinear::Bf16` → cuBLASLt 启发式 `h[0]`；
//! vLLM 走 `F.linear`（cuBLAS）。与 fp8 同类：**同一份输入，不同 algo，差 1-ulp**。
//!
//! 数据由 `scripts/dev/kaon3_d17_router.py` 生成到 `/w/d17b/router2/`：
//! `x_f32.bin`(24×2816，进 proj 前的真实输入)、`w_bf16.bin`(128×2816)、`out_ref_f32.bin`(24×128)。
//!
//! 跑：`D17_ROUTER=/w/d17b/router2 CUDA_ARCHS=120 cargo test --release -p dinfer-kernels \
//!        --test bf16_algo_calib -- --ignored --nocapture`

use dinfer_kernels::{bf16, Device, Gemm};

fn read_f32_le(p: &str) -> Vec<f32> {
    let b = std::fs::read(p).expect("读文件");
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[test]
#[ignore = "需要 GPU + router 标定数据"]
fn calibrate_bf16_algo() {
    let dir = std::env::var("D17_ROUTER").unwrap_or_else(|_| "/w/d17b/router2".to_string());
    let (m, n, k) = (24usize, 128usize, 2816usize);

    let x_f32 = read_f32_le(&format!("{dir}/x_f32.bin"));
    let ref_out = read_f32_le(&format!("{dir}/out_ref_f32.bin"));
    let w_bits: Vec<u16> = {
        let b = std::fs::read(format!("{dir}/w_bf16.bin")).expect("w_bf16.bin");
        b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
    };
    assert_eq!(x_f32.len(), m * k);
    assert_eq!(ref_out.len(), m * n);
    assert_eq!(w_bits.len(), n * k, "w 是 [n,k]");

    let dev = Device::new(0).expect("cuda");
    let st = dev.default_stream();
    let gemm = Gemm::new(&st).expect("gemm");

    let a: Vec<bf16> = x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let w: Vec<bf16> = w_bits.iter().map(|&b| bf16::from_bits(b)).collect();
    let da = st.htod_copy(a).expect("a");
    let dw = st.htod_copy(w).expect("w");

    let cnt = gemm.bf16_algo_count(m, n, k, false).expect("count");
    println!("bf16 候选 {cnt} 个（m={m} n={n} k={k}）");
    for idx in 0..cnt {
        let mut o = st.alloc_zeros::<bf16>(m * n).expect("alloc");
        if gemm.forward_bf16_algo(&da, &dw, None, &mut o, m, n, k, idx).is_err() {
            continue;
        }
        let g = st.dtoh_sync_copy(&o).expect("dl");
        let d = (0..m * n).filter(|&i| g[i].to_f32() != ref_out[i]).count();
        let desc = gemm.bf16_algo_desc(m, n, k, idx).unwrap_or_default();
        let mark = if d == 0 { " ✓" } else { "" };
        println!("  [{idx}] diff {d:<6} {desc}{mark}");
    }
    println!("DONE");
}
