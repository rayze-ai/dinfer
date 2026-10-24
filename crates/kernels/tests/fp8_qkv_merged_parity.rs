//! 合并 qkv（N=8192）的 fp8 GEMM 对拍：我们的 `forward_fp8_cublaslt` vs FlashInfer。
//!
//! 目的：验证「q/k/v 合并成一次 N=8192 的 GEMM」能否让我们的输出与参照逐位一致。
//!
//! 数据（`/w/d17qkv/`，由 oracle 容器导出）：
//!   `A_fp8.bin`        `[24, 2816]` e4m3 字节（参照的 `input_norm` 量化结果）
//!   `W_qkv.bin`        `[8192, 2816]` e4m3 字节（q|k|v 三个权重拼接，物理 [N,K]）
//!   `ref_qkv_merged.bin` `[24, 8192]` bf16 —— FlashInfer 合并单次 GEMM 的输出
//!
//! 注意：参照用的是 **q 的 weight_scale**（三个 scale 不同，这里只为验证"形状"的影响，
//! 所以比较的是同一条路径下的形状差异，不涉及 scale 语义）。
//!
//! 运行：`D17QKV=/w/d17qkv CUDA_ARCHS=120 cargo test --release -p dinfer-kernels \
//!        --test fp8_qkv_merged_parity -- --ignored --nocapture`
use dinfer_kernels::{bf16, Device, Gemm};

#[test]
#[ignore]
fn fp8_qkv_merged_matches_flashinfer() {
    let dir = std::env::var("D17QKV").unwrap_or_else(|_| "/w/d17qkv".to_string());
    let (m, n, k) = (24usize, 8192usize, 2816usize);
    let scale_a: f32 = 0.59375;
    let scale_b: f32 = 0.0011247907532379031; // q_proj.weight_scale

    let a = std::fs::read(format!("{dir}/A_fp8.bin")).expect("A_fp8.bin");
    let w = std::fs::read(format!("{dir}/W_qkv.bin")).expect("W_qkv.bin");
    let rb = std::fs::read(format!("{dir}/ref_qkv_merged.bin")).expect("ref");
    assert_eq!(a.len(), m * k, "A 字节数");
    assert_eq!(w.len(), n * k, "W 字节数");
    let r: Vec<bf16> = rb.chunks_exact(2).map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect();
    assert_eq!(r.len(), m * n, "ref 元素数");

    let dev = Device::new(0).expect("cuda");
    let st = dev.default_stream();
    let gemm = Gemm::new(&st).expect("gemm");
    let da = st.htod_copy(a).expect("a");
    let dw = st.htod_copy(w).expect("w");
    let mut dout = st.alloc_zeros::<bf16>(m * n).expect("out");
    gemm.forward_fp8_prequant(&da, scale_a, &dw, scale_b, &mut dout, m, n, k).expect("gemm");
    let got = st.dtoh_sync_copy(&dout).expect("d2h");

    let mut nd = 0usize;
    let mut maxd = 0f32;
    for i in 0..m * n {
        if got[i].to_bits() != r[i].to_bits() {
            nd += 1;
            maxd = maxd.max((got[i].to_f32() - r[i].to_f32()).abs());
        }
    }
    // 分段看：q = [0,4096)、k = [4096,6144)、v = [6144,8192)
    let mut seg = [0usize; 3];
    for i in 0..m * n {
        if got[i].to_bits() != r[i].to_bits() {
            let col = i % n;
            if col < 4096 { seg[0] += 1 } else if col < 6144 { seg[1] += 1 } else { seg[2] += 1 }
        }
    }
    println!("FP8_QKV_MERGED [m={m} n={n} k={k}] ndiff={nd}/{} max|d|={maxd:.3e}", m * n);
    println!("  分段不同：q={} k={} v={}", seg[0], seg[1], seg[2]);
}
