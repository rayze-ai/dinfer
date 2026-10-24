//! 我方 cuBLASLt fp8 GEMM vs FlashInfer bmm_fp8，同一份真实 qkv 输入/权重，逐位比（用户 2026-10-24）。
//! 数据由 `scripts/dev/fi_qkv_ref.py` 造：A_fp8[T,2816] / W_qkv_fp8[8192,2816] / ref_out[T,8192] f32 / meta。
//! 跑：`FIQKV_DIR=/w/fiqkv cargo test --release -p dinfer-kernels --test fp8_qkv_fi_parity -- --ignored --nocapture`
use dinfer_kernels::{bf16, Device, Gemm};

#[test]
#[ignore]
fn qkv_cublaslt_vs_flashinfer() {
    let dir = std::env::var("FIQKV_DIR").unwrap_or_else(|_| "/w/fiqkv".into());
    let meta: Vec<f32> = std::fs::read(format!("{dir}/meta.bin")).unwrap()
        .chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let (a_scale, w_scale, tmax) = (meta[0], meta[1], meta[2] as usize);
    // T 覆盖：ref 造了 tmax 行，这里只取前 t 行（qkv 每行独立）。默认 = tmax。
    let t: usize = std::env::var("T").ok().and_then(|v| v.parse().ok()).unwrap_or(tmax);
    let (n, k) = (8192usize, 2816usize);
    let a_all = std::fs::read(format!("{dir}/A_fp8.bin")).unwrap();   // [tmax,k] u8
    let w = std::fs::read(format!("{dir}/W_qkv_fp8.bin")).unwrap();   // [n,k] u8
    let rb_all: Vec<f32> = std::fs::read(format!("{dir}/ref_out.bin")).unwrap()
        .chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert!(t <= tmax, "T {t} > ref tmax {tmax}");
    let a = a_all[..t * k].to_vec();
    let rb = rb_all[..t * n].to_vec();
    assert_eq!(w.len(), n * k, "W size");

    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let gemm = Gemm::new(&st).unwrap();
    let da = st.htod_copy(a).unwrap();
    let dw = st.htod_copy(w).unwrap();
    let das = st.htod_copy(vec![a_scale]).unwrap();
    let dws = st.htod_copy(vec![w_scale]).unwrap();
    let mut out = st.alloc_zeros::<bf16>(t * n).unwrap();
    gemm.forward_fp8_cublaslt(&da, &das, &dw, &dws, &mut out, t, n, k).unwrap();
    st.synchronize().unwrap();
    let got = st.dtoh_sync_copy(&out).unwrap();

    let (mut nd, mut mx, mut nrm, mut dot) = (0usize, 0f32, 0f64, 0f64);
    let mut seg = [0usize; 3]; // q[0,4096) k[4096,6144) v[6144,8192)
    for i in 0..t * n {
        let g = got[i].to_f32();
        let r = bf16::from_f32(rb[i]).to_f32();
        if bf16::from_f32(g).to_bits() != bf16::from_f32(r).to_bits() {
            nd += 1;
            let col = i % n;
            if col < 4096 { seg[0] += 1 } else if col < 6144 { seg[1] += 1 } else { seg[2] += 1 }
        }
        mx = mx.max((g - r).abs());
        nrm += (r as f64) * (r as f64);
        dot += ((g - r) as f64).powi(2);
    }
    let fa = std::env::var("FP8_FAST_ACCUM").unwrap_or_else(|_| "1(default)".into());
    println!("qkv cuBLASLt vs FlashInfer [FP8_FAST_ACCUM={fa}] T={t} 不同 {nd}/{} max|d|={mx:.4e} rel={:.3e} 分段 q={} k={} v={}",
             t * n, dot.sqrt() / nrm.sqrt().max(1e-30), seg[0], seg[1], seg[2]);
}
