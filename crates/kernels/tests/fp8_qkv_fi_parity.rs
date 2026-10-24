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

    let diff = |got: &[bf16]| -> (usize, f32, [usize; 3]) {
        let (mut nd, mut mx, mut seg) = (0usize, 0f32, [0usize; 3]);
        for i in 0..t * n {
            let g = got[i].to_f32();
            let r = bf16::from_f32(rb[i]).to_f32();
            if bf16::from_f32(g).to_bits() != bf16::from_f32(r).to_bits() {
                nd += 1;
                let col = i % n;
                if col < 4096 { seg[0] += 1 } else if col < 6144 { seg[1] += 1 } else { seg[2] += 1 }
            }
            mx = mx.max((g - r).abs());
        }
        (nd, mx, seg)
    };

    // 默认（TABLE 选的 algo）
    let mut out = st.alloc_zeros::<bf16>(t * n).unwrap();
    gemm.forward_fp8_cublaslt(&da, &das, &dw, &dws, &mut out, t, n, k).unwrap();
    st.synchronize().unwrap();
    let (nd, mx, seg) = diff(&st.dtoh_sync_copy(&out).unwrap());
    println!("[默认 TABLE] T={t} 不同 {nd}/{} max|d|={mx:.4e} q={} k={} v={}", t * n, seg[0], seg[1], seg[2]);

    // ALGO_SWEEP=1：枚举全部候选 algo，找与 FlashInfer 逐位（或最小 diff）的那个
    if std::env::var("ALGO_SWEEP").is_ok() {
        let cnt = gemm.fp8_cublaslt_algo_count(&das, &dws, t, n, k).unwrap();
        println!("  候选 algo 数 = {cnt}");
        let (mut best, mut best_nd) = (usize::MAX, usize::MAX);
        for ai in 0..cnt {
            let mut o = st.alloc_zeros::<bf16>(t * n).unwrap();
            if gemm.forward_fp8_cublaslt_algo(&da, &das, &dw, &dws, &mut o, t, n, k, ai).is_err() { continue; }
            st.synchronize().unwrap();
            let (nd, mx, _) = diff(&st.dtoh_sync_copy(&o).unwrap());
            let desc = gemm.fp8_cublaslt_algo_desc(&das, &dws, t, n, k, ai).unwrap_or_default();
            println!("  algo[{ai:2}] nd={nd:5} max|d|={mx:.3e}  {desc}");
            if nd < best_nd { best_nd = nd; best = ai; }
        }
        println!("  >>> 最佳 algo[{best}] nd={best_nd}  desc={}",
                 gemm.fp8_cublaslt_algo_desc(&das, &dws, t, n, k, best).unwrap_or_default());
    }
}
