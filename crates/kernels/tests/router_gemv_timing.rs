//! R1 微基准（nsys 外，D174）：router_gemv_bf16 vs cuBLASLt forward_bf16，M=1/2/4/8，E=128,H=2816。
//! cargo test -p dinfer-kernels --test router_gemv_timing -- --ignored --nocapture
use dinfer_kernels as k;
use k::{bf16, ops, Device, Gemm};

#[test]
#[ignore]
fn router_gemv_vs_cublas() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let gemm = Gemm::new(&st).expect("gemm");
    let (e, h) = (128usize, 2816usize);
    let w: Vec<bf16> = (0..e * h).map(|i| bf16::from_f32(((i * 7919) % 1000) as f32 / 1e4 - 0.05)).collect();
    let x: Vec<bf16> = (0..16 * h).map(|i| bf16::from_f32(((i * 104729) % 1000) as f32 / 500.0 - 1.0)).collect();
    let dw = st.htod_copy(w).unwrap();
    let dx = st.htod_copy(x).unwrap();
    let iters = 500usize;
    for &m in &[1usize, 2, 4, 8, 16] {
        let mut o1 = st.alloc_zeros::<bf16>(m * e).unwrap();
        let mut o2 = st.alloc_zeros::<bf16>(m * e).unwrap();
        for _ in 0..20 { ops::router_gemv_bf16(&st, &dx, &dw, &mut o1, m, e, h).unwrap(); }
        st.synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..iters { ops::router_gemv_bf16(&st, &dx, &dw, &mut o1, m, e, h).unwrap(); }
        st.synchronize().unwrap();
        let us1 = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        for _ in 0..20 { gemm.forward_bf16(&dx, &dw, None, &mut o2, m, e, h).unwrap(); }
        st.synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..iters { gemm.forward_bf16(&dx, &dw, None, &mut o2, m, e, h).unwrap(); }
        st.synchronize().unwrap();
        let us2 = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let a = st.dtoh_sync_copy(&o1).unwrap();
        let b = st.dtoh_sync_copy(&o2).unwrap();
        let worst = a.iter().zip(&b).map(|(p, q)| (p.to_bits() as i32 - q.to_bits() as i32).abs()).max().unwrap();
        eprintln!("router_gemv m={m}: gemv {us1:.2} us  cublas {us2:.2} us  worst|ulp| vs cublas={worst}");
    }
}
