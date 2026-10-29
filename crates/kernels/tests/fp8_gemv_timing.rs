//! B31（loop kaon3-decode-opt D148）：fp8 gemv 微基准。形状 = Kaon-V3 decode 三件：dense down N=2816 K=2112、qkv N=3072 K=2816、
//! o_proj 滑窗 N=2816 K=4096（q16×256）；m = 1 / 8。打印 us/次与达到的有效带宽（权重字节 / 时间），与 1.79 TB/s 比。
//! `cargo test --release -p dinfer-kernels --test fp8_gemv_timing -- --ignored --nocapture`
use dinfer_kernels::{bf16, ops, Device};

struct Rng(u64);
impl Rng { fn next(&mut self) -> u8 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; (self.0 >> 40) as u8 } }

#[test]
#[ignore]
fn fp8_gemv_timing() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let iters = 300usize;
    for (name, n, k) in [("dense_down", 2816usize, 2112usize), ("qkv_sliding", 3072, 2816), ("oproj_sliding", 2816, 4096), ("dense_gateup_half", 2112, 2816)] {
        for m in [1usize, 8] {
            let mut rng = Rng(0x51 + n as u64 * 3 + k as u64);
            let wq = |rng: &mut Rng, cnt: usize| -> Vec<u8> { (0..cnt).map(|_| { let e = 4 + (rng.next() % 8); let mant = rng.next() & 7; (e << 3) | mant }).collect() };
            let w = wq(&mut rng, n * k);
            let a = wq(&mut rng, m * k);
            let dw = st.htod_copy(w).unwrap();
            let da = st.htod_copy(a).unwrap();
            let mut dout = st.alloc_zeros::<bf16>(m * n).unwrap();
            for _ in 0..20 { ops::mlp_fp8_gemv(&st, &da, &dw, &mut dout, m, n, k, 1e-3).unwrap(); }
            st.synchronize().unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..iters { ops::mlp_fp8_gemv(&st, &da, &dw, &mut dout, m, n, k, 1e-3).unwrap(); }
            st.synchronize().unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            let bytes = (n * k) as f64;
            println!("FP8_GEMV_TIMING {name} N={n} K={k} m={m}: {us:.2} us  {:.0} GB/s  (下限 {:.2} us @1.79TB/s)", bytes / us / 1e3, bytes / 1.79e12 * 1e6);
        }
    }
}
