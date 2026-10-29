//! B33a（loop kaon3-decode-opt）：`rmsnorm_quant_fp8`（mode 1）vs `rmsnorm_bf16_inplace` → `quant_fp8_bf16` 两步，fp8 码逐位同。
//! hidden=2816，tokens 1/8/400，a_scale 若干。`cargo test --release -p dinfer-kernels --test norm_quant_fused_parity -- --ignored --nocapture`
use dinfer_kernels::{bf16, Device};

struct Rng(u64);
impl Rng {
    fn uni(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; (self.0 >> 40) as f32 / (1u64 << 24) as f32 }
    fn normal(&mut self) -> f32 { let (u1, u2) = (self.uni().max(1e-7), self.uni()); (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() }
}

#[test]
#[ignore]
fn norm_quant_fused_parity() {
    assert_eq!(std::env::var("RMSNORM_MODE").ok().as_deref().unwrap_or("1"), "1", "该测试只定义 mode 1 口径");
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let (h, eps) = (2816usize, 1e-6f32);
    let mut fails = Vec::new();
    for tokens in [1usize, 8, 400] {
        for a_scale in [0.05f32, 0.0123, 1.0] {
            let mut rng = Rng(0xb33a + tokens as u64 + (a_scale * 1000.0) as u64);
            let rb = |x: f32| bf16::from_f32(x).to_f32();
            let n = tokens * h;
            let x: Vec<f32> = (0..n).map(|_| rb(4.0 * rng.normal())).collect();
            let w: Vec<f32> = (0..h).map(|_| rb(1.0 + 0.3 * rng.normal())).collect();
            let up = |v: &[f32]| st.htod_copy(v.iter().map(|&f| bf16::from_f32(f)).collect::<Vec<_>>()).unwrap();
            let dx = up(&x); let dw = up(&w);
            // 两步
            let mut nrm = up(&x);
            dinfer_kernels::rmsnorm_bf16_inplace(&st, &mut nrm, Some(&dw), tokens, h, eps).unwrap();
            let mut q_ref = st.alloc_zeros::<u8>(n).unwrap();
            let rc = unsafe { dinfer_kernels::ffi::quant_fp8_bf16(st.ptr(&nrm) as *const std::ffi::c_void, a_scale, st.ptr_mut(&mut q_ref) as *mut std::ffi::c_void, n as i64, st.raw() as _) };
            assert_eq!(rc, 0);
            // 融合
            let mut q_f = st.alloc_zeros::<u8>(n).unwrap();
            dinfer_kernels::rmsnorm_quant_fp8(&st, &dx, &dw, &mut q_f, tokens, h, eps, a_scale).unwrap();
            st.synchronize().unwrap();
            let (a, b) = (st.dtoh_sync_copy(&q_ref).unwrap(), st.dtoh_sync_copy(&q_f).unwrap());
            let nd = a.iter().zip(&b).filter(|(p, q)| p != q).count();
            println!("norm+quant tokens={tokens} a_scale={a_scale}: fp8 码位差 {nd}/{n} => {}", if nd == 0 { "ok" } else { "FAIL" });
            if nd != 0 { fails.push(format!("tokens={tokens} a_scale={a_scale} nd={nd}")); }
        }
    }
    assert!(fails.is_empty(), "失败: {fails:?}");
}
