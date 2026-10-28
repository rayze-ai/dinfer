//! B30（loop kaon3-decode-opt D149）：两处三合一 vs 原三步，逐位。
//! a) `fused_add_rmsnorm_add_scale_bf16_n(h2, norm, w, x, s)` vs `add_inplace(h2→norm) → rmsnorm_add(norm,w→x) → scale(x,s)`
//! b) `rmsnorm_scale2_bf16(x, scale, rs)`（mode 1）vs `rmsnorm_bf16_noweight → scale_bf16 → mul_bf16`
//! hidden=2816（Kaon-V3），tokens 1 / 8 / 400。`RMSNORM_MODE` 默认 1。
//! `cargo test --release -p dinfer-kernels --test layer_tail_fused_parity -- --ignored --nocapture`
use dinfer_kernels::{bf16, Device};

struct Rng(u64);
impl Rng {
    fn uni(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; (self.0 >> 40) as f32 / (1u64 << 24) as f32 }
    fn normal(&mut self) -> f32 { let (u1, u2) = (self.uni().max(1e-7), self.uni()); (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() }
}

#[test]
#[ignore]
fn layer_tail_fused_parity() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let (h, eps) = (2816usize, 1e-6f32);
    let mut fails = Vec::new();
    for tokens in [1usize, 8, 400] {
        let mut rng = Rng(0xb30 + tokens as u64);
        let rb = |x: f32| bf16::from_f32(x).to_f32();
        let n = tokens * h;
        let x: Vec<f32> = (0..n).map(|_| rb(4.0 * rng.normal())).collect();
        let h2: Vec<f32> = (0..n).map(|_| rb(2.0 * rng.normal())).collect();
        let nm: Vec<f32> = (0..n).map(|_| rb(2.0 * rng.normal())).collect();
        let w: Vec<f32> = (0..h).map(|_| rb(1.0 + 0.3 * rng.normal())).collect();
        let sc: Vec<f32> = (0..h).map(|_| rb(0.5 + 0.2 * rng.normal())).collect();
        let up = |v: &[f32]| st.htod_copy(v.iter().map(|&f| bf16::from_f32(f)).collect::<Vec<_>>()).unwrap();
        let dw = up(&w); let dsc = up(&sc);
        for s in [1.0f32, rb(0.7071)] {
            // a) 参照三步
            let dh2 = up(&h2);
            let mut r_norm = up(&nm); let mut r_x = up(&x);
            dinfer_kernels::add_inplace_bf16_n(&st, &dh2, &mut r_norm, n).unwrap();
            dinfer_kernels::rmsnorm_add_bf16_n(&st, &r_norm, &dw, &mut r_x, tokens, h, eps).unwrap();
            if s != 1.0 { dinfer_kernels::scale_bf16_n(&st, &mut r_x, n, s).unwrap(); }
            // a) 融合
            let mut f_norm = up(&nm); let mut f_x = up(&x);
            dinfer_kernels::fused_add_rmsnorm_add_scale_bf16_n(&st, &dh2, &mut f_norm, &dw, &mut f_x, tokens, h, eps, s).unwrap();
            st.synchronize().unwrap();
            let cmp = |a: &dinfer_kernels::cudarc::driver::CudaSlice<bf16>, b: &dinfer_kernels::cudarc::driver::CudaSlice<bf16>| -> (usize, f32) {
                let p = st.dtoh_sync_copy(a).unwrap(); let q = st.dtoh_sync_copy(b).unwrap();
                let mut nd = 0; let mut mx = 0f32;
                for i in 0..p.len() { if p[i].to_bits() != q[i].to_bits() { nd += 1; } mx = mx.max((p[i].to_f32() - q[i].to_f32()).abs()); }
                (nd, mx)
            };
            let (n1, m1) = cmp(&r_norm, &f_norm); let (n2, m2) = cmp(&r_x, &f_x);
            println!("tail tokens={tokens} s={s}: norm nd={n1} mx={m1:.4} | x nd={n2} mx={m2:.4} => {}", if n1 + n2 == 0 { "ok" } else { "FAIL" });
            if n1 + n2 != 0 { fails.push(format!("tail tokens={tokens} s={s} norm{n1} x{n2}")); }
        }
        // b) router 预处理
        let rs = f32::from(bf16::from_f32((h as f32).powf(-0.5)));
        let dx = up(&x);
        let mut r_out = st.alloc_zeros::<bf16>(n).unwrap();
        dinfer_kernels::rmsnorm_bf16_noweight(&st, &dx, &mut r_out, tokens, h, eps).unwrap();
        dinfer_kernels::scale_bf16_n(&st, &mut r_out, n, rs).unwrap();
        dinfer_kernels::mul_bf16_n(&st, &mut r_out, &dsc, tokens, h).unwrap();
        let mut f_out = st.alloc_zeros::<bf16>(n).unwrap();
        dinfer_kernels::rmsnorm_scale2_bf16(&st, &dx, &dsc, &mut f_out, tokens, h, eps, rs).unwrap();
        st.synchronize().unwrap();
        let p = st.dtoh_sync_copy(&r_out).unwrap(); let q = st.dtoh_sync_copy(&f_out).unwrap();
        let mut nd = 0; let mut mx = 0f32;
        for i in 0..n { if p[i].to_bits() != q[i].to_bits() { nd += 1; } mx = mx.max((p[i].to_f32() - q[i].to_f32()).abs()); }
        println!("router tokens={tokens}: nd={nd} mx={mx:.4} => {}", if nd == 0 { "ok" } else { "FAIL" });
        if nd != 0 { fails.push(format!("router tokens={tokens} nd={nd}")); }
    }
    assert!(fails.is_empty(), "失败: {fails:?}");
}
