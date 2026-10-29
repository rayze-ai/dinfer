//! B34③（loop kaon3-decode-opt D175）：`moe_gelu_grouped_quant`（gelu·mul mode 1 + grouped NVFP4 量化一次）vs
//! `gelu_tanh_mul_round2_bf16` → `moe_grouped_quant` 两步，aq / a_s 字节逐位同。K=704（Kaon down 的 K）、E=128、rows = tokens×8。
//! `cargo test --release -p dinfer-kernels --test moe_gelu_quant_fused_parity -- --ignored --nocapture`
use dinfer_kernels::{bf16, ops, Device};

struct Rng(u64);
impl Rng {
    fn uni(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; (self.0 >> 40) as f32 / (1u64 << 24) as f32 }
    fn normal(&mut self) -> f32 { let (u, v) = (self.uni().max(1e-7), self.uni()); (-2.0 * u.ln()).sqrt() * (6.2831853 * v).cos() }
}

#[test]
#[ignore]
fn moe_gelu_quant_fused_parity() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let (kd, e, k) = (704usize, 128usize, 8usize);
    let mut fails = Vec::new();
    for tokens in [1usize, 8, 64] {
        let mut rng = Rng(0xb343 + tokens as u64);
        let rows = tokens * k;
        let rb = |x: f32| bf16::from_f32(x);
        let g: Vec<bf16> = (0..rows * kd).map(|_| rb(2.0 * rng.normal())).collect();
        let u: Vec<bf16> = (0..rows * kd).map(|_| rb(2.0 * rng.normal())).collect();
        let row2e: Vec<i32> = (0..rows).map(|r| ((r * 7 + 3) % e) as i32).collect();
        let agscale: Vec<f32> = (0..e).map(|_| 0.5 + rng.uni()).collect();
        let dg = st.htod_copy(g).unwrap(); let du = st.htod_copy(u).unwrap();
        let dr2e = st.htod_copy(row2e).unwrap(); let dag = st.htod_copy(agscale).unwrap();
        for exact in [false, true] {
            let mut y = st.alloc_zeros::<bf16>(rows * kd).unwrap();
            ops::gelu_tanh_mul_round2_bf16(&st, &dg, &du, &mut y, rows, kd).unwrap();
            let mut aq1 = st.alloc_zeros::<u8>(rows * kd / 2).unwrap(); let mut as1 = st.alloc_zeros::<u8>(rows * kd / 16).unwrap();
            ops::moe_grouped_quant(&st, &y, &dr2e, None, &dag, &mut aq1, &mut as1, rows, kd, exact).unwrap();
            let mut aq2 = st.alloc_zeros::<u8>(rows * kd / 2).unwrap(); let mut as2 = st.alloc_zeros::<u8>(rows * kd / 16).unwrap();
            ops::moe_gelu_grouped_quant(&st, &dg, &du, &dr2e, &dag, &mut aq2, &mut as2, rows, kd, exact).unwrap();
            st.synchronize().unwrap();
            let (a1, a2) = (st.dtoh_sync_copy(&aq1).unwrap(), st.dtoh_sync_copy(&aq2).unwrap());
            let (s1, s2) = (st.dtoh_sync_copy(&as1).unwrap(), st.dtoh_sync_copy(&as2).unwrap());
            let nq = a1.iter().zip(&a2).filter(|(p, q)| p != q).count();
            let ns = s1.iter().zip(&s2).filter(|(p, q)| p != q).count();
            println!("gelu+quant tokens={tokens} rows={rows} exact_rq={exact}: aq 位差 {nq}/{} | a_s 位差 {ns}/{} => {}", a1.len(), s1.len(), if nq + ns == 0 { "ok" } else { "FAIL" });
            if nq + ns != 0 { fails.push(format!("tokens={tokens} exact={exact} aq{nq} as{ns}")); }
        }
    }
    assert!(fails.is_empty(), "失败: {fails:?}");
}
