//! B34①（loop kaon3-decode-opt）：`moe_rmsnorm_grouped_quant`（norm + grouped NVFP4 量化一次）vs `rmsnorm_bf16` → `moe_grouped_quant` 两步，
//! aq / a_s 字节逐位同。K=2816（Kaon gate/up 的 K）、E=128、tokens 1/8/64 × top-8 行（sorted_tok 乱序）。mode 默认 1。
//! `cargo test --release -p dinfer-kernels --test moe_norm_quant_fused_parity -- --ignored --nocapture`
use dinfer_kernels::{bf16, ops, Device};

struct Rng(u64);
impl Rng {
    fn uni(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; (self.0 >> 40) as f32 / (1u64 << 24) as f32 }
    fn normal(&mut self) -> f32 { let (u1, u2) = (self.uni().max(1e-7), self.uni()); (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() }
}

#[test]
#[ignore]
fn moe_norm_quant_fused_parity() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let (h, e, k, eps) = (2816usize, 128usize, 8usize, 1e-6f32);
    let mut fails = Vec::new();
    for tokens in [1usize, 8, 64] {
        let mut rng = Rng(0xb34 + tokens as u64);
        let rb = |x: f32| bf16::from_f32(x).to_f32();
        let x: Vec<f32> = (0..tokens * h).map(|_| rb(3.0 * rng.normal())).collect();
        let w: Vec<f32> = (0..h).map(|_| rb(1.0 + 0.3 * rng.normal())).collect();
        let rows = tokens * k;
        let mut sorted_tok: Vec<i32> = (0..rows).map(|r| (r % tokens) as i32).collect();
        for i in (1..rows).rev() { let j = (rng.uni() * (i + 1) as f32) as usize % (i + 1); sorted_tok.swap(i, j); }
        let row2e: Vec<i32> = (0..rows).map(|r| ((r * 7 + 3) % e) as i32).collect();
        let agscale: Vec<f32> = (0..e).map(|_| 0.5 + rng.uni()).collect();
        let up = |v: &[f32]| st.htod_copy(v.iter().map(|&f| bf16::from_f32(f)).collect::<Vec<_>>()).unwrap();
        let dx = up(&x); let dw = up(&w);
        let dst = st.htod_copy(sorted_tok).unwrap(); let dr2e = st.htod_copy(row2e).unwrap(); let dag = st.htod_copy(agscale).unwrap();
        for exact in [false, true] {
            // 两步
            let mut h2 = st.alloc_zeros::<bf16>(tokens * h).unwrap();
            ops::rmsnorm_bf16_n(&st, &dx, &dw, &mut h2, tokens, h, eps).unwrap();
            let mut aq1 = st.alloc_zeros::<u8>(rows * h / 2).unwrap(); let mut as1 = st.alloc_zeros::<u8>(rows * h / 16).unwrap();
            ops::moe_grouped_quant(&st, &h2, &dr2e, Some(&dst), &dag, &mut aq1, &mut as1, rows, h, exact).unwrap();
            // 融合
            let mut aq2 = st.alloc_zeros::<u8>(rows * h / 2).unwrap(); let mut as2 = st.alloc_zeros::<u8>(rows * h / 16).unwrap();
            ops::moe_rmsnorm_grouped_quant(&st, &dx, &dw, &dr2e, Some(&dst), &dag, &mut aq2, &mut as2, rows, h, eps, exact).unwrap();
            st.synchronize().unwrap();
            let (a1, a2) = (st.dtoh_sync_copy(&aq1).unwrap(), st.dtoh_sync_copy(&aq2).unwrap());
            let (s1, s2) = (st.dtoh_sync_copy(&as1).unwrap(), st.dtoh_sync_copy(&as2).unwrap());
            let nq = a1.iter().zip(&a2).filter(|(p, q)| p != q).count();
            let ns = s1.iter().zip(&s2).filter(|(p, q)| p != q).count();
            println!("tokens={tokens} rows={rows} exact_rq={exact}: aq 位差 {nq}/{} | a_s 位差 {ns}/{} => {}", a1.len(), s1.len(), if nq + ns == 0 { "ok" } else { "FAIL" });
            if nq + ns != 0 { fails.push(format!("tokens={tokens} exact={exact} aq{nq} as{ns}")); }
        }
    }
    assert!(fails.is_empty(), "失败: {fails:?}");
}
