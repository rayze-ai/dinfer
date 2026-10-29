//! B45（loop kaon3-decode-opt D192）：`moe_combine_norm_bf16`（combine + post_ff_norm_2 一次）vs
//! `moe_combine_bf16 → rmsnorm_bf16_layout(Block256)` 两步，y 与 h2 均逐位同。H=2816、K=8、tokens 1/8/64；mode 1 与 0；nmode = 1（默认口径）与 2。
//! cargo test --release -p dinfer-kernels --test moe_combine_norm_parity -- --ignored --nocapture
use dinfer_kernels::{arch::RmsnormLayout, bf16, ops, Device};

struct Rng(u64);
impl Rng {
    fn uni(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; (self.0 >> 40) as f32 / (1u64 << 24) as f32 }
    fn normal(&mut self) -> f32 { let (u, v) = (self.uni().max(1e-7), self.uni()); (-2.0 * u.ln()).sqrt() * (6.2831853 * v).cos() }
}

#[test]
#[ignore]
fn moe_combine_norm_parity() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let (h, k, eps) = (2816usize, 8usize, 1e-6f32);
    let mut fails = Vec::new();
    for tokens in [1usize, 8, 64] {
        for mode in [1i32, 0] {
            let mut rng = Rng(0xb45 + tokens as u64 * 31 + mode as u64);
            let rows = tokens * k;
            let yg: Vec<bf16> = (0..rows * h).map(|_| bf16::from_f32(0.5 * rng.normal())).collect();
            let topk_w: Vec<f32> = (0..rows).map(|_| 0.05 + 0.2 * rng.uni()).collect();
            // inv：每 token 的 k 个位置是 [0, rows) 的一个置换片段
            let mut perm: Vec<i32> = (0..rows as i32).collect();
            for i in (1..rows).rev() { let j = (rng.uni() * (i + 1) as f32) as usize % (i + 1); perm.swap(i, j); }
            let topk_ids: Vec<i32> = (0..rows).map(|r| ((r * 7 + 3) % 128) as i32).collect();
            let w: Vec<bf16> = (0..h).map(|_| bf16::from_f32(1.0 + 0.3 * rng.normal())).collect();
            let dyg = st.htod_copy(yg).unwrap(); let dtw = st.htod_copy(topk_w).unwrap();
            let dinv = st.htod_copy(perm).unwrap(); let dids = st.htod_copy(topk_ids).unwrap(); let dw = st.htod_copy(w).unwrap();
            // 两步
            let mut y1 = st.alloc_zeros::<bf16>(tokens * h).unwrap();
            ops::moe_combine_bf16(&st, &dyg, &dtw, &dinv, None, &dids, 1.0, &mut y1, tokens, k, h, mode).unwrap();
            let mut h1 = st.alloc_zeros::<bf16>(tokens * h).unwrap();
            ops::rmsnorm_bf16_layout(&st, &y1, &dw, &mut h1, tokens, h, eps, Some(RmsnormLayout::Block256)).unwrap();
            // 融合（nmode 取与两步同源的 RMSNORM_MODE 默认 1）
            let nmode = std::env::var("RMSNORM_MODE").ok().and_then(|s| s.parse::<i32>().ok()).unwrap_or(1);
            let mut y2 = st.alloc_zeros::<bf16>(tokens * h).unwrap();
            let mut h2 = st.alloc_zeros::<bf16>(tokens * h).unwrap();
            ops::moe_combine_norm_bf16(&st, &dyg, &dtw, &dinv, None, &dids, 1.0, &mut y2, &dw, &mut h2, tokens, k, h, mode, eps, nmode).unwrap();
            st.synchronize().unwrap();
            let (a1, a2) = (st.dtoh_sync_copy(&y1).unwrap(), st.dtoh_sync_copy(&y2).unwrap());
            let (b1, b2) = (st.dtoh_sync_copy(&h1).unwrap(), st.dtoh_sync_copy(&h2).unwrap());
            let ny = a1.iter().zip(&a2).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
            let nh = b1.iter().zip(&b2).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
            println!("combine+norm tokens={tokens} mode={mode} nmode={nmode}: y 位差 {ny}/{} | h2 位差 {nh}/{} => {}", a1.len(), b1.len(), if ny + nh == 0 { "ok" } else { "FAIL" });
            if ny + nh != 0 { fails.push(format!("tokens={tokens} mode={mode} y{ny} h{nh}")); }
        }
    }
    assert!(fails.is_empty(), "失败: {fails:?}");
}
