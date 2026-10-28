//! B10：`moe_route_bucket_small`（融合）vs 旧链 `moe_route_softmax_f32` + `moe_bucket` + `moe_row2expert`。
//! 判据：topk_ids / topk_w（位）/ offsets / tile_off / row2e 逐位同；sorted_tok/sorted_k 每个专家区间作为 (tok,k) 集合相同。
//! 场景：E=128 K=8，tokens ∈ {1, 7, 64}；含 per_expert_scale 与无；logits 含并列值。
//! 运行：`cargo test --release -p dinfer-kernels --test moe_route_fused_parity -- --ignored --nocapture`
use dinfer_kernels::{bf16, ops, Device};

struct Rng(u64);
impl Rng {
    fn uni(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
}

#[test]
#[ignore]
fn moe_route_fused_parity() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let (e, k) = (128usize, 8usize);
    let mut fails = Vec::new();
    for (tokens, use_pes) in [(1usize, false), (7, true), (64, true), (64, false), (33, true)] {
        let mut rng = Rng(0xb10 + tokens as u64 * 17 + use_pes as u64);
        // logits：正态近似 ×3，并人为制造并列（每 token 把两个位置设成同值）
        let mut lg: Vec<f32> = (0..tokens * e).map(|_| (rng.uni() + rng.uni() + rng.uni() + rng.uni() - 2.0) * 3.0).collect();
        for t in 0..tokens { let a = (rng.uni() * e as f32) as usize % e; let b = (a + 1) % e; lg[t * e + b] = lg[t * e + a]; }
        let pes: Vec<bf16> = (0..e).map(|_| bf16::from_f32(0.5 + rng.uni())).collect();
        let d_lg = st.htod_copy(lg.clone()).unwrap();
        let d_pes = st.htod_copy(pes.clone()).unwrap();
        let pes_opt = if use_pes { Some(&d_pes) } else { None };
        // 旧链
        let mut ids_a = st.alloc_zeros::<i32>(tokens * k).unwrap();
        let mut w_a = st.alloc_zeros::<f32>(tokens * k).unwrap();
        let mut b_a = ops::MoeBuckets::new(&st, e, 64, k).unwrap();
        let mut r2e_a = st.alloc_zeros::<i32>(tokens * k).unwrap();
        ops::moe_route_softmax_f32(&st, &d_lg, pes_opt, &mut ids_a, &mut w_a, tokens, e, k).unwrap();
        ops::moe_bucket(&st, &ids_a, &mut b_a, tokens, k).unwrap();
        ops::moe_row2expert(&st, &b_a.offsets, &mut r2e_a, e, tokens * k).unwrap();
        // 融合
        let mut ids_b = st.alloc_zeros::<i32>(tokens * k).unwrap();
        let mut w_b = st.alloc_zeros::<f32>(tokens * k).unwrap();
        let mut b_b = ops::MoeBuckets::new(&st, e, 64, k).unwrap();
        let mut r2e_b = st.alloc_zeros::<i32>(tokens * k).unwrap();
        ops::moe_route_bucket_small(&st, &d_lg, pes_opt, &mut ids_b, &mut w_b, &mut b_b, &mut r2e_b, tokens, k).unwrap();
        st.synchronize().unwrap();
        let g = |x: &dinfer_kernels::cudarc::driver::CudaSlice<i32>| st.dtoh_sync_copy(x).unwrap();
        let (ia, ib) = (g(&ids_a), g(&ids_b));
        let (wa, wb) = (st.dtoh_sync_copy(&w_a).unwrap(), st.dtoh_sync_copy(&w_b).unwrap());
        let (oa, ob) = (g(&b_a.offsets), g(&b_b.offsets));
        let (ta, tb) = (g(&b_a.tile_off), g(&b_b.tile_off));
        let (ca, cb) = (g(&b_a.counts), g(&b_b.counts));
        let (cua, cub) = (g(&b_a.cursor), g(&b_b.cursor));
        let (ra, rb) = (g(&r2e_a), g(&r2e_b));
        let (sta, stb, ska, skb) = (g(&b_a.sorted_tok), g(&b_b.sorted_tok), g(&b_a.sorted_k), g(&b_b.sorted_k));
        let n = tokens * k;
        let ids_eq = ia[..n] == ib[..n];
        let w_bits = wa[..n].iter().zip(&wb[..n]).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        let off_eq = oa[..=e] == ob[..=e] && ta[..=e] == tb[..=e] && ca[..e] == cb[..e] && cua[..e] == cub[..e];
        let r2e_eq = ra[..n] == rb[..n];
        let mut sets_eq = true;
        for ex in 0..e {
            let (lo, hi) = (oa[ex] as usize, oa[ex + 1] as usize);
            let mut sa: Vec<(i32, i32)> = (lo..hi).map(|i| (sta[i], ska[i])).collect();
            let mut sb: Vec<(i32, i32)> = (lo..hi).map(|i| (stb[i], skb[i])).collect();
            sa.sort(); sb.sort();
            if sa != sb { sets_eq = false; }
        }
        let ok = ids_eq && w_bits == 0 && off_eq && r2e_eq && sets_eq;
        println!("tokens={tokens} pes={use_pes}: ids {} | w 位差 {w_bits}/{n} | offsets/tile_off/counts/cursor {} | row2e {} | sorted 区间集合 {} => {}",
            ids_eq, off_eq, r2e_eq, sets_eq, if ok { "ok" } else { "FAIL" });
        if !ok { fails.push(format!("tokens={tokens} pes={use_pes}")); }
    }
    assert!(fails.is_empty(), "失败: {fails:?}");
}
