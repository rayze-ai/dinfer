//! B23（loop kaon3-decode-opt D148）：`attn_pre_fused_bf16` vs 原 5 步（rmsnorm_bf16_inplace ×3 → rope_span_bf16 → gqa_cache_write_fp8rt_bf16）
//! 逐位：q/k/v 原地结果 + cache 内容。形状 = Kaon-V3 两种层：D=256 q16/kv8 rope_dim 256 span 128；D=512 q16/kv2 rope_dim 128 span 256（proportional）。
//! cache e4m3 与 bf16 两种。`cargo test --release -p dinfer-kernels --test attn_pre_fused_parity -- --ignored --nocapture`
use dinfer_kernels::{bf16, Device, RopeStyle};

struct Rng(u64);
impl Rng {
    fn uni(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; (self.0 >> 40) as f32 / (1u64 << 24) as f32 }
    fn normal(&mut self) -> f32 { let (u1, u2) = (self.uni().max(1e-7), self.uni()); (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() }
}

#[test]
#[ignore]
fn attn_pre_fused_parity() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let (h, bs, eps) = (16usize, 32usize, 1e-6f32);
    let mut fails = Vec::new();
    for (d, nkv, rope_dim, span, tokens) in [(256usize, 8usize, 256usize, 128usize, 13usize), (512, 2, 128, 256, 13), (256, 8, 256, 128, 1), (512, 2, 128, 256, 400)] {
        for kv_fp8 in [true, false] {
            let mut rng = Rng(0xb23 + d as u64 + tokens as u64 + kv_fp8 as u64);
            let rb = |x: f32| bf16::from_f32(x).to_f32();
            let q: Vec<f32> = (0..tokens * h * d).map(|_| rb(3.0 * rng.normal())).collect();
            let k: Vec<f32> = (0..tokens * nkv * d).map(|_| rb(3.0 * rng.normal())).collect();
            let v: Vec<f32> = (0..tokens * nkv * d).map(|_| rb(rng.normal())).collect();
            let qw: Vec<f32> = (0..d).map(|_| rb(1.0 + 0.3 * rng.normal())).collect();
            let kw: Vec<f32> = (0..d).map(|_| rb(1.0 + 0.3 * rng.normal())).collect();
            let max_pos = 4096usize;
            let half = rope_dim / 2;
            let mut table = vec![0f32; max_pos * rope_dim];
            for p in 0..max_pos { for i in 0..half { let f = p as f64 * (10000f64).powf(-2.0 * i as f64 / d as f64); table[p * rope_dim + i] = f.cos() as f32; table[p * rope_dim + half + i] = f.sin() as f32; } }
            let positions: Vec<i32> = (0..tokens).map(|t| (t as i32 * 7 + 3) % 4000).collect();
            let nblk = tokens.div_ceil(bs) + 1;
            let slots: Vec<i64> = (0..tokens).map(|t| ((1 + t / bs) * bs + t % bs) as i64).collect();
            let up = |x: &[f32]| st.htod_copy(x.iter().map(|&f| bf16::from_f32(f)).collect::<Vec<_>>()).unwrap();
            let dqw = up(&qw); let dkw = up(&kw);
            let dcs = st.htod_copy(table.clone()).unwrap();
            let dpos = st.htod_copy(positions.clone()).unwrap();
            let dsl = st.htod_copy(slots.clone()).unwrap();
            let elems = nblk * bs * nkv * 2 * d;
            let celems = if kv_fp8 { elems / 2 } else { elems };
            // ---- 参照：5 步 ----
            let mut q1 = up(&q); let mut k1 = up(&k); let mut v1 = up(&v);
            let mut c1 = st.alloc_zeros::<bf16>(celems).unwrap();
            dinfer_kernels::rmsnorm_bf16_inplace(&st, &mut q1, Some(&dqw), tokens * h, d, eps).unwrap();
            dinfer_kernels::rmsnorm_bf16_inplace(&st, &mut k1, Some(&dkw), tokens * nkv, d, eps).unwrap();
            dinfer_kernels::rmsnorm_bf16_inplace(&st, &mut v1, None, tokens * nkv, d, eps).unwrap();
            dinfer_kernels::rope_span_bf16(&st, &mut q1, Some(&mut k1), &dpos, &dcs, tokens, h, nkv, d, rope_dim, span, RopeStyle::Neox).unwrap();
            dinfer_kernels::gqa_cache_write_fp8rt_bf16(&st, &mut q1, &mut k1, &mut v1, &mut c1, &dsl, tokens, h, nkv, d, bs, kv_fp8).unwrap();
            // ---- 融合：1 步 ----
            let mut q2 = up(&q); let mut k2 = up(&k); let mut v2 = up(&v);
            let mut c2 = st.alloc_zeros::<bf16>(celems).unwrap();
            dinfer_kernels::attn_pre_fused_bf16(&st, &mut q2, &mut k2, &mut v2, &dqw, &dkw, &dpos, &dcs, &mut c2, &dsl, tokens, h, nkv, d, rope_dim, span, bs, eps, kv_fp8).unwrap();
            st.synchronize().unwrap();
            let cmp = |a: &dinfer_kernels::cudarc::driver::CudaSlice<bf16>, b: &dinfer_kernels::cudarc::driver::CudaSlice<bf16>, n: usize| -> (usize, f32) {
                let x = st.dtoh_sync_copy(a).unwrap(); let y = st.dtoh_sync_copy(b).unwrap();
                let mut nd = 0; let mut mx = 0f32;
                for i in 0..n { if x[i].to_bits() != y[i].to_bits() { nd += 1; } mx = mx.max((x[i].to_f32() - y[i].to_f32()).abs()); }
                (nd, mx)
            };
            let (nq, mq) = cmp(&q1, &q2, tokens * h * d);
            let (nk, mk) = cmp(&k1, &k2, tokens * nkv * d);
            let (nv, mv) = cmp(&v1, &v2, tokens * nkv * d);
            let (nc, mc) = cmp(&c1, &c2, celems);
            println!("D={d} nkv={nkv} rope={rope_dim}/span{span} tokens={tokens} kv_fp8={kv_fp8}: q nd={nq} mx={mq:.4} | k nd={nk} mx={mk:.4} | v nd={nv} mx={mv:.4} | cache nd={nc} mx={mc:.4} => {}",
                if nq + nk + nv + nc == 0 { "ok" } else { "FAIL" });
            if nq + nk + nv + nc != 0 { fails.push(format!("D={d} tokens={tokens} fp8={kv_fp8}: q{nq} k{nk} v{nv} cache{nc}")); }
        }
    }
    assert!(fails.is_empty(), "失败: {fails:?}");
}
