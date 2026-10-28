//! 纯 decode split-K attention（`gqa_decode_win_bf16`：v3 D=256 带滑窗 / 新 D=512 kernel）对 CPU f64 参照，
//! 同时跑通用慢路径（`gqa_paged_win_bf16` cfg −1 口径的同一入口）作第二参照。
//! 场景 = Kaon-V3 decode：16 q 头；滑窗层 D=256 / 8 kv 头 / window 1024；全注意力层 D=512 / 2 kv 头 / window 0。
//! 分数量级：scale=1.0、QK-norm 后 O(10)（amp 1.0）与极尖（amp 2.0）。
//! 运行：`cargo test --release -p dinfer-kernels --test gqa_decode_win_parity -- --ignored --nocapture`
//! loop kaon3-decode-opt B3。
use dinfer_kernels::{bf16, Device};

struct Rng(u64);
impl Rng {
    fn uni(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
    fn normal(&mut self) -> f32 {
        (self.uni() + self.uni() + self.uni() + self.uni() - 2.0) * 1.732
    }
}

/// q: [B, H, D]（每请求 1 行）；k/v: 按请求拼接的 [Σn_b, Hkv, D]；返回 [B, H, D]。
#[allow(clippy::too_many_arguments)]
fn cpu_ref(q: &[f32], k: &[f32], v: &[f32], seq: &[usize], h: usize, hkv: usize, d: usize, window: usize) -> Vec<f32> {
    let b = seq.len();
    let mut out = vec![0f32; b * h * d];
    let mut off = 0;
    for (bi, &n) in seq.iter().enumerate() {
        let j0 = if window > 0 { n.saturating_sub(window) } else { 0 };
        for hh in 0..h {
            let hk = hh / (h / hkv);
            let qi = &q[(bi * h + hh) * d..][..d];
            let s: Vec<f64> = (j0..n)
                .map(|j| qi.iter().zip(&k[((off + j) * hkv + hk) * d..][..d]).map(|(a, b)| *a as f64 * *b as f64).sum())
                .collect();
            let m = s.iter().copied().fold(f64::MIN, f64::max);
            let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
            let z: f64 = e.iter().sum();
            for c in 0..d {
                let acc: f64 = (j0..n).zip(&e).map(|(j, p)| p * v[((off + j) * hkv + hk) * d + c] as f64).sum();
                out[(bi * h + hh) * d + c] = (acc / z) as f32;
            }
        }
        off += n;
    }
    out
}

#[test]
#[ignore]
fn gqa_decode_win_parity() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let (h, bs) = (16usize, 32usize);
    let mut fails = Vec::new();
    // (D, Hkv, window, seq_lens, amp)
    let cases: Vec<(usize, usize, usize, Vec<usize>, f32)> = vec![
        (256, 8, 1024, vec![1600], 1.0),
        (256, 8, 1024, vec![1600, 70, 1, 1024, 1025], 2.0),
        (256, 8, 0, vec![300, 1], 1.0),
        (512, 2, 0, vec![1600], 1.0),
        (512, 2, 0, vec![1600, 70, 1, 33, 64], 2.0),
        (512, 2, 1024, vec![1600, 900], 1.0),
    ];
    for (d, hkv, window, seq, amp) in cases {
        let b = seq.len();
        let total: usize = seq.iter().sum();
        let mut rng = Rng(0x5eed_0000 + d as u64 * 31 + window as u64 + total as u64);
        let rb = |x: f32| bf16::from_f32(x).to_f32();
        let q: Vec<f32> = (0..b * h * d).map(|_| rb(amp * rng.normal())).collect();
        let k: Vec<f32> = (0..total * hkv * d).map(|_| rb(amp * rng.normal())).collect();
        let v: Vec<f32> = (0..total * hkv * d).map(|_| rb(rng.normal())).collect();
        let want = cpu_ref(&q, &k, &v, &seq, h, hkv, d, window);
        // paged cache：每请求连续分配 block，slot = blk*bs + off
        let max_blocks = seq.iter().map(|n| n.div_ceil(bs)).max().unwrap();
        let num_blocks: usize = seq.iter().map(|n| n.div_ceil(bs)).sum::<usize>() + 1;
        let mut bt = vec![0i32; b * max_blocks];
        let mut slots = Vec::with_capacity(total);
        let mut next_blk = 1i32; // block 0 留空，暴露越界
        for (bi, &n) in seq.iter().enumerate() {
            for (i, blk) in (0..n.div_ceil(bs)).enumerate() {
                bt[bi * max_blocks + i] = next_blk + blk as i32;
            }
            for t in 0..n {
                slots.push(((next_blk as usize + t / bs) * bs + t % bs) as i64);
            }
            next_blk += n.div_ceil(bs) as i32;
        }
        let up = |x: &[f32]| st.htod_copy(x.iter().map(|&f| bf16::from_f32(f)).collect::<Vec<_>>()).unwrap();
        let (dq, dk, dv) = (up(&q), up(&k), up(&v));
        let dsl = st.htod_copy(slots.clone()).unwrap();
        let mut cache = st.alloc_zeros::<bf16>(num_blocks * bs * hkv * 2 * d).unwrap();
        dinfer_kernels::gqa_cache_write_bf16(&st, &dk, &dv, &mut cache, &dsl, total, hkv, d, bs, false).unwrap();
        let dbt = st.htod_copy(bt).unwrap();
        let dseq = st.htod_copy(seq.iter().map(|&n| n as i32).collect::<Vec<_>>()).unwrap();
        let cu: Vec<i32> = (0..=b as i32).collect();
        let dcu = st.htod_copy(cu).unwrap();
        let mut scratch = st.alloc_zeros::<f32>(dinfer_kernels::gqa_decode_scratch_len(b, h, d)).unwrap();
        for path in ["decode_win", "paged_slow"] {
            let mut out = st.alloc_zeros::<bf16>(b * h * d).unwrap();
            let r = if path == "decode_win" {
                dinfer_kernels::gqa_decode_win_bf16(
                    &st, &dq, &cache, &dbt, &dseq, &mut out, &mut scratch, b, h, hkv, d, bs, max_blocks, 1.0, 0, None, window, false,
                )
            } else {
                dinfer_kernels::gqa_paged_win_bf16(
                    &st, &dq, &cache, &dbt, &dseq, &dcu, &mut out, b, 1, h, hkv, d, bs, max_blocks, 1.0, 0, false, window, false, false,
                )
            };
            st.synchronize().unwrap();
            let got = st.dtoh_sync_copy(&out).unwrap();
            let (mut worst, mut bad) = (0f32, 0usize);
            for (g, w) in got.iter().zip(&want) {
                let e = (g.to_f32() - w).abs();
                worst = worst.max(e);
                bad += (e > 0.03 + 0.02 * w.abs()) as usize;
            }
            println!(
                "D={d} Hkv={hkv} win={window:>4} seq={seq:?} amp={amp} {path:>10}: rc={} 最大误差 {worst:.4} 超差 {bad}/{}",
                r.is_ok(),
                got.len()
            );
            if r.is_err() || bad > 0 {
                fails.push(format!("D={d} win={window} seq={seq:?} amp={amp} {path}: rc={r:?} bad={bad} worst={worst}"));
            }
        }
    }
    assert!(fails.is_empty(), "失败：\n{}", fails.join("\n"));
}
