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

/// B15：KV cache 真存 e4m3 vs bf16 cache（元素为往返值）—— 同一 q/k/v（k/v 先 fp8 往返），decode v3/d512 与 paged 慢路径
/// 两种 cache 的输出**逐位同**（读路径展开 = 写路径往返的同一 cvt）。场景同上（D=256 窗 1024 / D=512）。
#[test]
#[ignore]
fn gqa_decode_fp8cache_bitwise() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let (h, bs) = (16usize, 32usize);
    let rt = |x: f32| -> f32 {
        // e4m3 往返（SATFINITE、RNE）的 CPU 口径：用 GPU 侧同一 cvt 更稳 —— 这里借 kernel：先写 bf16 cache 再读回不方便，
        // 直接用 dinfer_kernels 的 fp8_e4m3_roundtrip_bf16_n 在 GPU 上做。
        x
    };
    let _ = rt;
    let mut fails = Vec::new();
    let cases: Vec<(usize, usize, usize, Vec<usize>, f32)> = vec![
        (256, 8, 1024, vec![1600, 70, 1, 1024, 1025], 2.0),
        (512, 2, 0, vec![1600, 70, 1, 33, 64], 2.0),
        (256, 8, 0, vec![300, 1], 1.0),
    ];
    for (d, hkv, window, seq, amp) in cases {
        let b = seq.len();
        let total: usize = seq.iter().sum();
        let mut rng = Rng(0xf8 + d as u64 * 31 + window as u64 + total as u64);
        let rb = |x: f32| bf16::from_f32(x).to_f32();
        let q: Vec<f32> = (0..b * h * d).map(|_| rb(amp * rng.normal())).collect();
        let k: Vec<f32> = (0..total * hkv * d).map(|_| rb(amp * rng.normal())).collect();
        let v: Vec<f32> = (0..total * hkv * d).map(|_| rb(rng.normal())).collect();
        let max_blocks = seq.iter().map(|n| n.div_ceil(bs)).max().unwrap();
        let num_blocks: usize = seq.iter().map(|n| n.div_ceil(bs)).sum::<usize>() + 1;
        let mut bt = vec![0i32; b * max_blocks];
        let mut slots = Vec::with_capacity(total);
        let mut next_blk = 1i32;
        for (i, &n) in seq.iter().enumerate() {
            let nb = n.div_ceil(bs);
            for j in 0..nb { bt[i * max_blocks + j] = next_blk + j as i32; }
            for t in 0..n { slots.push(((next_blk as usize + t / bs) * bs + t % bs) as i64); }
            next_blk += nb as i32;
        }
        let up = |x: &[f32]| st.htod_copy(x.iter().map(|&f| bf16::from_f32(f)).collect::<Vec<_>>()).unwrap();
        let dq = up(&q);
        let mut dk = up(&k);
        let mut dv = up(&v);
        // k/v 先 fp8 往返（kv_fp8 层的写入前处理）
        dinfer_kernels::ops::fp8_e4m3_roundtrip_bf16_n(&st, &mut dk, total * hkv * d, 1.0).unwrap();
        dinfer_kernels::ops::fp8_e4m3_roundtrip_bf16_n(&st, &mut dv, total * hkv * d, 1.0).unwrap();
        let dsl = st.htod_copy(slots.clone()).unwrap();
        let elems = num_blocks * bs * hkv * 2 * d;
        let mut cache_bf = st.alloc_zeros::<bf16>(elems).unwrap();
        let mut cache_f8 = st.alloc_zeros::<bf16>(elems / 2).unwrap();
        dinfer_kernels::gqa_cache_write_bf16(&st, &dk, &dv, &mut cache_bf, &dsl, total, hkv, d, bs, false).unwrap();
        dinfer_kernels::gqa_cache_write_bf16(&st, &dk, &dv, &mut cache_f8, &dsl, total, hkv, d, bs, true).unwrap();
        let dbt = st.htod_copy(bt).unwrap();
        let dseq = st.htod_copy(seq.iter().map(|&n| n as i32).collect::<Vec<_>>()).unwrap();
        let cu: Vec<i32> = (0..=b as i32).collect();
        let dcu = st.htod_copy(cu).unwrap();
        let mut scratch = st.alloc_zeros::<f32>(dinfer_kernels::gqa_decode_scratch_len(b, h, d)).unwrap();
        // B15b：fa_v5 PAGED（cfg=0）两侧同 kernel 结构（bf16 vs KV_FP8 模板），只 D=256（fa_v5 支持 128/256）
        let paths: Vec<&str> = if d == 256 { vec!["decode_win", "paged_slow", "paged_v5"] } else { vec!["decode_win", "paged_slow"] };
        for path in paths {
            let mut outs = Vec::new();
            for (cache, f8) in [(&cache_bf, false), (&cache_f8, true)] {
                let mut out = st.alloc_zeros::<bf16>(b * h * d).unwrap();
                if path == "decode_win" {
                    dinfer_kernels::gqa_decode_win_bf16(&st, &dq, cache, &dbt, &dseq, &mut out, &mut scratch, b, h, hkv, d, bs, max_blocks, 1.0, 0, None, window, f8).unwrap();
                } else {
                    // paged_slow：两侧都强制 cfg=-1 通用慢路径（D67）；paged_v5：两侧 cfg=0 → fa_v5 PAGED（bf16 / KV_FP8 模板同结构）
                    let cfg = if path == "paged_slow" { -1 } else { 0 };
                    let rc = unsafe {
                        dinfer_kernels::ffi::gqa_paged_win_bf16(
                            st.ptr(&dq) as *const std::ffi::c_void, st.ptr(cache) as *const std::ffi::c_void, st.ptr(&dbt) as *const i32,
                            st.ptr(&dseq) as *const i32, st.ptr(&dcu) as *const i32, st.ptr_mut(&mut out) as *mut std::ffi::c_void,
                            b as i64, 1, h as i64, hkv as i64, d as i64, bs as i64, max_blocks as i64, 1.0, 0, 0, cfg, window as i32, 0,
                            f8 as i32, st.raw() as _,
                        )
                    };
                    assert_eq!(rc, 0, "gqa_paged_win_bf16 rc");
                }
                st.synchronize().unwrap();
                outs.push(st.dtoh_sync_copy(&out).unwrap());
            }
            let nd = outs[0].iter().zip(&outs[1]).filter(|(a, c)| a.to_bits() != c.to_bits()).count();
            let nonzero = outs[0].iter().filter(|x| x.to_f32() != 0.0).count();
            println!("fp8cache D={d} win={window} seq={seq:?} {path}: 位差 {nd}/{} (非零 {nonzero}) => {}", b * h * d, if nd == 0 && nonzero > 0 { "ok" } else { "FAIL" });
            if nd != 0 || nonzero == 0 { fails.push(format!("D={d} win={window} {path}")); }
        }
    }
    assert!(fails.is_empty(), "失败: {fails:?}");
}
