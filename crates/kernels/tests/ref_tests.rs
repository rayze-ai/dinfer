//! 与 PyTorch 期望值对拍。先跑 `scripts/verify/gen_kernel_refs.py <dir>`，
//! 再 `KERNEL_REFS_DIR=<dir> cargo test -p dinfer-kernels --features ref-tests -- --ignored`。
#![cfg(feature = "ref-tests")]

use dinfer_kernels::{bf16, Device, Gemm, RopeStyle, Stream};
use std::path::PathBuf;

/// 期望值目录。相对路径按 **工作区根** 解析（cargo 跑集成测试时 cwd 是 crate 目录）。
fn refs() -> PathBuf {
    let p = PathBuf::from(std::env::var("KERNEL_REFS_DIR").unwrap_or_else(|_| "test/kernel_refs".into()));
    if p.is_absolute() {
        return p;
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    root.join(p)
}

fn load_f32(name: &str) -> (Vec<usize>, Vec<f32>) {
    let bytes =
        std::fs::read(refs().join(format!("{name}.npy"))).unwrap_or_else(|e| panic!("{name}.npy: {e}"));
    let arr: npyz::NpyFile<_> = npyz::NpyFile::new(&bytes[..]).unwrap();
    let shape: Vec<usize> = arr.shape().iter().map(|&d| d as usize).collect();
    let data: Vec<f32> = arr.into_vec::<f32>().unwrap();
    (shape, data)
}

fn load_i32(name: &str) -> (Vec<usize>, Vec<i32>) {
    let bytes =
        std::fs::read(refs().join(format!("{name}.npy"))).unwrap_or_else(|e| panic!("{name}.npy: {e}"));
    let arr: npyz::NpyFile<_> = npyz::NpyFile::new(&bytes[..]).unwrap();
    let shape: Vec<usize> = arr.shape().iter().map(|&d| d as usize).collect();
    let data: Vec<i32> = arr.into_vec::<i32>().unwrap();
    (shape, data)
}

fn to_bf16(v: &[f32]) -> Vec<bf16> {
    v.iter().map(|&x| bf16::from_f32(x)).collect()
}

fn assert_close(name: &str, got: &[bf16], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let mut worst = 0f32;
    let mut bad = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let g = g.to_f32();
        let tol = atol + rtol * w.abs();
        let d = (g - w).abs();
        if d > tol {
            bad += 1;
            if bad <= 5 {
                eprintln!("{name} mismatch at {i}: got {g} want {w} (tol {tol})");
            }
        }
        worst = worst.max(d);
    }
    assert_eq!(bad, 0, "{name}: {bad} mismatches, worst abs diff {worst}");
}

fn dev() -> (Device, Stream) {
    let d = Device::new(0).unwrap();
    let s = d.default_stream();
    (d, s)
}

#[test]
#[ignore]
fn rmsnorm_ref() {
    let (_d, s) = dev();
    let (shape, x) = load_f32("rmsnorm_x");
    let (_, w) = load_f32("rmsnorm_w");
    let (_, want) = load_f32("rmsnorm_out");
    let h = shape[1];
    let dx = s.htod_copy(to_bf16(&x)).unwrap();
    let dw = s.htod_copy(to_bf16(&w)).unwrap();
    let mut out = s.alloc_zeros::<bf16>(x.len()).unwrap();
    dinfer_kernels::rmsnorm_bf16(&s, &dx, &dw, &mut out, h, 1e-5).unwrap();
    assert_close("rmsnorm", &s.dtoh_sync_copy(&out).unwrap(), &want, 1e-2, 1e-2);
}

#[test]
#[ignore]
fn fused_add_rmsnorm_ref() {
    let (_d, s) = dev();
    let (shape, x) = load_f32("rmsnorm_x");
    let (_, w) = load_f32("rmsnorm_w");
    let (_, res_in) = load_f32("fused_res_in");
    let (_, res_want) = load_f32("fused_res_out");
    let (_, want) = load_f32("fused_out");
    let h = shape[1];
    let dx = s.htod_copy(to_bf16(&x)).unwrap();
    let dw = s.htod_copy(to_bf16(&w)).unwrap();
    let mut dres = s.htod_copy(to_bf16(&res_in)).unwrap();
    let mut out = s.alloc_zeros::<bf16>(x.len()).unwrap();
    dinfer_kernels::fused_add_rmsnorm_bf16(&s, &dx, &mut dres, &dw, &mut out, h, 1e-5).unwrap();
    assert_close("fused_add_rmsnorm.res", &s.dtoh_sync_copy(&dres).unwrap(), &res_want, 1e-2, 1e-2);
    assert_close("fused_add_rmsnorm.out", &s.dtoh_sync_copy(&out).unwrap(), &want, 1e-2, 1e-2);
}

fn rope_case(style: RopeStyle, suffix: &str) {
    let (_d, s) = dev();
    let (_, pos) = load_i32("rope_pos");
    let (cs_shape, cs) = load_f32("rope_cos_sin");
    let (qs, q) = load_f32("rope_q");
    let (ks, k) = load_f32("rope_k");
    let (_, q_want) = load_f32(&format!("rope_q_out_{suffix}"));
    let (_, k_want) = load_f32(&format!("rope_k_out_{suffix}"));
    let (q_heads, k_heads, head_dim, rope_dim) = (qs[1], ks[1], qs[2], cs_shape[1]);
    let dpos = s.htod_copy(pos).unwrap();
    let dcs = s.htod_copy(cs).unwrap();
    let mut dq = s.htod_copy(to_bf16(&q)).unwrap();
    let mut dk = s.htod_copy(to_bf16(&k)).unwrap();
    dinfer_kernels::rope_bf16(
        &s,
        &mut dq,
        Some(&mut dk),
        &dpos,
        &dcs,
        q_heads,
        k_heads,
        head_dim,
        rope_dim,
        style,
    )
    .unwrap();
    assert_close(&format!("rope_q_{suffix}"), &s.dtoh_sync_copy(&dq).unwrap(), &q_want, 2e-2, 2e-2);
    assert_close(&format!("rope_k_{suffix}"), &s.dtoh_sync_copy(&dk).unwrap(), &k_want, 2e-2, 2e-2);
}

#[test]
#[ignore]
fn rope_interleave_ref() {
    rope_case(RopeStyle::Interleave, "interleave");
}

#[test]
#[ignore]
fn rope_neox_ref() {
    rope_case(RopeStyle::Neox, "neox");
}

#[test]
#[ignore]
fn silu_and_mul_ref() {
    let (_d, s) = dev();
    let (shape, x) = load_f32("silu_x");
    let (_, want) = load_f32("silu_out");
    let inter = shape[1] / 2;
    let dx = s.htod_copy(to_bf16(&x)).unwrap();
    let mut out = s.alloc_zeros::<bf16>(shape[0] * inter).unwrap();
    dinfer_kernels::silu_and_mul_bf16(&s, &dx, &mut out, inter).unwrap();
    assert_close("silu_and_mul", &s.dtoh_sync_copy(&out).unwrap(), &want, 1e-2, 1e-2);
}

#[test]
#[ignore]
fn embedding_ref() {
    let (_d, s) = dev();
    let (ts, table) = load_f32("emb_table");
    let (_, ids) = load_i32("emb_ids");
    let (_, want) = load_f32("emb_out");
    let h = ts[1];
    let dt = s.htod_copy(to_bf16(&table)).unwrap();
    let di = s.htod_copy(ids.clone()).unwrap();
    let mut out = s.alloc_zeros::<bf16>(ids.len() * h).unwrap();
    dinfer_kernels::embedding_bf16(&s, &di, &dt, &mut out, h).unwrap();
    assert_close("embedding", &s.dtoh_sync_copy(&out).unwrap(), &want, 0.0, 0.0);
}

#[test]
#[ignore]
fn gemm_ref() {
    let (_d, s) = dev();
    let g = Gemm::new(&s).unwrap();
    let (as_, a) = load_f32("gemm_a");
    let (ws, w) = load_f32("gemm_w");
    let (_, bias) = load_f32("gemm_bias");
    let (_, want) = load_f32("gemm_out");
    let (m, k, n) = (as_[0], as_[1], ws[0]);
    let da = s.htod_copy(to_bf16(&a)).unwrap();
    let dw = s.htod_copy(to_bf16(&w)).unwrap();
    let db = s.htod_copy(to_bf16(&bias)).unwrap();
    let mut out = s.alloc_zeros::<bf16>(m * n).unwrap();
    g.forward_bf16(&da, &dw, Some(&db), &mut out, m, n, k).unwrap();
    assert_close("gemm", &s.dtoh_sync_copy(&out).unwrap(), &want, 3e-2, 2e-2);
}

#[test]
#[ignore]
fn argmax_ref() {
    let (_d, s) = dev();
    let (shape, logits) = load_f32("argmax_logits");
    let (_, want) = load_i32("argmax_out");
    let dl = s.htod_copy(logits).unwrap();
    let mut out = s.alloc_zeros::<i32>(shape[0]).unwrap();
    dinfer_kernels::argmax_f32(&s, &dl, &mut out, shape[1]).unwrap();
    assert_eq!(s.dtoh_sync_copy(&out).unwrap(), want);
}

#[test]
#[ignore]
fn topk_topp_mask_ref() {
    let (_d, s) = dev();
    let (shape, logits) = load_f32("mask_logits");
    let (_, top_k) = load_i32("mask_top_k");
    let (_, top_p) = load_f32("mask_top_p");
    let (_, want_isinf) = load_i32("mask_out_isinf");
    let mut dl = s.htod_copy(logits).unwrap();
    let dk = s.htod_copy(top_k).unwrap();
    let dp = s.htod_copy(top_p).unwrap();
    dinfer_kernels::topk_topp_mask_f32(&s, &mut dl, &dk, &dp, shape[1]).unwrap();
    let got = s.dtoh_sync_copy(&dl).unwrap();
    let (b, v) = (shape[0], shape[1]);
    for r in 0..b {
        let got_inf: Vec<i32> = got[r * v..(r + 1) * v].iter().map(|x| x.is_infinite() as i32).collect();
        let want_row = &want_isinf[r * v..(r + 1) * v];
        let diff = got_inf.iter().zip(want_row).filter(|(a, b)| a != b).count();
        let kept = got_inf.iter().filter(|&&x| x == 0).count();
        // kernel 用 fp32 二分阈值，参考用 sort+cumsum；top-p 边界附近概率几乎相同的 token
        // 两边取舍可能不同。允许差异 ≤ max(2, 0.1% of kept)。
        let tol = 2.max(kept / 1000);
        assert!(diff <= tol, "topk_topp_mask row {r}: {diff} differing positions (kept {kept}, tol {tol})");
    }
}

#[test]
#[ignore]
fn sample_distribution_matches_softmax() {
    // 分布级检验：对一行 64-vocab 采样 100k 次，频率与 softmax 概率的 L1 距离 < 0.03。
    // 100k 次下 L1 的随机波动期望 ≈ sum_i sqrt(p_i(1-p_i)/n) * 0.8 ≈ 0.018，阈值留 1.6 倍余量。
    let (_d, s) = dev();
    let v = 64usize;
    let n = 100_000usize;
    let row: Vec<f32> = (0..v).map(|i| ((i * 7) % 11) as f32 * 0.4 - 1.0).collect();
    let temp = 0.8f32;
    let logits: Vec<f32> = (0..n).flat_map(|_| row.iter().copied()).collect();
    let dl = s.htod_copy(logits).unwrap();
    let dt = s.htod_copy(vec![temp; n]).unwrap();
    let seeds = s.htod_copy(vec![42u64; n]).unwrap();
    let offs = s.htod_copy((0..n as u64).collect::<Vec<_>>()).unwrap();
    let mut out = s.alloc_zeros::<i32>(n).unwrap();
    dinfer_kernels::sample_f32(&s, &dl, &dt, &seeds, &offs, &mut out, v).unwrap();
    let got = s.dtoh_sync_copy(&out).unwrap();
    let mut counts = vec![0f32; v];
    for &t in &got {
        counts[t as usize] += 1.0;
    }
    let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = row.iter().map(|&x| ((x - m) / temp).exp()).collect();
    let z: f32 = exps.iter().sum();
    let l1: f32 = (0..v).map(|i| (counts[i] / n as f32 - exps[i] / z).abs()).sum();
    assert!(l1 < 0.03, "sample L1 distance {l1}");
}

// ---- GQA paged attention（Q1）----

fn load_i64(name: &str) -> Vec<i64> {
    let bytes =
        std::fs::read(refs().join(format!("{name}.npy"))).unwrap_or_else(|e| panic!("{name}.npy: {e}"));
    let arr: npyz::NpyFile<_> = npyz::NpyFile::new(&bytes[..]).unwrap();
    arr.into_vec::<i64>().unwrap()
}

#[test]
#[ignore]
fn gqa_prefill_ref() {
    let (_d, s) = dev();
    let (qs, q) = load_f32("gqa_q");
    let (ks, k) = load_f32("gqa_k");
    let (_, v) = load_f32("gqa_v");
    let (_, cu) = load_i32("gqa_cu");
    let (_, want) = load_f32("gqa_prefill_out");
    let (t, h, d) = (qs[0], qs[1], qs[2]);
    let hk = ks[1];
    let dq = s.htod_copy(to_bf16(&q)).unwrap();
    let dk = s.htod_copy(to_bf16(&k)).unwrap();
    let dv = s.htod_copy(to_bf16(&v)).unwrap();
    let dcu = s.htod_copy(cu.clone()).unwrap();
    let mut out = s.alloc_zeros::<bf16>(t * h * d).unwrap();
    dinfer_kernels::gqa_prefill_bf16(
        &s,
        &dq,
        &dk,
        &dv,
        &mut out,
        &dcu,
        cu.len() - 1,
        cu.windows(2).map(|w| (w[1] - w[0]) as usize).max().unwrap(),
        h,
        hk,
        d,
        (d as f32).powf(-0.5),
        0,
        0,
    )
    .unwrap();
    assert_close("gqa_prefill", &s.dtoh_sync_copy(&out).unwrap(), &want, 1e-2, 1e-2);
}

/// D=128（Qwen3 真实形状）走 flash tile kernel：3 个请求 70/33/5，覆盖多片、尾片不满、块内不活跃 warp。
#[test]
#[ignore]
fn gqa_prefill_d128_flash_ref() {
    let (_d, s) = dev();
    let (qs, q) = load_f32("gqa128_q");
    let (ks, k) = load_f32("gqa128_k");
    let (_, v) = load_f32("gqa128_v");
    let (_, cu) = load_i32("gqa128_cu");
    let (_, want) = load_f32("gqa128_prefill_out");
    let (t, h, d) = (qs[0], qs[1], qs[2]);
    let hk = ks[1];
    assert_eq!(d, 128);
    let dq = s.htod_copy(to_bf16(&q)).unwrap();
    let dk = s.htod_copy(to_bf16(&k)).unwrap();
    let dv = s.htod_copy(to_bf16(&v)).unwrap();
    let dcu = s.htod_copy(cu.clone()).unwrap();
    let mut out = s.alloc_zeros::<bf16>(t * h * d).unwrap();
    let max_len = cu.windows(2).map(|w| (w[1] - w[0]) as usize).max().unwrap();
    dinfer_kernels::gqa_prefill_bf16(
        &s,
        &dq,
        &dk,
        &dv,
        &mut out,
        &dcu,
        cu.len() - 1,
        max_len,
        h,
        hk,
        d,
        (d as f32).powf(-0.5),
        0,
        0,
    )
    .unwrap();
    assert_close("gqa_prefill_d128", &s.dtoh_sync_copy(&out).unwrap(), &want, 1e-2, 1e-2);
}

#[test]
#[ignore]
fn gqa_cache_write_and_paged_ref() {
    let (_d, s) = dev();
    let (ks, k) = load_f32("gqa_k");
    let (_, v) = load_f32("gqa_v");
    let (q2s, q2) = load_f32("gqa_q2");
    let (_, k2) = load_f32("gqa_k2");
    let (_, v2) = load_f32("gqa_v2");
    let hist_slots = load_i64("gqa_hist_slots");
    let new_slots = load_i64("gqa_new_slots");
    let (cs, cache_want) = load_f32("gqa_cache_expect");
    let (bts, bt) = load_i32("gqa_bt");
    let (_, seq_lens) = load_i32("gqa_seq_lens");
    let (_, cu_q) = load_i32("gqa_cu_q");
    let (_, want) = load_f32("gqa_paged_out");
    let (hk, d) = (ks[1], ks[2]);
    let (tq, h) = (q2s[0], q2s[1]);
    let (num_blocks, bs) = (cs[0], cs[1]);
    let mut cache = s.alloc_zeros::<bf16>(num_blocks * bs * hk * 2 * d).unwrap();
    // 写历史 + 新 token
    let dk = s.htod_copy(to_bf16(&k)).unwrap();
    let dv = s.htod_copy(to_bf16(&v)).unwrap();
    let dsl = s.htod_copy(hist_slots.clone()).unwrap();
    dinfer_kernels::gqa_cache_write_bf16(&s, &dk, &dv, &mut cache, &dsl, hist_slots.len(), hk, d, bs)
        .unwrap();
    let dk2 = s.htod_copy(to_bf16(&k2)).unwrap();
    let dv2 = s.htod_copy(to_bf16(&v2)).unwrap();
    let dsl2 = s.htod_copy(new_slots.clone()).unwrap();
    dinfer_kernels::gqa_cache_write_bf16(&s, &dk2, &dv2, &mut cache, &dsl2, new_slots.len(), hk, d, bs)
        .unwrap();
    assert_close("gqa_cache", &s.dtoh_sync_copy(&cache).unwrap(), &cache_want, 0.0, 0.0);
    // paged attention（req0 3 行带历史 11，req1 1 行 decode 带历史 6）
    let dq2 = s.htod_copy(to_bf16(&q2)).unwrap();
    let dbt = s.htod_copy(bt).unwrap();
    let dseq = s.htod_copy(seq_lens.clone()).unwrap();
    let dcu = s.htod_copy(cu_q.clone()).unwrap();
    let mut out = s.alloc_zeros::<bf16>(tq * h * d).unwrap();
    dinfer_kernels::gqa_paged_bf16(
        &s,
        &dq2,
        &cache,
        &dbt,
        &dseq,
        &dcu,
        &mut out,
        seq_lens.len(),
        cu_q.windows(2).map(|w| (w[1] - w[0]) as usize).max().unwrap(),
        h,
        hk,
        d,
        bs,
        bts[1],
        (d as f32).powf(-0.5),
        0,
        false,
    )
    .unwrap();
    assert_close("gqa_paged", &s.dtoh_sync_copy(&out).unwrap(), &want, 1e-2, 1e-2);
}

/// D=128 paged（flash tile kernel 读 paged cache）：历史 40/70 + 新 5/1 行，block 16。
#[test]
#[ignore]
fn gqa_paged_d128_flash_ref() {
    let (_d, s) = dev();
    let (ks, kh) = load_f32("gqa128p_k_hist");
    let (_, vh) = load_f32("gqa128p_v_hist");
    let sh = load_i64("gqa128p_slots_hist");
    let (qs, q) = load_f32("gqa128p_q");
    let (_, kn) = load_f32("gqa128p_k_new");
    let (_, vn) = load_f32("gqa128p_v_new");
    let sn = load_i64("gqa128p_slots_new");
    let (bts, bt) = load_i32("gqa128p_bt");
    let (_, seq_lens) = load_i32("gqa128p_seq_lens");
    let (_, cu_q) = load_i32("gqa128p_cu_q");
    let (_, want) = load_f32("gqa128p_out");
    let (_, meta) = load_i32("gqa128p_meta");
    let (num_blocks, bs) = (meta[0] as usize, meta[1] as usize);
    let (hk, d) = (ks[1], ks[2]);
    let (tq, h) = (qs[0], qs[1]);
    assert_eq!(d, 128);
    let mut cache = s.alloc_zeros::<bf16>(num_blocks * bs * hk * 2 * d).unwrap();
    let dkh = s.htod_copy(to_bf16(&kh)).unwrap();
    let dvh = s.htod_copy(to_bf16(&vh)).unwrap();
    let dsh = s.htod_copy(sh.clone()).unwrap();
    dinfer_kernels::gqa_cache_write_bf16(&s, &dkh, &dvh, &mut cache, &dsh, sh.len(), hk, d, bs).unwrap();
    let dkn = s.htod_copy(to_bf16(&kn)).unwrap();
    let dvn = s.htod_copy(to_bf16(&vn)).unwrap();
    let dsn = s.htod_copy(sn.clone()).unwrap();
    dinfer_kernels::gqa_cache_write_bf16(&s, &dkn, &dvn, &mut cache, &dsn, sn.len(), hk, d, bs).unwrap();
    let dq = s.htod_copy(to_bf16(&q)).unwrap();
    let dbt = s.htod_copy(bt).unwrap();
    let dseq = s.htod_copy(seq_lens.clone()).unwrap();
    let dcu = s.htod_copy(cu_q.clone()).unwrap();
    let mut out = s.alloc_zeros::<bf16>(tq * h * d).unwrap();
    let max_q = cu_q.windows(2).map(|w| (w[1] - w[0]) as usize).max().unwrap();
    dinfer_kernels::gqa_paged_bf16(
        &s,
        &dq,
        &cache,
        &dbt,
        &dseq,
        &dcu,
        &mut out,
        seq_lens.len(),
        max_q,
        h,
        hk,
        d,
        bs,
        bts[1],
        (d as f32).powf(-0.5),
        0,
        false,
    )
    .unwrap();
    assert_close("gqa_paged_d128", &s.dtoh_sync_copy(&out).unwrap(), &want, 1e-2, 1e-2);
}

/// D=128 混批分流：同一 paged fixture（含 q_len==1 的请求）走 gqa_mixed_bf16（decode_v2 行映射 + paged_tc skip_single），结果须与纯 paged 一致。
#[test]
#[ignore]
fn gqa_mixed_d128_ref() {
    let (_d, s) = dev();
    let (ks, kh) = load_f32("gqa128p_k_hist");
    let (_, vh) = load_f32("gqa128p_v_hist");
    let sh = load_i64("gqa128p_slots_hist");
    let (qs, q) = load_f32("gqa128p_q");
    let (_, kn) = load_f32("gqa128p_k_new");
    let (_, vn) = load_f32("gqa128p_v_new");
    let sn = load_i64("gqa128p_slots_new");
    let (bts, bt) = load_i32("gqa128p_bt");
    let (_, seq_lens) = load_i32("gqa128p_seq_lens");
    let (_, cu_q) = load_i32("gqa128p_cu_q");
    let (_, want) = load_f32("gqa128p_out");
    let (_, meta) = load_i32("gqa128p_meta");
    let (num_blocks, bs) = (meta[0] as usize, meta[1] as usize);
    let (hk, d) = (ks[1], ks[2]);
    let (tq, h) = (qs[0], qs[1]);
    assert_eq!(d, 128);
    let mut cache = s.alloc_zeros::<bf16>(num_blocks * bs * hk * 2 * d).unwrap();
    let dkh = s.htod_copy(to_bf16(&kh)).unwrap();
    let dvh = s.htod_copy(to_bf16(&vh)).unwrap();
    let dsh = s.htod_copy(sh.clone()).unwrap();
    dinfer_kernels::gqa_cache_write_bf16(&s, &dkh, &dvh, &mut cache, &dsh, sh.len(), hk, d, bs).unwrap();
    let dkn = s.htod_copy(to_bf16(&kn)).unwrap();
    let dvn = s.htod_copy(to_bf16(&vn)).unwrap();
    let dsn = s.htod_copy(sn.clone()).unwrap();
    dinfer_kernels::gqa_cache_write_bf16(&s, &dkn, &dvn, &mut cache, &dsn, sn.len(), hk, d, bs).unwrap();
    let dq = s.htod_copy(to_bf16(&q)).unwrap();
    let dbt = s.htod_copy(bt).unwrap();
    let dseq = s.htod_copy(seq_lens.clone()).unwrap();
    let dcu = s.htod_copy(cu_q.clone()).unwrap();
    let mut out = s.alloc_zeros::<bf16>(tq * h * d).unwrap();
    let mut scratch = s.alloc_zeros::<f32>(dinfer_kernels::gqa_decode_scratch_len(seq_lens.len(), h, d)).unwrap();
    assert!(cu_q.windows(2).any(|w| w[1] - w[0] == 1), "fixture must contain a q_len==1 request");
    let max_q = cu_q.windows(2).map(|w| (w[1] - w[0]) as usize).max().unwrap();
    dinfer_kernels::gqa_mixed_bf16(
        &s,
        &dq,
        &cache,
        &dbt,
        &dseq,
        &dcu,
        &mut out,
        &mut scratch,
        seq_lens.len(),
        max_q,
        h,
        hk,
        d,
        bs,
        bts[1],
        (d as f32).powf(-0.5),
        0,
    )
    .unwrap();
    assert_close("gqa_mixed_d128", &s.dtoh_sync_copy(&out).unwrap(), &want, 1e-2, 1e-2);
}

/// D=128 纯 decode split-K kernel：3 个请求长度 300 / 70 / 1（多 split、短段、单 key），block 16。
#[test]
#[ignore]
fn gqa_decode_d128_splitk_ref() {
    let (_d, s) = dev();
    let (ks, kall) = load_f32("gqa128d_k");
    let (_, vall) = load_f32("gqa128d_v");
    let slots = load_i64("gqa128d_slots");
    let (qs, q) = load_f32("gqa128d_q");
    let (bts, bt) = load_i32("gqa128d_bt");
    let (_, seq_lens) = load_i32("gqa128d_seq_lens");
    let (_, want) = load_f32("gqa128d_out");
    let (_, meta) = load_i32("gqa128d_meta");
    let (num_blocks, bs) = (meta[0] as usize, meta[1] as usize);
    let (hk, d) = (ks[1], ks[2]);
    let (b, h) = (qs[0], qs[1]);
    let mut cache = s.alloc_zeros::<bf16>(num_blocks * bs * hk * 2 * d).unwrap();
    let dk = s.htod_copy(to_bf16(&kall)).unwrap();
    let dv = s.htod_copy(to_bf16(&vall)).unwrap();
    let dsl = s.htod_copy(slots.clone()).unwrap();
    dinfer_kernels::gqa_cache_write_bf16(&s, &dk, &dv, &mut cache, &dsl, slots.len(), hk, d, bs).unwrap();
    let dq = s.htod_copy(to_bf16(&q)).unwrap();
    let dbt = s.htod_copy(bt).unwrap();
    let dseq = s.htod_copy(seq_lens.clone()).unwrap();
    let mut out = s.alloc_zeros::<bf16>(b * h * d).unwrap();
    let mut scratch = s.alloc_zeros::<f32>(dinfer_kernels::gqa_decode_scratch_len(b, h, d)).unwrap();
    dinfer_kernels::gqa_decode_bf16(
        &s,
        &dq,
        &cache,
        &dbt,
        &dseq,
        &mut out,
        &mut scratch,
        b,
        h,
        hk,
        d,
        bs,
        bts[1],
        (d as f32).powf(-0.5),
        0,
        None,
    )
    .unwrap();
    assert_close("gqa_decode_d128", &s.dtoh_sync_copy(&out).unwrap(), &want, 1e-2, 1e-2);
}

/// D=128 decode，B=32（B×Hkv=256 → splits=1，v2 直接写出路径），长度 1..64 随机。
#[test]
#[ignore]
fn gqa_decode_d128_b32_direct_ref() {
    let (_d, s) = dev();
    let (ks, kall) = load_f32("gqa128d32_k");
    let (_, vall) = load_f32("gqa128d32_v");
    let slots = load_i64("gqa128d32_slots");
    let (qs, q) = load_f32("gqa128d32_q");
    let (bts, bt) = load_i32("gqa128d32_bt");
    let (_, seq_lens) = load_i32("gqa128d32_seq_lens");
    let (_, want) = load_f32("gqa128d32_out");
    let (_, meta) = load_i32("gqa128d32_meta");
    let (num_blocks, bs) = (meta[0] as usize, meta[1] as usize);
    let (hk, d) = (ks[1], ks[2]);
    let (b, h) = (qs[0], qs[1]);
    let mut cache = s.alloc_zeros::<bf16>(num_blocks * bs * hk * 2 * d).unwrap();
    let dk = s.htod_copy(to_bf16(&kall)).unwrap();
    let dv = s.htod_copy(to_bf16(&vall)).unwrap();
    let dsl = s.htod_copy(slots.clone()).unwrap();
    dinfer_kernels::gqa_cache_write_bf16(&s, &dk, &dv, &mut cache, &dsl, slots.len(), hk, d, bs).unwrap();
    let dq = s.htod_copy(to_bf16(&q)).unwrap();
    let dbt = s.htod_copy(bt).unwrap();
    let dseq = s.htod_copy(seq_lens.clone()).unwrap();
    let mut out = s.alloc_zeros::<bf16>(b * h * d).unwrap();
    let mut scratch = s.alloc_zeros::<f32>(dinfer_kernels::gqa_decode_scratch_len(b, h, d)).unwrap();
    dinfer_kernels::gqa_decode_bf16(
        &s,
        &dq,
        &cache,
        &dbt,
        &dseq,
        &mut out,
        &mut scratch,
        b,
        h,
        hk,
        d,
        bs,
        bts[1],
        (d as f32).powf(-0.5),
        0,
        None,
    )
    .unwrap();
    assert_close("gqa_decode_d128_b32", &s.dtoh_sync_copy(&out).unwrap(), &want, 1e-2, 1e-2);
}

/// Qwen3 QKV 后处理融合 kernel：q/k per-head RMSNorm + neox RoPE 原地 + K/V 写 cache，与 torch 逐步参考比。
#[test]
#[ignore]
fn qkv_post_fused_ref() {
    let (_d, s) = dev();
    let (shape, qkv) = load_f32("qkvpost_in");
    let (_, wq) = load_f32("qkvpost_wq");
    let (_, wk) = load_f32("qkvpost_wk");
    let (_, pos) = load_i32("qkvpost_pos");
    let (_, cs) = load_f32("qkvpost_cos_sin");
    let slots = load_i64("qkvpost_slots");
    let (_, want) = load_f32("qkvpost_out");
    let (_, cache_want) = load_f32("qkvpost_cache");
    let (_, meta) = load_i32("qkvpost_meta");
    let (num_blocks, bs) = (meta[0] as usize, meta[1] as usize);
    let (t, row) = (shape[0], shape[1]);
    let (h, hk, d) = (16usize, 8usize, 128usize);
    assert_eq!(row, (h + 2 * hk) * d);
    let mut dq = s.htod_copy(to_bf16(&qkv)).unwrap();
    let dwq = s.htod_copy(to_bf16(&wq)).unwrap();
    let dwk = s.htod_copy(to_bf16(&wk)).unwrap();
    let dpos = s.htod_copy(pos).unwrap();
    let dcs = s.htod_copy(cs).unwrap();
    let dsl = s.htod_copy(slots).unwrap();
    let mut cache = s.alloc_zeros::<bf16>(num_blocks * bs * hk * 2 * d).unwrap();
    dinfer_kernels::qkv_post_bf16(
        &s, &mut dq, &dwq, &dwk, &dpos, &dcs, &mut cache, &dsl, t, h, hk, d, 1e-6, bs,
    )
    .unwrap();
    let got = s.dtoh_sync_copy(&dq).unwrap();
    // q/k 段比；v 段应原样
    let qk = (h + hk) * d;
    for r in 0..t {
        assert_close(
            &format!("qkv_post row {r} q|k"),
            &got[r * row..r * row + qk],
            &want[r * row..r * row + qk],
            2e-2,
            2e-2,
        );
        assert_close(
            &format!("qkv_post row {r} v untouched"),
            &got[r * row + qk..(r + 1) * row],
            &qkv[r * row + qk..(r + 1) * row],
            0.0,
            0.0,
        );
    }
    assert_close("qkv_post cache", &s.dtoh_sync_copy(&cache).unwrap(), &cache_want, 2e-2, 2e-2);
}

// ---- Q2 采样：penalties / temperature / min_p / logprobs ----

fn assert_f32_close(name: &str, got: &[f32], want: &[f32], atol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let mut bad = 0;
    let mut worst = 0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = if g.is_infinite() && w.is_infinite() { 0.0 } else { (g - w).abs() };
        if d > atol {
            bad += 1;
            if bad <= 5 {
                eprintln!("{name} mismatch at {i}: got {g} want {w}");
            }
        }
        worst = worst.max(d);
    }
    assert_eq!(bad, 0, "{name}: {bad} mismatches, worst {worst}");
}

#[test]
#[ignore]
fn penalties_ref() {
    let (_d, s) = dev();
    let (shape, logits) = load_f32("pen_logits");
    let (_, ids) = load_i32("pen_ids");
    let (_, offs) = load_i32("pen_offsets");
    let (_, plen) = load_i32("pen_prompt_len");
    let (_, rep) = load_f32("pen_rep");
    let (_, freq) = load_f32("pen_freq");
    let (_, pres) = load_f32("pen_pres");
    let (_, want) = load_f32("pen_out");
    let mut dl = s.htod_copy(logits).unwrap();
    let dids = s.htod_copy(ids).unwrap();
    let doffs = s.htod_copy(offs).unwrap();
    let dplen = s.htod_copy(plen).unwrap();
    let drep = s.htod_copy(rep).unwrap();
    let dfreq = s.htod_copy(freq).unwrap();
    let dpres = s.htod_copy(pres).unwrap();
    dinfer_kernels::penalties_f32(&s, &mut dl, &dids, &doffs, &dplen, &drep, &dfreq, &dpres, shape[1])
        .unwrap();
    assert_f32_close("penalties", &s.dtoh_sync_copy(&dl).unwrap(), &want, 1e-5);
}

#[test]
#[ignore]
fn temperature_ref() {
    let (_d, s) = dev();
    let (shape, logits) = load_f32("temp_in");
    let (_, t) = load_f32("temp_t");
    let (_, want) = load_f32("temp_out");
    let mut dl = s.htod_copy(logits).unwrap();
    let dt = s.htod_copy(t).unwrap();
    dinfer_kernels::temperature_f32(&s, &mut dl, &dt, shape[1]).unwrap();
    assert_f32_close("temperature", &s.dtoh_sync_copy(&dl).unwrap(), &want, 1e-5);
}

#[test]
#[ignore]
fn min_p_mask_ref() {
    let (_d, s) = dev();
    let (shape, logits) = load_f32("minp_in");
    let (_, p) = load_f32("minp_p");
    let (_, want_inf) = load_i32("minp_out_isinf");
    let mut dl = s.htod_copy(logits).unwrap();
    let dp = s.htod_copy(p).unwrap();
    dinfer_kernels::min_p_mask_f32(&s, &mut dl, &dp, shape[1]).unwrap();
    let got = s.dtoh_sync_copy(&dl).unwrap();
    let got_inf: Vec<i32> = got.iter().map(|x| x.is_infinite() as i32).collect();
    let diff = got_inf.iter().zip(&want_inf).filter(|(a, b)| a != b).count();
    assert!(diff <= 2, "min_p mask: {diff} positions differ (fp32 threshold edge)");
}

#[test]
#[ignore]
fn logprobs_ref() {
    let (_d, s) = dev();
    let (shape, logits) = load_f32("lp_logits");
    let (_, chosen) = load_i32("lp_chosen");
    let (ids_shape, want_ids) = load_i32("lp_ids");
    let (_, want_vals) = load_f32("lp_vals");
    let (_, want_rank) = load_i32("lp_rank");
    let (b, v) = (shape[0], shape[1]);
    let k = ids_shape[1] - 1;
    let dl = s.htod_copy(logits).unwrap();
    let dc = s.htod_copy(chosen).unwrap();
    let mut out_ids = s.alloc_zeros::<i32>(b * (k + 1)).unwrap();
    let mut out_lp = s.alloc_zeros::<f32>(b * (k + 1)).unwrap();
    let mut rank = s.alloc_zeros::<i32>(b).unwrap();
    dinfer_kernels::logprobs_f32(&s, &dl, &dc, k, &mut out_ids, &mut out_lp, &mut rank, v).unwrap();
    assert_eq!(s.dtoh_sync_copy(&out_ids).unwrap(), want_ids, "logprobs ids");
    assert_f32_close("logprobs vals", &s.dtoh_sync_copy(&out_lp).unwrap(), &want_vals, 1e-4);
    assert_eq!(s.dtoh_sync_copy(&rank).unwrap(), want_rank, "logprobs rank");
}
