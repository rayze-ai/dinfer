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
