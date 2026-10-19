//! KDA 算子对拍（Kimi-K3 线）。参考由 `scripts/verify/gen_kda_refs.py` 生成（纯 torch，与 fla Triton 同式）。
//!
//! 运行：`KDA_REFS_DIR=<dir> cargo test -p dinfer-kernels --features ref-tests --test kda_parity -- --ignored --test-threads=1`
//! 汇总入口 `scripts/verify/k3_kda.sh`（exit 0 = 通过）。
//!
//! 容差口径（`docs/pitfalls.md`）：软判决比 cos 与 max abs，不比逐位；bf16 出 1e-2/1e-2，
//! fp32 递归状态 1e-4（状态是累乘结果，比单步输出更敏感）。
#![cfg(feature = "ref-tests")]

use dinfer_kernels::{bf16, ops, Device, Stream};
use std::path::PathBuf;

fn refs_root() -> PathBuf {
    let p = PathBuf::from(std::env::var("KDA_REFS_DIR").unwrap_or_else(|_| "test/kimik3/2026-10-19/kda_refs".into()));
    if p.is_absolute() {
        p
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join(p)
    }
}

fn load(dir: &str, name: &str) -> (Vec<usize>, Vec<f32>) {
    // bf16 以原始位存（`<u2`），这里统一还原成 f32；名字不带后缀，自动试 `_bf16`。
    let base = refs_root().join(dir);
    let p = base.join(format!("{name}_bf16.npy"));
    let bytes = match std::fs::read(&p) {
        Ok(b) => b,
        Err(_) => std::fs::read(base.join(format!("{name}.npy"))).unwrap_or_else(|e| panic!("{}: {e}", p.display())),
    };
    let arr = npyz::NpyFile::new(&bytes[..]).unwrap();
    let shape: Vec<usize> = arr.shape().iter().map(|&d| d as usize).collect();
    // bf16 以原始位存成 `u2`（见 `gen_kda_refs.py` 的 save）。npyz 的 into_vec 只认精确的 Rust 类型，
    // 先看 npy 头里的 descr 再选分支。
    let descr = String::from_utf8_lossy(&bytes[10..128.min(bytes.len())]).to_string();
    let v: Vec<f32> = if descr.contains("'<u2'") || descr.contains("\"<u2\"") {
        arr.into_vec::<u16>().unwrap().iter().map(|&b| bf16::from_bits(b).to_f32()).collect()
    } else if descr.contains("<f4") {
        arr.into_vec::<f32>().unwrap()
    } else {
        panic!("{name}: 未支持的 descr {descr}")
    };
    (shape, v)
}

fn load_i32(dir: &str, name: &str) -> Vec<i32> {
    let bytes = std::fs::read(refs_root().join(dir).join(format!("{name}.npy"))).unwrap();
    let arr = npyz::NpyFile::new(&bytes[..]).unwrap();
    arr.into_vec::<i32>().unwrap()
}

fn to_bf16(v: &[f32]) -> Vec<bf16> {
    v.iter().map(|&x| bf16::from_f32(x)).collect()
}

/// 软判决：max abs + cos 相似度。bf16 出 atol 1e-2、cos > 0.999。
fn assert_close(name: &str, got: &[bf16], want: &[f32], atol: f32, min_cos: f32) {
    assert_eq!(got.len(), want.len(), "{name}: 长度 {} vs {}", got.len(), want.len());
    let mut max_err = 0f32;
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    let mut worst = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let g = g.to_f32();
        let e = (g - w).abs();
        if e > max_err {
            max_err = e;
            worst = i;
        }
        dot += (g as f64) * (*w as f64);
        na += (g as f64) * (g as f64);
        nb += (*w as f64) * (*w as f64);
    }
    let cos = if na > 0.0 && nb > 0.0 { dot / (na.sqrt() * nb.sqrt()) } else { 1.0 };
    let min_cos = min_cos as f64;
    assert!(
        max_err <= atol && cos >= min_cos,
        "{name}: max_err {max_err:.5} (at {worst}: {} vs {}) cos {cos:.6}（要求 <= {atol} / >= {min_cos}）",
        got[worst].to_f32(),
        want[worst]
    );
    println!("  {name:<28} max_err {max_err:.5}  cos {cos:.7}");
}

fn assert_close_f32(name: &str, got: &[f32], want: &[f32], atol: f32, min_cos: f32) {
    let g: Vec<f32> = got.to_vec();
    assert_eq!(g.len(), want.len(), "{name}: 长度");
    let mut max_err = 0f32;
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    let mut worst = 0usize;
    for (i, (a, b)) in g.iter().zip(want).enumerate() {
        let e = (a - b).abs();
        if e > max_err {
            max_err = e;
            worst = i;
        }
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64) * (*a as f64);
        nb += (*b as f64) * (*b as f64);
    }
    let cos = if na > 0.0 && nb > 0.0 { dot / (na.sqrt() * nb.sqrt()) } else { 1.0 };
    let min_cos = min_cos as f64;
    assert!(
        max_err <= atol && cos >= min_cos,
        "{name}: max_err {max_err:.7} (at {worst}: {} vs {}) cos {cos:.7}（要求 <= {atol} / >= {min_cos}）",
        g[worst], want[worst]
    );
    println!("  {name:<28} max_err {max_err:.7}  cos {cos:.7}");
}

fn dev() -> (Device, Stream) {
    let d = Device::new(0).unwrap();
    let s = d.default_stream();
    (d, s)
}

fn up_bf16(st: &Stream, v: &[f32]) -> dinfer_kernels::cudarc::driver::CudaSlice<bf16> {
    st.htod_copy(to_bf16(v)).unwrap()
}

// --------------------------------------------------------------------------- cases

#[test]
#[ignore]
fn kda_conv_prefill() {
    let (dev, st) = dev();
    let _ = &dev;
    let dir = "conv_prefill";
    let (shape, x) = load(dir, "x");
    let (_, w) = load(dir, "w");
    let (ss, st_in) = load(dir, "state_in");
    let (_, want_out) = load(dir, "out");
    let (_, want_state) = load(dir, "state_out");
    let dim = shape[1];
    let width = w.len() / dim;
    let state_len = ss[2];
    let cu = load_i32(dir, "cu_seqlens");
    let has = load_i32(dir, "has_initial");

    let dx = up_bf16(&st, &x);
    let dw = up_bf16(&st, &w);
    let mut dout = st.alloc_uninit::<bf16>(x.len()).unwrap();
    let mut dstate = up_bf16(&st, &st_in);
    let dcu = st.htod_copy(cu).unwrap();
    let dhas = st.htod_copy(has).unwrap();
    ops::kda_conv1d_prefill(&st, &dx, &dw, &mut dout, &mut dstate, &dcu, None, Some(&dhas), 1, dim, width, state_len).unwrap();
    st.synchronize().unwrap();
    assert_close("conv_prefill/out", &st.dtoh_sync_copy(&dout).unwrap(), &want_out, 1e-2, 0.999);
    assert_close("conv_prefill/state", &st.dtoh_sync_copy(&dstate).unwrap(), &want_state, 1e-2, 0.999);
}

#[test]
#[ignore]
fn kda_conv_prefill_varlen() {
    let (dev, st) = dev();
    let _ = &dev;
    let dir = "conv_prefill_varlen";
    let (shape, x) = load(dir, "x");
    let (_, w) = load(dir, "w");
    let (ss, st_in) = load(dir, "state_in");
    let (_, want_out) = load(dir, "out");
    let (_, want_state) = load(dir, "state_out");
    let dim = shape[1];
    let width = w.len() / dim;
    let cu = load_i32(dir, "cu_seqlens");
    let has = load_i32(dir, "has_initial");
    let nseq = cu.len() - 1;

    let dx = up_bf16(&st, &x);
    let dw = up_bf16(&st, &w);
    let mut dout = st.alloc_uninit::<bf16>(x.len()).unwrap();
    let mut dstate = up_bf16(&st, &st_in);
    let dcu = st.htod_copy(cu).unwrap();
    let dhas = st.htod_copy(has).unwrap();
    ops::kda_conv1d_prefill(&st, &dx, &dw, &mut dout, &mut dstate, &dcu, None, Some(&dhas), nseq, dim, width, ss[2]).unwrap();
    st.synchronize().unwrap();
    assert_close("conv_prefill_varlen/out", &st.dtoh_sync_copy(&dout).unwrap(), &want_out, 1e-2, 0.999);
    assert_close("conv_prefill_varlen/state", &st.dtoh_sync_copy(&dstate).unwrap(), &want_state, 1e-2, 0.999);
}

#[test]
#[ignore]
fn kda_conv_update() {
    let (dev, st) = dev();
    let _ = &dev;
    let dir = "conv_update";
    let (shape, w) = load(dir, "w");
    let dim = shape[0];
    let width = shape[1];
    let (ss, st0) = load(dir, "state0");
    let batch = ss[0];
    let dw = up_bf16(&st, &w);
    let mut dstate = up_bf16(&st, &st0);
    for s in 0..4 {
        let (_, x) = load(dir, &format!("x{s}"));
        let (_, want) = load(dir, &format!("out{s}"));
        let dx = up_bf16(&st, &x);
        let mut dout = st.alloc_uninit::<bf16>(x.len()).unwrap();
        ops::kda_conv1d_update(&st, &dx, &dw, &mut dout, &mut dstate, None, batch, dim, width, ss[2]).unwrap();
        st.synchronize().unwrap();
        assert_close(&format!("conv_update/out{s}"), &st.dtoh_sync_copy(&dout).unwrap(), &want, 1e-2, 0.999);
    }
    let (_, want_state) = load(dir, "state_final");
    assert_close("conv_update/state", &st.dtoh_sync_copy(&dstate).unwrap(), &want_state, 1e-2, 0.999);
}

#[test]
#[ignore]
fn kda_gate_op() {
    let (dev, st) = dev();
    let _ = &dev;
    let dir = "gate";
    let (gs, g) = load(dir, "g");
    let heads = gs[1];
    let head_dim = gs[2];
    let tokens = gs[0];
    let (_, a_log) = load(dir, "a_log");
    let (_, dt_bias) = load(dir, "dt_bias");
    let (_, want_lb) = load(dir, "out_lb");
    let (_, want_sp) = load(dir, "out_softplus");
    let dg = up_bf16(&st, &g);
    let da = st.htod_copy(a_log).unwrap();
    let db = st.htod_copy(dt_bias).unwrap();
    for (label, lb, want) in [("out_lb", Some(-5.0f32), &want_lb), ("out_softplus", None, &want_sp)] {
        let mut dout = st.alloc_uninit::<f32>(g.len()).unwrap();
        ops::kda_gate(&st, &dg, &da, &db, &mut dout, tokens, heads, head_dim, lb).unwrap();
        st.synchronize().unwrap();
        assert_close_f32(&format!("gate/{label}"), &st.dtoh_sync_copy(&dout).unwrap(), want, 1e-4, 0.9999);
    }
}

fn recurrent_case(dir: &str) {
    let (dev, st) = dev();
    let _ = &dev;
    let (qs, q) = load(dir, "q");
    let (_, k) = load(dir, "k");
    let (_, v) = load(dir, "v");
    let (_, raw_g) = load(dir, "raw_g");
    let (_, raw_beta) = load(dir, "raw_beta");
    let (_, a_log) = load(dir, "a_log");
    let (_, dt_bias) = load(dir, "dt_bias");
    let (_, want) = load(dir, "out");
    let (_, want_state) = load(dir, "state_out");
    let (st_shape, st_in) = load(dir, "state_in");
    let cu = load_i32(dir, "cu_seqlens");
    let heads = qs[1];
    let head_dim = qs[2];
    let nseq = cu.len() - 1;

    let dq = up_bf16(&st, &q);
    let dk = up_bf16(&st, &k);
    let dv = up_bf16(&st, &v);
    let dg = up_bf16(&st, &raw_g);
    let dbeta = st.htod_copy(raw_beta).unwrap();
    let da = st.htod_copy(a_log).unwrap();
    let db = st.htod_copy(dt_bias).unwrap();
    let mut dout = st.alloc_uninit::<bf16>(q.len()).unwrap();
    let mut dstate = st.htod_copy(st_in).unwrap();
    // 初态里 has_state=0 的槽必须由调用方清零（与调度器语义一致）
    let cu_host = cu.clone();
    let _ = cu_host;
    ops::kda_recurrent(&st, &dq, &dk, &dv, &dg, &dbeta, &da, &db, &mut dout, &mut dstate, &st.htod_copy(cu).unwrap(), None,
                       nseq, heads, head_dim, (head_dim as f32).powf(-0.5), Some(-5.0)).unwrap();
    st.synchronize().unwrap();
    assert_close(&format!("{dir}/out"), &st.dtoh_sync_copy(&dout).unwrap(), &want, 1e-2, 0.999);
    assert_eq!(st_shape[0], nseq, "{dir}: state 槽数");
    assert_close_f32(&format!("{dir}/state"), &st.dtoh_sync_copy(&dstate).unwrap(), &want_state, 1e-4, 0.999);
}

#[test]
#[ignore]
fn kda_recurrent_prefill() { recurrent_case("recurrent_prefill"); }

#[test]
#[ignore]
fn kda_recurrent_varlen() { recurrent_case("recurrent_varlen"); }

#[test]
#[ignore]
fn kda_recurrent_decode_steps() {
    let (dev, st) = dev();
    let _ = &dev;
    let dir = "recurrent_decode";
    let (_, a_log) = load(dir, "a_log");
    let (_, dt_bias) = load(dir, "dt_bias");
    let (ss, st0) = load(dir, "state_in");
    let batch = ss[0];
    let heads = ss[1];
    let head_dim = ss[2];
    let da = st.htod_copy(a_log).unwrap();
    let db = st.htod_copy(dt_bias).unwrap();
    let mut dstate = st.htod_copy(st0).unwrap();
    for s in 0..4 {
        let (_, q) = load(dir, &format!("q{s}"));
        let (_, k) = load(dir, &format!("k{s}"));
        let (_, v) = load(dir, &format!("v{s}"));
        let (_, rg) = load(dir, &format!("raw_g{s}"));
        let (_, rb) = load(dir, &format!("raw_beta{s}"));
        let (_, want) = load(dir, &format!("out{s}"));
        let (_, want_state) = load(dir, &format!("state{s}"));
        let cu = load_i32(dir, &format!("cu_seqlens{s}"));
        let dq = up_bf16(&st, &q);
        let dk = up_bf16(&st, &k);
        let dv = up_bf16(&st, &v);
        let dg = up_bf16(&st, &rg);
        let dbeta = st.htod_copy(rb).unwrap();
        let mut dout = st.alloc_uninit::<bf16>(q.len()).unwrap();
        ops::kda_recurrent(&st, &dq, &dk, &dv, &dg, &dbeta, &da, &db, &mut dout, &mut dstate,
                           &st.htod_copy(cu).unwrap(), None, batch, heads, head_dim,
                           (head_dim as f32).powf(-0.5), Some(-5.0)).unwrap();
        st.synchronize().unwrap();
        assert_close(&format!("recurrent_decode/out{s}"), &st.dtoh_sync_copy(&dout).unwrap(), &want, 1e-2, 0.999);
        assert_close_f32(&format!("recurrent_decode/state{s}"), &st.dtoh_sync_copy(&dstate).unwrap(),
                         &want_state, 1e-4, 0.999);
    }
}

#[test]
#[ignore]
fn kda_gated_rmsnorm_op() {
    let (dev, st) = dev();
    let _ = &dev;
    let dir = "gated_rmsnorm";
    let (xs, x) = load(dir, "x");
    let (_, w) = load(dir, "w");
    let (_, gate) = load(dir, "gate");
    let (_, want) = load(dir, "out");
    let rows = xs[0];
    let head_dim = xs[1];
    let dx = up_bf16(&st, &x);
    let dw = up_bf16(&st, &w);
    let dgt = up_bf16(&st, &gate);
    let mut dout = st.alloc_uninit::<bf16>(x.len()).unwrap();
    ops::kda_gated_rmsnorm(&st, &dx, &dw, &dgt, &mut dout, rows, head_dim, 1e-5).unwrap();
    st.synchronize().unwrap();
    assert_close("gated_rmsnorm", &st.dtoh_sync_copy(&dout).unwrap(), &want, 1e-2, 0.999);
}


// --------------------------------------------------------------------------- chunk 形式 prefill

/// 相对口径（coordinator 2026-10-19 确认）：state 用 cos ≥ 0.9999 + 相对 max_err ≤ 5e-3
/// （相对 `|want|max`）；口径依据 `docs/pitfalls.md`「软判决不比逐位」。
fn assert_close_rel(name: &str, got: &[f32], want: &[f32], rel: f32, min_cos: f32) {
    assert_eq!(got.len(), want.len(), "{name}: 长度");
    let mut max_err = 0f32;
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    let mut worst = 0usize;
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        let e = (a - b).abs();
        if e > max_err {
            max_err = e;
            worst = i;
        }
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64) * (*a as f64);
        nb += (*b as f64) * (*b as f64);
    }
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
    let cos = if na > 0.0 && nb > 0.0 { dot / (na.sqrt() * nb.sqrt()) } else { 1.0 };
    let min_cos = min_cos as f64;
    assert!(
        max_err / scale <= rel && cos >= min_cos,
        "{name}: max_err {max_err:.3e}（相对 {:.3e}，at {worst}: {} vs {}）cos {cos:.7}（要求 <= {rel:.1e} / >= {min_cos}）",
        max_err / scale,
        got[worst],
        want[worst]
    );
    println!("  {name:<28} max_err {max_err:.3e}  rel {:.3e}  cos {cos:.7}", max_err / scale);
}

const BT: usize = 64;

/// 跑 chunk 形式全链（prep → wy → ho），返回 (out, state)。
#[allow(clippy::too_many_arguments)]
fn run_chunk(
    st: &Stream,
    q: &[f32], k: &[f32], v: &[f32], raw_g: &[f32], raw_beta: &[f32],
    a_log: &[f32], dt_bias: &[f32], state0: &[f32], cu: &[i32],
    heads: usize, head_dim: usize, lower_bound: f32,
) -> (Vec<bf16>, Vec<f32>) {
    let tokens = q.len() / (heads * head_dim);
    let plan = ops::KdaChunkPlan::new(st, cu, BT).unwrap();
    let mut sc = ops::KdaChunkScratch::new(st, tokens, heads, head_dim, plan.num_chunks, BT).unwrap();
    let dq = up_bf16(st, q);
    let dk = up_bf16(st, k);
    let dv = up_bf16(st, v);
    let dg = up_bf16(st, raw_g);
    let dbeta = st.htod_copy(raw_beta.to_vec()).unwrap();
    let da = st.htod_copy(a_log.to_vec()).unwrap();
    let db = st.htod_copy(dt_bias.to_vec()).unwrap();
    let dcu = st.htod_copy(cu.to_vec()).unwrap();
    let mut dout = st.alloc_uninit::<bf16>(q.len()).unwrap();
    let mut dstate = st.htod_copy(state0.to_vec()).unwrap();
    let scale = (head_dim as f32).powf(-0.5);
    ops::kda_chunk_prep(st, &dq, &dk, &dg, &dbeta, &da, &db, &dcu, &plan, &mut sc, tokens, heads, head_dim, scale, Some(lower_bound)).unwrap();
    ops::kda_chunk_wy(st, &dv, &dcu, &plan, &mut sc, tokens, heads, head_dim).unwrap();
    ops::kda_chunk_ho(st, &dcu, &plan, &sc, &mut dout, &mut dstate, None, tokens, heads, head_dim).unwrap();
    st.synchronize().unwrap();
    (st.dtoh_sync_copy(&dout).unwrap(), st.dtoh_sync_copy(&dstate).unwrap())
}

fn chunk_case(dir: &str) {
    let (dev, st) = dev();
    let _ = &dev;
    let (qs, q) = load(dir, "q");
    let (_, k) = load(dir, "k");
    let (_, v) = load(dir, "v");
    let (_, raw_g) = load(dir, "raw_g");
    let (_, raw_beta) = load(dir, "raw_beta");
    let (_, a_log) = load(dir, "a_log");
    let (_, dt_bias) = load(dir, "dt_bias");
    let (_, want) = load(dir, "out");
    let (_, want_state) = load(dir, "state_out");
    let (_, st_in) = load(dir, "state_in");
    let cu = load_i32(dir, "cu_seqlens");
    let heads = qs[1];
    let head_dim = qs[2];
    let (got, got_state) =
        run_chunk(&st, &q, &k, &v, &raw_g, &raw_beta, &a_log, &dt_bias, &st_in, &cu, heads, head_dim, -5.0);
    // out：atol 1e-2 + cos ≥ 0.9999
    assert_close(&format!("{dir}/out"), &got, &want, 1e-2, 0.9999);
    // state：相对 5e-3 + cos ≥ 0.9999
    assert_close_rel(&format!("{dir}/state"), &got_state, &want_state, 5e-3, 0.9999);
}

#[test]
#[ignore]
fn kda_chunk_varlen() { chunk_case("recurrent_chunk"); }

#[test]
#[ignore]
fn kda_chunk_tail() { chunk_case("recurrent_chunk_tail"); }

/// 计时：K3 真实形状 T=1024 / 96 头 / head_dim 128，chunk 全链 vs 顺序递归。
/// 门槛见 `docs/loops/k3-kda.md`：chunk ≤ 顺序版 1/5。
#[test]
#[ignore]
fn kda_chunk_bench() {
    let (dev, st) = dev();
    let _ = &dev;
    let (t, heads, kd) = (1024usize, 96usize, 128usize);
    let n = t * heads * kd;
    // 用真实量级的 gate：-5·sigmoid(a·x)
    let raw_g: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 - 48.0) / 32.0).collect();
    let q = vec![0.03f32; n];
    let v = vec![0.03f32; n];
    let raw_beta = vec![0.5f32; t * heads];
    let a_log: Vec<f32> = (0..heads).map(|h| (0.2 + 0.16 * h as f32).ln()).collect();
    let dt_bias = vec![0.1f32; heads * kd];
    let cu = vec![0i32, t as i32];
    let scale = (kd as f32).powf(-0.5);

    // 顺序版基线
    let dq = up_bf16(&st, &q);
    let dk = up_bf16(&st, &q);
    let dv = up_bf16(&st, &v);
    let dg = up_bf16(&st, &raw_g);
    let dbeta = st.htod_copy(raw_beta.clone()).unwrap();
    let da = st.htod_copy(a_log.clone()).unwrap();
    let db = st.htod_copy(dt_bias.clone()).unwrap();
    let dcu = st.htod_copy(cu.clone()).unwrap();
    let mut out_seq = st.alloc_uninit::<bf16>(n).unwrap();
    let mut state_seq = st.alloc_zeros::<f32>(heads * kd * kd).unwrap();
    let seq_once = |st: &Stream, out: &mut _, state: &mut _| {
        ops::kda_recurrent(st, &dq, &dk, &dv, &dg, &dbeta, &da, &db, out, state, &dcu, None, 1, heads, kd, scale, Some(-5.0)).unwrap();
    };
    seq_once(&st, &mut out_seq, &mut state_seq);
    st.synchronize().unwrap();
    let iters = 3;
    let t0 = std::time::Instant::now();
    for _ in 0..iters { seq_once(&st, &mut out_seq, &mut state_seq); }
    st.synchronize().unwrap();
    let ms_seq = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

    // chunk 全链
    let plan = ops::KdaChunkPlan::new(&st, &cu, BT).unwrap();
    let mut sc = ops::KdaChunkScratch::new(&st, t, heads, kd, plan.num_chunks, BT).unwrap();
    let mut out_ch = st.alloc_uninit::<bf16>(n).unwrap();
    let mut state_ch = st.alloc_zeros::<f32>(heads * kd * kd).unwrap();
    let chunk_once = |st: &Stream, sc: &mut ops::KdaChunkScratch, out: &mut _, state: &mut _| {
        ops::kda_chunk_prep(st, &dq, &dk, &dg, &dbeta, &da, &db, &dcu, &plan, sc, t, heads, kd, scale, Some(-5.0)).unwrap();
        ops::kda_chunk_wy(st, &dv, &dcu, &plan, sc, t, heads, kd).unwrap();
        ops::kda_chunk_ho(st, &dcu, &plan, sc, out, state, None, t, heads, kd).unwrap();
    };
    chunk_once(&st, &mut sc, &mut out_ch, &mut state_ch);
    st.synchronize().unwrap();
    let t1 = std::time::Instant::now();
    for _ in 0..iters { chunk_once(&st, &mut sc, &mut out_ch, &mut state_ch); }
    st.synchronize().unwrap();
    let ms_chunk = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;

    let speedup = ms_seq / ms_chunk;
    println!("  T={t} H={heads} K={kd}: 顺序 {ms_seq:.3} ms, chunk {ms_chunk:.3} ms, 加速 {speedup:.2}x（门槛 5x）");
    assert!(speedup >= 5.0, "chunk 加速 {speedup:.2}x < 5x（顺序 {ms_seq:.3} ms / chunk {ms_chunk:.3} ms）");
}
