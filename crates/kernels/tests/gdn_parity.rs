//! GDN 算子对拍（Qwen3.5 线）。参考由 `scripts/verify/gen_gdn_refs.py` 生成（纯 torch，与 vLLM / fla Triton 同式）。
//!
//! 运行：`GDN_REFS_DIR=<dir> cargo test -p dinfer-kernels --features ref-tests --test gdn_parity -- --ignored --test-threads=1`
//! 汇总入口 `scripts/verify/qwen35_gdn.sh`（exit 0 = 通过）。
//!
//! 容差口径（`docs/pitfalls.md`）：软判决比 cos 与 max abs，不比逐位；bf16 出 1e-2/0.999，
//! f32 递归状态 2e-3（状态是累乘结果，比单步输出敏感）。
#![cfg(feature = "ref-tests")]

use dinfer_kernels::{bf16, ops, ops::GdnAct, Device, Stream};
use std::path::PathBuf;

fn refs_root() -> PathBuf {
    let p = PathBuf::from(std::env::var("GDN_REFS_DIR").unwrap_or_else(|_| "test/qwen35/2026-10-19/gdn_refs".into()));
    if p.is_absolute() {
        p
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join(p)
    }
}

fn load(dir: &str, name: &str) -> (Vec<usize>, Vec<f32>) {
    let base = refs_root().join(dir);
    let pb = base.join(format!("{name}_bf16.npy"));
    let bytes = match std::fs::read(&pb) {
        Ok(b) => b,
        Err(_) => std::fs::read(base.join(format!("{name}.npy"))).unwrap_or_else(|e| panic!("{}: {e}", pb.display())),
    };
    let arr = npyz::NpyFile::new(&bytes[..]).unwrap();
    let shape: Vec<usize> = arr.shape().iter().map(|&d| d as usize).collect();
    // bf16 以原始位存成 `u2`（见 `gen_gdn_refs.py` 的 save）。npyz 的 into_vec 只认精确的 Rust 类型，
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

fn up_bf16(st: &Stream, v: &[f32]) -> dinfer_kernels::cudarc::driver::CudaSlice<bf16> {
    st.htod_copy(v.iter().map(|&x| bf16::from_f32(x)).collect()).unwrap()
}

/// 软判决：max abs + cos 相似度。
fn assert_close(name: &str, got: &[bf16], want: &[f32], atol: f32, min_cos: f64) {
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
    assert!(
        max_err <= atol && cos >= min_cos,
        "{name}: max_err {max_err:.6} (at {worst}: {} vs {}) cos {cos:.7}（要求 <= {atol} / >= {min_cos}）",
        got[worst].to_f32(),
        want[worst]
    );
    println!("  {name:<30} max_err {max_err:.6}  cos {cos:.7}");
}

fn assert_close_f32(name: &str, got: &[f32], want: &[f32], atol: f32, min_cos: f64) {
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
    let cos = if na > 0.0 && nb > 0.0 { dot / (na.sqrt() * nb.sqrt()) } else { 1.0 };
    assert!(
        max_err <= atol && cos >= min_cos,
        "{name}: max_err {max_err:.7} (at {worst}: {} vs {}) cos {cos:.7}（要求 <= {atol} / >= {min_cos}）",
        got[worst],
        want[worst]
    );
    println!("  {name:<30} max_err {max_err:.7}  cos {cos:.7}");
}

fn dev() -> (Device, Stream) {
    let d = Device::new(0).unwrap();
    let s = d.default_stream();
    (d, s)
}

// --------------------------------------------------------------------------- cases

fn conv_case(tag: &str, k: usize, v: usize, h: usize, hv: usize, width: usize) {
    let (dev, st) = dev();
    let _ = dev;
    let dir = format!("conv1d_{tag}");
    let (xs, x) = load(&dir, "x");
    let (_, w) = load(&dir, "w");
    let (_, st_in) = load(&dir, "state_in");
    let (_, want) = load(&dir, "out");
    let (_, want_st) = load(&dir, "state_out");
    let cu = load_i32(&dir, "cu_seqlens");
    let hi = load_i32(&dir, "has_initial");
    let si = load_i32(&dir, "state_indices");
    let dim = xs[1];
    assert_eq!(dim, 2 * h * k + hv * v, "conv1d/{tag}: dim 与形状参数不符");
    let state_len = width - 1;
    let num_seqs = cu.len() - 1;

    let dx = up_bf16(&st, &x);
    let dw = up_bf16(&st, &w);
    let mut dst = st.htod_copy(st_in.iter().map(|&x| bf16::from_f32(x)).collect::<Vec<_>>()).unwrap();
    let dcu = st.htod_copy(cu).unwrap();
    let dhi = st.htod_copy(hi).unwrap();
    let dsi = st.htod_copy(si).unwrap();
    let mut dout = st.alloc_uninit::<bf16>(x.len()).unwrap();
    ops::gdn_conv1d_prefill(
        &st, &dx, &dw, &mut dout, &mut dst, &dcu, Some(&dsi), Some(&dhi), num_seqs, dim, width,
        state_len, GdnAct::Silu,
    )
    .unwrap();
    st.synchronize().unwrap();
    assert_close(&format!("conv1d_{tag}/out"), &st.dtoh_sync_copy(&dout).unwrap(), &want, 1e-2, 0.999);
    let got_st: Vec<f32> = st.dtoh_sync_copy(&dst).unwrap().iter().map(|b| b.to_f32()).collect();
    assert_close_f32(&format!("conv1d_{tag}/state"), &got_st, &want_st, 1e-2, 0.999);

    // update 连续 4 步（decode）：状态继续累积
    let udir = format!("conv_update_{tag}");
    let (_, w_u) = load(&udir, "w");
    let (_, st_u) = load(&udir, "state_in");
    let si_u = load_i32(&udir, "state_indices");
    let (xs_u, _) = load(&udir, "x0");
    let dim_u = xs_u[1];
    let batch = si_u.len();
    let dwu = up_bf16(&st, &w_u);
    let dsiu = st.htod_copy(si_u).unwrap();
    let mut dst_u = st.htod_copy(st_u.iter().map(|&x| bf16::from_f32(x)).collect::<Vec<_>>()).unwrap();
    let mut nstep = 0;
    while refs_root().join(&udir).join(format!("x{nstep}.npy")).exists()
        || refs_root().join(&udir).join(format!("x{nstep}_bf16.npy")).exists()
    {
        let (_, xk) = load(&udir, &format!("x{nstep}"));
        let (_, want_k) = load(&udir, &format!("out{nstep}"));
        let (_, want_s) = load(&udir, &format!("state{nstep}"));
        let dxk = up_bf16(&st, &xk);
        let mut dok = st.alloc_uninit::<bf16>(xk.len()).unwrap();
        ops::gdn_conv1d_update(&st, &dxk, &dwu, &mut dok, &mut dst_u, Some(&dsiu), batch, dim_u, width,
                               state_len, GdnAct::Silu)
            .unwrap();
        st.synchronize().unwrap();
        assert_close(&format!("conv_update_{tag}/out{nstep}"), &st.dtoh_sync_copy(&dok).unwrap(), &want_k, 1e-2, 0.999);
        let got_s: Vec<f32> = st.dtoh_sync_copy(&dst_u).unwrap().iter().map(|b| b.to_f32()).collect();
        assert_close_f32(&format!("conv_update_{tag}/state{nstep}"), &got_s, &want_s, 1e-2, 0.999);
        nstep += 1;
    }
    assert!(nstep >= 2, "conv_update_{tag}: 没有找到任何 update 步");
}

#[test]
#[ignore]
fn gdn_conv1d_qwen_shape() {
    conv_case("qwen", 128, 128, 16, 48, 4);
}

#[test]
#[ignore]
fn gdn_conv1d_small_shape() {
    conv_case("small", 64, 32, 2, 4, 3);
}

fn gate_case(tag: &str, hv: usize) {
    let (dev, st) = dev();
    let _ = dev;
    let dir = format!("gate_{tag}");
    let (as_, a) = load(&dir, "a");
    let (_, b) = load(&dir, "b");
    let (_, a_log) = load(&dir, "a_log");
    let (_, dt_bias) = load(&dir, "dt_bias");
    let (_, want_g) = load(&dir, "g");
    let (_, want_beta) = load(&dir, "beta");
    let tokens = as_[0];
    assert_eq!(as_[1], hv);
    let da = up_bf16(&st, &a);
    let db = up_bf16(&st, &b);
    let dal = st.htod_copy(a_log).unwrap();
    let ddb = st.htod_copy(dt_bias).unwrap();
    let mut dg = st.alloc_uninit::<f32>(tokens * hv).unwrap();
    let mut dbeta = st.alloc_uninit::<f32>(tokens * hv).unwrap();
    ops::gdn_gate(&st, &da, &db, &dal, &ddb, &mut dg, &mut dbeta, tokens, hv).unwrap();
    st.synchronize().unwrap();
    assert_close_f32(&format!("gate_{tag}/g"), &st.dtoh_sync_copy(&dg).unwrap(), &want_g, 1e-4, 0.999);
    assert_close_f32(&format!("gate_{tag}/beta"), &st.dtoh_sync_copy(&dbeta).unwrap(), &want_beta, 1e-4, 0.999);
}

#[test]
#[ignore]
fn gdn_gate_qwen_shape() {
    gate_case("qwen", 48);
}

#[test]
#[ignore]
fn gdn_gate_small_shape() {
    gate_case("small", 4);
}

fn recurrent_case(tag: &str, decode: bool, k: usize, v: usize, h: usize, hv: usize) {
    let (dev, st) = dev();
    let _ = dev;
    let dir = format!("recurrent_{tag}{}", if decode { "_decode" } else { "" });
    let (_, a_log) = load(&dir, "a_log");
    let (_, dt_bias) = load(&dir, "dt_bias");
    let (ss, st_in) = load(&dir, "state_in");
    let slots = ss[0];
    let cu0 = load_i32(&dir, "cu_seqlens");
    let dal = st.htod_copy(a_log).unwrap();
    let ddb = st.htod_copy(dt_bias).unwrap();
    let mut dstate = st.htod_copy(st_in).unwrap();
    let si: Vec<i32> = (0..slots as i32).collect();
    let dsi = st.htod_copy(si).unwrap();
    let scale = (k as f32).powf(-0.5);

    let steps = if decode {
        let mut n = 0;
        while refs_root().join(&dir).join(format!("mixed{n}.npy")).exists()
            || refs_root().join(&dir).join(format!("mixed{n}_bf16.npy")).exists()
        {
            n += 1;
        }
        n
    } else {
        1
    };
    assert!(steps >= 1, "recurrent_{tag}: 没有找到任何步");
    for step in 0..steps {
        let mixed_name = if decode { format!("mixed{step}") } else { "mixed".to_string() };
        let (ms, mixed) = load(&dir, &mixed_name);
        let names: Vec<(&str, String)> = if decode {
            vec![("a", format!("a{step}")), ("b", format!("b{step}")), ("out", format!("out{step}")), ("state", format!("state{step}"))]
        } else {
            vec![("a", "a".into()), ("b", "b".into()), ("out", "out".into()), ("state", "state_out".into())]
        };
        let name_of = |k: &str| names.iter().find(|(n, _)| *n == k).unwrap().1.clone();
        let (_, a) = load(&dir, &name_of("a"));
        let (_, b) = load(&dir, &name_of("b"));
        let (_, want) = load(&dir, &name_of("out"));
        let (_, want_st) = load(&dir, &name_of("state"));
        let tokens = ms[0];
        let row = ms[1];
        assert_eq!(row, 2 * h * k + hv * v, "recurrent/{tag}: mixed 宽度与形状参数不符");
        let dcu = st.htod_copy(cu0.clone()).unwrap();
        let dm = up_bf16(&st, &mixed);
        let da = up_bf16(&st, &a);
        let db = up_bf16(&st, &b);
        let mut dout = st.alloc_uninit::<bf16>(tokens * hv * v).unwrap();
        ops::gdn_recurrent(
            &st, &dm, None, None, Some(&da), Some(&db), &dal, &ddb, &mut dout, &mut dstate, &dcu,
            Some(&dsi), cu0.len() - 1, h, hv, k, v, scale,
        )
        .unwrap();
        st.synchronize().unwrap();
        assert_close(&format!("recurrent_{tag}/out{step}"), &st.dtoh_sync_copy(&dout).unwrap(), &want, 1e-2, 0.999);
        let got_st: Vec<f32> = st.dtoh_sync_copy(&dstate).unwrap();
        assert_close_f32(&format!("recurrent_{tag}/state{step}"), &got_st, &want_st, 2e-3, 0.999);
    }
}

#[test]
#[ignore]
fn gdn_recurrent_prefill_qwen_shape() {
    recurrent_case("qwen", false, 128, 128, 4, 12);
}

#[test]
#[ignore]
fn gdn_recurrent_decode_qwen_shape() {
    recurrent_case("qwen", true, 128, 128, 4, 12);
}

#[test]
#[ignore]
fn gdn_recurrent_prefill_small_shape() {
    recurrent_case("small", false, 64, 32, 2, 4);
}

#[test]
#[ignore]
fn gdn_recurrent_decode_small_shape() {
    recurrent_case("small", true, 64, 32, 2, 4);
}

fn norm_case(act: &str, act_enum: GdnAct, tag: &str, v: usize) {
    let (dev, st) = dev();
    let _ = dev;
    let dir = format!("gated_rmsnorm_{act}_{tag}");
    let (xs, x) = load(&dir, "x");
    let (_, w) = load(&dir, "w");
    let (_, z) = load(&dir, "z");
    let (_, want) = load(&dir, "out");
    let rows = xs[0];
    let head_dim = xs[1];
    assert_eq!(head_dim, v);
    let dx = up_bf16(&st, &x);
    let dw = up_bf16(&st, &w);
    let dz = up_bf16(&st, &z);
    let mut dout = st.alloc_uninit::<bf16>(x.len()).unwrap();
    ops::gdn_gated_rmsnorm(&st, &dx, &dw, &dz, &mut dout, rows, head_dim, 1e-6, act_enum).unwrap();
    st.synchronize().unwrap();
    assert_close(&format!("gated_rmsnorm_{act}_{tag}"), &st.dtoh_sync_copy(&dout).unwrap(), &want, 1e-2, 0.999);
}

#[test]
#[ignore]
fn gdn_gated_rmsnorm_silu_qwen_shape() {
    norm_case("silu", GdnAct::Silu, "qwen", 128);
}

#[test]
#[ignore]
fn gdn_gated_rmsnorm_sigmoid_small_shape() {
    norm_case("sigmoid", GdnAct::Sigmoid, "small", 32);
}

// --------------------------------------------------------------------------- chunk 形式 prefill（gdn_chunk.cu）
// 参考仍是顺序递归（gen_gdn_refs.py 的 gdn_ref）：这正是 chunk 版等价性的证明。
// 容差（任务口径，比 KDA chunk 的 1e-2 严）：out bf16 atol 5e-4 + cos ≥ 0.9999；state 相对 |want|max 5e-3 + cos ≥ 0.9999。

const BT: usize = 64;

fn assert_close_rel(name: &str, got: &[f32], want: &[f32], rel: f32, min_cos: f64) {
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
    assert!(
        max_err / scale <= rel && cos >= min_cos,
        "{name}: max_err {max_err:.3e}（相对 {:.3e}，at {worst}: {} vs {}）cos {cos:.7}（要求 <= {rel:.1e} / >= {min_cos}）",
        max_err / scale,
        got[worst],
        want[worst]
    );
    println!("  {name:<28} max_err {max_err:.3e}  rel {:.3e}  cos {cos:.7}", max_err / scale);
}

/// 跑 chunk 全链（gate → prep → wy → ho），`state` 原地更新，返回 out。
#[allow(clippy::too_many_arguments)]
fn run_gdn_chunk(
    st: &Stream,
    mixed: &dinfer_kernels::cudarc::driver::CudaSlice<bf16>,
    a: &dinfer_kernels::cudarc::driver::CudaSlice<bf16>,
    b: &dinfer_kernels::cudarc::driver::CudaSlice<bf16>,
    a_log: &dinfer_kernels::cudarc::driver::CudaSlice<f32>,
    dt_bias: &dinfer_kernels::cudarc::driver::CudaSlice<f32>,
    state: &mut dinfer_kernels::cudarc::driver::CudaSlice<f32>,
    cu: &[i32],
    tokens: usize,
    h: usize,
    hv: usize,
    k: usize,
    v: usize,
) -> dinfer_kernels::cudarc::driver::CudaSlice<bf16> {
    let plan = ops::KdaChunkPlan::new(st, cu, BT).unwrap();
    let mut sc = ops::GdnChunkScratch::new(st, tokens, h, hv, k, v, plan.num_chunks, BT).unwrap();
    let mut g = st.alloc_uninit::<f32>(tokens * hv).unwrap();
    let mut beta = st.alloc_uninit::<f32>(tokens * hv).unwrap();
    let dcu = st.htod_copy(cu.to_vec()).unwrap();
    let mut out = st.alloc_uninit::<bf16>(tokens * hv * v).unwrap();
    let scale = (k as f32).powf(-0.5);
    ops::gdn_gate(st, a, b, a_log, dt_bias, &mut g, &mut beta, tokens, hv).unwrap();
    ops::gdn_chunk_prep(st, mixed, &g, &dcu, &plan, &mut sc, tokens, h, hv, k, v, scale).unwrap();
    ops::gdn_chunk_wy(st, mixed, &beta, &dcu, &plan, &mut sc, tokens, h, hv, k, v).unwrap();
    ops::gdn_chunk_ho(st, &dcu, &plan, &sc, &mut out, state, None, tokens, h, hv, k, v).unwrap();
    st.synchronize().unwrap();
    out
}

fn chunk_case(dir: &str, k: usize, v: usize, h: usize, hv: usize) {
    let (dev, st) = dev();
    let _ = dev;
    let (_, a_log) = load(dir, "a_log");
    let (_, dt_bias) = load(dir, "dt_bias");
    let cu = load_i32(dir, "cu_seqlens");
    let slots = cu.len() - 1;
    // chunk_qwen 的 state_in 不落盘（3.1 MB/槽）：缺文件 = 零初始
    let st_in = if refs_root().join(dir).join("state_in.npy").exists() {
        load(dir, "state_in").1
    } else {
        vec![0f32; slots * hv * v * k]
    };
    let (ms, mixed) = load(dir, "mixed");
    let (_, a) = load(dir, "a");
    let (_, b) = load(dir, "b");
    let (_, want) = load(dir, "out");
    let (_, want_st) = load(dir, "state_out");
    let tokens = ms[0];
    assert_eq!(ms[1], 2 * h * k + hv * v, "{dir}: mixed 宽度与形状参数不符");
    let dal = st.htod_copy(a_log).unwrap();
    let ddb = st.htod_copy(dt_bias).unwrap();
    let mut dstate = st.htod_copy(st_in).unwrap();
    let dm = up_bf16(&st, &mixed);
    let da = up_bf16(&st, &a);
    let db = up_bf16(&st, &b);
    let out = run_gdn_chunk(&st, &dm, &da, &db, &dal, &ddb, &mut dstate, &cu, tokens, h, hv, k, v);
    assert_close(&format!("{dir}/out"), &st.dtoh_sync_copy(&out).unwrap(), &want, 5e-4, 0.9999);
    let got_st: Vec<f32> = st.dtoh_sync_copy(&dstate).unwrap();
    assert_close_rel(&format!("{dir}/state"), &got_st, &want_st, 5e-3, 0.9999);
}

#[test]
#[ignore]
fn gdn_chunk_single_t200() {
    chunk_case("chunk", 128, 128, 4, 12);
}

#[test]
#[ignore]
fn gdn_chunk_varlen() {
    chunk_case("chunk_varlen", 128, 128, 4, 12);
}

#[test]
#[ignore]
fn gdn_chunk_qwen_shape() {
    chunk_case("chunk_qwen", 128, 128, 16, 48);
}

/// 计时：27B 形状 T=1024 / H=16 HV=48 / K=V=128，顺序递归 vs chunk 全链（都含 gate）。
/// 数字记到 `docs/loops/qwen35-perf.md` 记录表。门槛：chunk ≤ 顺序 1/5。
#[test]
#[ignore]
fn gdn_chunk_bench() {
    let (dev, st) = dev();
    let _ = dev;
    let (t, h, hv, k, v) = (1024usize, 16usize, 48usize, 128usize, 128usize);
    let row = 2 * h * k + hv * v;
    let mixed: Vec<f32> = (0..t * row).map(|i| (((i * 7919) % 1000) as f32 / 1000.0 - 0.5) * 0.5).collect();
    let a: Vec<f32> = (0..t * hv).map(|i| ((i % 23) as f32 - 11.0) / 4.0).collect();
    let b: Vec<f32> = (0..t * hv).map(|i| ((i % 17) as f32 - 8.0) / 4.0).collect();
    let a_log: Vec<f32> = (0..hv).map(|i| (0.2 + 0.03 * i as f32).ln()).collect();
    let dt_bias = vec![0.1f32; hv];
    let cu = vec![0i32, t as i32];
    let scale = (k as f32).powf(-0.5);
    let dm = up_bf16(&st, &mixed);
    let da = up_bf16(&st, &a);
    let db = up_bf16(&st, &b);
    let dal = st.htod_copy(a_log).unwrap();
    let ddb = st.htod_copy(dt_bias).unwrap();
    let dcu = st.htod_copy(cu.clone()).unwrap();
    let iters = 5;

    // 顺序版（gate + recurrent）
    let mut g = st.alloc_uninit::<f32>(t * hv).unwrap();
    let mut beta = st.alloc_uninit::<f32>(t * hv).unwrap();
    let mut out_seq = st.alloc_uninit::<bf16>(t * hv * v).unwrap();
    let mut state_seq = st.alloc_zeros::<f32>(hv * v * k).unwrap();
    let seq_once = |st: &Stream, g: &mut _, beta: &mut _, out: &mut _, state: &mut _| {
        ops::gdn_gate(st, &da, &db, &dal, &ddb, g, beta, t, hv).unwrap();
        ops::gdn_recurrent(st, &dm, Some(&*g), Some(&*beta), None, None, &dal, &ddb, out, state, &dcu, None, 1, h, hv, k, v, scale).unwrap();
    };
    seq_once(&st, &mut g, &mut beta, &mut out_seq, &mut state_seq);
    st.synchronize().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        seq_once(&st, &mut g, &mut beta, &mut out_seq, &mut state_seq);
    }
    st.synchronize().unwrap();
    let ms_seq = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

    // chunk 全链（gate + prep + wy + ho）
    let plan = ops::KdaChunkPlan::new(&st, &cu, BT).unwrap();
    let mut sc = ops::GdnChunkScratch::new(&st, t, h, hv, k, v, plan.num_chunks, BT).unwrap();
    let mut out_ch = st.alloc_uninit::<bf16>(t * hv * v).unwrap();
    let mut state_ch = st.alloc_zeros::<f32>(hv * v * k).unwrap();
    let chunk_once = |st: &Stream, g: &mut _, beta: &mut _, sc: &mut ops::GdnChunkScratch, out: &mut _, state: &mut _| {
        ops::gdn_gate(st, &da, &db, &dal, &ddb, g, beta, t, hv).unwrap();
        ops::gdn_chunk_prep(st, &dm, &*g, &dcu, &plan, sc, t, h, hv, k, v, scale).unwrap();
        ops::gdn_chunk_wy(st, &dm, &*beta, &dcu, &plan, sc, t, h, hv, k, v).unwrap();
        ops::gdn_chunk_ho(st, &dcu, &plan, sc, out, state, None, t, h, hv, k, v).unwrap();
    };
    chunk_once(&st, &mut g, &mut beta, &mut sc, &mut out_ch, &mut state_ch);
    st.synchronize().unwrap();
    let t1 = std::time::Instant::now();
    for _ in 0..iters {
        chunk_once(&st, &mut g, &mut beta, &mut sc, &mut out_ch, &mut state_ch);
    }
    st.synchronize().unwrap();
    let ms_chunk = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;

    let speedup = ms_seq / ms_chunk;
    println!("  T={t} H={h} HV={hv} K={k} V={v}: 顺序 {ms_seq:.3} ms, chunk {ms_chunk:.3} ms, 加速 {speedup:.2}x（门槛 5x）");
    assert!(speedup >= 5.0, "chunk 加速 {speedup:.2}x < 5x（顺序 {ms_seq:.3} ms / chunk {ms_chunk:.3} ms）");
}
