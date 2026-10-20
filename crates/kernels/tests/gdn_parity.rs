//! GDN 算子对拍（Qwen3.5 线）。参考由 `scripts/verify/gen_gdn_refs.py` 生成（纯 torch，与 vLLM / fla Triton 同式）。
//!
//! 运行：`GDN_REFS_DIR=<dir> cargo test -p dinfer-kernels --features ref-tests --test gdn_parity -- --ignored --test-threads=1`
//! 汇总入口 `scripts/verify/qwen35_gdn.sh`（exit 0 = 通过）。
//!
//! 容差口径（`docs/pitfalls.md`）：软判决比 cos 与 max abs，不比逐位；bf16 出 1e-2/0.999，
//! f32 递归状态 2e-3（状态是累乘结果，比单步输出敏感）。
#![cfg(feature = "ref-tests")]

use dinfer_kernels::arch::GdnDecodeVariant;
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
    let dcu = st.htod_copy(cu).unwrap();
    let dhi = st.htod_copy(hi).unwrap();
    let dsi = st.htod_copy(si).unwrap();
    let mut dout = st.alloc_uninit::<bf16>(x.len()).unwrap();
    let tokens = xs[0];
    // 老 kernel（rows=0）与新按 token 分块 kernel（rows ∈ {width-1（极端小块，逼出跨块 halo / 状态补齐）, 32, 默认}）都对参考
    let mut variants: Vec<(String, usize)> = vec![("legacy".into(), 0), (format!("rows{}", width - 1), width - 1), ("rows32".into(), 32)];
    variants.push(("default".into(), usize::MAX));
    for (vname, rows) in variants {
        let mut dst = st.htod_copy(st_in.iter().map(|&x| bf16::from_f32(x)).collect::<Vec<_>>()).unwrap();
        if rows == usize::MAX {
            ops::gdn_conv1d_prefill(
                &st, &dx, &dw, &mut dout, &mut dst, &dcu, Some(&dsi), Some(&dhi), num_seqs, tokens, dim, width,
                state_len, GdnAct::Silu,
            )
            .unwrap();
        } else {
            ops::gdn_conv1d_prefill_with(
                &st, &dx, &dw, &mut dout, &mut dst, &dcu, Some(&dsi), Some(&dhi), num_seqs, tokens, dim, width,
                state_len, GdnAct::Silu, rows,
            )
            .unwrap();
        }
        st.synchronize().unwrap();
        assert_close(&format!("conv1d_{tag}/{vname}/out"), &st.dtoh_sync_copy(&dout).unwrap(), &want, 1e-2, 0.999);
        let got_st: Vec<f32> = st.dtoh_sync_copy(&dst).unwrap().iter().map(|b| b.to_f32()).collect();
        assert_close_f32(&format!("conv1d_{tag}/{vname}/state"), &got_st, &want_st, 1e-2, 0.999);
    }
    let mut dst = st.htod_copy(st_in.iter().map(|&x| bf16::from_f32(x)).collect::<Vec<_>>()).unwrap();
    ops::gdn_conv1d_prefill(
        &st, &dx, &dw, &mut dout, &mut dst, &dcu, Some(&dsi), Some(&dhi), num_seqs, tokens, dim, width,
        state_len, GdnAct::Silu,
    )
    .unwrap();
    st.synchronize().unwrap();

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
            Some(&dsi), cu0.len() - 1, tokens, h, hv, k, v, scale,
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
    ho: ops::GdnChunkHo,
    wy_mma: bool,
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
    ops::gdn_chunk_wy_with(st, mixed, &beta, &dcu, &plan, &mut sc, tokens, h, hv, k, v, wy_mma).unwrap();
    ops::gdn_chunk_ho_with(st, &dcu, &plan, &sc, &mut out, state, None, tokens, h, hv, k, v, ho).unwrap();
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
    let dm = up_bf16(&st, &mixed);
    let da = up_bf16(&st, &a);
    let db = up_bf16(&st, &b);
    // ho：老 fp32 版（bv=0）与 tensor core 版（bv 16 / 32，第 5A 轮）都对参考。tensor core 版 h / w / kg / Aq / v' 过 bf16，
    // 容差放宽到 out 2e-3 / state rel 1e-2（fla 同口径；参考是 fp32 顺序递归）。
    // (wy 实现, ho bv, ho 预取, out 容差, state rel 容差)：老 fp32 全链；新 ho（预取 / 不预取，r9 第 1 轮）；新 wy + 新 ho（默认路径）
    // 第 7 列 split：两段版 h-pass + o-pass（r9 第 2 轮；仅 K=V=128 / 64 有 o-pass 特化，其余形状跳过）
    let split_ok = matches!(k, 64 | 128 | 256) && v % 16 == 0;
    for (wy, bv, pf, split, tol_o, tol_s) in [
        (false, 0usize, false, false, 5e-4f32, 5e-3f32),
        (false, 16, false, false, 2e-3, 1e-2),
        (false, 32, false, false, 2e-3, 1e-2),
        (false, 16, true, false, 2e-3, 1e-2),
        (false, 32, true, false, 2e-3, 1e-2),
        (true, 64, false, false, 2e-3, 1e-2),   // r10 第 6 轮：BV=64（仅 K=128 有特化）
        (true, 64, true, false, 2e-3, 1e-2),
        (true, 32, false, false, 2e-3, 1e-2),
        (true, 16, true, false, 2e-3, 1e-2),
        (false, 16, false, true, 2e-3, 1e-2),
        (false, 32, true, true, 2e-3, 1e-2),
        (true, 32, true, true, 2e-3, 1e-2),
        (true, 16, false, true, 2e-3, 1e-2),
    ] {
        if split && !split_ok {
            continue;
        }
        if bv == 64 && !(k == 128 && v % 64 == 0) {
            continue;
        }
        let mut dstate = st.htod_copy(st_in.clone()).unwrap();
        let ho = ops::GdnChunkHo { bv, prefetch: pf, split };
        let out = run_gdn_chunk(&st, &dm, &da, &db, &dal, &ddb, &mut dstate, &cu, tokens, h, hv, k, v, ho, wy);
        let tag = format!("{dir}/wy{}ho{bv}{}{}", if wy { "mma" } else { "32" }, if pf { "pf" } else { "" }, if split { "sp" } else { "" });
        assert_close(&format!("{tag}/out"), &st.dtoh_sync_copy(&out).unwrap(), &want, tol_o, 0.9999);
        let got_st: Vec<f32> = st.dtoh_sync_copy(&dstate).unwrap();
        assert_close_rel(&format!("{tag}/state"), &got_st, &want_st, tol_s, 0.9999);
    }
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

/// wy 老 fp32 版 vs tensor core 版：直接比 scratch 的 aq / u / w / kg（定位用）。
#[test]
#[ignore]
fn gdn_chunk_wy_mma_vs_legacy() {
    let (dev, st) = dev();
    let _ = dev;
    let (h, hv, k, v) = (4usize, 12usize, 128usize, 128usize);
    let cu = vec![0i32, 130, 194, 201];
    let t = 201usize;
    let row = 2 * h * k + hv * v;
    let mixed: Vec<f32> = (0..t * row).map(|i| (((i * 7919) % 1000) as f32 / 1000.0 - 0.5) * 0.5).collect();
    let g: Vec<f32> = (0..t * hv).map(|i| -0.01 - ((i % 23) as f32) / 200.0).collect();
    let beta: Vec<f32> = (0..t * hv).map(|i| 0.3 + ((i % 17) as f32) / 40.0).collect();
    let dm = up_bf16(&st, &mixed);
    let dg = st.htod_copy(g).unwrap();
    let dbeta = st.htod_copy(beta).unwrap();
    let dcu = st.htod_copy(cu.clone()).unwrap();
    let plan = ops::KdaChunkPlan::new(&st, &cu, BT).unwrap();
    let scale = (k as f32).powf(-0.5);
    let mut outs = vec![];
    for how in [ops::GdnChunkWy::Legacy, ops::GdnChunkWy::Mma, ops::GdnChunkWy::MmaSerial, ops::GdnChunkWy::Mma8] {
        let mut sc = ops::GdnChunkScratch::new(&st, t, h, hv, k, v, plan.num_chunks, BT).unwrap();
        ops::gdn_chunk_prep(&st, &dm, &dg, &dcu, &plan, &mut sc, t, h, hv, k, v, scale).unwrap();
        ops::gdn_chunk_wy_with(&st, &dm, &dbeta, &dcu, &plan, &mut sc, t, h, hv, k, v, how).unwrap();
        st.synchronize().unwrap();
        // r10：scratch 中间量是 bf16，比较前升 f32
        let up = |v: Vec<bf16>| -> Vec<f32> { v.into_iter().map(f32::from).collect() };
        let aq = up(st.dtoh_sync_copy(&sc.aq).unwrap());
        let u = up(st.dtoh_sync_copy(&sc.u).unwrap());
        let w = up(st.dtoh_sync_copy(&sc.w).unwrap());
        let kg = up(st.dtoh_sync_copy(&sc.kg).unwrap());
        outs.push((aq, u, w, kg));
    }
    let (a0, u0, w0, k0) = &outs[0];
    let n_tok = t * hv;
    // 只比有效 token 行（scratch 多余部分未写）
    let cmp = |name: &str, x: &[f32], y: &[f32], n: usize, cols: usize| {
        let mut worst = (0f32, 0usize);
        let mut nz_y = 0usize;
        for i in 0..n * cols {
            let e = (x[i] - y[i]).abs();
            if y[i] != 0.0 { nz_y += 1; }
            if e > worst.0 { worst = (e, i); }
        }
        let i = worst.1;
        println!("  wy {name:<3} max_err {:.3e} at {} (tok {} hv {} col {}): old {} new {}  new 非零数 {nz_y}/{}", worst.0, i, i / cols / hv, (i / cols) % hv, i % cols, x[i], y[i], n * cols);
        worst.0
    };
    // Mma（分块逆）与 MmaSerial（64 步）都对老 fp32 版
    for (name, (a1, u1, w1, k1)) in [("mma", &outs[1]), ("mma_serial", &outs[2]), ("mma8", &outs[3])] {
        println!("  -- {name}");
        let ea = cmp("aq", a0, a1, plan.num_chunks * hv, BT * BT);
        let eu = cmp("u", u0, u1, n_tok, v);
        let ew = cmp("w", w0, w1, n_tok, k);
        let ek = cmp("kg", k0, k1, n_tok, k);
        assert!(ea < 1e-2 && eu < 2e-2 && ew < 2e-2 && ek < 1e-3, "wy {name} 与老版不一致");
    }
}

/// 计时：27B 形状 H=16 HV=48 / K=V=128，三种批形状（1×1024、8×1024、32×128）：
/// 顺序递归 vs chunk 全链（都含 gate），再分 kernel：conv1d prefill 老 / 新、ho 老 / 新（bv 16 / 32）、wy。
/// 数字记到 `docs/loops/qwen35-perf.md` 记录表。门槛：1×1024 chunk ≤ 顺序 1/5。
#[test]
#[ignore]
fn gdn_chunk_bench() {
    let (dev, st) = dev();
    let _ = dev;
    let (h, hv, k, v) = (16usize, 48usize, 128usize, 128usize);
    // r9（docs/loops/qwen35-gdn-r9.md）加 1×4096 / 1×8192：单长序列是 ho 串行最差情形
    for (nseq, len) in [(1usize, 1024usize), (8, 1024), (32, 128), (1, 4096), (1, 8192)] {
        bench_case(&st, nseq, len, h, hv, k, v);
    }
}

fn time_us(st: &Stream, iters: usize, mut f: impl FnMut()) -> f64 {
    f();
    st.synchronize().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        f();
    }
    st.synchronize().unwrap();
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

#[allow(clippy::too_many_arguments)]
fn bench_case(st: &Stream, nseq: usize, len: usize, h: usize, hv: usize, k: usize, v: usize) {
    let t = nseq * len;
    let row = 2 * h * k + hv * v;
    let mixed: Vec<f32> = (0..t * row).map(|i| (((i * 7919) % 1000) as f32 / 1000.0 - 0.5) * 0.5).collect();
    let a: Vec<f32> = (0..t * hv).map(|i| ((i % 23) as f32 - 11.0) / 4.0).collect();
    let b: Vec<f32> = (0..t * hv).map(|i| ((i % 17) as f32 - 8.0) / 4.0).collect();
    let a_log: Vec<f32> = (0..hv).map(|i| (0.2 + 0.03 * i as f32).ln()).collect();
    let dt_bias = vec![0.1f32; hv];
    let cu: Vec<i32> = (0..=nseq).map(|i| (i * len) as i32).collect();
    let scale = (k as f32).powf(-0.5);
    let dm = up_bf16(st, &mixed);
    let da = up_bf16(st, &a);
    let db = up_bf16(st, &b);
    let dal = st.htod_copy(a_log).unwrap();
    let ddb = st.htod_copy(dt_bias).unwrap();
    let dcu = st.htod_copy(cu.clone()).unwrap();
    let iters = 5;

    let mut g = st.alloc_uninit::<f32>(t * hv).unwrap();
    let mut beta = st.alloc_uninit::<f32>(t * hv).unwrap();
    let mut out = st.alloc_uninit::<bf16>(t * hv * v).unwrap();
    let mut state = st.alloc_zeros::<f32>(nseq * hv * v * k).unwrap();

    // 顺序版（gate + recurrent）
    let us_seq = time_us(st, iters, || {
        ops::gdn_gate(st, &da, &db, &dal, &ddb, &mut g, &mut beta, t, hv).unwrap();
        ops::gdn_recurrent(st, &dm, Some(&g), Some(&beta), None, None, &dal, &ddb, &mut out, &mut state, &dcu, None, nseq, t, h, hv, k, v, scale).unwrap();
    });

    // chunk 全链（gate + prep + wy + ho 默认）
    let plan = ops::KdaChunkPlan::new(st, &cu, BT).unwrap();
    let mut sc = ops::GdnChunkScratch::new(st, t, h, hv, k, v, plan.num_chunks, BT).unwrap();
    let us_chunk = time_us(st, iters, || {
        ops::gdn_gate(st, &da, &db, &dal, &ddb, &mut g, &mut beta, t, hv).unwrap();
        ops::gdn_chunk_prep(st, &dm, &g, &dcu, &plan, &mut sc, t, h, hv, k, v, scale).unwrap();
        ops::gdn_chunk_wy(st, &dm, &beta, &dcu, &plan, &mut sc, t, h, hv, k, v).unwrap();
        ops::gdn_chunk_ho(st, &dcu, &plan, &sc, &mut out, &mut state, None, t, h, hv, k, v).unwrap();
    });
    // 分 kernel
    let us_prep = time_us(st, iters, || ops::gdn_chunk_prep(st, &dm, &g, &dcu, &plan, &mut sc, t, h, hv, k, v, scale).unwrap());
    let us_wy0 = time_us(st, iters, || ops::gdn_chunk_wy_with(st, &dm, &beta, &dcu, &plan, &mut sc, t, h, hv, k, v, ops::GdnChunkWy::Legacy).unwrap());
    let us_wy = time_us(st, iters, || ops::gdn_chunk_wy_with(st, &dm, &beta, &dcu, &plan, &mut sc, t, h, hv, k, v, ops::GdnChunkWy::Mma).unwrap());
    let us_wy64 = time_us(st, iters, || ops::gdn_chunk_wy_with(st, &dm, &beta, &dcu, &plan, &mut sc, t, h, hv, k, v, ops::GdnChunkWy::MmaSerial).unwrap());
    let us_wy8 = time_us(st, iters, || ops::gdn_chunk_wy_with(st, &dm, &beta, &dcu, &plan, &mut sc, t, h, hv, k, v, ops::GdnChunkWy::Mma8).unwrap());
    // ho 变体：bv0 老 fp32；bv16 / bv32 融合 mma；bv16pf / bv32pf 寄存器预取（r9 第 1 轮）
    let mut ho = vec![];
    // + sp = 两段版（r9 第 2 轮）
    for (name, how) in [
        ("bv0", ops::GdnChunkHo { bv: 0, prefetch: false, split: false }),
        ("bv16", ops::GdnChunkHo { bv: 16, prefetch: false, split: false }),
        ("bv32", ops::GdnChunkHo { bv: 32, prefetch: false, split: false }),
        ("bv16pf", ops::GdnChunkHo { bv: 16, prefetch: true, split: false }),
        ("bv32pf", ops::GdnChunkHo { bv: 32, prefetch: true, split: false }),
        ("bv64", ops::GdnChunkHo { bv: 64, prefetch: false, split: false }),
        ("bv64pf", ops::GdnChunkHo { bv: 64, prefetch: true, split: false }),
        ("bv16sp", ops::GdnChunkHo { bv: 16, prefetch: false, split: true }),
        ("bv32sp", ops::GdnChunkHo { bv: 32, prefetch: false, split: true }),
        ("bv16pfsp", ops::GdnChunkHo { bv: 16, prefetch: true, split: true }),
        ("bv32pfsp", ops::GdnChunkHo { bv: 32, prefetch: true, split: true }),
    ] {
        ho.push((name, time_us(st, iters, || ops::gdn_chunk_ho_with(st, &dcu, &plan, &sc, &mut out, &mut state, None, t, h, hv, k, v, how).unwrap())));
    }
    // conv1d prefill（dim = row，width 4）
    let width = 4;
    let wv: Vec<f32> = (0..row * width).map(|i| ((i % 13) as f32 - 6.0) / 20.0).collect();
    let dw = up_bf16(st, &wv);
    let mut cout = st.alloc_uninit::<bf16>(t * row).unwrap();
    let mut cst = st.alloc_zeros::<bf16>(nseq * row * (width - 1)).unwrap();
    let mut conv = vec![];
    for rows in [0usize, 8, 16, 32, 64, 128, 256] {
        conv.push((rows, time_us(st, iters, || ops::gdn_conv1d_prefill_with(st, &dm, &dw, &mut cout, &mut cst, &dcu, None, None, nseq, t, row, width, width - 1, GdnAct::Silu, rows).unwrap())));
    }

    println!("  [{nseq}×{len}] 顺序 {:.0} µs | chunk 全链 {:.0} µs（{:.1}x）| prep {:.0} | wy old {:.0} mma {:.0} mma64 {:.0} mma8 {:.0} | ho {} | conv {}",
        us_seq, us_chunk, us_seq / us_chunk, us_prep, us_wy0, us_wy, us_wy64, us_wy8,
        ho.iter().map(|(name, u)| format!("{name}={u:.0}")).collect::<Vec<_>>().join(" "),
        conv.iter().map(|(r, u)| format!("rows{r}={u:.0}")).collect::<Vec<_>>().join(" "));
    if nseq == 1 {
        let speedup = us_seq / us_chunk;
        assert!(speedup >= 5.0, "chunk 加速 {speedup:.2}x < 5x");
    }
}

// ---------------------------------------------------------------- decode 寄存器版 kernel（第 4 轮，docs/loops/qwen35-perf.md）
// 27B 真形状、B=32、非零初始状态：状态 100 MB 不进仓库，两侧按同一公式程序化生成（gen_gdn_refs.py::proc_state）。

fn proc_state(slots: usize, hv: usize, v: usize, k: usize) -> Vec<f32> {
    (0..slots * hv * v * k)
        .map(|i| {
            let h = (i as u64).wrapping_mul(2654435761) & 0xFFFF_FFFF;
            ((h >> 8) as f32) / 16777216.0f32 * 0.1f32 - 0.05f32
        })
        .collect()
}

/// state [S, HV, V, K] → K 维求和 [S, HV, V] 与 V 维求和 [S, HV, K]（f64 累加，与参考侧同）。
fn state_sums(st: &[f32], slots: usize, hv: usize, v: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let mut sk = vec![0f64; slots * hv * v];
    let mut sv = vec![0f64; slots * hv * k];
    for s in 0..slots {
        for h in 0..hv {
            for vv in 0..v {
                let row = &st[((s * hv + h) * v + vv) * k..][..k];
                sk[(s * hv + h) * v + vv] = row.iter().map(|&x| x as f64).sum();
                for kk in 0..k {
                    sv[(s * hv + h) * k + kk] += row[kk] as f64;
                }
            }
        }
    }
    (sk.into_iter().map(|x| x as f32).collect(), sv.into_iter().map(|x| x as f32).collect())
}

#[test]
#[ignore]
fn gdn_recurrent_decode_b32_qwen_shape() {
    let (dev, st) = dev();
    let _ = dev;
    let (k, v, h, hv) = (128usize, 128usize, 16usize, 48usize);
    let dir = "recurrent_qwen_decode_b32";
    let (_, a_log) = load(dir, "a_log");
    let (_, dt_bias) = load(dir, "dt_bias");
    let cu = load_i32(dir, "cu_seqlens");
    let b = cu.len() - 1;
    let (ms, mixed) = load(dir, "mixed");
    assert_eq!(ms, vec![b, 2 * h * k + hv * v]);
    let (_, a) = load(dir, "a");
    let (_, bb) = load(dir, "b");
    let (_, want) = load(dir, "out");
    let (_, want_sk) = load(dir, "state_sum_k");
    let (_, want_sv) = load(dir, "state_sum_v");
    let dal = st.htod_copy(a_log).unwrap();
    let ddb = st.htod_copy(dt_bias).unwrap();
    let dcu = st.htod_copy(cu).unwrap();
    let si: Vec<i32> = (0..b as i32).collect();
    let dsi = st.htod_copy(si).unwrap();
    let dm = up_bf16(&st, &mixed);
    let da = up_bf16(&st, &a);
    let db = up_bf16(&st, &bb);
    let scale = (k as f32).powf(-0.5);
    let init = proc_state(b, hv, v, k);

    // 新 kernel 的三个 BV 档 × 四个变体（第 7 轮：base / tr / pipe / pipe_base）+ 老 kernel（decode_bv=0），
    // 每个都对参考；再拿老 vs 新全量状态逐元素对（tr 变体改了求和结合顺序，容差 1e-5 相对仍应过）。
    let mut legacy_state: Option<Vec<f32>> = None;
    let mut combos: Vec<(usize, Option<GdnDecodeVariant>)> = vec![(0, None)];
    for bv in [16usize, 32, 64] {
        for var in GdnDecodeVariant::ALL {
            combos.push((bv, Some(var)));
        }
    }
    for (bv, var) in combos {
        let mut dstate = st.htod_copy(init.clone()).unwrap();
        let mut dout = st.alloc_uninit::<bf16>(b * hv * v).unwrap();
        ops::gdn_recurrent_variant(
            &st, &dm, None, None, Some(&da), Some(&db), &dal, &ddb, &mut dout, &mut dstate, &dcu, Some(&dsi), b, b, h,
            hv, k, v, scale, Some(bv), var, None,
        )
        .unwrap();
        st.synchronize().unwrap();
        let vname = var.map_or("legacy", |v| v.name());
        let tag = format!("recurrent_qwen_decode_b32/bv{bv}_{vname}");
        assert_close(&format!("{tag}/out"), &st.dtoh_sync_copy(&dout).unwrap(), &want, 1e-2, 0.999);
        let got_st: Vec<f32> = st.dtoh_sync_copy(&dstate).unwrap();
        let (sk, sv) = state_sums(&got_st, b, hv, v, k);
        assert_close_f32(&format!("{tag}/state_sum_k"), &sk, &want_sk, 2e-3, 0.999);
        assert_close_f32(&format!("{tag}/state_sum_v"), &sv, &want_sv, 2e-3, 0.999);
        match &legacy_state {
            None => legacy_state = Some(got_st),
            Some(lg) => assert_close_f32(&format!("{tag}/state_vs_legacy"), &got_st, lg, 1e-5, 0.999_999),
        }
    }
}

/// decode kernel 计时：老 shared 版 vs 寄存器版四个变体（自动 BV）+ tr 的三档 BV，B ∈ {1, 8, 32, 64}，27B 形状，fused gate。
/// 数字写进 `docs/loops/qwen35-perf.md` 第 4 / 7 行；只断言 B=32 默认变体不慢于老版。
#[test]
#[ignore]
fn gdn_decode_bench() {
    let (dev, st) = dev();
    let _ = dev;
    let (k, v, h, hv) = (128usize, 128usize, 16usize, 48usize);
    let row = 2 * h * k + hv * v;
    let scale = (k as f32).powf(-0.5);
    let a_log: Vec<f32> = (0..hv).map(|i| (0.2 + 0.03 * i as f32).ln()).collect();
    let dal = st.htod_copy(a_log).unwrap();
    let ddb = st.htod_copy(vec![0.1f32; hv]).unwrap();
    let iters = 20;
    let mut worst_ratio = 0f64;
    let mut cases: Vec<(String, Option<usize>, Option<GdnDecodeVariant>)> = vec![("legacy".into(), Some(0), None)];
    for var in GdnDecodeVariant::ALL {
        cases.push((var.name().into(), None, Some(var)));
    }
    for bv in [16usize, 32] {
        cases.push((format!("tr/bv{bv}"), Some(bv), Some(GdnDecodeVariant::Tr)));
        cases.push((format!("pipe/bv{bv}"), Some(bv), Some(GdnDecodeVariant::Pipe)));
    }
    for b in [1usize, 8, 32, 64] {
        let mixed: Vec<f32> = (0..b * row).map(|i| (((i * 7919) % 1000) as f32 / 1000.0 - 0.5) * 0.5).collect();
        let a: Vec<f32> = (0..b * hv).map(|i| ((i % 23) as f32 - 11.0) / 4.0).collect();
        let bb: Vec<f32> = (0..b * hv).map(|i| ((i % 17) as f32 - 8.0) / 4.0).collect();
        let dm = up_bf16(&st, &mixed);
        let da = up_bf16(&st, &a);
        let db = up_bf16(&st, &bb);
        let cu: Vec<i32> = (0..=b as i32).collect();
        let dcu = st.htod_copy(cu).unwrap();
        let mut dstate = st.htod_copy(proc_state(b, hv, v, k)).unwrap();
        let mut dout = st.alloc_uninit::<bf16>(b * hv * v).unwrap();
        let mut line = format!("  B={b:>2}:");
        let mut us_legacy = 0f64;
        for (name, bv, var) in &cases {
            let once = |st: &Stream, dstate: &mut _, dout: &mut _| {
                ops::gdn_recurrent_variant(
                    st, &dm, None, None, Some(&da), Some(&db), &dal, &ddb, dout, dstate, &dcu, None, b, b, h, hv, k, v,
                    scale, *bv, *var, None,
                )
                .unwrap();
            };
            once(&st, &mut dstate, &mut dout);
            st.synchronize().unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                once(&st, &mut dstate, &mut dout);
            }
            st.synchronize().unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            if name == "legacy" {
                us_legacy = us;
            } else if name == "tr" && b == 32 {
                worst_ratio = us / us_legacy;
            }
            line.push_str(&format!(" {name} {us:6.1}"));
        }
        println!("{line}  (µs)");
    }
    assert!(worst_ratio <= 1.0, "B=32 新 decode kernel 慢于老版（{worst_ratio:.2}x）");
}
