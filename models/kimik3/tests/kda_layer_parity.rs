//! 里程碑 C-1：Kimi-K3 真权重**单层** KDA 对拍（`docs/loops/k3-kda.md` "里程碑 C-1"）。
//!
//! 参考：`scripts/verify/gen_k3_kda_layer_refs.py`（vLLM 容器里跑：HF 权重 + vLLM 自带的 fla `chunk_kda_with_fused_gate`，
//! 另给一份纯 torch 顺序递归 f32 参考做三方核对）产出 `.npy`：
//! - `x_bf16.npy [T, 7168]`：随机输入（seed 固定）；
//! - `h_bf16.npy [T, 7168]`：`input_layernorm(x)`；
//! - `y_fla.npy / y_torch.npy [T, 7168]` f32：`kda(h)` 两种参考；`n_fla.npy [T, 12288]` f32：`o_norm` 之后、`o_proj` 之前。
//!
//! Rust 侧：只 mmap 含该层的分片（`weights::shards_for_layer`），`KdaLayer::load` 后：
//! 1. `prefill`：整段 T 走 chunk 路径，与 `y_fla` 比（cos ≥ 0.999，相对 max_err ≤ 5e-2；bf16 链路口径见 loop 记录）；
//! 2. `decode`：prefill 前 T-1 个 token 后 decode 第 T 个，与整段 prefill 的最后一行比（chunk vs 递归，cos ≥ 0.9999）。
//!
//! 跑法（B300）：`KIMIK3_DIR=/data/models/moonshotai/Kimi-K3 KIMIK3_REFS=test/kimik3/2026-10-19/kda_layer \
//!   cargo test -p dinfer-kimik3 --features ref-tests --test kda_layer_parity -- --ignored --nocapture`
#![cfg(feature = "ref-tests")]
use dinfer_core::runner::BatchMode;
use dinfer_kimik3::config::KimiK3Config;
use dinfer_kimik3::kda::{KdaChunkWorkspace, KdaLayer, KdaMeta};
use dinfer_kimik3::scratch::ActScratch;
use dinfer_kimik3::weights::Shard;
use dinfer_kimik3::weights::{shards_for_layer, KdaShapes, Table};
use dinfer_layers::{bf16, Device, Gemm, Stream, Tensor, WeightSource};
use std::cell::RefCell;
use std::path::PathBuf;

const LAYER: usize = 1;

fn refs_dir() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("KIMIK3_REFS").ok()?);
    Some(if p.is_absolute() { p } else { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join(p) })
}

fn npy(dir: &PathBuf, name: &str) -> (Vec<usize>, Vec<f32>) {
    let bytes = std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    let arr = npyz::NpyFile::new(&bytes[..]).unwrap();
    let shape: Vec<usize> = arr.shape().iter().map(|&d| d as usize).collect();
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

/// max_err（相对参考的 max|want|）+ cos。返回 (rel, cos)。
fn compare(name: &str, got: &[f32], want: &[f32]) -> (f32, f64) {
    assert_eq!(got.len(), want.len(), "{name}: 长度");
    let mut max_err = 0f32;
    let mut worst = 0usize;
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
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
    let mean_abs = got.iter().zip(want).map(|(a, b)| (a - b).abs()).sum::<f32>() / got.len() as f32;
    println!(
        "  {name:<22} max_err {max_err:.4e} (rel {:.3e}, at {worst}: {:.5} vs {:.5})  mean_abs {mean_abs:.3e}  cos {cos:.7}",
        max_err / scale,
        got[worst],
        want[worst]
    );
    (max_err / scale, cos)
}

struct Env {
    s: Stream,
    gemm: Gemm,
    layer: KdaLayer,
    act: ActScratch,
    sh: KdaShapes,
}

fn load(t_max: usize) -> Option<Env> {
    let dir = PathBuf::from(std::env::var("KIMIK3_DIR").ok()?);
    let cfg = KimiK3Config::from_json(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let dev = Device::new(0).unwrap();
    let s = dev.default_stream();
    let gemm = Gemm::new(&s).unwrap();
    let files = shards_for_layer(&dir, LAYER).unwrap();
    println!(
        "  第 {LAYER} 层分片：{:?}",
        files.iter().map(|f| f.file_name().unwrap().to_string_lossy().to_string()).collect::<Vec<_>>()
    );
    let t0 = std::time::Instant::now();
    let src = WeightSource::open_files(&files).unwrap();
    let table = Table::from_names(&src.names().collect::<Vec<_>>()).unwrap();
    let layer = KdaLayer::load(&s, &src, &table, LAYER, &cfg.text_config, 2).unwrap();
    let act = ActScratch::new(&s, &cfg.text_config, t_max, Shard::SINGLE).unwrap();
    s.synchronize().unwrap();
    println!("  装载 {:.1} s", t0.elapsed().as_secs_f32());
    let sh = KdaShapes::new(&cfg.text_config);
    Some(Env { s, gemm, layer, act, sh })
}

/// 一次 forward：`x [t, hidden]` bf16 → y f32。`mode` / `cu` 决定 prefill 还是 decode。
fn run(e: &mut Env, x: &[bf16], t: usize, mode: BatchMode, use_chunk: bool) -> (Vec<f32>, Vec<f32>) {
    let s = &e.s;
    let x = Tensor::from_vec(s, x.to_vec(), &[t, e.sh.hidden]).unwrap();
    let mut out = Tensor::<bf16>::zeros(s, &[t, e.sh.hidden]).unwrap();
    let cu_host = vec![0i32, t as i32];
    let cu = Tensor::from_vec(s, cu_host.clone(), &[2]).unwrap();
    let si = Tensor::from_vec(s, vec![0i32], &[1]).unwrap();
    let hi = Tensor::from_vec(s, vec![i32::from(mode == BatchMode::Decode)], &[1]).unwrap();
    let ws = RefCell::new(KdaChunkWorkspace::new());
    if use_chunk {
        ws.borrow_mut().prepare(s, &cu_host, &e.sh).unwrap();
    }
    let meta = KdaMeta {
        cu_seqlens: &cu,
        state_indices: &si,
        has_initial: &hi,
        num_reqs: 1,
        mode,
        chunk: if use_chunk { Some(&ws) } else { None },
    };
    e.layer.forward(s, &e.gemm, t, &x, &meta, &mut e.act.kda, &mut out).unwrap();
    s.synchronize().unwrap();
    let y = out.to_vec_f32(s).unwrap();
    let n = e.act.kda.n.to_vec_f32(s).unwrap();
    (y, n[..t * e.sh.proj()].to_vec())
}

#[test]
#[ignore]
fn kda_layer_real_weights() {
    let Some(refs) = refs_dir() else {
        eprintln!("skip: KIMIK3_REFS 未设");
        return;
    };
    let (xs, h) = npy(&refs, "h_bf16.npy");
    let (_, y_fla) = npy(&refs, "y_fla.npy");
    let (_, n_fla) = npy(&refs, "n_fla.npy");
    let y_torch = refs.join("y_torch.npy").exists().then(|| npy(&refs, "y_torch.npy").1);
    let t = xs[0];
    let Some(mut e) = load(t) else {
        eprintln!("skip: KIMIK3_DIR 未设");
        return;
    };
    assert_eq!(xs[1], e.sh.hidden);
    let h16: Vec<bf16> = h.iter().map(|&v| bf16::from_f32(v)).collect();

    // 1. prefill 整段（chunk 路径）
    println!("== prefill T={t}（chunk）");
    e.layer.reset_slot(&e.s, 0).unwrap();
    let (y_chunk, n_chunk) = run(&mut e, &h16, t, BatchMode::Prefill, true);
    let (rel_n, cos_n) = compare("o_norm vs fla", &n_chunk, &n_fla);
    let (rel, cos) = compare("out vs fla", &y_chunk, &y_fla);
    if let Some(yt) = &y_torch {
        compare("out vs torch f32", &y_chunk, yt);
        compare("fla vs torch f32", &y_fla, yt);
    }
    // 2. prefill 整段（顺序递归路径）—— chunk 与递归的数值等价性
    println!("== prefill T={t}（recurrent）");
    e.layer.reset_slot(&e.s, 0).unwrap();
    let (y_rec, _) = run(&mut e, &h16, t, BatchMode::Prefill, false);
    let (rel_cr, cos_cr) = compare("chunk vs recurrent", &y_chunk, &y_rec);
    compare("recurrent vs fla", &y_rec, &y_fla);

    // 3. decode：prefill T-1 后 decode 第 T 个，与整段 prefill 的最后一行比
    println!("== decode：prefill {} + decode 1", t - 1);
    e.layer.reset_slot(&e.s, 0).unwrap();
    let hid = e.sh.hidden;
    let _ = run(&mut e, &h16[..(t - 1) * hid], t - 1, BatchMode::Prefill, true);
    let (y_dec, _) = run(&mut e, &h16[(t - 1) * hid..], 1, BatchMode::Decode, false);
    let (rel_d, cos_d) = compare("decode vs prefill[-1]", &y_dec, &y_chunk[(t - 1) * hid..]);
    compare("decode vs fla[-1]", &y_dec, &y_fla[(t - 1) * hid..]);

    // 容差：bf16 链路（投影 GEMM / conv / o_norm 输出都是 bf16）对 fla 参考：cos ≥ 0.999、相对 max_err ≤ 5e-2；
    // 同口径两条路径（chunk vs 递归、decode vs prefill）：cos ≥ 0.9999。
    assert!(cos_n >= 0.999 && rel_n <= 5e-2, "o_norm 对拍不过：rel {rel_n:.3e} cos {cos_n:.6}");
    assert!(cos >= 0.999 && rel <= 5e-2, "out 对拍不过：rel {rel:.3e} cos {cos:.6}");
    assert!(cos_cr >= 0.9999 && rel_cr <= 2e-2, "chunk vs recurrent：rel {rel_cr:.3e} cos {cos_cr:.6}");
    assert!(cos_d >= 0.9999 && rel_d <= 2e-2, "decode vs prefill：rel {rel_d:.3e} cos {cos_d:.6}");
}
