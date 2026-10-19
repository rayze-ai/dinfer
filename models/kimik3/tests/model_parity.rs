//! 里程碑 C-3：Kimi-K3 真权重**前 N 层主干**对拍（`docs/loops/k3-model.md`）。
//!
//! 参考：`scripts/verify/gen_k3_model_refs.py`（vLLM 容器，纯 torch 按 HF 代码逐层算；KDA 用 vLLM 自带 fla）。
//! 输入 `ids.npy`（真 token，HF tokenizer）；每层输出 `layer{l}.npy`（prefix_sum，f32）、`res_out / final / logits_last`。
//!
//! 口径（同 Qwen3.5 `trace_norm`）：每层 cos ≥ 0.999 并报每层 rel / cos / 末 token 范数；final cos ≥ 0.999；
//! 末 token logits top-1 一致（top-5 报出来）；decode（prefill T-1 + decode 1）vs 整段 prefill 末行 cos ≥ 0.999。
//!
//! 跑法（B300）：`KIMIK3_DIR=/data/models/moonshotai/Kimi-K3 KIMIK3_REFS=test/kimik3/2026-10-20/model KIMIK3_LAYERS=5 \
//!   cargo test --release -p dinfer-kimik3 --features ref-tests --test model_parity -- --ignored --nocapture`
#![cfg(feature = "ref-tests")]
use dinfer_core::runner::BatchMode;
use dinfer_kimik3::config::KimiK3Config;
use dinfer_kimik3::kda::{KdaChunkWorkspace, KdaMeta};
use dinfer_kimik3::layer::{Attn, LayerBudget, StepMeta};
use dinfer_kimik3::mla::MlaMeta;
use dinfer_kimik3::model::KimiK3Model;
use dinfer_kimik3::weights::KdaShapes;
use dinfer_layers::{bf16, Device, Gemm, Stream, Tensor};
use std::cell::RefCell;
use std::path::{Path, PathBuf};

fn refs_dir() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("KIMIK3_REFS").ok()?);
    Some(if p.is_absolute() { p } else { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join(p) })
}

fn npy_f32(dir: &Path, name: &str) -> (Vec<usize>, Vec<f32>) {
    let bytes = std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    let arr = npyz::NpyFile::new(&bytes[..]).unwrap();
    let shape: Vec<usize> = arr.shape().iter().map(|&d| d as usize).collect();
    (shape, arr.into_vec::<f32>().unwrap())
}

fn npy_i32(dir: &Path, name: &str) -> Vec<i32> {
    let bytes = std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    npyz::NpyFile::new(&bytes[..]).unwrap().into_vec::<i32>().unwrap()
}

/// 返回 (rel max_err, cos)。
fn compare(name: &str, got: &[f32], want: &[f32]) -> (f32, f64) {
    assert_eq!(got.len(), want.len(), "{name}: len {} vs {}", got.len(), want.len());
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
        "  {name:<26} max_err {max_err:.4e} (rel {:.3e}, at {worst}: {:.5} vs {:.5})  mean_abs {mean_abs:.3e}  cos {cos:.7}",
        max_err / scale,
        got[worst],
        want[worst]
    );
    (max_err / scale, cos)
}

fn norm(v: &[f32]) -> f64 {
    v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt()
}

fn rows(t: &Tensor<bf16>, s: &Stream, n: usize) -> Vec<f32> {
    t.to_vec_f32(s).unwrap()[..n].to_vec()
}

fn top5(v: &[f32]) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
    idx[..5].iter().map(|&i| (i, v[i])).collect()
}

#[test]
#[ignore]
fn model_first_layers_real_weights() {
    let _ = tracing_subscriber_init();
    let Some(refs) = refs_dir() else {
        eprintln!("skip: KIMIK3_REFS 未设");
        return;
    };
    let Ok(dir) = std::env::var("KIMIK3_DIR").map(PathBuf::from) else {
        eprintln!("skip: KIMIK3_DIR 未设");
        return;
    };
    let n_layers: usize = std::env::var("KIMIK3_LAYERS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let ids = npy_i32(&refs, "ids.npy");
    let t = ids.len();
    let cfg = KimiK3Config::from_json(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let tc = &cfg.text_config;
    let hidden = tc.hidden_size as usize;
    let vocab = tc.vocab_size as usize;
    let sh = KdaShapes::new(tc);

    let dev = Device::new(0).unwrap();
    let s = dev.default_stream();
    let gemm = Gemm::new(&s).unwrap();
    let block_size = 16usize;
    let kv_blocks = t.div_ceil(block_size) + 1;
    let budget = LayerBudget { max_tokens: t, num_slots: 1, kv_blocks, block_size };
    let (free0, _) = s.mem_info().unwrap();
    let t0 = std::time::Instant::now();
    let mut model = KimiK3Model::load(&s, &dir, tc, n_layers, budget).unwrap();
    s.synchronize().unwrap();
    let (free1, _) = s.mem_info().unwrap();
    println!("  装载前 {n_layers} 层 + 顶层：{:.1} s，显存 +{:.2} GB", t0.elapsed().as_secs_f32(), (free0 - free1) as f64 / 1e9);
    model.trace = true;

    // meta：单请求
    let ids_d = Tensor::from_vec(&s, ids.clone(), &[t]).unwrap();
    let si = Tensor::from_vec(&s, vec![0i32], &[1]).unwrap();
    let bt: Vec<i32> = (0..kv_blocks as i32).collect();
    let block_table = Tensor::from_vec(&s, bt, &[1, kv_blocks]).unwrap();
    let mut logits = Tensor::<f32>::zeros(&s, &[t, vocab]).unwrap();

    let run = |model: &mut KimiK3Model, ids_d: &Tensor<i32>, n: usize, pos0: usize, mode: BatchMode, logits: &mut Tensor<f32>| -> f32 {
        let cu_host = vec![0i32, n as i32];
        let cu = Tensor::from_vec(&s, cu_host.clone(), &[2]).unwrap();
        let hi = Tensor::from_vec(&s, vec![i32::from(mode == BatchMode::Decode)], &[1]).unwrap();
        let seq_lens = Tensor::from_vec(&s, vec![(pos0 + n) as i32], &[1]).unwrap();
        let slots: Vec<i64> = (pos0..pos0 + n).map(|p| p as i64).collect();
        let slot_mapping = Tensor::from_vec(&s, slots, &[n]).unwrap();
        let ws = RefCell::new(KdaChunkWorkspace::new());
        let use_chunk = mode == BatchMode::Prefill;
        if use_chunk {
            ws.borrow_mut().prepare(&s, &cu_host, &sh).unwrap();
        }
        let meta = StepMeta {
            kda: KdaMeta { cu_seqlens: &cu, state_indices: &si, has_initial: &hi, num_reqs: 1, mode, chunk: if use_chunk { Some(&ws) } else { None } },
            mla: MlaMeta { mode, cu_seqlens: &cu, block_table: &block_table, seq_lens: &seq_lens, num_reqs: 1, max_blocks: kv_blocks },
            slot_mapping: &slot_mapping,
        };
        if mode == BatchMode::Prefill {
            for l in model.layers.iter_mut() {
                if let Attn::Kda(k) = &mut l.attn {
                    k.reset_slot(&s, 0).unwrap();
                }
            }
        }
        let t0 = std::time::Instant::now();
        model.forward(&s, &gemm, n, ids_d, &meta, logits).unwrap();
        s.synchronize().unwrap();
        t0.elapsed().as_secs_f32() * 1e3
    };

    // 1. 整段 prefill，逐层对拍
    println!("== prefill T={t}，前 {n_layers} 层逐层对拍（末 token 范数：ours / HF）");
    let ms = run(&mut model, &ids_d, t, 0, BatchMode::Prefill, &mut logits);
    let n = t * hidden;
    let (_, emb_ref) = npy_f32(&refs, "embed.npy");
    let mut worst_cos = 1.0f64;
    let mut per_layer = Vec::new();
    for l in 0..n_layers {
        let (_, want) = npy_f32(&refs, &format!("layer{l}.npy"));
        let got = rows(&model.layer_out[l], &s, n);
        let kind = format!("{}+{}", if tc.is_kda_layer(l as u32) { "kda" } else { "mla" }, if tc.is_moe_layer(l as u32) { "moe" } else { "dense" });
        let (rel, cos) = compare(&format!("layer {l} [{kind}]"), &got, &want);
        let (ng, nw) = (norm(&got[(t - 1) * hidden..]), norm(&want[(t - 1) * hidden..]));
        println!("      末 token 范数 {ng:.3} / {nw:.3}（embed 末 token 范数 {:.3}）", norm(&emb_ref[(t - 1) * hidden..]));
        worst_cos = worst_cos.min(cos);
        per_layer.push((l, kind, rel, cos));
    }
    let (_, res_ref) = npy_f32(&refs, "res_out.npy");
    let (_, final_ref) = npy_f32(&refs, "final.npy");
    let (_, logits_ref) = npy_f32(&refs, "logits_last.npy");
    let (rel_r, cos_r) = compare("output_attn_res", &rows(model.last_res_out(), &s, n), &res_ref);
    let (rel_f, cos_f) = compare("final (norm)", &rows(model.last_normed(), &s, n), &final_ref);
    let lg = logits.to_vec(&s).unwrap();
    let last = &lg[(t - 1) * vocab..t * vocab];
    let (rel_l, cos_l) = compare("logits 末 token", last, &logits_ref);
    let (o5, r5) = (top5(last), top5(&logits_ref));
    println!("  top-5 ours {:?}", o5.iter().map(|(i, v)| format!("{i}:{v:.3}")).collect::<Vec<_>>());
    println!("  top-5 HF   {:?}", r5.iter().map(|(i, v)| format!("{i}:{v:.3}")).collect::<Vec<_>>());
    println!("  块残差数 {}，prefill 耗时 {ms:.1} ms（首轮，含 D2H 同步）", model.num_blocks());

    // 2. decode：prefill T-1 + decode 1，与整段 prefill 的末行比（KDA 状态接力 + MLA paged cache 读）
    println!("== decode：prefill {} + decode 1", t - 1);
    let final_full = rows(model.last_normed(), &s, n)[(t - 1) * hidden..].to_vec();
    let ids_head = Tensor::from_vec(&s, ids[..t - 1].to_vec(), &[t - 1]).unwrap();
    let ids_tail = Tensor::from_vec(&s, ids[t - 1..].to_vec(), &[1]).unwrap();
    let _ = run(&mut model, &ids_head, t - 1, 0, BatchMode::Prefill, &mut logits);
    let ms_d = run(&mut model, &ids_tail, 1, t - 1, BatchMode::Decode, &mut logits);
    let final_dec = rows(model.last_normed(), &s, hidden);
    let (rel_d, cos_d) = compare("decode final vs prefill[-1]", &final_dec, &final_full);
    let lg_d = logits.to_vec(&s).unwrap()[..vocab].to_vec();
    let (_rel_dl, cos_dl) = compare("decode logits vs HF", &lg_d, &logits_ref);
    println!("  decode 1 token 耗时 {ms_d:.1} ms（含 D2H 同步 + AttnRes host softmax）");

    println!("== 汇总（每层 rel / cos）");
    for (l, kind, rel, cos) in &per_layer {
        println!("  layer {l:<2} {kind:<10} rel {rel:.3e}  cos {cos:.6}");
    }
    for (l, kind, rel, cos) in &per_layer {
        assert!(*cos >= 0.999 && *rel <= 5e-2, "layer {l} [{kind}]：rel {rel:.3e} cos {cos:.6}");
    }
    assert!(cos_r >= 0.999 && rel_r <= 5e-2, "output_attn_res：rel {rel_r:.3e} cos {cos_r:.6}");
    assert!(cos_f >= 0.999 && rel_f <= 5e-2, "final：rel {rel_f:.3e} cos {cos_f:.6}");
    assert!(cos_l >= 0.999 && rel_l <= 5e-2, "logits：rel {rel_l:.3e} cos {cos_l:.6}");
    assert_eq!(o5[0].0, r5[0].0, "末 token top-1 不一致");
    assert!(cos_d >= 0.999 && rel_d <= 5e-2, "decode vs prefill：rel {rel_d:.3e} cos {cos_d:.6}");
    assert!(cos_dl >= 0.999, "decode logits：cos {cos_dl:.6}");
    let _ = worst_cos;
}

fn tracing_subscriber_init() -> Result<(), ()> {
    // RUST_LOG=info 时打印每层装载时间 / 显存（model.rs 的 tracing::info!）
    if std::env::var("RUST_LOG").is_ok() {
        tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).try_init().map_err(|_| ())
    } else {
        Ok(())
    }
}
