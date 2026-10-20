//! 里程碑 C-3：Kimi-K3 真权重**前 N 层主干**对拍（`docs/loops/k3-model.md`）。
//!
//! 参考：`scripts/verify/gen_k3_model_refs.py`（vLLM 容器，纯 torch 按 HF 代码逐层算；KDA 用 vLLM 自带 fla）。
//! 输入 `ids.npy`（真 token，HF tokenizer）；每层 `in{l}.npy` / `layer{l}.npy`（输入 / 输出 prefix_sum，f32）、`res_out / final / logits_last`、
//! MoE 层 `moe{l}/topk_ids.npy`（路由）。
//!
//! 两个口径：
//! 1. **隔离**（断言，同 C-2）：每层喂 HF 的输入（`in{l}` + 块残差用 HF 的 `in0`），只比本层；MoE 层只在**路由一致的行**判数值
//!    （平局行专家不同，硬判决不可逐位复现，见 docs/pitfalls.md）。门槛：cos ≥ 0.999、rel ≤ 5e-2；路由一致行 ≥ T/2。
//! 2. **链式**（端到端，报数字 + 弱断言）：embed → N 层 → output_attn_res → norm → lm_head 一气算完，每层 cos / 末 token 范数；
//!    误差逐层累积（每层多一次平局行 + bf16 链），只断言 final cos ≥ 0.998、末 token logits cos ≥ 0.999 且 **top-1 一致**；
//!    decode（prefill T-1 + decode 1）vs 整段 prefill 末行 cos ≥ 0.999（KDA 状态接力 + MLA paged cache）。
//!
//! 跑法（B300）：`KIMIK3_DIR=/data/models/moonshotai/Kimi-K3 KIMIK3_REFS=test/kimik3/2026-10-20/model KIMIK3_LAYERS=5 \
//!   cargo test --release -p dinfer-kimik3 --features ref-tests --test model_parity -- --ignored --nocapture`
//!
//! TP/EP（C-5，`docs/loops/k3-tp.md`）：`KIMIK3_TP=2`（配 `CUDA_VISIBLE_DEVICES=2,3`）→ 每 rank 一线程，`Device::new(rank)` +
//! `Shard{rank, world}` 分片装载 + NCCL `Comm`（经 `StepMeta.comm`），所有 rank 跑同样的三段；只 rank 0 打印 / 断言。
//! `KIMIK3_DUMP=<dir>`：rank 0 把隔离 / 链式每层输出与末 token logits 写成 raw f32（LE）到该目录，供 tp2 vs 单卡直接对比
//!（`scripts/verify/k3_tp_compare.py`）。
#![cfg(feature = "ref-tests")]
use dinfer_core::runner::BatchMode;
use dinfer_kimik3::config::KimiK3Config;
use dinfer_kimik3::kda::{KdaChunkWorkspace, KdaMeta};
use dinfer_kimik3::layer::{Attn, Ffn, LayerBudget, StepMeta};
use dinfer_kimik3::mla::MlaMeta;
use dinfer_kimik3::model::KimiK3Model;
use dinfer_kimik3::weights::{KdaShapes, Shard};
use dinfer_kernels::comm::{Comm, CommId};
use std::cell::Cell;

thread_local! {
    /// 非 0 rank 静默（只算不打印 / 不 dump）。**线程局部**：各 rank 一线程，不能用进程级静态（否则 rank 1 会把 rank 0 也静默）。
    static QUIET: Cell<bool> = const { Cell::new(false) };
}
fn quiet() -> bool {
    QUIET.with(|q| q.get())
}
macro_rules! pr {
    ($($t:tt)*) => { if !quiet() { println!($($t)*); } };
}

fn dump_f32(dir: Option<&Path>, name: &str, v: &[f32]) {
    if quiet() {
        return;
    }
    if let Some(d) = dir {
        let mut b = Vec::with_capacity(v.len() * 4);
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::write(d.join(format!("{name}.f32")), b).unwrap_or_else(|e| panic!("dump {name}: {e}"));
    }
}
use dinfer_layers::{bf16, Device, Gemm, Stream, Tensor};
use std::cell::RefCell;
use std::collections::HashSet;
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
    pr!(
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
    if refs_dir().is_none() || std::env::var("KIMIK3_DIR").is_err() {
        eprintln!("skip: KIMIK3_REFS / KIMIK3_DIR 未设");
        return;
    }
    let world: usize = std::env::var("KIMIK3_TP").ok().and_then(|s| s.parse().ok()).unwrap_or(1).max(1);
    let id = if world > 1 { Some(CommId::new().unwrap()) } else { None };
    let workers: Vec<_> = (1..world)
        .map(|r| std::thread::Builder::new().name(format!("tp-rank{r}")).spawn(move || run_rank(r, world, id)).unwrap())
        .collect();
    run_rank(0, world, id);
    for w in workers {
        w.join().expect("tp rank panicked");
    }
}

/// 一个 rank 的全部工作（world=1 即原单卡测试）。rank>0 静默。
fn run_rank(rank: usize, world: usize, comm_id: Option<CommId>) {
    if rank > 0 {
        QUIET.with(|q| q.set(true));
    }
    let refs = refs_dir().unwrap();
    let dir = PathBuf::from(std::env::var("KIMIK3_DIR").unwrap());
    let dump = std::env::var("KIMIK3_DUMP").ok().map(PathBuf::from);
    if let Some(d) = &dump {
        std::fs::create_dir_all(d).unwrap();
    }
    let dump = dump.as_deref();
    let n_layers: usize = std::env::var("KIMIK3_LAYERS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let ids = npy_i32(&refs, "ids.npy");
    let t = ids.len();
    let cfg = KimiK3Config::from_json(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let tc = &cfg.text_config;
    let hidden = tc.hidden_size as usize;
    let vocab = tc.vocab_size as usize;
    let shard = Shard::new(rank, world).unwrap();
    let sh = KdaShapes::with_shard(tc, shard);

    let dev = Device::new(rank).unwrap();
    let s = dev.default_stream();
    let gemm = Gemm::new(&s).unwrap();
    let block_size = 16usize;
    let kv_blocks = t.div_ceil(block_size) + 1;
    let budget = LayerBudget { max_tokens: t, num_slots: 1, kv_blocks, block_size };
    let (free0, _) = s.mem_info().unwrap();
    let t0 = std::time::Instant::now();
    let mut model = KimiK3Model::load_tp(&s, &dir, tc, n_layers, budget, t, shard).unwrap();
    s.synchronize().unwrap();
    let (free1, _) = s.mem_info().unwrap();
    pr!(
        "  装载前 {n_layers} 层 + 顶层（rank {rank}/{world}）：{:.1} s，显存 +{:.2} GB",
        t0.elapsed().as_secs_f32(),
        (free0 - free1) as f64 / 1e9
    );
    let comm = comm_id.map(|id| Comm::init(&s, rank, world, id).unwrap());
    let comm = comm.as_ref();
    if world > 1 {
        pr!("  TP/EP world {world}：NCCL comm ready（rank {rank}）");
    }
    model.trace = true;

    // meta：单请求
    let ids_d = Tensor::from_vec(&s, ids.clone(), &[t]).unwrap();
    let si = Tensor::from_vec(&s, vec![0i32], &[1]).unwrap();
    let bt: Vec<i32> = (0..kv_blocks as i32).collect();
    let block_table = Tensor::from_vec(&s, bt, &[1, kv_blocks]).unwrap();
    let mut logits = Tensor::<f32>::zeros(&s, &[t, vocab]).unwrap();

    let run = |model: &mut KimiK3Model,
               ids_d: &Tensor<i32>,
               n: usize,
               pos0: usize,
               mode: BatchMode,
               logits: &mut Tensor<f32>|
     -> f32 {
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
            kda: KdaMeta {
                cu_seqlens: &cu,
                state_indices: &si,
                has_initial: &hi,
                num_reqs: 1,
                mode,
                chunk: if use_chunk { Some(&ws) } else { None },
            },
            mla: MlaMeta {
                mode,
                cu_seqlens: &cu,
                block_table: &block_table,
                seq_lens: &seq_lens,
                num_reqs: 1,
                max_blocks: kv_blocks,
            },
            slot_mapping: &slot_mapping,
            comm,
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

    let n = t * hidden;
    let (_, emb_ref) = npy_f32(&refs, "embed.npy");
    let kind_of = |l: usize| {
        format!(
            "{}+{}",
            if tc.is_kda_layer(l as u32) { "kda" } else { "mla" },
            if tc.is_moe_layer(l as u32) { "moe" } else { "dense" }
        )
    };

    // ===== 1. 隔离口径：每层喂 HF 输入 =====
    pr!("== 隔离口径：每层喂 HF 输入（in{{l}}），块残差 = HF in0；MoE 层只在路由一致行判");
    let to16 = |v: &[f32]| v.iter().map(|&f| bf16::from_f32(f)).collect::<Vec<_>>();
    let (_, in0) = npy_f32(&refs, "in0.npy");
    let b0 = Tensor::from_vec(&s, to16(&in0), &[t, hidden]).unwrap();
    let mut iso = Vec::new();
    {
        let cu_host = vec![0i32, t as i32];
        let cu = Tensor::from_vec(&s, cu_host.clone(), &[2]).unwrap();
        let hi = Tensor::from_vec(&s, vec![0i32], &[1]).unwrap();
        let seq_lens = Tensor::from_vec(&s, vec![t as i32], &[1]).unwrap();
        let slot_mapping = Tensor::from_vec(&s, (0..t as i64).collect::<Vec<_>>(), &[t]).unwrap();
        let ws = RefCell::new(KdaChunkWorkspace::new());
        ws.borrow_mut().prepare(&s, &cu_host, &sh).unwrap();
        let meta = StepMeta {
            kda: KdaMeta {
                cu_seqlens: &cu,
                state_indices: &si,
                has_initial: &hi,
                num_reqs: 1,
                mode: BatchMode::Prefill,
                chunk: Some(&ws),
            },
            mla: MlaMeta {
                mode: BatchMode::Prefill,
                cu_seqlens: &cu,
                block_table: &block_table,
                seq_lens: &seq_lens,
                num_reqs: 1,
                max_blocks: kv_blocks,
            },
            slot_mapping: &slot_mapping,
            comm,
        };
        let mut out = Tensor::<bf16>::zeros(&s, &[t, hidden]).unwrap();
        for l in 0..n_layers {
            let (_, xin) = npy_f32(&refs, &format!("in{l}.npy"));
            let (_, want) = npy_f32(&refs, &format!("layer{l}.npy"));
            let xd = Tensor::from_vec(&s, to16(&xin), &[t, hidden]).unwrap();
            let layer = &mut model.layers[l];
            if let Attn::Kda(k) = &mut layer.attn {
                k.reset_slot(&s, 0).unwrap();
            }
            let blocks: Vec<&Tensor<bf16>> = if l == 0 { vec![] } else { vec![&b0] };
            layer.forward(&s, &gemm, t, &xd, &blocks, &meta, &mut model.act, &mut out).unwrap();
            s.synchronize().unwrap();
            let got = rows(&out, &s, n);
            dump_f32(dump, &format!("iso{l}"), &got);
            if let Ffn::LatentMoe(moe) = &layer.ffn {
                // 定位 TP 漂移用：路由 id（f32 存）、专家加权和（norm/up 之前）、routed 输出（up 之后）
                let (ids_g, w_g) = moe.last_routing(&s, &model.act.ffn, t).unwrap();
                dump_f32(dump, &format!("iso{l}_topk_ids"), &ids_g.iter().map(|&i| i as f32).collect::<Vec<_>>());
                dump_f32(dump, &format!("iso{l}_topk_w"), &w_g);
                dump_f32(dump, &format!("iso{l}_expert_sum"), &rows(&model.act.ffn.ysum, &s, t * model.act.ffn.ysum.shape[1]));
                dump_f32(dump, &format!("iso{l}_routed_out"), &rows(&model.act.ffn.yup, &s, n));
                if let Attn::Kda(_) = &layer.attn {
                    dump_f32(dump, &format!("iso{l}_attn_normed"), &rows(&model.act.kda.n, &s, t * model.act.kda.n.shape[1]));
                }
                dump_f32(dump, &format!("iso{l}_attn_out"), &rows(&model.act.a, &s, n));
                dump_f32(dump, &format!("iso{l}_mlp_in"), &rows(&model.act.mn, &s, n));
            }
            let kind = kind_of(l);
            // MoE：路由一致的行
            let (matched, hit_pairs, total_pairs) = if let Ffn::LatentMoe(moe) = &layer.ffn {
                let kk = moe.router.topk;
                let ids_ref = npy_i32(&refs.join(format!("moe{l}")), "topk_ids.npy");
                let (ids_got, _) = moe.last_routing(&s, &model.act.ffn, t).unwrap();
                let mut m = Vec::new();
                let (mut hp, mut tp) = (0usize, 0usize);
                for i in 0..t {
                    let g: HashSet<i32> = ids_got[i * kk..(i + 1) * kk].iter().cloned().collect();
                    let w: HashSet<i32> = ids_ref[i * kk..(i + 1) * kk].iter().cloned().collect();
                    let inter = g.intersection(&w).count();
                    hp += inter;
                    tp += kk;
                    if inter == kk {
                        m.push(i);
                    }
                }
                (m, hp, tp)
            } else {
                ((0..t).collect::<Vec<_>>(), 0, 0)
            };
            let (rel_all, cos_all) = compare(&format!("layer {l} [{kind}] 全部行"), &got, &want);
            let sel = |v: &[f32]| -> Vec<f32> {
                matched.iter().flat_map(|&i| v[i * hidden..(i + 1) * hidden].iter().cloned()).collect()
            };
            let (rel, cos) = if matched.len() < t {
                compare(&format!("layer {l} 路由一致 {} 行", matched.len()), &sel(&got), &sel(&want))
            } else {
                (rel_all, cos_all)
            };
            if total_pairs > 0 {
                pr!(
                    "      top-16 pair 级 {hit_pairs}/{total_pairs}（{:.2}%），路由整行一致 {}/{t}",
                    100.0 * hit_pairs as f64 / total_pairs as f64,
                    matched.len()
                );
            }
            iso.push((l, kind, rel, cos, matched.len()));
        }
    }

    // ===== 2. 链式口径：端到端 =====
    pr!("== 链式口径：prefill T={t}，前 {n_layers} 层一气算完（末 token 范数：ours / HF）");
    let ms = run(&mut model, &ids_d, t, 0, BatchMode::Prefill, &mut logits);
    let mut chain = Vec::new();
    for l in 0..n_layers {
        let (_, want) = npy_f32(&refs, &format!("layer{l}.npy"));
        let got = rows(&model.layer_out[l], &s, n);
        dump_f32(dump, &format!("chain{l}"), &got);
        let kind = kind_of(l);
        let (rel, cos) = compare(&format!("layer {l} [{kind}]"), &got, &want);
        let (ng, nw) = (norm(&got[(t - 1) * hidden..]), norm(&want[(t - 1) * hidden..]));
        pr!("      末 token 范数 {ng:.3} / {nw:.3}");
        chain.push((l, kind, rel, cos, ng, nw));
    }
    pr!("      embed 末 token 范数 {:.3}", norm(&emb_ref[(t - 1) * hidden..]));
    let (_, res_ref) = npy_f32(&refs, "res_out.npy");
    let (_, final_ref) = npy_f32(&refs, "final.npy");
    let (_, logits_ref) = npy_f32(&refs, "logits_last.npy");
    let (rel_r, cos_r) = compare("output_attn_res", &rows(model.last_res_out(), &s, n), &res_ref);
    let (rel_f, cos_f) = compare("final (norm)", &rows(model.last_normed(), &s, n), &final_ref);
    let lg = logits.to_vec(&s).unwrap();
    let last = &lg[(t - 1) * vocab..t * vocab];
    dump_f32(dump, "final", &rows(model.last_normed(), &s, n));
    dump_f32(dump, "logits_last", last);
    let (rel_l, cos_l) = compare("logits 末 token", last, &logits_ref);
    let (o5, r5) = (top5(last), top5(&logits_ref));
    pr!("  top-5 ours {:?}", o5.iter().map(|(i, v)| format!("{i}:{v:.3}")).collect::<Vec<_>>());
    pr!("  top-5 HF   {:?}", r5.iter().map(|(i, v)| format!("{i}:{v:.3}")).collect::<Vec<_>>());
    pr!("  块残差数 {}，prefill 耗时 {ms:.1} ms（首轮，含 D2H 同步）", model.num_blocks());

    // ===== 3. decode：prefill T-1 + decode 1 =====
    pr!("== decode：prefill {} + decode 1", t - 1);
    let final_full = rows(model.last_normed(), &s, n)[(t - 1) * hidden..].to_vec();
    let ids_head = Tensor::from_vec(&s, ids[..t - 1].to_vec(), &[t - 1]).unwrap();
    let ids_tail = Tensor::from_vec(&s, ids[t - 1..].to_vec(), &[1]).unwrap();
    let _ = run(&mut model, &ids_head, t - 1, 0, BatchMode::Prefill, &mut logits);
    let ms_d = run(&mut model, &ids_tail, 1, t - 1, BatchMode::Decode, &mut logits);
    let final_dec = rows(model.last_normed(), &s, hidden);
    let (rel_d, cos_d) = compare("decode final vs prefill[-1]", &final_dec, &final_full);
    let lg_d = logits.to_vec(&s).unwrap()[..vocab].to_vec();
    dump_f32(dump, "decode_logits", &lg_d);
    let (_rel_dl, cos_dl) = compare("decode logits vs HF", &lg_d, &logits_ref);
    let d5 = top5(&lg_d);
    pr!("  decode top-5 {:?}", d5.iter().map(|(i, v)| format!("{i}:{v:.3}")).collect::<Vec<_>>());
    pr!("  decode 1 token 耗时 {ms_d:.1} ms（含 D2H 同步 + AttnRes host softmax）");

    pr!("== 汇总");
    pr!("  隔离口径（HF 输入，MoE 路由一致行）：");
    for (l, kind, rel, cos, m) in &iso {
        pr!("    layer {l:<2} {kind:<10} rel {rel:.3e}  cos {cos:.6}  行 {m}/{t}");
    }
    pr!("  链式口径（端到端）：");
    for (l, kind, rel, cos, ng, nw) in &chain {
        pr!("    layer {l:<2} {kind:<10} rel {rel:.3e}  cos {cos:.6}  末 token 范数 {ng:.3}/{nw:.3}");
    }
    pr!("    res_out cos {cos_r:.6}  final cos {cos_f:.6}  logits cos {cos_l:.6}  decode-vs-prefill cos {cos_d:.6}  decode logits cos {cos_dl:.6}");

    // 断言
    for (l, kind, rel, cos, m) in &iso {
        assert!(*m * 2 >= t, "layer {l}：路由一致行太少 {m}/{t}");
        assert!(*cos >= 0.999 && *rel <= 5e-2, "隔离 layer {l} [{kind}]：rel {rel:.3e} cos {cos:.6}");
    }
    let _ = (rel_r, rel_f, rel_l);
    // TP：all-reduce 的 bf16 求和顺序 + MoE 平局行路由差异使链式末端对 HF 的 cos 略低（tp2 实测 0.998999 vs 单卡 0.999897）；
    // tp 与单卡的直接对比用 KIMIK3_DUMP + k3_tp_compare.py（≥ 0.9998）判，这里对 HF 的门槛放到 0.998。
    let logits_min = if world > 1 { 0.998 } else { 0.999 };
    assert!(cos_f >= 0.998, "链式 final：cos {cos_f:.6}");
    assert!(cos_l >= logits_min, "链式 logits：cos {cos_l:.6}（门槛 {logits_min}）");
    assert_eq!(o5[0].0, r5[0].0, "末 token top-1 不一致");
    // TP：decode 与整段 prefill 各自经历不同的 MoE 平局翻转（tp2 实测 cos 0.99837 / rel 6e-2；单卡 0.99997 / 9e-3），放到 0.998 / 1e-1
    let (dmin, drel) = if world > 1 { (0.998, 1e-1) } else { (0.999, 5e-2) };
    assert!(cos_d >= dmin && rel_d <= drel, "decode vs prefill：rel {rel_d:.3e} cos {cos_d:.6}（门槛 {dmin} / {drel}）");
    assert!(cos_dl >= logits_min, "decode logits：cos {cos_dl:.6}（门槛 {logits_min}）");
    assert_eq!(d5[0].0, r5[0].0, "decode top-1 不一致");
}

fn tracing_subscriber_init() -> Result<(), ()> {
    // RUST_LOG=info 时打印每层装载时间 / 显存（model.rs 的 tracing::info!）
    if std::env::var("RUST_LOG").is_ok() {
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .try_init()
            .map_err(|_| ())
    } else {
        Ok(())
    }
}
