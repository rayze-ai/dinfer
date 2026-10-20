//! Qwen35 里程碑 4：tiny 权重对拍 `scripts/dev/make_qwen35_tiny.py` 的纯 torch golden。
//!
//! 1. `golden_prefill_logits`：3 条 prompt 全位置 logits（max abs + argmax 只在非平局行断言）；
//! 2. `golden_greedy`：Scheduler + Runner + Executor greedy 16 步；
//! 3. `golden_two_requests`：两条 prompt 并发（chunked prefill + 混批，覆盖 GDN 双 slot）。
//!
//! 跑法：`cargo test -p dinfer-qwen35 --features ref-tests --test golden_parity -- --ignored --test-threads=1`
#![cfg(feature = "ref-tests")]

use dinfer_core::config::{CacheConfig, SchedulerConfig};
use dinfer_core::kv::{GroupInfo, KvCacheSpec};
use dinfer_core::request::Request;
use dinfer_core::runner::{BatchMode, Runner};
use dinfer_core::sample::SamplingParams;
use dinfer_core::sched::{Scheduler, SchedulerParams};
use dinfer_layers::{Device, Gemm, Stream, Tensor, WeightSource};
use dinfer_qwen35::config::Qwen35Config;
use dinfer_qwen35::executor::Qwen35Executor;
use dinfer_qwen35::model::{Forward, Qwen35Model};
use std::path::PathBuf;

const BLOCK: usize = 16;
const NUM_BLOCKS: usize = 64;
/// 与 golden 的比较口径：同一套 kernel 口径（bf16 边界），只剩累加顺序差。
const MAX_ABS: f32 = 0.6;
const TIE_GAP: f32 = 0.15;

fn golden_dir() -> PathBuf {
    let p = PathBuf::from(std::env::var("QWEN35_GOLDEN").unwrap_or_else(|_| "test/qwen35/golden".into()));
    if p.is_absolute() {
        p
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join(p)
    }
}
fn npy_f32(name: &str) -> (Vec<usize>, Vec<f32>) {
    let b = std::fs::read(golden_dir().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    let a = npyz::NpyFile::new(&b[..]).unwrap();
    (a.shape().iter().map(|&d| d as usize).collect(), a.into_vec::<f32>().unwrap())
}
fn npy_i32(name: &str) -> (Vec<usize>, Vec<i32>) {
    let b = std::fs::read(golden_dir().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    let a = npyz::NpyFile::new(&b[..]).unwrap();
    (a.shape().iter().map(|&d| d as usize).collect(), a.into_vec::<i32>().unwrap())
}
fn argmax(row: &[f32]) -> usize {
    (1..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b })
}
fn top_gap(row: &[f32]) -> f32 {
    let mut s = row.to_vec();
    s.sort_by(|a, b| b.partial_cmp(a).unwrap());
    s[0] - s[1]
}
fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

struct Env {
    s: Stream,
    gemm: Gemm,
    cfg: Qwen35Config,
    model: Qwen35Model,
}

fn load() -> Env {
    let dev = Device::new(0).unwrap();
    let s = dev.default_stream();
    let gemm = Gemm::new(&s).unwrap();
    let dir = golden_dir().join("tiny");
    let cfg = Qwen35Config::from_json(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let src = WeightSource::open_dir(&dir).unwrap();
    let model = Qwen35Model::load(&s, &src, cfg.clone(), 128, Some(NUM_BLOCKS), BLOCK, 8).unwrap();
    Env { s, gemm, cfg, model }
}

/// 单请求整段 prefill，返回全部位置的 logits。
fn prefill(e: &mut Env, prompt: &[i32]) -> Vec<f32> {
    let t = prompt.len();
    let v = e.cfg.vocab_size as usize;
    // 每个 prompt 从零状态开始（executor 在"位置 0"的请求上做同样的事）
    e.model.reset_slots(&e.s, &[0]).unwrap();
    let s = &e.s;
    let ids = Tensor::from_vec(s, prompt.to_vec(), &[t]).unwrap();
    let pos = Tensor::from_vec(s, (0..t as i32).collect(), &[t]).unwrap();
    let slots = Tensor::from_vec(s, (0..t as i64).collect(), &[t]).unwrap();
    let cu_host = vec![0i32, t as i32];
    let ws = chunk_ws(s, &e.model.cfg, &cu_host);
    let cu = Tensor::from_vec(s, cu_host, &[2]).unwrap();
    let bt = Tensor::from_vec(s, (0..NUM_BLOCKS as i32).collect(), &[1, NUM_BLOCKS]).unwrap();
    let seq = Tensor::from_vec(s, vec![t as i32], &[1]).unwrap();
    let rows = Tensor::from_vec(s, (0..t as i32).collect(), &[t]).unwrap();
    let state = Tensor::from_vec(s, vec![0i32], &[1]).unwrap();
    let has_init = Tensor::from_vec(s, vec![0i32], &[1]).unwrap();
    let mut logits = Tensor::<f32>::zeros(s, &[t, v]).unwrap();
    let req_ids = [1u64];
    let f = Forward {
        input_ids: &ids,
        positions: &pos,
        slot_mapping: &slots,
        meta: dinfer_qwen35::layer::StepMeta {
            mode: BatchMode::Prefill,
            cu_seqlens: &cu,
            seq_lens: &seq,
            block_table: &bt,
            state_indices: &state,
            has_initial: &has_init,
            num_reqs: 1,
            max_blocks: NUM_BLOCKS,
            max_seqlen: t,
            gdn_chunk: ws.as_ref(),
            gdn_snapshot: None,
        },
        logits_rows: &rows,
        t,
        rows: t,
        req_ids: &req_ids,
    };
    e.model.forward(s, &e.gemm, &f, &mut logits).unwrap();
    logits.to_vec(s).unwrap()
}

fn check_rows(tag: &str, got: &[f32], want: &[f32], v: usize) -> usize {
    let rows = want.len() / v;
    let (mut worst, mut bad, mut ties) = (0f32, 0, 0);
    for r in 0..rows {
        let (g, w) = (&got[r * v..(r + 1) * v], &want[r * v..(r + 1) * v]);
        worst = worst.max(max_abs(g, w));
        let gap = top_gap(w);
        if argmax(g) != argmax(w) {
            if gap > TIE_GAP {
                bad += 1;
                eprintln!("{tag}: row {r}: argmax {} vs ref {} (gap {gap:.4})", argmax(g), argmax(w));
            } else {
                ties += 1;
            }
        }
    }
    eprintln!("{tag}: {rows} rows, worst max abs {worst:.4}, argmax mismatches {bad} (near-tie {ties})");
    assert!(worst < MAX_ABS, "{tag}: max abs {worst} > {MAX_ABS}");
    assert_eq!(bad, 0, "{tag}: argmax differs on a non-tie row");
    rows
}


/// GDN chunk prefill 工作区（与 executor 同口径：prefill 步建表，`GDN_PREFILL=recurrent` 回退 → None）。
fn chunk_ws(
    s: &dinfer_layers::Stream,
    cfg: &dinfer_qwen35::config::Qwen35Config,
    cu: &[i32],
) -> Option<std::cell::RefCell<dinfer_qwen35::gdn::GdnChunkWorkspace>> {
    if matches!(std::env::var("GDN_PREFILL").as_deref(), Ok("recurrent")) {
        return None;
    }
    let n = cu.len() - 1;
    let t = *cu.last().unwrap() as usize;
    if t <= n {
        return None;
    }
    let mut ws = dinfer_qwen35::gdn::GdnChunkWorkspace::new();
    ws.prepare(s, cu, cfg).unwrap();
    Some(std::cell::RefCell::new(ws))
}

#[test]
#[ignore]
fn golden_prefill_logits() {
    let mut e = load();
    let v = e.cfg.vocab_size as usize;
    for name in ["a", "b", "c"] {
        let (_, prompt) = npy_i32(&format!("prompt_{name}.npy"));
        let (_, want) = npy_f32(&format!("expect_logits_{name}.npy"));
        let got = prefill(&mut e, &prompt);
        check_rows(&format!("prefill {name}"), &got, &want, v);
    }
}

fn run_engine(e: Env, prompts: &[Vec<i32>], gen: u32, budget: u32) -> Vec<Vec<u32>> {
    run_engine_mode(e, prompts, gen, budget, false)
}

/// `overlap`：照 engine 的异步流水——`begin(n+1)` 在 `finish(n)` 之前（scheduler `async_scheduling` 用 placeholder），
/// 步 n+1 的输入 token 在设备上从 `prev_chosen` gather；false = `Runner::execute`（finish 后再调度）。
fn run_engine_mode(e: Env, prompts: &[Vec<i32>], gen: u32, budget: u32, overlap: bool) -> Vec<Vec<u32>> {
    let stages: Vec<Vec<Vec<i32>>> = vec![prompts.to_vec()];
    run_engine_cfg(e, &stages, gen, budget, overlap, 0, NUM_BLOCKS).0
}

/// 递归快照（design §15.7）口径的 engine 驱动：`stages[k]` 是第 k 批一起提交的 prompt（前一批**全部完成**后再提交下一批，
/// 让后一批能命中前一批留下的 KV 块 + GDN 快照）；`snapshot_grain > 0` 开 prefix cache + 快照（槽数 = 模型快照区）。
/// 返回（按提交顺序的输出，scheduler prefix cache 命中 token 数，executor 快照恢复次数，快照保存次数，抢占次数）。
fn run_engine_cfg(
    e: Env,
    stages: &[Vec<Vec<i32>>],
    gen: u32,
    budget: u32,
    overlap: bool,
    snapshot_grain: u32,
    num_blocks: usize,
) -> (Vec<Vec<u32>>, u64, u64, u64, usize) {
    let num_snapshots = e.model.num_snapshots as u32;
    let exec = Qwen35Executor::new(e.s.clone(), e.gemm, e.model, num_blocks).unwrap();
    let mut runner = Runner::new(exec, BLOCK);
    let params = SchedulerParams {
        sched: SchedulerConfig {
            max_num_seqs: 8,
            max_num_batched_tokens: budget,
            enable_chunked_prefill: true,
            long_prefill_token_threshold: 0,
            decode_exempt_from_budget: false,
        },
        cache: CacheConfig {
            block_size: BLOCK as u32,
            hash_block_size: BLOCK as u32,
            enable_prefix_caching: snapshot_grain > 0,
            ..Default::default()
        },
        max_model_len: 512,
        num_spec_tokens: 0,
        async_scheduling: overlap,
        eos_token_ids: vec![],
        recurrent_snapshot_tokens: snapshot_grain,
        recurrent_snapshot_slots: if snapshot_grain > 0 { num_snapshots } else { 0 },
        recurrent_snapshot_chunk: 64,
    };
    let g = vec![GroupInfo {
        spec: KvCacheSpec::FullAttention {
            block_size: BLOCK as u32,
            kv_heads: e.cfg.num_key_value_heads,
            head_size: e.cfg.head_dim,
            dtype_bytes: 2,
        },
        block_size: BLOCK as u32,
    }];
    let mut sched = Scheduler::new(params, g, num_blocks as u32);
    let sp = SamplingParams { max_tokens: gen, temperature: 0.0, ..Default::default() };
    let total: usize = stages.iter().map(|s| s.len()).sum();
    let mut next_id = 1u64;
    let mut stage_iter = stages.iter();
    let mut submit = |sched: &mut Scheduler, batch: &[Vec<i32>]| {
        for p in batch {
            sched.add_request(Request::new(next_id, p.iter().map(|&x| x as u32).collect(), sp.clone(), 0));
            next_id += 1;
        }
    };
    submit(&mut sched, stage_iter.next().unwrap());
    let mut out = vec![Vec::new(); total];
    let mut preempted = 0usize;
    let mut steps = 0;
    let mut async_steps = 0;
    let mut pending: Option<(dinfer_core::sched::SchedulerOutput, dinfer_core::runner::Pending)> = None;
    loop {
        if !sched.has_requests() && pending.is_none() {
            match stage_iter.next() {
                Some(batch) => {
                    submit(&mut sched, batch);
                    continue;
                }
                None => break,
            }
        }
        // engine 同序：有在飞步 n 时先 schedule + begin n+1，再 finish n；n+1 若是同步步（prefill）begin 已跑完
        let (so, mro) = match pending.take() {
            Some((so_n, pend_n)) => {
                let mut next = None;
                if sched.has_requests() {
                    let so_next = sched.schedule();
                    let pend_next = runner.begin(&so_next).unwrap();
                    if runner.last_begin_was_async() {
                        async_steps += 1;
                        next = Some((so_next, pend_next));
                    } else {
                        let mro_next = runner.finish(pend_next).unwrap();
                        let res = sched.update_from_output(&so_next, &mro_next);
                        for (rid, toks, _) in res.iter() {
                            out[rid as usize - 1].extend_from_slice(toks);
                        }
                        steps += 1;
                    }
                }
                let mro_n = runner.finish(pend_n).unwrap();
                pending = next;
                (so_n, mro_n)
            }
            None => {
                let so = sched.schedule();
                let pend = runner.begin(&so).unwrap();
                if overlap && runner.last_begin_was_async() {
                    async_steps += 1;
                    pending = Some((so, pend));
                    continue;
                }
                (so, runner.finish(pend).unwrap())
            }
        };
        preempted += so.preempted_req_ids.len();
        if std::env::var("GOLDEN_TRACE").is_ok() {
            eprintln!(
                "step {steps}: sched {:?} new {:?} cached {:?} resume {:?} snap {:?} preempted {:?} sampled {:?}",
                so.num_scheduled_tokens,
                so.new_reqs.iter().map(|n| (n.req_id, n.num_computed_tokens, n.resume_from.map(|x| x.0))).collect::<Vec<_>>(),
                so.cached_reqs.req_ids.iter().zip(&so.cached_reqs.resumed).zip(&so.cached_reqs.num_computed_tokens).map(|((r, re), c)| (*r, *re, *c)).collect::<Vec<_>>(),
                so.recurrent_resume,
                so.recurrent_snapshot,
                so.preempted_req_ids,
                mro.sampled
            );
        }
        let res = sched.update_from_output(&so, &mro);
        for (rid, toks, _) in res.iter() {
            out[rid as usize - 1].extend_from_slice(toks);
        }
        steps += 1;
        assert!(steps < 10 * gen as usize * total + 40, "runaway");
    }
    if overlap {
        assert!(async_steps > 0, "overlap requested but no step went through the async path");
        eprintln!("qwen35: engine overlap: {async_steps} async steps");
    }
    let (g, e) = (runner.exec.graph_steps, runner.exec.eager_steps);
    eprintln!("qwen35: engine {steps} steps for {total} prompts × {gen}; graph steps {g}, eager steps {e}");
    // decode 步必须真走了图重放（NO_GRAPH 时全 eager）；两条路径 golden 都要过
    if std::env::var("NO_GRAPH").is_ok() {
        assert_eq!(g, 0, "NO_GRAPH set but graph steps > 0");
    } else {
        assert!(g > 0, "no decode step went through the cuda graph");
    }
    let (_, hits) = sched.prefix_cache_stats();
    (out, hits, runner.exec.snapshot_restores, runner.exec.snapshot_saves, preempted)
}

/// 异步流水（engine 口径：begin n+1 在 finish n 之前，设备侧 gather 在飞 token）下 greedy 与 golden 一致。
#[test]
#[ignore]
fn golden_greedy_overlap() {
    let e = load();
    let v = e.cfg.vocab_size as usize;
    let (_, prompt) = npy_i32("prompt_a.npy");
    let (_, want) = npy_i32("expect_greedy_a.npy");
    let (_, steps) = npy_f32("expect_step_logits_a.npy");
    let out = run_engine_mode(e, &[prompt], want.len() as u32, 128, true);
    let n = compare_greedy("greedy a (async overlap)", &out[0], &want, &steps, v);
    assert!(n >= 1, "too few comparable greedy steps: {n}");
}

/// 两请求并发 + 异步流水（chunked prefill 的插队同步步 + decode 异步步交替，slot 表 / gather 行号跨步变化）。
#[test]
#[ignore]
fn golden_two_requests_overlap() {
    let e = load();
    let v = e.cfg.vocab_size as usize;
    let (_, p1) = npy_i32("prompt_a.npy");
    let (_, p2) = npy_i32("prompt_b.npy");
    let (_, w1) = npy_i32("expect_greedy_a.npy");
    let (_, w2) = npy_i32("expect_greedy_b.npy");
    let (_, l1) = npy_f32("expect_step_logits_a.npy");
    let (_, l2) = npy_f32("expect_step_logits_b.npy");
    let out = run_engine_mode(e, &[p1, p2], w1.len() as u32, 8, true);
    let n1 = compare_greedy("a (concurrent, async overlap)", &out[0], &w1, &l1, v);
    let n2 = compare_greedy("b (concurrent, async overlap)", &out[1], &w2, &l2, v);
    assert!(n1 + n2 >= 1, "too few comparable steps: {n1} / {n2}");
}

/// 逐步比 greedy token：非平局步（ref top-2 gap > TIE_GAP）必须一致；平局步跳过不计。
/// 一旦某步 token 与 ref 不同（只可能发生在平局步），之后的序列就分叉了，后续步不再可比 → 返回已比较的决定性步数。
fn compare_greedy(tag: &str, got: &[u32], want: &[i32], step_logits: &[f32], v: usize) -> usize {
    let mut n = 0;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let gap = top_gap(&step_logits[i * v..(i + 1) * v]);
        if g != w as u32 {
            assert!(gap <= TIE_GAP, "{tag}: step {i} token {g} != ref {w} with gap {gap:.4} (not a tie)");
            eprintln!("qwen35: {tag}: diverged at step {i} on a near-tie (gap {gap:.4}); decisive steps compared {n}");
            return n;
        }
        if gap > TIE_GAP {
            n += 1;
        }
    }
    eprintln!("qwen35: {tag}: all {} greedy tokens match ({n} decisive)", got.len());
    n
}

#[test]
#[ignore]
fn golden_greedy() {
    let e = load();
    let v = e.cfg.vocab_size as usize;
    let (_, prompt) = npy_i32("prompt_a.npy");
    let (_, want) = npy_i32("expect_greedy_a.npy");
    let (_, steps) = npy_f32("expect_step_logits_a.npy");
    let out = run_engine(e, &[prompt], want.len() as u32, 128);
    let n = compare_greedy("greedy a", &out[0], &want, &steps, v);
    assert!(n >= 1, "too few comparable greedy steps: {n}");
}

/// chunked prefill（块 4/4/1）+ 教师强制的逐 token decode：`prompt_c` 的 greedy 在步 1 有平局（ref top-2 gap 0.03），
/// 所以不用"自己采样再喂回去"，而是每步喂 golden token、只断言非平局行的 argmax。
#[test]
#[ignore]
fn golden_chunked_teacher_forced() {
    let mut e = load();
    let v = e.cfg.vocab_size as usize;
    let (_, prompt) = npy_i32("prompt_c.npy");
    let (_, want) = npy_f32("expect_logits_c.npy");
    let (_, gr) = npy_i32("expect_greedy_c.npy");
    let (_, steps) = npy_f32("expect_step_logits_c.npy");
    let t = prompt.len();
    e.model.reset_slots(&e.s, &[0]).unwrap();
    let s = e.s.clone();
    let bt = Tensor::from_vec(&s, (0..NUM_BLOCKS as i32).collect(), &[1, NUM_BLOCKS]).unwrap();
    let state = Tensor::from_vec(&s, vec![0i32], &[1]).unwrap();
    let req_ids = [1u64];
    // prefill 分块 4 / 4 / 1
    let mut pos0 = 0usize;
    while pos0 < t {
        let n = 4.min(t - pos0);
        let mode =
            if pos0 == 0 { BatchMode::Prefill } else if n == 1 { BatchMode::Decode } else { BatchMode::Paged };
        let ids = Tensor::from_vec(&s, prompt[pos0..pos0 + n].to_vec(), &[n]).unwrap();
        let posv = Tensor::from_vec(&s, (pos0 as i32..(pos0 + n) as i32).collect(), &[n]).unwrap();
        let slots = Tensor::from_vec(&s, (pos0 as i64..(pos0 + n) as i64).collect(), &[n]).unwrap();
        let cu_host = vec![0i32, n as i32];
        let ws = chunk_ws(&s, &e.model.cfg, &cu_host);
        let cu = Tensor::from_vec(&s, cu_host, &[2]).unwrap();
        let seq = Tensor::from_vec(&s, vec![(pos0 + n) as i32], &[1]).unwrap();
        let rows = Tensor::from_vec(&s, vec![(n - 1) as i32], &[1]).unwrap();
        let has_init = Tensor::from_vec(&s, vec![i32::from(pos0 > 0)], &[1]).unwrap();
        let mut logits = Tensor::<f32>::zeros(&s, &[1, v]).unwrap();
        let f = Forward {
            input_ids: &ids,
            positions: &posv,
            slot_mapping: &slots,
            meta: dinfer_qwen35::layer::StepMeta {
                mode,
                cu_seqlens: &cu,
                seq_lens: &seq,
                block_table: &bt,
                state_indices: &state,
                has_initial: &has_init,
                num_reqs: 1,
                max_blocks: NUM_BLOCKS,
                max_seqlen: n,
                gdn_chunk: ws.as_ref(),
                gdn_snapshot: None,
            },
            logits_rows: &rows,
            t: n,
            rows: 1,
            req_ids: &req_ids,
        };
        e.model.forward(&s, &e.gemm, &f, &mut logits).unwrap();
        let got = logits.to_vec(&s).unwrap();
        // 中间 chunk 的末行只与 golden 的对应位置比（非最后一块）
        if pos0 + n < t {
            let w = &want[(pos0 + n - 1) * v..(pos0 + n) * v];
            assert!(max_abs(&got, w) < MAX_ABS, "chunk end {pos0}+{n}: max abs {}", max_abs(&got, w));
        }
        pos0 += n;
    }
    // decode：逐步喂 golden token
    let mut checked = 0;
    for (i, &tok) in gr.iter().take(gr.len() - 1).enumerate() {
        let pos = t + i;
        let ids = Tensor::from_vec(&s, vec![tok], &[1]).unwrap();
        let posv = Tensor::from_vec(&s, vec![pos as i32], &[1]).unwrap();
        let slots = Tensor::from_vec(&s, vec![pos as i64], &[1]).unwrap();
        let cu = Tensor::from_vec(&s, vec![0i32, 1], &[2]).unwrap();
        let seq = Tensor::from_vec(&s, vec![(pos + 1) as i32], &[1]).unwrap();
        let rows = Tensor::from_vec(&s, vec![0i32], &[1]).unwrap();
        let has_init = Tensor::from_vec(&s, vec![1i32], &[1]).unwrap();
        let mut logits = Tensor::<f32>::zeros(&s, &[1, v]).unwrap();
        let f = Forward {
            input_ids: &ids,
            positions: &posv,
            slot_mapping: &slots,
            meta: dinfer_qwen35::layer::StepMeta {
                mode: BatchMode::Decode,
                cu_seqlens: &cu,
                seq_lens: &seq,
                block_table: &bt,
                state_indices: &state,
                has_initial: &has_init,
                num_reqs: 1,
                max_blocks: NUM_BLOCKS,
                max_seqlen: 1,
                gdn_chunk: None,
                gdn_snapshot: None,
            },
            logits_rows: &rows,
            t: 1,
            rows: 1,
            req_ids: &req_ids,
        };
        e.model.forward(&s, &e.gemm, &f, &mut logits).unwrap();
        let got = logits.to_vec(&s).unwrap();
        // 喂 gr[i] 之后的 logits 预测第 i+1 个 token → 对 `steps[i+1]`
        let w = &steps[(i + 1) * v..(i + 2) * v];
        let mx = max_abs(&got, w);
        assert!(mx < MAX_ABS, "decode step {i}: max abs {mx}");
        if top_gap(w) > TIE_GAP {
            assert_eq!(argmax(&got), argmax(w), "decode step {i}: argmax on a non-tie row (gap {})", top_gap(w));
        }
        checked += 1;
    }
    assert!(checked >= 12, "too few decode steps: {checked}");
    eprintln!("qwen35: chunked prefill (4/4/1) + {checked} teacher-forced decode steps OK");
}

/// 两条 prompt 并发（预算 8 < prompt 长度 → chunked prefill + 混批，覆盖两个 GDN slot 与 paged cache）。
/// 用 a / b（greedy 16 步内没有平局的行）；`prompt_c` 留给 teacher-forced 用例。
#[test]
#[ignore]
fn golden_two_requests() {
    let e = load();
    let v = e.cfg.vocab_size as usize;
    let (_, p1) = npy_i32("prompt_a.npy");
    let (_, p2) = npy_i32("prompt_b.npy");
    let (_, w1) = npy_i32("expect_greedy_a.npy");
    let (_, w2) = npy_i32("expect_greedy_b.npy");
    let (_, l1) = npy_f32("expect_step_logits_a.npy");
    let (_, l2) = npy_f32("expect_step_logits_b.npy");
    let out = run_engine(e, &[p1, p2], w1.len() as u32, 8);
    let n1 = compare_greedy("a (concurrent, chunked)", &out[0], &w1, &l1, v);
    let n2 = compare_greedy("b (concurrent)", &out[1], &w2, &l2, v);
    assert!(n1 + n2 >= 1, "too few comparable steps: {n1} / {n2}");
}

// ---------------------------------------------------------------- 递归状态按块缓存（design §15.7）

/// 带快照区的 tiny 模型（`snapshots` 个槽；`blocks` 个 KV 块）。
fn load_snap(snapshots: usize, blocks: usize) -> Env {
    let dev = Device::new(0).unwrap();
    let s = dev.default_stream();
    let gemm = Gemm::new(&s).unwrap();
    let dir = golden_dir().join("tiny");
    let cfg = Qwen35Config::from_json(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let src = WeightSource::open_dir(&dir).unwrap();
    let bytes = <Qwen35Model>::gdn_state_bytes_per_slot(&cfg) * snapshots;
    let model: Qwen35Model = Qwen35Model::load_with_budget(&s, &src, cfg.clone(), 128, 128, Some(blocks), BLOCK, 8, (0, 1), 0.9, 0, Some(bytes))
        .unwrap();
    assert_eq!(model.num_snapshots, snapshots);
    Env { s, gemm, cfg, model }
}

/// 一步 forward（单请求，slot / has_init / 位置区间显式），返回 `rows` 行 logits。
#[allow(clippy::too_many_arguments)]
fn step_forward(e: &mut Env, ids: &[i32], pos0: usize, slot: i32, has_init: bool, mode: BatchMode, all_rows: bool) -> Vec<f32> {
    let n = ids.len();
    let v = e.cfg.vocab_size as usize;
    let s = e.s.clone();
    let idt = Tensor::from_vec(&s, ids.to_vec(), &[n]).unwrap();
    let posv = Tensor::from_vec(&s, (pos0 as i32..(pos0 + n) as i32).collect(), &[n]).unwrap();
    let slots = Tensor::from_vec(&s, (pos0 as i64..(pos0 + n) as i64).collect(), &[n]).unwrap();
    let cu_host = vec![0i32, n as i32];
    let ws = if n > 1 { chunk_ws(&s, &e.model.cfg, &cu_host) } else { None };
    let cu = Tensor::from_vec(&s, cu_host, &[2]).unwrap();
    let bt = Tensor::from_vec(&s, (0..NUM_BLOCKS as i32).collect(), &[1, NUM_BLOCKS]).unwrap();
    let seq = Tensor::from_vec(&s, vec![(pos0 + n) as i32], &[1]).unwrap();
    let rows_host: Vec<i32> = if all_rows { (0..n as i32).collect() } else { vec![n as i32 - 1] };
    let nrows = rows_host.len();
    let rows = Tensor::from_vec(&s, rows_host, &[nrows]).unwrap();
    let state = Tensor::from_vec(&s, vec![slot], &[1]).unwrap();
    let hi = Tensor::from_vec(&s, vec![i32::from(has_init)], &[1]).unwrap();
    let mut logits = Tensor::<f32>::zeros(&s, &[nrows, v]).unwrap();
    let req_ids = [1u64];
    let f = Forward {
        input_ids: &idt,
        positions: &posv,
        slot_mapping: &slots,
        meta: dinfer_qwen35::layer::StepMeta {
            mode,
            cu_seqlens: &cu,
            seq_lens: &seq,
            block_table: &bt,
            state_indices: &state,
            has_initial: &hi,
            num_reqs: 1,
            max_blocks: NUM_BLOCKS,
            max_seqlen: n,
            gdn_chunk: ws.as_ref(),
            gdn_snapshot: None,
        },
        logits_rows: &rows,
        t: n,
        rows: nrows,
        req_ids: &req_ids,
    };
    e.model.forward(&s, &e.gemm, &f, &mut logits).unwrap();
    logits.to_vec(&s).unwrap()
}

/// 把 a / b / c 三条 golden prompt 拼成一条 ≥ 2 个 block 的长 prompt（tiny 词表内的 token，确定性）。
fn long_prompt(min_len: usize) -> Vec<i32> {
    let (_, a) = npy_i32("prompt_a.npy");
    let (_, b) = npy_i32("prompt_b.npy");
    let (_, c) = npy_i32("prompt_c.npy");
    let mut p = Vec::new();
    let mut k = 0usize;
    while p.len() < min_len {
        let src = [&a, &b, &c][k % 3];
        // 轮换 + 平移，避免完全周期重复
        p.extend(src.iter().map(|&t| (t + k as i32) % 64));
        k += 1;
    }
    p.truncate(min_len);
    p
}

/// 教师强制参考：整段 prefill `prompt`，再逐 token greedy decode `gen` 步；返回 (greedy token, 每步 logits 行)。
/// 用 slot 0、单请求、与 `golden_chunked_teacher_forced` 相同的 kernel 路径。
fn reference_greedy(e: &mut Env, prompt: &[i32], gen: usize) -> (Vec<i32>, Vec<f32>) {
    let v = e.cfg.vocab_size as usize;
    e.model.reset_slots(&e.s, &[0]).unwrap();
    let first = step_forward(e, prompt, 0, 0, false, BatchMode::Prefill, false);
    let mut toks = vec![argmax(&first) as i32];
    let mut steps = first;
    let mut pos = prompt.len();
    for _ in 1..gen {
        let l = step_forward(e, &[*toks.last().unwrap()], pos, 0, true, BatchMode::Decode, false);
        toks.push(argmax(&l) as i32);
        steps.extend_from_slice(&l);
        pos += 1;
    }
    assert_eq!(steps.len(), gen * v);
    (toks, steps)
}

/// 模型级：对齐点保存 / 恢复快照后续算 = 同一活槽直接续算（逐位相同），且与整段 prefill 的末行一致（MAX_ABS 内）。
#[test]
#[ignore]
fn snapshot_save_restore_bitwise() {
    let mut e = load_snap(2, NUM_BLOCKS);
    let v = e.cfg.vocab_size as usize;
    let p = long_prompt(40);
    let (head, tail) = p.split_at(32);
    // 路径 A：slot 0 算 0..32 → 快照槽 1 → slot 0 继续 32..40
    e.model.reset_slots(&e.s, &[0, 1]).unwrap();
    let _ = step_forward(&mut e, head, 0, 0, false, BatchMode::Prefill, false);
    e.model.save_snapshots(&e.s, &[(0, 1)]).unwrap();
    let a = step_forward(&mut e, tail, 32, 0, true, BatchMode::Paged, true);
    // 路径 B：快照槽 1 → slot 1（另一个活槽），从 32 续算
    e.model.restore_snapshots(&e.s, &[(1, 1)]).unwrap();
    let b = step_forward(&mut e, tail, 32, 1, true, BatchMode::Paged, true);
    assert_eq!(a.len(), b.len());
    let diff = a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    assert_eq!(diff, 0, "restore-from-snapshot must be bitwise identical to continuing in place ({diff} / {} differ)", a.len());
    // 与整段 prefill 的对应行比（chunk 切分不同，数值不逐位，但在 golden 口径内）
    e.model.reset_slots(&e.s, &[0]).unwrap();
    let full = step_forward(&mut e, &p, 0, 0, false, BatchMode::Prefill, true);
    let n = check_rows("snapshot-resume vs full prefill (rows 32..40)", &b, &full[32 * v..], v);
    assert!(n >= 1);
    eprintln!("qwen35: snapshot save/restore bitwise OK ({} logits), {n} decisive rows vs full prefill", a.len());
}

/// engine 级（同 prompt 两次）：第二次命中 prefix cache，对齐到 32 从快照续算；输出 = 教师强制参考（非平局步逐个一致）；
/// 并断言真的发生了命中 / 恢复 / 保存。`ASYNC_SCHED=0` 时同步路径也过（overlap = false）。
#[test]
#[ignore]
fn golden_prefix_hit_resumes_from_snapshot() {
    let v: usize;
    let p = long_prompt(40);
    let gen = 12usize;
    let (want, steps) = {
        let mut e = load();
        v = e.cfg.vocab_size as usize;
        reference_greedy(&mut e, &p, gen)
    };
    let overlap = std::env::var("ASYNC_SCHED").map(|x| x != "0").unwrap_or(true);
    let e = load_snap(4, NUM_BLOCKS);
    let stages = vec![vec![p.clone()], vec![p.clone()]];
    let (out, hits, restores, saves, _) = run_engine_cfg(e, &stages, gen as u32, 64, overlap, 32, NUM_BLOCKS);
    assert_eq!(hits, 32, "second request must hit 32 aligned tokens (KV hit 39 → align 32)");
    assert!(saves >= 1, "first request must save a snapshot at 32 (saves {saves})");
    assert_eq!(restores, 1, "second request must restore exactly one snapshot");
    let n1 = compare_greedy("first (full prefill)", &out[0], &want, &steps, v);
    let n2 = compare_greedy("second (resume @32)", &out[1], &want, &steps, v);
    assert!(n1 >= 4 && n2 >= 4, "too few decisive steps: {n1} / {n2}");
    assert_eq!(out[0], out[1], "same prompt, same greedy output with and without the prefix hit");
}

/// 多轮：先 A（40），再 A+B（56）：命中 A 的 32 对齐点，续算 8 + 16 个新 token；输出 = A+B 的教师强制参考。
#[test]
#[ignore]
fn golden_multi_turn_resume() {
    let a = long_prompt(40);
    let ab = long_prompt(56);
    assert_eq!(&ab[..40], &a[..]);
    let gen = 10usize;
    let (want, steps, v) = {
        let mut e = load();
        let v = e.cfg.vocab_size as usize;
        let (w, s) = reference_greedy(&mut e, &ab, gen);
        (w, s, v)
    };
    let e = load_snap(4, NUM_BLOCKS);
    let overlap = std::env::var("ASYNC_SCHED").map(|x| x != "0").unwrap_or(true);
    let (out, hits, restores, _, _) = run_engine_cfg(e, &[vec![a], vec![ab]], gen as u32, 64, overlap, 32, NUM_BLOCKS);
    assert_eq!(hits, 32, "A+B hits A's 32-aligned prefix");
    assert_eq!(restores, 1);
    let n = compare_greedy("A+B (resume @32)", &out[1], &want, &steps, v);
    assert!(n >= 3, "too few decisive steps: {n}");
}

/// 混批 + 命中 + 抢占重算：第 2 批 = [命中 P 的请求, 新 prompt Q] 同时提交（混批）；KV 池只有 8 块 → 两条 40+ 的请求
/// 跑 decode 时抢占，被抢占者重排后再命中自己留下的前缀 + 快照续算。三条的输出都要 = 各自的教师强制参考。
#[test]
#[ignore]
fn golden_mixed_hit_and_preempt() {
    let p = long_prompt(40);
    let q: Vec<i32> = long_prompt(84)[40..].to_vec(); // 与 p 无共同前缀（long_prompt 的另一段）
    assert_ne!(&q[..BLOCK], &p[..BLOCK], "first KV block must differ (no prefix hit for Q)");
    let gen = 14usize;
    let (wp, sp_, wq, sq, v) = {
        let mut e = load();
        let v = e.cfg.vocab_size as usize;
        let (wp, sp_) = reference_greedy(&mut e, &p, gen);
        let (wq, sq) = reference_greedy(&mut e, &q, gen);
        (wp, sp_, wq, sq, v)
    };
    // 8 块 = 128 token：p (40+14) + q (44+14) = 112 ≤ 128 装得下但 block 粒度下 4+4 块 + 命中块共享……用 7 块逼出抢占
    let blocks = 7usize;
    let e = load_snap(4, blocks);
    let grain = std::env::var("GOLDEN_GRAIN").ok().and_then(|x| x.parse().ok()).unwrap_or(32u32);
    let overlap = std::env::var("ASYNC_SCHED").map(|x| x != "0").unwrap_or(true);
    let (out, hits, restores, saves, preempted) =
        run_engine_cfg(e, &[vec![p.clone()], vec![p.clone(), q.clone()]], gen as u32, 64, overlap, grain, blocks);
    if grain == 0 {
        let n2 = compare_greedy("Q (fresh, mixed, no prefix cache)", &out[2], &wq, &sq, v);
        eprintln!("qwen35: grain 0 variant: preempted {preempted}, Q decisive {n2}");
        return;
    }
    eprintln!("qwen35: mixed: hits {hits} restores {restores} saves {saves} preempted {preempted}");
    assert!(hits >= 32, "second P must hit (hits {hits})");
    assert!(restores >= 1);
    assert!(preempted >= 1, "expected at least one preemption with {blocks} blocks");
    let n0 = compare_greedy("P (first)", &out[0], &wp, &sp_, v);
    let n1 = compare_greedy("P (hit, mixed with Q)", &out[1], &wp, &sp_, v);
    let n2 = compare_greedy("Q (fresh, mixed)", &out[2], &wq, &sq, v);
    // tiny 随机权重平局多：每条至少 2 个可判定步（分叉只允许发生在平局步，compare_greedy 已断言）
    assert!(n0 >= 2 && n1 >= 2 && n2 >= 2, "too few decisive steps: {n0} / {n1} / {n2}");
}

/// r8b：快照由 chunk prefill kernel 在 chunk 边界顺带写出、调度不切步。粒度 64 = kernel chunk：120 token 的 prompt
/// 一步算完（跨对齐点 64，不切步；该快照由 ho kernel 在 chunk 1 末、conv 快照 kernel 在短卷积前写出，不是步末 dtod）；
/// 同 prompt 第二次命中 64 从该快照续算，输出 = 教师强制参考且与第一次逐字同。`saves == 1 && hits == 64` 证明第一次
/// 没有在 64 切步（切步的话 r8 口径下仍 saves 1，但 `test/` 里 `GOLDEN_TRACE=1` 可看到 sched [(1,120)] 一步）。
/// 跨多个对齐点的一步在 tiny 的 max_tokens 128 下放不下，由 B300 engine 口径（1024 粒度、6 K 单发）覆盖。
#[test]
#[ignore]
fn golden_kernel_boundary_snapshot_one_step() {
    let p = long_prompt(120);
    let gen = 10usize;
    let (want, steps, v) = {
        let mut e = load();
        let v = e.cfg.vocab_size as usize;
        let (w, s) = reference_greedy(&mut e, &p, gen);
        (w, s, v)
    };
    let e = load_snap(4, NUM_BLOCKS);
    let overlap = std::env::var("ASYNC_SCHED").map(|x| x != "0").unwrap_or(true);
    // 预算 128 ≥ 120：第一条请求一步 prefill 完（跨 64）
    let (out, hits, restores, saves, _) = run_engine_cfg(e, &[vec![p.clone()], vec![p.clone()]], gen as u32, 128, overlap, 64, NUM_BLOCKS);
    assert_eq!(saves, 1, "one-step prefill of 120 tokens must register the snapshot at 64 (saves {saves})");
    assert_eq!(hits, 64, "second request hits 64 (KV hit 119 → 112 → align 64)");
    assert_eq!(restores, 1);
    let n1 = compare_greedy("first (one-step prefill, kernel-written snapshot)", &out[0], &want, &steps, v);
    let n2 = compare_greedy("second (resume @64 from kernel-written snapshot)", &out[1], &want, &steps, v);
    assert!(n1 >= 3 && n2 >= 3, "too few decisive steps: {n1} / {n2}");
    assert_eq!(out[0], out[1], "same prompt, same greedy output with and without the prefix hit");
}
