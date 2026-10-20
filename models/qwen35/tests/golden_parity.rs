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
    let exec = Qwen35Executor::new(e.s.clone(), e.gemm, e.model, NUM_BLOCKS).unwrap();
    let mut runner = Runner::new(exec, BLOCK);
    let params = SchedulerParams {
        sched: SchedulerConfig {
            max_num_seqs: 8,
            max_num_batched_tokens: budget,
            enable_chunked_prefill: true,
            long_prefill_token_threshold: 0,
            decode_exempt_from_budget: false,
        },
        cache: CacheConfig { block_size: BLOCK as u32, enable_prefix_caching: false, ..Default::default() },
        max_model_len: 512,
        num_spec_tokens: 0,
        async_scheduling: overlap,
        eos_token_ids: vec![],
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
    let mut sched = Scheduler::new(params, g, NUM_BLOCKS as u32);
    let sp = SamplingParams { max_tokens: gen, temperature: 0.0, ..Default::default() };
    for (i, p) in prompts.iter().enumerate() {
        sched.add_request(Request::new(i as u64 + 1, p.iter().map(|&x| x as u32).collect(), sp.clone(), 0));
    }
    let mut out = vec![Vec::new(); prompts.len()];
    let mut steps = 0;
    let mut async_steps = 0;
    let mut pending: Option<(dinfer_core::sched::SchedulerOutput, dinfer_core::runner::Pending)> = None;
    while sched.has_requests() || pending.is_some() {
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
        let res = sched.update_from_output(&so, &mro);
        for (rid, toks, _) in res.iter() {
            out[rid as usize - 1].extend_from_slice(toks);
        }
        steps += 1;
        assert!(steps < 10 * gen as usize + 20, "runaway");
    }
    if overlap {
        assert!(async_steps > 0, "overlap requested but no step went through the async path");
        eprintln!("qwen35: engine overlap: {async_steps} async steps");
    }
    let (g, e) = (runner.exec.graph_steps, runner.exec.eager_steps);
    eprintln!("qwen35: engine {steps} steps for {} prompts × {gen}; graph steps {g}, eager steps {e}", prompts.len());
    // decode 步必须真走了图重放（NO_GRAPH 时全 eager）；两条路径 golden 都要过
    if std::env::var("NO_GRAPH").is_ok() {
        assert_eq!(g, 0, "NO_GRAPH set but graph steps > 0");
    } else {
        assert!(g > 0, "no decode step went through the cuda graph");
    }
    out
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
