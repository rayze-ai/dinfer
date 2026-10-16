//! Q1：Qwen3-0.6B **真权重** vs HF（同机 3090，HF 侧 bf16 SDPA）。golden：`scripts/verify/gen_qwen3_golden.py`。
//!
//! 1. `real_prefill_logits_match_hf`：chat prompt 全部位置 logits 对拍（max abs；argmax 在 HF gap > TIE_GAP 的行一致）。
//! 2. `real_greedy_matches_hf`：Scheduler+Runner+Executor greedy 32 步，与 HF 逐 token 一致（HF 平局步后停止比较）。
//! 3. `real_two_requests_concurrent`：两条 prompt 并发（含 enable_thinking=false 的第二条），各自与 HF 一致。
//!
//! 跑法：QWEN3_DIR=/data/ref/models/qwen3-0.6b QWEN3_GOLDEN=$(realpath test/qwen3_golden) \
//!   cargo test -p dinfer-qwen3 --features ref-tests --test real_parity -- --ignored --nocapture --test-threads=1
#![cfg(feature = "ref-tests")]

use dinfer_core::config::{CacheConfig, SchedulerConfig};
use dinfer_core::kv::{GroupInfo, KvCacheSpec};
use dinfer_core::request::Request;
use dinfer_core::runner::{BatchMode, Runner};
use dinfer_core::sample::SamplingParams;
use dinfer_core::sched::{Scheduler, SchedulerParams};
use dinfer_layers::{Device, Gemm, Tensor, WeightSource};
use dinfer_qwen3::config::Qwen3Config;
use dinfer_qwen3::executor::Qwen3Executor;
use dinfer_qwen3::model::{AttnMeta, Forward, Qwen3Model};
use std::path::PathBuf;

fn model_dir() -> PathBuf {
    PathBuf::from(std::env::var("QWEN3_DIR").unwrap_or_else(|_| "/data/ref/models/qwen3-0.6b".into()))
}
fn golden_dir() -> PathBuf {
    let p = PathBuf::from(std::env::var("QWEN3_GOLDEN").unwrap_or_else(|_| "test/qwen3_golden".into()));
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

const BLOCK: usize = 16;
const NUM_BLOCKS: usize = 256;
/// 真权重 bf16 噪声底：HF 自己 eager vs sdpa 的 logits 差就有 0.68（|logits| ~36，28 层累积）；我们逐层 bisect 显示
/// attention 输出前 10 层 ≤ 0.03、往后 bf16 累积到 ~1，与 HF 两种 attention 实现之间的差同量级 → 不是 bug。
/// 口径：全行 logits max abs < 2.0；argmax 不一致只允许出现在 HF top-2 gap ≤ TIE_GAP 的行，且一致行 ≥ 80%。
const MAX_ABS: f32 = 2.0;
const TIE_GAP: f32 = 1.0;

struct Env {
    s: dinfer_layers::Stream,
    gemm: Gemm,
    cfg: Qwen3Config,
    model: Qwen3Model,
}

fn load(max_tokens: usize) -> Env {
    let dev = Device::new(0).unwrap();
    let s = dev.default_stream();
    let gemm = Gemm::new(&s).unwrap();
    let cfg =
        Qwen3Config::from_json(&std::fs::read_to_string(model_dir().join("config.json")).unwrap()).unwrap();
    let src = WeightSource::open_dir(&model_dir()).unwrap();
    let t0 = std::time::Instant::now();
    let model = Qwen3Model::load(&s, &src, cfg.clone(), max_tokens, Some(NUM_BLOCKS), BLOCK).unwrap();
    eprintln!("q1: loaded Qwen3-0.6B real weights in {:.1}s", t0.elapsed().as_secs_f32());
    Env { s, gemm, cfg, model }
}

#[test]
#[ignore]
fn real_prefill_logits_match_hf() {
    let mut e = load(512);
    let (_, prompt) = npy_i32("prompt_ids.npy");
    let t = prompt.len();
    let v = e.cfg.vocab_size as usize;
    let (shape, want) = npy_f32("expect_prefill_logits.npy");
    assert_eq!(shape, vec![t, v]);
    let s = &e.s;
    let ids = Tensor::from_vec(s, prompt.clone(), &[t]).unwrap();
    let pos = Tensor::from_vec(s, (0..t as i32).collect(), &[t]).unwrap();
    let slots = Tensor::from_vec(s, (0..t as i64).collect(), &[t]).unwrap();
    let cu = Tensor::from_vec(s, vec![0i32, t as i32], &[2]).unwrap();
    let bt = Tensor::from_vec(s, (0..NUM_BLOCKS as i32).collect(), &[1, NUM_BLOCKS]).unwrap();
    let seq = Tensor::from_vec(s, vec![t as i32], &[1]).unwrap();
    let rows = Tensor::from_vec(s, (0..t as i32).collect(), &[t]).unwrap();
    let mut logits = Tensor::<f32>::zeros(s, &[t, v]).unwrap();
    let f = Forward {
        input_ids: &ids,
        positions: &pos,
        slot_mapping: &slots,
        meta: AttnMeta {
            mode: BatchMode::Prefill,
            cu_seqlens: &cu,
            seq_lens: &seq,
            block_table: &bt,
            num_reqs: 1,
            max_seqlen: prompt.len(),
            max_blocks: NUM_BLOCKS,
        },
        logits_rows: &rows,
        t,
        rows: t,
    };
    let t0 = std::time::Instant::now();
    e.model.forward(s, &e.gemm, &f, &mut logits).unwrap();
    let got = logits.to_vec(s).unwrap();
    eprintln!("q1: prefill {t} tokens in {:.1} ms", t0.elapsed().as_secs_f32() * 1e3);
    let (mut worst, mut bad, mut ties) = (0f32, 0, 0);
    for r in 0..t {
        let (g, w) = (&got[r * v..(r + 1) * v], &want[r * v..(r + 1) * v]);
        let mx = max_abs(g, w);
        worst = worst.max(mx);
        let gap = top_gap(w);
        if argmax(g) != argmax(w) {
            if gap > TIE_GAP {
                bad += 1;
                eprintln!(
                    "q1: row {r}: argmax {} vs HF {} (gap {gap:.4}, max abs {mx:.4})",
                    argmax(g),
                    argmax(w)
                );
            } else {
                ties += 1;
            }
        }
    }
    eprintln!("q1: prefill logits vs HF: worst max abs {worst:.4}, argmax mismatches {bad} (near-tie rows {ties}) over {t} rows");
    assert!(worst < MAX_ABS, "prefill logits max abs {worst} > {MAX_ABS}");
    assert_eq!(bad, 0, "argmax differs on a row where HF top-2 gap > {TIE_GAP}");
    assert!((t - ties) * 5 >= t * 4, "too many near-tie argmax flips: {ties}/{t}");
}

/// Scheduler + Runner + Executor 跑 n 条请求，返回每条的输出 token。
fn run_engine(e: Env, prompts: &[Vec<i32>], gen: u32, budget: u32) -> Vec<Vec<u32>> {
    let exec = Qwen3Executor::new(e.s.clone(), e.gemm, e.model).unwrap();
    let mut runner = Runner::new(exec, BLOCK);
    let params = SchedulerParams {
        sched: SchedulerConfig {
            max_num_seqs: 8,
            max_num_batched_tokens: budget,
            enable_chunked_prefill: true,
            long_prefill_token_threshold: 0,
        },
        cache: CacheConfig { block_size: BLOCK as u32, enable_prefix_caching: false, ..Default::default() },
        max_model_len: 1024,
        num_spec_tokens: 0,
        async_scheduling: false,
        eos_token_ids: vec![], // 对拍：不提前停
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
    let t0 = std::time::Instant::now();
    while sched.has_requests() {
        let so = sched.schedule();
        let mro = runner.execute(&so).unwrap();
        let res = sched.update_from_output(&so, &mro);
        for (rid, toks, _) in res.iter() {
            out[rid as usize - 1].extend_from_slice(toks);
        }
        steps += 1;
        assert!(steps < 10 * gen as usize + 20, "runaway");
    }
    eprintln!(
        "q1: engine {steps} steps for {} prompts × {gen} tokens in {:.2}s",
        prompts.len(),
        t0.elapsed().as_secs_f32()
    );
    out
}

/// 与 HF greedy 比：逐 token 相等，直到第一个 HF 平局步（top-2 gap ≤ TIE_GAP）。返回 (比较步数, 是否全对)。
fn compare_greedy(tag: &str, got: &[u32], want: &[i32], step_logits: &[f32], v: usize) -> usize {
    let mut n = 0;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let gap = top_gap(&step_logits[i * v..(i + 1) * v]);
        if g != w as u32 {
            assert!(gap <= TIE_GAP, "{tag}: step {i} token {g} != HF {w} with gap {gap:.4} (not a tie)");
            eprintln!("q1: {tag}: diverged at step {i} on an HF near-tie (gap {gap:.4}); compared {n} steps");
            return n;
        }
        n += 1;
    }
    eprintln!("q1: {tag}: all {n} greedy tokens match HF");
    n
}

#[test]
#[ignore]
fn real_greedy_matches_hf() {
    let e = load(512);
    let v = e.cfg.vocab_size as usize;
    let (_, prompt) = npy_i32("prompt_ids.npy");
    let (_, want) = npy_i32("expect_greedy.npy");
    let (_, step_logits) = npy_f32("expect_step_logits.npy");
    let out = run_engine(e, &[prompt], want.len() as u32, 512);
    let n = compare_greedy("prompt", &out[0], &want, &step_logits, v);
    assert!(n >= 8, "too few comparable greedy steps: {n}");
}

#[test]
#[ignore]
fn real_two_requests_concurrent() {
    let e = load(512);
    let v = e.cfg.vocab_size as usize;
    let (_, p1) = npy_i32("prompt_ids.npy");
    let (_, p2) = npy_i32("prompt2_ids.npy");
    let (_, w1) = npy_i32("expect_greedy.npy");
    let (_, w2) = npy_i32("expect_greedy2.npy");
    let (_, l1) = npy_f32("expect_step_logits.npy");
    let (_, l2) = npy_f32("expect_step_logits2.npy");
    // 预算 16 < prompt2 27 → chunked prefill + 混合 batch
    let out = run_engine(e, &[p1, p2], w1.len() as u32, 16);
    let n1 = compare_greedy("prompt (concurrent, chunked)", &out[0], &w1, &l1, v);
    let n2 = compare_greedy("prompt2 (no_think, concurrent)", &out[1], &w2, &l2, v);
    assert!(n1 >= 8 && n2 >= 8, "too few comparable steps: {n1} / {n2}");
}
