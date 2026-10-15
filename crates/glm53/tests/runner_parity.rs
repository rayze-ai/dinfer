//! Scheduler + Runner + Glm53Executor 串起来跑两个并发请求（prompt 长 14 与 11，块 16 不对齐）：
//! 一个 step 同时 prefill 两个请求，之后每 step 同时 decode 两个请求，直到 max_tokens 结束。
//! 每步两个请求的 logits 都与 HF 单请求全序列前向对拍（喂 HF 的 greedy token，避免误差累积）。
//!
//! 先 `python3 scripts/verify/gen_tiny_glm.py test/tiny_glm`（或 `--first-k-dense 1` 到 test/tiny_glm_moe）。
#![cfg(feature = "ref-tests")]

use dinfer_core::config::{CacheConfig, SchedulerConfig};
use dinfer_core::kv::{GroupInfo, KvCacheSpec};
use dinfer_core::request::{FinishReason, Request};
use dinfer_core::runner::Runner;
use dinfer_core::sample::SamplingParams;
use dinfer_core::sched::{Scheduler, SchedulerParams};
use dinfer_glm53::config::GlmMoeDsaConfig;
use dinfer_glm53::executor::Glm53Executor;
use dinfer_glm53::model::Glm53Model;
use dinfer_layers::{Device, Gemm, WeightSource};
use std::path::PathBuf;

fn dir() -> PathBuf {
    let p = PathBuf::from(std::env::var("TINY_GLM_DIR").unwrap_or_else(|_| "test/tiny_glm".into()));
    if p.is_absolute() {
        p
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join(p)
    }
}
fn npy_f32(name: &str) -> (Vec<usize>, Vec<f32>) {
    let b = std::fs::read(dir().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    let a = npyz::NpyFile::new(&b[..]).unwrap();
    (a.shape().iter().map(|&d| d as usize).collect(), a.into_vec::<f32>().unwrap())
}
fn npy_i32(name: &str) -> Vec<i32> {
    let b = std::fs::read(dir().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    npyz::NpyFile::new(&b[..]).unwrap().into_vec::<i32>().unwrap()
}
/// 相等取小下标（与 torch.argmax / GPU argmax kernel 一致）。
fn argmax(row: &[f32]) -> i32 {
    let mut best = 0usize;
    for i in 1..row.len() {
        if row[i] > row[best] {
            best = i;
        }
    }
    best as i32
}
fn top_gap(row: &[f32]) -> f32 {
    let mut s = row.to_vec();
    s.sort_by(|a, b| b.partial_cmp(a).unwrap());
    s[0] - s[1]
}

const BLOCK: usize = 16;
const MAX_ABS: f32 = 0.05;
const TIE_GAP: f32 = 0.02;
const GEN: u32 = 32;

/// 两请求同时进入、预算充足：1 步双 prefill（展开 K/V 路径）+ 31 步双 decode（吸收路径）。
#[test]
#[ignore]
fn runner_two_requests_match_hf() {
    run(1024, false, 0, GEN as usize);
}

/// 预算 8（< 两个 prompt 长 14 / 11）、开 chunked prefill：req1 分 14 = 8 + 6（第二块带历史 → Paged 路径），
/// 之后与 req2 的 chunk / decode 混合 → 覆盖 `mla_prefill_paged` 的三种情形（纯 chunk、chunk+decode 混合、全 decode）。
#[test]
#[ignore]
fn runner_chunked_mixed_match_hf() {
    // 总 token = 14 + 11 + 2×31 = 87；每步最多 8 → 步数 ≥ 11；具体步数由调度器决定，不做精确断言
    run(8, true, 0, 0);
}

fn run(budget: u32, chunked: bool, _unused: u32, expect_steps: usize) {
    let dev = Device::new(0).unwrap();
    let s = dev.default_stream();
    let gemm = Gemm::new(&s).unwrap();
    let cfg =
        GlmMoeDsaConfig::from_json(&std::fs::read_to_string(dir().join("config.json")).unwrap()).unwrap();
    let mut src = WeightSource::open_dir(&dir()).unwrap();
    if let Some((bn, bk)) = cfg.fp8_block_size() {
        src = src.with_quant_block(bn, bk);
    }
    let num_blocks = 64;
    let model = Glm53Model::load(&s, &src, cfg.clone(), 128, num_blocks, BLOCK).unwrap();
    let exec = Glm53Executor::new(s.clone(), gemm, model, None).unwrap();
    let mut runner = Runner::new(exec, BLOCK);

    let params = SchedulerParams {
        sched: SchedulerConfig {
            max_num_seqs: 8,
            max_num_batched_tokens: budget,
            enable_chunked_prefill: chunked,
            long_prefill_token_threshold: 0,
        },
        cache: CacheConfig { block_size: BLOCK as u32, enable_prefix_caching: false, ..Default::default() },
        max_model_len: 256,
        num_spec_tokens: 0,
        async_scheduling: false,
        eos_token_ids: vec![], // tiny 模型随机权重，不用 eos
    };
    let g = vec![GroupInfo {
        spec: KvCacheSpec::Mla { block_size: BLOCK as u32, head_size: 48, dtype_bytes: 2 },
        block_size: BLOCK as u32,
    }];
    let mut sched = Scheduler::new(params, g, num_blocks as u32);

    let prompts = [npy_i32("prompt.npy"), npy_i32("prompt2.npy")];
    let want_tok = [npy_i32("expect_greedy.npy"), npy_i32("expect_greedy2.npy")];
    let (shape, l1) = npy_f32("expect_step_logits.npy");
    let (_, l2) = npy_f32("expect_step_logits2.npy");
    let v = shape[1];
    let want_logits = [l1, l2];
    let sp = SamplingParams { max_tokens: GEN, temperature: 0.0, ..Default::default() };
    for (i, p) in prompts.iter().enumerate() {
        sched.add_request(Request::new(i as u64 + 1, p.iter().map(|&x| x as u32).collect(), sp.clone(), 0));
    }

    // 每请求：已对拍的步数；一旦某步 HF 自身平局（gap ≤ TIE_GAP）且我们选了另一个 token，之后序列不可比 → 停止该请求的对拍。
    let mut compared = [0usize; 2];
    let mut comparable = [true; 2];
    let mut worst = 0f32;
    let mut steps = 0;
    let mut modes = std::collections::BTreeMap::<String, usize>::new();
    let mut finished = 0usize;
    while sched.has_requests() {
        let so = sched.schedule();
        finished += so.finished_req_ids.len();
        let mro = runner.execute(&so).unwrap();
        steps += 1;
        *modes.entry(format!("{:?}", runner.last_plan().mode)).or_default() += 1;
        let n = so.num_scheduled_tokens.len();
        let logits = runner.exec.last_logits(n).unwrap();
        for (bi, &(rid, _)) in so.num_scheduled_tokens.iter().enumerate() {
            let r = rid as usize - 1;
            if !comparable[r] || mro.sampled[bi].is_empty() {
                continue; // 不可比，或 prefill 中间 chunk（无采样）
            }
            let step = compared[r];
            let got = &logits[bi * v..(bi + 1) * v];
            let w = &want_logits[r][step * v..(step + 1) * v];
            let mx = got.iter().zip(w).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            worst = worst.max(mx);
            let (ga, wa) = (argmax(got), argmax(w));
            let gap = top_gap(w);
            eprintln!("step {steps:2} req {rid} out#{step:2}: max abs {mx:.4} argmax got {ga} want {wa} gap {gap:.4}");
            assert!(mx < MAX_ABS, "req {rid} step {step}: logits max abs diff {mx} > {MAX_ABS}");
            if mro.sampled[bi] != [ga as u32] {
                let st = mro.sampled[bi][0] as usize;
                let (a, b) = (got[st], got[ga as usize]);
                eprintln!("  GPU argmax {st} ({a}) vs host argmax {ga} ({b}) diff {}", a - b);
                assert_eq!(a, b, "sampled token must be an argmax (tie allowed)");
            }
            if ga != wa {
                assert!(gap <= TIE_GAP, "req {rid} step {step}: argmax differs with clear HF gap {gap:.4}");
                comparable[r] = false; // 平局分叉，后续不可比
            } else {
                assert_eq!(ga, want_tok[r][step], "greedy token mismatch");
            }
            compared[r] += 1;
        }
        let res = sched.update_from_output(&so, &mro);
        for (rid, _, fin) in res.iter() {
            if let Some(f) = fin {
                assert_eq!(f, FinishReason::Length, "req {rid} finished with {f:?}");
            }
        }
    }
    eprintln!(
        "steps = {steps}, modes = {modes:?}, compared = {compared:?}, comparable_to_end = {comparable:?}, worst = {worst}"
    );
    if expect_steps > 0 {
        assert_eq!(steps, expect_steps, "1 prefill step + {} decode steps", GEN - 1);
    }
    if chunked {
        assert!(modes.contains_key("Paged"), "chunked run must exercise the Paged path: {modes:?}");
    }
    // 完成的请求在下一次 schedule 的 finished_req_ids 里才传给 worker（两请求可能不同步结束）
    let so = sched.schedule();
    finished += so.finished_req_ids.len();
    assert_eq!(finished, 2);
    runner.execute(&so).unwrap();
    assert_eq!(runner.num_states(), 0, "runner states must be cleaned after finish");
    // 两个请求都应至少对拍到相当多步（平局分叉是允许的，但不能一开始就分叉）
    assert!(compared.iter().all(|&c| c >= 8), "compared too few steps: {compared:?}");
}
