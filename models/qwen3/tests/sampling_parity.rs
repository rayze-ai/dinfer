//! Q2-a：Qwen3-0.6B 真权重上的采样语义验证（对照 vLLM `Sampler.forward` 顺序，host 侧用同公式算参考）。
//!
//! 1. `seed_deterministic_and_batch_independent`：同 seed 两次运行逐 token 相同；与另一个请求混批时也不变。
//! 2. `greedy_unchanged_in_mixed_batch`：temperature=0 的请求与随机请求混批，输出仍与 HF greedy golden 一致。
//! 3. `mask_and_distribution_match_host_reference`：100 个同 prompt 不同 seed 的请求一次 prefill（位置选 greedy 24 步里 mask 后候选最多的一步）；每个采样 token 必须落在
//!    host 侧按 temperature/top_k/top_p 算出的候选集合里；频率与 host softmax 的 L1 < 0.35（100 样本的统计噪声上界）。
//! 4. `penalties_match_host_formula`：repetition / frequency / presence 惩罚后 greedy argmax == host 侧对原始 logits 套
//!    vLLM 公式后的 argmax（逐步，用 executor 原始 logits）。
//! 5. `logprobs_match_host_log_softmax`：logprobs=Some(3) 的 chosen logprob / rank / top-3 与 host log_softmax 一致。
//!
//! 跑法：QWEN3_DIR=/data/ref/models/qwen3-0.6b QWEN3_GOLDEN=$(realpath test/qwen3_golden) \
//!   cargo test -p dinfer-qwen3 --features ref-tests --test sampling_parity -- --ignored --nocapture --test-threads=1
#![cfg(feature = "ref-tests")]

use dinfer_core::config::{CacheConfig, SchedulerConfig};
use dinfer_core::kv::{GroupInfo, KvCacheSpec};
use dinfer_core::request::Request;
use dinfer_core::runner::Runner;
use dinfer_core::sample::SamplingParams;
use dinfer_core::sampler::TokenLogprobs;
use dinfer_core::sched::{Scheduler, SchedulerParams};
use dinfer_layers::{Device, Gemm, WeightSource};
use dinfer_qwen3::config::Qwen3Config;
use dinfer_qwen3::executor::Qwen3Executor;
use dinfer_qwen3::model::Qwen3Model;
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
fn npy_i32(name: &str) -> Vec<i32> {
    let b = std::fs::read(golden_dir().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    npyz::NpyFile::new(&b[..]).unwrap().into_vec::<i32>().unwrap()
}
fn npy_f32(name: &str) -> Vec<f32> {
    let b = std::fs::read(golden_dir().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    npyz::NpyFile::new(&b[..]).unwrap().into_vec::<f32>().unwrap()
}
fn prompt() -> Vec<u32> {
    npy_i32("prompt_ids.npy").into_iter().map(|x| x as u32).collect()
}

const BLOCK: usize = 16;
const NUM_BLOCKS: usize = 512;
const TIE_GAP: f32 = 1.0;

struct Engine {
    runner: Runner<Qwen3Executor>,
    sched: Scheduler,
    vocab: usize,
}

fn engine(max_tokens: usize, max_seqs: u32, budget: u32) -> Engine {
    let dev = Device::new(0).unwrap();
    let s = dev.default_stream();
    let gemm = Gemm::new(&s).unwrap();
    let cfg =
        Qwen3Config::from_json(&std::fs::read_to_string(model_dir().join("config.json")).unwrap()).unwrap();
    let src = WeightSource::open_dir(&model_dir()).unwrap();
    // logits 只取每请求最后一行：行数上限 = max_seqs（按 max_tokens=8192 分 fp32 logits ×3 份会 OOM）
    let model = Qwen3Model::load_with_budget(
        &s,
        &src,
        cfg.clone(),
        max_tokens,
        max_seqs as usize,
        Some(NUM_BLOCKS),
        BLOCK,
        0.9,
        512 << 20,
    )
    .unwrap();
    let exec = Qwen3Executor::new(s, gemm, model).unwrap();
    let runner = Runner::new(exec, BLOCK);
    let params = SchedulerParams {
        sched: SchedulerConfig {
            max_num_seqs: max_seqs,
            max_num_batched_tokens: budget,
            enable_chunked_prefill: true,
            long_prefill_token_threshold: 0,
        decode_exempt_from_budget: false,
        },
        cache: CacheConfig { block_size: BLOCK as u32, enable_prefix_caching: false, ..Default::default() },
        max_model_len: 1024,
        num_spec_tokens: 0,
        async_scheduling: false,
        eos_token_ids: vec![],
        recurrent_snapshot_tokens: 0,
        recurrent_snapshot_slots: 0,
    };
    let g = vec![GroupInfo {
        spec: KvCacheSpec::FullAttention {
            block_size: BLOCK as u32,
            kv_heads: cfg.num_key_value_heads,
            head_size: cfg.head_dim,
            dtype_bytes: 2,
        },
        block_size: BLOCK as u32,
    }];
    let sched = Scheduler::new(params, g, NUM_BLOCKS as u32);
    Engine { runner, sched, vocab: cfg.vocab_size as usize }
}

struct Out {
    tokens: Vec<Vec<u32>>,
    logprobs: Vec<Vec<TokenLogprobs>>,
    /// 每 step 的 (req_ids, 原始 logits 行) —— 只在 `keep_logits` 时记录。
    step_logits: Vec<(Vec<u64>, Vec<f32>)>,
}

/// 跑到所有请求结束。请求 id = 下标 + 1。
fn run(e: &mut Engine, reqs: &[(Vec<u32>, SamplingParams)], keep_logits: bool) -> Out {
    for (i, (p, sp)) in reqs.iter().enumerate() {
        e.sched.add_request(Request::new(i as u64 + 1, p.clone(), sp.clone(), 0));
    }
    let mut out = Out {
        tokens: vec![Vec::new(); reqs.len()],
        logprobs: vec![Vec::new(); reqs.len()],
        step_logits: vec![],
    };
    let mut steps = 0;
    while e.sched.has_requests() {
        let so = e.sched.schedule();
        let mro = e.runner.execute(&so).unwrap();
        if keep_logits {
            let n = mro.req_ids.len();
            out.step_logits.push((mro.req_ids.clone(), e.runner.exec.last_logits(n).unwrap()));
        }
        let res = e.sched.update_from_output(&so, &mro);
        for (i, (rid, toks, _)) in res.iter().enumerate() {
            out.tokens[rid as usize - 1].extend_from_slice(toks);
            out.logprobs[rid as usize - 1].extend(res.logprobs_of(i).iter().flatten().cloned());
        }
        steps += 1;
        assert!(steps < 4000, "runaway");
    }
    out
}

fn softmax(row: &[f32]) -> Vec<f32> {
    let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = row.iter().map(|&x| if x == f32::NEG_INFINITY { 0.0 } else { (x - m).exp() }).collect();
    let z: f32 = e.iter().sum();
    e.iter().map(|x| x / z).collect()
}
fn argmax(row: &[f32]) -> usize {
    (1..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b })
}
fn top_gap(row: &[f32]) -> f32 {
    let mut s = row.to_vec();
    s.sort_by(|a, b| b.partial_cmp(a).unwrap());
    s[0] - s[1]
}

/// vLLM `apply_top_k_top_p` host 参考（与 gen_kernel_refs.py::mask_ref 同口径）：返回 masked 后的 logits。
fn host_mask(row: &[f32], temperature: f32, top_k: i32, top_p: f32) -> Vec<f32> {
    let mut r: Vec<f32> = row.iter().map(|x| x / temperature).collect();
    if top_k > 0 && (top_k as usize) < r.len() {
        let mut s = r.clone();
        s.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let th = s[top_k as usize - 1];
        for x in r.iter_mut() {
            if *x < th {
                *x = f32::NEG_INFINITY;
            }
        }
    }
    if top_p > 0.0 && top_p < 1.0 {
        let mut idx: Vec<usize> = (0..r.len()).collect();
        idx.sort_by(|&a, &b| r[a].partial_cmp(&r[b]).unwrap()); // 升序
        let p = softmax(&r);
        let mut cum = 0.0f32;
        for (j, &i) in idx.iter().enumerate() {
            cum += p[i];
            if cum <= 1.0 - top_p && j + 1 < idx.len() {
                r[i] = f32::NEG_INFINITY;
            }
        }
    }
    r
}

/// vLLM `apply_penalties` host 参考。
fn host_penalties(row: &[f32], prompt: &[u32], output: &[u32], rep: f32, freq: f32, pres: f32) -> Vec<f32> {
    let mut r = row.to_vec();
    let v = r.len();
    let mut in_prompt = vec![false; v];
    let mut cnt = vec![0i32; v];
    for &t in prompt {
        in_prompt[t as usize] = true;
    }
    for &t in output {
        cnt[t as usize] += 1;
    }
    for i in 0..v {
        if in_prompt[i] || cnt[i] > 0 {
            r[i] = if r[i] > 0.0 { r[i] / rep } else { r[i] * rep };
        }
        r[i] -= freq * cnt[i] as f32;
        if cnt[i] > 0 {
            r[i] -= pres;
        }
    }
    r
}

#[test]
#[ignore]
fn seed_deterministic_and_batch_independent() {
    let mut e = engine(512, 8, 512);
    let p = prompt();
    let sp = SamplingParams {
        max_tokens: 24,
        temperature: 0.8,
        top_p: 0.95,
        top_k: 40,
        seed: Some(1234),
        ..Default::default()
    };
    let a = run(&mut e, &[(p.clone(), sp.clone())], false);
    let b = run(&mut e, &[(p.clone(), sp.clone())], false);
    assert_eq!(a.tokens[0], b.tokens[0], "same seed → same tokens");
    // 与另一个（不同 seed 的）请求混批，第二个槽位的输出必须不变
    let other = SamplingParams { seed: Some(99), ..sp.clone() };
    let c = run(&mut e, &[(p.clone(), other), (p.clone(), sp.clone())], false);
    assert_eq!(c.tokens[1], a.tokens[0], "batch position must not change the sample");
    assert_ne!(c.tokens[0], a.tokens[0], "different seed should (almost surely) differ");
    let d = run(&mut e, &[(p, SamplingParams { seed: None, ..sp })], false);
    eprintln!("q2: seeded 24 tokens = {:?}\nq2: unseeded = {:?}", a.tokens[0], d.tokens[0]);
}

#[test]
#[ignore]
fn greedy_unchanged_in_mixed_batch() {
    let mut e = engine(512, 8, 512);
    let p = prompt();
    let want = npy_i32("expect_greedy.npy");
    let step_logits = npy_f32("expect_step_logits.npy");
    let v = e.vocab;
    let greedy = SamplingParams { max_tokens: want.len() as u32, temperature: 0.0, ..Default::default() };
    let random = SamplingParams {
        max_tokens: want.len() as u32,
        temperature: 1.2,
        top_k: 10,
        seed: Some(7),
        ..Default::default()
    };
    let o = run(&mut e, &[(p.clone(), random), (p, greedy)], false);
    let mut n = 0;
    for (i, (&g, &w)) in o.tokens[1].iter().zip(&want).enumerate() {
        if g != w as u32 {
            let gap = top_gap(&step_logits[i * v..(i + 1) * v]);
            assert!(gap <= TIE_GAP, "greedy step {i}: {g} != HF {w}, gap {gap:.3}");
            break;
        }
        n += 1;
    }
    eprintln!("q2: greedy-in-mixed-batch matched HF for {n} steps");
    assert!(n >= 8);
}

#[test]
#[ignore]
fn mask_and_distribution_match_host_reference() {
    // N × (13 + ≤24) token prompt 要一次 prefill 放下：预算 8192、每请求 ≤ 3 块（512 块池）
    const N: usize = 100;
    let mut e = engine(8192, N as u32, 8192);
    let p = prompt();
    let v = e.vocab;
    let (t, k, tp) = (0.7f32, 20i32, 0.9f32);
    // 第一个 token（`<think>`）几乎确定（logprob ≈ -2e-4），mask 后只剩 1 个候选 → 先 greedy 跑 24 步，
    // 选 mask 后候选最多的一步，用 prompt + greedy[..step] 当采样 prompt。
    let pre = run(
        &mut e,
        &[(p.clone(), SamplingParams { max_tokens: 24, temperature: 0.0, ..Default::default() })],
        true,
    );
    let (step, _) = pre
        .step_logits
        .iter()
        .enumerate()
        .map(|(i, (_, l))| (i, host_mask(&l[..v], t, k, tp).iter().filter(|x| x.is_finite()).count()))
        .max_by_key(|&(_, c)| c)
        .unwrap();
    let mut p = p;
    p.extend_from_slice(&pre.tokens[0][..step]);
    eprintln!("q2: sampling at greedy step {step} (prompt {} tokens)", p.len());
    let reqs: Vec<_> = (0..N)
        .map(|i| {
            (
                p.clone(),
                SamplingParams {
                    max_tokens: 1,
                    temperature: t,
                    top_k: k,
                    top_p: tp,
                    seed: Some(1000 + i as u64),
                    ..Default::default()
                },
            )
        })
        .collect();
    let o = run(&mut e, &reqs, true);
    let (ids, logits) = &o.step_logits[0];
    assert_eq!(ids.len(), N, "expected all {N} requests in one prefill step");
    // 所有行同 prompt → 原始 logits 同一行（取第 0 行做参考）
    let raw = &logits[..v];
    let masked = host_mask(raw, t, k, tp);
    let probs = softmax(&masked);
    let allowed: Vec<usize> = (0..v).filter(|&i| masked[i] > f32::NEG_INFINITY).collect();
    eprintln!("q2: host candidate set size {} (top_k {k}, top_p {tp})", allowed.len());
    let mut counts = vec![0usize; v];
    for tok in o.tokens.iter().map(|t| t[0] as usize) {
        assert!(masked[tok] > f32::NEG_INFINITY, "sampled token {tok} outside host top-k/top-p set");
        counts[tok] += 1;
    }
    let l1: f32 = (0..v).map(|i| (counts[i] as f32 / N as f32 - probs[i]).abs()).sum();
    let distinct = counts.iter().filter(|&&c| c > 0).count();
    eprintln!("q2: {N} samples, {distinct} distinct tokens, L1 vs host softmax = {l1:.3}");
    assert!(distinct >= 2, "sampling collapsed to a single token");
    assert!(l1 < 0.35, "frequency vs softmax L1 {l1:.3} too large");
}

#[test]
#[ignore]
fn penalties_match_host_formula() {
    let mut e = engine(512, 8, 512);
    let p = prompt();
    let v = e.vocab;
    let cases = [(1.5f32, 0.0f32, 0.0f32), (1.0, 0.7, 0.0), (1.0, 0.0, 1.2), (1.3, 0.4, 0.5)];
    for (rep, freq, pres) in cases {
        let sp = SamplingParams {
            max_tokens: 16,
            temperature: 0.0,
            repetition_penalty: rep,
            frequency_penalty: freq,
            presence_penalty: pres,
            ..Default::default()
        };
        let o = run(&mut e, &[(p.clone(), sp)], true);
        let toks = &o.tokens[0];
        assert_eq!(o.step_logits.len(), toks.len());
        let mut mismatches = 0;
        for (step, (_, logits)) in o.step_logits.iter().enumerate() {
            let raw = &logits[..v];
            let want = host_penalties(raw, &p, &toks[..step], rep, freq, pres);
            let am = argmax(&want);
            if am != toks[step] as usize {
                // 允许 fp32 平局（惩罚后两个候选差 < 1e-4）
                let mut s = want.clone();
                s.sort_by(|a, b| b.partial_cmp(a).unwrap());
                assert!(
                    s[0] - s[1] < 1e-4,
                    "rep {rep} freq {freq} pres {pres}: step {step} got {} want {am} (gap {:.5})",
                    toks[step],
                    s[0] - s[1]
                );
                mismatches += 1;
            }
        }
        eprintln!(
            "q2: penalties ({rep},{freq},{pres}) 16 greedy steps match host formula ({mismatches} fp32 ties)"
        );
    }
    // 统计口径：rep 1.5 下 16 步的重复率不高于不加惩罚
    let plain = run(
        &mut e,
        &[(p.clone(), SamplingParams { max_tokens: 16, temperature: 0.0, ..Default::default() })],
        false,
    );
    let pen = run(
        &mut e,
        &[(
            p,
            SamplingParams {
                max_tokens: 16,
                temperature: 0.0,
                repetition_penalty: 1.5,
                ..Default::default()
            },
        )],
        false,
    );
    let dup = |t: &[u32]| t.len() - t.iter().collect::<std::collections::HashSet<_>>().len();
    eprintln!("q2: duplicates plain {} vs rep1.5 {}", dup(&plain.tokens[0]), dup(&pen.tokens[0]));
    assert!(dup(&pen.tokens[0]) <= dup(&plain.tokens[0]));
}

#[test]
#[ignore]
fn logprobs_match_host_log_softmax() {
    let mut e = engine(512, 8, 512);
    let p = prompt();
    let v = e.vocab;
    let sp = SamplingParams {
        max_tokens: 8,
        temperature: 0.6,
        top_k: 30,
        seed: Some(5),
        logprobs: Some(3),
        ..Default::default()
    };
    let o = run(&mut e, &[(p, sp)], true);
    assert_eq!(o.logprobs[0].len(), 8, "one logprob entry per token");
    for (step, lp) in o.logprobs[0].iter().enumerate() {
        let raw = &o.step_logits[step].1[..v];
        let m = raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lse = m + raw.iter().map(|x| (x - m).exp()).sum::<f32>().ln();
        let want = raw[lp.token_id as usize] - lse;
        assert_eq!(lp.token_id, o.tokens[0][step]);
        assert!((lp.logprob - want).abs() < 1e-3, "step {step}: logprob {} vs host {want}", lp.logprob);
        let rank = raw.iter().filter(|&&x| x > raw[lp.token_id as usize]).count() as u32;
        assert_eq!(lp.rank, rank, "step {step}: rank");
        assert_eq!(lp.top.len(), 3);
        let mut idx: Vec<usize> = (0..v).collect();
        idx.sort_by(|&a, &b| raw[b].partial_cmp(&raw[a]).unwrap());
        for (j, (id, l)) in lp.top.iter().enumerate() {
            assert!(
                *id as usize == idx[j] || (raw[*id as usize] - raw[idx[j]]).abs() < 1e-5,
                "step {step} top{j}: {id} vs {}",
                idx[j]
            );
            assert!((l - (raw[*id as usize] - lse)).abs() < 1e-3);
        }
    }
    eprintln!(
        "q2: logprobs (chosen / rank / top-3) match host log_softmax on 8 steps; first = {:?}",
        o.logprobs[0][0]
    );
}
