//! P6-d：MTP（nextn=1）对拍。fixture：`gen_tiny_glm.py --dsa --mtp`（HF decoder 层 + 自写胶水当参考）。
//!
//! 1. `mtp_draft_logits_match_hf`：主干 prefill 长 prompt（注入 HF top-k 隔离索引器平局）→ MTP 对右移一位的输入
//!    跑 prefill → draft logits vs `mtp_draft_logits_long.npy`（[T-1, V]）。断言只在主干 hidden 未漂移的行上做。
//! 2. `mtp_spec_decode_matches_plain_greedy`：Scheduler+Runner+Executor 开 MTP（num_spec_tokens=1）与不开，greedy
//!    输出 token 序列必须 **完全一致**（投机解码不改变分布；greedy 下逐 token相等），且上报接受率。
//!
//! 跑法：TINY_GLM_DIR=$(realpath test/tiny_glm_dsa_mtp) cargo test -p dinfer-glm53 --features ref-tests --test mtp_parity -- --ignored --nocapture --test-threads=1
#![cfg(feature = "ref-tests")]

use dinfer_core::config::{CacheConfig, SchedulerConfig};
use dinfer_core::kv::{GroupInfo, KvCacheSpec};
use dinfer_core::request::Request;
use dinfer_core::runner::Runner;
use dinfer_core::sample::SamplingParams;
use dinfer_core::sched::{Scheduler, SchedulerParams};
use dinfer_glm53::config::GlmMoeDsaConfig;
use dinfer_glm53::executor::Glm53Executor;
use dinfer_glm53::model::{Forward, Glm53Model};
use dinfer_glm53::mtp::MtpLayer;
use dinfer_layers::mla::AttnMeta;
use dinfer_layers::{Device, Gemm, Tensor, WeightSource};
use std::path::PathBuf;

fn dir() -> PathBuf {
    let p = PathBuf::from(std::env::var("TINY_GLM_DIR").unwrap_or_else(|_| "test/tiny_glm_dsa_mtp".into()));
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
fn npy_i32(name: &str) -> (Vec<usize>, Vec<i32>) {
    let b = std::fs::read(dir().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    let a = npyz::NpyFile::new(&b[..]).unwrap();
    (a.shape().iter().map(|&d| d as usize).collect(), a.into_vec::<i32>().unwrap())
}
fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}
fn argmax(row: &[f32]) -> usize {
    (1..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b })
}
fn top_gap(row: &[f32]) -> f32 {
    let mut s = row.to_vec();
    s.sort_by(|a, b| b.partial_cmp(a).unwrap());
    s[0] - s[1]
}

const BLOCK: usize = 16;
const NUM_BLOCKS: usize = 64;
const MAX_ABS: f32 = 0.05;
const TIE_GAP: f32 = 0.02;
const DRIFT: f32 = 0.02;

#[test]
#[ignore]
fn mtp_draft_logits_match_hf() {
    let dev = Device::new(0).unwrap();
    let s = dev.default_stream();
    let gemm = Gemm::new(&s).unwrap();
    let cfg =
        GlmMoeDsaConfig::from_json(&std::fs::read_to_string(dir().join("config.json")).unwrap()).unwrap();
    assert_eq!(cfg.num_nextn_predict_layers, 1, "fixture must have an MTP layer");
    let src = WeightSource::open_dir(&dir()).unwrap();
    let mut model = Glm53Model::load(&s, &src, cfg.clone(), 512, NUM_BLOCKS, BLOCK).unwrap();
    let mut mtp = MtpLayer::load(&s, &src, &cfg, 512, NUM_BLOCKS, BLOCK).unwrap();
    let (_, prompt) = npy_i32("prompt_long.npy");
    let t = prompt.len();
    let v = cfg.vocab_size as usize;
    let h = cfg.hidden_size as usize;

    // 主干 prefill，注入 HF top-k（隔离索引器平局；MTP 的 shared 索引器层复用这份 top-k，与 HF 参考一致）
    let full: Vec<usize> =
        (0..cfg.num_hidden_layers).filter(|&l| cfg.has_indexer(l)).map(|l| l as usize).collect();
    model.topk_override =
        Some(full.iter().map(|&l| (l, npy_i32(&format!("indexer_topk_L{l}_long.npy")).1)).collect());
    model.capture_layers = Some(Vec::new());
    let ids = Tensor::from_vec(&s, prompt.clone(), &[t]).unwrap();
    let pos = Tensor::from_vec(&s, (0..t as i32).collect(), &[t]).unwrap();
    let slots = Tensor::from_vec(&s, (0..t as i64).collect(), &[t]).unwrap();
    let cu = Tensor::from_vec(&s, vec![0i32, t as i32], &[2]).unwrap();
    let bt = Tensor::from_vec(&s, (0..NUM_BLOCKS as i32).collect(), &[1, NUM_BLOCKS]).unwrap();
    let seq = Tensor::from_vec(&s, vec![t as i32], &[1]).unwrap();
    let rows_all = Tensor::from_vec(&s, (0..t as i32).collect(), &[t]).unwrap();
    let mut logits = Tensor::<f32>::zeros(&s, &[t, v]).unwrap();
    let meta = AttnMeta::Prefill {
        cu_seqlens: &cu,
        num_reqs: 1,
        block_table: &bt,
        seq_lens: &seq,
        max_blocks: NUM_BLOCKS,
    };
    let f = Forward { input_ids: &ids, positions: &pos, slot_mapping: &slots, meta, logits_rows: &rows_all };
    model.forward(&s, &gemm, &f, &mut logits).unwrap();
    model.topk_override = None;
    // 主干 hidden 漂移行（vs HF hidden_L{last}_long）：这些行的 MTP 输入已不同，不比
    let caps = model.capture_layers.take().unwrap();
    let (_, _, last_hidden) = caps.last().unwrap();
    let last = cfg.num_hidden_layers as usize - 1;
    let (_, hf_hidden) = npy_f32(&format!("hidden_L{last}_long.npy"));
    let drift: Vec<bool> = (0..t)
        .map(|r| max_abs(&last_hidden[r * h..(r + 1) * h], &hf_hidden[r * h..(r + 1) * h]) > DRIFT)
        .collect();

    // MTP：输入 = prompt 右移一位（第 i 行吃 prompt[i+1]，最后一行占位 0），previous_hidden = 主干最后一层输出
    let mut next = prompt[1..].to_vec();
    next.push(0);
    let next_t = Tensor::from_vec(&s, next, &[t]).unwrap();
    let prev = model.last_hidden();
    let prev_copy =
        Tensor::<dinfer_layers::bf16>::from_vec(&s, prev.to_vec(&s).unwrap()[..t * h].to_vec(), &[t, h])
            .unwrap();
    let mut dlogits = Tensor::<f32>::zeros(&s, &[t, v]).unwrap();
    let meta = AttnMeta::Prefill {
        cu_seqlens: &cu,
        num_reqs: 1,
        block_table: &bt,
        seq_lens: &seq,
        max_blocks: NUM_BLOCKS,
    };
    mtp.forward(&s, &gemm, &mut model, &next_t, &prev_copy, &pos, &slots, &meta, &rows_all, &mut dlogits)
        .unwrap();
    let got = dlogits.to_vec(&s).unwrap();
    let (shape, want) = npy_f32("mtp_draft_logits_long.npy");
    assert_eq!(shape, vec![t - 1, v]);

    let (mut worst, mut bad, mut n, mut skipped) = (0f32, 0, 0, 0);
    for r in 0..t - 1 {
        if drift[r] {
            skipped += 1;
            continue;
        }
        let g = &got[r * v..(r + 1) * v];
        let w = &want[r * v..(r + 1) * v];
        let mx = max_abs(g, w);
        worst = worst.max(mx);
        n += 1;
        if argmax(g) != argmax(w) && top_gap(w) > TIE_GAP {
            bad += 1;
            eprintln!(
                "mtp: row {r} argmax {} vs HF {} (gap {:.4}, max abs {mx:.4})",
                argmax(g),
                argmax(w),
                top_gap(w)
            );
        }
    }
    eprintln!("mtp: draft logits vs HF on {n}/{t} rows ({skipped} drifted skipped): worst max abs {worst:.4}, argmax mismatches {bad}");
    assert!(n * 2 >= t, "too few comparable rows");
    assert!(worst < MAX_ABS, "mtp draft logits max abs {worst} > {MAX_ABS}");
    assert_eq!(bad, 0);
}

/// 用 Scheduler + Runner + Executor 跑一条请求 `gen` 步 greedy，返回输出 token 与接受统计。
fn run_greedy(mtp: bool, prompt: &[i32], gen: u32) -> (Vec<u32>, usize, usize) {
    let dev = Device::new(0).unwrap();
    let s = dev.default_stream();
    let gemm = Gemm::new(&s).unwrap();
    let cfg =
        GlmMoeDsaConfig::from_json(&std::fs::read_to_string(dir().join("config.json")).unwrap()).unwrap();
    let src = WeightSource::open_dir(&dir()).unwrap();
    let model = Glm53Model::load(&s, &src, cfg.clone(), 256, NUM_BLOCKS, BLOCK).unwrap();
    let mtp_layer =
        if mtp { Some(MtpLayer::load(&s, &src, &cfg, 256, NUM_BLOCKS, BLOCK).unwrap()) } else { None };
    let exec = Glm53Executor::new(s.clone(), gemm, model, mtp_layer).unwrap();
    let mut runner = Runner::new(exec, BLOCK);
    let params = SchedulerParams {
        sched: SchedulerConfig {
            max_num_seqs: 4,
            max_num_batched_tokens: 256,
            enable_chunked_prefill: true,
            long_prefill_token_threshold: 0,
        },
        cache: CacheConfig { block_size: BLOCK as u32, enable_prefix_caching: false, ..Default::default() },
        max_model_len: 512,
        num_spec_tokens: if mtp { 1 } else { 0 },
        async_scheduling: false,
        eos_token_ids: vec![],
    };
    let entry = cfg.kv_lora_rank + cfg.qk_rope_head_dim;
    let g = vec![GroupInfo {
        spec: KvCacheSpec::Mla { block_size: BLOCK as u32, head_size: entry, dtype_bytes: 2 },
        block_size: BLOCK as u32,
    }];
    let mut sched = Scheduler::new(params, g, NUM_BLOCKS as u32);
    let sp = SamplingParams { max_tokens: gen, temperature: 0.0, ..Default::default() };
    sched.add_request(Request::new(1, prompt.iter().map(|&x| x as u32).collect(), sp, 0));
    let mut out = Vec::new();
    let (mut drafts_total, mut accepted_total) = (0usize, 0usize);
    let mut steps = 0;
    while sched.has_requests() {
        let so = sched.schedule();
        let mro = runner.execute(&so).unwrap();
        for (d, a) in &runner.exec.last_accept {
            drafts_total += d;
            accepted_total += a;
        }
        let res = sched.update_from_output(&so, &mro);
        if mro.drafts.iter().any(|d| !d.is_empty()) {
            let dr: Vec<_> = mro
                .req_ids
                .iter()
                .copied()
                .zip(mro.drafts.iter().cloned())
                .filter(|(_, d)| !d.is_empty())
                .collect();
            sched.update_draft_tokens(&dr);
        }
        for (_, toks, _) in res.iter() {
            out.extend_from_slice(toks);
        }
        steps += 1;
        assert!(steps < 10 * gen as usize + 10, "runaway loop");
    }
    eprintln!("mtp={mtp}: {steps} steps for {gen} tokens; drafts {drafts_total}, accepted {accepted_total}");
    (out, drafts_total, accepted_total)
}

/// 把 MTP 的 draft 换成 "正确答案"（plain greedy 的下一 token）的执行器包装：verify 路径必须几乎全接受。
/// 若接受率低，说明 verify / KV 回滚 / 位置有 bug（真 draft 质量不影响这条）。
struct OracleDraft {
    inner: Glm53Executor,
    answer: Vec<u32>, // plain greedy 输出
    prompt_len: usize,
    pub drafts: usize,
    pub accepted: usize,
}
impl dinfer_core::runner::ModelExecutor for OracleDraft {
    fn forward_sample(
        &mut self,
        plan: &dinfer_core::runner::BatchPlan,
        generated: &mut Vec<Vec<u32>>,
        drafts: &mut Vec<Vec<u32>>,
    ) -> anyhow::Result<()> {
        self.inner.forward_sample(plan, generated, drafts)?;
        for (i, (d, a)) in self.inner.last_accept.iter().enumerate() {
            self.drafts += d;
            self.accepted += a;
            // 下一步 draft = 该请求下一个位置的真 token：序列长 = seq_lens[i] - 被拒绝的 draft + 新生成
            let n_gen = generated[i].len();
            let base = plan.num_tokens[i] as usize - plan.spec_tokens[i].len();
            let seq_after = plan.seq_lens[i] as usize - plan.spec_tokens[i].len() + n_gen; // = 已确认 token 数（prompt+out）
            let _ = base;
            let out_idx = seq_after - self.prompt_len; // 下一个要生成的是 answer[out_idx]
            if plan.next_known[i] < 0 && out_idx < self.answer.len() && !drafts[i].is_empty() {
                drafts[i] = vec![self.answer[out_idx]];
            }
        }
        Ok(())
    }
}

#[test]
#[ignore]
fn mtp_verify_accepts_oracle_drafts() {
    let (_, prompt) = npy_i32("prompt.npy");
    let gen = 32u32;
    let (plain, _, _) = run_greedy(false, &prompt, gen);
    let dev = Device::new(0).unwrap();
    let s = dev.default_stream();
    let gemm = Gemm::new(&s).unwrap();
    let cfg =
        GlmMoeDsaConfig::from_json(&std::fs::read_to_string(dir().join("config.json")).unwrap()).unwrap();
    let src = WeightSource::open_dir(&dir()).unwrap();
    let model = Glm53Model::load(&s, &src, cfg.clone(), 256, NUM_BLOCKS, BLOCK).unwrap();
    let mtp_layer = MtpLayer::load(&s, &src, &cfg, 256, NUM_BLOCKS, BLOCK).unwrap();
    let exec = Glm53Executor::new(s.clone(), gemm, model, Some(mtp_layer)).unwrap();
    let exec =
        OracleDraft { inner: exec, answer: plain.clone(), prompt_len: prompt.len(), drafts: 0, accepted: 0 };
    let mut runner = Runner::new(exec, BLOCK);
    let params = SchedulerParams {
        sched: SchedulerConfig {
            max_num_seqs: 4,
            max_num_batched_tokens: 256,
            enable_chunked_prefill: true,
            long_prefill_token_threshold: 0,
        },
        cache: CacheConfig { block_size: BLOCK as u32, enable_prefix_caching: false, ..Default::default() },
        max_model_len: 512,
        num_spec_tokens: 1,
        async_scheduling: false,
        eos_token_ids: vec![],
    };
    let entry = cfg.kv_lora_rank + cfg.qk_rope_head_dim;
    let g = vec![GroupInfo {
        spec: KvCacheSpec::Mla { block_size: BLOCK as u32, head_size: entry, dtype_bytes: 2 },
        block_size: BLOCK as u32,
    }];
    let mut sched = Scheduler::new(params, g, NUM_BLOCKS as u32);
    let sp = SamplingParams { max_tokens: gen, temperature: 0.0, ..Default::default() };
    sched.add_request(Request::new(1, prompt.iter().map(|&x| x as u32).collect(), sp, 0));
    let mut out = Vec::new();
    let mut steps = 0;
    while sched.has_requests() {
        let so = sched.schedule();
        let mro = runner.execute(&so).unwrap();
        let res = sched.update_from_output(&so, &mro);
        if mro.drafts.iter().any(|d| !d.is_empty()) {
            let dr: Vec<_> = mro
                .req_ids
                .iter()
                .copied()
                .zip(mro.drafts.iter().cloned())
                .filter(|(_, d)| !d.is_empty())
                .collect();
            sched.update_draft_tokens(&dr);
        }
        for (_, toks, _) in res.iter() {
            out.extend_from_slice(toks);
        }
        steps += 1;
        assert!(steps < 200, "runaway");
    }
    let (d, a) = (runner.exec.drafts, runner.exec.accepted);
    eprintln!("mtp: oracle drafts: {steps} steps for {gen} tokens, accepted {a}/{d}");
    assert_eq!(out, plain, "oracle-draft spec output differs from plain greedy");
    assert!(d >= 10, "too few drafts verified: {d}");
    assert!(a * 10 >= d * 9, "oracle drafts should be accepted ≥ 90%: {a}/{d}");
    assert!(steps <= (gen as usize) * 2 / 3 + 3, "with ~100% acceptance, steps should be ~gen/2: {steps}");
}

#[test]
#[ignore]
fn mtp_spec_decode_matches_plain_greedy() {
    let (_, prompt) = npy_i32("prompt.npy");
    let gen = 32;
    let (plain, _, _) = run_greedy(false, &prompt, gen);
    let (spec, drafts, accepted) = run_greedy(true, &prompt, gen);
    assert_eq!(plain.len(), gen as usize);
    assert_eq!(spec.len(), gen as usize, "spec run must produce exactly max_tokens tokens");
    // greedy 投机解码不改变输出：逐 token 相等
    let first_diff = plain.iter().zip(&spec).position(|(a, b)| a != b);
    eprintln!("mtp: plain vs spec first diff at {first_diff:?}; acceptance {accepted}/{drafts}");
    assert_eq!(plain, spec, "speculative greedy output differs from plain greedy");
    assert!(drafts >= gen as usize / 2, "MTP should have drafted on most decode steps: {drafts}");
    // HF greedy（无 MTP）也应一致（主干正确性已由 p4/p5 覆盖；这里只做弱检查：前 4 个 token）
    let (_, hf) = npy_i32("expect_greedy.npy");
    let hf: Vec<u32> = hf.iter().map(|&x| x as u32).collect();
    let k = plain.iter().zip(&hf).take_while(|(a, b)| a == b).count();
    eprintln!("mtp: plain greedy matches HF for the first {k}/{} tokens", hf.len());
}
