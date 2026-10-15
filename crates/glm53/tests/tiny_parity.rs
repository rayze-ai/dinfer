//! tiny GLM（HF DeepseekV3 实现，MLA + dense MLP）对拍：
//! 1) prefill logits 与 HF 一致；2) greedy 32 token 与 HF 一致（prefill 一次 + 31 次 decode）。
//!
//! 先 `python3 scripts/verify/gen_tiny_glm.py test/tiny_glm`。
#![cfg(feature = "ref-tests")]

use dinfer_glm53::config::GlmMoeDsaConfig;
use dinfer_glm53::model::{Forward, Glm53Model};
use dinfer_layers::mla::AttnMeta;
use dinfer_layers::{Device, Gemm, Tensor, WeightSource};
use std::path::PathBuf;

fn dir() -> PathBuf {
    let p = PathBuf::from(std::env::var("DINFER_TINY_GLM").unwrap_or_else(|_| "test/tiny_glm".into()));
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

const BLOCK: usize = 16;
/// bf16 两层模型：全词表 logits 允许的最大绝对偏差（实测 ~0.007）。
const MAX_ABS: f32 = 0.05;
/// HF top1/top2 差 ≤ 此值视为 bf16 平局（实测分叉步 gap = 0.0039 = 1 ULP）。
const TIE_GAP: f32 = 0.02;

struct Harness {
    dev: Device,
    gemm: Gemm,
    model: Glm53Model,
    cfg: GlmMoeDsaConfig,
}

fn load(max_tokens: usize) -> Harness {
    let dev = Device::new(0).unwrap();
    let s = dev.default_stream();
    let gemm = Gemm::new(&s).unwrap();
    let cfg =
        GlmMoeDsaConfig::from_json(&std::fs::read_to_string(dir().join("config.json")).unwrap()).unwrap();
    let mut src = WeightSource::open_dir(&dir()).unwrap();
    if let Some((bn, bk)) = cfg.fp8_block_size() {
        src = src.with_quant_block(bn, bk);
    }
    let model = Glm53Model::load(&s, &src, cfg.clone(), max_tokens, 64, BLOCK).unwrap();
    Harness { dev, gemm, model, cfg }
}

/// 单请求 prefill：slot = position（块 0.. 连续分配），返回最后一个 token 的 logits 行以及全部行。
fn prefill(h: &mut Harness, prompt: &[i32], all_rows: bool) -> Vec<f32> {
    let s = h.dev.default_stream();
    let t = prompt.len();
    let v = h.cfg.vocab_size as usize;
    let ids = Tensor::from_vec(&s, prompt.to_vec(), &[t]).unwrap();
    let pos = Tensor::from_vec(&s, (0..t as i32).collect(), &[t]).unwrap();
    let slots = Tensor::from_vec(&s, (0..t as i64).collect(), &[t]).unwrap();
    let cu = Tensor::from_vec(&s, vec![0i32, t as i32], &[2]).unwrap();
    let rows: Vec<i32> = if all_rows { (0..t as i32).collect() } else { vec![t as i32 - 1] };
    let nrows = rows.len();
    let rows_t = Tensor::from_vec(&s, rows, &[nrows]).unwrap();
    let mut logits = Tensor::<f32>::zeros(&s, &[nrows, v]).unwrap();
    let f = Forward {
        input_ids: &ids,
        positions: &pos,
        slot_mapping: &slots,
        meta: AttnMeta::Prefill { cu_seqlens: &cu, num_reqs: 1 },
        logits_rows: &rows_t,
    };
    h.model.forward(&s, &h.gemm, &f, &mut logits).unwrap();
    logits.to_vec(&s).unwrap()
}

/// 单请求 decode 一步：token 放在位置 pos，slot = pos，seq_len = pos+1。
fn decode(h: &mut Harness, token: i32, pos: usize) -> Vec<f32> {
    let s = h.dev.default_stream();
    let v = h.cfg.vocab_size as usize;
    let ids = Tensor::from_vec(&s, vec![token], &[1]).unwrap();
    let positions = Tensor::from_vec(&s, vec![pos as i32], &[1]).unwrap();
    let slots = Tensor::from_vec(&s, vec![pos as i64], &[1]).unwrap();
    let max_blocks = 64;
    let bt: Vec<i32> = (0..max_blocks as i32).collect(); // 块 i 就是物理块 i（连续分配）
    let bt_t = Tensor::from_vec(&s, bt, &[1, max_blocks]).unwrap();
    let seq = Tensor::from_vec(&s, vec![pos as i32 + 1], &[1]).unwrap();
    let rows_t = Tensor::from_vec(&s, vec![0i32], &[1]).unwrap();
    let mut logits = Tensor::<f32>::zeros(&s, &[1, v]).unwrap();
    let f = Forward {
        input_ids: &ids,
        positions: &positions,
        slot_mapping: &slots,
        meta: AttnMeta::Decode { block_table: &bt_t, seq_lens: &seq, max_blocks },
        logits_rows: &rows_t,
    };
    h.model.forward(&s, &h.gemm, &f, &mut logits).unwrap();
    logits.to_vec(&s).unwrap()
}

fn argmax(row: &[f32]) -> i32 {
    let mut best = 0usize;
    for i in 1..row.len() {
        if row[i] > row[best] {
            best = i;
        }
    }
    best as i32
}

#[test]
#[ignore]
fn tiny_glm_prefill_logits_parity() {
    let prompt = npy_i32("prompt.npy");
    let (shape, want) = npy_f32("expect_prefill_logits.npy");
    let v = shape[1];
    let mut h = load(64);
    let got = prefill(&mut h, &prompt, true);
    assert_eq!(got.len(), want.len());
    // 逐行：argmax 一致（HF top1/top2 平局 ≤ TIE_GAP 的行除外）+ 数值接近（bf16 两层累积误差）
    let mut max_abs = 0f32;
    let mut argmax_mismatch = 0;
    for r in 0..shape[0] {
        let g = &got[r * v..(r + 1) * v];
        let w = &want[r * v..(r + 1) * v];
        if argmax(g) != argmax(w) {
            let mut sorted: Vec<f32> = w.to_vec();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let gap = sorted[0] - sorted[1];
            eprintln!("row {r}: argmax got {} want {} (hf top1-top2 gap {gap:.4})", argmax(g), argmax(w));
            if gap > TIE_GAP {
                argmax_mismatch += 1;
            }
        }
        for i in 0..v {
            max_abs = max_abs.max((g[i] - w[i]).abs());
        }
    }
    eprintln!("prefill logits max abs diff = {max_abs}");
    assert!(max_abs < MAX_ABS, "logits max abs diff {max_abs} too large (limit {MAX_ABS})");
    assert_eq!(argmax_mismatch, 0, "{argmax_mismatch} rows have different argmax");
}

#[test]
#[ignore]
fn tiny_glm_greedy_parity() {
    let prompt = npy_i32("prompt.npy");
    let want = npy_i32("expect_greedy.npy");
    let mut h = load(64);
    let last = prefill(&mut h, &prompt, false);
    let mut got = vec![argmax(&last)];
    let mut pos = prompt.len();
    while got.len() < want.len() {
        let tok = *got.last().unwrap();
        let row = decode(&mut h, tok, pos);
        got.push(argmax(&row));
        pos += 1;
    }
    let (shape, step_logits) = npy_f32("expect_step_logits.npy");
    let v = shape[1];
    let first_diff = got.iter().zip(&want).position(|(a, b)| a != b);
    match first_diff {
        None => {}
        Some(i) => {
            // 允许在 HF 自身 top1/top2 几乎平局（≤ TIE_GAP）的那一步分叉；之后的序列不再可比。
            let w = &step_logits[i * v..(i + 1) * v];
            let mut sorted: Vec<f32> = w.to_vec();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let gap = sorted[0] - sorted[1];
            eprintln!(
                "greedy diverges at step {i}: got {} want {} (hf top1-top2 gap {gap:.4})",
                got[i], want[i]
            );
            assert!(gap <= TIE_GAP, "greedy diverges at step {i} with a clear HF gap {gap:.4} > {TIE_GAP}");
            // 分叉 token 必须是 HF 的 top-2 之一
            let top2: Vec<i32> = {
                let mut idx: Vec<usize> = (0..v).collect();
                idx.sort_by(|&a, &b| w[b].partial_cmp(&w[a]).unwrap());
                idx[..2].iter().map(|&x| x as i32).collect()
            };
            assert!(top2.contains(&got[i]), "diverged token {} not in HF top-2 {top2:?}", got[i]);
        }
    }
}

/// 逐步对拍：每个 decode 步的 logits 与 HF 全序列前向的最后一行比较（喂 HF 的 greedy token，避免误差累积）。
#[test]
#[ignore]
fn tiny_glm_step_logits_parity() {
    let prompt = npy_i32("prompt.npy");
    let want_tokens = npy_i32("expect_greedy.npy");
    let (shape, want) = npy_f32("expect_step_logits.npy");
    let v = shape[1];
    let mut h = load(64);
    let mut rows: Vec<Vec<f32>> = vec![prefill(&mut h, &prompt, false)];
    let mut pos = prompt.len();
    for &tok in &want_tokens[..want_tokens.len() - 1] {
        rows.push(decode(&mut h, tok, pos));
        pos += 1;
    }
    let mut worst = 0f32;
    let mut first_argmax_diff = None;
    for (i, got) in rows.iter().enumerate() {
        let w = &want[i * v..(i + 1) * v];
        let mut mx = 0f32;
        for j in 0..v {
            mx = mx.max((got[j] - w[j]).abs());
        }
        worst = worst.max(mx);
        let (ga, wa) = (argmax(got), argmax(w));
        let mut sorted: Vec<f32> = w.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let gap = sorted[0] - sorted[1];
        eprintln!("step {i:2}: max abs diff {mx:.4}  argmax got {ga} want {wa}  hf top1-top2 gap {gap:.4}");
        // bf16 下 top1/top2 相差 ≤ TIE_GAP 视为平局：两边都对，argmax 不要求一致。
        if ga != wa && gap > TIE_GAP && first_argmax_diff.is_none() {
            first_argmax_diff = Some(i);
        }
    }
    eprintln!("worst max abs diff over {} steps = {worst}", rows.len());
    assert!(worst < MAX_ABS, "step logits max abs diff {worst} too large (limit {MAX_ABS})");
    assert_eq!(first_argmax_diff, None, "argmax differs (beyond tie gap) at step {first_argmax_diff:?}");
}
