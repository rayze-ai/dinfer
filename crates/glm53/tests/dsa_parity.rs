//! P5 DSA 对拍（tiny GlmMoeDsa：3 层 full/shared/full，index_topk 8，稀疏真正生效）。
//!
//! bf16 下两类 "硬判决" 都可能在两边翻转，翻转后下游不可比，这不是 bug（见 docs/pitfalls.md）：
//! - 索引器 top-k：q/k 各自 bf16 舍入后 fp32 点积，分差量级 ~1e-2；HF 自身 k-th 与 (k+1)-th 分差中位数仅 0.006。
//! - MoE 路由 top-2：第 2/3 名专家 sigmoid 分差 < ~0.01 时专家选择会不同（tiny 4 专家随机权重下约 2% 的行）。
//! 三道测试据此设计：
//! 1. `dsa_topk_indices_match_hf`：每个 full 层每行 top-k 集合 vs HF；不一致的行，对称差元素的 HF 分数必须落在
//!    "HF k-th 分数 ± 本行两边分数实测最大偏差" 之内（用我们自己的 index_scores 量误差，不猜阈值）。
//! 2. `dsa_sparse_attention_with_hf_topk`：注入 HF top-k，只验稀疏 attention：长 prompt（256）prefill + 32 步 decode；
//!    logits 断言只在 HF 路由可判定（margin > ROUTER_TIE）的行/步上做，其余只上报。
//! 3. `dsa_self_run_long_prompt`：全程自算；断言只在 "索引器 k-th 分差 > 本行实测分数误差 且 路由可判定" 的行上做。
//!
//! 先 `/data/ref/venv-tf519/bin/python scripts/verify/gen_tiny_glm.py test/tiny_glm_dsa --layers 3 --first-k-dense 1 --dsa`。
#![cfg(feature = "ref-tests")]

use dinfer_glm53::config::GlmMoeDsaConfig;
use dinfer_glm53::model::{Forward, Glm53Model};
use dinfer_layers::mla::AttnMeta;
use dinfer_layers::{Device, Gemm, Tensor, WeightSource};
use std::collections::BTreeSet;
use std::path::PathBuf;

fn dir() -> PathBuf {
    let p = PathBuf::from(std::env::var("DINFER_TINY_GLM").unwrap_or_else(|_| "test/tiny_glm_dsa".into()));
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
fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..row.len() {
        if row[i] > row[best] {
            best = i;
        }
    }
    best
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
const NUM_BLOCKS: usize = 64; // 1024 token 容量
const MAX_ABS: f32 = 0.05;
const TIE_GAP: f32 = 0.02;
/// MoE 路由 "第 2 vs 第 3 名专家" 分差 ≤ 此值 → 该行路由不可判定，下游 logits 不比较。
const ROUTER_TIE: f32 = 0.01;
/// 索引器分数误差带的下限（即使实测误差更小也至少放这么宽；fp32 累加顺序量级）。
const SCORE_BAND_MIN: f32 = 1e-4;

struct H {
    dev: Device,
    gemm: Gemm,
    model: Glm53Model,
    cfg: GlmMoeDsaConfig,
}

fn load(max_tokens: usize) -> H {
    let dev = Device::new(0).unwrap();
    let s = dev.default_stream();
    let gemm = Gemm::new(&s).unwrap();
    let cfg =
        GlmMoeDsaConfig::from_json(&std::fs::read_to_string(dir().join("config.json")).unwrap()).unwrap();
    assert!(cfg.dsa, "this test needs a --dsa tiny model");
    let src = WeightSource::open_dir(&dir()).unwrap();
    let model = Glm53Model::load(&s, &src, cfg.clone(), max_tokens, NUM_BLOCKS, BLOCK).unwrap();
    H { dev, gemm, model, cfg }
}
fn full_layers(h: &H) -> Vec<usize> {
    (0..h.cfg.num_hidden_layers).filter(|&l| h.cfg.has_indexer(l)).map(|l| l as usize).collect()
}

type Capture = Vec<(usize, Vec<i32>, Vec<f32>, usize)>;

/// 单请求 prefill（slot = position）。`override_tag`：Some(tag) → 用 `indexer_topk_L{l}{tag}.npy` 注入。
fn prefill(h: &mut H, prompt: &[i32], override_tag: Option<&str>) -> (Vec<f32>, Capture) {
    let s = h.dev.default_stream();
    let t = prompt.len();
    let v = h.cfg.vocab_size as usize;
    let ids = Tensor::from_vec(&s, prompt.to_vec(), &[t]).unwrap();
    let pos = Tensor::from_vec(&s, (0..t as i32).collect(), &[t]).unwrap();
    let slots = Tensor::from_vec(&s, (0..t as i64).collect(), &[t]).unwrap();
    let cu = Tensor::from_vec(&s, vec![0i32, t as i32], &[2]).unwrap();
    let bt = Tensor::from_vec(&s, (0..NUM_BLOCKS as i32).collect(), &[1, NUM_BLOCKS]).unwrap();
    let seq = Tensor::from_vec(&s, vec![t as i32], &[1]).unwrap();
    let rows_t = Tensor::from_vec(&s, (0..t as i32).collect(), &[t]).unwrap();
    let mut logits = Tensor::<f32>::zeros(&s, &[t, v]).unwrap();
    h.model.capture_topk = Some(Vec::new());
    h.model.topk_override = override_tag.map(|tag| {
        full_layers(h).into_iter().map(|l| (l, npy_i32(&format!("indexer_topk_L{l}{tag}.npy")).1)).collect()
    });
    let f = Forward {
        input_ids: &ids,
        positions: &pos,
        slot_mapping: &slots,
        meta: AttnMeta::Prefill {
            cu_seqlens: &cu,
            num_reqs: 1,
            block_table: &bt,
            seq_lens: &seq,
            max_blocks: NUM_BLOCKS,
        },
        logits_rows: &rows_t,
    };
    h.model.forward(&s, &h.gemm, &f, &mut logits).unwrap();
    h.model.topk_override = None;
    let caps = h.model.capture_topk.take().unwrap();
    (logits.to_vec(&s).unwrap(), caps)
}

/// 单请求 decode 一步。`override_step`：Some((steps 表, step i)) → 注入 HF 该步的 top-k。
fn decode(
    h: &mut H,
    token: i32,
    pos: usize,
    override_step: Option<(&[(usize, Vec<i32>)], usize)>,
) -> Vec<f32> {
    let s = h.dev.default_stream();
    let v = h.cfg.vocab_size as usize;
    let k = h.cfg.index_topk as usize;
    let ids = Tensor::from_vec(&s, vec![token], &[1]).unwrap();
    let positions = Tensor::from_vec(&s, vec![pos as i32], &[1]).unwrap();
    let slots = Tensor::from_vec(&s, vec![pos as i64], &[1]).unwrap();
    let bt = Tensor::from_vec(&s, (0..NUM_BLOCKS as i32).collect(), &[1, NUM_BLOCKS]).unwrap();
    let seq = Tensor::from_vec(&s, vec![pos as i32 + 1], &[1]).unwrap();
    let rows_t = Tensor::from_vec(&s, vec![0i32], &[1]).unwrap();
    let mut logits = Tensor::<f32>::zeros(&s, &[1, v]).unwrap();
    h.model.topk_override = override_step
        .map(|(tab, i)| tab.iter().map(|(l, all)| (*l, all[i * k..(i + 1) * k].to_vec())).collect());
    let f = Forward {
        input_ids: &ids,
        positions: &positions,
        slot_mapping: &slots,
        meta: AttnMeta::Decode { block_table: &bt, seq_lens: &seq, max_blocks: NUM_BLOCKS },
        logits_rows: &rows_t,
    };
    h.model.forward(&s, &h.gemm, &f, &mut logits).unwrap();
    h.model.topk_override = None;
    logits.to_vec(&s).unwrap()
}

/// HF 分数行 (k-th − (k+1)-th)；可见数 ≤ k 时 +inf。
fn kth_margin(row: &[f32], k: usize) -> f32 {
    if row.len() <= k {
        return f32::INFINITY;
    }
    let mut srt = row.to_vec();
    srt.sort_by(|a, b| b.partial_cmp(a).unwrap());
    srt[k - 1] - srt[k]
}

/// 对一个 prompt：每个 full 层每行 top-k 集合 vs HF。返回 (比较行数, 平局容忍行数, 索引器分数实测最大误差)。
fn check_topk(h: &mut H, tag: &str, prompt: &[i32]) -> (usize, usize, f32) {
    let t = prompt.len();
    let k = h.cfg.index_topk as usize;
    let (_, caps) = prefill(h, prompt, None);
    assert_eq!(caps.iter().map(|c| c.0).collect::<Vec<_>>(), full_layers(h), "captured layers");
    let (mut rows, mut tolerated, mut worst_score) = (0, 0, 0f32);
    for (li, got, ours_sc, stride) in &caps {
        let (shape, hf) = npy_i32(&format!("indexer_topk_L{li}{tag}.npy"));
        let (sshape, score) = npy_f32(&format!("indexer_score_L{li}{tag}.npy"));
        assert_eq!(shape, vec![t, k]);
        assert_eq!(sshape, vec![t, t]);
        for r in 0..t {
            let vis = r + 1;
            let hf_row = &score[r * t..r * t + vis];
            let my_row = &ours_sc[r * stride..r * stride + vis];
            let err = max_abs(hf_row, my_row);
            worst_score = worst_score.max(err);
            let ours: BTreeSet<i32> = got[r * k..(r + 1) * k].iter().copied().filter(|&x| x >= 0).collect();
            let theirs: BTreeSet<i32> =
                hf[r * k..(r + 1) * k].iter().copied().filter(|&x| (x as usize) < vis).collect();
            assert_eq!(ours.len(), vis.min(k), "L{li}{tag} row {r}: {} valid indices", ours.len());
            rows += 1;
            if ours == theirs {
                continue;
            }
            // 平局判定：对称差元素的 HF 分数都在 [kth − band, kth + band]，band = 本行两边分数的实测最大偏差
            let kth = theirs.iter().map(|&i| hf_row[i as usize]).fold(f32::INFINITY, f32::min);
            let band = err.max(SCORE_BAND_MIN) * 1.01;
            for &i in ours.symmetric_difference(&theirs) {
                let sc = hf_row[i as usize];
                assert!(
                    (sc - kth).abs() <= band,
                    "L{li}{tag} row {r}: index {i} HF score {sc} vs k-th {kth}, band {band} (row score err {err}); ours {ours:?} hf {theirs:?}"
                );
            }
            tolerated += 1;
        }
    }
    (rows, tolerated, worst_score)
}

#[test]
#[ignore]
fn dsa_topk_indices_match_hf() {
    let mut h = load(512);
    let (_, p1) = npy_i32("prompt.npy");
    let (r1, t1, e1) = check_topk(&mut h, "", &p1);
    let (_, p2) = npy_i32("prompt_long.npy");
    let (r2, t2, e2) = check_topk(&mut h, "_long", &p2);
    eprintln!(
        "topk rows compared: short {r1} (tie-tolerated {t1}, score err {e1:.4}), long {r2} (tie-tolerated {t2}, score err {e2:.4})"
    );
    assert!(e2 < 0.1, "indexer score error too large: {e2}");
    assert!(t2 * 4 <= r2, "too many tie-tolerated rows on long prompt: {t2}/{r2}");
}

/// 注入 HF top-k → 稀疏 attention 本身的正确性（prefill 展开路径 + decode 吸收路径 + shared 层复用）。
#[test]
#[ignore]
fn dsa_sparse_attention_with_hf_topk() {
    let mut h = load(512);
    let (_, prompt) = npy_i32("prompt_long.npy");
    let t = prompt.len();
    let (shape, want_pf) = npy_f32("expect_prefill_logits_long.npy");
    let v = shape[1];
    let (_, rmargin) = npy_f32("router_margin_long.npy");
    let (got_pf, _) = prefill(&mut h, &prompt, Some("_long"));
    let (mut worst, mut bad, mut skipped, mut worst_skipped) = (0f32, 0, 0, 0f32);
    for r in 0..t {
        let (g, w) = (&got_pf[r * v..(r + 1) * v], &want_pf[r * v..(r + 1) * v]);
        let mx = max_abs(g, w);
        if rmargin[r] <= ROUTER_TIE {
            skipped += 1;
            worst_skipped = worst_skipped.max(mx);
            continue;
        }
        worst = worst.max(mx);
        if argmax(g) != argmax(w) && top_gap(w) > TIE_GAP {
            bad += 1;
            eprintln!(
                "prefill row {r}: argmax got {} want {} gap {:.4} maxabs {mx:.4}",
                argmax(g),
                argmax(w),
                top_gap(w)
            );
        }
    }
    eprintln!(
        "[hf-topk] long prefill: {} router-decidable rows worst max abs {worst}; {skipped} router-near-tie rows (worst {worst_skipped}, not asserted)",
        t - skipped
    );
    assert!(worst < MAX_ABS, "[hf-topk] prefill logits max abs {worst} > {MAX_ABS}");
    assert_eq!(bad, 0, "[hf-topk] {bad} prefill rows argmax differ beyond tie gap");
    assert!(skipped * 10 <= t, "[hf-topk] too many router-near-tie rows: {skipped}/{t}");

    let (_, want_tok) = npy_i32("expect_greedy_long.npy");
    let (sshape, want_step) = npy_f32("expect_step_logits_long.npy");
    let (_, rmargin_steps) = npy_f32("router_margin_long_steps.npy");
    assert_eq!(sshape[1], v);
    let steps: Vec<(usize, Vec<i32>)> = full_layers(&h)
        .into_iter()
        .map(|l| (l, npy_i32(&format!("indexer_topk_L{l}_steps.npy")).1))
        .collect();
    let mut rows: Vec<Vec<f32>> = vec![got_pf[(t - 1) * v..].to_vec()];
    let mut pos = t;
    for (i, &tok) in want_tok[..want_tok.len() - 1].iter().enumerate() {
        rows.push(decode(&mut h, tok, pos, Some((&steps, i + 1))));
        pos += 1;
    }
    let (mut worst, mut first_bad, mut dec_steps) = (0f32, None, 0);
    for (i, got) in rows.iter().enumerate() {
        let w = &want_step[i * v..(i + 1) * v];
        let mx = max_abs(got, w);
        let (ga, wa, gap) = (argmax(got), argmax(w), top_gap(w));
        let dec = rmargin_steps[i] > ROUTER_TIE;
        eprintln!(
            "[hf-topk] step {i:2}: max abs {mx:.4} argmax got {ga} want {wa} gap {gap:.4} router-margin {:.4}{}",
            rmargin_steps[i],
            if dec { "" } else { " (near-tie, not asserted)" }
        );
        if !dec {
            continue;
        }
        dec_steps += 1;
        worst = worst.max(mx);
        if ga != wa && gap > TIE_GAP && first_bad.is_none() {
            first_bad = Some(i);
        }
    }
    eprintln!("[hf-topk] long decode: {dec_steps}/{} decidable steps, worst max abs {worst}", rows.len());
    assert!(worst < MAX_ABS, "[hf-topk] step logits max abs {worst} > {MAX_ABS}");
    assert_eq!(first_bad, None, "[hf-topk] argmax differs beyond tie gap at step {first_bad:?}");
    assert!(dec_steps * 4 >= rows.len() * 3, "[hf-topk] too few decidable decode steps: {dec_steps}");
}

/// 全程自算：索引器 + 稀疏 attention。断言只在 "索引器 top-k 可判定 且 路由可判定" 的行/步上做。
#[test]
#[ignore]
fn dsa_self_run_long_prompt() {
    let mut h = load(512);
    let k = h.cfg.index_topk as usize;
    let (_, prompt) = npy_i32("prompt_long.npy");
    let t = prompt.len();
    let (shape, want_pf) = npy_f32("expect_prefill_logits_long.npy");
    let v = shape[1];
    let (_, rmargin) = npy_f32("router_margin_long.npy");
    let (got_pf, caps) = prefill(&mut h, &prompt, None);
    // 行 r 的索引器可判定：所有 full 层 HF k-th 分差 > 本行两边分数实测误差
    let hf_scores: Vec<(usize, Vec<f32>)> = full_layers(&h)
        .into_iter()
        .map(|l| (l, npy_f32(&format!("indexer_score_L{l}_long.npy")).1))
        .collect();
    let idx_decidable = |r: usize| -> bool {
        let vis = r + 1;
        caps.iter().zip(&hf_scores).all(|((_, _, mine, stride), (_, hf))| {
            let hf_row = &hf[r * t..r * t + vis];
            let err = max_abs(hf_row, &mine[r * stride..r * stride + vis]).max(SCORE_BAND_MIN);
            kth_margin(hf_row, k) > err * 1.01
        })
    };
    let (mut decidable, mut bad, mut worst_dec, mut worst_all) = (0, 0, 0f32, 0f32);
    for r in 0..t {
        let (g, w) = (&got_pf[r * v..(r + 1) * v], &want_pf[r * v..(r + 1) * v]);
        let mx = max_abs(g, w);
        worst_all = worst_all.max(mx);
        if rmargin[r] > ROUTER_TIE && idx_decidable(r) {
            decidable += 1;
            worst_dec = worst_dec.max(mx);
            if argmax(g) != argmax(w) && top_gap(w) > TIE_GAP {
                bad += 1;
                eprintln!(
                    "[self] prefill row {r}: argmax got {} want {} gap {:.4} maxabs {mx:.4}",
                    argmax(g),
                    argmax(w),
                    top_gap(w)
                );
            }
        }
    }
    eprintln!(
        "[self] long prefill: {decidable}/{t} decidable rows (worst max abs {worst_dec}); all rows worst {worst_all}"
    );
    assert!(worst_dec < MAX_ABS, "[self] decidable-row logits max abs {worst_dec} > {MAX_ABS}");
    assert_eq!(bad, 0, "[self] {bad} decidable rows argmax differ");
    assert!(decidable >= 64, "[self] too few decidable rows: {decidable}/{t}");

    // decode：喂 HF greedy token；每步可判定 = 路由可判定 且 该步索引器 k-th 分差 > 索引器分数误差上界（prefill 实测最大值）
    let (_, want_tok) = npy_i32("expect_greedy_long.npy");
    let (sshape, want_step) = npy_f32("expect_step_logits_long.npy");
    let (_, rmargin_steps) = npy_f32("router_margin_long_steps.npy");
    assert_eq!(sshape[1], v);
    let idx_err_bound = caps
        .iter()
        .zip(&hf_scores)
        .map(|((_, _, mine, stride), (_, hf))| {
            (0..t)
                .map(|r| max_abs(&hf[r * t..r * t + r + 1], &mine[r * stride..r * stride + r + 1]))
                .fold(0f32, f32::max)
        })
        .fold(0f32, f32::max)
        .max(SCORE_BAND_MIN);
    let step_scores: Vec<(usize, Vec<f32>, usize)> = full_layers(&h)
        .into_iter()
        .map(|l| {
            let (sh, sc) = npy_f32(&format!("indexer_score_L{l}_steps.npy"));
            (l, sc, sh[1])
        })
        .collect();
    let mut rows: Vec<Vec<f32>> = vec![got_pf[(t - 1) * v..].to_vec()];
    let mut pos = t;
    for &tok in &want_tok[..want_tok.len() - 1] {
        rows.push(decode(&mut h, tok, pos, None));
        pos += 1;
    }
    let (mut dec_steps, mut bad_steps, mut worst) = (0, 0, 0f32);
    for (i, got) in rows.iter().enumerate() {
        let w = &want_step[i * v..(i + 1) * v];
        let mx = max_abs(got, w);
        let (ga, wa, gap) = (argmax(got), argmax(w), top_gap(w));
        let imargin = step_scores
            .iter()
            .map(|(_, sc, stride)| kth_margin(&sc[i * stride..i * stride + t + i], k))
            .fold(f32::INFINITY, f32::min);
        let dec = rmargin_steps[i] > ROUTER_TIE && imargin > idx_err_bound * 1.01;
        eprintln!(
            "[self] step {i:2}: max abs {mx:.4} argmax got {ga} want {wa} gap {gap:.4} indexer-margin {imargin:.4} router-margin {:.4}{}",
            rmargin_steps[i],
            if dec { "" } else { " (near-tie, not asserted)" }
        );
        if dec {
            dec_steps += 1;
            worst = worst.max(mx);
            if ga != wa && gap > TIE_GAP {
                bad_steps += 1;
            }
        }
    }
    eprintln!(
        "[self] decode: {dec_steps}/{} decidable steps (indexer err bound {idx_err_bound:.4}), worst max abs {worst}",
        rows.len()
    );
    assert!(worst < MAX_ABS, "[self] decidable-step logits max abs {worst} > {MAX_ABS}");
    assert_eq!(bad_steps, 0, "[self] {bad_steps} decidable steps argmax differ");
}
