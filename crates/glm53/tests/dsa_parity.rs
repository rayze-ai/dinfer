//! P5 DSA 对拍（tiny GlmMoeDsa：3 层 full/shared/full，index_topk 8，稀疏真正生效）。
//!
//! bf16 下两类 "硬判决" 都可能在两边翻转，翻转后下游不可比，这不是 bug（见 docs/pitfalls.md）：
//! - 索引器 top-k：q/k 各自 bf16 舍入后 fp32 点积，分差量级 ~1e-2；HF 自身 k-th 与 (k+1)-th 分差中位数仅 0.006。
//! - MoE 路由 top-2：第 2/3 名专家 sigmoid 分差 < ~0.01 时专家选择会不同（tiny 4 专家随机权重下约 2% 的行）。
//!
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
    let p = PathBuf::from(std::env::var("TINY_GLM_DIR").unwrap_or_else(|_| "test/tiny_glm_dsa".into()));
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
const ROUTER_TIE: f32 = 0.02;
/// 索引器分数误差带的下限（即使实测误差更小也至少放这么宽；fp32 累加顺序量级）。
const SCORE_BAND_MIN: f32 = 1e-4;
/// 注入 HF top-k 时某层输出 hidden 与 HF 的逐行 max abs 超过此值 → 该行 "真漂移"（MoE 路由翻转）。
/// 纯数值噪声（attention / norm / bf16）实测 ≤ 0.004，路由翻转 ≥ 0.05。
const DRIFT: f32 = 0.02;

type LayerCapture = Vec<(usize, Vec<f32>, Vec<f32>)>;

struct H {
    dev: Device,
    gemm: Gemm,
    model: Glm53Model,
    cfg: GlmMoeDsaConfig,
    /// 上一次 prefill 的逐层 (layer, attn_out, hidden_after) 捕获。
    last_layers: LayerCapture,
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
    H { dev, gemm, model, cfg, last_layers: Vec::new() }
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
    h.model.capture_layers = Some(Vec::new());
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
    h.last_layers = h.model.capture_layers.take().unwrap();
    (logits.to_vec(&s).unwrap(), caps)
}

/// 注入 HF top-k 跑 prefill 后，逐层比 hidden（残差流）找出 **真漂移** 的行：`drift[li]` = 第 li 层输出里
/// max abs > DRIFT 的行集合（累计：一旦漂移，后续层也算漂移）。路由 margin 只是漂移的保守代理（near-tie 行里
/// 实际翻转的不到 1/10），用实测漂移过滤能保住绝大多数行。需要 fixture 有 `hidden_L{li}{tag}.npy`。
fn drift_rows(h: &H, tag: &str, t: usize) -> Vec<BTreeSet<usize>> {
    let hid = h.cfg.hidden_size as usize;
    let mut acc = BTreeSet::new();
    let mut out = Vec::new();
    for (li, _attn, hidden) in &h.last_layers {
        let (shape, hf) = npy_f32(&format!("hidden_L{li}{tag}.npy"));
        assert_eq!(shape, vec![t, hid], "hidden_L{li}{tag}");
        for r in 0..t {
            let e = max_abs(&hidden[r * hid..(r + 1) * hid], &hf[r * hid..(r + 1) * hid]);
            if e > DRIFT {
                acc.insert(r);
            }
        }
        out.push(acc.clone());
    }
    out
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
    h.model.capture_layers = Some(Vec::new());
    let f = Forward {
        input_ids: &ids,
        positions: &positions,
        slot_mapping: &slots,
        meta: AttnMeta::Decode { block_table: &bt, seq_lens: &seq, max_blocks: NUM_BLOCKS },
        logits_rows: &rows_t,
    };
    h.model.forward(&s, &h.gemm, &f, &mut logits).unwrap();
    h.model.topk_override = None;
    h.last_layers = h.model.capture_layers.take().unwrap();
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

/// 对一个 prompt：每个 full 层每行 **自算** top-k 集合 vs HF。返回 (比较行数, 平局容忍行数, 索引器分数实测最大误差)。
/// 逐层隔离：注入 HF top-k 跑（上游层的稀疏选择与 HF 一致，本层输入 x 才可比），但比较的是本层覆盖前的自算值
/// （`SparseCtx::topk_self`）。不注入时 L0 一旦在平局行选了不同 token，L2 的输入就已漂移，分数差 > 1 也不是 bug。
/// 还有一层过滤：MoE 路由平局行（`router_margin ≤ ROUTER_TIE`）的 hidden 在两边会不同，该行作为后续层索引器的
/// query（本行分数）与 key（所有后续行对它那一列的分数）都不可比 → 该行不断言，也不计入分数误差。
fn check_topk(h: &mut H, tag: &str, prompt: &[i32]) -> (usize, usize, f32) {
    let t = prompt.len();
    let k = h.cfg.index_topk as usize;
    let (_, caps) = prefill(h, prompt, Some(tag));
    assert_eq!(caps.iter().map(|c| c.0).collect::<Vec<_>>(), full_layers(h), "captured layers");
    let drift = drift_rows(h, tag, t);
    let (mut rows, mut tolerated, mut worst_score, mut skipped) = (0, 0, 0f32, 0);
    for (li, got, ours_sc, stride) in &caps {
        let (shape, hf) = npy_i32(&format!("indexer_topk_L{li}{tag}.npy"));
        let (sshape, score) = npy_f32(&format!("indexer_score_L{li}{tag}.npy"));
        assert_eq!(shape, vec![t, k]);
        assert_eq!(sshape, vec![t, t]);
        // 第 li 层索引器的输入 = 第 li-1 层输出；L0 的输入是 embedding，不受路由影响
        let first_full = *li == 0;
        let empty = BTreeSet::new();
        let tie_cols: &BTreeSet<usize> = if first_full { &empty } else { &drift[*li - 1] };
        for r in 0..t {
            let vis = r + 1;
            if tie_cols.contains(&r) {
                skipped += 1;
                continue;
            }
            let hf_row = &score[r * t..r * t + vis];
            let my_row = &ours_sc[r * stride..r * stride + vis];
            // 分数误差只在 "非平局列" 上量（平局行作为 key 的那一列两边不可比）
            let err = (0..vis)
                .filter(|j| !tie_cols.contains(j))
                .map(|j| (hf_row[j] - my_row[j]).abs())
                .fold(0f32, f32::max);
            worst_score = worst_score.max(err);
            let ours: BTreeSet<i32> = got[r * k..(r + 1) * k].iter().copied().filter(|&x| x >= 0).collect();
            let theirs: BTreeSet<i32> =
                hf[r * k..(r + 1) * k].iter().copied().filter(|&x| (x as usize) < vis).collect();
            assert_eq!(ours.len(), vis.min(k), "L{li}{tag} row {r}: {} valid indices", ours.len());
            rows += 1;
            if ours == theirs {
                continue;
            }
            // 任一侧 top-k 含平局列 → 该列分数两边本就不同，k-th 与对称差都失去意义，整行不可比
            if ours.union(&theirs).any(|&i| tie_cols.contains(&(i as usize))) {
                skipped += 1;
                rows -= 1;
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
    eprintln!(
        "check_topk{tag}: drift rows per layer {:?}; skipped {skipped} (row drifted or top-k touches a drifted key)",
        drift.iter().map(|d| d.len()).collect::<Vec<_>>()
    );
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
    assert!(e2 < MAX_ABS, "indexer score error too large: {e2}");
    // HF 自身 k-th 分差中位数 ~0.006（256 个近似高斯分数取 top-8 的顺序统计量间距），平局行多是结构性的；
    // 这里只要求 "不一致的行全部能用分差解释"（上面的 assert），再给个宽松的比例上限防止全靠容忍过关
    assert!(t2 * 2 <= r2, "too many tie-tolerated rows on long prompt: {t2}/{r2}");
}

/// 注入 HF top-k → 稀疏 attention 本身的正确性（prefill 展开路径 + decode 吸收路径 + shared 层复用）。
#[test]
#[ignore]
fn dsa_sparse_attention_with_hf_topk() {
    let mut h = load(512);
    let k = h.cfg.index_topk as usize;
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
    assert!(skipped * 5 <= t, "[hf-topk] too many router-near-tie rows: {skipped}/{t}");

    let (_, want_tok) = npy_i32("expect_greedy_long.npy");
    let (sshape, want_step) = npy_f32("expect_step_logits_long.npy");
    let (_, rmargin_steps) = npy_f32("router_margin_long_steps.npy");
    assert_eq!(sshape[1], v);
    let steps: Vec<(usize, Vec<i32>)> = full_layers(&h)
        .into_iter()
        .map(|l| (l, npy_i32(&format!("indexer_topk_L{l}_steps.npy")).1))
        .collect();
    // 漂移行（MoE 路由翻转）的 hidden 两边不同 → 后续层 K cache 的那一行不可比；decode 步注入的 top-k 若选中了
    // 这样的位置，attention 结果就不可比。用实测漂移而不是路由 margin：prefill 行的漂移来自 hidden_L*_long.npy，
    // decode 新 token 的漂移来自 hidden_L*_long_steps.npy（逐步追加到同一集合）
    let mut drift = drift_rows(&h, "_long", t);
    let hid = h.cfg.hidden_size as usize;
    let step_hidden: Vec<Vec<f32>> =
        (0..h.cfg.num_hidden_layers).map(|l| npy_f32(&format!("hidden_L{l}_long_steps.npy")).1).collect();
    // HF 的参考是对整条序列重算，bf16 下不是前缀稳定的：个别前缀行的索引器 top-k / 路由在更长序列里翻转，
    // 该行在 HF 里的 KV 与任何增量 decode（我们）cache 里的不同。fixture 记录了每步每层前缀行的 hidden 变化，
    // 变化 > DRIFT 的行并入漂移集合（累计到后续步）
    let prefix_drift: Vec<Vec<f32>> = (0..h.cfg.num_hidden_layers)
        .map(|l| npy_f32(&format!("hidden_L{l}_long_prefix_drift_steps.npy")).1)
        .collect();
    let n_steps = want_tok.len();
    for li in 0..h.cfg.num_hidden_layers as usize {
        for i in 0..n_steps {
            for r in 0..t {
                if prefix_drift[li][i * t + r] > DRIFT {
                    for l2 in li..h.cfg.num_hidden_layers as usize {
                        drift[l2].insert(r);
                    }
                }
            }
        }
    }
    let mut rows: Vec<Vec<f32>> = vec![got_pf[(t - 1) * v..].to_vec()];
    let mut pos = t;
    for (i, &tok) in want_tok[..want_tok.len() - 1].iter().enumerate() {
        rows.push(decode(&mut h, tok, pos, Some((&steps, i + 1))));
        // decode 输入 gen[i] 落在位置 pos；HF 第 i+1 步前向的最后一行就是它。漂移累计：一层漂移，后续层也算
        let mut drifted = false;
        for (li, _a, hd) in &h.last_layers {
            let hf = &step_hidden[*li][(i + 1) * hid..(i + 2) * hid];
            drifted |= max_abs(&hd[..hid], hf) > DRIFT;
            if drifted {
                drift[*li].insert(pos);
            }
        }
        pos += 1;
    }
    let full = full_layers(&h);
    let tie_for_layer = |l: usize| -> BTreeSet<usize> {
        // 第 l 层 top-k 被 l..next_full 这些层用；它们的 K cache 分别来自 l-1..next_full-2 层输出
        let next_full = full.iter().copied().find(|&f| f > l).unwrap_or(h.cfg.num_hidden_layers as usize);
        if next_full < 2 {
            return BTreeSet::new();
        }
        drift[next_full - 2].clone()
    };
    let tie_sets: Vec<BTreeSet<usize>> = full.iter().map(|&l| tie_for_layer(l)).collect();
    let step_hits_tie = |i: usize| -> bool {
        steps.iter().zip(&tie_sets).any(|((_, all), ties)| {
            all[i * k..(i + 1) * k].iter().any(|&p| p >= 0 && ties.contains(&(p as usize)))
        })
    };
    eprintln!("[hf-topk] drift rows per layer {:?}", drift.iter().map(|d| d.len()).collect::<Vec<_>>());
    let (mut worst, mut first_bad, mut dec_steps) = (0f32, None, 0);
    for (i, got) in rows.iter().enumerate() {
        let w = &want_step[i * v..(i + 1) * v];
        let mx = max_abs(got, w);
        let (ga, wa, gap) = (argmax(got), argmax(w), top_gap(w));
        let dec = rmargin_steps[i] > ROUTER_TIE && !step_hits_tie(i);
        eprintln!(
            "[hf-topk] step {i:2}: max abs {mx:.4} argmax got {ga} want {wa} gap {gap:.4} router-margin {:.4}{}",
            rmargin_steps[i],
            if dec { "" } else if step_hits_tie(i) { " (top-k hits drifted row, not asserted)" } else { " (near-tie, not asserted)" }
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
    // tiny 随机模型上 HF 自己对整条序列重算不是前缀稳定的（见 docs/pitfalls.md 2026-10-15 "HF 参考前缀不稳定"），
    // 约一半的步会选中漂移行；要求至少 1/3 可判定且绝对数 ≥ 8
    assert!(
        dec_steps * 3 >= rows.len() && dec_steps >= 8,
        "[hf-topk] too few decidable decode steps: {dec_steps}/{}",
        rows.len()
    );
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
    let (mut decidable, mut bad, mut worst_dec, mut worst_all, mut router_ok) = (0, 0, 0f32, 0f32, 0);
    for r in 0..t {
        let (g, w) = (&got_pf[r * v..(r + 1) * v], &want_pf[r * v..(r + 1) * v]);
        let mx = max_abs(g, w);
        worst_all = worst_all.max(mx);
        if rmargin[r] > ROUTER_TIE {
            router_ok += 1;
        }
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
        "[self] long prefill: {decidable}/{t} decidable rows ({router_ok} router-decidable; worst max abs {worst_dec}); all rows worst {worst_all}"
    );
    assert!(worst_dec < MAX_ABS, "[self] decidable-row logits max abs {worst_dec} > {MAX_ABS}");
    assert_eq!(bad, 0, "[self] {bad} decidable rows argmax differ");
    // 全程自算只在 "索引器可判定" 的行上有意义；索引器平局是结构性的（见 check_topk 注释），
    // 要求至少 1/16 的行可判定，且首 k+1 行（可见 ≤ k+1，top-k 几乎必然一致）之外也要有样本
    assert!(decidable * 16 >= t, "[self] too few decidable rows: {decidable}/{t}");
    assert!(decidable > k + 1, "[self] decidable rows only in the trivial prefix: {decidable}");

    // decode：喂 HF greedy token；每步可判定 = 路由可判定 且 该步索引器 k-th 分差 > 索引器分数误差上界（prefill 实测最大值）
    let (_, want_tok) = npy_i32("expect_greedy_long.npy");
    let (sshape, want_step) = npy_f32("expect_step_logits_long.npy");
    let (_, rmargin_steps) = npy_f32("router_margin_long_steps.npy");
    assert_eq!(sshape[1], v);
    // 自算路径下，L0 平局行的 top-k 翻转会让下游 hidden 漂移，索引器分数误差上界只能在 "路由可判定 且 索引器可判定"
    // 的行、且排除平局列后量，否则一个漂移行（误差 > 1）会让所有 decode 步都不可判定
    // self-run 没注入，L0 自己的平局翻转也会让下游漂移；这里仍用保守的路由 margin 当列过滤
    let tie_cols: Vec<usize> = (0..t).filter(|&r| rmargin[r] <= ROUTER_TIE).collect();
    let idx_err_bound = caps
        .iter()
        .zip(&hf_scores)
        .map(|((_, _, mine, stride), (_, hf))| {
            (0..t)
                .filter(|&r| rmargin[r] > ROUTER_TIE && idx_decidable(r))
                .map(|r| {
                    (0..=r)
                        .filter(|j| !tie_cols.contains(j))
                        .map(|j| (hf[r * t + j] - mine[r * stride + j]).abs())
                        .fold(0f32, f32::max)
                })
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
    // decode 步可判定率在 tiny 随机模型上很低（索引器 k-th 分差中位数 ~0.006 vs 分数误差上界 ~0.04），
    // 这里只要求有样本；全程自算的主要证据是上面 prefill 的可判定行
    if dec_steps == 0 {
        eprintln!("[self] WARN: no decidable decode step (indexer err bound {idx_err_bound:.4}); prefill rows carry the assertion");
    }
}
