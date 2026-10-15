//! P5 诊断（不是验收）：索引器分数误差的分布 —— 按层、按位置分桶、按 rope/pass 维无法拆（分数已求和），
//! 但能看出误差是否随位置增长（rope 表 / 位置问题）、是否集中在个别行（bf16 舍入放大）、是否某层独大。
//! 跑法：TINY_GLM_DIR=$(realpath test/tiny_glm_dsa) cargo test -p dinfer-glm53 --features ref-tests --test dsa_diag -- --ignored --nocapture
#![cfg(feature = "ref-tests")]

use dinfer_glm53::config::GlmMoeDsaConfig;
use dinfer_glm53::model::{Forward, Glm53Model};
use dinfer_layers::mla::AttnMeta;
use dinfer_layers::{Device, Gemm, Tensor, WeightSource};
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

const BLOCK: usize = 16;
const NUM_BLOCKS: usize = 64;

fn profile(inject: bool) {
    let dev = Device::new(0).unwrap();
    let s = dev.default_stream();
    let gemm = Gemm::new(&s).unwrap();
    let cfg =
        GlmMoeDsaConfig::from_json(&std::fs::read_to_string(dir().join("config.json")).unwrap()).unwrap();
    let src = WeightSource::open_dir(&dir()).unwrap();
    let mut model = Glm53Model::load(&s, &src, cfg.clone(), 512, NUM_BLOCKS, BLOCK).unwrap();
    let (_, prompt) = npy_i32("prompt_long.npy");
    let t = prompt.len();
    let v = cfg.vocab_size as usize;
    let ids = Tensor::from_vec(&s, prompt.clone(), &[t]).unwrap();
    let pos = Tensor::from_vec(&s, (0..t as i32).collect(), &[t]).unwrap();
    let slots = Tensor::from_vec(&s, (0..t as i64).collect(), &[t]).unwrap();
    let cu = Tensor::from_vec(&s, vec![0i32, t as i32], &[2]).unwrap();
    let bt = Tensor::from_vec(&s, (0..NUM_BLOCKS as i32).collect(), &[1, NUM_BLOCKS]).unwrap();
    let seq = Tensor::from_vec(&s, vec![t as i32], &[1]).unwrap();
    let rows_t = Tensor::from_vec(&s, (0..t as i32).collect(), &[t]).unwrap();
    let mut logits = Tensor::<f32>::zeros(&s, &[t, v]).unwrap();
    model.capture_topk = Some(Vec::new());
    model.topk_override = if inject {
        Some(
            (0..cfg.num_hidden_layers)
                .filter(|&l| cfg.has_indexer(l))
                .map(|l| (l as usize, npy_i32(&format!("indexer_topk_L{l}_long.npy")).1))
                .collect(),
        )
    } else {
        None
    };
    eprintln!(
        "=== mode: {} ===",
        if inject {
            "HF top-k injected (x per layer == HF up to attn error)"
        } else {
            "self-run (x per layer drifts when upstream top-k differs)"
        }
    );
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
    model.capture_layers = Some(Vec::new());
    model.capture_routing = Some(Vec::new());
    model.forward(&s, &gemm, &f, &mut logits).unwrap();
    model.topk_override = None;
    let caps = model.capture_topk.take().unwrap();
    if inject {
        for (li, ids, w) in model.capture_routing.take().unwrap() {
            let kk = ids.len() / t;
            let txt: Vec<String> = (0..t)
                .map(|r| {
                    let a: Vec<String> =
                        (0..kk).map(|j| format!("{} {:.6}", ids[r * kk + j], w[r * kk + j])).collect();
                    a.join(" ")
                })
                .collect();
            std::fs::write(format!("/tmp/dsa_ours_routing_L{li}.txt"), txt.join("\n")).unwrap();
        }
    }
    // 逐层 bisect：HF 侧 dump 在 /tmp/dsa_hf_layers/{attn,hidden}_L{li}.npy（gen 之外的临时脚本产出）
    let hid = cfg.hidden_size as usize;
    for (li, a, hd) in model.capture_layers.take().unwrap() {
        let rd = |n: &str| -> Option<Vec<f32>> {
            let b = std::fs::read(format!("/tmp/dsa_hf_layers/{n}_L{li}.npy")).ok()?;
            npyz::NpyFile::new(&b[..]).ok()?.into_vec::<f32>().ok()
        };
        if let (Some(ha), Some(hh)) = (rd("attn"), rd("hidden")) {
            let worst_row = |x: &[f32], y: &[f32]| -> (f32, usize) {
                (0..t)
                    .map(|r| {
                        ((0..hid).map(|c| (x[r * hid + c] - y[r * hid + c]).abs()).fold(0f32, f32::max), r)
                    })
                    .fold((0f32, 0), |acc, e| if e.0 > acc.0 { e } else { acc })
            };
            let (ea, ra) = worst_row(&a, &ha);
            let (eh, rh) = worst_row(&hd, &hh);
            let n_bad_h = (0..t)
                .filter(|&r| (0..hid).any(|c| (hd[r * hid + c] - hh[r * hid + c]).abs() > 0.05))
                .count();
            eprintln!("  layer {li}: attn_out max abs {ea:.4} (row {ra}); hidden-after-layer max abs {eh:.4} (row {rh}), {n_bad_h} rows > 0.05");
            if inject {
                // 也落盘我们的 attn_out / hidden（f32），供 python 侧按行拆 attn vs mlp
                let dump = |n: &str, v: &[f32]| {
                    let txt: Vec<String> = v[..t * hid].iter().map(|e| format!("{e:.6}")).collect();
                    std::fs::write(format!("/tmp/dsa_ours_{n}_L{li}.txt"), txt.join("\n")).unwrap();
                };
                dump("attn", &a);
                dump("hidden", &hd);
                let per_row: Vec<f32> = (0..t)
                    .map(|r| (0..hid).map(|c| (hd[r * hid + c] - hh[r * hid + c]).abs()).fold(0f32, f32::max))
                    .collect();
                let txt: Vec<String> = per_row.iter().map(|e| format!("{e:.5}")).collect();
                std::fs::write(format!("/tmp/dsa_row_drift_L{li}.txt"), txt.join("\n")).unwrap();
            }
        }
    }

    for (li, _topk, mine, stride) in &caps {
        let (_, hf) = npy_f32(&format!("indexer_score_L{li}_long.npy"));
        // 每行：max abs err、HF 行内 |score| 最大值、误差最大的列 j 及其位置距离 (r - j)
        let mut row_err = vec![0f32; t];
        let mut worst = (0f32, 0usize, 0usize, 0f32, 0f32);
        let mut col_hist = [0f32; 8]; // 误差按 (r-j) 距离分 8 桶
        let mut col_cnt = [0usize; 8];
        let mut pos_hist = [0f32; 8]; // 行误差按 r 分 8 桶（取桶内最大）
        for r in 0..t {
            let vis = r + 1;
            let hr = &hf[r * t..r * t + vis];
            let mr = &mine[r * stride..r * stride + vis];
            for j in 0..vis {
                let e = (hr[j] - mr[j]).abs();
                if e > row_err[r] {
                    row_err[r] = e;
                }
                if e > worst.0 {
                    worst = (e, r, j, hr[j], mr[j]);
                }
                let b = ((r - j) * 8 / t).min(7);
                col_hist[b] += e;
                col_cnt[b] += 1;
            }
            let pb = r * 8 / t;
            if row_err[r] > pos_hist[pb] {
                pos_hist[pb] = row_err[r];
            }
        }
        let mut sorted = row_err.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n_big = row_err.iter().filter(|&&e| e > 0.05).count();
        eprintln!(
            "L{li}: row-err median {:.4} p90 {:.4} max {:.4} ({} rows > 0.05); worst at row {} col {} (dist {}): hf {:.4} vs ours {:.4}",
            sorted[t / 2], sorted[t * 9 / 10], worst.0, n_big, worst.1, worst.2, worst.1 - worst.2, worst.3, worst.4
        );
        eprintln!(
            "L{li}: max row-err by position bucket (r/32): {}",
            pos_hist.iter().map(|e| format!("{e:.3}")).collect::<Vec<_>>().join(" ")
        );
        eprintln!(
            "L{li}: mean cell-err by query-key distance bucket ((r-j)/32): {}",
            col_hist
                .iter()
                .zip(&col_cnt)
                .map(|(e, c)| format!("{:.4}", if *c > 0 { e / *c as f32 } else { 0.0 }))
                .collect::<Vec<_>>()
                .join(" ")
        );
        // 误差最大的 6 行
        let mut idx: Vec<usize> = (0..t).collect();
        idx.sort_by(|&a, &b| row_err[b].partial_cmp(&row_err[a]).unwrap());
        eprintln!(
            "L{li}: top rows: {}",
            idx[..6].iter().map(|&r| format!("r{r}={:.3}", row_err[r])).collect::<Vec<_>>().join(" ")
        );
    }
}

#[test]
#[ignore]
fn dsa_diag_indexer_score_error_profile() {
    profile(true);
    profile(false);
}

/// decode 第 N 步逐层 bisect（注入 HF top-k）：HF 侧 `/tmp/dsa_hf_dec6/{attn,hidden}_L{li}_step6.npy`（最后一行）。
#[test]
#[ignore]
fn dsa_diag_decode_step6_layer_bisect() {
    let dev = Device::new(0).unwrap();
    let s = dev.default_stream();
    let gemm = Gemm::new(&s).unwrap();
    let cfg =
        GlmMoeDsaConfig::from_json(&std::fs::read_to_string(dir().join("config.json")).unwrap()).unwrap();
    let src = WeightSource::open_dir(&dir()).unwrap();
    let mut model = Glm53Model::load(&s, &src, cfg.clone(), 512, NUM_BLOCKS, BLOCK).unwrap();
    let (_, prompt) = npy_i32("prompt_long.npy");
    let t = prompt.len();
    let v = cfg.vocab_size as usize;
    let k = cfg.index_topk as usize;
    let hid = cfg.hidden_size as usize;
    let full: Vec<usize> =
        (0..cfg.num_hidden_layers).filter(|&l| cfg.has_indexer(l)).map(|l| l as usize).collect();
    let bt = Tensor::from_vec(&s, (0..NUM_BLOCKS as i32).collect(), &[1, NUM_BLOCKS]).unwrap();
    // prefill with HF top-k
    {
        let ids = Tensor::from_vec(&s, prompt.clone(), &[t]).unwrap();
        let pos = Tensor::from_vec(&s, (0..t as i32).collect(), &[t]).unwrap();
        let slots = Tensor::from_vec(&s, (0..t as i64).collect(), &[t]).unwrap();
        let cu = Tensor::from_vec(&s, vec![0i32, t as i32], &[2]).unwrap();
        let seq = Tensor::from_vec(&s, vec![t as i32], &[1]).unwrap();
        let rows_t = Tensor::from_vec(&s, (0..t as i32).collect(), &[t]).unwrap();
        let mut logits = Tensor::<f32>::zeros(&s, &[t, v]).unwrap();
        model.topk_override =
            Some(full.iter().map(|&l| (l, npy_i32(&format!("indexer_topk_L{l}_long.npy")).1)).collect());
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
        model.forward(&s, &gemm, &f, &mut logits).unwrap();
        // prefill 刚结束：L0/L1/L2 latent cache 全量落盘（[256, entry]），供 python 与 HF kv_a_layernorm 对比
        for li in 0..3 {
            let kvc = &model.kv[li];
            let entry = kvc.cache.numel() / (NUM_BLOCKS * BLOCK);
            let all = kvc.cache.to_vec_f32(&s).unwrap();
            let txt: Vec<String> = (0..t)
                .map(|r| {
                    all[r * entry..(r + 1) * entry]
                        .iter()
                        .map(|x| format!("{x:.6}"))
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .collect();
            std::fs::write(format!("/tmp/dsa_ours_kvcache_prefill_L{li}.txt"), txt.join("\n")).unwrap();
        }
    }
    let (_, want_tok) = npy_i32("expect_greedy_long.npy");
    let (sshape, want_step) = npy_f32("expect_step_logits_long.npy");
    assert_eq!(sshape[1], v);
    let steps: Vec<(usize, Vec<i32>)> =
        full.iter().map(|&l| (l, npy_i32(&format!("indexer_topk_L{l}_steps.npy")).1)).collect();
    let target: usize = std::env::var("STEP").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
    let mut pos_i = t;
    for (i, &tok) in want_tok.iter().enumerate().take(target) {
        let ids = Tensor::from_vec(&s, [tok].to_vec(), &[1]).unwrap();
        let positions = Tensor::from_vec(&s, vec![pos_i as i32], &[1]).unwrap();
        let slots = Tensor::from_vec(&s, vec![pos_i as i64], &[1]).unwrap();
        let seq = Tensor::from_vec(&s, vec![pos_i as i32 + 1], &[1]).unwrap();
        let rows_t = Tensor::from_vec(&s, vec![0i32], &[1]).unwrap();
        let mut logits = Tensor::<f32>::zeros(&s, &[1, v]).unwrap();
        let step_i = i + 1; // decode input gen[i] produces step i+1 logits
        model.topk_override =
            Some(steps.iter().map(|(l, all)| (*l, all[step_i * k..(step_i + 1) * k].to_vec())).collect());
        model.capture_layers = if step_i == target { Some(Vec::new()) } else { None };
        model.capture_attn_pre = if step_i == target { Some(Vec::new()) } else { None };
        let f = Forward {
            input_ids: &ids,
            positions: &positions,
            slot_mapping: &slots,
            meta: AttnMeta::Decode { block_table: &bt, seq_lens: &seq, max_blocks: NUM_BLOCKS },
            logits_rows: &rows_t,
        };
        model.forward(&s, &gemm, &f, &mut logits).unwrap();
        pos_i += 1;
        if step_i == target {
            let got = logits.to_vec(&s).unwrap();
            let w = &want_step[step_i * v..(step_i + 1) * v];
            let mx = got.iter().zip(w).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            eprintln!("decode step {step_i}: logits max abs {mx:.4}");
            // 共享 top-k 缓冲此刻的内容（最后一个 full 层 L2 写入/注入的）vs 注入值
            let buf = model.topk.as_ref().unwrap().to_vec(&s).unwrap();
            for (l, all) in &steps {
                let inj = &all[step_i * k..(step_i + 1) * k];
                eprintln!("  injected L{l} step top-k: {:?}", inj);
            }
            eprintln!("  shared topk buffer row0 after forward: {:?}", &buf[..k]);
            // L1 latent cache（slot = position；block_table 恒等）rows 254..=pos_i-1 → /tmp/dsa_ours_kvcache_L1.txt
            {
                let kvc = &model.kv[1];
                let entry = kvc.cache.numel() / (NUM_BLOCKS * BLOCK);
                let all = kvc.cache.to_vec_f32(&s).unwrap();
                let lo = 250usize;
                let hi = pos_i; // exclusive; pos_i already advanced past the token just written
                let mut txt = Vec::new();
                // 也带上 L0 注入 top-k 选中的 8 个 key 行（L1 复用它们）
                let (_, l0_all) = &steps[0];
                let sel: Vec<usize> = l0_all[step_i * k..(step_i + 1) * k]
                    .iter()
                    .filter(|&&p| p >= 0)
                    .map(|&p| p as usize)
                    .collect();
                for &r in &sel {
                    let row: Vec<String> =
                        all[r * entry..(r + 1) * entry].iter().map(|x| format!("{x:.6}")).collect();
                    txt.push(format!("{r} {}", row.join(" ")));
                }
                for r in lo..hi {
                    let row: Vec<String> =
                        all[r * entry..(r + 1) * entry].iter().map(|x| format!("{x:.6}")).collect();
                    txt.push(format!("{r} {}", row.join(" ")));
                }
                std::fs::write("/tmp/dsa_ours_kvcache_L1.txt", txt.join("\n")).unwrap();
                eprintln!("  dumped L1 cache rows {lo}..{hi} (entry {entry})");
                let (ql, ol) = model.layers[1].attn.last_absorbed(&s, 1).unwrap();
                let f = |v: &[f32]| v.iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join(" ");
                std::fs::write("/tmp/dsa_ours_qlat_L1.txt", f(&ql)).unwrap();
                std::fs::write("/tmp/dsa_ours_olat_L1.txt", f(&ol)).unwrap();
                let hh = cfg.num_attention_heads as usize;
                let (qa, oa) = (ql.len() / hh, ol.len() / hh);
                eprintln!("  dumped L1 q_lat [{hh},{qa}] and o_lat [{hh},{oa}]");
            }
            let (hh, vv) = (cfg.num_attention_heads as usize, cfg.v_head_dim as usize);
            for (li, pre) in model.capture_attn_pre.take().unwrap() {
                let b = std::fs::read(format!("/tmp/dsa_hf_dec6/preproj_L{li}_step{target}.npy")).ok();
                if let Some(b) = b {
                    let hf = npyz::NpyFile::new(&b[..]).unwrap().into_vec::<f32>().unwrap();
                    let per_head: Vec<String> = (0..hh)
                        .map(|hd| {
                            let e = (0..vv)
                                .map(|c| (pre[hd * vv + c] - hf[hd * vv + c]).abs())
                                .fold(0f32, f32::max);
                            let mag = (0..vv).map(|c| hf[hd * vv + c].abs()).fold(0f32, f32::max);
                            format!("h{hd}:{e:.4}/{mag:.3}")
                        })
                        .collect();
                    eprintln!("  layer {li} pre-o_proj err/|hf| per head: {}", per_head.join(" "));
                }
            }
            for (li, a, hd) in model.capture_layers.take().unwrap() {
                let rd = |n: &str| -> Option<Vec<f32>> {
                    let b = std::fs::read(format!("/tmp/dsa_hf_dec6/{n}_L{li}_step{target}.npy")).ok()?;
                    npyz::NpyFile::new(&b[..]).ok()?.into_vec::<f32>().ok()
                };
                if let (Some(ha), Some(hh), Some(hm)) = (rd("attn"), rd("hidden"), rd("mlp")) {
                    let ea = a[..hid].iter().zip(&ha).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                    let eh = hd[..hid].iter().zip(&hh).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                    // ours mlp = hidden_after - hidden_prev - attn（prev 用 HF 的，误差上界内）
                    let hprev: Vec<f32> = if li == 0 {
                        // 没 dump embedding；用 HF hidden_L0 - attn_L0 - mlp_L0 反推
                        (0..hid).map(|c| hh[c] - ha[c] - hm[c]).collect()
                    } else {
                        let b =
                            std::fs::read(format!("/tmp/dsa_hf_dec6/hidden_L{}_step{target}.npy", li - 1))
                                .unwrap();
                        npyz::NpyFile::new(&b[..]).unwrap().into_vec::<f32>().unwrap()
                    };
                    let em = (0..hid).map(|c| ((hd[c] - hprev[c] - a[c]) - hm[c]).abs()).fold(0f32, f32::max);
                    let hm_max = hm.iter().fold(0f32, |m, x| m.max(x.abs()));
                    eprintln!("  layer {li}: attn_out err {ea:.4}; mlp_out err {em:.4} (|mlp| max {hm_max:.3}); hidden-after err {eh:.4}");
                }
            }
        }
    }
    model.topk_override = None;
}
