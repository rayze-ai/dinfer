//! 里程碑 C-2：Kimi-K3 真权重**整层**对拍（第 1 层 = KDA + latent MoE + AttnRes；`docs/loops/k3-moe.md`「C-2」）。
//!
//! 参考：`scripts/verify/gen_k3_layer_refs.py`（vLLM 容器：fla kda + 纯 torch MoE，MXFP4 按 realcheck 口径解包）。
//! 输入 `x_bf16.npy [T, 7168]`（上一层 prefix_sum）、`b0_bf16.npy`（第 0 层写入的块残差）；中间量与输出 f32。
//!
//! 容差（bf16 链 + MXFP4 专家）：整层 `out` cos ≥ 0.999；子块各报 rel / cos；路由 top-16 命中率只在
//! `topk_margin > 2e-3` 的可判定行断言（硬判决口径见 docs/pitfalls.md）。
//!
//! 跑法（B300）：`KIMIK3_DIR=/data/models/moonshotai/Kimi-K3 KIMIK3_REFS=test/kimik3/2026-10-20/layer \
//!   cargo test -p dinfer-kimik3 --features ref-tests --test layer_parity -- --ignored --nocapture`
#![cfg(feature = "ref-tests")]
use dinfer_core::runner::BatchMode;
use dinfer_kimik3::config::KimiK3Config;
use dinfer_kimik3::kda::{KdaChunkWorkspace, KdaMeta};
use dinfer_kimik3::layer::{Attn, Ffn, KimiDecoderLayer, LayerBudget, StepMeta};
use dinfer_kimik3::mla::MlaMeta;
use dinfer_kimik3::weights::{shards_for_layer, KdaShapes, Table};
use dinfer_layers::{bf16, Device, Gemm, Stream, Tensor, WeightSource};
use std::cell::RefCell;
use std::collections::HashSet;
use std::path::PathBuf;

const LAYER: usize = 1;

fn refs_dir() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("KIMIK3_REFS").ok()?);
    Some(if p.is_absolute() { p } else { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join(p) })
}

fn npy_bytes(dir: &PathBuf, name: &str) -> (Vec<usize>, String, Vec<u8>) {
    let bytes = std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    let arr = npyz::NpyFile::new(&bytes[..]).unwrap();
    let shape: Vec<usize> = arr.shape().iter().map(|&d| d as usize).collect();
    let descr = String::from_utf8_lossy(&bytes[10..128.min(bytes.len())]).to_string();
    (shape, descr, bytes)
}

fn npy_f32(dir: &PathBuf, name: &str) -> (Vec<usize>, Vec<f32>) {
    let (shape, descr, bytes) = npy_bytes(dir, name);
    let arr = npyz::NpyFile::new(&bytes[..]).unwrap();
    let v: Vec<f32> = if descr.contains("<u2") {
        arr.into_vec::<u16>().unwrap().into_iter().map(|b| bf16::from_bits(b).to_f32()).collect()
    } else if descr.contains("<f4") {
        arr.into_vec::<f32>().unwrap()
    } else {
        panic!("{name}: 未支持的 descr {descr}")
    };
    (shape, v)
}

fn npy_i32(dir: &PathBuf, name: &str) -> (Vec<usize>, Vec<i32>) {
    let (shape, descr, bytes) = npy_bytes(dir, name);
    assert!(descr.contains("<i4"), "{name}: 期望 int32，descr {descr}");
    (shape, npyz::NpyFile::new(&bytes[..]).unwrap().into_vec::<i32>().unwrap())
}

/// max_err（相对参考的 max|want|）+ cos。
fn compare(name: &str, got: &[f32], want: &[f32]) -> (f32, f64) {
    assert_eq!(got.len(), want.len(), "{name}: 长度");
    let mut max_err = 0f32;
    let mut worst = 0usize;
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        let e = (a - b).abs();
        if e > max_err {
            max_err = e;
            worst = i;
        }
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64) * (*a as f64);
        nb += (*b as f64) * (*b as f64);
    }
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
    let cos = if na > 0.0 && nb > 0.0 { dot / (na.sqrt() * nb.sqrt()) } else { 1.0 };
    let mean_abs = got.iter().zip(want).map(|(a, b)| (a - b).abs()).sum::<f32>() / got.len() as f32;
    println!(
        "  {name:<22} max_err {max_err:.4e} (rel {:.3e}, at {worst}: {:.5} vs {:.5})  mean_abs {mean_abs:.3e}  cos {cos:.7}",
        max_err / scale,
        got[worst],
        want[worst]
    );
    (max_err / scale, cos)
}

fn rows(t: &Tensor<bf16>, s: &Stream, n: usize) -> Vec<f32> {
    t.to_vec_f32(s).unwrap()[..n].to_vec()
}

#[test]
#[ignore]
fn layer1_real_weights() {
    let Some(refs) = refs_dir() else {
        eprintln!("skip: KIMIK3_REFS 未设");
        return;
    };
    let Ok(dir) = std::env::var("KIMIK3_DIR").map(PathBuf::from) else {
        eprintln!("skip: KIMIK3_DIR 未设");
        return;
    };
    let (xs, x) = npy_f32(&refs, "x_bf16.npy");
    let (_, b0) = npy_f32(&refs, "b0_bf16.npy");
    let (t, hidden) = (xs[0], xs[1]);
    let cfg = KimiK3Config::from_json(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let tc = &cfg.text_config;
    let sh = KdaShapes::new(tc);
    assert_eq!(hidden, sh.hidden);

    let dev = Device::new(0).unwrap();
    let s = dev.default_stream();
    let gemm = Gemm::new(&s).unwrap();
    let files = shards_for_layer(&dir, LAYER).unwrap();
    println!(
        "  第 {LAYER} 层分片：{:?}",
        files.iter().map(|f| f.file_name().unwrap().to_string_lossy().to_string()).collect::<Vec<_>>()
    );
    let t0 = std::time::Instant::now();
    let src = WeightSource::open_files(&files).unwrap();
    let table = Table::from_names(&src.names().collect::<Vec<_>>()).unwrap();
    let (free0, _) = s.mem_info().unwrap();
    let mut layer = KimiDecoderLayer::load(
        &s,
        &src,
        &table,
        LAYER,
        tc,
        LayerBudget { max_tokens: t, num_slots: 1, kv_blocks: 1, block_size: 16 },
    )
    .unwrap();
    s.synchronize().unwrap();
    let (free1, _) = s.mem_info().unwrap();
    println!("  装载 {:.1} s，显存 +{:.2} GB", t0.elapsed().as_secs_f32(), (free0 - free1) as f64 / 1e9);
    if let Ffn::LatentMoe(m) = &layer.ffn {
        println!(
            "  专家 packed 常驻 {:.3} GB（{} 专家 × {:.2} MB），每 token 专家 GEMM {} 次（grouped 2 条 kernel）",
            m.experts.bytes as f64 / 1e9,
            m.experts.num_experts,
            m.experts.bytes as f64 / m.experts.num_experts as f64 / 1e6,
            m.gemms_per_token()
        );
    }
    assert!(!layer.writes_block(), "第 1 层不是写块层");

    let to16 = |v: &[f32]| v.iter().map(|&f| bf16::from_f32(f)).collect::<Vec<_>>();
    let xd = Tensor::from_vec(&s, to16(&x), &[t, hidden]).unwrap();
    let b0d = Tensor::from_vec(&s, to16(&b0), &[t, hidden]).unwrap();
    let mut out = Tensor::<bf16>::zeros(&s, &[t, hidden]).unwrap();
    let cu_host = vec![0i32, t as i32];
    let cu = Tensor::from_vec(&s, cu_host.clone(), &[2]).unwrap();
    let si = Tensor::from_vec(&s, vec![0i32], &[1]).unwrap();
    let hi = Tensor::from_vec(&s, vec![0i32], &[1]).unwrap();
    let ws = RefCell::new(KdaChunkWorkspace::new());
    ws.borrow_mut().prepare(&s, &cu_host, &sh).unwrap();
    let kda = KdaMeta {
        cu_seqlens: &cu,
        state_indices: &si,
        has_initial: &hi,
        num_reqs: 1,
        mode: BatchMode::Prefill,
        chunk: Some(&ws),
    };
    // 第 1 层是 KDA 层，MLA 侧 meta 只是占位
    let slots = Tensor::from_vec(&s, vec![-1i64; t], &[t]).unwrap();
    let meta = StepMeta {
        kda,
        mla: MlaMeta {
            mode: BatchMode::Prefill,
            cu_seqlens: &cu,
            block_table: &si,
            seq_lens: &si,
            num_reqs: 1,
            max_blocks: 0,
        },
        slot_mapping: &slots,
    };

    let run = |layer: &mut KimiDecoderLayer, out: &mut Tensor<bf16>| {
        match &mut layer.attn {
            Attn::Kda(k) => k.reset_slot(&s, 0).unwrap(),
            Attn::Mla(_) => panic!("第 1 层应是 KDA 层"),
        }
        let t0 = std::time::Instant::now();
        layer.forward(&s, &gemm, t, &xd, &[&b0d], &meta, out).unwrap();
        s.synchronize().unwrap();
        t0.elapsed().as_secs_f32() * 1e3
    };
    let ms0 = run(&mut layer, &mut out);
    let n = t * hidden;

    println!("== 第 {LAYER} 层 T={t} 整层对拍");
    let (_, h_ref) = npy_f32(&refs, "h.npy");
    let (_, a_ref) = npy_f32(&refs, "a.npy");
    let (_, p_ref) = npy_f32(&refs, "p.npy");
    let (_, m_ref) = npy_f32(&refs, "m.npy");
    let (_, mn_ref) = npy_f32(&refs, "mn.npy");
    let (_, out_ref) = npy_f32(&refs, "out.npy");
    let (rel_h, cos_h) = compare("attn 门 h vs HF", &rows(layer.last_attn_in(), &s, n), &h_ref);
    let (rel_a, cos_a) = compare("attn out vs fla", &rows(layer.last_attn_out(), &s, n), &a_ref);
    compare("prefix p vs HF", &rows(layer.last_prefix_sum(), &s, n), &p_ref);
    let (rel_m, cos_m) = compare("mlp 门 m vs HF", &rows(layer.last_mlp_gate_out(), &s, n), &m_ref);
    compare("post_norm mn vs HF", &rows(layer.last_mlp_in(), &s, n), &mn_ref);

    // MoE 子块
    let Ffn::LatentMoe(moe) = &layer.ffn else { panic!("第 1 层应是 MoE 层") };
    let e = moe.router.num_experts;
    let kk = moe.router.topk;
    let (_, logits_ref) = npy_f32(&refs, "logits.npy");
    let (_, ids_ref) = npy_i32(&refs, "topk_ids.npy");
    let (_, margin) = npy_f32(&refs, "topk_margin.npy");
    let (_, ysum_ref) = npy_f32(&refs, "ysum.npy");
    let (_, yrouted_ref) = npy_f32(&refs, "yrouted.npy");
    let (_, ymoe_ref) = npy_f32(&refs, "y_moe.npy");
    println!("== MoE 子块");
    let (rel_lg, cos_lg) =
        compare("router logits vs HF", &moe.last_logits(&s, t).unwrap(), &logits_ref[..t * e]);
    let (ids, _w) = moe.last_routing(&s, t).unwrap();
    let thresh = 2e-3f32;
    let (mut decidable, mut hit_rows, mut hit_pairs, mut total_pairs) = (0usize, 0usize, 0usize, 0usize);
    let mut matched = Vec::new(); // 路由整行一致的 token
    let mut tie_mismatch = 0usize; // 路由不一致且 margin ≤ thresh（不可判定的平局，预期）
    for i in 0..t {
        let got: HashSet<i32> = ids[i * kk..(i + 1) * kk].iter().cloned().collect();
        let want: HashSet<i32> = ids_ref[i * kk..(i + 1) * kk].iter().cloned().collect();
        let inter = got.intersection(&want).count();
        total_pairs += kk;
        hit_pairs += inter;
        if inter == kk {
            matched.push(i);
        } else if margin[i] <= thresh {
            tie_mismatch += 1;
        }
        if margin[i] > thresh {
            decidable += 1;
            if inter == kk {
                hit_rows += 1;
            }
        }
    }
    println!(
        "  top-{kk} 命中：pair 级 {hit_pairs}/{total_pairs}（{:.2}%）；可判定行（margin>{thresh:.0e}）{decidable}/{t}，其中整行全中 {hit_rows}；路由整行一致 {}/{t}（不一致行全是平局：{tie_mismatch}）",
        100.0 * hit_pairs as f64 / total_pairs as f64,
        matched.len()
    );
    // 路由在平局行选了不同专家 → 该行专家输出按定义不同（硬判决不可逐位复现），MoE 子块的数值容差只在路由一致的行上判。
    let sel = |v: &[f32], w: usize| -> Vec<f32> {
        matched.iter().flat_map(|&i| v[i * w..(i + 1) * w].iter().cloned()).collect()
    };
    let latent = moe.experts.latent;
    let ysum_got = rows(moe.last_expert_sum(), &s, t * latent);
    let yr_got = rows(moe.last_routed_out(), &s, n);
    let ym_got = rows(layer.last_mlp_out(), &s, n);
    compare("ysum（全部行）", &ysum_got, &ysum_ref);
    compare("routed up（全部行）", &yr_got, &yrouted_ref);
    compare("MoE out（全部行）", &ym_got, &ymoe_ref);
    println!("  —— 下面三行只取路由一致的 {} 行 ——", matched.len());
    let (rel_ys, cos_ys) = compare("专家加权和 ysum", &sel(&ysum_got, latent), &sel(&ysum_ref, latent));
    let (rel_yr, cos_yr) = compare("routed up_proj", &sel(&yr_got, hidden), &sel(&yrouted_ref, hidden));
    let (rel_ym, cos_ym) = compare("MoE out（+shared）", &sel(&ym_got, hidden), &sel(&ymoe_ref, hidden));
    println!("== 整层");
    let (rel_o, cos_o) = compare("layer out vs HF", &rows(&out, &s, n), &out_ref);

    // 计时（已装载、已 warm）：5 次均值
    let mut ms = Vec::new();
    for _ in 0..5 {
        ms.push(run(&mut layer, &mut out));
    }
    let mean = ms.iter().sum::<f32>() / ms.len() as f32;
    println!("  单层 T={t} 耗时：首轮 {ms0:.1} ms，之后 5 次均值 {mean:.1} ms（含 2 次 D2H 同步 + AttnRes host softmax）");

    // 断言
    assert!(cos_h >= 0.999 && rel_h <= 5e-2, "attn 门 h：rel {rel_h:.3e} cos {cos_h:.6}");
    assert!(cos_a >= 0.999 && rel_a <= 5e-2, "attn out：rel {rel_a:.3e} cos {cos_a:.6}");
    assert!(cos_m >= 0.999 && rel_m <= 5e-2, "mlp 门 m：rel {rel_m:.3e} cos {cos_m:.6}");
    assert!(cos_lg >= 0.999 && rel_lg <= 2e-2, "router logits：rel {rel_lg:.3e} cos {cos_lg:.6}");
    assert!(decidable == 0 || hit_rows == decidable, "可判定行 top-{kk} 应全中：{hit_rows}/{decidable}");
    assert!(matched.len() + tie_mismatch == t, "路由不一致的行必须都是平局（margin ≤ {thresh:.0e}）");
    assert!(matched.len() * 2 >= t, "路由一致行太少：{}/{t}", matched.len());
    assert!(cos_ys >= 0.999 && rel_ys <= 5e-2, "ysum：rel {rel_ys:.3e} cos {cos_ys:.6}");
    assert!(cos_yr >= 0.999 && rel_yr <= 5e-2, "routed up：rel {rel_yr:.3e} cos {cos_yr:.6}");
    assert!(cos_ym >= 0.999 && rel_ym <= 5e-2, "MoE out：rel {rel_ym:.3e} cos {cos_ym:.6}");
    assert!(cos_o >= 0.999 && rel_o <= 5e-2, "layer out：rel {rel_o:.3e} cos {cos_o:.6}");
}
