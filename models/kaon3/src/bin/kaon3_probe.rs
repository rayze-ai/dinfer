//! Kaon-V3 端到端探针：加载真实 checkpoint → 前向一次 → 与 vLLM oracle 逐位置比对。
//!
//! 用法：
//!   kaon3_probe <model_dir>                                   # 权重格式探针（老行为）
//!   kaon3_probe <model_dir> --forward <ids...>                # 前向一次，打印最后一行 top-5
//!   kaon3_probe <model_dir> --forward --oracle x.json <ids...> # 与 oracle 比对，PASS 退出 0、FAIL 退出 3
//!   kaon3_probe <model_dir> --forward --dump-layer 5 <ids...>
//!   kaon3_probe <model_dir> --forward --layers 6 --oracle L6.json <ids...>   # 截层二分（oracle 用同样截层的 vLLM）
//!   kaon3_probe <model_dir> --batch test/kaon3/oracle --out test/kaon3/ours   # 加载一次跑完目录里全部 <name>.ids
//!
//! oracle = 基线 vLLM `/v1/completions` 的原始响应（`prompt_logprobs:5` + `logprobs:5` +
//! `return_tokens_as_token_ids`，由 `scripts/dev/kaon3_oracle.sh` 抓取）。第 r 行 logits 对应
//! `prompt_logprobs[r+1]`（预测第 r+1 个 prompt token），最后一行对应 `logprobs.top_logprobs[0]`。

use anyhow::{bail, ensure, Context, Result};
use dinfer_kernels::{Device, Gemm, Stream};
use dinfer_kaon3::config::KaonConfig;
use dinfer_kaon3::layer::{ActBuf, LayerScratch};
use dinfer_kaon3::model::{check_weights, KaonModel, StepMeta};
use dinfer_layers::{DefaultOps, Tensor, WeightSource};
use std::path::Path;

const BLOCK_SIZE: usize = 32;

/// 比对判据（loop 规格 `docs/loops/kaon3-parity.md` 的停止条件，改这里要同步改规格）：
/// 可判定行（oracle top1 − top2 ≥ MARGIN）top-1 必须一致；oracle top-5 的 |Δlogprob| 均值 ≤ MEAN_TOL；
/// oracle top-1 token 的 |Δlogprob| 最大 ≤ TOP1_TOL。
const MARGIN: f64 = 0.5;
const MEAN_TOL: f64 = 0.15;
const TOP1_TOL: f64 = 0.6;

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(!args.is_empty(), "usage: kaon3_probe <model_dir> [--forward [--oracle f.json] [--dump-layer N] <ids...>]");
    let dir = args.remove(0);
    let src = WeightSource::open_dir(Path::new(&dir))?;

    if args.iter().any(|a| a == "--scales") {
        return scales_probe(&src);
    }
    if !args.iter().any(|a| a == "--forward" || a == "--batch") {
        return format_probe(&src);
    }
    args.retain(|a| a != "--forward");
    let mut take = |flag: &str| -> Option<String> {
        let i = args.iter().position(|a| a == flag)?;
        let v = args.get(i + 1).cloned();
        args.drain(i..(i + 2).min(args.len()));
        v
    };
    let dump_layer = take("--dump-layer").map(|s| s.parse::<usize>().expect("--dump-layer 需要整数"));
    let oracle = take("--oracle");
    let layers = take("--layers").map(|s| s.parse::<usize>().expect("--layers 需要整数"));
    let dump = take("--dump-oracle");
    let hidden_out = take("--hidden-out");
    let hidden_ref = take("--hidden-ref");
    // --batch <目录> --out <目录>：加载一次，把目录里每个 <name>.ids 前向一遍、写 <out>/<name>.json（vLLM 同格式）。
    // 单用例模式每次都要 ~30 s 的权重加载 + RoPE 表构建，verify 30 个用例要 17 分钟；批量模式只付一次。
    if let Some(bdir) = take("--batch") {
        let out = take("--out").context("--batch 需要 --out <目录>")?;
        std::fs::create_dir_all(&out)?;
        let mut names: Vec<String> = std::fs::read_dir(&bdir)?
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter_map(|f| f.strip_suffix(".ids").map(str::to_string))
            .collect();
        names.sort();
        // --shard i/n：只跑排序后下标 ≡ i (mod n) 的用例（多卡并行 verify 用）
        if let Some(s) = take("--shard") {
            let (i, n) = s.split_once('/').context("--shard 格式 i/n")?;
            let (i, n): (usize, usize) = (i.parse()?, n.parse()?);
            names = names.into_iter().enumerate().filter(|(k, _)| k % n == i).map(|(_, v)| v).collect();
        }
        let mut ctx = setup(&dir, &src, layers)?;
        for n in &names {
            let ids: Vec<i32> = std::fs::read_to_string(format!("{bdir}/{n}.ids"))?
                .split_whitespace()
                .map(|s| s.parse::<i32>())
                .collect::<std::result::Result<_, _>>()?;
            let t = std::time::Instant::now();
            forward(&mut ctx, &ids, None, None, Some(&format!("{out}/{n}.json")), (None, None))?;
            println!("BATCH {n}: T={} {:.2} s", ids.len(), t.elapsed().as_secs_f64());
        }
        return Ok(());
    }
    let ids: Vec<i32> = args.iter().map(|s| s.parse::<i32>().expect("token id")).collect();
    ensure!(!ids.is_empty(), "--forward 需要至少一个 token id");
    let mut ctx = setup(&dir, &src, layers)?;
    let pass = forward(&mut ctx, &ids, dump_layer, oracle.as_deref(), dump.as_deref(), (hidden_out.as_deref(), hidden_ref.as_deref()))?;
    if !pass {
        std::process::exit(3);
    }
    Ok(())
}

/// 各层 FP8 / NVFP4 线性层的 `input_scale`（静态激活 scale，amax ≈ 448·s）：> 1 说明标定 amax 超过 e4m3 上限。
fn scales_probe(src: &WeightSource) -> Result<()> {
    let names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "experts.0.gate_proj", "experts.0.down_proj"];
    for l in 0..30 {
        let mut line = format!("L{l:02}");
        for n in names {
            let key = format!("model.language_model.layers.{l}.{n}.input_scale");
            match src.load_f32_host(&key) {
                Ok((_, v)) => line += &format!(" {}={:.3}", n.rsplit('.').next().unwrap_or(n), v[0]),
                Err(_) => line += &format!(" {}=-", n.rsplit('.').next().unwrap_or(n)),
            }
        }
        println!("{line}");
    }
    Ok(())
}

/// 老行为：三种格式的识别结果与显存账。
fn format_probe(src: &WeightSource) -> Result<()> {
    let dev = Device::new(0)?;
    let stream = dev.new_stream()?;
    println!("tensors: {}", src.names().count());
    let p = "model.language_model.layers.0";
    for name in [
        format!("{p}.experts.0.gate_proj.weight"),
        format!("{p}.experts.0.down_proj.weight"),
        format!("{p}.mlp.gate_proj.weight"),
        format!("{p}.self_attn.q_proj.weight"),
        format!("{p}.router.proj.weight"),
    ] {
        let w = dinfer_kaon3::weights::load_linear(src, &stream, &name)?;
        let (n, kk) = w.shape();
        let desc = match &w {
            dinfer_kaon3::weights::QWeight::NvFp4 { packed, scale, global, .. } => format!(
                "NVFP4 {n}x{kk} packed={:.2}MB scale={:.2}MB global={global:e}",
                packed.len() as f64 / 1e6,
                scale.len() as f64 / 1e6
            ),
            dinfer_kaon3::weights::QWeight::Fp8 { w, scale, .. } =>
                format!("FP8 {n}x{kk} {:.2}MB scale={scale:e}", w.len() as f64 / 1e6),
            dinfer_kaon3::weights::QWeight::Bf16 { w, .. } =>
                format!("BF16 {n}x{kk} {:.2}MB", w.len() as f64 * 2.0 / 1e6),
        };
        println!("{:<52} {desc}", name.trim_start_matches("model.language_model."));
    }
    Ok(())
}

/// 一次加载、多次前向的上下文（批量模式按用例复用）。
struct Ctx {
    _dev: Device,
    stream: Stream,
    gemm: Gemm,
    cfg: KaonConfig,
    model: KaonModel,
}

fn setup(dir: &str, src: &WeightSource, layers: Option<usize>) -> Result<Ctx> {
    let cfg_path = Path::new(dir).join("config.json");
    let cfg = KaonConfig::from_json(&std::fs::read_to_string(&cfg_path)?)?;
    let q_path = Path::new(dir).join("hf_quant_config.json");
    let cfg = if q_path.exists() { cfg.with_quant_json(&std::fs::read_to_string(&q_path)?)? } else { cfg };
    println!(
        "config: {} 层 / {} 专家 top-{} / hidden {} / vocab {} / kv_fp8 {}",
        cfg.num_layers, cfg.num_experts, cfg.top_k, cfg.hidden, cfg.vocab_size, cfg.kv_fp8
    );
    check_weights(src, &cfg)?;

    let dev = Device::new(0)?;
    let stream = dev.new_stream()?;
    let gemm = Gemm::new(&stream)?;

    let t0 = std::time::Instant::now();
    let mut model = KaonModel::load(&stream, src, cfg.clone())?;
    // --layers N：只跑前 N 层（再接 final norm / lm_head / softcap），与 vLLM `--hf-overrides` 截层的 oracle 逐层二分
    if let Some(n) = layers {
        ensure!(n >= 1 && n <= model.layers.len(), "--layers {n} 越界");
        model.layers.truncate(n);
        model.cfg.num_layers = n;
        println!("只跑前 {n} 层");
    }
    stream.synchronize()?;
    println!("权重加载 {:.1} s", t0.elapsed().as_secs_f64());
    Ok(Ctx { _dev: dev, stream, gemm, cfg, model })
}

fn forward(
    ctx: &mut Ctx,
    ids: &[i32],
    dump_layer: Option<usize>,
    oracle: Option<&str>,
    dump: Option<&str>,
    hidden_io: (Option<&str>, Option<&str>),
) -> Result<bool> {
    let (stream, gemm, cfg) = (&ctx.stream, &ctx.gemm, ctx.cfg.clone());
    let model = &mut ctx.model;
    let ops = DefaultOps;
    let tokens = ids.len();
    let num_blocks = (tokens + BLOCK_SIZE - 1) / BLOCK_SIZE + 1;
    model.alloc_kv(&stream, num_blocks, BLOCK_SIZE)?;

    // ---- 一维 batch：单请求 prefill ----
    let cu_seqlens = Tensor::from_vec(&stream, vec![0i32, tokens as i32], &[2])?;
    let seq_lens = Tensor::from_vec(&stream, vec![tokens as i32], &[1])?;
    let positions = Tensor::from_vec(&stream, (0..tokens as i32).collect::<Vec<_>>(), &[tokens])?;
    let slot_mapping = Tensor::from_vec(&stream, (0..tokens as i64).collect::<Vec<_>>(), &[tokens])?;
    let block_table = Tensor::from_vec(&stream, (0..num_blocks as i32).collect::<Vec<_>>(), &[num_blocks])?;

    let mut sc = LayerScratch::new(&stream, &cfg, tokens)?;
    let mut acts = ActBuf::new(&stream, &cfg)?;
    let mut logits = Tensor::uninit(&stream, &[tokens, cfg.vocab_size])?;
    let mut hidden = Tensor::uninit(&stream, &[tokens, cfg.hidden])?;
    let ids_t = Tensor::from_vec(&stream, ids.to_vec(), &[tokens])?;

    let meta = StepMeta {
        mode: dinfer_core::runner::BatchMode::Prefill,
        cu_seqlens: &cu_seqlens,
        seq_lens: &seq_lens,
        block_table: &block_table,
        positions: &positions,
        slot_mapping: &slot_mapping,
        num_reqs: 1,
        max_blocks: num_blocks,
        max_seqlen: tokens,
    };

    let t1 = std::time::Instant::now();
    model.forward(&stream, &gemm, &ops, &ids_t, &meta, &mut sc, &mut acts, &mut logits, &mut hidden, tokens, dump_layer)?;
    stream.synchronize()?;
    println!("一次前向（T={tokens}）{:.1} ms", t1.elapsed().as_secs_f64() * 1e3);

    if let Some(l) = dump_layer {
        let v = hidden.to_vec_f32(&stream)?;
        let n = (v.iter().map(|x| x * x).sum::<f32>()).sqrt();
        println!("layer {l} 残差流: 前 8 = {:?}  ‖x‖ = {n:.4}", &v[..8.min(v.len())]);
        // --hidden-out：整块残差流写盘（f32 小端）；--hidden-ref：与另一次运行逐行比相对误差（定位误差在哪层放大）
        if let Some(p) = hidden_io.0 {
            std::fs::write(p, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
        }
        if let Some(p) = hidden_io.1 {
            let b = std::fs::read(p)?;
            let r: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            ensure!(r.len() == v.len(), "--hidden-ref 长度 {} != {}", r.len(), v.len());
            let h = cfg.hidden;
            let rel: Vec<f32> = (0..tokens)
                .map(|t| {
                    let (a, b) = (&v[t * h..(t + 1) * h], &r[t * h..(t + 1) * h]);
                    let d: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
                    d / b.iter().map(|y| y * y).sum::<f32>().sqrt().max(1e-20)
                })
                .collect();
            let mx = rel.iter().copied().fold(0f32, f32::max);
            let mean = rel.iter().sum::<f32>() / rel.len() as f32;
            println!("HIDDEN layer {l}: 逐行相对误差 均值 {mean:.2e} 最大 {mx:.2e} | 前 12 行 {:?}", rel.iter().take(12).map(|x| format!("{x:.1e}")).collect::<Vec<_>>());
        }
    }

    let lg = logits.to_vec_f32(&stream)?;
    let vocab = cfg.vocab_size;
    let last = &lg[(tokens - 1) * vocab..];
    let mut top: Vec<(usize, f32)> = last.iter().copied().enumerate().collect();
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("最后一行 top-5: {:?}", &top[..5]);
    if let Some(p) = dump {
        dump_oracle(&lg, vocab, ids, p)?;
        println!("已写 {p}（vLLM 同格式，可作 --oracle 做实现间对照）");
    }
    match oracle {
        Some(path) => compare_oracle(&lg, vocab, ids, path),
        None => Ok(true),
    }
}

/// 一行 oracle：top-5（按 logprob 降序）+ 实际下一个 prompt token 的 (id, logprob)。
struct OracleRow {
    top: Vec<(usize, f64)>,
    actual: Option<(usize, f64)>,
}

fn parse_oracle(path: &str, ids: &[i32]) -> Result<Vec<OracleRow>> {
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).with_context(|| format!("读 oracle {path}"))?)?;
    let ch = &v["choices"][0];
    let pl = ch["prompt_logprobs"].as_array().context("oracle 缺 prompt_logprobs")?;
    let t = ids.len();
    ensure!(pl.len() == t, "oracle prompt_logprobs 长度 {} != tokens {t}（ids 与 oracle 不配套？）", pl.len());
    let mut rows = Vec::with_capacity(t);
    for r in 0..t {
        let mut row = OracleRow { top: Vec::new(), actual: None };
        if r + 1 < t {
            let obj = pl[r + 1].as_object().with_context(|| format!("prompt_logprobs[{}] 不是对象", r + 1))?;
            for (k, e) in obj {
                let id: usize = k.parse().with_context(|| format!("token id {k}"))?;
                let lp = e["logprob"].as_f64().context("logprob")?;
                if e["rank"].as_u64().context("rank")? <= 5 {
                    row.top.push((id, lp));
                }
                if id == ids[r + 1] as usize {
                    row.actual = Some((id, lp));
                }
            }
        } else {
            let obj = ch["logprobs"]["top_logprobs"][0].as_object().context("oracle 缺 logprobs.top_logprobs[0]")?;
            for (k, lp) in obj {
                let id: usize = k.trim_start_matches("token_id:").parse().with_context(|| format!("token {k}"))?;
                row.top.push((id, lp.as_f64().context("logprob")?));
            }
        }
        row.top.sort_by(|a, b| b.1.total_cmp(&a.1));
        ensure!(row.top.len() >= 2, "第 {r} 行 oracle top 不足 2 个");
        rows.push(row);
    }
    Ok(rows)
}

fn compare_oracle(lg: &[f32], vocab: usize, ids: &[i32], path: &str) -> Result<bool> {
    let rows = parse_oracle(path, ids)?;
    let (mut decisive, mut miss, mut sum_d, mut n_d, mut max_top1, mut sum_act, mut n_act) = (0, 0, 0.0, 0usize, 0.0f64, 0.0, 0);
    let mut worst: Vec<(f64, String)> = Vec::new();
    for (r, o) in rows.iter().enumerate() {
        let x = &lg[r * vocab..(r + 1) * vocab];
        if x.iter().any(|v| !v.is_finite()) {
            bail!("第 {r} 行 logits 含 NaN/Inf");
        }
        let (am, mx) = x.iter().enumerate().fold((0, f32::MIN), |a, (i, &v)| if v > a.1 { (i, v) } else { a });
        let lse = mx as f64 + x.iter().map(|&v| ((v - mx) as f64).exp()).sum::<f64>().ln();
        let lp = |id: usize| x[id] as f64 - lse;
        let (o1, lp1) = o.top[0];
        let margin = lp1 - o.top[1].1;
        let d1 = (lp(o1) - lp1).abs();
        max_top1 = max_top1.max(d1);
        for &(id, l) in &o.top {
            sum_d += (lp(id) - l).abs();
            n_d += 1;
        }
        let mut dact = f64::NAN;
        if let Some((id, l)) = o.actual {
            dact = (lp(id) - l).abs();
            sum_act += dact;
            n_act += 1;
        }
        let bad = margin >= MARGIN && am != o1;
        if margin >= MARGIN {
            decisive += 1;
            miss += bad as usize;
        }
        let line = format!(
            "row {r:>4}: oracle top1 {o1:>6} ({lp1:>7.3}, margin {margin:.2}) | ours top1 {am:>6} ({:>7.3}) | Δtop1 {d1:.3} | Δact {dact:.3}{}",
            lp(am),
            if bad { "  ← top1 不一致" } else { "" }
        );
        if r < 32 {
            println!("{line}");
        }
        worst.push((if bad { 1e9 } else { d1 }, line));
    }
    worst.sort_by(|a, b| b.0.total_cmp(&a.0));
    if rows.len() > 8 {
        for (_, l) in worst.iter().take(5) {
            println!("worst {l}");
        }
    }
    let mean = sum_d / n_d.max(1) as f64;
    let mean_act = if n_act > 0 { sum_act / n_act as f64 } else { 0.0 };
    let pass = miss == 0 && mean <= MEAN_TOL && max_top1 <= TOP1_TOL;
    println!(
        "PARITY {}: rows {} | 可判定 {decisive} 行 top1 不一致 {miss} | top5 |Δlp| 均值 {mean:.4}（≤{MEAN_TOL}）| top1 |Δlp| 最大 {max_top1:.4}（≤{TOP1_TOL}）| 实际 token |Δlp| 均值 {mean_act:.4}",
        if pass { "PASS" } else { "FAIL" },
        rows.len()
    );
    Ok(pass)
}

/// 把本次 logits 写成 vLLM `/v1/completions` 同格式（`prompt_logprobs` 每行 top-5 + 实际 token，末行 `top_logprobs`），
/// 用于两种实现互相当 oracle：量化"数值敏感度"与"系统性错误"各占多少。
fn dump_oracle(lg: &[f32], vocab: usize, ids: &[i32], path: &str) -> Result<()> {
    use serde_json::{json, Map, Value};
    let t = ids.len();
    let mut pl = vec![Value::Null];
    let mut last = Map::new();
    for r in 0..t {
        let x = &lg[r * vocab..(r + 1) * vocab];
        let mx = x.iter().copied().fold(f32::MIN, f32::max);
        let lse = mx as f64 + x.iter().map(|&v| ((v - mx) as f64).exp()).sum::<f64>().ln();
        let mut idx: Vec<usize> = (0..vocab).collect();
        idx.select_nth_unstable_by(5, |&a, &b| x[b].total_cmp(&x[a]));
        let mut top = idx[..5].to_vec();
        top.sort_by(|&a, &b| x[b].total_cmp(&x[a]));
        if r + 1 < t {
            let mut m = Map::new();
            for (i, &id) in top.iter().enumerate() {
                m.insert(id.to_string(), json!({"logprob": x[id] as f64 - lse, "rank": i + 1}));
            }
            let a = ids[r + 1] as usize;
            if !m.contains_key(&a.to_string()) {
                let rank = x.iter().filter(|&&v| v > x[a]).count() + 1;
                m.insert(a.to_string(), json!({"logprob": x[a] as f64 - lse, "rank": rank}));
            }
            pl.push(Value::Object(m));
        } else {
            for &id in &top {
                last.insert(format!("token_id:{id}"), json!(x[id] as f64 - lse));
            }
        }
    }
    let v = json!({"choices": [{"prompt_logprobs": pl, "logprobs": {"top_logprobs": [last]}}]});
    std::fs::write(path, serde_json::to_string(&v)?)?;
    Ok(())
}
