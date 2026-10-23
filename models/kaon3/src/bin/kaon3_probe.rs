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
    if let Some(i) = args.iter().position(|a| a == "--dump-weights") {
        let d = args.get(i + 1).cloned().context("--dump-weights 需要目录")?;
        return dump_weights(&src, &d);
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
    // --hidden-all <dir>：一次前向 dump 全部层残差流（与 vLLM 侧 30 个 L{i}.bin 逐层比）
    let hidden_all = take("--hidden-all");
    // --logits-out <file>：把完整 logits（[T,vocab]，softcap 之后）写成 f32 LE，用于与参照逐元素比
    let logits_out = take("--logits-out");
    // --layer-iso <l>:<ref.bin>：单层隔离 —— 用参照的第 l-1 层残差流灌进第 l 层，只跑这一层，
    // 输出写 <hidden-all>/L{l}.bin（与参照同层直接比）。用来逐层判"实现本身"对不对。
    let layer_iso = take("--layer-iso");
    // --layer-iso-eps <f32>：注入时给输入加**确定性**相对扰动（幅度 eps，用行内索引做伪随机符号），
    // 用来测"该层对输入微小扰动的放大倍数"。因为注入本身要把参照 f32 量化成 bf16（~2e-3 相对），
    // 这个放大倍数决定了仪器下限——才能分清某层的高误差是"实现错"还是"层本身敏感"。
    let layer_iso_eps = take("--layer-iso-eps").map(|s| s.parse::<f32>().expect("eps 需要浮点"));
    // --hidden-cmp <refdir>：把 --hidden-all 写的逐层残差流与参照目录逐层比（纯 Rust，不用 Python 参考）。
    let hidden_cmp = take("--hidden-cmp");
    // --sub-cmp <refdir>：把 --hidden-all 目录里**所有同名 .bin** 与参照目录逐项比（layer 0 子模块对拍）。
    let sub_cmp = take("--sub-cmp");
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
            forward(&mut ctx, &ids, None, None, Some(&format!("{out}/{n}.json")), (None, None), None, None)?;
            println!("BATCH {n}: T={} {:.2} s", ids.len(), t.elapsed().as_secs_f64());
        }
        return Ok(());
    }
    let ids: Vec<i32> = args.iter().map(|s| s.parse::<i32>().expect("token id")).collect();
    ensure!(!ids.is_empty(), "--forward 需要至少一个 token id");
    let mut ctx = setup(&dir, &src, layers)?;
    if let Some(spec) = layer_iso.as_deref() {
        let (l, path) = spec.split_once(':').context("--layer-iso 需要 l:ref.bin")?;
        let l: usize = l.parse()?;
        let b = std::fs::read(path)?;
        let mut data: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        if let Some(eps) = layer_iso_eps {
            // 确定性扰动：符号由元素下标决定，幅度 = eps * |x|（相对扰动），不引入随机源。
            for (i, x) in data.iter_mut().enumerate() {
                let sign = if (i * 2654435761usize) & 1 == 0 { 1.0 } else { -1.0 };
                *x *= 1.0 + sign * eps;
            }
        }
        let dir = hidden_all.as_deref().context("--layer-iso 需要 --hidden-all <dir>")?;
        std::fs::create_dir_all(dir)?;
        run_layer_iso(&mut ctx, &ids, l, &data, dir)?;
        // 隔离 + `SUB_LAYER=<l>` 时可一次跑完子模块对拍（原来 sub-cmp 在早退之后，永远走不到）。
        if let Some(r) = sub_cmp.as_deref() {
            compare_sub_dir(dir, r, ids.len())?;
        }
        return Ok(());
    }
    let pass = forward(&mut ctx, &ids, dump_layer, oracle.as_deref(), dump.as_deref(), (hidden_out.as_deref(), hidden_ref.as_deref()), hidden_all.as_deref(), logits_out.as_deref())?;
    if let (Some(a), Some(r)) = (hidden_all.as_deref(), sub_cmp.as_deref()) {
        compare_sub_dir(a, r, ids.len())?;
        return Ok(());
    }
    if let (Some(a), Some(r)) = (hidden_all.as_deref(), hidden_cmp.as_deref()) {
        compare_hidden_dir(a, r, ctx.cfg.hidden)?;
        return Ok(());
    }
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
    hidden_all: Option<&str>,
    logits_out: Option<&str>,
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
    if let Some(d) = hidden_all {
        std::fs::create_dir_all(d)?;
    }
    model.forward(&stream, &gemm, &ops, &ids_t, &meta, &mut sc, &mut acts, &mut logits, &mut hidden, tokens, dump_layer, hidden_all, None)?;
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
    if let Some(p) = logits_out {
        let bytes: Vec<u8> = lg.iter().take(tokens * vocab).flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(p, bytes)?;
        println!("已写 {p}（完整 logits [{}×{}] f32 LE）", tokens, vocab);
    }
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

/// 逐层残差流对拍：`ours` 与 `ref` 目录下的 `L{i}.bin`（f32 LE，[T,H]）逐层、逐行比。
/// 输出每层「逐行相对误差」的均值 / 最大，以及相对上一层的放大倍数 —— 用来定位误差从哪层开始。
fn compare_hidden_dir(ours: &str, rf: &str, h: usize) -> Result<()> {
    let read = |p: &str| -> Result<Vec<f32>> {
        let b = std::fs::read(p).with_context(|| format!("读 {p}"))?;
        ensure!(b.len() % 4 == 0, "{p} 长度不是 4 的倍数");
        Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    };
    println!("# 逐层残差流相对误差（逐行 L2 差 / 参照 L2），参照 = {rf}");
    println!("# layer  rows  mean_rel  max_rel  rel_vs_prev");
    let mut prev: Option<f64> = None;
    // Lm1 = embed 输出（layer 0 之前），用来把 "embed 错" 与 "layer 0 错" 分开
    for l in (-1i64..64).collect::<Vec<_>>() {
        let tag = if l < 0 { "Lm1".to_string() } else { format!("L{l}") };
        let (pa, pb) = (format!("{ours}/{tag}.bin"), format!("{rf}/{tag}.bin"));
        if !std::path::Path::new(&pb).exists() {
            break;
        }
        let (a, b) = (read(&pa)?, read(&pb)?);
        ensure!(a.len() == b.len(), "{tag}: 长度 {} != {}", a.len(), b.len());
        ensure!(a.len() % h == 0, "{tag}: 长度 {} 不是 H={h} 的倍数", a.len());
        let rows = a.len() / h;
        let mut rels = Vec::with_capacity(rows);
        for t in 0..rows {
            let (x, y) = (&a[t * h..(t + 1) * h], &b[t * h..(t + 1) * h]);
            let d: f64 = x.iter().zip(y).map(|(p, q)| ((p - q) as f64).powi(2)).sum::<f64>().sqrt();
            let n: f64 = y.iter().map(|q| (*q as f64).powi(2)).sum::<f64>().sqrt();
            rels.push(d / n.max(1e-30));
        }
        let mean = rels.iter().sum::<f64>() / rows as f64;
        let mx = rels.iter().copied().fold(0f64, f64::max);
        let jump = prev.map(|p| if p > 0.0 { mean / p } else { f64::INFINITY });
        println!(
            "{tag:<4} {rows:>4}  {mean:.3e}  {mx:.3e}  {}",
            jump.map(|j| format!("{j:.2}x")).unwrap_or_else(|| "-".into())
        );
        prev = Some(mean);
    }
    Ok(())
}

/// layer 0 子模块逐项对拍：遍历参照目录里的每个 `.bin`，在 ours 目录找同名文件，
/// 按"最后一维 = 块宽"切成行，逐行比相对 L2 误差。块宽由文件大小与行数推断：
/// 对 `router_logits`（E=128）与 hidden（H=2816）都成立的做法是**按参照文件大小 / 行数**算。
fn compare_sub_dir(ours: &str, rf: &str, rows_want: usize) -> Result<()> {
    let read = |p: &std::path::Path| -> Result<Vec<f32>> {
        let b = std::fs::read(p).with_context(|| format!("读 {}", p.display()))?;
        ensure!(b.len() % 4 == 0, "{} 长度不是 4 的倍数", p.display());
        Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    };
    let mut names: Vec<String> = std::fs::read_dir(rf)?
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|f| f.ends_with(".bin"))
        .collect();
    names.sort();
    ensure!(!names.is_empty(), "参照目录 {rf} 里没有 .bin");
    println!("# layer 0 子模块对拍：参照 = {rf}");
    println!("# {:<18} {:>6} {:>11} {:>11}", "name", "rows", "mean_rel", "max_rel");
    for n in names {
        let pa = std::path::Path::new(ours).join(&n);
        if !pa.exists() {
            println!("# {n:<18} 我们侧缺这个文件");
            continue;
        }
        let pb = std::path::Path::new(rf).join(&n);
        let (a, b) = (read(&pa)?, read(&pb)?);
        ensure!(a.len() == b.len(), "{n}: 长度 {} != {}", a.len(), b.len());
        // 行宽直接用 T 算：这些 .bin 都是 [T, w]（w = H / E / heads*head_dim / nkv*head_dim）
        ensure!(a.len() % rows_want == 0, "{n}: 长度 {} 不是 T={rows_want} 的倍数", a.len());
        let w = a.len() / rows_want;
        let rows = a.len() / w;
        let mut rels = Vec::with_capacity(rows);
        for t in 0..rows {
            let (x, y) = (&a[t * w..(t + 1) * w], &b[t * w..(t + 1) * w]);
            let d: f64 = x.iter().zip(y).map(|(p, q)| ((p - q) as f64).powi(2)).sum::<f64>().sqrt();
            let nn: f64 = y.iter().map(|q| (*q as f64).powi(2)).sum::<f64>().sqrt();
            rels.push(d / nn.max(1e-30));
        }
        let mean = rels.iter().sum::<f64>() / rows as f64;
        let mx = rels.iter().copied().fold(0f64, f64::max);
        println!("{:<20} {rows:>6} {mean:>11.3e} {mx:>11.3e}", n.trim_end_matches(".bin"));
    }
    Ok(())
}

/// 把关键权重原样导出成 bf16 LE，用于和 safetensors 直接 `cmp` 校验加载路径没有改数值。
fn dump_weights(src: &WeightSource, dir: &str) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let dev = Device::new(0)?;
    let stream = dev.new_stream()?;
    // router 三件按 `RLAYER=<l>` 导出（默认 0；bf16 algo 要按真实层的输入标定）
    let rl: usize = std::env::var("RLAYER").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let p = format!("model.language_model.layers.{rl}");
    // qkv 组：走与 `GemmaLayer::load` 完全相同的 `load_fp8_group`，导出重量化后的 fp8 字节与 scale
    // 层号：`WLAYERS=0,9,11`（默认 0,11）
    let wl: Vec<usize> = std::env::var("WLAYERS").unwrap_or_else(|_| "0,11".into())
        .split(',').filter_map(|x| x.trim().parse().ok()).collect();
    for l in wl {
        let lp = format!("model.language_model.layers.{l}.self_attn");
        let cfg_path = std::path::Path::new(dir).join("cfg.json");
        let _ = &cfg_path;
        let k_eq_v = (l + 1) % 6 == 0; // 本模型：full 层（5/11/17/23/29）k_eq_v 两槽，其余滑窗层三槽（与 config.k_eq_v_at 一致）
        let names: Vec<String> = if k_eq_v {
            vec![format!("{lp}.q_proj.weight"), format!("{lp}.k_proj.weight")]
        } else {
            vec![format!("{lp}.q_proj.weight"), format!("{lp}.k_proj.weight"), format!("{lp}.v_proj.weight")]
        };
        let parts = dinfer_kaon3::weights::load_fp8_group(src, &stream, &names)?;
        for (i, qw) in parts.iter().enumerate() {
            if let dinfer_kaon3::weights::QWeight::Fp8 { w, scale, a_scale, n, k } = qw {
                println!("OUR L{l} part{i}: {n}x{k} scale={scale:e} a_scale={a_scale:e} bytes={}", w.len());
                let host = stream.dtoh_sync_copy(w)?;
                std::fs::write(std::path::Path::new(dir).join(format!("our_qkv_L{l}_p{i}.bin")), &host)?;
            }
        }
    }
    for (name, file) in [
        (format!("{p}.router.proj.weight"), "router_proj_bf16.bin"),
        (format!("{p}.router.scale"), "router_scale_bf16.bin"),
        (format!("{p}.router.per_expert_scale"), "router_pes_bf16.bin"),
    ] {
        let (shape, v) = src.load_f32_host(&name)?;
        let bits: Vec<u8> = v
            .iter()
            .flat_map(|x| half::bf16::from_f32(*x).to_bits().to_le_bytes())
            .collect();
        std::fs::write(std::path::Path::new(dir).join(file), &bits)?;
        println!("{file}: shape {shape:?} {} bytes", bits.len());
    }
    Ok(())
}

/// 单层隔离：把参照的第 `l-1` 层残差流灌进第 `l` 层，只跑这一层，写 `<dir>/L{l}.bin`。
/// 前置各层照跑（提供本层注意力要读的 KV），但它们的输出不影响本层输入（输入是被灌进去的）。
fn run_layer_iso(ctx: &mut Ctx, ids: &[i32], l: usize, data: &[f32], dir: &str) -> Result<()> {
    let (stream, gemm, cfg) = (&ctx.stream, &ctx.gemm, ctx.cfg.clone());
    let model = &mut ctx.model;
    let ops = DefaultOps;
    let tokens = ids.len();
    let h = cfg.hidden;
    ensure!(data.len() >= tokens * h, "layer-iso: 参照数据 {} < T*H {}", data.len(), tokens * h);
    let num_blocks = (tokens + BLOCK_SIZE - 1) / BLOCK_SIZE + 1;
    model.alloc_kv(stream, num_blocks, BLOCK_SIZE)?;
    let cu_seqlens = Tensor::from_vec(stream, vec![0i32, tokens as i32], &[2])?;
    let seq_lens = Tensor::from_vec(stream, vec![tokens as i32], &[1])?;
    let positions = Tensor::from_vec(stream, (0..tokens as i32).collect::<Vec<_>>(), &[tokens])?;
    let slot_mapping = Tensor::from_vec(stream, (0..tokens as i64).collect::<Vec<_>>(), &[tokens])?;
    let block_table = Tensor::from_vec(stream, (0..num_blocks as i32).collect::<Vec<_>>(), &[num_blocks])?;
    let mut sc = LayerScratch::new(stream, &cfg, tokens)?;
    let mut acts = ActBuf::new(stream, &cfg)?;
    let mut logits = Tensor::uninit(stream, &[tokens, cfg.vocab_size])?;
    let mut hidden = Tensor::uninit(stream, &[tokens, h])?;
    let ids_t = Tensor::from_vec(stream, ids.to_vec(), &[tokens])?;
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
    model.forward(stream, gemm, &ops, &ids_t, &meta, &mut sc, &mut acts, &mut logits, &mut hidden,
                  tokens, None, Some(dir), Some((data, l, 0)))?;
    stream.synchronize()?;
    // 隔离结果写到 `iso_L{l}.bin`（不与 `--hidden-all` 写的逐层残差流抢同一个文件名）
    let v = hidden.to_vec_f32(stream)?;
    let bytes: Vec<u8> = v.iter().take(tokens * h).flat_map(|x| x.to_le_bytes()).collect();
    let out = std::path::Path::new(dir).join(format!("iso_L{l}.bin"));
    std::fs::write(&out, bytes)?;
    println!("layer-iso L{l} 完成 → {}", out.display());
    Ok(())
}
