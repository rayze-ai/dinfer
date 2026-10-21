//! Kaon-V3 端到端探针：加载真实 checkpoint → 前向一次 → 逐层/token 打印摘要。
//!
//! 用法：
//!   kaon3_probe <model_dir>                 # 权重格式探针（老行为）
//!   kaon3_probe <model_dir> --forward <ids...>
//!   kaon3_probe <model_dir> --forward --dump-layer 5 <ids...>
//!
//! 设计意图：作为与 torch 参考（`work/kaon_forward.py`）对拍的入口。
//! `--dump-layer N` 打印第 N 层输出的前若干个数，便于逐层比对"从哪一层开始错"。

use anyhow::{ensure, Result};
use dinfer_kernels::{self as k, bf16, Device, Gemm, Stream};
use dinfer_kaon3::config::KaonConfig;
use dinfer_kaon3::layer::{ActBuf, LayerScratch};
use dinfer_kaon3::model::{check_weights, KaonModel, StepMeta};
use dinfer_layers::{DefaultOps, Tensor, WeightSource};
use std::path::Path;

const BLOCK_SIZE: usize = 32;

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(!args.is_empty(), "usage: kaon3_probe <model_dir> [--forward [--dump-layer N] <ids...>]");
    let dir = args.remove(0);
    let src = WeightSource::open_dir(Path::new(&dir))?;

    if !args.iter().any(|a| a == "--forward") {
        return format_probe(&src);
    }
    args.retain(|a| a != "--forward");
    let dump_layer = args
        .iter()
        .position(|a| a == "--dump-layer")
        .map(|i| {
            let n: usize = args[i + 1].parse().expect("--dump-layer 需要整数");
            args.drain(i..i + 2);
            n
        });
    let ids: Vec<i32> = args.iter().map(|s| s.parse::<i32>().expect("token id")).collect();
    ensure!(!ids.is_empty(), "--forward 需要至少一个 token id");
    forward(&dir, &src, &ids, dump_layer)
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

fn forward(dir: &str, src: &WeightSource, ids: &[i32], dump_layer: Option<usize>) -> Result<()> {
    let cfg_path = Path::new(dir).join("config.json");
    let cfg = KaonConfig::from_json(&std::fs::read_to_string(&cfg_path)?)?;
    println!(
        "config: {} 层 / {} 专家 top-{} / hidden {} / vocab {}",
        cfg.num_layers, cfg.num_experts, cfg.top_k, cfg.hidden, cfg.vocab_size
    );
    check_weights(src, &cfg)?;

    let dev = Device::new(0)?;
    let stream = dev.new_stream()?;
    let gemm = Gemm::new(&stream)?;
    let ops = DefaultOps;
    let tokens = ids.len();

    let t0 = std::time::Instant::now();
    let mut model = KaonModel::load(&stream, src, cfg.clone())?;
    stream.synchronize()?;
    println!("权重加载 {:.1} s", t0.elapsed().as_secs_f64());

    let num_blocks = (tokens + BLOCK_SIZE - 1) / BLOCK_SIZE + 1;
    model.alloc_kv(&stream, num_blocks, BLOCK_SIZE)?;
    println!("kv 层数 = {} / num_blocks = {}", model.kv.len(), num_blocks);

    // ---- 一维 batch：单请求 prefill ----
    let cu_seqlens = Tensor::from_vec(&stream, vec![0i32, tokens as i32], &[2])?;
    let seq_lens = Tensor::from_vec(&stream, vec![tokens as i32], &[1])?;
    let positions = Tensor::from_vec(&stream, (0..tokens as i32).collect::<Vec<_>>(), &[tokens])?;
    let sl = (0..tokens)
        .map(|t| (t / BLOCK_SIZE * BLOCK_SIZE + t % BLOCK_SIZE) as i64)
        .collect::<Vec<_>>();
    let slot_mapping = Tensor::from_vec(&stream, sl, &[tokens])?;
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

    // 逐层对拍用：打印第 l 层输出（残差流）的前 8 个值与范数
    if let Some(l) = dump_layer {
        let v = hidden.to_vec_f32(&stream)?;
        let n = (v.iter().map(|x| x * x).sum::<f32>()).sqrt();
        println!("layer {l} 残差流: 前 8 = {:?}  ‖x‖ = {n:.4}", &v[..8.min(v.len())]);
    }

    // logits 摘要：最后一行 top-5
    let lg = logits.to_vec_f32(&stream)?;
    let last = &lg[(tokens - 1) * cfg.vocab_size..];
    let mut top: Vec<(usize, f32)> = last.iter().enumerate().map(|(i, &x)| (i, x)).collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("最后一行 top-5: {:?}", &top[..5]);
    let _ = k::bf16::default;
    Ok(())
}
