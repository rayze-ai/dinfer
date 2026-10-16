//! Q1 诊断（不进 q1.sh）：真权重 prefill 逐层 bisect vs HF（`/tmp/qwen3_hf/{embed,attn_L*,hidden_L*}.npy`，由临时脚本产出）。
#![cfg(feature = "ref-tests")]

use dinfer_core::runner::BatchMode;
use dinfer_layers::{Device, Gemm, Tensor, WeightSource};
use dinfer_qwen3::config::Qwen3Config;
use dinfer_qwen3::model::{AttnMeta, Forward, Qwen3Model};
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
fn npy_f32_at(path: &str) -> Option<Vec<f32>> {
    let b = std::fs::read(path).ok()?;
    npyz::NpyFile::new(&b[..]).ok()?.into_vec::<f32>().ok()
}
fn npy_i32(name: &str) -> Vec<i32> {
    let b = std::fs::read(golden_dir().join(name)).unwrap();
    npyz::NpyFile::new(&b[..]).unwrap().into_vec::<i32>().unwrap()
}
fn row_err(a: &[f32], b: &[f32], t: usize, h: usize) -> Vec<f32> {
    (0..t).map(|r| (0..h).map(|c| (a[r * h + c] - b[r * h + c]).abs()).fold(0f32, f32::max)).collect()
}

#[test]
#[ignore]
fn qwen3_layer_bisect() {
    let dev = Device::new(0).unwrap();
    let s = dev.default_stream();
    let gemm = Gemm::new(&s).unwrap();
    let cfg =
        Qwen3Config::from_json(&std::fs::read_to_string(model_dir().join("config.json")).unwrap()).unwrap();
    let src = WeightSource::open_dir(&model_dir()).unwrap();
    let mut model = Qwen3Model::load(&s, &src, cfg.clone(), 512, Some(64), 16).unwrap();
    let prompt = npy_i32("prompt_ids.npy");
    let t = prompt.len();
    let h = cfg.hidden_size as usize;
    let v = cfg.vocab_size as usize;
    let ids = Tensor::from_vec(&s, prompt.clone(), &[t]).unwrap();
    let pos = Tensor::from_vec(&s, (0..t as i32).collect(), &[t]).unwrap();
    let slots = Tensor::from_vec(&s, (0..t as i64).collect(), &[t]).unwrap();
    let cu = Tensor::from_vec(&s, vec![0i32, t as i32], &[2]).unwrap();
    let bt = Tensor::from_vec(&s, (0..64i32).collect(), &[1, 64]).unwrap();
    let seq = Tensor::from_vec(&s, vec![t as i32], &[1]).unwrap();
    let rows = Tensor::from_vec(&s, (0..t as i32).collect(), &[t]).unwrap();
    let mut logits = Tensor::<f32>::zeros(&s, &[t, v]).unwrap();
    model.capture_layers = Some(Vec::new());
    let f = Forward {
        input_ids: &ids,
        positions: &pos,
        slot_mapping: &slots,
        meta: AttnMeta {
            mode: BatchMode::Prefill,
            cu_seqlens: &cu,
            seq_lens: &seq,
            block_table: &bt,
            num_reqs: 1,
            max_blocks: 64,
        },
        logits_rows: &rows,
    };
    model.forward(&s, &gemm, &f, &mut logits).unwrap();
    let caps = model.capture_layers.take().unwrap();
    // embedding check
    if let Some(e) = npy_f32_at("/tmp/qwen3_hf/embed.npy") {
        // 我们没单独 capture embed；用 layer0 的 hidden - attn - mlp 反推不方便，跳过
        let _ = e;
    }
    for (li, a, hd) in &caps {
        let ha = npy_f32_at(&format!("/tmp/qwen3_hf/attn_L{li}.npy"));
        let hh = npy_f32_at(&format!("/tmp/qwen3_hf/hidden_L{li}.npy"));
        let (Some(ha), Some(hh)) = (ha, hh) else {
            eprintln!("layer {li}: no HF dump");
            continue;
        };
        let ea = row_err(a, &ha, t, h);
        let eh = row_err(hd, &hh, t, h);
        let mag = (0..t * h).map(|i| hh[i].abs()).fold(0f32, f32::max);
        let worst_a = ea.iter().cloned().fold(0f32, f32::max);
        let worst_h = eh.iter().cloned().fold(0f32, f32::max);
        eprintln!(
            "layer {li:2}: attn_out max abs {worst_a:.4} (row {}); hidden max abs {worst_h:.4} (row {}) |hidden| {mag:.2}",
            ea.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|x| x.0).unwrap(),
            eh.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|x| x.0).unwrap(),
        );
    }
}
