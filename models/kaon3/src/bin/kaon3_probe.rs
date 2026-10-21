//! 权重加载探针：读真实 checkpoint，报告三种格式的识别结果与显存占用。
use anyhow::Result;
use dinfer_kernels::Device;
use dinfer_layers::WeightSource;
use std::path::Path;

fn main() -> Result<()> {
    let dir = std::env::args().nth(1).expect("usage: kaon3_probe <model_dir>");
    let src = WeightSource::open_dir(Path::new(&dir))?;
    println!("tensors: {}", src.names().count());
    let dev = Device::new(0)?;
    let stream = dev.new_stream()?;
    let p = "model.language_model.layers.0";
    for name in [
        format!("{p}.experts.0.gate_proj.weight"),
        format!("{p}.experts.0.down_proj.weight"),
        format!("{p}.mlp.gate_proj.weight"),
        format!("{p}.self_attn.q_proj.weight"),
        format!("{p}.router.proj.weight"),
    ] {
        let w = dinfer_kaon3::weights::load_linear(&src, &stream, &name)?;
        let (n, k) = w.shape();
        let mb = match &w {
            dinfer_kaon3::weights::QWeight::NvFp4 { packed, scale, global, .. } => {
                format!("NVFP4 {}x{} packed={:.2}MB scale={:.2}MB global={:e}",
                    n, k, packed.len() as f64/1e6, scale.len() as f64/1e6, global)
            }
            dinfer_kaon3::weights::QWeight::Fp8 { w, scale, .. } =>
                format!("FP8 {}x{} {:.2}MB scale={:e}", n, k, w.len() as f64/1e6, scale),
            dinfer_kaon3::weights::QWeight::Bf16 { w, .. } =>
                format!("BF16 {}x{} {:.2}MB", n, k, w.len() as f64*2.0/1e6),
        };
        println!("{:<52} {}", name.trim_start_matches("model.language_model."), mb);
    }
    // 全模型专家显存核算
    let (inter, hidden, e) = (704usize, 2816usize, 128usize);
    let per_expert = inter*hidden/2*2 + hidden*inter/2;
    let per_scale = inter*hidden/16*2 + hidden*inter/16;
    let total = (per_expert + per_scale) * e * 30;
    println!("\n专家总量: {:.2} GB (packed {:.2} + scale {:.2})",
        total as f64/1e9, (per_expert*e*30) as f64/1e9, (per_scale*e*30) as f64/1e9);
    Ok(())
}
