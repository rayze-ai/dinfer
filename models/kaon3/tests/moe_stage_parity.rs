//! 用**参照真实的 MoE 输入**走我们的专家路径（top-8 → 逐专家 NVFP4 gate/up/down → gelu·mul → combine），
//! 与参照的 MoE 输出逐位比。
//!
//! 数据（`scripts/dev/moe_capture.py`，p02 / layer 0）：
//!   `x.bin`        `[24, 2816]` bf16 —— experts 的真实输入
//!   `router_y.bin` `[24, 128]`  f32   —— router logits（用来算 top-8 与权重）
//!   `y.bin`        `[24, 2816]` bf16 —— 参照的 MoE 输出（combine 之后）
//!
//! 判据：逐位相同。差异大就分两段定位：① 单专家 GEMM 对不对（拿最大权重那个专家单独比）；
//! ② combine 口径对不对（`MOE_COMBINE_F32` 两种都试）。
//!
//! 运行：`MOE_DIR=/w/tmp/moe1 MODEL=<ckpt> CUDA_ARCHS=120 cargo test --release -p dinfer-kaon3 \
//!        --test moe_stage_parity -- --ignored --nocapture`
use dinfer_kaon3::config::KaonConfig;
use dinfer_kaon3::layer::QLinear;
use dinfer_kernels::{bf16, Device};
use dinfer_layers::{Tensor, WeightSource};

const CFG: &str = r#"{"text_config": {
    "vocab_size": 262144, "hidden_size": 2816, "intermediate_size": 2112, "num_hidden_layers": 2,
    "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256, "global_head_dim": 512,
    "num_global_key_value_heads": 2, "moe_intermediate_size": 704, "num_experts": 128, "top_k_experts": 8,
    "sliding_window": 1024, "max_position_embeddings": 262144, "rms_norm_eps": 1e-6,
    "layer_types": ["sliding_attention", "full_attention"], "attention_k_eq_v": true, "tie_word_embeddings": true,
    "rope_parameters": {
        "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}}}}"#;

fn rd_bf16(p: &str) -> Vec<bf16> {
    let b = std::fs::read(p).expect("读 bf16 bin");
    b.chunks_exact(2).map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect()
}
fn rd_f32(p: &str) -> Vec<f32> {
    let b = std::fs::read(p).expect("读 f32 bin");
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[test]
#[ignore]
fn moe_stage_matches_vllm() {
    let dir = std::env::var("MOE_DIR").expect("设 MOE_DIR");
    let model = std::env::var("MODEL").expect("设 MODEL");
    let x = rd_bf16(&format!("{dir}/x.bin"));
    let logits = rd_f32(&format!("{dir}/router_y.bin"));
    let refy = rd_bf16(&format!("{dir}/y.bin"));

    let cfg = KaonConfig::from_json(CFG).expect("config");
    let (t, h, e, k) = (24usize, cfg.hidden, cfg.num_experts, cfg.top_k);
    assert_eq!(x.len(), t * h);
    assert_eq!(logits.len(), t * e);
    assert_eq!(refy.len(), t * h);

    let dev = Device::new(0).expect("cuda");
    let st = dev.default_stream();
    let gemm = dinfer_kernels::Gemm::new(&st).expect("gemm");
    let src = WeightSource::open_dir(std::path::Path::new(&model)).expect("ckpt");
    let p = "model.language_model.layers.0";

    // `MOE_USE_REF_TOPK=1`：直接用参照抓的 topk_ids / topk_w（它们已含 per_expert_scale），
    // 把我方 routing 的贡献完全排除，只测"专家 GEMM + combine"。
    let use_ref_topk = std::env::var("MOE_USE_REF_TOPK").map(|v| v == "1").unwrap_or(false);
    // ---- 参照语义的 top-k 权重（f32，全专家 softmax → top-k 重归一化 → × per_expert_scale）----
    let pes = {
        let v = src.load_bf16(&st, &format!("{p}.router.per_expert_scale")).expect("pes");
        st.dtoh_sync_copy(&v.data).expect("pes d2h").iter().map(|b| f32::from(*b)).collect::<Vec<f32>>()
    };
    let mut ids = vec![0i32; t * k];
    let mut w = vec![0f32; t * k];
    let mut ref_ids: Vec<i32> = Vec::new();
    let mut ref_w: Vec<f32> = Vec::new();
    if use_ref_topk {
        let b = std::fs::read(format!("{dir}/../cmo/topk_ids.bin")).expect("cmo/topk_ids.bin");
        ref_ids = b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let b = std::fs::read(format!("{dir}/../cmo/topk_w.bin")).expect("cmo/topk_w.bin");
        ref_w = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        assert_eq!(ref_ids.len(), t * k);
        ids.copy_from_slice(&ref_ids);
        w.copy_from_slice(&ref_w);
    }
    for r in 0..t {
        if use_ref_topk { break; }
        let row = &logits[r * e..(r + 1) * e];
        let mut idx: Vec<usize> = (0..e).collect();
        idx.sort_by(|&i, &j| row[j].partial_cmp(&row[i]).unwrap_or(std::cmp::Ordering::Equal).then(i.cmp(&j)));
        let top: Vec<usize> = idx[..k].to_vec();
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let ex: Vec<f32> = top.iter().map(|&i| (row[i] - m).exp()).collect();
        let s: f32 = ex.iter().sum();
        for (j, &i) in top.iter().enumerate() {
            ids[r * k + j] = i as i32;
            // 默认口径（MOE_COMBINE_F32=1）：per_expert_scale **折进 down 的 global scale**，
            // 路由权重里不再乘（`layer.rs:319-323` + `973`）。
            w[r * k + j] = ex[j] / s;
        }
    }

    // ---- 逐专家做 gate/up → gelu·mul → down，再用权重累加 ----
    let mut acc = vec![0f32; t * h];
    let mut per_expert_y: Vec<Vec<f32>> = Vec::new();
    let dstd = st.htod_copy(x.clone()).expect("x");
    for eid in 0..e {
        // 该专家是否被任何 token 选中；没选中就跳过（省时间）
        let used = (0..t * k).any(|i| ids[i] == eid as i32);
        if !used {
            continue;
        }
        let ep = format!("{p}.experts.{eid}");
        let gate = QLinear::load(&st, &src, &format!("{ep}.gate_proj.weight")).expect("gate");
        let up = QLinear::load(&st, &src, &format!("{ep}.up_proj.weight")).expect("up");
        let mut down = QLinear::load(&st, &src, &format!("{ep}.down_proj.weight")).expect("down");
        // 与 `layer.rs:319` 同：默认口径把 per_expert_scale 折进 down 的全局 scale（线性等价）
        if !use_ref_topk {
            // 只有"自己算路由权重"时才把 pes 折进 down（那时 w 里没乘 pes）
            if let QLinear::NvFp4 { global, .. } = &mut down {
                *global *= pes[eid];
            }
        }
        let inter = cfg.moe_inter;
        let mut g = Tensor::<bf16>::uninit(&st, &[t, inter]).unwrap();
        let mut u = Tensor::<bf16>::uninit(&st, &[t, inter]).unwrap();
        gate.forward(&gemm, &st, &Tensor { data: dstd.clone(), shape: vec![t, h] }, &mut g, t).expect("gate fwd");
        up.forward(&gemm, &st, &Tensor { data: dstd.clone(), shape: vec![t, h] }, &mut u, t).expect("up fwd");
        // gelu(tanh)·mul（compile 口径：全 f32 一次舍入）
        let mut act = Tensor::<bf16>::uninit(&st, &[t, inter]).unwrap();
        dinfer_kernels::ops::gelu_tanh_mul_bf16(&st, &g.data, &u.data, &mut act.data, t, inter).expect("gelu");
        let mut y = Tensor::<bf16>::uninit(&st, &[t, h]).unwrap();
        down.forward(&gemm, &st, &act, &mut y, t).expect("down fwd");
        let yv = st.dtoh_sync_copy(&y.data).expect("y d2h");
        let yv: Vec<f32> = yv.iter().map(|b| f32::from(*b)).collect();
        per_expert_y.push(yv.clone());
        // 诊断用：把该专家输出落盘，宿主侧做最小二乘反推真实权重
        if let Ok(dd) = std::env::var("MOE_DUMP") {
            let _ = std::fs::create_dir_all(&dd);
            let bits: Vec<u8> = yv.iter().flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes()).collect();
            let _ = std::fs::write(format!("{dd}/e{eid}.bin"), bits);
        }
        // 累加：只在该专家被选中的 (row, slot) 上加 w·y
        for r in 0..t {
            for j in 0..k {
                if ids[r * k + j] == eid as i32 {
                    let ww = w[r * k + j];
                    for c in 0..h {
                        acc[r * h + c] += ww * yv[r * h + c];
                    }
                }
            }
        }
    }

    // ---- 比对（acc 是 f32；参照 y 是 bf16，先舍 bf16 再比）----
    let mut ndiff = 0usize;
    let mut maxd = 0f32;
    for i in 0..t * h {
        let ours = f32::from(bf16::from_f32(acc[i]));
        let r = f32::from(refy[i]);
        if ours.to_bits() != r.to_bits() {
            ndiff += 1;
            maxd = maxd.max((ours - r).abs());
        }
    }
    println!("MOE_STAGE [T={t} H={h} E={e} K={k}] 选中专家 {} 个 | ndiff={ndiff}/{} max|d|={maxd:.3e}", per_expert_y.len(), t * h);
    let oursv: Vec<f32> = (0..8).map(|i| f32::from(bf16::from_f32(acc[i]))).collect();
    let refv: Vec<f32> = (0..8).map(|i| f32::from(refy[i])).collect();
    println!("   ours[0..8] = {:?}", oursv);
    println!("   ref [0..8] = {:?}", refv);
    let onorm = (0..t * h).map(|i| acc[i] * acc[i]).sum::<f32>().sqrt();
    let rnorm = (0..t * h).map(|i| { let v = f32::from(refy[i]); v * v }).sum::<f32>().sqrt();
    println!("   ‖ours‖={onorm:.4}  ‖ref‖={rnorm:.4}  比值={:.4}", onorm / rnorm);
    let s0: Vec<f32> = w[..k].to_vec();
    println!("   row0 weights = {:?}", s0);
    println!("   pes[..4] = {:?}", &pes[..4.min(pes.len())]);
}
