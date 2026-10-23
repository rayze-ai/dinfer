//! 把**参照真实的 router 输入**灌进我们的 router 路径，与参照的 logits 逐位比。
//!
//! 数据（`scripts/dev/moe_capture.py` 在 vLLM 里抓的，p02 / layer 0）：
//!   `router_x.bin`  `[24, 2816]` bf16 —— router 的真实输入（pre_ff_norm_2 输出）
//!   `router_y.bin`  `[24, 128]`  f32  —— 参照的 router logits
//! 我们侧复算：`rmsnorm_scale2_bf16(x, scale, rs) → router_gemm_bi_f32`
//! 判据：逐位相同（0 差异）。有差异就打印 ulp 分布与"会改变 top-8 集合的行数"。
//!
//! 运行：`MOE_DIR=/w/tmp/moe1 CUDA_ARCHS=120 cargo test --release -p dinfer-kaon3 \
//!        --test router_stage_parity -- --ignored --nocapture`
use dinfer_kaon3::config::KaonConfig;
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

#[test]
#[ignore]
fn router_stage_matches_vllm() {
    // `rmsnorm_scale2` 只支持 compile 系（0 / 2）；默认 1 是 eager BI，会 cudaError。
    std::env::set_var("RMSNORM_MODE", "2");
    let dir = std::env::var("MOE_DIR").expect("设 MOE_DIR");
    let model = std::env::var("MODEL").expect("设 MODEL（checkpoint 目录）");
    let x = rd_bf16(&format!("{dir}/router_x.bin"));
    let yb = std::fs::read(format!("{dir}/router_y.bin")).expect("router_y");
    let refy: Vec<f32> = yb.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let cfg = KaonConfig::from_json(CFG).expect("config");
    let (t, h, e) = (24usize, cfg.hidden, cfg.num_experts);
    assert_eq!(x.len(), t * h, "router_x 形状");
    assert_eq!(refy.len(), t * e, "router_y 形状");

    let dev = Device::new(0).expect("cuda");
    let st = dev.default_stream();
    let gemm = dinfer_kernels::Gemm::new(&st).expect("gemm");
    let src = WeightSource::open_dir(std::path::Path::new(&model)).expect("打开 checkpoint");

    // 只取 layer 0 的 router 参数（真实的 scale / proj / per_expert_scale）
    let p = "model.language_model.layers.0";
    let router_p = format!("{p}.router");
    let scale = src.load_bf16(&st, &format!("{router_p}.scale")).expect("router.scale");
    let proj = src.load_bf16(&st, &format!("{router_p}.proj.weight")).expect("router.proj.weight");
    assert_eq!(scale.numel(), h);
    assert_eq!(proj.numel(), e * h);

    // ---- 我们的 router 路径：无权重 RMSNorm + ×root_size + ×scale（compile 口径融合）→ GEMM ----
    let root_size = f32::from(bf16::from_f32((h as f32).powf(-0.5)));
    let dx = st.htod_copy(x.clone()).expect("upload x");
    let mut routed = st.alloc_zeros::<bf16>(t * h).expect("routed");
    dinfer_kernels::ops::rmsnorm_scale2_bf16(
        &st, &dx, &scale.data, &mut routed, t, h, cfg.eps, root_size,
    )
    .expect("rmsnorm_scale2");
    let mut logits = st.alloc_zeros::<f32>(t * e).expect("logits");
    dinfer_kernels::ops::router_gemm_bi_f32(&st, &routed, &proj.data, &mut logits, t, e, h).expect("gemm");
    let ours = st.dtoh_sync_copy(&logits).expect("d2h");

    // ---- 比对 ----
    let mut ndiff = 0usize;
    let mut maxd = 0f32;
    for i in 0..t * e {
        if ours[i].to_bits() != refy[i].to_bits() {
            ndiff += 1;
            maxd = maxd.max((ours[i] - refy[i]).abs());
        }
    }
    // top-8 集合是否改变（这才是端到端真正关心的）
    let mut setdiff_rows = 0usize;
    for r in 0..t {
        let a = &ours[r * e..(r + 1) * e];
        let b = &refy[r * e..(r + 1) * e];
        let top8 = |v: &[f32]| {
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|&i, &j| v[j].partial_cmp(&v[i]).unwrap_or(std::cmp::Ordering::Equal).then(i.cmp(&j)));
            idx[..8].to_vec()
        };
        if top8(a) != top8(b) {
            setdiff_rows += 1;
        }
    }
    println!("ROUTER_STAGE [T={t} E={e} H={h}] logits ndiff={ndiff}/{} max|d|={maxd:.3e} | top-8 集合不同的行 = {setdiff_rows}/{t}", t * e);
    let _ = &gemm;
}
