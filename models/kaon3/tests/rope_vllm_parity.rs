//! Gemma4 RoPE 对 vLLM 语义（CPU f64 参考）：sliding = 标准 neox 全维旋转（θ=1e4，分母 256）；
//! full = `Gemma4RotaryEmbedding`：整头 512 维 neox，前 rope_dim/2 = 64 对频率 θ^(-2i/512)（θ=1e6），其余零频（恒等）。
//! 运行：`cargo test --release -p dinfer-kaon3 --test rope_vllm_parity -- --ignored --nocapture`
use dinfer_kaon3::config::KaonConfig;
use dinfer_kernels::{bf16, Device, RopeStyle};
use dinfer_layers::{Rope, Tensor};

const CFG: &str = r#"{"text_config": {
    "vocab_size": 262144, "hidden_size": 2816, "intermediate_size": 2112, "num_hidden_layers": 2,
    "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256, "global_head_dim": 512,
    "num_global_key_value_heads": 2, "moe_intermediate_size": 704, "num_experts": 128, "top_k_experts": 8,
    "sliding_window": 1024, "max_position_embeddings": 262144, "rms_norm_eps": 1e-6,
    "layer_types": ["sliding_attention", "full_attention"], "attention_k_eq_v": true, "tie_word_embeddings": true,
    "rope_parameters": {
        "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}}}}"#;

/// vLLM 语义的参考：inv_freq 长 head/2（不旋转的对补 0），x1 = x[..head/2]、x2 = x[head/2..]
fn vllm_rope(x: &[f32], pos: f64, head: usize, angles: usize, denom: f64, theta: f64) -> Vec<f32> {
    let half = head / 2;
    let mut out = x.to_vec();
    for i in 0..half {
        let inv = if i < angles { 1.0 / theta.powf(2.0 * i as f64 / denom) } else { 0.0 };
        let (c, s) = ((pos * inv).cos(), (pos * inv).sin());
        let (a, b) = (x[i] as f64, x[half + i] as f64);
        out[i] = (a * c - b * s) as f32;
        out[half + i] = (b * c + a * s) as f32;
    }
    out
}

#[test]
#[ignore]
fn gemma4_rope_matches_vllm_semantics() {
    let cfg = KaonConfig::from_json(CFG).unwrap();
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let positions = [0i32, 1, 2, 7, 1000, 1023];
    let t = positions.len();
    let mut seed = 12345u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 40) as f32) / (1u64 << 24) as f32 * 2.0 - 1.0
    };
    // (层型, head_dim, 被旋转的频率对数, 分母, θ, RopeSpec)
    let cases = [
        ("sliding", cfg.head_dim_sliding, cfg.rope_sliding.rope_dim / 2, 256.0, 1e4, cfg.rope_sliding.clone()),
        ("full", cfg.head_dim_full, cfg.rope_full.rope_dim / 2, 512.0, 1e6, cfg.rope_full.clone()),
    ];
    for (name, d, angles, denom, theta, spec) in cases {
        let rope = Rope::with_denom(&st, spec.rope_dim, spec.denom, 2048, spec.theta, RopeStyle::Neox)
            .unwrap()
            .with_span(spec.span)
            .unwrap();
        let (hq, hk) = (2usize, 1usize);
        let q: Vec<f32> = (0..t * hq * d).map(|_| bf16::from_f32(rnd()).to_f32()).collect();
        let k: Vec<f32> = (0..t * hk * d).map(|_| bf16::from_f32(rnd()).to_f32()).collect();
        let mut dq = Tensor::from_vec(&st, q.iter().map(|&x| bf16::from_f32(x)).collect(), &[t, hq * d]).unwrap();
        let mut dk = Tensor::from_vec(&st, k.iter().map(|&x| bf16::from_f32(x)).collect(), &[t, hk * d]).unwrap();
        let dp = Tensor::from_vec(&st, positions.to_vec(), &[t]).unwrap();
        rope.apply(&st, &mut dq, Some(&mut dk), &dp, t, hq, hk, d).unwrap();
        let (gq, gk) = (dq.to_vec_f32(&st).unwrap(), dk.to_vec_f32(&st).unwrap());
        let mut worst = 0f32;
        for (got, src, heads) in [(&gq, &q, hq), (&gk, &k, hk)] {
            for ti in 0..t {
                for h in 0..heads {
                    let o = (ti * heads + h) * d;
                    let want = vllm_rope(&src[o..o + d], positions[ti] as f64, d, angles, denom, theta);
                    for c in 0..d {
                        worst = worst.max((got[o + c] - want[c]).abs());
                    }
                }
            }
        }
        println!("{name}: head {d} 旋转 {angles} 对 / 分母 {denom} / θ {theta:e}：最大绝对误差 {worst:.4}");
        assert!(worst < 0.02, "{name} RoPE 与 vLLM 语义不符：最大误差 {worst}");
    }
}
