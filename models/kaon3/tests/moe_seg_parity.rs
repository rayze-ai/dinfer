//! 逐专家比对：用**参照自己的专家输入**喂我们的 gate/up，与参照的 `c1` 逐位比。
//!
//! 这一步把"输入 bf16 张量"和"权重"都固定成参照的值，只测 **量化 + GEMM** 这一段。
//! 上一轮已证明 dense 路径逐位 0，但那是自造输入；这里换成参照真实 routed 数据、
//! 真实专家权重、真实 per-expert global scale。
//!
//! 数据（`scripts/dev/moe_seg_capture.py`，compile 口径，`/w/tmp/moe3`）：
//!   `a1_eoff.bin`  int32 [E+1]   —— 每个专家在 shuffled 输入里的行区间
//!   `a1_in.bin`    bf16 [m*topk, k] —— `shuffle_rows` 之后、专家量化器**之前**的输入
//!   `c1_gateup.bin` bf16 [m*topk, 2*inter] —— 参照的 gate/up GEMM 输出（activation 之前）
//!
//! 运行：
//!   SEG_DIR=/w/tmp/moe3 MODEL=<ckpt> EXPERT=0 CUDA_ARCHS=120 \
//!     cargo test --release -p dinfer-kaon3 --test moe_seg_parity -- --ignored --nocapture
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
fn rd_i32(p: &str) -> Vec<i32> {
    let b = std::fs::read(p).expect("读 i32 bin");
    b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[test]
#[ignore]
fn moe_seg_matches_vllm() {
    let dir = std::env::var("SEG_DIR").unwrap_or_else(|_| "/w/tmp/moe3".to_string());
    let model = std::env::var("MODEL").expect("设 MODEL");
    let expert: usize = std::env::var("EXPERT").ok().and_then(|v| v.parse().ok()).unwrap_or(0);

    let cfg = KaonConfig::from_json(CFG).expect("config");
    let (h, inter) = (cfg.hidden, cfg.moe_inter);           // 2816, 704
    let ncol = inter * 2;                                   // 1408 = [gate | up]

    let eoff = rd_i32(&format!("{dir}/a1_eoff.bin"));
    let a_in = rd_bf16(&format!("{dir}/a1_in.bin"));
    let c1 = rd_bf16(&format!("{dir}/c1_gateup.bin"));
    let rows = a_in.len() / h;
    assert_eq!(c1.len(), rows * ncol, "c1 形状 {ncol}");
    let (r0, r1) = (eoff[expert] as usize, eoff[expert + 1] as usize);
    let cnt = r1 - r0;
    println!("SEG rows={rows} expert={expert} 行区间=[{r0},{r1}) 共 {cnt} 行");

    if cnt == 0 {
        println!("SEG 该专家没被选中，换 EXPERT 再跑");
        return;
    }

    let dev = Device::new(0).expect("cuda");
    let st = dev.default_stream();
    let gemm = dinfer_kernels::Gemm::new(&st).expect("gemm");
    let src = WeightSource::open_dir(std::path::Path::new(&model)).expect("ckpt");
    let ep = format!("model.language_model.layers.0.experts.{expert}");

    let x = &a_in[r0 * h..r1 * h];
    let dx = st.htod_copy(x.to_vec()).expect("x");
    let xt = Tensor { data: dx, shape: vec![cnt, h] };

    let mut diff = |name: &str, got: &[bf16], base_col: usize| {
        let mut nd = 0usize;
        let mut maxd = 0f32;
        let mut nd_ulp1 = 0usize;
        for i in 0..cnt {
            for c in 0..inter {
                let g = got[i * inter + c];
                let r = c1[(r0 + i) * ncol + base_col + c];
                if g.to_bits() != r.to_bits() {
                    nd += 1;
                    let (gf, rf) = (f32::from(g), f32::from(r));
                    maxd = maxd.max((gf - rf).abs());
                    // 差 1 个 bf16 ULP？
                    let step = (f32::from(g) - f32::from(bf16::from_bits(r.to_bits() ^ 1))).abs();
                    if (gf - rf).abs() <= step * 1.01 {
                        nd_ulp1 += 1;
                    }
                }
            }
        }
        println!("SEG {name} ndiff={nd}/{} max|d|={maxd:.3e} 其中 ≤1ULP {nd_ulp1}",
                 cnt * inter);
    };

    let gate = QLinear::load(&st, &src, &format!("{ep}.gate_proj.weight")).expect("gate");
    let up = QLinear::load(&st, &src, &format!("{ep}.up_proj.weight")).expect("up");

    let mut g = Tensor::<bf16>::uninit(&st, &[cnt, inter]).unwrap();
    gate.forward(&gemm, &st, &xt, &mut g, cnt).expect("gate fwd");
    diff("gate", &st.dtoh_sync_copy(&g.data).expect("d2h"), 0);

    let mut u = Tensor::<bf16>::uninit(&st, &[cnt, inter]).unwrap();
    up.forward(&gemm, &st, &xt, &mut u, cnt).expect("up fwd");
    diff("up  ", &st.dtoh_sync_copy(&u.data).expect("d2h"), inter);

    // ---- gelu(tanh)·mul：与参照的 `c2`（activation 之后）逐位比 ----
    if std::path::Path::new(&format!("{dir}/c2_act.bin")).exists() {
        let c2 = rd_bf16(&format!("{dir}/c2_act.bin"));
        assert_eq!(c2.len(), rows * inter, "c2 形状");
        let mut act = Tensor::<bf16>::uninit(&st, &[cnt, inter]).unwrap();
        dinfer_kernels::ops::gelu_tanh_mul_bf16(&st, &g.data, &u.data, &mut act.data, cnt, inter)
            .expect("gelu");
        let av = st.dtoh_sync_copy(&act.data).expect("d2h");
        let mut nd = 0usize;
        let mut maxd = 0f32;
        for i in 0..cnt * inter {
            let r = c2[r0 * inter + i];
            if av[i].to_bits() != r.to_bits() {
                nd += 1;
                maxd = maxd.max((f32::from(av[i]) - f32::from(r)).abs());
            }
        }
        println!("SEG act ndiff={nd}/{} max|d|={maxd:.3e}", cnt * inter);
    } else {
        println!("SEG act 跳过（没有 c2_act.bin）");
    }
}
