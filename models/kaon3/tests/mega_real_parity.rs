//! 真实权重下 dense MLP 支路的 megakernel 对拍：mega（一次 launch）vs 现有逐 op 路径。
//!
//! 与 `crates/kernels/tests/mega_mlp_parity.rs`（合成随机权重）互补：那份证明 kernel 自身正确，
//! 这份证明**接线**（真实 per-tensor scale / 真实激活分布 / 真实 H、I）不漏东西。
//!
//! 跑：`KAON3_M=<M> cargo test --release -p dinfer-kaon3 --test mega_real_parity -- --ignored --nocapture`
//! 需要 `/hf_data/hub/.../Kaon-V3-nvpf4-mixed`（远端 5090）。
use dinfer_kernels::{bf16, cudarc::driver::CudaSlice, Device, Gemm, Stream};
use dinfer_kaon3::config::KaonConfig;
use dinfer_kaon3::layer::{LayerScratch, QLinear};
use dinfer_layers::{Norm as _, OpSet, Tensor, WeightSource};
use std::ffi::c_void;

fn model_dir() -> String {
    std::env::var("KAON3_DIR").unwrap_or_else(|_| {
        "/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478/"
            .to_string()
    })
}

fn m_val() -> usize {
    std::env::var("KAON3_M").ok().and_then(|v| v.parse().ok()).unwrap_or(1)
}

/// 4 个阶段各自的逐位一致计数，便于在失败时判断是哪一段先偏。
fn cmp(stream: &Stream, a: &CudaSlice<bf16>, b: &CudaSlice<bf16>, n: usize, tag: &str) -> bool {
    let va = stream.dtoh_sync_copy(a).unwrap();
    let vb = stream.dtoh_sync_copy(b).unwrap();
    let (mut nd, mut mx) = (0usize, 0f32);
    let (mut na, mut nb) = (0f64, 0f64);
    for i in 0..n {
        let (x, y) = (va[i].to_f32(), vb[i].to_f32());
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
        if va[i].to_bits() != vb[i].to_bits() {
            nd += 1;
        }
        mx = mx.max((x - y).abs());
    }
    let rel = if na > 0.0 { ((nb - na) / na).abs() } else { 0.0 };
    // 退化检查：两侧都必须非零，否则"全零比全零"会假通过
    let ok_nonzero = na.sqrt() > 1.0 && nb.sqrt() > 1.0;
    println!(
        "[{tag}] 不同 {nd}/{n}  max|d|={mx:.4e}  ‖ref‖={:.4} ‖mega‖={:.4}  {}",
        na.sqrt(),
        nb.sqrt(),
        if !ok_nonzero { "退化(有全零!)" } else if nd == 0 { "逐位一致" } else { "有差异" }
    );
    assert!(ok_nonzero, "{tag}: 有一侧范数异常（全零？）");
    nd == 0
}

#[test]
#[ignore]
fn mega_real_weights_matches_eager() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let gemm = Gemm::new(&st).unwrap();
    let ops = dinfer_layers::DefaultOps;

    let dir = model_dir();
    let dpath = std::path::Path::new(&dir);
    let cfg = KaonConfig::from_json(&std::fs::read_to_string(dpath.join("config.json")).unwrap()).unwrap();
    let src = WeightSource::open_dir(dpath).expect("打开权重失败");
    let (h, inter) = (cfg.hidden, cfg.dense_inter);
    let eps = cfg.eps;
    let m = m_val();
    let p = dinfer_kaon3::weights::layer_prefix(0);

    // 真实权重（走与生产完全相同的加载口径：load_fp8_group 会 requantize 到组内最大 scale）
    let dense = format!("{p}.mlp");
    let gu = QLinear::load_group(
        &st,
        &src,
        &[format!("{dense}.gate_proj.weight"), format!("{dense}.up_proj.weight")],
    )
    .unwrap();
    let (gate, up) = (&gu[0], &gu[1]);
    let down = QLinear::load(&st, &src, &format!("{dense}.down_proj.weight")).unwrap();
    let pre_w = src.load_bf16(&st, &format!("{p}.pre_feedforward_layernorm.weight")).unwrap();
    let post_w = src.load_bf16(&st, &format!("{p}.post_feedforward_layernorm_1.weight")).unwrap();

    let seed0: u64 = std::env::var("KAON3_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let iters: usize = std::env::var("KAON3_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    // 输入：用确定性伪随机 + 真实量级（Kaon 残差流 ≈ ±1）
    let xv: Vec<bf16> = (0..m * h)
        .map(|i| {
            let s = (i as u64 + seed0 * 1000003).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let s = s ^ (s >> 29);
            bf16::from_f32(((s >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * 1.5)
        })
        .collect();
    let dx = st.htod_copy(xv).unwrap();

    println!("m={m} H={h} I={inter} eps={eps} iters={iters}");

    // ---- mega 路径：与 layer.rs::mega_dense_branch 相同的缓冲与参数 ----
    let mut mb = LayerScratch::new(&st, &cfg, 8).unwrap();
    let Some(mb) = mb.mega.as_mut() else { panic!("KAON3_MEGA 未开：测试里直接置 env 或改 LayerScratch::new") };
    let (gw, gw_s, gw_a, _, _) = gate.fp8_parts().expect("gate 不是 FP8");
    let (uw, uw_s, uw_a, _, _) = up.fp8_parts().expect("up 不是 FP8");
    let (dw, dw_s, dw_a, _, _) = down.fp8_parts().expect("down 不是 FP8");

    let mut total = [0usize; 6];
    let mut first_bad: Option<usize> = None;
    let mut dx = dx;
    for it in 0..iters {
        if it > 0 {
            // 换一组输入（真实权重保持不变）
            let seed = (it as u64 + seed0) * 1000003;
            let xv: Vec<bf16> = (0..m * h)
                .map(|i| {
                    let s = (i as u64).wrapping_add(seed).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let s = s ^ (s >> 29);
                    bf16::from_f32(((s >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * 1.5)
                })
                .collect();
            dx = st.htod_copy(xv).unwrap();
        }

        // ---- 参考路径：与 layer.rs 的逐 op 路径逐字对齐 ----
        let mut ref_norm1 = Tensor::zeros(&st, &[m, h]).unwrap();
        ops.norm().rmsnorm(&st, &dx, &pre_w.data, &mut ref_norm1.data, m, h, eps).unwrap();
        let mut gate_out = Tensor::zeros(&st, &[m, inter]).unwrap();
        let mut up_out = Tensor::zeros(&st, &[m, inter]).unwrap();
        let mut act_bf = Tensor::zeros(&st, &[m, inter]).unwrap();
        let mut dn_out = Tensor::zeros(&st, &[m, h]).unwrap();
        let mut ref_ff1 = Tensor::zeros(&st, &[m, h]).unwrap();
        gate.forward(&gemm, &st, &ref_norm1, &mut gate_out, m).unwrap();
        up.forward(&gemm, &st, &ref_norm1, &mut up_out, m).unwrap();
        dinfer_kernels::ops::gelu_tanh_mul_round2_bf16(&st, &gate_out.data, &up_out.data, &mut act_bf.data, m, inter)
            .unwrap();
        down.forward(&gemm, &st, &act_bf, &mut dn_out, m).unwrap();
        ops.norm().rmsnorm(&st, &dn_out.data, &post_w.data, &mut ref_ff1.data, m, h, eps).unwrap();
        st.synchronize().unwrap();

        let mut mega_ff1 = Tensor::zeros(&st, &[m, h]).unwrap();
        st.memset_zeros_n(&mut mb.ready, 6).unwrap();
        let rc = unsafe {
            dinfer_kernels::ffi::mega_mlp_dense_branch(
                st.ptr(&dx) as *const c_void,
                st.ptr_mut(&mut mega_ff1.data) as *mut c_void,
                st.ptr(&pre_w.data) as *const c_void,
                st.ptr(&post_w.data) as *const c_void,
                st.ptr(gw) as *const c_void,
                gw_s,
                1.0 / gw_a,
                st.ptr(uw) as *const c_void,
                uw_s,
                1.0 / uw_a,
                st.ptr(dw) as *const c_void,
                dw_s,
                1.0 / dw_a,
                st.ptr_mut(&mut mb.norm1.data) as *mut c_void,
                st.ptr_mut(&mut mb.norm1_q) as *mut c_void,
                st.ptr_mut(&mut mb.gate_buf.data) as *mut c_void,
                st.ptr_mut(&mut mb.up_buf.data) as *mut c_void,
                st.ptr_mut(&mut mb.act_bf.data) as *mut c_void,
                st.ptr_mut(&mut mb.act_q) as *mut c_void,
                st.ptr_mut(&mut mb.down_out.data) as *mut c_void,
                m as i32,
                h as i32,
                inter as i32,
                eps,
                inter.div_ceil(128) as i32,
                h.div_ceil(128) as i32,
                st.ptr_mut(&mut mb.dep) as *mut i32,
                st.ptr_mut(&mut mb.count) as *mut i32,
                st.ptr_mut(&mut mb.start) as *mut i32,
                st.ptr_mut(&mut mb.ready) as *mut i32,
                2,
                0,
                st.raw() as _,
            )
        };
        assert_eq!(rc, 0, "mega 返回 {rc}");
        st.synchronize().unwrap();

        // ---- 逐段比较，第一处不一致即根因所在（只在首轮打印细节）----
        let stages: [(&str, &CudaSlice<bf16>, &CudaSlice<bf16>, usize); 6] = [
            ("1_pre_ff_norm", &ref_norm1.data, &mb.norm1.data, m * h),
            ("2_gate", &gate_out.data, &mb.gate_buf.data, m * inter),
            ("3_up", &up_out.data, &mb.up_buf.data, m * inter),
            ("4_gelu", &act_bf.data, &mb.act_bf.data, m * inter),
            ("5_down", &dn_out.data, &mb.down_out.data, m * h),
            ("6_post_ff_norm", &ref_ff1.data, &mega_ff1.data, m * h),
        ];
        for (si, (tag, a, b, n)) in stages.iter().enumerate() {
            let nd = ndiff(&st, a, b, *n);
            total[si] += nd;
            if nd > 0 && first_bad.is_none() {
                first_bad = Some(si);
                cmp(&st, a, b, *n, tag); // 打印首个不一致段的统计
            }
        }
        if it == 0 {
            for (si, (tag, a, b, n)) in stages.iter().enumerate() {
                if total[si] == 0 {
                    cmp(&st, a, b, *n, tag);
                }
            }
        }
    }

    let names = ["1_pre_ff_norm", "2_gate", "3_up", "4_gelu", "5_down", "6_post_ff_norm"];
    println!("---- {iters} 组输入累计（每组 m·维度） ----");
    let mut ok = true;
    for (si, name) in names.iter().enumerate() {
        let denom = iters * if si == 1 || si == 2 || si == 3 { m * inter } else { m * h };
        println!("  {name}: 累计不同 {}/{}", total[si], denom);
        ok &= total[si] == 0;
    }
    println!("VERDICT: {}", if ok { "全段逐位一致" } else { "有段不一致（见上，第一处即根因）" });
    assert!(ok, "真实权重下有段不一致");
}

/// 只数不一致元素个数（不打印），用于多轮累计。
fn ndiff(stream: &Stream, a: &CudaSlice<bf16>, b: &CudaSlice<bf16>, n: usize) -> usize {
    let va = stream.dtoh_sync_copy(a).unwrap();
    let vb = stream.dtoh_sync_copy(b).unwrap();
    (0..n).filter(|&i| va[i].to_bits() != vb[i].to_bits()).count()
}
