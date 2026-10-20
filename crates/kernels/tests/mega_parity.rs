//! 质变实验第一阶段：Qwen3.5 单 GDN 层 decode 的 persistent kernel（mega_sm100.cu）vs 现路径 11 个 kernel。
//! 规格 / 记录：docs/loops/mega-sm100.md。
//!
//! 两边同输入、同随机权重（27B 形状，B ∈ {1, 8, 32}）；基线按生产口径走 Gemm 自调优 + CUDA graph，
//! 实验侧一次 launch 也进 graph；各重放 MEGA_ITERS（默认 1000）次取均值 = ms/层；对拍 res（层输出）与递归状态。
//! 跑法：cargo test -p dinfer-kernels --features ref-tests --test mega_parity -- --ignored --nocapture
//! 环境：MEGA_B=1,8,32（B 列表）、MEGA_PARAMS=groups=4,unroll=4,stages=3,ctas=1（覆盖 ArchProfile）、
//!       MEGA_SKIP_PARITY=1（只计时）、MEGA_MIN_SPEEDUP（断言任一档 ≥ 该倍率，默认不断言）。
#![cfg(feature = "ref-tests")]

use dinfer_kernels::mega::{mega_gdn_layer_decode, MegaScratch, MegaShape, MegaWeights};
use dinfer_kernels::{bf16, ops, ops::GdnAct, ArchProfile, Device, Gemm, GemmTuneEntry, Stream};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unif(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

fn rand_bf16(s: &Stream, rng: &mut Rng, n: usize, scale: f32) -> cudarc::driver::CudaSlice<bf16> {
    let v: Vec<bf16> = (0..n).map(|_| bf16::from_f32(rng.unif() * scale)).collect();
    s.htod_copy(v).unwrap()
}
fn rand_f32(s: &Stream, rng: &mut Rng, n: usize, scale: f32) -> cudarc::driver::CudaSlice<f32> {
    let v: Vec<f32> = (0..n).map(|_| rng.unif() * scale).collect();
    s.htod_copy(v).unwrap()
}
fn const_bf16(s: &Stream, n: usize, v: f32) -> cudarc::driver::CudaSlice<bf16> {
    s.htod_copy(vec![bf16::from_f32(v); n]).unwrap()
}

/// (cos, max_rel) —— max_rel 以 ref 的最大绝对值做分母（bf16 输出链 ulp 口径见 docs/knowledge/kernel-rules.md）。
fn compare(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut dot, mut na, mut nb, mut maxd, mut maxb) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += (x as f64).powi(2);
        nb += (y as f64).powi(2);
        maxd = maxd.max((x as f64 - y as f64).abs());
        maxb = maxb.max((y as f64).abs());
    }
    (dot / (na.sqrt() * nb.sqrt()).max(1e-30), maxd / maxb.max(1e-30))
}

fn to_f32(v: &[bf16]) -> Vec<f32> {
    v.iter().map(|x| x.to_f32()).collect()
}

#[test]
#[ignore]
fn mega_gdn_layer_decode_parity_and_timing() {
    let dev = Device::new(0).unwrap();
    dev.disable_event_tracking();
    let s = dev.new_stream().unwrap();
    let prof = ArchProfile::detect(&dev).unwrap();
    let params = prof.mega_params();
    println!("GPU {} cc={:?} sm={} arch={:?} mega_params={:?}", dev.name().unwrap(), prof.cc, prof.sm_count, prof.arch, params);

    let sh = MegaShape { hidden: 5120, inter: 17408, heads_k: 16, heads_v: 48, k_dim: 128, v_dim: 128, width: 4, eps: 1e-6, scale: 128f32.powf(-0.5) };
    let (hid, inter, hv, kd, vd, width) = (sh.hidden, sh.inter, sh.heads_v, sh.k_dim, sh.v_dim, sh.width);
    let conv_dim = sh.conv_dim();
    let in_row = sh.in_row();
    let (z_off, a_off, b_off) = (conv_dim, conv_dim + hv * vd, conv_dim + hv * vd + hv);
    let bs: Vec<usize> = std::env::var("MEGA_B")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| vec![1, 8, 32]);
    let bmax = bs.iter().copied().max().unwrap();
    let iters: usize = std::env::var("MEGA_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(1000);
    let skip_parity = std::env::var("MEGA_SKIP_PARITY").is_ok();

    let mut rng = Rng(0x9e3779b97f4a7c15);
    // 权重：标准差 ~ 1/sqrt(K)（随机 bf16，形状对就行）
    let in_w = rand_bf16(&s, &mut rng, in_row * hid, 0.02);
    let out_w = rand_bf16(&s, &mut rng, hid * hv * vd, 0.02);
    let gate_up_w = rand_bf16(&s, &mut rng, 2 * inter * hid, 0.02);
    let down_w = rand_bf16(&s, &mut rng, hid * inter, 0.01);
    let conv_w = rand_bf16(&s, &mut rng, conv_dim * width, 0.5);
    let norm_w = rand_bf16(&s, &mut rng, hid, 1.0);
    let post_norm_w = rand_bf16(&s, &mut rng, hid, 1.0);
    let gnorm_w = rand_bf16(&s, &mut rng, vd, 1.0);
    let a_log = rand_f32(&s, &mut rng, hv, 1.0);
    let dt_bias = rand_f32(&s, &mut rng, hv, 1.0);
    let _ = const_bf16(&s, 1, 0.0);
    // 初态
    let res0 = rand_bf16(&s, &mut rng, bmax * hid, 1.0);
    let conv0 = rand_bf16(&s, &mut rng, bmax * conv_dim * (width - 1), 1.0);
    let rec0 = rand_f32(&s, &mut rng, bmax * hv * vd * kd, 0.1);
    let cu: Vec<i32> = (0..=bmax as i32).collect();
    let cu_seqlens = s.htod_copy(cu).unwrap();

    // 基线缓冲
    let mut res_b = s.alloc_zeros::<bf16>(bmax * hid).unwrap();
    let mut conv_b = s.alloc_zeros::<bf16>(conv0.len()).unwrap();
    let mut rec_b = s.alloc_zeros::<f32>(rec0.len()).unwrap();
    let mut xn = s.alloc_zeros::<bf16>(bmax * hid).unwrap();
    let mut qkvzab = s.alloc_zeros::<bf16>(bmax * in_row).unwrap();
    let mut conv_out = s.alloc_zeros::<bf16>(bmax * conv_dim).unwrap();
    let mut o = s.alloc_zeros::<bf16>(bmax * hv * vd).unwrap();
    let mut n = s.alloc_zeros::<bf16>(bmax * hv * vd).unwrap();
    let mut h2 = s.alloc_zeros::<bf16>(bmax * hid).unwrap();
    let mut xn2 = s.alloc_zeros::<bf16>(bmax * hid).unwrap();
    let mut gu = s.alloc_zeros::<bf16>(bmax * 2 * inter).unwrap();
    let mut act = s.alloc_zeros::<bf16>(bmax * inter).unwrap();
    let mut d = s.alloc_zeros::<bf16>(bmax * hid).unwrap();
    // 实验缓冲
    let mut res_m = s.alloc_zeros::<bf16>(bmax * hid).unwrap();
    let mut conv_m = s.alloc_zeros::<bf16>(conv0.len()).unwrap();
    let mut rec_m = s.alloc_zeros::<f32>(rec0.len()).unwrap();
    let w = MegaWeights { norm_w: &norm_w, post_norm_w: &post_norm_w, in_w: &in_w, conv_w: &conv_w, a_log: &a_log, dt_bias: &dt_bias, gnorm_w: &gnorm_w, out_w: &out_w, gate_up_w: &gate_up_w, down_w: &down_w };
    let mut sc = MegaScratch::new(&s, &sh, bmax, params, prof.sm_count).unwrap();
    println!("mega splits (in/out/gu/down) = {:?}  tile_n={}", sc.splits, params.tile_n());

    // Gemm 自调优（生产口径：Linear::forward_rows 走 plan 表）
    let gemm = Gemm::new(&s).unwrap();
    let entries = vec![
        GemmTuneEntry { name: "gdn.in_proj".into(), n: in_row, k: hid, w: &in_w, bias: None },
        GemmTuneEntry { name: "gdn.out_proj".into(), n: hid, k: hv * vd, w: &out_w, bias: None },
        GemmTuneEntry { name: "mlp.gate_up".into(), n: 2 * inter, k: hid, w: &gate_up_w, bias: None },
        GemmTuneEntry { name: "mlp.down".into(), n: hid, k: inter, w: &down_w, bias: None },
    ];
    let rep = gemm.autotune(&entries).unwrap();
    for l in rep.summary_lines().iter().take(12) {
        println!("  {l}");
    }
    s.synchronize().unwrap();

    let reset = |s: &Stream, res: &mut cudarc::driver::CudaSlice<bf16>, conv: &mut cudarc::driver::CudaSlice<bf16>, rec: &mut cudarc::driver::CudaSlice<f32>| {
        s.dtod_copy_n(&res0, res, res0.len()).unwrap();
        s.dtod_copy_n(&conv0, conv, conv0.len()).unwrap();
        s.dtod_copy_n(&rec0, rec, rec0.len()).unwrap();
    };

    let mut results = Vec::new();
    for &b in &bs {
        let mut baseline = |s: &Stream, res_b: &mut cudarc::driver::CudaSlice<bf16>, conv_b: &mut cudarc::driver::CudaSlice<bf16>, rec_b: &mut cudarc::driver::CudaSlice<f32>| {
            ops::rmsnorm_bf16_n(s, res_b, &norm_w, &mut xn, b, hid, sh.eps).unwrap();
            gemm.forward_bf16(&xn, &in_w, None, &mut qkvzab, b, in_row, hid).unwrap();
            ops::gdn_conv1d_update_strided(s, &qkvzab.slice(..), &conv_w, &mut conv_out, conv_b, None, b, conv_dim, in_row, width, width - 1, GdnAct::Silu).unwrap();
            ops::gdn_recurrent_strided(s, &conv_out, None, None, Some(&qkvzab.slice(a_off..)), Some(&qkvzab.slice(b_off..)), in_row, &a_log, &dt_bias, &mut o, rec_b, &cu_seqlens, None, b, b, sh.heads_k, hv, kd, vd, sh.scale, None, None, None).unwrap();
            ops::gdn_gated_rmsnorm_strided(s, &o, &gnorm_w, &qkvzab.slice(z_off..), &mut n, b * hv, vd, sh.eps, GdnAct::Silu, hv, in_row).unwrap();
            gemm.forward_bf16(&n, &out_w, None, &mut h2, b, hid, hv * vd).unwrap();
            ops::fused_add_rmsnorm_bf16_n(s, &h2, res_b, &post_norm_w, &mut xn2, b, hid, sh.eps).unwrap();
            gemm.forward_bf16(&xn2, &gate_up_w, None, &mut gu, b, 2 * inter, hid).unwrap();
            ops::silu_and_mul_bf16_n(s, &gu, &mut act, b, inter).unwrap();
            gemm.forward_bf16(&act, &down_w, None, &mut d, b, hid, inter).unwrap();
            ops::add_inplace_bf16_n(s, &d, res_b, b * hid).unwrap();
        };

        // ---- 对拍：各跑一次（eager），比 res 与 rec_state
        reset(&s, &mut res_b, &mut conv_b, &mut rec_b);
        reset(&s, &mut res_m, &mut conv_m, &mut rec_m);
        s.synchronize().unwrap();
        baseline(&s, &mut res_b, &mut conv_b, &mut rec_b);
        s.synchronize().unwrap();
        mega_gdn_layer_decode(&s, &sh, &w, &mut sc, &mut res_m, &mut conv_m, &mut rec_m, None, b).unwrap();
        s.synchronize().unwrap();
        println!("B={b}: mega grid={} occ={} smem={} B", sc.info[0], sc.info[1], sc.info[2]);
        let (mut cos_res, mut rel_res, mut cos_rec, mut rel_rec, mut cos_conv, mut rel_conv) = (1.0, 0.0, 1.0, 0.0, 1.0, 0.0);
        if !skip_parity {
            let rb = to_f32(&s.dtoh_sync_copy(&res_b).unwrap()[..b * hid]);
            let rm = to_f32(&s.dtoh_sync_copy(&res_m).unwrap()[..b * hid]);
            (cos_res, rel_res) = compare(&rm, &rb);
            let kb = s.dtoh_sync_copy(&rec_b).unwrap();
            let km = s.dtoh_sync_copy(&rec_m).unwrap();
            (cos_rec, rel_rec) = compare(&km[..b * hv * vd * kd], &kb[..b * hv * vd * kd]);
            let cb = to_f32(&s.dtoh_sync_copy(&conv_b).unwrap());
            let cm = to_f32(&s.dtoh_sync_copy(&conv_m).unwrap());
            (cos_conv, rel_conv) = compare(&cm[..b * conv_dim * (width - 1)], &cb[..b * conv_dim * (width - 1)]);
            println!("B={b}: parity res cos={cos_res:.6} max_rel={rel_res:.3e} | rec cos={cos_rec:.6} max_rel={rel_rec:.3e} | conv cos={cos_conv:.6} max_rel={rel_conv:.3e}");
        }

        if std::env::var("MEGA_PROF").is_ok() {
            // 阶段计时：对拍读完后再 eager 跑一次（状态再推一步，不影响计时口径），读 prof
            sc.profile = true;
            sc.reset_prof(&s).unwrap();
            mega_gdn_layer_decode(&s, &sh, &w, &mut sc, &mut res_m, &mut conv_m, &mut rec_m, None, b).unwrap();
            s.synchronize().unwrap();
            sc.profile = false;
            let pr = sc.read_prof(&s).unwrap();
            let grid = sc.info[0] as f64;
            println!("B={b}: stage      start→end (us)   span   Σwait/grid  Σbusy/grid   (Σbusy/span/grid = 占用率)");
            for (i, q) in pr.iter().enumerate() {
                let span = q.end_us - q.start_us;
                println!(
                    "  {:<10} {:>7.1}→{:>7.1} {:>7.1} {:>10.1} {:>10.1}   {:>5.2}",
                    dinfer_kernels::mega::MEGA_STAGE_NAMES[i], q.start_us, q.end_us, span, q.wait_us / grid, q.busy_us / grid,
                    q.busy_us / grid / span.max(1e-9)
                );
            }
        }
        // ---- 计时：两边都进 graph，重放 iters 次
        reset(&s, &mut res_b, &mut conv_b, &mut rec_b);
        s.begin_capture().unwrap();
        baseline(&s, &mut res_b, &mut conv_b, &mut rec_b);
        let g_base = s.end_capture().unwrap();
        reset(&s, &mut res_m, &mut conv_m, &mut rec_m);
        s.begin_capture().unwrap();
        mega_gdn_layer_decode(&s, &sh, &w, &mut sc, &mut res_m, &mut conv_m, &mut rec_m, None, b).unwrap();
        let g_mega = s.end_capture().unwrap();
        let time = |g: &dinfer_kernels::Graph| -> f32 {
            for _ in 0..20 {
                g.launch().unwrap();
            }
            s.synchronize().unwrap();
            let (e0, e1) = (s.new_timing_event().unwrap(), s.new_timing_event().unwrap());
            s.record_timing(&e0).unwrap();
            for _ in 0..iters {
                g.launch().unwrap();
            }
            s.record_timing(&e1).unwrap();
            s.synchronize().unwrap();
            dinfer_kernels::TimingEvent::elapsed_ms(&e0, &e1).unwrap() / iters as f32
        };
        // 两边都先重置状态（重放中状态会漂，不影响计时）
        reset(&s, &mut res_b, &mut conv_b, &mut rec_b);
        let t_base = time(&g_base);
        reset(&s, &mut res_m, &mut conv_m, &mut rec_m);
        let t_mega = time(&g_mega);
        let t_base2 = time(&g_base);
        let t_mega2 = time(&g_mega);
        let (tb, tm) = (t_base.min(t_base2), t_mega.min(t_mega2));
        println!("B={b}: baseline {:.1} us/layer (graph, 11 kernels) | mega {:.1} us/layer | speedup_B{b} = {:.3}x", tb * 1e3, tm * 1e3, tb / tm);
        results.push((b, tb, tm, cos_res, rel_res, cos_rec, rel_rec, cos_conv, rel_conv));
    }
    println!("\nsummary: B | baseline us | mega us | speedup | res cos / max_rel | rec cos / max_rel");
    for (b, tb, tm, c, r, ck, rk, _, _) in &results {
        println!("  {b:>2} | {:>8.1} | {:>8.1} | speedup_B{b} = {:.3}x | {c:.6} / {r:.2e} | {ck:.6} / {rk:.2e}", tb * 1e3, tm * 1e3, tb / tm);
    }
    if !skip_parity {
        for (b, _, _, c, r, ck, rk, cc, rc) in &results {
            assert!(*c >= 0.9999 && *r <= 1e-2, "B={b}: res parity cos={c} max_rel={r}");
            assert!(*ck >= 0.9999 && *rk <= 2e-3, "B={b}: rec_state parity cos={ck} max_rel={rk}");
            assert!(*cc >= 0.9999 && *rc <= 1e-2, "B={b}: conv_state parity cos={cc} max_rel={rc}");
        }
    }
    if let Some(min) = std::env::var("MEGA_MIN_SPEEDUP").ok().and_then(|v| v.parse::<f32>().ok()) {
        let best = results.iter().map(|r| r.1 / r.2).fold(0f32, f32::max);
        assert!(best >= min, "best speedup {best:.3} < {min}");
    }
}
