//! decode dense MLP fp8 路径（`mlp_fp8_norm_quant` / `mlp_fp8_gateup_act` / `mlp_fp8_gemv`）对 CPU f64 参照。
//! 场景 = Kaon-V3 dense MLP：h=2816 / inter=2112，M ∈ {1,3,8,16}；另一组小形状。
//! 判据：fp8 输出码差 ≤ 1（累加序不同可在舍入边界翻一码）；bf16 输出 |err| ≤ 0.01 + 0.01·|ref|。
//! 运行：`cargo test --release -p dinfer-kernels --test mlp_fp8_decode_parity -- --ignored --nocapture`
//! loop kaon3-decode-opt B5。
use dinfer_kernels::{bf16, ops, Device};

struct Rng(u64);
impl Rng {
    fn uni(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
    fn normal(&mut self) -> f32 {
        (self.uni() + self.uni() + self.uni() + self.uni() - 2.0) * 1.732
    }
}

/// e4m3 码 → f32（0x7f/0xff = NaN，这里不生成）。
fn e4m3_dec(c: u8) -> f32 {
    let s = if c & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((c >> 3) & 0xf) as i32;
    let m = (c & 7) as f32;
    if e == 0 { s * m / 8.0 * 2f32.powi(-6) } else { s * (1.0 + m / 8.0) * 2f32.powi(e - 7) }
}
/// f32 → e4m3（最近、同距取偶、SATFINITE）：对 254 个有限码穷举。
fn e4m3_enc(v: f32) -> u8 {
    let v = v.clamp(-448.0, 448.0);
    let mut best = 0u8;
    let mut bd = f32::INFINITY;
    for c in 0u16..=255 {
        let c = c as u8;
        if c & 0x7f == 0x7f { continue; }
        let d = (e4m3_dec(c) - v).abs();
        if d < bd || (d == bd && (c & 1) == 0) { bd = d; best = c; }
    }
    best
}
fn rb(x: f32) -> f32 { bf16::from_f32(x).to_f32() }
fn gelu_tanh(x: f32) -> f32 { 0.5 * x * (1.0 + (0.7978845608028654f32 * (x + 0.044715 * x * x * x)).tanh()) }
fn code_diff(a: u8, b: u8) -> i32 {
    // 符号-幅值码按数值序比较：把 (s, mag) 映射到有符号整数
    let f = |c: u8| -> i32 { let m = (c & 0x7f) as i32; if c & 0x80 != 0 { -m } else { m } };
    (f(a) - f(b)).abs()
}

#[test]
#[ignore]
fn mlp_fp8_decode_parity() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let mut fails = Vec::new();
    // (m, h, inter)
    let cases = [(1usize, 2816usize, 2112usize), (3, 2816, 2112), (8, 2816, 2112), (16, 2816, 2112), (5, 256, 160)];
    for (m, h, inter) in cases {
        let mut rng = Rng(0xb5 + m as u64 * 977 + h as u64);
        let eps = 1e-6f32;
        let (gs_a, dn_a) = (0.05f32, 0.08f32); // 激活 a_scale（gate/up 共用、down 单独）
        let (gw_s, uw_s, dw_s) = (0.0021f32, 0.0019f32, 0.0017f32);
        let x: Vec<f32> = (0..m * h).map(|_| rb(rng.normal())).collect();
        let w: Vec<f32> = (0..h).map(|_| rb(1.0 + 0.3 * rng.normal())).collect();
        // 权重 fp8 码：exp 4..11 范围内随机（避免 NaN / 极端幅值）
        let mut wq = |n: usize| -> Vec<u8> { (0..n).map(|_| { let e = 4 + (rng.uni() * 8.0) as u8; let mant = (rng.uni() * 8.0) as u8 & 7; let s = if rng.uni() < 0.5 { 0x80 } else { 0 }; s | (e << 3) | mant }).collect() };
        let gw = wq(inter * h);
        let uw = wq(inter * h);
        let dw = wq(h * inter);
        // ---- CPU 参照 ----
        let mut q_ref = vec![0u8; m * h];
        for r in 0..m {
            let row = &x[r * h..(r + 1) * h];
            let ss: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
            let inv = 1.0 / ((ss / h as f64) + eps as f64).sqrt();
            for i in 0..h { let nv = rb((row[i] as f64 * inv) as f32 * w[i]); q_ref[r * h + i] = e4m3_enc(nv / gs_a); }
        }
        let dot = |a: &[u8], wrow: &[u8]| -> f64 { a.iter().zip(wrow).map(|(&p, &q)| e4m3_dec(p) as f64 * e4m3_dec(q) as f64).sum() };
        let mut act_ref = vec![0u8; m * inter];
        let mut act_val = vec![0f32; m * inter];
        for r in 0..m {
            let aq = &q_ref[r * h..(r + 1) * h];
            for n in 0..inter {
                let g = rb((dot(aq, &gw[n * h..(n + 1) * h]) * (gw_s * gs_a) as f64) as f32);
                let u = rb((dot(aq, &uw[n * h..(n + 1) * h]) * (uw_s * gs_a) as f64) as f32);
                let y = rb(gelu_tanh(g));
                let act = rb(y * u);
                act_val[r * inter + n] = act;
                act_ref[r * inter + n] = e4m3_enc(act / dn_a);
            }
        }
        let mut down_ref = vec![0f32; m * h];
        for r in 0..m {
            let aq = &act_ref[r * inter..(r + 1) * inter];
            for n in 0..h { down_ref[r * h + n] = (dot(aq, &dw[n * inter..(n + 1) * inter]) * (dw_s * dn_a) as f64) as f32; }
        }
        // ---- GPU ----
        let dx = st.htod_copy(x.iter().map(|&f| bf16::from_f32(f)).collect::<Vec<_>>()).unwrap();
        let dw_norm = st.htod_copy(w.iter().map(|&f| bf16::from_f32(f)).collect::<Vec<_>>()).unwrap();
        let mut dq = st.alloc_zeros::<u8>(m * h).unwrap();
        ops::mlp_fp8_norm_quant(&st, &dx, &dw_norm, &mut dq, m, h, eps, 1.0 / gs_a).unwrap();
        let dgw = st.htod_copy(gw.clone()).unwrap();
        let duw = st.htod_copy(uw.clone()).unwrap();
        let ddw = st.htod_copy(dw.clone()).unwrap();
        let mut dact = st.alloc_zeros::<u8>(m * inter).unwrap();
        // gateup 用 CPU 的 q_ref 作输入（隔离 norm 阶段的 ±1 码差）
        let dq_ref = st.htod_copy(q_ref.clone()).unwrap();
        ops::mlp_fp8_gateup_act(&st, &dq_ref, &dgw, &duw, &mut dact, m, inter, h, gw_s * gs_a, uw_s * gs_a, 1.0 / dn_a).unwrap();
        let dact_ref = st.htod_copy(act_ref.clone()).unwrap();
        let mut dout = st.alloc_zeros::<bf16>(m * h).unwrap();
        ops::mlp_fp8_gemv(&st, &dact_ref, &ddw, &mut dout, m, h, inter, dw_s * dn_a).unwrap();
        st.synchronize().unwrap();
        let q_got = st.dtoh_sync_copy(&dq).unwrap();
        let act_got = st.dtoh_sync_copy(&dact).unwrap();
        let out_got = st.dtoh_sync_copy(&dout).unwrap();
        // ---- 判 ----
        let (mut q_bad, mut q_max) = (0usize, 0i32);
        for (a, b) in q_got.iter().zip(&q_ref) { let d = code_diff(*a, *b); q_max = q_max.max(d); if d > 1 { q_bad += 1; } }
        let q_eq = q_got.iter().zip(&q_ref).filter(|(a, b)| a == b).count();
        let (mut a_bad, mut a_max) = (0usize, 0i32);
        for (a, b) in act_got.iter().zip(&act_ref) { let d = code_diff(*a, *b); a_max = a_max.max(d); if d > 1 { a_bad += 1; } }
        let a_eq = act_got.iter().zip(&act_ref).filter(|(a, b)| a == b).count();
        let (mut o_bad, mut o_worst) = (0usize, 0f32);
        for (g, r) in out_got.iter().zip(&down_ref) { let e = (g.to_f32() - r).abs(); o_worst = o_worst.max(e); if e > 0.01 + 0.01 * r.abs() { o_bad += 1; } }
        let ok = q_bad == 0 && a_bad == 0 && o_bad == 0;
        println!(
            "m={m} h={h} inter={inter}: norm_q 全同 {q_eq}/{} 最大码差 {q_max} 超差 {q_bad} | act_q 全同 {a_eq}/{} 最大码差 {a_max} 超差 {a_bad} | down 最大误差 {o_worst:.4} 超差 {o_bad}/{} => {}",
            m * h, m * inter, m * h, if ok { "ok" } else { "FAIL" }
        );
        if !ok { fails.push(format!("m={m} h={h} inter={inter}")); }
    }
    assert!(fails.is_empty(), "失败: {fails:?}");
}
