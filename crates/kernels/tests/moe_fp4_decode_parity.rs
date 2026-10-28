//! MoE decode NVFP4 grouped GEMV（`moe_fp4_gateup_act` / `moe_fp4_gemv`）对 CPU f64 参照。
//! 场景 = Kaon-V3 MoE：K=2816 / inter=704 / down N=2816；专家 4 个，行分布含空专家与 >M_TILE 的专家（chunk 循环）。
//! 判据：bf16 输出 |err| ≤ 0.01 + 0.01·|ref|（单项 fp4·fp4·e4m3·e4m3 在 f32 精确，差只在累加序）。
//! 运行：`cargo test --release -p dinfer-kernels --test moe_fp4_decode_parity -- --ignored --nocapture`
//! loop kaon3-decode-opt B7。
use dinfer_kernels::{bf16, ops, Device};

struct Rng(u64);
impl Rng {
    fn uni(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
    fn nib(&mut self) -> u8 { (self.uni() * 16.0) as u8 & 15 }
    /// e4m3 码，exp 3..11，正（scale 永远正）
    fn e4m3_pos(&mut self) -> u8 { let e = 3 + (self.uni() * 9.0) as u8; let m = (self.uni() * 8.0) as u8 & 7; (e << 3) | m }
}
fn e2m1(n: u8) -> f32 { let mag = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(n & 7) as usize]; if n & 8 != 0 { -mag } else { mag } }
fn e4m3(c: u8) -> f32 {
    let s = if c & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((c >> 3) & 0xf) as i32; let m = (c & 7) as f32;
    if e == 0 { s * m / 8.0 * 2f32.powi(-6) } else { s * (1.0 + m / 8.0) * 2f32.powi(e - 7) }
}
fn rb(x: f32) -> f32 { bf16::from_f32(x).to_f32() }
fn gelu_tanh(x: f32) -> f32 { 0.5 * x * (1.0 + (0.7978845608028654f32 * (x + 0.044715 * x * x * x)).tanh()) }
/// 行 r 与权重行 w 的块缩放点积（f64）：Σ_blk (Σ_16 a·b)·sa·sb
fn dot(aq: &[u8], asc: &[u8], wq: &[u8], wsc: &[u8], k: usize) -> f64 {
    let mut s = 0f64;
    for b in 0..k / 16 {
        let mut p = 0f64;
        for i in 0..8 {
            let ab = aq[b * 8 + i]; let wb = wq[b * 8 + i];
            p += e2m1(ab & 15) as f64 * e2m1(wb & 15) as f64 + e2m1(ab >> 4) as f64 * e2m1(wb >> 4) as f64;
        }
        s += p * e4m3(asc[b]) as f64 * e4m3(wsc[b]) as f64;
    }
    s
}

#[test]
#[ignore]
fn moe_fp4_decode_parity() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let mut fails = Vec::new();
    // (K_gate(=hidden), inter, rows per expert)
    let cases: Vec<(usize, usize, Vec<usize>)> = vec![
        (2816, 704, vec![1, 1, 1, 1]),          // c1：每专家 1 行
        (2816, 704, vec![0, 3, 1, 2]),          // 空专家 + 不等
        (2816, 704, vec![9, 0, 4, 1]),          // 9 > M_TILE=8 → chunk 循环
        (256, 128, vec![2, 5, 0, 0]),           // 小形状
    ];
    for (kg, inter, rows_e) in cases {
        let e = rows_e.len();
        let r: usize = rows_e.iter().sum();
        let mut rng = Rng(0xb7 + kg as u64 * 7 + r as u64 * 131);
        let mut offsets = vec![0i32; e + 1];
        for i in 0..e { offsets[i + 1] = offsets[i] + rows_e[i] as i32; }
        let (kd, nd) = (inter, kg); // down: K=inter, N=hidden
        // A（gate/up 输入）[r, kg] fp4 + scale [r, kg/16]
        let aq: Vec<u8> = (0..r * kg / 2).map(|_| rng.nib() | (rng.nib() << 4)).collect();
        let asc: Vec<u8> = (0..r * kg / 16).map(|_| rng.e4m3_pos()).collect();
        // 专家权重：gate/up [inter, kg]，down [nd, kd]；各自 scale；alpha
        let mut gw = Vec::new(); let mut gs = Vec::new(); let mut uw = Vec::new(); let mut us = Vec::new();
        let mut dw = Vec::new(); let mut ds = Vec::new();
        let (mut ag, mut au, mut ad) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..e {
            gw.push((0..inter * kg / 2).map(|_| rng.nib() | (rng.nib() << 4)).collect::<Vec<u8>>());
            gs.push((0..inter * kg / 16).map(|_| rng.e4m3_pos()).collect::<Vec<u8>>());
            uw.push((0..inter * kg / 2).map(|_| rng.nib() | (rng.nib() << 4)).collect::<Vec<u8>>());
            us.push((0..inter * kg / 16).map(|_| rng.e4m3_pos()).collect::<Vec<u8>>());
            dw.push((0..nd * kd / 2).map(|_| rng.nib() | (rng.nib() << 4)).collect::<Vec<u8>>());
            ds.push((0..nd * kd / 16).map(|_| rng.e4m3_pos()).collect::<Vec<u8>>());
            ag.push(1e-4 * (0.5 + rng.uni())); au.push(1e-4 * (0.5 + rng.uni())); ad.push(1e-4 * (0.5 + rng.uni()));
        }
        // CPU 参照：gelu_o[r, inter]
        let mut gelu_ref = vec![0f32; r * inter];
        for ex in 0..e { for row in offsets[ex] as usize..offsets[ex + 1] as usize { for n in 0..inter {
            let a = &aq[row * kg / 2..(row + 1) * kg / 2]; let a_s = &asc[row * kg / 16..(row + 1) * kg / 16];
            let g = rb((dot(a, a_s, &gw[ex][n * kg / 2..(n + 1) * kg / 2], &gs[ex][n * kg / 16..(n + 1) * kg / 16], kg) * ag[ex] as f64) as f32);
            let u = rb((dot(a, a_s, &uw[ex][n * kg / 2..(n + 1) * kg / 2], &us[ex][n * kg / 16..(n + 1) * kg / 16], kg) * au[ex] as f64) as f32);
            gelu_ref[row * inter + n] = rb(rb(gelu_tanh(g)) * u);
        } } }
        // down 输入：独立随机 fp4（隔离 gelu 阶段），[r, kd]
        let dq: Vec<u8> = (0..r * kd / 2).map(|_| rng.nib() | (rng.nib() << 4)).collect();
        let dsc: Vec<u8> = (0..r * kd / 16).map(|_| rng.e4m3_pos()).collect();
        let mut down_ref = vec![0f32; r * nd];
        for ex in 0..e { for row in offsets[ex] as usize..offsets[ex + 1] as usize { for n in 0..nd {
            down_ref[row * nd + n] = (dot(&dq[row * kd / 2..(row + 1) * kd / 2], &dsc[row * kd / 16..(row + 1) * kd / 16],
                &dw[ex][n * kd / 2..(n + 1) * kd / 2], &ds[ex][n * kd / 16..(n + 1) * kd / 16], kd) * ad[ex] as f64) as f32;
        } } }
        // GPU
        let d_aq = st.htod_copy(aq).unwrap(); let d_asc = st.htod_copy(asc).unwrap();
        let d_dq = st.htod_copy(dq).unwrap(); let d_dsc = st.htod_copy(dsc).unwrap();
        let d_off = st.htod_copy(offsets.clone()).unwrap();
        let up_all = |v: &Vec<Vec<u8>>| -> (Vec<dinfer_kernels::cudarc::driver::CudaSlice<u8>>, Vec<u64>) {
            let slices: Vec<_> = v.iter().map(|x| st.htod_copy(x.clone()).unwrap()).collect();
            let ptrs: Vec<u64> = slices.iter().map(|s| st.ptr(s)).collect();
            (slices, ptrs)
        };
        let (_g, gp) = up_all(&gw); let (_gs, gsp) = up_all(&gs); let (_u, upp) = up_all(&uw); let (_us, usp) = up_all(&us);
        let (_d, dp) = up_all(&dw); let (_ds, dsp) = up_all(&ds);
        let d_gp = st.htod_copy(gp).unwrap(); let d_gsp = st.htod_copy(gsp).unwrap();
        let d_up = st.htod_copy(upp).unwrap(); let d_usp = st.htod_copy(usp).unwrap();
        let d_dp = st.htod_copy(dp).unwrap(); let d_dsp = st.htod_copy(dsp).unwrap();
        let d_ag = st.htod_copy(ag).unwrap(); let d_au = st.htod_copy(au).unwrap(); let d_ad = st.htod_copy(ad).unwrap();
        let mut d_gelu = st.alloc_zeros::<bf16>(r * inter).unwrap();
        let mut d_down = st.alloc_zeros::<bf16>(r * nd).unwrap();
        ops::moe_fp4_gateup_act(&st, &d_aq, &d_asc, &d_gp, &d_gsp, &d_ag, &d_up, &d_usp, &d_au, &d_off, &mut d_gelu, r, inter, kg, e).unwrap();
        ops::moe_fp4_gemv(&st, &d_dq, &d_dsc, &d_dp, &d_dsp, &d_ad, &d_off, &mut d_down, r, nd, kd, e).unwrap();
        st.synchronize().unwrap();
        let gelu_got = st.dtoh_sync_copy(&d_gelu).unwrap();
        let down_got = st.dtoh_sync_copy(&d_down).unwrap();
        let judge = |got: &[bf16], refv: &[f32]| -> (usize, f32) {
            let (mut bad, mut worst) = (0usize, 0f32);
            for (g, r) in got.iter().zip(refv) { let e = (g.to_f32() - r).abs(); worst = worst.max(e); if e > 0.01 + 0.01 * r.abs() { bad += 1; } }
            (bad, worst)
        };
        let (gb, gwst) = judge(&gelu_got, &gelu_ref);
        let (db, dwst) = judge(&down_got, &down_ref);
        let ok = gb == 0 && db == 0;
        println!("K={kg} inter={inter} rows={rows_e:?}: gateup_act 最大误差 {gwst:.4} 超差 {gb}/{} | down 最大误差 {dwst:.4} 超差 {db}/{} => {}",
            r * inter, r * nd, if ok { "ok" } else { "FAIL" });
        if !ok { fails.push(format!("K={kg} rows={rows_e:?}")); }
    }
    assert!(fails.is_empty(), "失败: {fails:?}");
}
