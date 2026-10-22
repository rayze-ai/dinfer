//! sm_120 低精度 GEMM 对 CPU 参考：FP8 静态激活 scale（`forward_fp8`）与 NVFP4 激活全局 scale（`forward_nvfp4`）。
//! 参考侧逐式复现 kernel 的激活量化（e4m3 饱和 RNE；NVFP4 块缩放 = e4m3(gs·amax/6)、值 = e2m1(x·gs/sf)），
//! 权重直接反量化，f64 累加。kernel 用 rcp.approx 而参考用精确除法，个别元素舍入可能不同 → 只比 GEMM 输出（容差 2%）。
//! 运行：`CUDA_ARCHS=120 cargo test --release -p dinfer-kernels --test gemm_lowp_parity -- --ignored --nocapture`
use dinfer_kernels::{bf16, Device, Gemm};

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

fn e4m3_to_f32(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let (e, m) = (((b >> 3) & 0xf) as i32, (b & 7) as f32);
    s * if e == 0 { m / 8.0 * 2f32.powi(-6) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) }
}
fn f32_to_e4m3(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs();
    if a >= 448.0 {
        return sign | 0x7e;
    }
    if a < 2f32.powi(-6) {
        return sign | (a * 512.0).round_ties_even() as u8;
    }
    let mut e = a.log2().floor() as i32;
    let mut m = ((a / 2f32.powi(e) - 1.0) * 8.0).round_ties_even() as i32;
    if m == 8 {
        m = 0;
        e += 1;
    }
    if e + 7 > 15 || (e + 7 == 15 && m > 6) {
        return sign | 0x7e;
    }
    sign | (((e + 7) as u8) << 3) | m as u8
}
const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
/// e2m1 就近偶数、饱和到 ±6（与 cvt.rn.satfinite.e2m1x2.f32 一致）
fn e2m1(x: f32) -> f32 {
    let a = x.abs().min(6.0);
    let mut best = 0usize;
    for i in 1..8 {
        let (d, db) = ((a - E2M1[i]).abs(), (a - E2M1[best]).abs());
        if d < db || (d == db && i % 2 == 0) {
            best = i;
        }
    }
    if x < 0.0 { -E2M1[best] } else { E2M1[best] }
}

#[test]
#[ignore]
fn gemm_fp8_static_scale_parity() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let gemm = Gemm::new(&st).unwrap();
    let (n, k) = (256usize, 2816usize);
    let mut fails = Vec::new();
    for &(m, a_scale, amp) in &[(1usize, 1.0f32, 1.0f32), (9, 0.05, 1.0), (70, 3.0, 300.0), (9, 1.0, 300.0)] {
        let mut r = Rng(7 + m as u64);
        let a: Vec<f32> = (0..m * k).map(|_| bf16::from_f32(amp * r.normal()).to_f32()).collect();
        let w: Vec<u8> = (0..n * k).map(|_| f32_to_e4m3(r.normal() * 8.0)).collect();
        let w_scale = 0.003f32;
        let da = st.htod_copy(a.iter().map(|&x| bf16::from_f32(x)).collect::<Vec<_>>()).unwrap();
        let dw = st.htod_copy(w.clone()).unwrap();
        let mut dout = st.alloc_zeros::<bf16>(m * n).unwrap();
        gemm.forward_fp8(&da, a_scale, &dw, w_scale, &mut dout, m, n, k).unwrap();
        let got = st.dtoh_sync_copy(&dout).unwrap();
        let inv = 1.0 / a_scale;
        let aq: Vec<f64> = a.iter().map(|&x| e4m3_to_f32(f32_to_e4m3(x * inv)) as f64 * a_scale as f64).collect();
        let (mut bad, mut worst) = (0usize, 0f64);
        for i in 0..m {
            for j in 0..n {
                let want: f64 = (0..k).map(|t| aq[i * k + t] * e4m3_to_f32(w[j * k + t]) as f64).sum::<f64>() * w_scale as f64;
                let g = got[i * n + j].to_f32() as f64;
                let e = (g - want).abs() / want.abs().max(1e-3 * amp as f64);
                worst = worst.max(e);
                bad += (e > 2e-2) as usize;
            }
        }
        println!("FP8   m={m:<3} a_scale={a_scale:<5} amp={amp:<5}: 最大相对误差 {worst:.2e} 超差 {bad}/{}", m * n);
        if bad > 0 {
            fails.push(format!("fp8 m={m} a_scale={a_scale}: bad={bad}"));
        }
    }
    assert!(fails.is_empty(), "{fails:?}");
}

#[test]
#[ignore]
fn gemm_nvfp4_gscale_parity() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let gemm = Gemm::new(&st).unwrap();
    let n = 256usize;
    let mut fails = Vec::new();
    // K=2816 走 BK=128；K=704（专家 down_proj）走 BK=64
    for &(m, k, gs, amp) in &[(1usize, 2816usize, 1.0f32, 1.0f32), (9, 2816, 900.0, 1.0), (70, 704, 25.0, 4.0), (9, 704, 1.0, 4.0)] {
        let mut r = Rng(11 + m as u64 + k as u64);
        let a: Vec<f32> = (0..m * k).map(|_| bf16::from_f32(amp * r.normal()).to_f32()).collect();
        let packed: Vec<u8> = (0..n * k / 2).map(|_| (r.uni() * 256.0) as u8).collect();
        let wsc: Vec<u8> = (0..n * k / 16).map(|i| [0x30u8, 0x38, 0x40, 0x3c, 0x34][i % 5]).collect();
        let global = 0.002f32;
        let da = st.htod_copy(a.iter().map(|&x| bf16::from_f32(x)).collect::<Vec<_>>()).unwrap();
        let (dp, ds) = (st.htod_copy(packed.clone()).unwrap(), st.htod_copy(wsc.clone()).unwrap());
        let mut dout = st.alloc_zeros::<bf16>(m * n).unwrap();
        // 接口约定：global 是最终输出乘子 = w_global · input_scale = w_global / gs（vLLM 的 alpha）
        gemm.forward_nvfp4(&da, gs, &dp, &ds, global / gs, &mut dout, m, n, k).unwrap();
        let got = st.dtoh_sync_copy(&dout).unwrap();
        // 参考：A 按块量化（gs 网格）后反量化回实数
        let mut aq = vec![0f64; m * k];
        for i in 0..m {
            for b in 0..k / 16 {
                let blk = &a[i * k + b * 16..][..16];
                let amax = blk.iter().fold(0f32, |x, y| x.max(y.abs()));
                if amax == 0.0 {
                    continue;
                }
                let sfd = e4m3_to_f32(f32_to_e4m3(gs * (amax / 6.0)));
                if sfd == 0.0 {
                    continue;
                }
                let scale = gs / sfd;
                for t in 0..16 {
                    aq[i * k + b * 16 + t] = e2m1(blk[t] * scale) as f64 * (sfd / gs) as f64;
                }
            }
        }
        let wv = |j: usize, t: usize| {
            let byte = packed[j * (k / 2) + t / 2];
            let nib = if t % 2 == 0 { byte & 0xf } else { byte >> 4 };
            let v = E2M1[(nib & 7) as usize] * if nib & 8 != 0 { -1.0 } else { 1.0 };
            v as f64 * e4m3_to_f32(wsc[j * (k / 16) + t / 16]) as f64
        };
        let (mut bad, mut worst) = (0usize, 0f64);
        for i in 0..m {
            for j in 0..n {
                let want: f64 = (0..k).map(|t| aq[i * k + t] * wv(j, t)).sum::<f64>() * global as f64;
                let g = got[i * n + j].to_f32() as f64;
                let e = (g - want).abs() / want.abs().max(0.05 * amp as f64 * global as f64 * (k as f64).sqrt());
                worst = worst.max(e);
                bad += (e > 2e-2) as usize;
            }
        }
        println!("NVFP4 m={m:<3} k={k:<5} gs={gs:<5} amp={amp:<3}: 最大相对误差 {worst:.2e} 超差 {bad}/{}", m * n);
        if bad > 0 {
            fails.push(format!("nvfp4 m={m} k={k} gs={gs}: bad={bad}"));
        }
    }
    assert!(fails.is_empty(), "{fails:?}");
}
