//! GQA prefill（含滑窗、大分数）：tensor-core kernel（cfg 0 = v5、cfg 3464 = v6）与通用慢路径（cfg −1）各自对 CPU f64 参考。
//!
//! 背景（2026-10-22，Kaon-V3 / Gemma4）：scaling = 1.0、QK-norm 后分数量级 O(100)，端到端里 v5 与慢路径互差 20 nat。
//! 运行：`cargo test --release -p dinfer-kernels --test gqa_window_parity -- --ignored --nocapture`
use dinfer_kernels::{bf16, ffi, Device};
use std::ffi::c_void;

struct Rng(u64);
impl Rng {
    fn uni(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
    /// 近似标准正态（4 个均匀数之和）
    fn normal(&mut self) -> f32 {
        (self.uni() + self.uni() + self.uni() + self.uni() - 2.0) * 1.732
    }
}

#[allow(clippy::too_many_arguments)]
fn cpu_ref(q: &[f32], k: &[f32], v: &[f32], t: usize, h: usize, hkv: usize, d: usize, window: usize) -> Vec<f32> {
    let mut out = vec![0f32; t * h * d];
    for i in 0..t {
        let j0 = if window > 0 { (i + 1).saturating_sub(window) } else { 0 };
        for hh in 0..h {
            let hk = hh / (h / hkv);
            let qi = &q[(i * h + hh) * d..][..d];
            let s: Vec<f64> = (j0..=i)
                .map(|j| qi.iter().zip(&k[(j * hkv + hk) * d..][..d]).map(|(a, b)| *a as f64 * *b as f64).sum())
                .collect();
            let m = s.iter().copied().fold(f64::MIN, f64::max);
            let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
            let z: f64 = e.iter().sum();
            for c in 0..d {
                let acc: f64 = (j0..=i).zip(&e).map(|(j, p)| p * v[(j * hkv + hk) * d + c] as f64).sum();
                out[(i * h + hh) * d + c] = (acc / z) as f32;
            }
        }
    }
    out
}

#[test]
#[ignore]
fn gqa_window_large_scores_parity() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let (h, hkv, d) = (16usize, 8usize, 256usize);
    let mut fails = Vec::new();
    // (T, window, q/k 幅度)：幅度 0.1 → softmax 平；2.0 → 分数 std≈64，极尖（Gemma4 的实际情形）
    for &(t, window, amp) in &[
        (25usize, 1024usize, 0.1f32),
        (25, 1024, 2.0),
        (25, 1024, 5.0),
        (25, 1024, 10.0),
        (25, 0, 2.0),
        (300, 64, 2.0),
        (300, 0, 0.5),
    ] {
        let mut rng = Rng(0x1234_5678 + t as u64 * 7 + window as u64);
        let rb = |x: f32| bf16::from_f32(x).to_f32();
        let q: Vec<f32> = (0..t * h * d).map(|_| rb(amp * rng.normal())).collect();
        let k: Vec<f32> = (0..t * hkv * d).map(|_| rb(amp * rng.normal())).collect();
        let v: Vec<f32> = (0..t * hkv * d).map(|_| rb(rng.normal())).collect();
        let want = cpu_ref(&q, &k, &v, t, h, hkv, d, window);
        let up = |x: &[f32]| st.htod_copy(x.iter().map(|&f| bf16::from_f32(f)).collect::<Vec<_>>()).unwrap();
        let (dq, dk, dv) = (up(&q), up(&k), up(&v));
        let cu = st.htod_copy(vec![0i32, t as i32]).unwrap();
        // −1 慢路径；0 v5；432 v6；2432 v6B；3464 v6F；1432 / 1864 v7（cfg 编码见 gqa.cu v6_dispatch）
        for cfg in [-1, 0, 432, 2432, 3464, 1432, 1864] {
            let mut out = st.alloc_zeros::<bf16>(t * h * d).unwrap();
            let rc = unsafe {
                ffi::gqa_prefill_win_bf16(
                    st.ptr(&dq) as *const c_void,
                    st.ptr(&dk) as *const c_void,
                    st.ptr(&dv) as *const c_void,
                    st.ptr_mut(&mut out) as *mut c_void,
                    st.ptr(&cu) as *const i32,
                    1,
                    t as i64,
                    h as i64,
                    hkv as i64,
                    d as i64,
                    1.0,
                    0,
                    0,
                    cfg,
                    window as i32,
                    st.raw() as _,
                )
            };
            st.synchronize().unwrap();
            let got = st.dtoh_sync_copy(&out).unwrap();
            let (mut worst, mut bad) = (0f32, 0usize);
            for (g, w) in got.iter().zip(&want) {
                let e = (g.to_f32() - w).abs();
                worst = worst.max(e);
                bad += (e > 0.03 + 0.02 * w.abs()) as usize;
            }
            println!("T={t:>3} win={window:>4} amp={amp:<4} cfg={cfg:>5}: rc={rc} 最大误差 {worst:.4}  超差 {bad}/{}", got.len());
            // rc≠0 = 该档在本卡不可用（如共享内存超 99 KB），只打印；数值只判 rc==0 的档
            if rc == 0 && bad > 0 {
                fails.push(format!("T={t} win={window} amp={amp} cfg={cfg}: rc={rc} bad={bad} worst={worst}"));
            }
        }
    }
    assert!(fails.is_empty(), "超差：{fails:#?}");
}
