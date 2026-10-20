//! 全注意力 prefill kernel（`gqa_prefill_bf16` → `gqa_fa_v5_kernel`）微基准（r11a，docs/loops/qwen35-attn-r11a.md）。
//! Qwen3.5-27B 形状 H=24 / Hkv=4 / D=256，causal 单请求 T ∈ {1024, 3009, 6019}（与 engine-gap 9e 的 nsys 分解同口径）。
//! 输出一行一形状：`[attn T=6019] fa_us=2040 tflops=218.1`，`scripts/dev/gqa_bench_to_probe.py` 转 probe.csv。
//! 变体靠环境变量（`GQA_PREFILL_NBUF` 等）在进程级切换，shell 侧多跑几次。
//! 运行：`cargo test -p dinfer-kernels --test gqa_fa_bench -- --ignored --nocapture`
#![cfg(not(feature = "no-nvcc"))]

use dinfer_kernels::{bf16, Device, Stream};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn normal(&mut self) -> f32 {
        let u1 = ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32).max(1e-7);
        let u2 = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

fn time_us(st: &Stream, iters: usize, mut f: impl FnMut()) -> f64 {
    f();
    st.synchronize().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        f();
    }
    st.synchronize().unwrap();
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

fn bench_t(st: &Stream, t: usize, h: usize, hk: usize, d: usize, iters: usize) {
    let w = (h + 2 * hk) * d;
    let mut r = Rng::new(0xA11A + t as u64);
    let qkv: Vec<bf16> = (0..t * w).map(|_| bf16::from_f32(r.normal())).collect();
    let dqkv = st.htod_copy(qkv).unwrap();
    let dcu = st.htod_copy(vec![0i32, t as i32]).unwrap();
    let mut out = st.alloc_zeros::<bf16>(t * h * d).unwrap();
    let qv = dqkv.slice(..);
    let kv = dqkv.slice(h * d..);
    let vv = dqkv.slice((h + hk) * d..);
    let scale = (d as f32).powf(-0.5);
    let us = time_us(st, iters, || {
        dinfer_kernels::gqa_prefill_bf16(st, &qv, &kv, &vv, &mut out, &dcu, 1, t, h, hk, d, scale, w, w).unwrap();
    });
    // causal：QK^T 与 PV 各 T²/2 · D 次乘加 → 2·T²·D·H FLOP
    let flop = 2.0 * (t as f64) * (t as f64) * (d as f64) * (h as f64);
    // 自对拍（B300 无 torch）：与 v5 旧路径（cfg 0，3090 ops_parity 已过）逐元素比，bf16 输出链一 ulp ≈ 4e-3·|x|
    let cfg = std::env::var("GQA_PREFILL_CFG").ok().and_then(|s| s.parse::<i32>().ok()).unwrap_or(-1);
    let mut diff = String::new();
    if cfg != 0 {
        let mut out_ref = st.alloc_zeros::<bf16>(t * h * d).unwrap();
        let rc = unsafe {
            dinfer_kernels::ffi::gqa_prefill_bf16(
                st.ptr(&qv) as *const _, st.ptr(&kv) as *const _, st.ptr(&vv) as *const _,
                st.ptr_mut(&mut out_ref) as *mut _, st.ptr(&dcu) as *const i32, 1, t as i64, h as i64, hk as i64, d as i64, scale,
                w as i64, w as i64, 0, st.raw() as _,
            )
        };
        assert_eq!(rc, 0, "v5 ref launch");
        let a = st.dtoh_sync_copy(&out).unwrap();
        let b = st.dtoh_sync_copy(&out_ref).unwrap();
        let (mut mx, mut dot, mut na, mut nb) = (0f32, 0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b.iter()) {
            let (x, y) = (x.to_f32(), y.to_f32());
            mx = mx.max((x - y).abs());
            dot += (x * y) as f64; na += (x * x) as f64; nb += (y * y) as f64;
        }
        let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-30);
        diff = format!(" vs_v5_maxabs={mx:.4} cos={cos:.7}");
        assert!(cos > 0.9999, "v6 cfg {cfg} 与 v5 偏离：cos {cos}");
    }
    println!("[attn T={t}] fa_us={us:.0} tflops={:.1}{diff}", flop / us / 1e6);
}

#[test]
#[ignore]
fn gqa_fa_bench() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let iters: usize = std::env::var("GQA_BENCH_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    for t in [1024usize, 3009, 6019] {
        bench_t(&st, t, 24, 4, 256, iters);
    }
}
