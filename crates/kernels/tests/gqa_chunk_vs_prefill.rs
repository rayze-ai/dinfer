//! D142（loop kaon3-decode-opt）：同一 q/k/v，21 行一次 `prefill_win`（连续 K/V，fresh）vs 写 cache 后 `paged_win` 只算行 16..21（hist=16，chunk）。
//! 两者行 16–20 的输出应 ≤ ulp（v5/v6 对 ref 都 ≤ 0.016）。差大 ⇒ cache 写 / paged 读的位置或内容有形状相关 bug。
//! 场景：D=256 nkv=8 win=1024（滑窗层）、D=512 nkv=2 win=0（全注意力层，k_eq_v：v=k）、prompt 21 / 400 / 1200；cfg 默认(464→v6) 与 0(v5)。
//! `cargo test --release -p dinfer-kernels --test gqa_chunk_vs_prefill -- --ignored --nocapture`
use dinfer_kernels::{bf16, Device};

struct Rng(u64);
impl Rng {
    fn uni(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; (self.0 >> 40) as f32 / (1u64 << 24) as f32 }
    fn normal(&mut self) -> f32 { let (u1, u2) = (self.uni().max(1e-7), self.uni()); (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() }
}

#[test]
#[ignore]
fn gqa_chunk_vs_prefill() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let (h, bs) = (16usize, 32usize);
    let mut fails = Vec::new();
    // (D, hkv, window, k_eq_v, total, hist)
    let cases: Vec<(usize, usize, usize, bool, usize, usize)> = vec![
        (256, 8, 1024, false, 21, 16), (512, 2, 0, true, 21, 16),
        (256, 8, 1024, false, 400, 384), (512, 2, 0, true, 400, 384),
        (256, 8, 1024, false, 1200, 1184), (256, 8, 1024, false, 25, 20), (512, 2, 0, true, 25, 20),
    ];
    for (d, hkv, window, k_eq_v, total, hist) in cases {
        let ql = total - hist;
        let mut rng = Rng(0xc0ffee + d as u64 * 7 + total as u64);
        let rb = |x: f32| bf16::from_f32(x).to_f32();
        let q: Vec<f32> = (0..total * h * d).map(|_| rb(2.0 * rng.normal())).collect();
        let k: Vec<f32> = (0..total * hkv * d).map(|_| rb(2.0 * rng.normal())).collect();
        let v: Vec<f32> = if k_eq_v { k.clone() } else { (0..total * hkv * d).map(|_| rb(rng.normal())).collect() };
        let up = |x: &[f32]| st.htod_copy(x.iter().map(|&f| bf16::from_f32(f)).collect::<Vec<_>>()).unwrap();
        let mut dq = up(&q); let mut dk = up(&k); let mut dv = up(&v);
        dinfer_kernels::ops::fp8_e4m3_roundtrip_bf16_n(&st, &mut dq, total * h * d, 1.0).unwrap();
        dinfer_kernels::ops::fp8_e4m3_roundtrip_bf16_n(&st, &mut dk, total * hkv * d, 1.0).unwrap();
        dinfer_kernels::ops::fp8_e4m3_roundtrip_bf16_n(&st, &mut dv, total * hkv * d, 1.0).unwrap();
        // ---- fresh：一次 prefill_win（连续）----
        let cu1 = st.htod_copy(vec![0i32, total as i32]).unwrap();
        let mut out_fresh = st.alloc_zeros::<bf16>(total * h * d).unwrap();
        dinfer_kernels::gqa_prefill_win_bf16(&st, &dq, &dk, &dv, &mut out_fresh, &cu1, 1, total, h, hkv, d, 1.0, 0, 0, window, false).unwrap(); // p_fp8=false 与下面 paged_win 的 p_fp8=0 一致
        // ---- chunk：全部 K/V 写 cache（模拟前一步写了 0..hist，本步写 hist..total），paged_win 只算 q 行 hist..total ----
        let nblk = total.div_ceil(bs);
        let bt: Vec<i32> = (1..=nblk as i32).collect();
        let slots: Vec<i64> = (0..total).map(|t| ((1 + t / bs) * bs + t % bs) as i64).collect();
        let dsl = st.htod_copy(slots).unwrap();
        let mut cache = st.alloc_zeros::<bf16>((nblk + 1) * bs * hkv * 2 * d).unwrap();
        dinfer_kernels::gqa_cache_write_bf16(&st, &dk, &dv, &mut cache, &dsl, total, hkv, d, bs, false).unwrap();
        let dbt = st.htod_copy(bt).unwrap();
        let dseq = st.htod_copy(vec![total as i32]).unwrap();
        let cu2 = st.htod_copy(vec![0i32, ql as i32]).unwrap();
        // q 的 chunk 行 = 原 q 的行 hist..total（连续切片视图：用 slice）
        let dq_chunk = dq.slice(hist * h * d..total * h * d);
        for cfg in [464i32, 0, -1] {
            let mut out_chunk = st.alloc_zeros::<bf16>(ql * h * d).unwrap();
            let rc = unsafe {
                dinfer_kernels::ffi::gqa_paged_win_bf16(
                    st.ptr(&dq_chunk) as *const std::ffi::c_void, st.ptr(&cache) as *const std::ffi::c_void, st.ptr(&dbt) as *const i32,
                    st.ptr(&dseq) as *const i32, st.ptr(&cu2) as *const i32, st.ptr_mut(&mut out_chunk) as *mut std::ffi::c_void,
                    1, ql as i64, h as i64, hkv as i64, d as i64, bs as i64, nblk as i64, 1.0, 0, 0, cfg, window as i32, 1, 0, st.raw() as _,
                )
            };
            assert_eq!(rc, 0, "paged_win rc D={d} cfg={cfg}");
            st.synchronize().unwrap();
            let f = st.dtoh_sync_copy(&out_fresh).unwrap();
            let c = st.dtoh_sync_copy(&out_chunk).unwrap();
            let (mut mx, mut nd, mut amax) = (0f32, 0usize, 0f32);
            for r in 0..ql {
                for j in 0..h * d {
                    let a = f[(hist + r) * h * d + j].to_f32(); let b = c[r * h * d + j].to_f32();
                    amax = amax.max(a.abs()); if a.to_bits() != b.to_bits() { nd += 1; } mx = mx.max((a - b).abs());
                }
            }
            let tol = (amax * 2e-2).max(2e-2);
            println!("D={d} win={window} k_eq_v={k_eq_v} total={total} hist={hist} ql={ql} cfg={cfg}: chunk vs fresh rows[{hist}..{total}] max|Δ|={mx:.4} nd={nd}/{} amax={amax:.2} => {}", ql * h * d, if mx <= tol { "ok" } else { "FAIL" });
            if mx > tol { fails.push(format!("D={d} total={total} cfg={cfg} mx={mx}")); }
        }
    }
    assert!(fails.is_empty(), "失败: {fails:?}");
}
