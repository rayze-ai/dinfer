//! D93（loop kaon3-decode-opt）：cacheprobe 实证 c1 串行 **prefix-cache 命中后的带历史 chunk**（Paged 步，hist>0、q_len 11–17）
//! 在 `KV_FP8_STORE=1` 下输出改变、`=0` 下不变 → 复现 fa_v5 PAGED + KV_FP8 + window + hist>0 多行 chunk。
//! 判据：cfg=0（v5 PAGED）bf16 cache vs e4m3 cache 位差 0（元素同值、同 kernel 结构）；并与 cfg=-1 慢路径（两种 cache）比 max|Δ|。
//! `cargo test --release -p dinfer-kernels --test gqa_paged_fp8_hist_parity -- --ignored --nocapture`
use dinfer_kernels::{bf16, Device};

struct Rng(u64);
impl Rng {
    fn uni(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; (self.0 >> 40) as f32 / (1u64 << 24) as f32 }
    fn normal(&mut self) -> f32 { let (u1, u2) = (self.uni().max(1e-7), self.uni()); (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() }
}

#[test]
#[ignore]
fn gqa_paged_fp8_hist_parity() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let (h, bs) = (16usize, 32usize);
    // (D, hkv, window, [(seq_len, q_len)])：滑窗层 D=256 nkv=8 win=1024；全注意力 D=512 nkv=2 win=0
    let cases: Vec<(usize, usize, usize, Vec<(usize, usize)>)> = vec![
        (256, 8, 1024, vec![(400, 16), (413, 17), (1200, 14), (90, 11), (33, 33), (1, 1)]),
        (256, 8, 1024, vec![(1300, 16), (2100, 300)]),
        (512, 2, 0, vec![(400, 16), (413, 17), (1200, 14), (1, 1)]),
    ];
    let mut fails = Vec::new();
    for (d, hkv, window, reqs) in cases {
        let b = reqs.len();
        let total: usize = reqs.iter().map(|r| r.0).sum();
        let tq: usize = reqs.iter().map(|r| r.1).sum();
        let max_q = reqs.iter().map(|r| r.1).max().unwrap();
        let mut rng = Rng(0xd93 + d as u64 + total as u64);
        let rb = |x: f32| bf16::from_f32(x).to_f32();
        let q: Vec<f32> = (0..tq * h * d).map(|_| rb(2.0 * rng.normal())).collect();
        let k: Vec<f32> = (0..total * hkv * d).map(|_| rb(2.0 * rng.normal())).collect();
        let v: Vec<f32> = (0..total * hkv * d).map(|_| rb(rng.normal())).collect();
        let max_blocks = reqs.iter().map(|r| r.0.div_ceil(bs)).max().unwrap();
        let num_blocks: usize = reqs.iter().map(|r| r.0.div_ceil(bs)).sum::<usize>() + 1;
        let mut bt = vec![0i32; b * max_blocks];
        let mut slots = Vec::with_capacity(total);
        let mut next_blk = 1i32;
        for (i, &(n, _)) in reqs.iter().enumerate() {
            let nb = n.div_ceil(bs);
            for j in 0..nb { bt[i * max_blocks + j] = next_blk + j as i32; }
            for t in 0..n { slots.push(((next_blk as usize + t / bs) * bs + t % bs) as i64); }
            next_blk += nb as i32;
        }
        let mut cu = vec![0i32];
        for &(_, ql) in &reqs { cu.push(cu.last().unwrap() + ql as i32); }
        let up = |x: &[f32]| st.htod_copy(x.iter().map(|&f| bf16::from_f32(f)).collect::<Vec<_>>()).unwrap();
        let mut dq = up(&q); let mut dk = up(&k); let mut dv = up(&v);
        dinfer_kernels::ops::fp8_e4m3_roundtrip_bf16_n(&st, &mut dq, tq * h * d, 1.0).unwrap();
        dinfer_kernels::ops::fp8_e4m3_roundtrip_bf16_n(&st, &mut dk, total * hkv * d, 1.0).unwrap();
        dinfer_kernels::ops::fp8_e4m3_roundtrip_bf16_n(&st, &mut dv, total * hkv * d, 1.0).unwrap();
        let dsl = st.htod_copy(slots).unwrap();
        let elems = num_blocks * bs * hkv * 2 * d;
        let mut cache_bf = st.alloc_zeros::<bf16>(elems).unwrap();
        let mut cache_f8 = st.alloc_zeros::<bf16>(elems / 2).unwrap();
        dinfer_kernels::gqa_cache_write_bf16(&st, &dk, &dv, &mut cache_bf, &dsl, total, hkv, d, bs, false).unwrap();
        dinfer_kernels::gqa_cache_write_bf16(&st, &dk, &dv, &mut cache_f8, &dsl, total, hkv, d, bs, true).unwrap();
        let dbt = st.htod_copy(bt).unwrap();
        let dseq = st.htod_copy(reqs.iter().map(|r| r.0 as i32).collect::<Vec<_>>()).unwrap();
        let dcu = st.htod_copy(cu.clone()).unwrap();
        let run = |cache: &dinfer_kernels::cudarc::driver::CudaSlice<bf16>, f8: bool, cfg: i32| -> Vec<bf16> {
            let mut out = st.alloc_zeros::<bf16>(tq * h * d).unwrap();
            let rc = unsafe {
                dinfer_kernels::ffi::gqa_paged_win_bf16(
                    st.ptr(&dq) as *const std::ffi::c_void, st.ptr(cache) as *const std::ffi::c_void, st.ptr(&dbt) as *const i32,
                    st.ptr(&dseq) as *const i32, st.ptr(&dcu) as *const i32, st.ptr_mut(&mut out) as *mut std::ffi::c_void,
                    b as i64, max_q as i64, h as i64, hkv as i64, d as i64, bs as i64, max_blocks as i64, 1.0, 0, 0, cfg, window as i32,
                    0, f8 as i32, st.raw() as _,
                )
            };
            assert_eq!(rc, 0, "rc D={d} f8={f8} cfg={cfg}");
            st.synchronize().unwrap();
            st.dtoh_sync_copy(&out).unwrap()
        };
        // cfg：D=256 用 0（v5 PAGED，kv_fp8 时 serve 实际也是 v5：cfg≥1000 被降 0）；D=512 用 0（B26 v6）
        let fast_bf = run(&cache_bf, false, 0);
        let fast_f8 = run(&cache_f8, true, 0);
        let ref_bf = run(&cache_bf, false, -1);
        let ref_f8 = run(&cache_f8, true, -1);
        let stat = |a: &[bf16], c: &[bf16]| -> (f32, usize) {
            let mut mx = 0f32; let mut nd = 0;
            for (x, y) in a.iter().zip(c) { if x.to_bits() != y.to_bits() { nd += 1; } mx = mx.max((x.to_f32() - y.to_f32()).abs()); }
            (mx, nd)
        };
        let amax = ref_bf.iter().map(|x| x.to_f32().abs()).fold(0f32, f32::max);
        let tol = (amax * 2e-2).max(2e-2);
        let (m1, n1) = stat(&fast_f8, &fast_bf);
        let (m2, n2) = stat(&fast_bf, &ref_bf);
        let (m3, n3) = stat(&fast_f8, &ref_bf);
        let (m4, n4) = stat(&ref_f8, &ref_bf);
        // 逐请求：哪个 (seq,q) 的行错
        let mut per = Vec::new();
        for (i, &(sl, ql)) in reqs.iter().enumerate() {
            let r0 = cu[i] as usize * h * d; let r1 = cu[i + 1] as usize * h * d;
            let (mx, nd) = stat(&fast_f8[r0..r1], &fast_bf[r0..r1]);
            per.push(format!("(seq{sl},q{ql}):nd={nd}/{} mx={mx:.3}", r1 - r0));
        }
        println!("D={d} win={window} reqs={reqs:?}\n  fast e4m3 vs fast bf16: nd={n1} mx={m1:.4} | fast bf16 vs ref: nd={n2} mx={m2:.4} (tol {tol:.3}) | fast e4m3 vs ref: mx={m3:.4} nd={n3} | ref e4m3 vs ref bf16: nd={n4} mx={m4:.4}\n  per-req: {}", per.join(" "));
        if n1 != 0 { fails.push(format!("D={d} fast e4m3≠bf16 nd={n1}")); }
        if m2 > tol { fails.push(format!("D={d} fast vs ref mx={m2}")); }
        if n4 != 0 { fails.push(format!("D={d} ref e4m3≠bf16 nd={n4}")); }
    }
    assert!(fails.is_empty(), "失败: {fails:?}");
}
