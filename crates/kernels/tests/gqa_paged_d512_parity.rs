//! B26（loop kaon3-decode-opt）：`gqa_paged_win_bf16` D=512 的 v6 PAGED 快分支 vs 通用慢路径（cfg=-1），
//! bf16 cache 与 e4m3 cache（KV_FP8 模板，同步读展开）两种。混批形状：q_len 37 / 300 / 1 行，带历史。
//! 判据：① v6 vs ref max|Δ| ≤ 阈（ref fp32 标量累加 / v6 bf16 mma，不逐位）；② v6 bf16 cache vs v6 e4m3 cache **位差 0**
//! （cache 元素是往返值，两侧读到同数值，mma 序相同）。
//! `cargo test --release -p dinfer-kernels --test gqa_paged_d512_parity -- --ignored --nocapture`
use cudarc::driver::CudaDevice as Device;
use half::bf16;

struct Rng(u64);
impl Rng {
    fn uni(&mut self) -> f32 {
        self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
    fn normal(&mut self) -> f32 {
        let (u1, u2) = (self.uni().max(1e-7), self.uni());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

#[test]
#[ignore]
fn gqa_paged_d512_parity() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let (h, hkv, d, bs) = (8usize, 2usize, 512usize, 32usize);
    // (seq_len, q_len)：行 0 = 混批里的 chunk（带历史），行 1 = 大 chunk，行 2 = decode 行（q_len 1，skip_single=0 两侧都算）
    let reqs: Vec<(usize, usize)> = vec![(600, 37), (300, 300), (77, 1)];
    let b = reqs.len();
    let total: usize = reqs.iter().map(|r| r.0).sum();
    let tq: usize = reqs.iter().map(|r| r.1).sum();
    let max_q = reqs.iter().map(|r| r.1).max().unwrap();
    let mut rng = Rng(0x512_b26);
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
    let mut dq = up(&q);
    let mut dk = up(&k);
    let mut dv = up(&v);
    // kv_fp8 层口径：q/k/v 先 e4m3 往返，cache 元素 = 往返值
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
    let dcu = st.htod_copy(cu).unwrap();
    let run = |cache: &cudarc::driver::CudaSlice<bf16>, f8: bool, cfg: i32| -> Vec<bf16> {
        let mut out = st.alloc_zeros::<bf16>(tq * h * d).unwrap();
        let rc = unsafe {
            dinfer_kernels::ffi::gqa_paged_win_bf16(
                st.ptr(&dq) as *const std::ffi::c_void, st.ptr(cache) as *const std::ffi::c_void, st.ptr(&dbt) as *const i32,
                st.ptr(&dseq) as *const i32, st.ptr(&dcu) as *const i32, st.ptr_mut(&mut out) as *mut std::ffi::c_void,
                b as i64, max_q as i64, h as i64, hkv as i64, d as i64, bs as i64, max_blocks as i64, 1.0, 0, 0, cfg, 0,
                0, f8 as i32, st.raw() as _,
            )
        };
        assert_eq!(rc, 0, "gqa_paged_win_bf16 rc (f8={f8} cfg={cfg})");
        st.synchronize().unwrap();
        st.dtoh_sync_copy(&out).unwrap()
    };
    let r_ref = run(&cache_bf, false, -1);
    let r_v6 = run(&cache_bf, false, 0);
    let r_v6f8 = run(&cache_f8, true, 0);
    let r_reff8 = run(&cache_f8, true, -1);
    let mut fails = Vec::new();
    let stat = |a: &[bf16], c: &[bf16]| -> (f32, usize, usize) {
        let mut mx = 0f32; let mut nd = 0; let mut nz = 0;
        for (x, y) in a.iter().zip(c) {
            if x.to_bits() != y.to_bits() { nd += 1; }
            if x.to_f32() != 0.0 { nz += 1; }
            mx = mx.max((x.to_f32() - y.to_f32()).abs());
        }
        (mx, nd, nz)
    };
    let amax = r_ref.iter().map(|x| x.to_f32().abs()).fold(0f32, f32::max);
    let tol = (amax * 2e-2).max(2e-2); // bf16 输出 8 bit 尾数：1 ulp ≈ 0.4%·|x|，mma 与标量 fp32 序差几 ulp
    let (m1, n1, z1) = stat(&r_v6, &r_ref);
    println!("D=512 PAGED v6(bf16 cache) vs ref: max|Δ|={m1:.4} (amax {amax:.3}, tol {tol:.4}) ndiff={n1}/{} nonzero={z1}", tq * h * d);
    if m1 > tol || z1 == 0 { fails.push(format!("v6 bf16 vs ref max {m1}")); }
    let (m2, n2, _) = stat(&r_v6f8, &r_v6);
    println!("D=512 PAGED v6 e4m3 cache vs v6 bf16 cache: 位差 {n2} max|Δ|={m2:.4} => {}", if n2 == 0 { "ok" } else { "FAIL" });
    if n2 != 0 { fails.push(format!("v6 fp8 vs bf16 bitwise {n2}")); }
    let (m3, n3, _) = stat(&r_reff8, &r_ref);
    println!("D=512 PAGED ref e4m3 vs ref bf16: 位差 {n3} max|Δ|={m3:.4}（参照自洽）");
    assert!(fails.is_empty(), "失败: {fails:?}");
}
