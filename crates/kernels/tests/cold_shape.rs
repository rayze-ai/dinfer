//! 诊断：新形状（新的 M）首次调用的冷停顿来自哪里。对 GEMM（cuBLASLt）与 gqa_prefill 分别计时：每个 M 调两次，
//! 打印 cold / warm 毫秒。跑法：cargo test -p dinfer-kernels --test cold_shape -- --ignored --nocapture
use dinfer_kernels::{Device, Gemm};
use half::bf16;
use std::time::Instant;

#[test]
#[ignore]
fn cold_vs_warm_by_shape() {
    let dev = Device::new(0).unwrap();
    let s = dev.default_stream();
    let g = Gemm::new(&s).unwrap();
    let (hidden, qkv_w, inter) = (1024usize, 4096usize, 3072usize);
    let max_m = 2048;
    let a = s.alloc_zeros::<bf16>(max_m * inter).unwrap();
    let w1 = s.alloc_zeros::<bf16>(qkv_w * hidden).unwrap();
    let w2 = s.alloc_zeros::<bf16>(2 * inter * hidden).unwrap();
    let w3 = s.alloc_zeros::<bf16>(hidden * inter).unwrap();
    let mut out = s.alloc_zeros::<bf16>(max_m * 2 * inter).unwrap();
    let time = |f: &mut dyn FnMut()| {
        s.synchronize().unwrap();
        let t = Instant::now();
        f();
        s.synchronize().unwrap();
        t.elapsed().as_secs_f64() * 1e3
    };
    println!("M      gemm_qkv cold/warm   gemm_gateup cold/warm   gemm_down cold/warm   (ms)");
    for &m in &[389usize, 639, 889, 1139, 1514, 1900] {
        let mut r = |i: usize| {
            let t1 = time(&mut || g.forward_bf16(&a, &w1, None, &mut out, m, qkv_w, hidden).unwrap());
            let t2 = time(&mut || g.forward_bf16(&a, &w2, None, &mut out, m, 2 * inter, hidden).unwrap());
            let t3 = time(&mut || g.forward_bf16(&a, &w3, None, &mut out, m, hidden, inter).unwrap());
            (i, t1, t2, t3)
        };
        let (_, c1, c2, c3) = r(0);
        let (_, w1t, w2t, w3t) = r(1);
        println!("{m:<6} {c1:>8.2}/{w1t:<8.2}   {c2:>8.2}/{w2t:<8.2}   {c3:>8.2}/{w3t:<8.2}");
    }
    // attention prefill：单请求 n 个 token
    let (h, hk, d) = (16usize, 8usize, 128usize);
    let q = s.alloc_zeros::<bf16>(max_m * h * d).unwrap();
    let k = s.alloc_zeros::<bf16>(max_m * hk * d).unwrap();
    let v = s.alloc_zeros::<bf16>(max_m * hk * d).unwrap();
    let mut o = s.alloc_zeros::<bf16>(max_m * h * d).unwrap();
    // decode attention：flash paged（q_len=1 → 每 (req,head) 1 个活跃 warp）vs split-K；B 个请求、每个 seq_len 个 key
    {
        let (bs, hk) = (16usize, 8usize);
        for &(b, n) in &[(1usize, 512usize), (8, 2048), (32, 2048), (64, 1024)] {
            let blocks_per = n.div_ceil(bs);
            let cache = s.alloc_zeros::<bf16>(b * blocks_per * bs * hk * 2 * d).unwrap();
            let bt: Vec<i32> = (0..b * blocks_per).map(|x| x as i32).collect();
            let dbt = s.htod_copy(bt).unwrap();
            let seq = s.htod_copy(vec![n as i32; b]).unwrap();
            let cu: Vec<i32> = (0..=b as i32).collect();
            let dcu = s.htod_copy(cu).unwrap();
            let qd = s.alloc_zeros::<bf16>(b * h * d).unwrap();
            let mut od = s.alloc_zeros::<bf16>(b * h * d).unwrap();
            let mut scratch = s.alloc_zeros::<f32>(dinfer_kernels::gqa_decode_scratch_len(b, h, d)).unwrap();
            let mut f_paged = || {
                dinfer_kernels::gqa_paged_bf16(
                    &s,
                    &qd,
                    &cache,
                    &dbt,
                    &seq,
                    &dcu,
                    &mut od,
                    b,
                    1,
                    h,
                    hk,
                    d,
                    bs,
                    blocks_per,
                    1.0 / (d as f32).sqrt(),
                    0,
                )
                .unwrap()
            };
            f_paged();
            let t_paged = time(&mut f_paged);
            let mut f_split = || {
                dinfer_kernels::gqa_decode_bf16(
                    &s,
                    &qd,
                    &cache,
                    &dbt,
                    &seq,
                    &mut od,
                    &mut scratch,
                    b,
                    h,
                    hk,
                    d,
                    bs,
                    blocks_per,
                    1.0 / (d as f32).sqrt(),
                    0,
                )
                .unwrap()
            };
            f_split();
            let t_split = time(&mut f_split);
            println!("decode B={b:<3} seq={n:<5} flash_paged {t_paged:>8.3} ms   splitk {t_split:>8.3} ms   ({:.1}x)", t_paged / t_split.max(1e-6));
        }
    }
    // 小 M（decode batch）GEMM：cuBLASLt 对 M ≤ 8 选 gemv 路径；对比把 M pad 到 16 走 tensorop
    println!("M     qkv(4096x1024)  gateup(6144x1024)  down(1024x3072)   [ms per GEMM, warm, 20 次平均]  | padded-to-16");
    for &m in &[1usize, 2, 4, 8, 16, 32, 64] {
        let mut run = |mm: usize| -> (f64, f64, f64) {
            let mut t = |w: &dinfer_kernels::cudarc::driver::CudaSlice<bf16>, nn: usize, kk: usize| {
                g.forward_bf16(&a, w, None, &mut out, mm, nn, kk).unwrap();
                s.synchronize().unwrap();
                let t0 = Instant::now();
                for _ in 0..20 {
                    g.forward_bf16(&a, w, None, &mut out, mm, nn, kk).unwrap();
                }
                s.synchronize().unwrap();
                t0.elapsed().as_secs_f64() * 1e3 / 20.0
            };
            (t(&w1, qkv_w, hidden), t(&w2, 2 * inter, hidden), t(&w3, hidden, inter))
        };
        let (a1, a2, a3) = run(m);
        let (p1, p2, p3) = if m < 16 { run(16) } else { (a1, a2, a3) };
        println!("{m:<5} {a1:>8.4}         {a2:>8.4}           {a3:>8.4}           | {p1:>8.4} {p2:>8.4} {p3:>8.4}");
    }
    println!("n      gqa_prefill cold/warm (ms)");
    for &n in &[389usize, 639, 889, 1139, 1514, 1900] {
        let cu = s.htod_copy(vec![0i32, n as i32]).unwrap();
        let mut f = || {
            dinfer_kernels::gqa_prefill_bf16(
                &s,
                &q,
                &k,
                &v,
                &mut o,
                &cu,
                1,
                n,
                h,
                hk,
                d,
                1.0 / (d as f32).sqrt(),
                0,
                0,
            )
            .unwrap()
        };
        let c = time(&mut f);
        let w = time(&mut f);
        println!("{n:<6} {c:>8.2}/{w:<8.2}");
    }
}
