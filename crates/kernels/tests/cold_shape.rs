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
        let r = |i: usize| {
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
                h,
                hk,
                d,
                1.0 / (d as f32).sqrt(),
            )
            .unwrap()
        };
        let c = time(&mut f);
        let w = time(&mut f);
        println!("{n:<6} {c:>8.2}/{w:<8.2}");
    }
}
