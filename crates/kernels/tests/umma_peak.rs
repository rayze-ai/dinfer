//! tcgen05（UMMA）工具链自证 + 峰值探针（r12a 第一步，docs/loops/qwen35-tcgen05.md）。
//!
//! ① 正确性：M=128 N=256 的一小块 GEMM，与 CPU 参考逐元素比（bf16 输入、f32 累加，容差按相对误差）。
//! ② 峰值：K 维循环拉长到几千个 64-块，量 TF/s，门槛 ≥ 900（mma.sync 实测 575）。
//!
//! 运行：`CUDA_VISIBLE_DEVICES=5 cargo test -p dinfer-kernels --test umma_peak -- --ignored --nocapture`
//! 未编入 sm100（CUDA_ARCHS 不含 100/103）时 skip 并打印原因，不失败。
#![cfg(not(feature = "no-nvcc"))]

use dinfer_kernels::cudarc::driver::CudaSlice;
use dinfer_kernels::{bf16, ffi, Device, Stream};

fn mk(st: &Stream, n: usize, scale: f32) -> Vec<bf16> {
    // 简单的确定性伪随机，避免额外依赖。
    let mut x: u64 = 0x1234_5678_9abc_def0;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let v = ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
            bf16::from_f32(v * scale)
        })
        .collect()
}

fn launch(
    st: &Stream,
    a: &CudaSlice<bf16>,
    b: &CudaSlice<bf16>,
    c: &mut CudaSlice<f32>,
    k_iters: i64,
    blocks: i64,
    check: bool,
) -> i32 {
    unsafe {
        ffi::umma_peak_launch(
            st.ptr(a) as *const _,
            st.ptr(b) as *const _,
            st.ptr_mut(c) as *mut _,
            k_iters,
            blocks,
            check as i64,
            st.raw() as _,
        )
    }
}

#[test]
#[ignore]
fn umma_peak() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    const M: usize = 128;
    const N: usize = 256;
    const BK: usize = 64;

    // ---- ① 正确性：k_iters = 3 ----
    let ki = 3usize;
    let av = mk(&st, ki * M * BK, 0.125);
    let bv = mk(&st, ki * N * BK, 0.125);
    let da = st.htod_copy(av.clone()).unwrap();
    let db = st.htod_copy(bv.clone()).unwrap();
    let mut dc = st.alloc_zeros::<f32>(M * N).unwrap();
    let rc = launch(&st, &da, &db, &mut dc, ki as i64, 1, true);
    if rc == ffi::CUDA_ERROR_NOT_SUPPORTED {
        println!("[umma_peak] skip：本机构建未编入 sm100 路径（CUDA_ARCHS 不含 100/103）");
        return;
    }
    assert_eq!(rc, 0, "umma_peak_launch rc={rc}");
    st.synchronize().unwrap();
    let got = st.dtoh_sync_copy(&dc).unwrap();

    // CPU 参考（f32 累加）。
    let mut want = vec![0f32; M * N];
    for it in 0..ki {
        for m in 0..M {
            for n in 0..N {
                let mut acc = 0f32;
                for k in 0..BK {
                    let a = av[it * M * BK + m * BK + k].to_f32();
                    let b = bv[it * N * BK + n * BK + k].to_f32();
                    acc += a * b;
                }
                want[m * N + n] += acc;
            }
        }
    }
    let (mut mx, mut dot, mut na, mut nb) = (0f32, 0f64, 0f64, 0f64);
    for (g, w) in got.iter().zip(want.iter()) {
        mx = mx.max((g - w).abs());
        dot += (*g as f64) * (*w as f64);
        na += (*g as f64) * (*g as f64);
        nb += (*w as f64) * (*w as f64);
    }
    let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-30);
    let scale_ref = want.iter().fold(0f32, |a, b| a.max(b.abs())).max(1e-6);
    println!("[umma_peak] 正确性 k={} maxabs={mx:.5} rel={:.3e} cos={cos:.8}", ki * BK, mx / scale_ref);
    assert!(cos > 0.9999 && mx / scale_ref < 1e-2, "UMMA 结果与 CPU 参考不符：cos={cos} maxabs={mx}");

    // ---- ② 峰值 ----
    // 口径与 scripts/dev/mma_peak.cu 一致：smem 只装一次、纯 MMA 循环，看 tcgen05 的吞吐上限。
    // flops = 2·M·N·64·k_iters·blocks。
    let kit: usize = std::env::var("UMMA_K_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(4096);
    let blocks: usize = std::env::var("UMMA_BLOCKS").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let blocks = if blocks == 0 { 2 * 148 } else { blocks }; // B300 148 SM；每 SM 2 个 block 看是否更好
    let av2 = mk(&st, M * BK, 0.125);
    let bv2 = mk(&st, N * BK, 0.125);
    let da2 = st.htod_copy(av2.clone()).unwrap();
    let db2 = st.htod_copy(bv2.clone()).unwrap();
    let mut dc2 = st.alloc_zeros::<f32>(blocks * M * N).unwrap();
    let flop = 2.0 * (M as f64) * (N as f64) * (BK as f64) * (kit as f64) * (blocks as f64);
    let iters = 20usize;
    let rc = launch(&st, &da2, &db2, &mut dc2, kit as i64, blocks as i64, false);
    assert_eq!(rc, 0, "peak launch rc={rc}");
    st.synchronize().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        launch(&st, &da2, &db2, &mut dc2, kit as i64, blocks as i64, false);
    }
    st.synchronize().unwrap();
    let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

    // 峰值模式也要验数：k_iters 个相同的 64-K 块累加 = 单个 64-K 块结果 × k_iters。
    // 这一条是"MMA 真的全部执行了"的硬证据——否则测出来的时间是排空不彻底的假象。
    let got2 = st.dtoh_sync_copy(&dc2).unwrap();
    let mut want1 = vec![0f32; M * N];
    for m in 0..M {
        for n in 0..N {
            let mut acc = 0f32;
            for k in 0..BK {
                acc += av2[m * BK + k].to_f32() * bv2[n * BK + k].to_f32();
            }
            want1[m * N + n] = acc * kit as f32;
        }
    }
    let (mut mx2, mut dot2, mut na2, mut nb2) = (0f32, 0f64, 0f64, 0f64);
    for nb in 0..blocks {
        for i in 0..M * N {
            let g = got2[nb * M * N + i];
            let w = want1[i];
            mx2 = mx2.max((g - w).abs());
            dot2 += (g as f64) * (w as f64);
            na2 += (g as f64) * (g as f64);
            nb2 += (w as f64) * (w as f64);
        }
    }
    let cos2 = dot2 / (na2.sqrt() * nb2.sqrt()).max(1e-30);
    println!(
        "[umma_peak] blocks={blocks} k_iters={kit} us={us:.1} tflops={:.1} 数值cos={cos2:.8} maxabs={mx2:.4}",
        flop / us / 1e6
    );
    assert!(cos2 > 0.9999, "峰值模式数值不符（MMA 未全部执行？）：cos={cos2}");
}
