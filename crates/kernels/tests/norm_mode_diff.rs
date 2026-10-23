//! 临时：融合 norm→fp8 与两步（norm bf16 → quant fp8）逐元素比（验证完即删）。
use dinfer_kernels as k;

#[test]
#[ignore]
fn fused_vs_two_step() {
    let dev = k::Device::new(0).unwrap();
    let s = dev.default_stream();
    let (rows, hidden) = (24usize, 2816usize);
    let x: Vec<k::bf16> = (0..rows * hidden)
        .map(|i| k::bf16::from_f32(((i * 37 % 1000) as f32 - 500.0) / 97.0))
        .collect();
    let w: Vec<k::bf16> = (0..hidden).map(|i| k::bf16::from_f32(0.5 + (i % 13) as f32 * 0.1)).collect();
    let dx = s.htod_copy(x).unwrap();
    let dw = s.htod_copy(w).unwrap();
    let a_scale = 0.0371f32;

    // 融合
    let mut fused = s.alloc_zeros::<u8>(rows * hidden).unwrap();
    k::ops::rmsnorm_quant_fp8(&s, &dx, &dw, &mut fused, rows, hidden, 1e-6, a_scale).unwrap();

    // 两步：norm 到 bf16，再 quant
    let mut normed = s.alloc_zeros::<k::bf16>(rows * hidden).unwrap();
    k::rmsnorm_bf16_n(&s, &dx, &dw, &mut normed, rows, hidden, 1e-6).unwrap();
    let mut two = s.alloc_zeros::<u8>(rows * hidden).unwrap();
    let g = k::Gemm::new(&s).unwrap();
    g.quant_fp8(&normed, a_scale, &mut two, rows * hidden).unwrap();

    let a = s.dtoh_sync_copy(&fused).unwrap();
    let b = s.dtoh_sync_copy(&two).unwrap();
    let nd = a.iter().zip(b.iter()).filter(|(p, q)| p != q).count();
    println!("FUSED vs TWO-STEP: {nd}/{} differ", rows * hidden);
}
