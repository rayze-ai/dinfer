//! 临时：同一进程内直接调 C ABI，传 mode=0/1 逐元素比（验证完即删）。
use dinfer_kernels as k;

#[test]
#[ignore]
fn norm_mode_differs() {
    let dev = k::Device::new(0).unwrap();
    let s = dev.default_stream();
    let (rows, hidden) = (8usize, 2816usize);
    let x: Vec<k::bf16> = (0..rows * hidden)
        .map(|i| k::bf16::from_f32(((i * 37 % 1000) as f32 - 500.0) / 97.0))
        .collect();
    let w: Vec<k::bf16> = (0..hidden).map(|i| k::bf16::from_f32(0.5 + (i % 13) as f32 * 0.1)).collect();
    let dx = s.htod_copy(x).unwrap();
    let dw = s.htod_copy(w).unwrap();
    let mut out = Vec::new();
    for mode in [0, 1] {
        let mut o = s.alloc_zeros::<k::bf16>(rows * hidden).unwrap();
        let rc = unsafe {
            k::ffi::rmsnorm_bf16(
                s.ptr(&dx) as *const std::ffi::c_void,
                s.ptr(&dw) as *const std::ffi::c_void,
                s.ptr_mut(&mut o) as *mut std::ffi::c_void,
                rows as i64,
                hidden as i64,
                1e-6,
                0,
                mode,
                s.raw() as _,
            )
        };
        assert_eq!(rc, 0, "rmsnorm mode={mode} rc={rc}");
        out.push(s.dtoh_sync_copy(&o).unwrap());
    }
    let nd = out[0].iter().zip(out[1].iter()).filter(|(a, b)| a != b).count();
    let mx = out[0]
        .iter()
        .zip(out[1].iter())
        .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
        .fold(0f32, f32::max);
    println!("NORM_MODE diff: {nd}/{} max={mx:.3e}", rows * hidden);
}
