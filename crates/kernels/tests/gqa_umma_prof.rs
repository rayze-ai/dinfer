#![cfg(not(feature = "no-nvcc"))]
use dinfer_kernels::cudarc::driver::CudaSlice;
use dinfer_kernels::{bf16, ffi, Device, Stream};

#[test]
#[ignore]
fn gqa_umma_prof() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let (t, h, hk, d) = (1024usize, 1usize, 1usize, 256usize);
    let w = (h + 2 * hk) * d;
    let qkv: Vec<bf16> = (0..t * w).map(|i| bf16::from_f32(((i % 97) as f32 - 48.0) * 0.02)).collect();
    let dqkv = st.htod_copy(qkv).unwrap();
    let dcu = st.htod_copy(vec![0i32, t as i32]).unwrap();
    let mut out = st.alloc_zeros::<bf16>(t * h * d).unwrap();
    let mut prof: CudaSlice<u64> = st.alloc_zeros::<u64>(8).unwrap();
    let qv = dqkv.slice(..);
    let kv = dqkv.slice(h * d..);
    let vv = dqkv.slice((h + hk) * d..);
    let scale = (d as f32).powf(-0.5);
    for _ in 0..3 {
        let rc = unsafe {
            ffi::gqa_fa_umma_launch(0, t as i32, h as i32, 1, st.ptr(&qv) as *const _, st.ptr(&kv) as *const _,
                st.ptr(&vv) as *const _, std::ptr::null(), std::ptr::null(), std::ptr::null(), st.ptr(&dcu) as *const i32,
                st.ptr_mut(&mut out) as *mut _, h as i32, hk as i32, scale * 1.4426950408889634, w as i32, w as i32,
                0, 0, 0, st.ptr_mut(&mut prof) as *mut _, st.raw() as _)
        };
        assert_eq!(rc, 0);
    }
    st.synchronize().unwrap();
    let p = st.dtoh_sync_copy(&prof).unwrap();
    let tot: u64 = p[0] + p[1] + p[2] + p[3];
    let seg = (p[5] + p[6] + p[7]).max(1) as f64;
    println!("[prof] pv+load 细分: PV发射={:.0}% 等K/V={:.0}% QK发射={:.0}%",
        100.0 * p[5] as f64 / seg, 100.0 * p[6] as f64 / seg, 100.0 * p[7] as f64 / seg);
    println!("[prof] qk_wait={:.0}% softmax={:.0}% rescale={:.0}% pv+load={:.0}% tiles={}",
        100.0 * p[0] as f64 / tot as f64, 100.0 * p[1] as f64 / tot as f64,
        100.0 * p[2] as f64 / tot as f64, 100.0 * p[3] as f64 / tot as f64, p[4] / 3);
}
