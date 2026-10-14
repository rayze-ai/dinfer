//! 类型安全的 kernel 包装。

use crate::device::Stream;
use crate::ffi;
use anyhow::{bail, Result};
use cudarc::driver::CudaSlice;

/// out = a + b（fp32）。
pub fn add_f32(
    stream: &Stream,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
) -> Result<()> {
    let n = a.len();
    if b.len() != n || out.len() != n {
        bail!("add_f32: length mismatch a={} b={} out={}", n, b.len(), out.len());
    }
    let pa = stream.ptr(a) as *const f32;
    let pb = stream.ptr(b) as *const f32;
    let po = stream.ptr_mut(out) as *mut f32;
    // SAFETY: 三个指针来自有效的 CudaSlice，长度已校验；stream 来自同一 context。
    let rc = unsafe { ffi::dinfer_add_f32(pa, pb, po, n as i64, stream.raw() as ffi::cudaStream_t) };
    if rc != 0 {
        bail!("dinfer_add_f32 failed: cudaError {rc}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    #[test]
    #[ignore = "needs a CUDA device; run with `cargo test -p dinfer-kernels -- --ignored`"]
    fn add_roundtrip() {
        let dev = Device::new(0).unwrap();
        let s = dev.default_stream();
        let n = 1 << 20;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (2 * i) as f32).collect();
        let da = s.htod_copy(a.clone()).unwrap();
        let db = s.htod_copy(b.clone()).unwrap();
        let mut dout = s.alloc_zeros::<f32>(n).unwrap();
        add_f32(&s, &da, &db, &mut dout).unwrap();
        let out = s.dtoh_sync_copy(&dout).unwrap();
        for i in (0..n).step_by(4097) {
            assert_eq!(out[i], a[i] + b[i]);
        }
    }
}
