//! 设备 / 流 的薄封装。基于 cudarc driver API。

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, ValidAsZeroBits,
};
use std::sync::Arc;

#[derive(Clone)]
pub struct Device {
    ctx: Arc<CudaContext>,
}

#[derive(Clone)]
pub struct Stream {
    inner: Arc<CudaStream>,
}

impl Device {
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal).with_context(|| format!("cuda device {ordinal}"))?;
        Ok(Self { ctx })
    }

    /// 关闭 cudarc 的自动事件同步（每次取指针时 record / wait 事件）。CUDA graph capture 期间这些事件操作会被
    /// 驱动拒绝（CUDA_ERROR_STREAM_CAPTURE_ISOLATION），而我们全程单流、按 step 同步，不需要它。
    /// 安全前提：调用方保证所有工作在同一条流上，或自行用事件同步。
    pub fn disable_event_tracking(&self) {
        // SAFETY: 见上；本项目 executor 单流顺序执行，H2D/D2H 用同流的拷贝。
        unsafe { self.ctx.disable_event_tracking() }
    }

    pub fn default_stream(&self) -> Stream {
        Stream { inner: self.ctx.default_stream() }
    }

    pub fn new_stream(&self) -> Result<Stream> {
        Ok(Stream { inner: self.ctx.new_stream()? })
    }

    pub fn name(&self) -> Result<String> {
        Ok(self.ctx.name()?)
    }

    /// (major, minor)
    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        use cudarc::driver::sys::CUdevice_attribute as A;
        let major = self.ctx.attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = self.ctx.attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        Ok((major, minor))
    }

    pub fn ordinal(&self) -> usize {
        self.ctx.ordinal()
    }
}

impl Stream {
    pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> Result<CudaSlice<T>> {
        Ok(self.inner.alloc_zeros::<T>(len)?)
    }

    pub fn htod_copy<T: DeviceRepr>(&self, src: Vec<T>) -> Result<CudaSlice<T>> {
        Ok(self.inner.clone_htod(&src)?)
    }

    pub fn dtoh_sync_copy<T: DeviceRepr>(&self, src: &CudaSlice<T>) -> Result<Vec<T>> {
        Ok(self.inner.clone_dtoh(src)?)
    }

    /// 异步清零前 `n` 个元素（在本流上排队）。
    pub fn memset_zeros_n<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        s: &mut CudaSlice<T>,
        n: usize,
    ) -> Result<()> {
        anyhow::ensure!(n <= s.len(), "memset_zeros_n: n {n} > len {}", s.len());
        let mut v = s.slice_mut(..n);
        Ok(self.inner.memset_zeros(&mut v)?)
    }

    /// 设备内拷贝前 n 个元素（同流异步）。
    pub fn dtod_copy_n<T: DeviceRepr>(
        &self,
        src: &CudaSlice<T>,
        dst: &mut CudaSlice<T>,
        n: usize,
    ) -> Result<()> {
        anyhow::ensure!(n <= src.len() && n <= dst.len(), "dtod_copy_n: n {n} > len");
        let s = src.slice(..n);
        let mut d = dst.slice_mut(..n);
        Ok(self.inner.memcpy_dtod(&s, &mut d)?)
    }

    pub fn synchronize(&self) -> Result<()> {
        Ok(self.inner.synchronize()?)
    }

    /// 当前设备 (free, total) 字节（`cuMemGetInfo`）。按显存算 KV block 数用。
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        Ok(cudarc::driver::result::mem_get_info()?)
    }

    /// 裸 cudaStream_t，传给 C ABI kernel。
    pub fn raw(&self) -> cudarc::driver::sys::CUstream {
        self.inner.cu_stream()
    }

    /// 给 cudarc 的其他库（cublasLt 等）用。
    pub fn inner_arc(&self) -> Arc<CudaStream> {
        self.inner.clone()
    }

    /// 只读设备指针（以 u64 形式，供 FFI）。
    pub fn ptr<T: DeviceRepr>(&self, s: &CudaSlice<T>) -> u64 {
        let (p, _guard) = s.device_ptr(&self.inner);
        p
    }

    pub fn ptr_mut<T: DeviceRepr>(&self, s: &mut CudaSlice<T>) -> u64 {
        let (p, _guard) = s.device_ptr_mut(&self.inner);
        p
    }
}
