//! 设备 / 流 的薄封装。基于 cudarc driver API。

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaEvent, CudaGraph, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, ValidAsZeroBits,
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

    /// 异步 H2D 到已有缓冲的前 `src.len()` 个元素（同流排队；host 侧 Vec 在本调用返回后即可复用，
    /// 因为 cudarc 对 pageable 内存的 memcpy_htod 在返回前完成拷贝进 staging）。
    pub fn htod_into<T: DeviceRepr>(&self, src: &[T], dst: &mut CudaSlice<T>) -> Result<()> {
        anyhow::ensure!(src.len() <= dst.len(), "htod_into: src {} > dst {}", src.len(), dst.len());
        if src.is_empty() {
            return Ok(());
        }
        let mut d = dst.slice_mut(..src.len());
        Ok(self.inner.memcpy_htod(src, &mut d)?)
    }

    /// 开始在本流上捕获 CUDA graph（线程局部模式）。要求：非默认流；期间不得同步 / 分配 / 跨流等待。
    pub fn begin_capture(&self) -> Result<()> {
        use cudarc::driver::sys::CUstreamCaptureMode;
        Ok(self.inner.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?)
    }

    /// 结束捕获并实例化；返回可重放的图。
    pub fn end_capture(&self) -> Result<Graph> {
        use cudarc::driver::sys::CUgraphInstantiate_flags;
        let g = self
            .inner
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)?
            .ok_or_else(|| anyhow::anyhow!("end_capture: empty graph"))?;
        Ok(Graph(g))
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

    /// 在本流上记录一个事件（跨流依赖用；禁用 timing 以免影响并发）。
    pub fn record_event(&self) -> Result<Event> {
        use cudarc::driver::sys::CUevent_flags;
        Ok(Event(self.inner.record_event(Some(CUevent_flags::CU_EVENT_DISABLE_TIMING))?))
    }
    /// 本流后续工作等待 `ev`（设备侧等待，不阻塞 host）。
    pub fn wait_event(&self, ev: &Event) -> Result<()> {
        Ok(self.inner.wait(&ev.0)?)
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

    /// 只读设备指针（以 u64 形式，供 FFI）。接受 `CudaSlice` / `CudaView`。
    pub fn ptr<T: DeviceRepr, S: DevicePtr<T> + ?Sized>(&self, s: &S) -> u64 {
        let (p, _guard) = s.device_ptr(&self.inner);
        p
    }

    /// 可写设备指针。接受 `CudaSlice` / `CudaViewMut`。
    pub fn ptr_mut<T: DeviceRepr, S: DevicePtrMut<T> + ?Sized>(&self, s: &mut S) -> u64 {
        let (p, _guard) = s.device_ptr_mut(&self.inner);
        p
    }

    /// 所属设备（同一 context）。
    pub fn device(&self) -> Device {
        Device { ctx: self.inner.context().clone() }
    }

    /// D2H 同步拷贝一个视图。
    pub fn dtoh_sync_copy_view<T: DeviceRepr, S: DevicePtr<T>>(&self, src: &S) -> Result<Vec<T>> {
        Ok(self.inner.clone_dtoh(src)?)
    }
}

/// 可重放的 CUDA graph。`cudarc::CudaGraph` 持裸句柄不是 `Send`；本类型声明 `Send`：图只在创建它的 executor /
/// 引擎线程上 launch（Runner 随 engine 线程整体 move 一次），不会并发访问。
pub struct Graph(CudaGraph);
// SAFETY: 见上。cuGraphLaunch 允许从任意线程调用，前提是同一时刻只有一个使用者；所有权随 executor 单线程移动。
unsafe impl Send for Graph {}
impl Graph {
    /// 在创建它的流上重放。
    pub fn launch(&self) -> Result<()> {
        Ok(self.0.launch()?)
    }
}

/// CUDA event（跨流同步）。
pub struct Event(CudaEvent);
