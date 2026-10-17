//! 设备 / 流 的薄封装。基于 cudarc driver API。

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaEvent, CudaGraph, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, ValidAsZeroBits,
};
use cudarc::driver::{result, sys};
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
    /// 不清零的显存分配（省一次 memset；17 GiB KV ≈ 60 ms）。
    /// 安全前提：调用方保证每个元素在被读之前都已被写（KV cache：块只在 slot_mapping 写入后才进 block_table）。
    pub fn alloc_uninit<T: DeviceRepr>(&self, len: usize) -> Result<CudaSlice<T>> {
        // SAFETY: 见上；返回的内存内容未定义，调用方负责先写后读。
        Ok(unsafe { self.inner.alloc::<T>(len)? })
    }
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
    /// 带时间戳的事件（量 GPU 忙时用；跨流同步请用 `record_event`）。走裸驱动 API：cudarc 的 safe event
    /// 在 disable_event_tracking 后 elapsed 报 INVALID_HANDLE。
    pub fn new_timing_event(&self) -> Result<TimingEvent> {
        self.bind_ctx()?;
        // SAFETY: 默认 flags（带 timing）；Drop 时 destroy。
        Ok(TimingEvent(result::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT)?))
    }
    pub fn record_timing(&self, ev: &TimingEvent) -> Result<()> {
        self.bind_ctx()?;
        // SAFETY: event / stream 都有效。
        Ok(unsafe { result::event::record(ev.0, self.raw())? })
    }
    /// 把本流的 CUDA context 绑到当前线程（裸驱动调用前必须；cudarc 的 safe 封装内部会做，裸 `result::*` 不会）。
    /// engine 线程不是创建 context 的线程，不绑会得到 CUDA_ERROR_INVALID_CONTEXT / INVALID_HANDLE。
    pub fn bind_ctx(&self) -> Result<()> {
        Ok(self.inner.context().bind_to_thread()?)
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

// ---------------------------------------------------------------- staged H2D（启动 loop 第 10 轮）
/// pinned 双缓冲 H2D：host 侧 memcpy 进 pinned 块 A 的同时，上一块 B 在异步 DMA 到设备。
/// 对比驱动的 pageable `cuMemcpyHtoD`（内部 staging + DMA 串行，4090 实测 ~8 GiB/s），
/// 可接近 PCIe 4 ×16 的 ~24 GiB/s。只用于一次性大批量上传（权重加载），线程局部、按需创建。
struct StagedUploader {
    bufs: [*mut u8; 2],
    events: [sys::CUevent; 2],
    chunk: usize,
    next: usize,
    inflight: [bool; 2],
}
const STAGE_CHUNK: usize = 64 << 20;
impl StagedUploader {
    fn new() -> Result<Self> {
        // SAFETY: 分配 pinned host 内存与事件；在 Drop 释放。flags=0（不用 WRITECOMBINED，memcpy 写入更快）。
        unsafe {
            let a = result::malloc_host(STAGE_CHUNK, 0)? as *mut u8;
            let b = result::malloc_host(STAGE_CHUNK, 0)? as *mut u8;
            let ea = result::event::create(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)?;
            let eb = result::event::create(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)?;
            Ok(Self { bufs: [a, b], events: [ea, eb], chunk: STAGE_CHUNK, next: 0, inflight: [false; 2] })
        }
    }
    /// 把 `src` 拷到设备地址 `dst`（按 chunk 轮转两块 pinned 缓冲）。
    fn upload(&mut self, stream: sys::CUstream, dst: u64, src: &[u8]) -> Result<()> {
        let mut off = 0usize;
        while off < src.len() {
            let n = (src.len() - off).min(self.chunk);
            let i = self.next;
            // SAFETY: 等该缓冲上一次 DMA 完成后再覆盖；memcpy 不越界；DMA 源是 pinned 内存，异步合法。
            unsafe {
                if self.inflight[i] {
                    result::event::synchronize(self.events[i])?;
                }
                std::ptr::copy_nonoverlapping(src.as_ptr().add(off), self.bufs[i], n);
                let host = std::slice::from_raw_parts(self.bufs[i], n);
                result::memcpy_htod_async(dst + off as u64, host, stream)?;
                result::event::record(self.events[i], stream)?;
            }
            self.inflight[i] = true;
            self.next = i ^ 1;
            off += n;
        }
        Ok(())
    }
    fn finish(&mut self) -> Result<()> {
        for i in 0..2 {
            if self.inflight[i] {
                // SAFETY: 事件有效。
                unsafe { result::event::synchronize(self.events[i])? };
                self.inflight[i] = false;
            }
        }
        Ok(())
    }
}
impl Drop for StagedUploader {
    fn drop(&mut self) {
        let _ = self.finish();
        // SAFETY: 本结构独占这些资源。
        unsafe {
            for b in self.bufs {
                let _ = result::free_host(b as *mut std::ffi::c_void);
            }
            for e in self.events {
                let _ = result::event::destroy(e);
            }
        }
    }
}
thread_local! {
    static STAGER: std::cell::RefCell<Option<StagedUploader>> = const { std::cell::RefCell::new(None) };
}
impl Stream {
    /// 大批量 H2D（见 `StagedUploader`）：异步入队，返回后 `src` 可释放（已复制进 pinned）。
    /// 用完整批后调 `staged_finish()` 等 DMA 全部落地。
    pub fn htod_into_staged<T: DeviceRepr>(&self, src: &[T], dst: &mut CudaSlice<T>) -> Result<()> {
        anyhow::ensure!(src.len() <= dst.len(), "htod_into_staged: src {} > dst {}", src.len(), dst.len());
        if src.is_empty() {
            return Ok(());
        }
        let dst_ptr = self.ptr_mut(dst);
        // SAFETY: T: DeviceRepr 是 POD，按字节视图拷贝。
        let bytes = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, std::mem::size_of_val(src)) };
        STAGER.with(|c| -> Result<()> {
            let mut c = c.borrow_mut();
            if c.is_none() {
                *c = Some(StagedUploader::new()?);
            }
            c.as_mut().unwrap().upload(self.raw(), dst_ptr, bytes)
        })
    }
    /// 等待本线程所有 staged H2D 完成并释放 pinned 缓冲。
    pub fn staged_finish(&self) -> Result<()> {
        STAGER.with(|c| -> Result<()> {
            if let Some(mut u) = c.borrow_mut().take() {
                u.finish()?;
            }
            Ok(())
        })
    }
}

/// 带时间戳的 CUDA event（裸句柄）；`elapsed_ms(a, b)` 先同步 end 再取差。
pub struct TimingEvent(sys::CUevent);
// SAFETY: 裸句柄只在 executor 线程的单条流上 record / 读取；跨线程移动不改变其有效性。
unsafe impl Send for TimingEvent {}
unsafe impl Sync for TimingEvent {}
impl TimingEvent {
    pub fn elapsed_ms(start: &TimingEvent, end: &TimingEvent) -> Result<f32> {
        // SAFETY: 两事件都已 record；synchronize 保证完成。
        unsafe {
            result::event::synchronize(end.0)?;
            Ok(result::event::elapsed(start.0, end.0)?)
        }
    }
}
impl Drop for TimingEvent {
    fn drop(&mut self) {
        // SAFETY: 独占句柄。
        unsafe { let _ = result::event::destroy(self.0); }
    }
}

// ---------------------------------------------------------------- PinnedUploader（Q3 第 50 轮）
/// 每 step 的小输入（ids / positions / block_table / 采样参数……十几个数组）打包进**一块 pinned host 缓冲**，
/// 一次异步 H2D 到设备 staging，再按记录的目标地址 D2D 异步散开。替代原先每数组一次同步 pageable
/// `cuMemcpyHtoD`（B=32 时 14 次 ≈ 600–750 µs/步，是 decode 步 host 开销的主项）。
/// 用法：每 step 开始 `reset_step()`（host 区回到 0）；之后可多轮 `push(..)*` → `flush()`，每轮 flush 只重置
/// 散开表、**不回卷 host 偏移**：同一 step 内多次 flush 用的是 pinned 缓冲的不同区域，前一轮的异步 H2D 还在
/// 读时不会被后一轮的 host memcpy 覆盖（第一版在每轮 begin 回卷到 0，B=32 压测下随机 ILLEGAL_ADDRESS）。
/// 安全前提：下一次 `reset_step()` 前本流上一步的 H2D 已完成（executor 每步末尾 D2H 同步满足）。
pub struct PinnedUploader {
    host: *mut u8,
    staging: CudaSlice<u8>,
    cap: usize,
    off: usize,
    scatter: Vec<(u64, usize, usize)>, // (dst_dev_ptr, host_off, bytes)
    /// PINNED_UPLOAD=1 走打包；默认 0 = 每数组一次 pageable `htod_into`（bench21 显示打包版 c8 更慢，待 A/B 定论）。
    enabled: bool,
}
// SAFETY: 裸 host 指针只在 executor 线程使用；跨线程移动不改变有效性。
unsafe impl Send for PinnedUploader {}
impl PinnedUploader {
    pub fn new(stream: &Stream, cap: usize) -> Result<Self> {
        stream.bind_ctx()?;
        // SAFETY: pinned 分配，Drop 释放。
        let host = unsafe { result::malloc_host(cap, 0)? as *mut u8 };
        let staging = stream.alloc_uninit::<u8>(cap)?;
        let enabled = std::env::var("PINNED_UPLOAD").is_ok_and(|v| v == "1");
        Ok(Self { host, staging, cap, off: 0, scatter: Vec::with_capacity(32), enabled })
    }
    /// 每 step 开始：host 区回到 0（上一步的 DMA 已随 step 末尾同步完成）。
    pub fn reset_step(&mut self) {
        self.off = 0;
        self.scatter.clear();
    }
    /// 一轮打包开始：只清散开表，host 偏移继续往后走。
    pub fn begin(&mut self) {
        self.scatter.clear();
    }
    /// 把 `src` 排进本次上传，目标是 `dst` 的前 src.len() 个元素。
    pub fn push<T: DeviceRepr>(&mut self, stream: &Stream, src: &[T], dst: &mut CudaSlice<T>) -> Result<()> {
        anyhow::ensure!(src.len() <= dst.len(), "PinnedUploader::push: src {} > dst {}", src.len(), dst.len());
        if !self.enabled {
            return stream.htod_into(src, dst);
        }
        let bytes = std::mem::size_of_val(src);
        if bytes == 0 {
            return Ok(());
        }
        let start = (self.off + 15) & !15;
        anyhow::ensure!(start + bytes <= self.cap, "PinnedUploader: staging {} B too small", self.cap);
        // SAFETY: 范围已校验；T 是 POD。
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr() as *const u8, self.host.add(start), bytes) };
        self.scatter.push((stream.ptr_mut(dst), start, bytes));
        self.off = start + bytes;
        Ok(())
    }
    /// 一次 H2D（pinned → staging，异步）+ 逐目标 D2D（异步），都在 `stream` 上。
    pub fn flush(&mut self, stream: &Stream) -> Result<()> {
        let Some(&(_, first_off, _)) = self.scatter.first() else { return Ok(()) };
        stream.bind_ctx()?;
        let base = stream.ptr_mut(&mut self.staging);
        // 只传本轮新增的区间 [first_off, off)
        // SAFETY: host 是 pinned；staging / 目标都是有效设备地址；同流顺序保证 D2D 在 H2D 之后。
        unsafe {
            let host = std::slice::from_raw_parts(self.host.add(first_off), self.off - first_off);
            result::memcpy_htod_async(base + first_off as u64, host, stream.raw())?;
            for &(dst, hoff, bytes) in &self.scatter {
                result::memcpy_dtod_async(dst, base + hoff as u64, bytes, stream.raw())?;
            }
        }
        self.scatter.clear();
        Ok(())
    }
}
impl Drop for PinnedUploader {
    fn drop(&mut self) {
        // SAFETY: 独占。
        unsafe { let _ = result::free_host(self.host as *mut std::ffi::c_void); }
    }
}

// ---------------------------------------------------------------- 异步 D2H（Q3-l）
/// 一块 pinned host 缓冲 + 一个事件：`enqueue_d2h` 在流上排一次异步 D2H 并记事件；`wait_read` 等事件后读。
pub struct PinnedSlot {
    host: *mut i32,
    cap: usize,
    event: sys::CUevent,
    pending: usize,
}
// SAFETY: 只在 executor 线程使用。
unsafe impl Send for PinnedSlot {}
impl PinnedSlot {
    pub fn new(stream: &Stream, cap: usize) -> Result<Self> {
        stream.bind_ctx()?;
        // SAFETY: pinned 分配 + 事件创建；Drop 释放。
        unsafe {
            let host = result::malloc_host(cap * 4, 0)? as *mut i32;
            let event = result::event::create(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)?;
            Ok(Self { host, cap, event, pending: 0 })
        }
    }
    /// 把 `src[..n]` 异步拷到本槽并记事件（调用方保证上次 `wait_read` 已完成）。
    pub fn enqueue_d2h(&mut self, stream: &Stream, src: &CudaSlice<i32>, n: usize) -> Result<()> {
        anyhow::ensure!(n <= self.cap && n <= src.len(), "PinnedSlot: n {n} > cap {}", self.cap);
        stream.bind_ctx()?;
        // SAFETY: 目标 pinned、长度已校验；流序保证读到此刻 src 的值。
        unsafe {
            let dst = std::slice::from_raw_parts_mut(self.host, n);
            result::memcpy_dtoh_async(dst, stream.ptr(src), stream.raw())?;
            result::event::record(self.event, stream.raw())?;
        }
        self.pending = n;
        Ok(())
    }
    /// 等事件完成，返回上次 enqueue 的 n 个值。
    pub fn wait_read(&mut self) -> Result<Vec<i32>> {
        // SAFETY: 事件已 record。
        unsafe { result::event::synchronize(self.event)? };
        let n = self.pending;
        self.pending = 0;
        // SAFETY: 范围内。
        Ok(unsafe { std::slice::from_raw_parts(self.host, n) }.to_vec())
    }
}
impl Drop for PinnedSlot {
    fn drop(&mut self) {
        // SAFETY: 独占。
        unsafe {
            let _ = result::free_host(self.host as *mut std::ffi::c_void);
            let _ = result::event::destroy(self.event);
        }
    }
}
