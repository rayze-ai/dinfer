//! 多卡通信：NCCL 的最小封装（TP all-reduce / all-gather）+ 小消息自定义 one-shot all-reduce。
//!
//! 设计（design.md §13）：单进程多线程，每个 rank 一个线程 + 一个 `Comm`，绑定该 rank 的 executor 流，
//! 集合通信入 CUDA graph 捕获（NCCL ≥ 2.9 支持）。4090 Pod 无 PCIe P2P，NCCL 自动走 SHM；
//! 若探测误判可设 `NCCL_P2P_DISABLE=1`。
//!
//! 自定义 one-shot all-reduce（design §13.6 T3，对齐 vLLM `custom_all_reduce`，§11 "采用"）：
//! world ∈ {2,4,8}、全 P2P、消息 ≤ 512 KB（world ≤ 4）/ 256 KB（world 8）时，各 rank 把输入 D2D 进自己的设备
//! 镜像区，一个 kernel 直接读全部 peer 镜像区求和（固定 rank 顺序 → 逐位一致），比 NCCL ring 省一半以上延迟
//! （2 卡 ~3 µs vs ~8 µs）。更大消息留 NCCL。开关：`CUSTOM_AR=0` 关。
//! 建立流程（信息交换由上层 `TpExecutor` 在进程内做，不走文件 / NCCL）：
//! `custom_ar_begin()` → 各 rank 的 `(device ordinal, 区设备地址, ipc handle)` 收齐 → `custom_ar_finish(peers)`；
//! 任一 rank 失败（无 P2P / peer access 失败）→ 所有 rank `custom_ar_disable()` 回退 NCCL（各 rank 必须一致，否则死锁）。
//! **同进程不能 `cuIpcOpenMemHandle` 自己进程导出的 handle**（CUDA 限制，B300 实测 2026-10-19；vLLM 多进程所以能用），
//! 一进程多线程下直接用对端 `cuMemAlloc` 的裸指针（UVA 进程内唯一）+ `cuCtxEnablePeerAccess(对端主 ctx)`；
//! handle 仍导出并随信息传递，留给将来多进程（DP / PD 跨进程）用。

use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::sys;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use cudarc::nccl::safe::{Comm as NcclComm, Id, ReduceOp};
use cudarc::nccl::NcclType;
use half::bf16;
use std::ffi::c_void;

use crate::device::Stream;
use crate::ffi;

/// 一个 TP 组里所有 rank 共享的 NCCL unique id（由 rank 0 线程生成后分发给其它线程）。
#[derive(Clone, Copy)]
pub struct CommId(pub [::core::ffi::c_char; 128]);

impl CommId {
    pub fn new() -> Result<Self> {
        let id = Id::new().map_err(|e| anyhow::anyhow!("ncclGetUniqueId: {e:?}"))?;
        Ok(Self(*id.internal()))
    }
}

// ---------------------------------------------------------------- cuda-ipc 设备缓冲

/// 本进程 `cuMemAlloc` + `cuIpcGetMemHandle` 的设备区（导出 handle 给其它 rank 打开）。Drop 释放。
pub struct IpcBuf {
    ptr: u64,
    bytes: usize,
    handle: [u8; 64],
}
unsafe impl Send for IpcBuf {}

impl IpcBuf {
    /// 在 `stream` 所属设备上分配 `bytes` 字节并清零（同步）。
    pub fn create(stream: &Stream, bytes: usize) -> Result<Self> {
        stream.bind_ctx()?;
        let mut ptr: sys::CUdeviceptr = 0;
        let mut h = sys::CUipcMemHandle { reserved: [0; 64] };
        // SAFETY: ctx 已绑定；失败走 result。
        unsafe {
            sys::cuMemAlloc_v2(&mut ptr, bytes).result().with_context(|| format!("cuMemAlloc {bytes} B (ipc)"))?;
            let r = sys::cuMemsetD8_v2(ptr, 0, bytes).result().and_then(|_| sys::cuIpcGetMemHandle(&mut h, ptr).result());
            if let Err(e) = r {
                let _ = sys::cuMemFree_v2(ptr);
                return Err(e).context("cuIpcGetMemHandle");
            }
        }
        let mut handle = [0u8; 64];
        for (d, s) in handle.iter_mut().zip(h.reserved.iter()) {
            *d = *s as u8;
        }
        Ok(Self { ptr, bytes, handle })
    }
    pub fn ptr(&self) -> u64 {
        self.ptr
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn handle(&self) -> [u8; 64] {
        self.handle
    }
}
impl Drop for IpcBuf {
    fn drop(&mut self) {
        // SAFETY: 本进程分配。
        unsafe {
            let _ = sys::cuMemFree_v2(self.ptr);
        }
    }
}

/// 让当前线程的 ctx（本 rank 设备的主 ctx）能直接访问 `peer_ordinal` 设备的内存（`cuCtxEnablePeerAccess`）。
/// 已开过（CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED）视为成功。
pub fn enable_peer_access(peer_ordinal: i32) -> Result<()> {
    let dev = cu_device(peer_ordinal)?;
    let mut ctx: sys::CUcontext = std::ptr::null_mut();
    // SAFETY: 主 ctx retain 只加引用计数（cudarc 已为该设备 retain 过，ctx 存活）；enable 对已开的返回 ALREADY_ENABLED。
    unsafe {
        sys::cuDevicePrimaryCtxRetain(&mut ctx, dev).result().context("cuDevicePrimaryCtxRetain(peer)")?;
        match sys::cuCtxEnablePeerAccess(ctx, 0) {
            sys::cudaError_enum::CUDA_SUCCESS | sys::cudaError_enum::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => {}
            e => {
                let _ = sys::cuDevicePrimaryCtxRelease_v2(dev);
                bail!("cuCtxEnablePeerAccess(device {peer_ordinal}): {e:?}");
            }
        }
        // 释放我们这次 retain 的计数（cudarc 的那份仍在，ctx 不会被销毁；peer access 属于 ctx，与计数无关）
        let _ = sys::cuDevicePrimaryCtxRelease_v2(dev);
    }
    Ok(())
}

/// `canAccessPeer(dev, peer)`。
pub fn can_access_peer(dev: sys::CUdevice, peer: sys::CUdevice) -> Result<bool> {
    let mut v: i32 = 0;
    // SAFETY: 纯查询。
    unsafe {
        sys::cuDeviceCanAccessPeer(&mut v, dev, peer).result().context("cuDeviceCanAccessPeer")?;
    }
    Ok(v != 0)
}

fn cu_device(ordinal: i32) -> Result<sys::CUdevice> {
    let mut d: sys::CUdevice = 0;
    // SAFETY: 纯查询。
    unsafe {
        sys::cuDeviceGet(&mut d, ordinal).result().context("cuDeviceGet")?;
    }
    Ok(d)
}

// ---------------------------------------------------------------- 自定义 one-shot all-reduce 状态

/// 建立时每个 rank 公布的信息。`ptr` = 本进程内可直接用的区设备地址（UVA）；`handle` 留给多进程场景。
#[derive(Clone, Copy, Debug)]
pub struct ArPeerInfo {
    pub device_ordinal: i32,
    pub ptr: u64,
    pub handle: [u8; 64],
}

/// 镜像区默认容量（B）；阈值以下的消息才走 one-shot，所以这就是上限。
pub const CUSTOM_AR_MAX_BYTES: usize = 1 << 20;

struct CustomAr {
    /// 本 rank 的区：[Signal | mirror(max_bytes)]
    local: IpcBuf,
    /// 各 rank Signal 设备地址（host 数组，传 kernel）
    signals: Vec<*const c_void>,
    /// 设备 RankData（ptrs[r] = rank r 镜像区）
    rank_data: CudaSlice<u8>,
    signal_bytes: usize,
    max_bytes: usize,
}
unsafe impl Send for CustomAr {}

impl CustomAr {
    fn mirror_ptr(&self) -> u64 {
        self.local.ptr() + self.signal_bytes as u64
    }
}

enum CustomState {
    /// 未开始 / 已禁用
    Off,
    /// `custom_ar_begin` 后：本地区已建，等 peers
    Pending(IpcBuf),
    Ready(CustomAr),
}

pub struct Comm {
    inner: NcclComm,
    stream: Stream,
    rank: usize,
    world: usize,
    custom: CustomState,
}

// cudarc 的 Comm 持 raw ncclComm_t；一个 Comm 只在创建它的线程 / 流上用，这里只需要 Send 以便放进 executor。
unsafe impl Send for Comm {}

/// 环境开关：`CUSTOM_AR=0` 关自定义 all-reduce（全走 NCCL）。
fn custom_ar_enabled_by_env() -> bool {
    std::env::var("CUSTOM_AR").map(|v| v != "0").unwrap_or(true)
}

impl Comm {
    /// 所有 rank 必须并发调用（ncclCommInitRank 阻塞到全组到齐）。`stream` 是集合通信使用的流。
    pub fn init(stream: &Stream, rank: usize, world: usize, id: CommId) -> Result<Self> {
        let inner = NcclComm::from_rank(stream.inner_arc(), rank, world, Id::uninit(id.0))
            .map_err(|e| anyhow::anyhow!("ncclCommInitRank rank {rank}/{world}: {e:?}"))?;
        Ok(Self { inner, stream: stream.clone(), rank, world, custom: CustomState::Off })
    }
    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn world(&self) -> usize {
        self.world
    }
    /// 自定义 one-shot all-reduce 是否已建立。
    pub fn custom_ar_ready(&self) -> bool {
        matches!(self.custom, CustomState::Ready(_))
    }
    /// 原地 sum all-reduce，前 `n` 个元素。
    pub fn all_reduce_sum<T: NcclType, B: DevicePtrMut<T>>(&self, buf: &mut B, n: usize) -> Result<()> {
        let _ = n; // cudarc 用 buf.len()；调用方传入已 slice 的视图
        self.inner.all_reduce_in_place(buf, &ReduceOp::Sum).map(|_| ()).map_err(|e| anyhow::anyhow!("ncclAllReduce: {e:?}"))
    }
    /// all-gather：每 rank 的 `send` 拼成 `recv`（按 rank 顺序）。
    pub fn all_gather<T: NcclType, S: DevicePtr<T>, R: DevicePtrMut<T>>(&self, send: &S, recv: &mut R) -> Result<()> {
        self.inner.all_gather(send, recv).map(|_| ()).map_err(|e| anyhow::anyhow!("ncclAllGather: {e:?}"))
    }

    // ------------------------------------------------------------ 自定义 all-reduce 建立（三步，见模块注释）

    /// 第 1 步：建本 rank 的 ipc 区，返回要公布给其它 rank 的信息；`None` = 本 rank 不参与（env 关 / world 不支持 /
    /// 分配失败），调用方应让所有 rank `custom_ar_disable`。
    pub fn custom_ar_begin(&mut self) -> Result<Option<ArPeerInfo>> {
        self.custom = CustomState::Off;
        if !custom_ar_enabled_by_env() {
            tracing::info!(rank = self.rank, "custom all-reduce: off (CUSTOM_AR=0)");
            return Ok(None);
        }
        if !matches!(self.world, 2 | 4 | 8) {
            tracing::info!(rank = self.rank, world = self.world, "custom all-reduce: off (world not in {{2,4,8}})");
            return Ok(None);
        }
        // SAFETY: 纯常量查询。
        let sig = unsafe { ffi::ar_signal_bytes() } as usize;
        if sig == 0 {
            return Ok(None); // no-nvcc 构建
        }
        let buf = match IpcBuf::create(&self.stream, sig + CUSTOM_AR_MAX_BYTES) {
            Ok(b) => b,
            Err(e) => {
                tracing::info!(rank = self.rank, err = format!("{e:#}"), "custom all-reduce: off (ipc buffer create failed)");
                return Ok(None);
            }
        };
        let info = ArPeerInfo { device_ordinal: self.stream.device().ordinal() as i32, ptr: buf.ptr(), handle: buf.handle() };
        self.custom = CustomState::Pending(buf);
        Ok(Some(info))
    }

    /// 第 2 步：拿到全部 rank 的信息（index = rank，含自己）后打开 peer 区、上传 RankData。
    /// 返回 `false` = 本 rank 建立失败（无 P2P / open 失败），已自行回到 Off；调用方应让所有 rank disable。
    pub fn custom_ar_finish(&mut self, peers: &[ArPeerInfo]) -> Result<bool> {
        let local = match std::mem::replace(&mut self.custom, CustomState::Off) {
            CustomState::Pending(b) => b,
            _ => return Ok(false),
        };
        ensure!(peers.len() == self.world, "custom_ar_finish: {} peers != world {}", peers.len(), self.world);
        match self.open_peers(local, peers) {
            Ok(c) => {
                tracing::info!(rank = self.rank, world = self.world, max_kb = c.max_bytes >> 10, "custom all-reduce: ready (one-shot, in-process peer access)");
                self.custom = CustomState::Ready(c);
                Ok(true)
            }
            Err(e) => {
                tracing::info!(rank = self.rank, err = format!("{e:#}"), "custom all-reduce: off, fallback NCCL");
                Ok(false)
            }
        }
    }

    /// 任一 rank 失败时所有 rank 调用：回到纯 NCCL。
    pub fn custom_ar_disable(&mut self) {
        if !matches!(self.custom, CustomState::Off) {
            tracing::info!(rank = self.rank, "custom all-reduce: disabled (peer failed), fallback NCCL");
        }
        self.custom = CustomState::Off;
    }

    fn open_peers(&self, local: IpcBuf, peers: &[ArPeerInfo]) -> Result<CustomAr> {
        // SAFETY: 常量查询。
        let signal_bytes = unsafe { ffi::ar_signal_bytes() } as usize;
        let rd_bytes = unsafe { ffi::ar_rank_data_bytes() } as usize;
        let me = cu_device(peers[self.rank].device_ordinal)?;
        self.stream.bind_ctx()?;
        let mut bases: Vec<u64> = Vec::with_capacity(self.world);
        for (r, p) in peers.iter().enumerate() {
            if r == self.rank {
                bases.push(local.ptr());
                continue;
            }
            let dev = cu_device(p.device_ordinal)?;
            ensure!(p.device_ordinal != peers[self.rank].device_ordinal, "rank {r} on same device");
            ensure!(can_access_peer(me, dev)?, "no P2P between device {} and {}", peers[self.rank].device_ordinal, p.device_ordinal);
            enable_peer_access(p.device_ordinal).with_context(|| format!("peer access to rank {r}"))?;
            bases.push(p.ptr);
        }
        let signals: Vec<*const c_void> = bases.iter().map(|b| *b as *const c_void).collect();
        // RankData.ptrs[r] = 镜像区（Signal 之后）；8 槽，未用的填 0
        let mut rd = vec![0u8; rd_bytes.max(8 * 8)];
        for (r, b) in bases.iter().enumerate() {
            let m = b + signal_bytes as u64;
            rd[r * 8..r * 8 + 8].copy_from_slice(&m.to_ne_bytes());
        }
        let rank_data = self.stream.htod_copy(rd)?;
        self.stream.synchronize()?;
        Ok(CustomAr { local, signals, rank_data, signal_bytes, max_bytes: CUSTOM_AR_MAX_BYTES })
    }

    /// `n` 个 bf16 的消息是否走 one-shot（vLLM 阈值：world ≤ 4 → <512 KB；world 8 → <256 KB）。
    fn custom_for(&self, n: usize) -> Option<&CustomAr> {
        let c = match &self.custom {
            CustomState::Ready(c) => c,
            _ => return None,
        };
        let bytes = n * 2;
        let limit = if self.world <= 4 { 512 << 10 } else { 256 << 10 };
        if n % 8 != 0 || bytes == 0 || bytes > limit || bytes > c.max_bytes {
            return None;
        }
        Some(c)
    }

    /// 把本 rank 输入 D2D 进镜像区（流序保证 kernel 看到）。
    fn stage(&self, c: &CustomAr, src: u64, bytes: usize) -> Result<()> {
        // SAFETY: 两段都是本 ctx 可见的设备地址，长度已校验。
        unsafe {
            sys::cuMemcpyDtoDAsync_v2(c.mirror_ptr(), src, bytes, self.stream.raw()).result().context("stage mirror (D2D)")?;
        }
        Ok(())
    }

    /// one-shot：`buf[..n] = Σ_r buf_r[..n]`（原地）。调用前已判定 `custom_for(n)` 为 Some。
    fn oneshot_inplace(&self, c: &CustomAr, buf: &mut CudaSlice<bf16>, n: usize) -> Result<()> {
        let p = self.stream.ptr_mut(buf);
        self.stage(c, p, n * 2)?;
        let rd = self.stream.ptr(&c.rank_data) as *mut c_void;
        // SAFETY: rank_data / signals 在 Ready 期间固定；out 容量已校验。
        let rc = unsafe {
            ffi::ar_oneshot_bf16(rd, c.signals.as_ptr(), p as *mut c_void, self.rank as i32, self.world as i32, n as i64, self.stream.raw() as _)
        };
        if rc != 0 {
            bail!("ar_oneshot_bf16 failed: cudaError {rc}");
        }
        Ok(())
    }
}

impl std::fmt::Debug for Comm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Comm(rank {}/{}, custom_ar={})", self.rank, self.world, self.custom_ar_ready())
    }
}

/// 把 `n` 个连续元素的 all-reduce 包一层，便于模型层调用（slice_mut 视图）。
/// 按消息大小分流：小消息（见 `custom_for`）走 one-shot 自定义 kernel，否则 NCCL。
// TODO(T3-3)：two-stage（reduce-scatter + all-gather）版本给 > 阈值的消息（prefill）；现在走 NCCL。
pub fn all_reduce_bf16(comm: &Comm, buf: &mut CudaSlice<bf16>, n: usize) -> Result<()> {
    ensure!(buf.len() >= n, "all_reduce_bf16: n {n} > len {}", buf.len());
    if let Some(c) = comm.custom_for(n) {
        return comm.oneshot_inplace(c, buf, n).context("all_reduce_bf16 (one-shot)");
    }
    let mut v = buf.slice_mut(..n);
    comm.all_reduce_sum(&mut v, n).context("all_reduce_bf16")
}

/// 强制走 NCCL 的 all-reduce（测试对拍用）。
pub fn all_reduce_bf16_nccl(comm: &Comm, buf: &mut CudaSlice<bf16>, n: usize) -> Result<()> {
    ensure!(buf.len() >= n, "all_reduce_bf16_nccl: n {n} > len {}", buf.len());
    let mut v = buf.slice_mut(..n);
    comm.all_reduce_sum(&mut v, n).context("all_reduce_bf16_nccl")
}

/// all-reduce + 残差 + RMSNorm：`res[t,:] += Σ_r part_r[t,:]; normed[t,:] = rmsnorm(res[t,:]) · w`。
/// 有 one-shot 时一个 kernel（`part` 不被改写，和不落回）；否则 NCCL 原地 all-reduce `part` 再 `fused_add_rmsnorm`。
#[allow(clippy::too_many_arguments)]
pub fn all_reduce_add_rmsnorm_bf16(
    comm: &Comm,
    part: &mut CudaSlice<bf16>,
    res: &mut CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    normed: &mut CudaSlice<bf16>,
    tokens: usize,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    let n = tokens * hidden;
    ensure!(
        part.len() >= n && res.len() >= n && normed.len() >= n && w.len() == hidden,
        "all_reduce_add_rmsnorm_bf16: shape"
    );
    if hidden % 8 == 0 && hidden <= 4096 {
        if let Some(c) = comm.custom_for(n) {
            let s = &comm.stream;
            let pp = s.ptr(part);
            comm.stage(c, pp, n * 2)?;
            let rd = s.ptr(&c.rank_data) as *mut c_void;
            let pr = s.ptr_mut(res) as *mut c_void;
            let pw = s.ptr(w) as *const c_void;
            let po = s.ptr_mut(normed) as *mut c_void;
            // SAFETY: 容量已校验；rank_data / signals 固定。
            let rc = unsafe {
                ffi::ar_oneshot_add_rmsnorm_bf16(
                    rd,
                    c.signals.as_ptr(),
                    pr,
                    pw,
                    po,
                    comm.rank as i32,
                    comm.world as i32,
                    tokens as i64,
                    hidden as i64,
                    eps,
                    s.raw() as _,
                )
            };
            if rc != 0 {
                bail!("ar_oneshot_add_rmsnorm_bf16 failed: cudaError {rc}");
            }
            return Ok(());
        }
    }
    all_reduce_bf16(comm, part, n)?;
    crate::ops::fused_add_rmsnorm_bf16_n(&comm.stream, part, res, w, normed, tokens, hidden, eps)
}

/// `recv[..world*n] = concat_r send_r[..n]`（bf16，按 rank 顺序连续）。只追加包装，供模型层按输出维切分的 GEMM
///（如 lm_head 列切）把各 rank 局部结果拼成全量；拼出来是 `[world][rows][cols]` 块布局，调用方再修正布局。
pub fn all_gather_bf16(
    comm: &Comm,
    send: &cudarc::driver::CudaSlice<half::bf16>,
    recv: &mut cudarc::driver::CudaSlice<half::bf16>,
    n: usize,
) -> Result<()> {
    let world = comm.world();
    anyhow::ensure!(send.len() >= n && recv.len() >= world * n, "all_gather_bf16: buffer ({} / {} vs n={n} world={world})", send.len(), recv.len());
    let s = send.slice(..n);
    let mut r = recv.slice_mut(..world * n);
    comm.all_gather(&s, &mut r).context("all_gather_bf16")
}
