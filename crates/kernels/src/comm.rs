//! 多卡通信：NCCL 的最小封装（TP all-reduce / all-gather）。
//!
//! 设计（design.md §13）：单进程多线程，每个 rank 一个线程 + 一个 `Comm`，绑定该 rank 的 executor 流，
//! 集合通信入 CUDA graph 捕获（NCCL ≥ 2.9 支持）。4090 Pod 无 PCIe P2P，NCCL 自动走 SHM；
//! 若探测误判可设 `NCCL_P2P_DISABLE=1`。

use anyhow::{Context, Result};
use cudarc::driver::{DevicePtr, DevicePtrMut};
use cudarc::nccl::safe::{Comm as NcclComm, Id, ReduceOp};
use cudarc::nccl::NcclType;

use crate::device::Stream;

/// 一个 TP 组里所有 rank 共享的 NCCL unique id（由 rank 0 线程生成后分发给其它线程）。
#[derive(Clone, Copy)]
pub struct CommId(pub [::core::ffi::c_char; 128]);

impl CommId {
    pub fn new() -> Result<Self> {
        let id = Id::new().map_err(|e| anyhow::anyhow!("ncclGetUniqueId: {e:?}"))?;
        Ok(Self(*id.internal()))
    }
}

pub struct Comm {
    inner: NcclComm,
    rank: usize,
    world: usize,
}

// cudarc 的 Comm 持 raw ncclComm_t；一个 Comm 只在创建它的线程 / 流上用，这里只需要 Send 以便放进 executor。
unsafe impl Send for Comm {}

impl Comm {
    /// 所有 rank 必须并发调用（ncclCommInitRank 阻塞到全组到齐）。`stream` 是集合通信使用的流。
    pub fn init(stream: &Stream, rank: usize, world: usize, id: CommId) -> Result<Self> {
        let inner = NcclComm::from_rank(stream.inner_arc(), rank, world, Id::uninit(id.0))
            .map_err(|e| anyhow::anyhow!("ncclCommInitRank rank {rank}/{world}: {e:?}"))?;
        Ok(Self { inner, rank, world })
    }
    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn world(&self) -> usize {
        self.world
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
}

impl std::fmt::Debug for Comm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Comm(rank {}/{})", self.rank, self.world)
    }
}

/// 把 `n` 个连续元素的 all-reduce 包一层，便于模型层调用（slice_mut 视图）。
// TODO(T3, 需 NVLink / P2P 机器才有意义，2026-10-18 用户决定暂不做)：
//   1) `fused_allreduce_rmsnorm`：all-reduce + residual add + RMSNorm 一个 kernel（vLLM 同名算子），省一次 hidden 读写；
//   2) 小消息（decode t×hidden ≤ 几十 KB）走自定义 one-shot all-reduce（vLLM `custom_all_reduce`），不走 NCCL ring；
//   3) 这两条都依赖 P2P；无 P2P 机器上 NCCL 走 SHM，上面任何优化都被 SHM 拷贝掩盖，不值得做。
//   接口保持 `all_reduce_bf16(comm, buf, n)` 不变，届时在这里按 n 分流。
pub fn all_reduce_bf16(comm: &Comm, buf: &mut cudarc::driver::CudaSlice<half::bf16>, n: usize) -> Result<()> {
    let mut v = buf.slice_mut(..n);
    comm.all_reduce_sum(&mut v, n).context("all_reduce_bf16")
}
