// megakernel 公共骨架（模型无关 + arch 无关）。从 Qwen3.5 的 `mega_sm100.cu` 抽出
// （见 docs/loops/mega-sm100.md / loop kaon3-mega）。**这里不放任何模型语义**：
//   - 阶段由调用方给一张 `StagePlan`（dep 表 + count 表 + start 偏移）；
//   - 任务体由调用方给一个 `Dispatch`（`run(shard, stage)`），本文件只负责调度/依赖/流水原语。
// 用法（模型侧 *_sm120.cu / *_sm100.cu）：
//   mega_run<Dispatch, MB, G, U, OCC>(plan, params, dispatch);
// 规则：kernel 体内不做 arch 分叉；arch 差异只读 `arch.cuh` 的有名常量；调优数值全由 host 传入。
#pragma once

#include <cstdint>
#include "common.cuh"

namespace dinfer {
namespace mega {

constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kChunkK = 32;  // 一个 K 块 = 两条 m16n8k16 = 每 lane 16 B（8 个 bf16）
constexpr int kMaxPipe = 6;

// ---------------------------------------------------------------- 同步原语
__device__ __forceinline__ int ld_acquire_gpu(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void red_release_gpu_add(int* p, int v) {
  asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
  return t;
}

/// 阶段依赖表（host 填）。`dep[s] < 0` 表示 stage s 无依赖（第一个阶段）。
/// `count[s]` = stage s 的任务数；`start[s]` = stage s 在全局 task 编号里的起始 offset。
/// 语义：stage s 的所有任务必须等 stage dep[s] 的 count[dep[s]] 个任务全部 signal 后才能开跑。
struct StagePlan {
  const int* dep;
  const int* count;
  const int* start;
  int* ready;  // [num_stages] 全局计数器（host 清零）
  int prof;
};

__shared__ unsigned long long s_wait_end;  // prof：本任务依赖等待结束时刻

/// 等依赖阶段全部完成（只 thread 0 自旋，其余在 barrier 等）。
template <typename P>
__device__ __forceinline__ void wait_stage(const StagePlan& sp, int stage) {
  if (stage < 0) return;
  const int dep = sp.dep[stage];
  if (dep < 0) {
    if (sp.prof && threadIdx.x == 0) s_wait_end = gtimer();
    return;
  }
  if (threadIdx.x == 0) {
    const int need = sp.count[dep];
    const int* flag = sp.ready + dep;
    while (ld_acquire_gpu(flag) < need) __nanosleep(40);
    if (sp.prof) s_wait_end = gtimer();
  }
  __syncthreads();
}

__device__ __forceinline__ void signal_stage(StagePlan& sp, int stage) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    red_release_gpu_add(sp.ready + stage, 1);
  }
}

// ---------------------------------------------------------------- cp.async（带 zfill）
__device__ __forceinline__ void cp_async16_zfill(void* smem, const void* gmem, int src_bytes) {
  const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(src_bytes));
}
/// 第 i 轮顶部已提交 pipe-1+i 个 group，需要 group i 完成 → 允许挂起 pipe-2 个。
/// wait_group 的立即数必须编译期常量；pipe 是 block 一致的运行时值（≥ 2）。
__device__ __forceinline__ void cp_async_wait_pipe(int pipe) {
  switch (pipe) {
    case 2: asm volatile("cp.async.wait_group 0;\n" ::); break;
    case 3: asm volatile("cp.async.wait_group 1;\n" ::); break;
    case 4: asm volatile("cp.async.wait_group 2;\n" ::); break;
    case 5: asm volatile("cp.async.wait_group 3;\n" ::); break;
    default: asm volatile("cp.async.wait_group 4;\n" ::); break;
  }
}

// ---------------------------------------------------------------- mma / swizzle
__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t a0, const uint32_t a1, const uint32_t a2,
                                         const uint32_t a3, const uint32_t b0, const uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

/// smem tile 行内 16 B 块的 swizzle：mma 片段读是 8 行 × 4 块一个 wavefront；
/// 行距 4U 块（U=2 → 32 B，4 → 64 B，8 → 128 B）。奇数行把块号的 bit2 翻转。
__device__ __forceinline__ int swz(int row, int chunk) { return chunk ^ ((row & 1) << 2); }

// ---------------------------------------------------------------- persistent 外壳
/// 全局 task 编号 → (stage, shard) 的二分反查。host 保证 start[] 单调。
template <typename P>
__device__ __forceinline__ int stage_of(const StagePlan& sp, int task, int num_stages) {
  // 找最大的 s 使 start[s] <= task
  int lo = 0, hi = num_stages - 1;
  while (lo < hi) {
    int mid = (lo + hi + 1) >> 1;
    if (sp.start[mid] <= task) lo = mid; else hi = mid - 1;
  }
  return lo;
}

/// 持久 CTA 主循环：blockIdx 轮转领取全局 task，领到后二分出 (stage, shard)，
/// 等依赖 → 跑任务体 → signal。dispatch.run(shard, stage) 由模型侧提供。
template <typename Dispatch, int OCC>
__global__ void __launch_bounds__(kThreads, OCC) mega_kernel(StagePlan sp, typename Dispatch::Params params,
                                                             Dispatch dispatch, int total_tasks, int num_stages,
                                                             int start_offset) {
  extern __shared__ uint8_t smem[];
  for (int task = (int)blockIdx.x + start_offset; task < total_tasks; task += (int)gridDim.x) {
    const int stage = stage_of<typename Dispatch::Params>(sp, task, num_stages);
    const int shard = task - sp.start[stage];
    wait_stage<typename Dispatch::Params>(sp, stage);
    dispatch.run(params, shard, stage, smem);
    signal_stage(sp, stage);
  }
}

}  // namespace mega
}  // namespace dinfer
