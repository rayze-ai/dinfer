// umma_peak：tcgen05（UMMA）最小 GEMM 探针（r12a 第一步，docs/loops/qwen35-tcgen05.md）。
//
// 形状 M=128, N=256，K 维循环累加。两个模式：
//   true  —— **正确性**：每轮从 gmem 重装一块 [*,64] 到 smem、每轮 commit+wait，与 CPU 参考比数值。
//   false —— **纯算力**：smem 只装一次，之后只发 MMA（与 `scripts/dev/mma_peak.cu` 同口径），
//            每 `kCommitEvery` 轮 commit 一次并等最早的那个，用来量 tcgen05 的真实峰值。
//
// 只在 CUDA_ARCHS 含 100/103 时编入（build.rs 的 sm100_only 白名单）。
#include "common.cuh"
#include "umma.cuh"

namespace {

using dinfer::bf16;
using umma::smem_u32;

constexpr int kM = 128;
constexpr int kN = 256;
constexpr int kBK = 64;   // = 128 B swizzle 的 K 跨度（bf16）
constexpr int kThreads = 128;
constexpr int kMmaPerBlock = kBK / 16;  // 一块 [*,64] = 4 条 K=16 的 MMA

// 流水深度：用 kDepth 个 mbarrier 轮转，各轮 commit 到自己的槽，等到 kDepth 轮之前那个。
// 这样 MMA 队列里始终有 kDepth 轮在飞，不被单个 commit 的完成延迟卡住。
constexpr int kDepth = 8;

// smem 布局：A tile | B tile | tmem 指针槽 | mbarrier[kDepth]
struct Layout {
  static constexpr int abytes = kM * kBK * 2;  // 16 KB
  static constexpr int bbytes = kN * kBK * 2;  // 32 KB
  static constexpr int tmem_slot = (abytes + bbytes + 15) & ~15;
  static constexpr int bar_off = tmem_slot + 16;
  static constexpr int bytes = bar_off + kDepth * 8;
};

// 把一个 [rows, 64] 的 bf16 行主序 tile 写成 128 B swizzle 布局（16 B 块号与 (row%8) 异或）。
__device__ __forceinline__ void stage_tile(bf16* dst, const bf16* src, int rows, int tid) {
  const int nblk = rows * 8;
  for (int i = tid; i < nblk; i += kThreads) {
    const int row = i >> 3, blk = i & 7;
    const int phys = row * 8 + (blk ^ (row & 7));
    *reinterpret_cast<uint4*>(dst + (size_t)phys * 8) =
        *reinterpret_cast<const uint4*>(src + (size_t)row * kBK + blk * 8);
  }
}

__global__ void __launch_bounds__(kThreads) umma_peak_kernel(const bf16* __restrict__ A,
                                                             const bf16* __restrict__ B, float* __restrict__ C,
                                                             int k_iters, int check) {
  extern __shared__ __align__(128) unsigned char smem[];
  bf16* sA = reinterpret_cast<bf16*>(smem);
  bf16* sB = sA + kM * kBK;
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(smem + Layout::tmem_slot);
  uint64_t* bar = reinterpret_cast<uint64_t*>(smem + Layout::bar_off);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int ncol = umma::tmem_cols_for(kN);

  if (tid == 0) {
    for (int i = 0; i < kDepth; ++i) umma::mbar_init(bar + i, 1);
    umma::mbar_init_fence();
  }
  __syncthreads();
  if (warp == 0) {
    umma::tmem_alloc(tmem_slot, ncol);
    umma::tmem_relinquish();
  }
  __syncthreads();
  const uint32_t tmem_c = *tmem_slot;
  const uint32_t idesc = umma::make_instr_desc_bf16(kM, kN, false, false);

  uint32_t phase = 0;
  if (check) {
    // ---------------- 正确性模式：每轮重装、每轮 commit + wait ----------------
    for (int it = 0; it < k_iters; ++it) {
      stage_tile(sA, A + (size_t)it * kM * kBK, kM, tid);
      stage_tile(sB, B + (size_t)it * kN * kBK, kN, tid);
      __syncthreads();
      if (warp == 0 && umma::elect_one()) {
        const uint64_t da = umma::k_major_128b_desc(smem_u32(sA));
        const uint64_t db = umma::k_major_128b_desc(smem_u32(sB));
#pragma unroll
        for (int k = 0; k < kMmaPerBlock; ++k) {
          const uint32_t off = (uint32_t)(k * 16 * 2) >> 4;  // 32 B → 2 个 16 B 单位
          umma::mma_f16(tmem_c, da + off, db + off, idesc, it > 0 || k > 0);
        }
        umma::mma_commit(bar);
      }
      umma::mbar_wait(bar, phase);
      phase ^= 1;
      umma::fence_after_sync();
      __syncthreads();
    }
  } else {
    // ---------------- 纯算力模式：smem 只装一次，MMA 流水 ----------------
    stage_tile(sA, A, kM, tid);
    stage_tile(sB, B, kN, tid);
    __syncthreads();
    const uint64_t da = umma::k_major_128b_desc(smem_u32(sA));
    const uint64_t db = umma::k_major_128b_desc(smem_u32(sB));
    if (warp == 0) {
      for (int it = 0; it < k_iters; ++it) {
        if (umma::elect_one()) {
#pragma unroll
          for (int k = 0; k < kMmaPerBlock; ++k) {
            const uint32_t off = (uint32_t)(k * 16 * 2) >> 4;
            umma::mma_f16(tmem_c, da + off, db + off, idesc, it > 0 || k > 0);
          }
          const int slot = it % kDepth;
          umma::mma_commit(bar + slot);
        }
        // 等 kDepth 轮之前那一轮的 commit：MMA 队列深度 = kDepth 轮。
        if (it >= kDepth - 1) {
          const int slot = (it - (kDepth - 1)) % kDepth;
          const uint32_t ph = ((it - (kDepth - 1)) / kDepth) & 1;
          umma::mbar_wait(bar + slot, ph);
        }
      }
      // 收尾：把还挂着的都等掉。
      for (int it = k_iters - (kDepth - 1); it < k_iters; ++it) {
        if (it < 0) continue;
        const int slot = it % kDepth;
        umma::mbar_wait(bar + slot, (it / kDepth) & 1);
      }
      umma::fence_before_sync();
    }
    __syncthreads();
  }

  // 取回：warp w 读 TMEM lane 32w..32w+31（地址高 16 位是 lane），每线程 4 列。
  float* Cout = C + (size_t)blockIdx.x * kM * kN;
  for (int n = 0; n < kN; n += 4) {
    uint32_t v[4];
    const uint32_t taddr = tmem_c + (uint32_t)(warp * 32u) * (1u << 16) + (uint32_t)n;
    umma::tmem_ld32_x4(taddr, v);
    umma::tmem_ld_wait();
    const int m = warp * 32 + (tid & 31);
#pragma unroll
    for (int j = 0; j < 4; ++j) Cout[(size_t)m * kN + n + j] = __uint_as_float(v[j]);
  }
  __syncthreads();
  if (warp == 0) umma::tmem_dealloc(tmem_c, ncol);
}

}  // namespace

extern "C" int umma_peak_launch(const void* a, const void* b, void* c, int64_t k_iters, int64_t blocks, int64_t check,
                                cudaStream_t stream) {
  const size_t smem = Layout::bytes;
  static int attr = 0;
  if (attr == 0) {
    attr = cudaFuncSetAttribute(umma_peak_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess
               ? 1
               : -1;
    if (attr < 0) (void)cudaGetLastError();
  }
  if (attr < 0) return (int)cudaErrorInvalidConfiguration;
  umma_peak_kernel<<<(unsigned)blocks, kThreads, smem, stream>>>(
      static_cast<const bf16*>(a), static_cast<const bf16*>(b), static_cast<float*>(c), (int)k_iters, (int)check);
  return (int)cudaGetLastError();
}
