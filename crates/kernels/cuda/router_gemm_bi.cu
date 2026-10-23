// router 的 BI persistent matmul（复刻 vLLM `batch_invariant.py::matmul_persistent`）。
//
// **为什么必须单独写**（docs/loops/kaon3-compile.md 第 5 轮）：
// vLLM 在 `VLLM_BATCH_INVARIANT=1` 下，`linear.py:209` 显式走 `linear_batch_invariant` →
// `matmul_persistent`（Triton），**即便 sm_120 不装 dispatcher patch 也一样**。
// 该 kernel 与 cuBLAS 在 router 形状（M×128×2816）上实测 **43% 输出元素不同**（max 7.8e-3），
// 而 router logits 差一个 bf16 ulp 就会在 top-8 边界换专家 → 端到端 16% flip。
//
// **参照的归约结构**（必须逐位照抄，换了就不对）：
//   accumulator = fp32 零
//   for ki in 0..K/64:  accumulator = mma(a[.., 64*ki : 64*ki+64], b[同段], accumulator)
//   c = bf16(accumulator)                    ← 只在最后舍一次
// 即 K 方向**严格顺序**，一个 fp32 累加器。BLOCK_SIZE_M/N/K = 128/128/64（bf16 配置）。
//
// 本 kernel 的结构（BM=128、BN=128、BK=64；N=128 时网格只有一列）：
//   - block 负责 BM=128 行 × BN=128 列（= 全部专家，num_experts 必须是 128 的倍数，当前 128）；
//   - 8 个 warp，每 warp 16 行（m16 → 每 warp 一条 mma 链），每 warp 管 BN/8 = 16 个 n8 tile；
//   - K 按 64 一块：每块内 4 条 `mma.m16n8k16`（k16）**顺序**累加，块间也顺序 → 与参照同序；
//   - A 分块进 smem（16 行 × 64 列 bf16 = 2 KB 每 warp），W 直接 global 读（N 小、L2 命中）；
//   - 输出 f32。
#include <cstdlib>
#include "common.cuh"
#include "kernels.h"

namespace {
using dinfer::bf16;
using dinfer::to_bf16;
using dinfer::to_f32;
using dinfer::err;
using dinfer::last_err;

constexpr int BI_BM = 128;   // BLOCK_SIZE_M
constexpr int BI_BN = 128;   // BLOCK_SIZE_N
constexpr int BI_BK = 64;    // BLOCK_SIZE_K
constexpr int BI_WARPS = 8;  // 每 warp 16 行

__device__ __forceinline__ void bi_mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                            uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// C[M,N] f32 = A[M,K] bf16 · W[N,K]^T bf16，K 方向严格顺序累加（= 参照 matmul_persistent）。
// grid = (ceil(M / BI_BM), ceil(N / BI_BN))，block = 32 * BI_WARPS。
__global__ void __launch_bounds__(32 * BI_WARPS) router_gemm_bi_kernel(const bf16* __restrict__ a,
                                                                      const bf16* __restrict__ w,
                                                                      float* __restrict__ c, int M, int N, int K) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row0 = (int)blockIdx.x * BI_BM + warp * 16;  // 本 warp 的 16 行
  const int col0 = (int)blockIdx.y * BI_BN;

  __shared__ __align__(16) bf16 a_tile[BI_WARPS][16][BI_BK + 8];  // **每 warp 一份**：8 warp 各 16 行

  const int ktiles = K / BI_BK;
  // 每 warp 负责 BN/8 个 n8 tile（n8 是 mma 的 N 维粒度）。
  constexpr int NT = BI_BN / 8;
  float acc[NT][4];
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[t][j] = 0.f;

  const int g = lane >> 2;  // m16 内行号组（0..7），每 lane 持 2 行：g 与 g+8
  const int th = lane & 3;  // k16 内列号组（0..3），每 lane 持 2 列：2*th 与 2*th+1

  for (int ki = 0; ki < ktiles; ++ki) {
    const int kbase = ki * BI_BK;
    // A 块进 smem：16 行 × 64 列。每 lane 搬 16*64/32 = 32 个元素 = 4 个 uint4。
    {
      const int r = lane >> 1;              // 0..15 行
      const int cq = lane & 1;              // 每行 2 个 16B 的哪一半
      const bf16* src = a + (int64_t)(row0 + r) * K + kbase + cq * 32;
      uint4* dst = reinterpret_cast<uint4*>(&a_tile[warp][r][cq * 32]);
      if (row0 + r < M) {
        *dst = *reinterpret_cast<const uint4*>(src);  // 32 个 bf16 = 4 × uint4 的 1/2
        *reinterpret_cast<uint4*>(&a_tile[warp][r][cq * 32 + 8]) = *reinterpret_cast<const uint4*>(src + 8);
        *reinterpret_cast<uint4*>(&a_tile[warp][r][cq * 32 + 16]) = *reinterpret_cast<const uint4*>(src + 16);
        *reinterpret_cast<uint4*>(&a_tile[warp][r][cq * 32 + 24]) = *reinterpret_cast<const uint4*>(src + 24);
      } else {
        *reinterpret_cast<uint4*>(&a_tile[warp][r][cq * 32]) = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(&a_tile[warp][r][cq * 32 + 8]) = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(&a_tile[warp][r][cq * 32 + 16]) = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(&a_tile[warp][r][cq * 32 + 24]) = make_uint4(0, 0, 0, 0);
      }
    }
    __syncwarp();

    // 4 条 k16 mma，严格顺序（参照 BK=64 内也是顺序累加）。
#pragma unroll
    for (int k16 = 0; k16 < 4; ++k16) {
      const int kc = k16 * 16;
      // A 片段（m16k16，4 个 32 位寄存器 = 8 个 bf16）：
      //   reg0 = (g,     2*th), (g,     2*th+1)
      //   reg1 = (g+8,   2*th), (g+8,   2*th+1)
      //   reg2 = (g,     2*th+8), (g,   2*th+9)
      //   reg3 = (g+8,   2*th+8), (g+8, 2*th+9)
      uint32_t a0 = *reinterpret_cast<const uint32_t*>(&a_tile[warp][g][kc + 2 * th]);
      uint32_t a1 = *reinterpret_cast<const uint32_t*>(&a_tile[warp][g + 8][kc + 2 * th]);
      uint32_t a2 = *reinterpret_cast<const uint32_t*>(&a_tile[warp][g][kc + 2 * th + 8]);
      uint32_t a3 = *reinterpret_cast<const uint32_t*>(&a_tile[warp][g + 8][kc + 2 * th + 8]);
      // B 片段：k16n8，W 是 [N,K] 行主序，nv 取第 (col0 + t*8 + g) 行。
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        const int nrow = col0 + t * 8 + g;
        uint32_t b0 = *reinterpret_cast<const uint32_t*>(w + (int64_t)nrow * K + kbase + kc + 2 * th);
        uint32_t b1 = *reinterpret_cast<const uint32_t*>(w + (int64_t)nrow * K + kbase + kc + 2 * th + 8);
        bi_mma16816(acc[t], a0, a1, a2, a3, b0, b1);
      }
    }
    __syncwarp();
  }

  // 写回：参照是 `c = accumulator.to(bf16)`（输出 dtype = 输入 dtype），**先舍 bf16 再转 f32**。
  // 忘了这一步会得到"原始 f32 累加和"，实测 max|d| 3.9e-3（正好一个 bf16 ulp，全部 8192 元素都不同）。
  // mma 的 c 布局 —— lane 持 (g, 2*th) (g, 2*th+1) (g+8, 2*th) (g+8, 2*th+1)。
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    const int nc = col0 + t * 8 + 2 * th;
    const int r0 = row0 + g, r1 = row0 + g + 8;
    if (r0 < M) {
      if (nc < N) c[(int64_t)r0 * N + nc] = to_f32(to_bf16(acc[t][0]));
      if (nc + 1 < N) c[(int64_t)r0 * N + nc + 1] = to_f32(to_bf16(acc[t][1]));
    }
    if (r1 < M) {
      if (nc < N) c[(int64_t)r1 * N + nc] = to_f32(to_bf16(acc[t][2]));
      if (nc + 1 < N) c[(int64_t)r1 * N + nc + 1] = to_f32(to_bf16(acc[t][3]));
    }
  }
}

}  // namespace

extern "C" int router_gemm_bi_f32(const void* x, const void* w, float* out, int64_t tokens, int64_t num_experts,
                                  int64_t hidden, cudaStream_t stream) {
  if (tokens <= 0) return 0;
  // 参照的 BLOCK_SIZE 配置要求这些整除；不满足时调用方应退回 cuBLAS。
  if (hidden % BI_BK != 0 || num_experts % 8 != 0) return err(cudaErrorInvalidValue);
  dim3 grid((unsigned)((tokens + BI_BM - 1) / BI_BM), (unsigned)((num_experts + BI_BN - 1) / BI_BN));
  router_gemm_bi_kernel<<<grid, 32 * BI_WARPS, 0, stream>>>(static_cast<const dinfer::bf16*>(x), static_cast<const dinfer::bf16*>(w),
                                                           out, (int)tokens, (int)num_experts, (int)hidden);
  return last_err();
}
