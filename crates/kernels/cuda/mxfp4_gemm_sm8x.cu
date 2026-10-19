// MXFP4 grouped GEMM（sm80+ 版：权重在 kernel 内解包，仿真 bf16 乘加、fp32 累加）。
// 规格见 docs/loops/k3-moe.md「3b 规格」。模型无关：只按形状与分组参数。
//
//   out[r, n] = Σ_k dequant(w[e, n, k]) · xg[r, k]        r ∈ 专家 e 的 token 段
//
// 分组语义与 moe.cu 一致：`xg` 已按 `moe_gather_bf16` 的排序结果收集，专家 e 的 token 占
// 行区间 `[expert_off[e], expert_off[e+1])`。权重 packed 布局见 mxfp4.cu 顶部：
// u8 [E, N, K/2] + e8m0 scale u8 [E, N, K/group]，group = 32。
//
// 本文件是 **sm8x 路径**（v1，正确性优先）：权重按 tile 解包到共享内存，然后逐元素乘加。
// 原生 FP4 tensor core 走 `mxfp4_gemm_sm100.cu`（空壳 + TODO），两份按 ArchProfile 由 host 选；
// kernel 体内不出现 `#if __CUDA_ARCH__`（CLAUDE.md 多卡型规则）。
//
// 线程分工：256 线程覆盖一个 BM×BN tile；每线程负责同一行 i 的 4 个连续输出列，
// 扫 K 时 x[i][k] 读一次喂 4 次 FMA（与 kda_chunk.cu 的 u/w 同款 4 路分块）。
#include "common.cuh"
#include "kernels.h"

using namespace dinfer;

namespace {

// 幅值表：下标 = 3 位指数（e2m1 的偏置 1）
__constant__ float kE2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

__device__ __forceinline__ float e2m1_val(unsigned nibble) {
  const float mag = kE2M1[nibble & 7u];
  return (nibble & 8u) ? -mag : mag;
}

// e8m0 = 2^(u8 - 127)：直接构造 fp32 指数位
__device__ __forceinline__ float e8m0_val(unsigned s) { return __uint_as_float(s << 23); }

// 每 block 一个 (专家, M tile, N tile)。块三元组由 host 展平（每专家 token 数不同，网格不是矩形）。
template <int BM, int BN, int BK>
__global__ void gemm_tile_kernel(const bf16* __restrict__ xg, const uint8_t* __restrict__ wp,
                                 const uint8_t* __restrict__ ws, const int32_t* __restrict__ expert_off,
                                 const int32_t* __restrict__ tile_e, const int32_t* __restrict__ tile_m,
                                 const int32_t* __restrict__ tile_n, bf16* __restrict__ out, int K, int N,
                                 int group) {
  const int bid = blockIdx.x;
  const int e = tile_e[bid];
  const int start = expert_off[e];
  const int m0 = start + tile_m[bid] * BM;
  const int rows = min(BM, expert_off[e + 1] - m0);
  if (rows <= 0) return;
  const int n0 = tile_n[bid] * BN;

  const int tid = threadIdx.x;
  const int nb = blockDim.x;
  const int KB = K / 2;
  const int NG = K / group;
  constexpr int JPT = 4;              // 每线程的输出列数
  constexpr int ITH = BN / JPT;       // 每行用掉的线程数
  static_assert(BN % JPT == 0, "BN % JPT");
  static_assert(BM * ITH == 256, "BM*BN 必须整除 256 线程（v1 的固定分工）");

  extern __shared__ char smraw[];
  bf16* sw = reinterpret_cast<bf16*>(smraw);              // [BN][BK]
  bf16* sx = sw + BN * BK;                                // [BM][BK]

  const uint8_t* wp_e = wp + (long)e * N * KB;
  const uint8_t* ws_e = ws + (long)e * N * NG;

  const int i = tid / ITH;            // tile 内的行
  const int j0 = (tid % ITH) * JPT;   // tile 内的首列
  const bool row_ok = i < rows;

  float acc[JPT];
  for (int t = 0; t < JPT; ++t) acc[t] = 0.f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    // ---- 解包权重 tile：sw[n][kk] = e2m1(nibble) · e8m0(scale)
    for (int idx = tid; idx < BN * BK; idx += nb) {
      const int n = idx / BK, kk = idx % BK;
      const int gn = n0 + n, gk = k0 + kk;
      const uint8_t byte = wp_e[(long)gn * KB + (gk >> 1)];
      const float v = (gk & 1) ? e2m1_val((byte >> 4) & 0xFu) : e2m1_val(byte & 0xFu);
      sw[idx] = to_bf16(v * e8m0_val(ws_e[(long)gn * NG + (gk / group)]));
    }
    // ---- 激活 tile
    for (int idx = tid; idx < BM * BK; idx += nb) {
      const int r = idx / BK, kk = idx % BK;
      sx[idx] = (r < rows) ? xg[(long)(m0 + r) * K + k0 + kk] : to_bf16(0.f);
    }
    __syncthreads();
    // ---- 累加：x 一行读一次，喂 JPT 个输出列
    if (row_ok) {
      const bf16* xr = sx + i * BK;
      for (int kk = 0; kk < BK; ++kk) {
        const float xv = to_f32(xr[kk]);
        const bf16* wr = sw + kk;
        for (int t = 0; t < JPT; ++t) acc[t] += xv * to_f32(wr[(j0 + t) * BK]);
      }
    }
    __syncthreads();
  }

  if (!row_ok) return;
  const long ob = (long)(m0 + i) * N + n0 + j0;
  for (int t = 0; t < JPT; ++t) out[ob + t] = to_bf16(acc[t]);
}

int set_smem(const void* fn, int bytes) {
  return err(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

}  // namespace

extern "C" {

// tile 三元组的块数由 host 算好（tiles 长度）；这里只校验参数并派发。
int mxfp4_gemm_grouped(const void* xg, const void* wp, const void* ws, const int32_t* expert_off,
                       const int32_t* tile_e, const int32_t* tile_m, const int32_t* tile_n, void* out,
                       int64_t num_tiles, int64_t K, int64_t N, int64_t group, cudaStream_t stream) {
  if (num_tiles <= 0) return 0;
  if (K <= 0 || N <= 0 || group <= 0 || K % group != 0 || K % 2 != 0) return err(cudaErrorInvalidValue);
  constexpr int BM = 16, BN = 64, BK = 64;
  if (N % BN != 0 || K % BK != 0) return err(cudaErrorInvalidValue);
  const int smem = (int)((BN * BK + BM * BK) * sizeof(bf16));
  if (const int rc = set_smem(reinterpret_cast<const void*>(&gemm_tile_kernel<BM, BN, BK>), smem)) return rc;
  gemm_tile_kernel<BM, BN, BK><<<(unsigned)num_tiles, 256, smem, stream>>>(
      static_cast<const bf16*>(xg), static_cast<const uint8_t*>(wp), static_cast<const uint8_t*>(ws), expert_off,
      tile_e, tile_m, tile_n, static_cast<bf16*>(out), (int)K, (int)N, (int)group);
  return last_err();
}

}  // extern "C"
