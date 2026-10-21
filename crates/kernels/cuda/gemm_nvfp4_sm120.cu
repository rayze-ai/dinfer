// NVFP4 块缩放 GEMM（sm_120 专用）：out[m,n] = global * Σ_k a[m,k] · b[n,k]
//   a      : bf16 激活 [m,k]，**载入时按 16 元素一块量化成 fp4**（scale = block amax / 6，e4m3 编码）
//   packed : [n, k/2] 权重，每字节两个 fp4（低半字节是偶数列）
//   scale  : [n, k/16] 的 e4m3 块缩放
//   global : per-tensor 标量，出栈前乘
// 结构照远端探针 v5（models/kaon3/cuda/nvfp4_gemm5.cu，4096³ 实测 502 TF/s）：
//   block 32x64xBK、8 warp = 2(M)x4(N)、每 warp 16x16（2 个 n8 累加器）、smem padding 16 B。
// 与探针的差别（阶段 3 新增，必须测）：
//   1. 探针把 sa/sb 写死 0x38383838（全是 1.0），真实块缩放路径从未验证；这里从 smem 读真 scale。
//   2. 探针 A 也是预量化字节；这里 A 是 bf16，量化融进载入。
// 语义细节见 docs/loops/kaon-v3.md「阶段 1/2」：每组 8 个 nibble 在 A/B 里沿 k 连续，可单条 u32 读。
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include "common.cuh"

using namespace dinfer;  // bf16 / to_f32 / to_bf16（common.cuh 放在 namespace 里）

namespace {

constexpr int BM = 32, BN = 64, PAD = 16;

/// e4m3 字节 → float（与探针 fp8_v2.cu 的 e4m3_f 同一套位运算）。
__device__ __forceinline__ float e4m3_f(unsigned b) {
  const unsigned s = (b >> 7) & 1u, e = (b >> 3) & 15u, m = b & 7u;
  float v;
  if (e == 0) {
    v = m * (1.f / 8.f) * (1.f / 64.f);  // 次正规：m/8 * 2^-6
  } else if (e == 15 && m == 7) {
    v = 448.f;
  } else {
    v = (1.f + m * (1.f / 8.f)) * exp2f((float)((int)e - 7));
  }
  return s ? -v : v;
}

__device__ __forceinline__ void mma_fp4(const uint32_t* a, const uint32_t* b, uint32_t sa, uint32_t sb, float* d) {
  const uint16_t z = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z),
        "h"(z));
}

/// 16 个 bf16 → 8 字节 fp4 + 1 字节 e4m3 块缩放。
/// 量化口径（CPU 参考侧必须逐位复现）：amax = max|x|；scale = amax/6 再编码成 e4m3；
/// 反量化出 deq 后用 1/deq 缩放再做 e2m1 就近舍入。amax == 0 时整块给 0、scale 给 1.0。
__device__ __forceinline__ void quant16(const bf16* src, bool valid, uint2* out8, uint8_t* out_s) {
  float x[16];
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    x[i] = valid ? to_f32(src[i]) : 0.f;
    amax = fmaxf(amax, fabsf(x[i]));
  }
  uint8_t sb8;
  float inv;
  if (amax > 0.f) {
    sb8 = __nv_cvt_float_to_fp8(amax * (1.f / 6.f), __NV_SATFINITE, __NV_E4M3);
    inv = 1.f / e4m3_f(sb8);
  } else {
    sb8 = 0x38u;  // e4m3 的 1.0：让硬件乘 1 后仍得 0，避免 0/0
    inv = 0.f;
  }
  uint32_t w[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    uint32_t v = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      // 一个字节装两个 fp4：低半字节 = 偶数列
      const uint8_t two = __nv_cvt_float2_to_fp4x2(make_float2(x[i * 8 + j * 2] * inv, x[i * 8 + j * 2 + 1] * inv),
                                                   __NV_E2M1, cudaRoundNearest);
      v |= static_cast<uint32_t>(two) << (8 * j);
    }
    w[i] = v;
  }
  out8->x = w[0];
  out8->y = w[1];
  *out_s = sb8;
}

template <int BK>
__global__ void __launch_bounds__(256) gemm_nvfp4_kernel(const bf16* __restrict__ A, const uint8_t* __restrict__ B,
                                                         const uint8_t* __restrict__ S, bf16* __restrict__ D,
                                                         int M, int N, int K, float global) {
  constexpr int RSA = BK / 2 + PAD, RSB = BK / 2 + PAD;
  constexpr int NB16 = BK / 16;  // 每行 16 元素块数
  constexpr int NQ4 = BK / 64;   // 每行 scale 四元组数（= scale_vec::4X 的 u32 个数）
  __shared__ uint8_t sA[BM * RSA];
  __shared__ uint8_t sB[BN * RSB];
  __shared__ uint32_t sfa[BM * NQ4];  // A 块缩放，按 4 个连续 k-block 打成 u32
  __shared__ uint32_t sfb[BN * NQ4];

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 2, wn = warp & 3;
  const int m_blk = blockIdx.y * BM, n_blk = blockIdx.x * BN;
  float acc[4] = {0, 0, 0, 0}, acc2[4] = {0, 0, 0, 0};
  const int nk = (K + BK - 1) / BK;
  const int rb = BK / 32;  // 每行 uint4 数


  for (int kb = 0; kb < nk; ++kb) {
    {
      // A 量化：每块 16 个 bf16 由一个线程处理（BM*NB16 块，块数随 BK 变，故用循环）
      for (int i = tid; i < BM * NB16; i += blockDim.x) {
        const int r = i / NB16, bi = i % NB16;
        const int gm = m_blk + r, gk = kb * BK + bi * 16;
        const bool ok = (gm < M) && (gk + 15 < K);
        uint2 pk;
        uint8_t sb8;
        quant16(A + static_cast<size_t>(gm < M ? gm : M - 1) * K + gk, ok, &pk, &sb8);
        *reinterpret_cast<uint2*>(&sA[r * RSA + bi * 8]) = pk;
        reinterpret_cast<uint8_t*>(sfa)[r * NB16 + bi] = sb8;
      }
    }
    {
      const uint4* src = reinterpret_cast<const uint4*>(B + static_cast<size_t>(n_blk) * (K / 2) + kb * (BK / 2));
      for (int i = tid; i < (BN * BK / 2) / 16; i += blockDim.x) {
        const int r = i / rb, c = i % rb;
        const uint4 z = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(&sB[r * RSB + c * 16]) =
            (n_blk + r < N) ? src[static_cast<size_t>(r) * (K / 32) + c] : z;
      }
    }
    {
      for (int i = tid; i < BN * NQ4; i += blockDim.x) {
        const int r = i / NQ4, q = i % NQ4;
        sfb[r * NQ4 + q] = (n_blk + r < N)
                               ? *reinterpret_cast<const uint32_t*>(S + static_cast<size_t>(n_blk + r) * (K / 16) +
                                                                    kb * NB16 + q * 4)
                               : 0u;
      }
    }
    __syncthreads();

    const int nbk = BK / 64;
    for (int b = 0; b < nbk; ++b) {
      const int kb0 = b * 64;
      uint32_t a[4], b0[2], b1[2];
      const int mrow = wm * 16, ncol = wn * 16;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const int m = (lane / 4) + 8 * (g % 2);
        const int kb_ = kb0 + 8 * (lane % 4) + 32 * (g / 2);
        a[g] = *reinterpret_cast<const uint32_t*>(&sA[(mrow + m) * RSA + kb_ / 2]);
      }
      const int brow = lane / 4;
#pragma unroll
      for (int g = 0; g < 2; ++g) {
        const int kb_ = kb0 + 8 * (lane % 4) + 32 * g;
        b0[g] = *reinterpret_cast<const uint32_t*>(&sB[(ncol + brow) * RSB + kb_ / 2]);
        b1[g] = *reinterpret_cast<const uint32_t*>(&sB[(ncol + 8 + brow) * RSB + kb_ / 2]);
      }
      // scale 的线程→行映射（探针实测，见 loop 文档）：A 看 t%2 选高/低 8 行，B 看 t/4
      const int sam = (8 * (lane & 1) + (lane >> 2)) & 15;
      const uint32_t sa = sfa[(mrow + sam) * NQ4 + b];
      const uint32_t sb0 = sfb[(ncol + brow) * NQ4 + b];
      const uint32_t sb1 = sfb[(ncol + 8 + brow) * NQ4 + b];
      mma_fp4(a, b0, sa, sb0, acc);
      mma_fp4(a, b1, sa, sb1, acc2);
    }
    __syncthreads();
  }

  const int r0 = lane / 4, cg = (lane % 4) * 2;
  const int g0 = m_blk + wm * 16 + r0, g1 = g0 + 8;
  const int c0 = n_blk + wn * 16 + cg, c1 = c0 + 8;
  if (g0 < M && c0 < N) {
    D[static_cast<size_t>(g0) * N + c0] = to_bf16(acc[0] * global);
    if (c0 + 1 < N) D[static_cast<size_t>(g0) * N + c0 + 1] = to_bf16(acc[1] * global);
  }
  if (g1 < M && c0 < N) {
    D[static_cast<size_t>(g1) * N + c0] = to_bf16(acc[2] * global);
    if (c0 + 1 < N) D[static_cast<size_t>(g1) * N + c0 + 1] = to_bf16(acc[3] * global);
  }
  if (g0 < M && c1 < N) {
    D[static_cast<size_t>(g0) * N + c1] = to_bf16(acc2[0] * global);
    if (c1 + 1 < N) D[static_cast<size_t>(g0) * N + c1 + 1] = to_bf16(acc2[1] * global);
  }
  if (g1 < M && c1 < N) {
    D[static_cast<size_t>(g1) * N + c1] = to_bf16(acc2[2] * global);
    if (c1 + 1 < N) D[static_cast<size_t>(g1) * N + c1 + 1] = to_bf16(acc2[3] * global);
  }
}

int gemm_last_err() {
  return static_cast<int>(cudaGetLastError());
}

}  // namespace

extern "C" {

int gemm_nvfp4_bf16(const void* a, const void* packed, const void* scale, float global, void* out, int64_t m, int64_t n,
                    int64_t k, cudaStream_t stream) {
  if (m < 0 || n < 0 || k < 0 || k == 0) return static_cast<int>(cudaErrorInvalidValue);
  if (m == 0 || n == 0) return 0;
  // 块缩放的 k-block 是 16，mma 是 k64：BK 取 128（K%128==0）否则 64（调用方保证 K%64==0）
  const dim3 grid((unsigned)((n + BN - 1) / BN), (unsigned)((m + BM - 1) / BM));
  if (k % 128 == 0) {
    gemm_nvfp4_kernel<128><<<grid, 256, 0, stream>>>(static_cast<const bf16*>(a), static_cast<const uint8_t*>(packed),
                                                     static_cast<const uint8_t*>(scale), static_cast<bf16*>(out),
                                                     (int)m, (int)n, (int)k, global);
  } else if (k % 64 == 0) {
    gemm_nvfp4_kernel<64><<<grid, 256, 0, stream>>>(static_cast<const bf16*>(a), static_cast<const uint8_t*>(packed),
                                                    static_cast<const uint8_t*>(scale), static_cast<bf16*>(out),
                                                    (int)m, (int)n, (int)k, global);
  } else {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return gemm_last_err();
}

}  // extern "C"
