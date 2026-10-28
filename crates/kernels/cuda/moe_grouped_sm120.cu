// MoE device-grouped NVFP4 GEMM（sm_120 专用，B3：去 host 路由 + 去 D2H）。
// 现状（host 版 models/kaon3/src/layer.rs moe_branch）：D2H 读 offsets → host 循环逐专家 launch gate/up/down。
// 本文件：offsets 留 device，一个 kernel（grid.z=专家）按 device 端 ranges 跑变长专家的 grouped GEMM，
//   launch 次数编译期固定（cudagraph 前置，B4）。
//
// 逐位一致（硬底线）：GEMM 逐行、K 固定 → 与单专家 gemm_nvfp4_sm120.cu 同 quant16 / 同 mma 归约序。
//   quant16 / mma_fp4 / e4m3_f / rcp_approx **逐字复制自 gemm_nvfp4_sm120.cu**（不共享头避免跨文件漂移；
//   任一侧改了另一侧要同步）。BM/BN/PAD 同。
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include "common.cuh"

using namespace dinfer;

namespace {

constexpr int BM = 32, BN = 64, PAD = 16;

// ---- 以下 4 个 device 函数逐字复制自 gemm_nvfp4_sm120.cu（保逐位）----
__device__ __forceinline__ float e4m3_f(unsigned b) {
  const unsigned s = (b >> 7) & 1u, e = (b >> 3) & 15u, m = b & 7u;
  float v;
  if (e == 0) {
    v = m * (1.f / 8.f) * exp2f(-6.f);
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

__device__ __forceinline__ float rcp_approx(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
  return b;
}

__device__ __forceinline__ void quant16(const bf16* src, bool valid, uint2* out8, uint8_t* out_s, float gs, int exact_rq) {
  float x[16];
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    x[i] = valid ? to_f32(src[i]) : 0.f;
    amax = fmaxf(amax, fabsf(x[i]));
  }
  uint8_t sb8 = 0;
  float inv = 0.f;
  if (amax > 0.f) {
    const float sfv = gs * (exact_rq ? __fdiv_rn(amax, 6.f) : (amax * rcp_approx(6.f)));
    sb8 = __nv_cvt_float_to_fp8(sfv, __NV_SATFINITE, __NV_E4M3);
    const float sfd = e4m3_f(sb8);
    inv = sfd != 0.f ? (exact_rq ? __fdiv_rn(gs, sfd) : rcp_approx(sfd * rcp_approx(gs))) : 0.f;
  }
  uint32_t w[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    uint32_t v = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
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
// ---- 复制结束 ----

// cp.async 流水 helper（照搬 gemm_fp8_sm120.cu 的 cp.async.cg 双缓冲范式）。
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// 16 B 异步拷贝（cg）：bytes=0 → zero-fill 不访问 global（越界行 fill 0，与旧版 ternary 填 0 一致）。
__device__ __forceinline__ void cp16(uint32_t s, const void* g, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(g), "r"(bytes));
}
// 4 B 异步拷贝（ca）：sfb 每元素 4 B；bytes=0 → zero-fill（越界 sfb fill 0，一致）。
__device__ __forceinline__ void cp4(uint32_t s, const void* g, int bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s), "l"(g), "r"(bytes));
}

// row2expert[off[e]..off[e+1]] = e。grid=E，block e 填自己的 range（device offsets，无 D2H）。
__global__ void row2expert_kernel(const int32_t* __restrict__ offsets, int32_t* __restrict__ row2e) {
  const int e = blockIdx.x;
  const int s = offsets[e], t = offsets[e + 1];
  for (int r = s + threadIdx.x; r < t; r += blockDim.x) row2e[r] = e;
}

// grouped 预量化：A[R,K] bf16 → packed[R,K/2] + 块缩放 [R,K/16]（自然布局 swz=0）。
// 每行用自己专家的 gs = agscale[row2e[r]]。一个 16-元素块一个线程（同 quantize_a_kernel）。
__global__ void grouped_quant_kernel(const bf16* __restrict__ A, const int32_t* __restrict__ row2e,
                                     const float* __restrict__ agscale, uint8_t* __restrict__ P,
                                     uint8_t* __restrict__ S, int R, int K, int exact_rq) {
  const int nb = K / 16;
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)R * nb) return;
  const int r = (int)(idx / nb), bi = (int)(idx % nb);
  const float gs = agscale[row2e[r]];
  uint2 pk;
  uint8_t sb8;
  quant16(A + (size_t)r * K + bi * 16, true, &pk, &sb8, gs, exact_rq);
  *reinterpret_cast<uint2*>(P + (size_t)r * (K / 2) + bi * 8) = pk;
  S[(size_t)r * nb + bi] = sb8;
}

// grouped prequant GEMM：A 已量化 [R,K/2]+块缩放 [R,K/16]（swz=0）。
// grid=(N/BN, ceil(R/BM), E)；blockIdx.z=e → 行段 [off[e], off[e+1])；m_blk=off[e]+by*BM，越界 return。
// B[e]=packed[e] [N,K/2]、S[e]=scale[e] [N,K/16]、global=alpha[e]。out D[R,N]。
template <int BK>
__global__ void __launch_bounds__(256)
    grouped_gemm_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Asc,
                        const uint8_t* const* __restrict__ Bptr, const uint8_t* const* __restrict__ Sptr,
                        const float* __restrict__ alpha, const int32_t* __restrict__ offsets, bf16* __restrict__ D,
                        int N, int K) {
  const int e = blockIdx.z;
  const int row_lo = offsets[e], row_hi = offsets[e + 1];
  const int m_blk = row_lo + blockIdx.y * BM;
  if (m_blk >= row_hi) return;  // 该专家没有这一行块（含 len==0 专家：row_lo==row_hi 直接退）

  constexpr int RSA = BK / 2 + PAD, RSB = BK / 2 + PAD;
  constexpr int NQ4 = BK / 64;
  constexpr int NB16 = BK / 16;
  __shared__ uint8_t sA[BM * RSA];
  __shared__ uint8_t sB[BN * RSB];
  __shared__ uint32_t sfa[BM * NQ4];
  __shared__ uint32_t sfb[BN * NQ4];

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 2, wn = warp & 3;
  const int n_blk = blockIdx.x * BN;
  const uint8_t* B = Bptr[e];
  const uint8_t* S = Sptr[e];
  const float global = alpha[e];
  // 行上界用专家段尾（row_hi），不是全局 R：段外行不载入（置 0 nibble）。
  const int M = row_hi;

  float acc[4] = {0, 0, 0, 0}, acc2[4] = {0, 0, 0, 0};
  const int nk = (K + BK - 1) / BK;
  const int rb = BK / 32;

  for (int kb = 0; kb < nk; ++kb) {
    {
      const uint4* sa_src = reinterpret_cast<const uint4*>(A + static_cast<size_t>(m_blk) * (K / 2) + kb * (BK / 2));
      for (int i = tid; i < (BM * BK / 2) / 16; i += blockDim.x) {
        const int r = i / rb, c = i % rb;
        const uint4 z = make_uint4(0x11, 0x11, 0x11, 0x11);
        *reinterpret_cast<uint4*>(&sA[r * RSA + c * 16]) =
            (m_blk + r < M) ? sa_src[static_cast<size_t>(r) * (K / 32) + c] : z;
      }
      for (int i = tid; i < BM * NQ4; i += blockDim.x) {
        const int r = i / NQ4, q = i % NQ4;
        sfa[r * NQ4 + q] =
            (m_blk + r < M)
                ? *reinterpret_cast<const uint32_t*>(Asc + static_cast<size_t>(m_blk + r) * (K / 16) + kb * NB16 + q * 4)
                : 0x38383838u;
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

// 流水版（第一刀，R71/R72/R73）：单缓冲 cp.async 版——结构与 grouped_gemm_kernel 逐行一致，
// 只把 B / sfb 的 global→smem 从「同步 STS」换成「cp.async + commit + wait_group 0」。
// A/sfa 原样 STS。先验证 cp.async 索引逐位对（wait 0 无重叠，不提速），过了再加双缓冲流水。
// 逐位硬门槛：落 smem 字节 / fill / mma 读法 / 归约序与旧版逐字相同。
template <int BK>
__global__ void __launch_bounds__(256)
    grouped_gemm_kernel_pipe(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Asc,
                             const uint8_t* const* __restrict__ Bptr, const uint8_t* const* __restrict__ Sptr,
                             const float* __restrict__ alpha, const int32_t* __restrict__ offsets,
                             bf16* __restrict__ D, int N, int K) {
  const int e = blockIdx.z;
  const int row_lo = offsets[e], row_hi = offsets[e + 1];
  const int m_blk = row_lo + blockIdx.y * BM;
  if (m_blk >= row_hi) return;

  constexpr int RSA = BK / 2 + PAD, RSB = BK / 2 + PAD;
  constexpr int NQ4 = BK / 64;
  constexpr int NB16 = BK / 16;
  __shared__ uint8_t sA[BM * RSA];
  __shared__ uint8_t sB[BN * RSB];
  __shared__ uint32_t sfa[BM * NQ4];
  __shared__ uint32_t sfb[BN * NQ4];

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 2, wn = warp & 3;
  const int n_blk = blockIdx.x * BN;
  const uint8_t* B = Bptr[e];
  const uint8_t* S = Sptr[e];
  const float global = alpha[e];
  const int M = row_hi;

  float acc[4] = {0, 0, 0, 0}, acc2[4] = {0, 0, 0, 0};
  const int nk = (K + BK - 1) / BK;
  const int rb = BK / 32;

  for (int kb = 0; kb < nk; ++kb) {
    // A：原样 STS（越界 fill 0x11 nibble）。
    {
      const uint4* sa_src = reinterpret_cast<const uint4*>(A + static_cast<size_t>(m_blk) * (K / 2) + kb * (BK / 2));
      for (int i = tid; i < (BM * BK / 2) / 16; i += blockDim.x) {
        const int r = i / rb, c = i % rb;
        const uint4 z = make_uint4(0x11111111u, 0x11111111u, 0x11111111u, 0x11111111u);
        *reinterpret_cast<uint4*>(&sA[r * RSA + c * 16]) =
            (m_blk + r < M) ? sa_src[static_cast<size_t>(r) * (K / 32) + c] : z;
      }
      for (int i = tid; i < BM * NQ4; i += blockDim.x) {
        const int r = i / NQ4, q = i % NQ4;
        sfa[r * NQ4 + q] =
            (m_blk + r < M)
                ? *reinterpret_cast<const uint32_t*>(Asc + static_cast<size_t>(m_blk + r) * (K / 16) + kb * NB16 + q * 4)
                : 0x38383838u;
      }
    }
    // B：cp.async 预取（越界 zfill 0，与旧版 ternary 填 0 一致）。
    {
      const uint8_t* bsrc = B + static_cast<size_t>(n_blk) * (K / 2) + kb * (BK / 2);
      for (int i = tid; i < (BN * BK / 2) / 16; i += blockDim.x) {
        const int r = i / rb, c = i % rb;
        const bool ok = (n_blk + r < N);
        cp16(smem_u32(&sB[r * RSB + c * 16]), bsrc + (static_cast<size_t>(r) * (K / 32) + c) * 16, ok ? 16 : 0);
      }
    }
    // sfb：cp.async 预取（越界 zfill 0）。
    {
      for (int i = tid; i < BN * NQ4; i += blockDim.x) {
        const int r = i / NQ4, q = i % NQ4;
        const bool ok = (n_blk + r < N);
        cp4(smem_u32(&sfb[r * NQ4 + q]), S + static_cast<size_t>(n_blk + r) * (K / 16) + kb * NB16 + q * 4,
            ok ? 4 : 0);
      }
    }
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");
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

int gerr() { return static_cast<int>(cudaGetLastError()); }

}  // namespace

extern "C" {

// offsets[E+1] (device) → row2e[R] (device)。R = offsets[E]（调用方传 E，不读 offsets 值）。
int moe_row2expert(const int32_t* offsets, int32_t* row2e, int64_t num_experts, cudaStream_t stream) {
  if (num_experts <= 0) return 0;
  row2expert_kernel<<<(unsigned)num_experts, 128, 0, stream>>>(offsets, row2e);
  return gerr();
}

// grouped A 预量化：A[R,K] bf16 → aq[R,K/2] + a_s[R,K/16]，每行 gs=agscale[row2e[r]]。
int moe_grouped_quant(const void* a, const int32_t* row2e, const float* agscale, void* aq, void* a_s, int64_t R,
                      int64_t K, int exact_rq, cudaStream_t stream) {
  if (R <= 0) return 0;
  if (K % 64 != 0) return static_cast<int>(cudaErrorInvalidValue);
  const long blocks = (long)R * (K / 16);
  const unsigned g = (unsigned)((blocks + 255) / 256);
  grouped_quant_kernel<<<g, 256, 0, stream>>>(static_cast<const bf16*>(a), row2e, agscale,
                                              static_cast<uint8_t*>(aq), static_cast<uint8_t*>(a_s), (int)R, (int)K,
                                              exact_rq);
  return gerr();
}

// grouped GEMM：out[R,N] = alpha[e] * aq · Bptr[e]ᵀ（按专家段）。Bptr/Sptr 是 device 端 [E] 指针数组。
// grid.y = ceil(R/BM)（R 是各专家段上界的最大可能，单专家最多 R 行）。cudagraph 友好：launch 配置编译期固定。
int moe_grouped_gemm(const void* aq, const void* a_s, const void* const* Bptr, const void* const* Sptr,
                     const float* alpha, const int32_t* offsets, void* out, int64_t R, int64_t N, int64_t K,
                     int64_t num_experts, cudaStream_t stream) {
  if (R <= 0 || N <= 0 || K <= 0 || num_experts <= 0) return 0;
  if (K % 64 != 0) return static_cast<int>(cudaErrorInvalidValue);
  // 诊断(MOE_GRID_CAP=<n>)：临时把 grid.z 从 num_experts 压到 n，测空 block 占 gemm3 多少时间。
  // **结果会错**(只覆盖前 n 个专家),仅用于计时验证"94% 空 block 是否真耗时",测完删。
  int gz = (int)num_experts;
  if (const char* cap = getenv("MOE_GRID_CAP")) { int c = atoi(cap); if (c > 0 && c < gz) gz = c; }
  const dim3 grid((unsigned)((N + BN - 1) / BN), (unsigned)((R + BM - 1) / BM), (unsigned)gz);
  const uint8_t* pa = static_cast<const uint8_t*>(aq);
  const uint8_t* psa = static_cast<const uint8_t*>(a_s);
  const uint8_t* const* pb = reinterpret_cast<const uint8_t* const*>(Bptr);
  const uint8_t* const* ps = reinterpret_cast<const uint8_t* const*>(Sptr);
  const bool pipe = []{ const char* v = getenv("MOE_GEMM_PIPE"); return v && v[0] == '1'; }();
  if (K % 128 == 0) {
    if (pipe)
      grouped_gemm_kernel_pipe<128><<<grid, 256, 0, stream>>>(pa, psa, pb, ps, alpha, offsets,
                                                              static_cast<bf16*>(out), (int)N, (int)K);
    else
      grouped_gemm_kernel<128><<<grid, 256, 0, stream>>>(pa, psa, pb, ps, alpha, offsets, static_cast<bf16*>(out),
                                                         (int)N, (int)K);
  } else {
    if (pipe)
      grouped_gemm_kernel_pipe<64><<<grid, 256, 0, stream>>>(pa, psa, pb, ps, alpha, offsets,
                                                             static_cast<bf16*>(out), (int)N, (int)K);
    else
      grouped_gemm_kernel<64><<<grid, 256, 0, stream>>>(pa, psa, pb, ps, alpha, offsets, static_cast<bf16*>(out),
                                                        (int)N, (int)K);
  }
  return gerr();
}

}  // extern "C"
