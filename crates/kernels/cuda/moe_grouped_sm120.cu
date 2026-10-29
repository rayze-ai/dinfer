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
#include "rmsnorm_core.cuh"

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
// B8：一维 tile 表。tile_off[e] = Σ_{e'<e} ceil(len_e'/BM)（moe_bucket 的 scan 产出）；tile t ∈ [0, tile_off[E]) 属于
// 使 tile_off[e] ≤ t < tile_off[e+1] 的专家 e，行块 m_blk = offsets[e] + (t − tile_off[e])·BM。二分 E=128 只 7 步。
// grid.y 取 host 上界 R/BM + E，超过 tile_off[E] 的 block 直接退（比旧版 (E × R/BM) 满格 grid 少 >100×）。
__device__ __forceinline__ bool tile_lookup(const int32_t* __restrict__ offsets, const int32_t* __restrict__ tile_off,
                                            int E, int t, int& e, int& m_blk, int& row_hi, int bm = BM) {
  if (t >= tile_off[E]) return false;
  int lo = 0, hi = E;  // 找最大的 e 使 tile_off[e] ≤ t
  while (hi - lo > 1) {
    const int mid = (lo + hi) >> 1;
    if (tile_off[mid] <= t) lo = mid; else hi = mid;
  }
  e = lo;
  m_blk = offsets[e] + (t - tile_off[e]) * bm;
  row_hi = offsets[e + 1];
  return true;
}


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
// B11：`sorted_tok` 非空时源行 = A[sorted_tok[r]]（gather 融进量化，去掉 xg 中间缓冲与一次 launch）；空则 A[r]。
__global__ void grouped_quant_kernel(const bf16* __restrict__ A, const int32_t* __restrict__ row2e,
                                     const int32_t* __restrict__ sorted_tok, const float* __restrict__ agscale,
                                     uint8_t* __restrict__ P, uint8_t* __restrict__ S, int R, int K, int exact_rq) {
  const int nb = K / 16;
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)R * nb) return;
  const int r = (int)(idx / nb), bi = (int)(idx % nb);
  const float gs = agscale[row2e[r]];
  const int src_r = sorted_tok ? sorted_tok[r] : r;
  uint2 pk;
  uint8_t sb8;
  quant16(A + (size_t)src_r * K + bi * 16, true, &pk, &sb8, gs, exact_rq);
  *reinterpret_cast<uint2*>(P + (size_t)r * (K / 2) + bi * 8) = pk;
  S[(size_t)r * nb + bi] = sb8;
}

// B34③（loop kaon3-decode-opt D175）：gelu·mul + down 的 grouped_quant 一次 launch。一个线程一个 16 元素块（同 grouped_quant_kernel），
// 先按 `gelu_tanh_mul_kernel` mode 1 的逐元素式算 y = bf16(bf16(gelu_tanh(g)) · u)（舍入两次，与 vLLM act_and_mul 同），再 `quant16`。
// 与 `gelu_tanh_mul_bf16(mode 1) → moe_grouped_quant` 两步逐位同（`moe_gelu_quant_fused_parity`）。省 gelu_o 一写一读 + 1 launch/层。
__global__ void gelu_grouped_quant_kernel(const bf16* __restrict__ G, const bf16* __restrict__ U,
                                          const int32_t* __restrict__ row2e, const float* __restrict__ agscale,
                                          uint8_t* __restrict__ P, uint8_t* __restrict__ S, int R, int K, int exact_rq) {
  const int nb = K / 16;
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)R * nb) return;
  const int r = (int)(idx / nb), bi = (int)(idx % nb);
  const float gs = agscale[row2e[r]];
  const bf16* g = G + (size_t)r * K + bi * 16;
  const bf16* u = U + (size_t)r * K + bi * 16;
  bf16 y[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float f = to_f32(g[i]);
    const float x_cube = f * f * f;
    const float inner = 0.7978845608028654f * (f + 0.044715f * x_cube);
    const float a = to_f32(to_bf16(0.5f * f * (1.f + tanhf(inner))));
    y[i] = to_bf16(a * to_f32(u[i]));
  }
  uint2 pk;
  uint8_t sb8;
  quant16(y, true, &pk, &sb8, gs, exact_rq);
  *reinterpret_cast<uint2*>(P + (size_t)r * (K / 2) + bi * 8) = pk;
  S[(size_t)r * nb + bi] = sb8;
}

// B44（split-K，D190）：gelu_grouped_quant 的 split-K 消费版：gate = bf16((Σ_s Pg[s])·alphaG[e])、up 同；其余与
// gelu_grouped_quant_kernel 同式。P 布局 [2·ksplit][Rtot][K]（前 ksplit 份 gate，后 ksplit 份 up）。
__global__ void gelu_grouped_quant_splitk_kernel(const float* __restrict__ P, int ksplit, int Rtot,
                                                 const float* __restrict__ alphaG, const float* __restrict__ alphaU,
                                                 const int32_t* __restrict__ row2e, const float* __restrict__ agscale,
                                                 uint8_t* __restrict__ Pq, uint8_t* __restrict__ S, int R, int K, int exact_rq) {
  const int nb = K / 16;
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)R * nb) return;
  const int r = (int)(idx / nb), bi = (int)(idx % nb);
  const int ex = row2e[r];
  const float gs = agscale[ex];
  const float ag = alphaG[ex], au = alphaU[ex];
  const size_t plane = static_cast<size_t>(Rtot) * K;
  const float* pg = P + static_cast<size_t>(r) * K + bi * 16;
  const float* pu = P + static_cast<size_t>(ksplit) * plane + static_cast<size_t>(r) * K + bi * 16;
  bf16 y[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    float sg = 0.f, su = 0.f;
    for (int sp = 0; sp < ksplit; ++sp) { sg += __ldcg(pg + sp * plane + i); su += __ldcg(pu + sp * plane + i); }
    const float f = to_f32(to_bf16(sg * ag));
    const float u = to_f32(to_bf16(su * au));
    const float x_cube = f * f * f;
    const float inner = 0.7978845608028654f * (f + 0.044715f * x_cube);
    const float a = to_f32(to_bf16(0.5f * f * (1.f + tanhf(inner))));
    y[i] = to_bf16(a * u);
  }
  uint2 pk;
  uint8_t sb8;
  quant16(y, true, &pk, &sb8, gs, exact_rq);
  *reinterpret_cast<uint2*>(Pq + (size_t)r * (K / 2) + bi * 8) = pk;
  S[(size_t)r * nb + bi] = sb8;
}

// B34①（loop kaon3-decode-opt D160）：pre_ff_norm_2 + grouped_quant 一次 launch。一 block 一行 r（源 token = sorted_tok[r]），
// 256 线程：① 复刻 `rmsnorm_vec_kernel<256,2>`（hidden ≤ 4096）的平方和（线程 t 持向量 t、t+256；gs 同序；`rms_ss_triton<256,2>`）
// 与 `rms_inv(mode)`；② 每线程若干 16 元素块：y = bf16(norm_mid(x·inv)·w)（与 vec kernel 逐元素同式）→ `quant16`（与
// grouped_quant 同）。结果与 `rmsnorm_bf16 → moe_grouped_quant` 两步逐位同（`moe_norm_quant_fused_parity`）。省 h2 一写一读。
__global__ void __launch_bounds__(256) rmsnorm_grouped_quant_kernel(const bf16* __restrict__ X, const bf16* __restrict__ W,
                                                                   const int32_t* __restrict__ row2e,
                                                                   const int32_t* __restrict__ sorted_tok,
                                                                   const float* __restrict__ agscale, uint8_t* __restrict__ P,
                                                                   uint8_t* __restrict__ S, int R, int K, float eps, int mode,
                                                                   int exact_rq) {
  constexpr int BLOCK = 256, VPT = 2;
  const int r = blockIdx.x;
  if (r >= R) return;
  const int src_r = sorted_tok ? sorted_tok[r] : r;
  const bf16* x = X + (size_t)src_r * K;
  const uint4* xr = reinterpret_cast<const uint4*>(x);
  const int nvec = K / 8;
  float gs[VPT];
#pragma unroll
  for (int j = 0; j < VPT; ++j) {
    const int i = threadIdx.x + j * BLOCK;
    gs[j] = 0.f;
    if (i < nvec) {
      const uint4 xa = xr[i];
      const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xa);
      float v[8];
#pragma unroll
      for (int c = 0; c < 4; ++c) { const float2 f = __bfloat1622float2(xp[c]); v[2 * c] = f.x; v[2 * c + 1] = f.y; }
      float acc = __fmul_rn(v[1], v[1]);
      acc = __fmaf_rn(v[0], v[0], acc);
#pragma unroll
      for (int e = 2; e < 8; ++e) acc = __fmaf_rn(v[e], v[e], acc);
      gs[j] = acc;
    }
  }
  const float inv = rms_inv(rms_ss_triton<BLOCK, VPT>(gs, nvec), K, eps, mode);
  const float gsc = agscale[row2e[r]];
  const int nb = K / 16;
  for (int bi = threadIdx.x; bi < nb; bi += BLOCK) {
    bf16 y[16];
    const bf16* xs = x + bi * 16;
    const bf16* ws = W + bi * 16;
#pragma unroll
    for (int e = 0; e < 16; e += 2) {
      const __nv_bfloat162 xx = *reinterpret_cast<const __nv_bfloat162*>(xs + e);
      const __nv_bfloat162 ww = *reinterpret_cast<const __nv_bfloat162*>(ws + e);
      const float2 fx = __bfloat1622float2(xx), fw = __bfloat1622float2(ww);
      const __nv_bfloat162 o = __floats2bfloat162_rn(norm_mid(fx.x * inv, mode) * fw.x, norm_mid(fx.y * inv, mode) * fw.y);
      y[e] = o.x; y[e + 1] = o.y;
    }
    uint2 pk;
    uint8_t sb8;
    quant16(y, true, &pk, &sb8, gsc, exact_rq);
    *reinterpret_cast<uint2*>(P + (size_t)r * (K / 2) + bi * 8) = pk;
    S[(size_t)r * nb + bi] = sb8;
  }
}

// grouped prequant GEMM：A 已量化 [R,K/2]+块缩放 [R,K/16]（swz=0）。
// grid=(N/BN, ceil(R/BM), E)；blockIdx.z=e → 行段 [off[e], off[e+1])；m_blk=off[e]+by*BM，越界 return。
// B[e]=packed[e] [N,K/2]、S[e]=scale[e] [N,K/16]、global=alpha[e]。out D[R,N]。
template <int BK>
__global__ void __launch_bounds__(256)
    grouped_gemm_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Asc,
                        const uint8_t* const* __restrict__ Bptr, const uint8_t* const* __restrict__ Sptr,
                        const float* __restrict__ alpha, const int32_t* __restrict__ offsets, const int32_t* __restrict__ tile_off, bf16* __restrict__ D,
                        int N, int K, int E) {
  int e, m_blk, row_hi;
  if (!tile_lookup(offsets, tile_off, E, (int)blockIdx.y, e, m_blk, row_hi)) return;

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

// 流水版（第一刀，R71/R72/R73）：cp.async.cg 双缓冲——B/sfb 异步预取下一 K 块与当前块 mma 重叠。
// 严格照搬已验证的 gemm_fp8_sm120.cu 双缓冲范式（prologue 先存 buf0；循环内 store_a(nxt) 在 wait+sync 之后，
//   compute 前不再写本轮 cur buf）。A/sfa 越界 fill 非 0（0x11/0x38），cp.async zfill 给不了 → 寄存器暂存 + STS。
// 逐位硬门槛：落 smem 字节 / fill / mma 读法 / 归约序与 grouped_gemm_kernel 逐字相同（R73c 单缓冲已验 bad=0）。
template <int BK>
__global__ void __launch_bounds__(256)
    grouped_gemm_kernel_pipe(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Asc,
                             const uint8_t* const* __restrict__ Bptr, const uint8_t* const* __restrict__ Sptr,
                             const float* __restrict__ alpha, const int32_t* __restrict__ offsets, const int32_t* __restrict__ tile_off,
                             bf16* __restrict__ D, int N, int K, int E) {
  int e, m_blk, row_hi;
  if (!tile_lookup(offsets, tile_off, E, (int)blockIdx.y, e, m_blk, row_hi)) return;

  constexpr int RSA = BK / 2 + PAD, RSB = BK / 2 + PAD;
  constexpr int NQ4 = BK / 64;
  constexpr int NB16 = BK / 16;
  __shared__ uint8_t sA[2][BM * RSA];
  __shared__ uint8_t sB[2][BN * RSB];
  __shared__ uint32_t sfa[2][BM * NQ4];
  __shared__ uint32_t sfb[2][BN * NQ4];

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
  constexpr int NA = (BM * BK / 2 / 16 + 255) / 256;
  constexpr int NFA = (BM * NQ4 + 255) / 256;

  // B/sfb → buffer st（cp.async，越界 zfill 0），提交 1 个 group。
  auto load_b = [&](int st, int kb) {
    const uint8_t* bsrc = B + static_cast<size_t>(n_blk) * (K / 2) + kb * (BK / 2);
    for (int i = tid; i < (BN * BK / 2) / 16; i += blockDim.x) {
      const int r = i / rb, c = i % rb;
      const bool ok = (n_blk + r < N);
      cp16(smem_u32(&sB[st][r * RSB + c * 16]), bsrc + (static_cast<size_t>(r) * (K / 32) + c) * 16, ok ? 16 : 0);
    }
    for (int i = tid; i < BN * NQ4; i += blockDim.x) {
      const int r = i / NQ4, q = i % NQ4;
      const bool ok = (n_blk + r < N);
      cp4(smem_u32(&sfb[st][r * NQ4 + q]), S + static_cast<size_t>(n_blk + r) * (K / 16) + kb * NB16 + q * 4,
          ok ? 4 : 0);
    }
    asm volatile("cp.async.commit_group;\n");
  };
  // A/sfa：global → 寄存器（本 block 每线程 NA/NFA 个，这里恒 1）。
  auto conv_a = [&](int kb, uint4 (&ar)[NA], uint32_t (&fr)[NFA]) {
    const uint8_t* asrc = A + static_cast<size_t>(m_blk) * (K / 2) + kb * (BK / 2);
#pragma unroll
    for (int t = 0; t < NA; ++t) {
      const int i = tid + t * 256;
      if (i < (BM * BK / 2) / 16) {
        const int r = i / rb, c = i % rb;
        const uint4 z = make_uint4(0x11111111u, 0x11111111u, 0x11111111u, 0x11111111u);
        ar[t] = (m_blk + r < M)
                    ? *reinterpret_cast<const uint4*>(asrc + (static_cast<size_t>(r) * (K / 32) + c) * 16)
                    : z;
      }
    }
#pragma unroll
    for (int t = 0; t < NFA; ++t) {
      const int i = tid + t * 256;
      if (i < BM * NQ4) {
        const int r = i / NQ4, q = i % NQ4;
        fr[t] = (m_blk + r < M)
                    ? *reinterpret_cast<const uint32_t*>(Asc + static_cast<size_t>(m_blk + r) * (K / 16) +
                                                         kb * NB16 + q * 4)
                    : 0x38383838u;
      }
    }
  };
  // 寄存器 → buffer st（STS）。
  auto store_a = [&](int st, const uint4 (&ar)[NA], const uint32_t (&fr)[NFA]) {
#pragma unroll
    for (int t = 0; t < NA; ++t) {
      const int i = tid + t * 256;
      if (i < (BM * BK / 2) / 16) {
        const int r = i / rb, c = i % rb;
        *reinterpret_cast<uint4*>(&sA[st][r * RSA + c * 16]) = ar[t];
      }
    }
#pragma unroll
    for (int t = 0; t < NFA; ++t) {
      const int i = tid + t * 256;
      if (i < BM * NQ4) {
        const int r = i / NQ4, q = i % NQ4;
        sfa[st][r * NQ4 + q] = fr[t];
      }
    }
  };

  // prologue：buf0 的 B(cp.async) + A/sfa(STS)。A 在循环首个 __syncthreads 后对 mma 可见。
  {
    uint4 ar[NA];
    uint32_t fr[NFA];
    load_b(0, 0);
    conv_a(0, ar, fr);
    store_a(0, ar, fr);
  }
  for (int kb = 0; kb < nk; ++kb) {
    const int cur = kb & 1, nxt = cur ^ 1;
    uint4 anext[NA];
    uint32_t fanext[NFA];
    if (kb + 1 < nk) {
      load_b(nxt, kb + 1);  // 预取下一块 B/sfb（异步）
      conv_a(kb + 1, anext, fanext);  // 下一块 A/sfa 读到寄存器
    } else {
      asm volatile("cp.async.commit_group;\n");  // 占位 group，保持 wait_group 计数一致
    }
    asm volatile("cp.async.wait_group 1;\n");  // 至多 1 group 在飞 → buf cur 的 B/sfb 已就绪
    __syncthreads();
    // buf nxt 两轮前被读完（上轮 trailing sync 保证），本轮 STS 下一块 A/sfa 进 nxt（与 compute cur 不冲突）
    if (kb + 1 < nk) store_a(nxt, anext, fanext);

    const int nbk = BK / 64;
    for (int b = 0; b < nbk; ++b) {
      const int kb0 = b * 64;
      uint32_t a[4], b0[2], b1[2];
      const int mrow = wm * 16, ncol = wn * 16;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const int m = (lane / 4) + 8 * (g % 2);
        const int kb_ = kb0 + 8 * (lane % 4) + 32 * (g / 2);
        a[g] = *reinterpret_cast<const uint32_t*>(&sA[cur][(mrow + m) * RSA + kb_ / 2]);
      }
      const int brow = lane / 4;
#pragma unroll
      for (int g = 0; g < 2; ++g) {
        const int kb_ = kb0 + 8 * (lane % 4) + 32 * g;
        b0[g] = *reinterpret_cast<const uint32_t*>(&sB[cur][(ncol + brow) * RSB + kb_ / 2]);
        b1[g] = *reinterpret_cast<const uint32_t*>(&sB[cur][(ncol + 8 + brow) * RSB + kb_ / 2]);
      }
      const int sam = (8 * (lane & 1) + (lane >> 2)) & 15;
      const uint32_t sa = sfa[cur][(mrow + sam) * NQ4 + b];
      const uint32_t sb0 = sfb[cur][(ncol + brow) * NQ4 + b];
      const uint32_t sb1 = sfb[cur][(ncol + 8 + brow) * NQ4 + b];
      mma_fp4(a, b0, sa, sb0, acc);
      mma_fp4(a, b1, sa, sb1, acc2);
    }
    __syncthreads();  // trailing：下一轮 store_a(nxt)/mma 前，本轮对 cur 的读已全部完成
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

// B16（loop kaon3-decode-opt）：STAGES 级 cp.async 环形流水（默认 4）。pipe 版是 2 级双缓冲 + 每 BK 步一次
// "等上一组落地 → barrier"，decode 每 CTA 每步只搬 ~4 KB，22 步 × (HBM 往返 + barrier) 是延迟绑定（0.85 us/步）。
// 这里 A/sfa 也改 cp.async（越界行 zfill 0：e2m1 0 × 任何 = 0，且这些行的输出不写；与旧版 0x11/0x38 填充同样不影响有效行），
// 预取深度 STAGES-1 组在飞，wait_group STAGES-1。mma 序、片段寻址与 pipe 版逐字同 → 有效行逐位一致。
// B16b：BM_T=16（decode）时 8 warp 排 1(M)×8(N)，每 warp 一个 n8 片段（只 acc）；A 只 16 行、smem/寄存器减半、
// 占用翻倍，M 侧不再为空行算 mma。每个输出元素的 k 累加序与 BM_T=32 版同 → 逐位。tile 表用 tile_off 的第二半（BM=16）。
// B41（loop kaon3-decode-opt D183）：BN_T 模板化（默认 BN=64）。decode 可用 BN_T=32 + 4 warp（blockDim 128）：CTA 数翻倍、
// 每 CTA smem 减半 → 每 SM 常驻 CTA 翻倍、在飞字节翻倍；每个输出元素的 k 累加序不变 → 逐位同。
template <int BK, int STAGES, int BM_T = BM, int BN_T = BN>
__global__ void __launch_bounds__(256)
    grouped_gemm_kernel_pipe4(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Asc,
                              const uint8_t* const* __restrict__ Bptr, const uint8_t* const* __restrict__ Sptr,
                              const float* __restrict__ alpha, const int32_t* __restrict__ offsets,
                              const int32_t* __restrict__ tile_off, bf16* __restrict__ D, int N, int K, int E,
                              // B36（gate‖up dual，grid.z=2）：blockIdx.z==1 的 block 改用第二套（up）指针。grid.z=1 时不触碰。
                              const uint8_t* const* __restrict__ Bu = nullptr, const uint8_t* const* __restrict__ Su = nullptr,
                              const float* __restrict__ alphaU = nullptr, bf16* __restrict__ Du = nullptr,
                              // B44（split-K，D190）：ksplit > 1 时 grid.z = half·ksplit + split，各 split 只算自己的 K 段，
                              // 把**未乘 alpha 的 fp32 部分和**写到 P[(half·ksplit+split)][Rtot][N]（不写 D）；归约与 alpha、bf16
                              // 舍入由消费 kernel（gelu_grouped_quant_splitk）做。ksplit == 1 时行为与原版完全同。
                              int ksplit = 1, float* __restrict__ P = nullptr, int Rtot = 0) {
  int e, m_blk, row_hi;
  if (!tile_lookup(offsets, BM_T == BM ? tile_off : tile_off + (E + 1), E, (int)blockIdx.y, e, m_blk, row_hi, BM_T)) return;
  const int half = ksplit > 1 ? (int)blockIdx.z / ksplit : (int)blockIdx.z;
  const int split = ksplit > 1 ? (int)blockIdx.z % ksplit : 0;
  if (half) { Bptr = Bu; Sptr = Su; alpha = alphaU; D = Du; }
  constexpr int RSA = BK / 2 + PAD, RSB = BK / 2 + PAD;
  constexpr int NQ4 = BK / 64;
  constexpr int NB16 = BK / 16;
  __shared__ __align__(16) uint8_t sA[STAGES][BM_T * RSA];
  __shared__ __align__(16) uint8_t sB[STAGES][BN_T * RSB];
  __shared__ __align__(16) uint32_t sfa[STAGES][BM_T * NQ4];
  __shared__ __align__(16) uint32_t sfb[STAGES][BN_T * NQ4];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  constexpr bool ONE_FRAG = BM_T == 16;              // 1×8 warp 布局：每 warp 一个 n8 片段
  const int wm = ONE_FRAG ? 0 : (warp >> 2), wn = ONE_FRAG ? warp : (warp & 3);
  const int n_blk = blockIdx.x * BN_T;
  static_assert(BN_T == BN || (BM_T == 16 && BN_T % 8 == 0), "BN_T 只在 BM16 布局下可变");
  const uint8_t* B = Bptr[e];
  const uint8_t* S = Sptr[e];
  const float global = alpha[e];
  const int M = row_hi;
  float acc[4] = {0, 0, 0, 0}, acc2[4] = {0, 0, 0, 0};
  const int nk_total = (K + BK - 1) / BK;
  const int nk_per = (nk_total + ksplit - 1) / ksplit;
  const int kb_base = split * nk_per;
  const int nk = kb_base < nk_total ? ((nk_total - kb_base) < nk_per ? (nk_total - kb_base) : nk_per) : 0;  // 本 split 的 K 块数
  const int rb = BK / 32;
  // 第 kb 块 → stage st：A/sfa/B/sfb 全部 cp.async，提交 1 组
  auto load_stage = [&](int st, int kb) {
    const uint8_t* asrc = A + static_cast<size_t>(m_blk) * (K / 2) + kb * (BK / 2);
    for (int i = tid; i < (BM_T * BK / 2) / 16; i += blockDim.x) {
      const int r = i / rb, c = i % rb;
      const bool ok = (m_blk + r < M) && (kb * BK + c * 32 < K);  // B40：K 尾块越界零填充
      cp16(smem_u32(&sA[st][r * RSA + c * 16]), asrc + (static_cast<size_t>(r) * (K / 32) + c) * 16, ok ? 16 : 0);
    }
    for (int i = tid; i < BM_T * NQ4; i += blockDim.x) {
      const int r = i / NQ4, q = i % NQ4;
      const bool ok = (m_blk + r < M) && (kb * NB16 + q * 4 < K / 16);
      cp4(smem_u32(&sfa[st][r * NQ4 + q]), Asc + static_cast<size_t>(m_blk + r) * (K / 16) + kb * NB16 + q * 4,
          ok ? 4 : 0);
    }
    const uint8_t* bsrc = B + static_cast<size_t>(n_blk) * (K / 2) + kb * (BK / 2);
    for (int i = tid; i < (BN_T * BK / 2) / 16; i += blockDim.x) {
      const int r = i / rb, c = i % rb;
      const bool ok = (n_blk + r < N) && (kb * BK + c * 32 < K);
      cp16(smem_u32(&sB[st][r * RSB + c * 16]), bsrc + (static_cast<size_t>(r) * (K / 32) + c) * 16, ok ? 16 : 0);
    }
    for (int i = tid; i < BN_T * NQ4; i += blockDim.x) {
      const int r = i / NQ4, q = i % NQ4;
      const bool ok = (n_blk + r < N) && (kb * NB16 + q * 4 < K / 16);
      cp4(smem_u32(&sfb[st][r * NQ4 + q]), S + static_cast<size_t>(n_blk + r) * (K / 16) + kb * NB16 + q * 4,
          ok ? 4 : 0);
    }
    asm volatile("cp.async.commit_group;\n");
  };
  // prologue：前 STAGES-1 块
#pragma unroll
  for (int st = 0; st < STAGES - 1; ++st) {
    if (st < nk) load_stage(st, kb_base + st);
    else asm volatile("cp.async.commit_group;\n");
  }
  for (int kb = 0; kb < nk; ++kb) {
    // 发出第 kb+STAGES-1 块（其 stage 在上轮 trailing barrier 后已无读者）
    {
      const int kn = kb + STAGES - 1;
      if (kn < nk) load_stage(kn % STAGES, kb_base + kn);
      else asm volatile("cp.async.commit_group;\n");
    }
    asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGES - 1));  // 第 kb 组已落地
    __syncthreads();
    const int cur = kb % STAGES;
    const int nbk = BK / 64;
    for (int b = 0; b < nbk; ++b) {
      const int kb0 = b * 64;
      uint32_t a[4], b0[2], b1[2];
      const int mrow = wm * 16, ncol = ONE_FRAG ? wn * 8 : wn * 16;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const int m = (lane / 4) + 8 * (g % 2);
        const int kb_ = kb0 + 8 * (lane % 4) + 32 * (g / 2);
        a[g] = *reinterpret_cast<const uint32_t*>(&sA[cur][(mrow + m) * RSA + kb_ / 2]);
      }
      const int brow = lane / 4;
#pragma unroll
      for (int g = 0; g < 2; ++g) {
        const int kb_ = kb0 + 8 * (lane % 4) + 32 * g;
        b0[g] = *reinterpret_cast<const uint32_t*>(&sB[cur][(ncol + brow) * RSB + kb_ / 2]);
        if (!ONE_FRAG) b1[g] = *reinterpret_cast<const uint32_t*>(&sB[cur][(ncol + 8 + brow) * RSB + kb_ / 2]);
      }
      const int sam = (8 * (lane & 1) + (lane >> 2)) & 15;
      const uint32_t sa = sfa[cur][(mrow + sam) * NQ4 + b];
      const uint32_t sb0 = sfb[cur][(ncol + brow) * NQ4 + b];
      mma_fp4(a, b0, sa, sb0, acc);
      if (!ONE_FRAG) {
        const uint32_t sb1 = sfb[cur][(ncol + 8 + brow) * NQ4 + b];
        mma_fp4(a, b1, sa, sb1, acc2);
      }
    }
    __syncthreads();  // trailing：本 stage 的读全部完成，下一轮可覆盖
  }
  const int r0 = lane / 4, cg = (lane % 4) * 2;
  const int g0 = m_blk + wm * 16 + r0, g1 = g0 + 8;
  const int c0 = n_blk + (ONE_FRAG ? wn * 8 : wn * 16) + cg, c1 = c0 + 8;
  if (P) {  // split-K：fp32 部分和，不乘 alpha
    float* Pp = P + (static_cast<size_t>(half * ksplit + split) * Rtot) * N;
    if (g0 < M && c0 < N) { Pp[static_cast<size_t>(g0) * N + c0] = acc[0]; if (c0 + 1 < N) Pp[static_cast<size_t>(g0) * N + c0 + 1] = acc[1]; }
    if (g1 < M && c0 < N) { Pp[static_cast<size_t>(g1) * N + c0] = acc[2]; if (c0 + 1 < N) Pp[static_cast<size_t>(g1) * N + c0 + 1] = acc[3]; }
    if (!ONE_FRAG) {
      if (g0 < M && c1 < N) { Pp[static_cast<size_t>(g0) * N + c1] = acc2[0]; if (c1 + 1 < N) Pp[static_cast<size_t>(g0) * N + c1 + 1] = acc2[1]; }
      if (g1 < M && c1 < N) { Pp[static_cast<size_t>(g1) * N + c1] = acc2[2]; if (c1 + 1 < N) Pp[static_cast<size_t>(g1) * N + c1 + 1] = acc2[3]; }
    }
    return;
  }
  if (g0 < M && c0 < N) {
    D[static_cast<size_t>(g0) * N + c0] = to_bf16(acc[0] * global);
    if (c0 + 1 < N) D[static_cast<size_t>(g0) * N + c0 + 1] = to_bf16(acc[1] * global);
  }
  if (g1 < M && c0 < N) {
    D[static_cast<size_t>(g1) * N + c0] = to_bf16(acc[2] * global);
    if (c0 + 1 < N) D[static_cast<size_t>(g1) * N + c0 + 1] = to_bf16(acc[3] * global);
  }
  if (!ONE_FRAG) {
    if (g0 < M && c1 < N) {
      D[static_cast<size_t>(g0) * N + c1] = to_bf16(acc2[0] * global);
      if (c1 + 1 < N) D[static_cast<size_t>(g0) * N + c1 + 1] = to_bf16(acc2[1] * global);
    }
    if (g1 < M && c1 < N) {
      D[static_cast<size_t>(g1) * N + c1] = to_bf16(acc2[2] * global);
      if (c1 + 1 < N) D[static_cast<size_t>(g1) * N + c1 + 1] = to_bf16(acc2[3] * global);
    }
  }
}
// 第二刀（R89+）：gate+up 融合单 launch。grid.z = 2E：z<E 跑 gate（Bg/Sg/alphaG/Dg），z>=E 跑 up（Bu/Su/alphaU/Du）。
// decode grid 小（~8激活×N/BN）占不满 170 SM；翻倍 grid.z 填满 SM = graph 吃不掉的并行度。
// body 与 grouped_gemm_kernel_pipe 逐字相同（仅 e/B/S/global/D 按 half 选）→ 逐位天然一致。
template <int BK>
__global__ void __launch_bounds__(256)
    grouped_gemm_kernel_dual(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Asc,
                             const uint8_t* const* __restrict__ Bg, const uint8_t* const* __restrict__ Sg,
                             const float* __restrict__ alphaG, bf16* __restrict__ Dg,
                             const uint8_t* const* __restrict__ Bu, const uint8_t* const* __restrict__ Su,
                             const float* __restrict__ alphaU, bf16* __restrict__ Du,
                             const int32_t* __restrict__ offsets, const int32_t* __restrict__ tile_off, int N, int K, int E) {
  const int half = blockIdx.z;              // 0=gate 1=up
  int e, m_blk, row_hi;
  if (!tile_lookup(offsets, tile_off, E, (int)blockIdx.y, e, m_blk, row_hi)) return;
  const uint8_t* const* Bptr = half ? Bu : Bg;
  const uint8_t* const* Sptr = half ? Su : Sg;
  const float* alpha = half ? alphaU : alphaG;
  bf16* D = half ? Du : Dg;

  constexpr int RSA = BK / 2 + PAD, RSB = BK / 2 + PAD;
  constexpr int NQ4 = BK / 64;
  constexpr int NB16 = BK / 16;
  __shared__ uint8_t sA[2][BM * RSA];
  __shared__ uint8_t sB[2][BN * RSB];
  __shared__ uint32_t sfa[2][BM * NQ4];
  __shared__ uint32_t sfb[2][BN * NQ4];

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
  constexpr int NA = (BM * BK / 2 / 16 + 255) / 256;
  constexpr int NFA = (BM * NQ4 + 255) / 256;

  auto load_b = [&](int st, int kb) {
    const uint8_t* bsrc = B + static_cast<size_t>(n_blk) * (K / 2) + kb * (BK / 2);
    for (int i = tid; i < (BN * BK / 2) / 16; i += blockDim.x) {
      const int r = i / rb, c = i % rb;
      const bool ok = (n_blk + r < N);
      cp16(smem_u32(&sB[st][r * RSB + c * 16]), bsrc + (static_cast<size_t>(r) * (K / 32) + c) * 16, ok ? 16 : 0);
    }
    for (int i = tid; i < BN * NQ4; i += blockDim.x) {
      const int r = i / NQ4, q = i % NQ4;
      const bool ok = (n_blk + r < N);
      cp4(smem_u32(&sfb[st][r * NQ4 + q]), S + static_cast<size_t>(n_blk + r) * (K / 16) + kb * NB16 + q * 4,
          ok ? 4 : 0);
    }
    asm volatile("cp.async.commit_group;\n");
  };
  auto conv_a = [&](int kb, uint4 (&ar)[NA], uint32_t (&fr)[NFA]) {
    const uint8_t* asrc = A + static_cast<size_t>(m_blk) * (K / 2) + kb * (BK / 2);
#pragma unroll
    for (int t = 0; t < NA; ++t) {
      const int i = tid + t * 256;
      if (i < (BM * BK / 2) / 16) {
        const int r = i / rb, c = i % rb;
        const uint4 z = make_uint4(0x11111111u, 0x11111111u, 0x11111111u, 0x11111111u);
        ar[t] = (m_blk + r < M)
                    ? *reinterpret_cast<const uint4*>(asrc + (static_cast<size_t>(r) * (K / 32) + c) * 16)
                    : z;
      }
    }
#pragma unroll
    for (int t = 0; t < NFA; ++t) {
      const int i = tid + t * 256;
      if (i < BM * NQ4) {
        const int r = i / NQ4, q = i % NQ4;
        fr[t] = (m_blk + r < M)
                    ? *reinterpret_cast<const uint32_t*>(Asc + static_cast<size_t>(m_blk + r) * (K / 16) +
                                                         kb * NB16 + q * 4)
                    : 0x38383838u;
      }
    }
  };
  auto store_a = [&](int st, const uint4 (&ar)[NA], const uint32_t (&fr)[NFA]) {
#pragma unroll
    for (int t = 0; t < NA; ++t) {
      const int i = tid + t * 256;
      if (i < (BM * BK / 2) / 16) {
        const int r = i / rb, c = i % rb;
        *reinterpret_cast<uint4*>(&sA[st][r * RSA + c * 16]) = ar[t];
      }
    }
#pragma unroll
    for (int t = 0; t < NFA; ++t) {
      const int i = tid + t * 256;
      if (i < BM * NQ4) {
        const int r = i / NQ4, q = i % NQ4;
        sfa[st][r * NQ4 + q] = fr[t];
      }
    }
  };

  {
    uint4 ar[NA];
    uint32_t fr[NFA];
    load_b(0, 0);
    conv_a(0, ar, fr);
    store_a(0, ar, fr);
  }
  for (int kb = 0; kb < nk; ++kb) {
    const int cur = kb & 1, nxt = cur ^ 1;
    uint4 anext[NA];
    uint32_t fanext[NFA];
    if (kb + 1 < nk) {
      load_b(nxt, kb + 1);
      conv_a(kb + 1, anext, fanext);
    } else {
      asm volatile("cp.async.commit_group;\n");
    }
    asm volatile("cp.async.wait_group 1;\n");
    __syncthreads();
    if (kb + 1 < nk) store_a(nxt, anext, fanext);

    const int nbk = BK / 64;
    for (int b = 0; b < nbk; ++b) {
      const int kb0 = b * 64;
      uint32_t a[4], b0[2], b1[2];
      const int mrow = wm * 16, ncol = wn * 16;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const int m = (lane / 4) + 8 * (g % 2);
        const int kb_ = kb0 + 8 * (lane % 4) + 32 * (g / 2);
        a[g] = *reinterpret_cast<const uint32_t*>(&sA[cur][(mrow + m) * RSA + kb_ / 2]);
      }
      const int brow = lane / 4;
#pragma unroll
      for (int g = 0; g < 2; ++g) {
        const int kb_ = kb0 + 8 * (lane % 4) + 32 * g;
        b0[g] = *reinterpret_cast<const uint32_t*>(&sB[cur][(ncol + brow) * RSB + kb_ / 2]);
        b1[g] = *reinterpret_cast<const uint32_t*>(&sB[cur][(ncol + 8 + brow) * RSB + kb_ / 2]);
      }
      const int sam = (8 * (lane & 1) + (lane >> 2)) & 15;
      const uint32_t sa = sfa[cur][(mrow + sam) * NQ4 + b];
      const uint32_t sb0 = sfb[cur][(ncol + brow) * NQ4 + b];
      const uint32_t sb1 = sfb[cur][(ncol + 8 + brow) * NQ4 + b];
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
// B40：decode（R ≤ 128）pipe4 的 K 块大小。默认 256；`MOE_GEMM_BK=64|128` 回退（A/B 用）。
// B41：decode pipe4 的 N 块宽（仅 BK=256 路径）。默认 64；`MOE_GEMM_BN=32` → 4 warp CTA，CTA 数翻倍（A/B 用，赢了再改默认）。
static int dec_bn() {
  static const int v = []{ const char* e = getenv("MOE_GEMM_BN"); const int n = e ? atoi(e) : 64; return n == 32 ? 32 : 64; }();
  return v;
}
// D197：decode 快路径（BM16 / BK256 / STAGES3，dual pipe4）的行数上限。默认 128；`MOE_DEC_RMAX=512` 让 c16–c64（R ≤ 512）也走它
// （M=32 激活 ~100 专家、每专家 2–3 行，BM=32 旧体 tile 93% 空：nsys M=32 dual<128> 108 us/层）。
static int dec_rmax() {
  // D198（2026-10-29 b48 两轮）：默认 512（c32 TPOT -14% / 吞吐 +10%，c64 -6%，c16 持平；R=264 与旧体逐位同）。`MOE_DEC_RMAX=128` 回旧默认。
  static const int v = []{ const char* e = getenv("MOE_DEC_RMAX"); const int n = e ? atoi(e) : 512; return (n >= 16 && n <= 4096) ? n : 512; }();
  return v;
}
static int dec_bk() {
  static const int v = []{ const char* e = getenv("MOE_GEMM_BK"); const int n = e ? atoi(e) : 256; return (n == 64 || n == 128) ? n : 256; }();
  return v;
}

}  // namespace

extern "C" {

// offsets[E+1] (device) → row2e[R] (device)。R = offsets[E]（调用方传 E，不读 offsets 值）。
int moe_row2expert(const int32_t* offsets, int32_t* row2e, int64_t num_experts, cudaStream_t stream) {
  if (num_experts <= 0) return 0;
  row2expert_kernel<<<(unsigned)num_experts, 128, 0, stream>>>(offsets, row2e);
  return gerr();
}

// grouped A 预量化：A[R,K] bf16 → aq[R,K/2] + a_s[R,K/16]，每行 gs=agscale[row2e[r]]。
int moe_grouped_quant(const void* a, const int32_t* row2e, const int32_t* sorted_tok, const float* agscale, void* aq,
                      void* a_s, int64_t R, int64_t K, int exact_rq, cudaStream_t stream) {
  if (R <= 0) return 0;
  if (K % 64 != 0) return static_cast<int>(cudaErrorInvalidValue);
  const long blocks = (long)R * (K / 16);
  const unsigned g = (unsigned)((blocks + 255) / 256);
  grouped_quant_kernel<<<g, 256, 0, stream>>>(static_cast<const bf16*>(a), row2e, sorted_tok, agscale,
                                              static_cast<uint8_t*>(aq), static_cast<uint8_t*>(a_s), (int)R, (int)K,
                                              exact_rq);
  return gerr();
}

// grouped GEMM：out[R,N] = alpha[e] * aq · Bptr[e]ᵀ（按专家段）。Bptr/Sptr 是 device 端 [E] 指针数组。
// grid.y = ceil(R/BM)（R 是各专家段上界的最大可能，单专家最多 R 行）。cudagraph 友好：launch 配置编译期固定。
// grid = (N/BN, tile 上界 = R/BM + E, 1)：tile 表（B8）代替满格 (R/BM × E)。编译期固定、cudagraph 友好。
int moe_grouped_gemm(const void* aq, const void* a_s, const void* const* Bptr, const void* const* Sptr,
                     const float* alpha, const int32_t* offsets, const int32_t* tile_off, void* out, int64_t R,
                     int64_t N, int64_t K, int64_t num_experts, cudaStream_t stream) {
  if (R <= 0 || N <= 0 || K <= 0 || num_experts <= 0) return 0;
  if (K % 64 != 0) return static_cast<int>(cudaErrorInvalidValue);
  const dim3 grid((unsigned)((N + BN - 1) / BN), (unsigned)((R + BM - 1) / BM + num_experts), 1u);
  const uint8_t* pa = static_cast<const uint8_t*>(aq);
  const uint8_t* psa = static_cast<const uint8_t*>(a_s);
  const uint8_t* const* pb = reinterpret_cast<const uint8_t* const*>(Bptr);
  const uint8_t* const* ps = reinterpret_cast<const uint8_t* const*>(Sptr);
  // cp.async 流水版默认开（D48 实测 decode TPOT -5.5%/吞吐+5.7%、逐位一致）；MOE_GEMM_PIPE=0 关（回退旧版）。
  static const bool pipe = []{ const char* v = getenv("MOE_GEMM_PIPE"); return !v || v[0] != '0'; }();  // B17a：一次读
  // B16：MOE_GEMM_STAGES（默认 4）= STAGES 级 cp.async 环形流水；=2 回退 pipe 双缓冲版（A/B 用）。
  static const int stages = []{ const char* v = getenv("MOE_GEMM_STAGES"); const int n = v ? atoi(v) : 4; return (n == 3 || n == 4 || n == 6) ? n : (n <= 2 ? 2 : 4); }();
  if (pipe && stages > 2) {
    // B16b：decode（R ≤ 128）用 BM=16 tile（tile 表第二半），grid.y 上界 R/16 + E
    static const bool bm16 = []{ const char* v = getenv("MOE_GEMM_BM16"); return !v || v[0] != '0'; }();
    if (bm16 && R <= dec_rmax()) {
      const dim3 g16((unsigned)((N + BN - 1) / BN), (unsigned)((R + 15) / 16 + num_experts), 1u);
      // B40（D181）：decode 的 pipe4 是串行 K 链瓶颈（22 迭代 × ~0.7 us），BK=256 把迭代减半；K 尾块零填充（K % 64 == 0 即可）。
      // 累加序仍 k 升序 → 与 BK=64/128 逐位同。`MOE_GEMM_BK=64|128|256`，默认 256。
      if (dec_bk() == 256 && dec_bn() == 32) {
        const dim3 g32((unsigned)((N + 31) / 32), g16.y, 1u);
        grouped_gemm_kernel_pipe4<256, 3, 16, 32><<<g32, 128, 0, stream>>>(pa, psa, pb, ps, alpha, offsets, tile_off, static_cast<bf16*>(out), (int)N, (int)K, (int)num_experts);
      }
      else if (dec_bk() == 256) grouped_gemm_kernel_pipe4<256, 3, 16><<<g16, 256, 0, stream>>>(pa, psa, pb, ps, alpha, offsets, tile_off, static_cast<bf16*>(out), (int)N, (int)K, (int)num_experts);
      else if (dec_bk() == 128 && K % 128 == 0) grouped_gemm_kernel_pipe4<128, 4, 16><<<g16, 256, 0, stream>>>(pa, psa, pb, ps, alpha, offsets, tile_off, static_cast<bf16*>(out), (int)N, (int)K, (int)num_experts);
      else grouped_gemm_kernel_pipe4<64, 4, 16><<<g16, 256, 0, stream>>>(pa, psa, pb, ps, alpha, offsets, tile_off, static_cast<bf16*>(out), (int)N, (int)K, (int)num_experts);
      return gerr();
    }
    if (K % 128 == 0) {
      if (stages == 3) grouped_gemm_kernel_pipe4<128, 3><<<grid, 256, 0, stream>>>(pa, psa, pb, ps, alpha, offsets, tile_off, static_cast<bf16*>(out), (int)N, (int)K, (int)num_experts);
      // STAGES=6 × BK=128 静态 smem 50.7 KB > 48 KB 上限 → BK=128 最多 4 级
      else grouped_gemm_kernel_pipe4<128, 4><<<grid, 256, 0, stream>>>(pa, psa, pb, ps, alpha, offsets, tile_off, static_cast<bf16*>(out), (int)N, (int)K, (int)num_experts);
    } else {
      if (stages == 3) grouped_gemm_kernel_pipe4<64, 3><<<grid, 256, 0, stream>>>(pa, psa, pb, ps, alpha, offsets, tile_off, static_cast<bf16*>(out), (int)N, (int)K, (int)num_experts);
      else if (stages == 6) grouped_gemm_kernel_pipe4<64, 6><<<grid, 256, 0, stream>>>(pa, psa, pb, ps, alpha, offsets, tile_off, static_cast<bf16*>(out), (int)N, (int)K, (int)num_experts);
      else grouped_gemm_kernel_pipe4<64, 4><<<grid, 256, 0, stream>>>(pa, psa, pb, ps, alpha, offsets, tile_off, static_cast<bf16*>(out), (int)N, (int)K, (int)num_experts);
    }
    return gerr();
  }
  if (K % 128 == 0) {
    if (pipe)
      grouped_gemm_kernel_pipe<128><<<grid, 256, 0, stream>>>(pa, psa, pb, ps, alpha, offsets, tile_off,
                                                              static_cast<bf16*>(out), (int)N, (int)K, (int)num_experts);
    else
      grouped_gemm_kernel<128><<<grid, 256, 0, stream>>>(pa, psa, pb, ps, alpha, offsets, tile_off, static_cast<bf16*>(out),
                                                         (int)N, (int)K, (int)num_experts);
  } else {
    if (pipe)
      grouped_gemm_kernel_pipe<64><<<grid, 256, 0, stream>>>(pa, psa, pb, ps, alpha, offsets, tile_off,
                                                             static_cast<bf16*>(out), (int)N, (int)K, (int)num_experts);
    else
      grouped_gemm_kernel<64><<<grid, 256, 0, stream>>>(pa, psa, pb, ps, alpha, offsets, tile_off, static_cast<bf16*>(out),
                                                        (int)N, (int)K, (int)num_experts);
  }
  return gerr();
}

// 第二刀：gate+up 融合单 launch。grid.z=2E（前 E gate、后 E up）。两路共用 aq/a_s/offsets，权重/scale/alpha/out 各一套。
// K%128==0 用 pipe body(BK=128)。逐位与分开两次 moe_grouped_gemm 一致（body 逐字同 pipe）。
int moe_grouped_gemm_dual(const void* aq, const void* a_s,
                          const void* const* Bg, const void* const* Sg, const float* alphaG, void* outG,
                          const void* const* Bu, const void* const* Su, const float* alphaU, void* outU,
                          const int32_t* offsets, const int32_t* tile_off, int64_t R, int64_t N, int64_t K,
                          int64_t num_experts, cudaStream_t stream) {
  if (R <= 0 || N <= 0 || K <= 0 || num_experts <= 0) return 0;
  if (K % 64 != 0) return static_cast<int>(cudaErrorInvalidValue);
  const dim3 grid((unsigned)((N + BN - 1) / BN), (unsigned)((R + BM - 1) / BM + num_experts), 2u);
  const uint8_t* pa = static_cast<const uint8_t*>(aq);
  const uint8_t* psa = static_cast<const uint8_t*>(a_s);
  const uint8_t* const* pbg = reinterpret_cast<const uint8_t* const*>(Bg);
  const uint8_t* const* psg = reinterpret_cast<const uint8_t* const*>(Sg);
  const uint8_t* const* pbu = reinterpret_cast<const uint8_t* const*>(Bu);
  const uint8_t* const* psu = reinterpret_cast<const uint8_t* const*>(Su);
  // B36：decode（R ≤ 128）走 pipe4 BM=16 的 dual（grid.z=2；每个输出元素 k 累加序与单路 pipe4<BM16> 同 → 逐位）。
  // 旧 dual body 是 BM=32 的 pipe 体，decode 时丢了 B16b 的占用收益。`MOE_DUAL_PIPE4=0` 回退旧 body。
  static const bool dual_pipe4 = []{ const char* v = getenv("MOE_DUAL_PIPE4"); return !v || v[0] != '0'; }();
  if (dual_pipe4 && R <= dec_rmax()) {
    const dim3 g16((unsigned)((N + BN - 1) / BN), (unsigned)((R + 15) / 16 + num_experts), 2u);
    if (dec_bk() == 256 && dec_bn() == 32) {
      const dim3 g32((unsigned)((N + 31) / 32), g16.y, 2u);
      grouped_gemm_kernel_pipe4<256, 3, 16, 32><<<g32, 128, 0, stream>>>(pa, psa, pbg, psg, alphaG, offsets, tile_off, static_cast<bf16*>(outG), (int)N, (int)K, (int)num_experts, pbu, psu, alphaU, static_cast<bf16*>(outU));
    }
    else if (dec_bk() == 256)
      grouped_gemm_kernel_pipe4<256, 3, 16><<<g16, 256, 0, stream>>>(pa, psa, pbg, psg, alphaG, offsets, tile_off, static_cast<bf16*>(outG), (int)N, (int)K, (int)num_experts, pbu, psu, alphaU, static_cast<bf16*>(outU));
    else if (dec_bk() == 128 && K % 128 == 0)
      grouped_gemm_kernel_pipe4<128, 4, 16><<<g16, 256, 0, stream>>>(pa, psa, pbg, psg, alphaG, offsets, tile_off, static_cast<bf16*>(outG), (int)N, (int)K, (int)num_experts, pbu, psu, alphaU, static_cast<bf16*>(outU));
    else
      grouped_gemm_kernel_pipe4<64, 4, 16><<<g16, 256, 0, stream>>>(pa, psa, pbg, psg, alphaG, offsets, tile_off, static_cast<bf16*>(outG), (int)N, (int)K, (int)num_experts, pbu, psu, alphaU, static_cast<bf16*>(outU));
    return gerr();
  }
  if (K % 128 == 0) {
    grouped_gemm_kernel_dual<128><<<grid, 256, 0, stream>>>(pa, psa, pbg, psg, alphaG, static_cast<bf16*>(outG),
                                                            pbu, psu, alphaU, static_cast<bf16*>(outU), offsets, tile_off,
                                                            (int)N, (int)K, (int)num_experts);
  } else {
    grouped_gemm_kernel_dual<64><<<grid, 256, 0, stream>>>(pa, psa, pbg, psg, alphaG, static_cast<bf16*>(outG),
                                                           pbu, psu, alphaU, static_cast<bf16*>(outU), offsets, tile_off,
                                                           (int)N, (int)K, (int)num_experts);
  }
  return gerr();
}


// B34①：见 rmsnorm_grouped_quant_kernel。K % 64 == 0 且 K ≤ 4096（vec<256,2> 口径）。
extern "C" int moe_rmsnorm_grouped_quant(const void* x, const void* w, const int32_t* row2e, const int32_t* sorted_tok,
                                         const float* agscale, void* aq, void* a_s, int64_t rows, int64_t k, float eps,
                                         int mode, int exact_rq, cudaStream_t stream) {
  if (rows <= 0) return 0;
  if (k <= 0 || k % 64 != 0 || k > 4096) return static_cast<int>(cudaErrorInvalidValue);
  rmsnorm_grouped_quant_kernel<<<(unsigned)rows, 256, 0, stream>>>(static_cast<const bf16*>(x), static_cast<const bf16*>(w),
                                                                   row2e, sorted_tok, agscale, static_cast<uint8_t*>(aq),
                                                                   static_cast<uint8_t*>(a_s), (int)rows, (int)k, eps, mode, exact_rq);
  return gerr();
}

// B34③：见 gelu_grouped_quant_kernel（gelu mode 1 口径）。
extern "C" int moe_gelu_grouped_quant(const void* gate, const void* up, const int32_t* row2e, const float* agscale, void* aq,
                                      void* a_s, int64_t R, int64_t K, int exact_rq, cudaStream_t stream) {
  if (R <= 0) return 0;
  if (K % 64 != 0) return static_cast<int>(cudaErrorInvalidValue);
  const long blocks = (long)R * (K / 16);
  const unsigned g = (unsigned)((blocks + 255) / 256);
  gelu_grouped_quant_kernel<<<g, 256, 0, stream>>>(static_cast<const bf16*>(gate), static_cast<const bf16*>(up), row2e,
                                                   agscale, static_cast<uint8_t*>(aq), static_cast<uint8_t*>(a_s),
                                                   (int)R, (int)K, exact_rq);
  return gerr();
}

// B44（split-K，D190）：decode（R ≤ 128）gate‖up 的 split-K 版 dual：grid.z = 2·ksplit，输出 fp32 部分和 P[2·ksplit][R][N]（不乘 alpha）。
// 只走 BK=256/BM16 路径（ksplit·256 ≤ K 才有意义；K=2816 → ksplit ≤ 11）。
extern "C" int moe_grouped_gemm_dual_splitk(const void* aq, const void* a_s,
                                            const void* const* Bg, const void* const* Sg, const float* alphaG,
                                            const void* const* Bu, const void* const* Su, const float* alphaU,
                                            const int32_t* offsets, const int32_t* tile_off, float* P,
                                            int64_t R, int64_t N, int64_t K, int64_t num_experts, int ksplit, cudaStream_t stream) {
  if (R <= 0 || N <= 0 || K <= 0 || num_experts <= 0) return 0;
  if (K % 64 != 0 || R > 128 || ksplit < 1 || ksplit > 8) return static_cast<int>(cudaErrorInvalidValue);
  const uint8_t* pa = static_cast<const uint8_t*>(aq);
  const uint8_t* psa = static_cast<const uint8_t*>(a_s);
  const uint8_t* const* pbg = reinterpret_cast<const uint8_t* const*>(Bg);
  const uint8_t* const* psg = reinterpret_cast<const uint8_t* const*>(Sg);
  const uint8_t* const* pbu = reinterpret_cast<const uint8_t* const*>(Bu);
  const uint8_t* const* psu = reinterpret_cast<const uint8_t* const*>(Su);
  const dim3 g((unsigned)((N + BN - 1) / BN), (unsigned)((R + 15) / 16 + num_experts), (unsigned)(2 * ksplit));
  grouped_gemm_kernel_pipe4<256, 3, 16><<<g, 256, 0, stream>>>(pa, psa, pbg, psg, alphaG, offsets, tile_off, nullptr, (int)N, (int)K, (int)num_experts,
                                                              pbu, psu, alphaU, nullptr, ksplit, P, (int)R);
  return gerr();
}
extern "C" int moe_gelu_grouped_quant_splitk(const float* P, int ksplit, const float* alphaG, const float* alphaU,
                                             const int32_t* row2e, const float* agscale, void* aq, void* a_s,
                                             int64_t R, int64_t K, int exact_rq, cudaStream_t stream) {
  if (R <= 0) return 0;
  if (K % 64 != 0 || ksplit < 1) return static_cast<int>(cudaErrorInvalidValue);
  const long blocks = (long)R * (K / 16);
  const unsigned g = (unsigned)((blocks + 255) / 256);
  gelu_grouped_quant_splitk_kernel<<<g, 256, 0, stream>>>(P, ksplit, (int)R, alphaG, alphaU, row2e, agscale,
                                                          static_cast<uint8_t*>(aq), static_cast<uint8_t*>(a_s), (int)R, (int)K, exact_rq);
  return gerr();
}

}  // extern "C"
