// decode（M ≤ 16）dense MLP 的带宽型 fp8 路径：norm→fp8 / gate‖up GEMV + gelu·mul→fp8 / down GEMV。
//
// 为什么不用 persistent mega（mega_mlp_sm120.cu）：decode 时 M=1，三个 GEMM 都是纯访存；mega 的阶段链
// 1 CTA/SM 自旋等依赖，nsys 实测 86.9 us/层 vs 权重 17.8 MB 的带宽下限 10 us（loop kaon3-decode-opt D60）。
// 这里每个 warp 负责权重的 2 行（gate‖up 同行号 / down 相邻两行，共用一次 A 读），lane 一次 16 B 向量读、
// K 维流式扫完；A（fp8 → bf16 精确展开）整体驻共享内存被 block 内所有 warp 复用；f32 FMA 累加、warp shuffle 归约。
// 数值口径与 mega 一致：fp8 输入、f32 累加、epilogue 乘 (w_scale·a_scale) 后舍 bf16、gelu 先舍 bf16 再乘 up 再舍、
// fp8 用 SATFINITE。累加顺序不同于 mma，非逐位，差在 bf16 ulp 级（Rust 测试 mlp_fp8_decode_parity）。
// 结构常量：kWarps（block 内 warp 数）、M_TILE 模板档位 {1,2,4,8,16}（决定共享内存 = M_TILE·K·2 B）。
// 没有卡型调优数值。所有 arch 共用（fp8 转换用 cuda_fp8.h）。限制：K % 16 == 0。
#include <cuda_fp8.h>

#include "common.cuh"
#include "kernels.h"

namespace dinfer {
namespace {
constexpr int kWarps = 4;
constexpr int kBlock = kWarps * 32;
constexpr int kMaxM = 16;

__device__ __forceinline__ float gelu_tanh_f32(float x) {
  constexpr float k0 = 0.7978845608028654f;
  return 0.5f * x * (1.f + tanhf(k0 * (x + 0.044715f * x * x * x)));
}
__device__ __forceinline__ uint8_t f32_to_e4m3(float v) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}
// 16 个 e4m3（一个 uint4）→ 16 个 f32
__device__ __forceinline__ void e4m3x16_to_f32(uint4 v, float* f) {
  const unsigned short* p = reinterpret_cast<const unsigned short*>(&v);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const __half2 h2 = __half2(__nv_cvt_fp8x2_to_halfraw2(p[i], __NV_E4M3));
    const float2 x = __half22float2(h2);
    f[2 * i] = x.x;
    f[2 * i + 1] = x.y;
  }
}

// A_q [m][K] fp8 → 共享内存 bf16 [M_TILE][K]（e4m3 ⊂ bf16，精确）；行 ≥ m 不读不写。末尾 __syncthreads。
template <int M_TILE>
__device__ __forceinline__ void load_a(const uint8_t* __restrict__ a_q, bf16* sa, int m, int K) {
  const int nvec = K / 16;
  for (int idx = threadIdx.x; idx < M_TILE * nvec; idx += kBlock) {
    const int r = idx / nvec, c = idx % nvec;
    if (r < m) {
      float f[16];
      e4m3x16_to_f32(*reinterpret_cast<const uint4*>(a_q + (long)r * K + c * 16), f);
      uint4 o[2];
      __nv_bfloat162* ob = reinterpret_cast<__nv_bfloat162*>(o);
#pragma unroll
      for (int i = 0; i < 8; ++i) ob[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
      uint4* dst = reinterpret_cast<uint4*>(sa + (long)r * K + c * 16);
      dst[0] = o[0];
      dst[1] = o[1];
    }
  }
  __syncthreads();
}

// 一个 warp：权重两行 w0/w1（各 K 个 fp8）与 A 的 M_TILE 行点积，acc[j][r] 为本 lane 的部分和（未归约）。
template <int M_TILE>
__device__ __forceinline__ void dot2(const bf16* sa, int K, const uint8_t* __restrict__ w0,
                                     const uint8_t* __restrict__ w1, float (&acc)[2][M_TILE], int lane) {
  const int nvec = K / 16;
  for (int c = lane; c < nvec; c += 32) {
    float wf[2][16];
    e4m3x16_to_f32(__ldg(reinterpret_cast<const uint4*>(w0 + c * 16)), wf[0]);
    e4m3x16_to_f32(__ldg(reinterpret_cast<const uint4*>(w1 + c * 16)), wf[1]);
#pragma unroll
    for (int r = 0; r < M_TILE; ++r) {
      const uint4* ap = reinterpret_cast<const uint4*>(sa + (long)r * K + c * 16);
      const uint4 a0 = ap[0], a1 = ap[1];
      const __nv_bfloat162* b0 = reinterpret_cast<const __nv_bfloat162*>(&a0);
      const __nv_bfloat162* b1 = reinterpret_cast<const __nv_bfloat162*>(&a1);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 x = __bfloat1622float2(b0[i]);
        const float2 y = __bfloat1622float2(b1[i]);
        acc[0][r] = fmaf(wf[0][2 * i], x.x, acc[0][r]);
        acc[0][r] = fmaf(wf[0][2 * i + 1], x.y, acc[0][r]);
        acc[0][r] = fmaf(wf[0][8 + 2 * i], y.x, acc[0][r]);
        acc[0][r] = fmaf(wf[0][8 + 2 * i + 1], y.y, acc[0][r]);
        acc[1][r] = fmaf(wf[1][2 * i], x.x, acc[1][r]);
        acc[1][r] = fmaf(wf[1][2 * i + 1], x.y, acc[1][r]);
        acc[1][r] = fmaf(wf[1][8 + 2 * i], y.x, acc[1][r]);
        acc[1][r] = fmaf(wf[1][8 + 2 * i + 1], y.y, acc[1][r]);
      }
    }
  }
#pragma unroll
  for (int j = 0; j < 2; ++j) {
#pragma unroll
    for (int r = 0; r < M_TILE; ++r) {
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) acc[j][r] += __shfl_xor_sync(0xffffffffu, acc[j][r], off);
    }
  }
}

// ---- norm → fp8：q[r][i] = fp8(bf16(x·rsqrt(mean x²+eps)·w) · a_inv)。grid = m，block = 256。
__global__ void __launch_bounds__(256) norm_quant_kernel(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                                         uint8_t* __restrict__ q, int H, float eps, float a_inv) {
  __shared__ float red[8];
  const bf16* xr = x + (long)blockIdx.x * H;
  float ss = 0.f;
  for (int i = threadIdx.x; i < H; i += 256) {
    const float v = to_f32(xr[i]);
    ss = fmaf(v, v, ss);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) tot += red[i];
  const float inv = rsqrtf(tot / (float)H + eps);
  uint8_t* qr = q + (long)blockIdx.x * H;
  for (int i = threadIdx.x; i < H; i += 256) {
    const bf16 nv = to_bf16(to_f32(xr[i]) * inv * to_f32(w[i]));
    qr[i] = f32_to_e4m3(to_f32(nv) * a_inv);
  }
}

// ---- gate‖up GEMV + gelu·mul → fp8。grid = ceil(I / kWarps)，每 warp 一个行号 n（gate 行 n + up 行 n）。
template <int M_TILE>
__global__ void __launch_bounds__(kBlock) gateup_act_kernel(const uint8_t* __restrict__ a_q,
                                                            const uint8_t* __restrict__ gw,
                                                            const uint8_t* __restrict__ uw,
                                                            uint8_t* __restrict__ act_q, int m, int I, int K,
                                                            float gs, float us, float dn_a_inv) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  bf16* sa = reinterpret_cast<bf16*>(smem_raw);
  load_a<M_TILE>(a_q, sa, m, K);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = blockIdx.x * kWarps + warp;
  if (n >= I) return;  // load_a 已过 barrier，之后无 barrier
  float acc[2][M_TILE];
#pragma unroll
  for (int r = 0; r < M_TILE; ++r) acc[0][r] = acc[1][r] = 0.f;
  dot2<M_TILE>(sa, K, gw + (long)n * K, uw + (long)n * K, acc, lane);
#pragma unroll
  for (int r = 0; r < M_TILE; ++r) {
    if (lane == r && r < m) {
      const bf16 g = to_bf16(acc[0][r] * gs);
      const bf16 u = to_bf16(acc[1][r] * us);
      const float y = to_f32(to_bf16(gelu_tanh_f32(to_f32(g))));
      const bf16 act = to_bf16(y * to_f32(u));
      act_q[(long)r * I + n] = f32_to_e4m3(to_f32(act) * dn_a_inv);
    }
  }
}

// ---- 通用 fp8 GEMV：out[r][n] = bf16(Σ_k a_q[r][k]·w[n][k] · scale)。grid = ceil(N / (2·kWarps))，每 warp 两行。
template <int M_TILE>
__global__ void __launch_bounds__(kBlock) gemv_kernel(const uint8_t* __restrict__ a_q, const uint8_t* __restrict__ w,
                                                      bf16* __restrict__ out, int m, int N, int K, float scale) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  bf16* sa = reinterpret_cast<bf16*>(smem_raw);
  load_a<M_TILE>(a_q, sa, m, K);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n0 = (blockIdx.x * kWarps + warp) * 2;
  if (n0 >= N) return;
  const bool two = n0 + 1 < N;
  float acc[2][M_TILE];
#pragma unroll
  for (int r = 0; r < M_TILE; ++r) acc[0][r] = acc[1][r] = 0.f;
  dot2<M_TILE>(sa, K, w + (long)n0 * K, w + (long)(two ? n0 + 1 : n0) * K, acc, lane);
#pragma unroll
  for (int r = 0; r < M_TILE; ++r) {
    if (lane == r && r < m) {
      out[(long)r * N + n0] = to_bf16(acc[0][r] * scale);
      if (two) out[(long)r * N + n0 + 1] = to_bf16(acc[1][r] * scale);
    }
  }
}

template <int M_TILE>
int launch_gateup(const uint8_t* a, const uint8_t* g, const uint8_t* u, uint8_t* act, int m, int I, int K, float gs,
                  float us, float dinv, cudaStream_t s) {
  const int smem = M_TILE * K * 2;
  if (smem > 48 * 1024) {
    if (const int rc = (int)cudaFuncSetAttribute(gateup_act_kernel<M_TILE>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem))
      return rc;
  }
  gateup_act_kernel<M_TILE><<<(unsigned)((I + kWarps - 1) / kWarps), kBlock, smem, s>>>(a, g, u, act, m, I, K, gs, us, dinv);
  return last_err();
}
template <int M_TILE>
int launch_gemv(const uint8_t* a, const uint8_t* w, bf16* out, int m, int N, int K, float scale, cudaStream_t s) {
  const int smem = M_TILE * K * 2;
  if (smem > 48 * 1024) {
    if (const int rc = (int)cudaFuncSetAttribute(gemv_kernel<M_TILE>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem))
      return rc;
  }
  gemv_kernel<M_TILE><<<(unsigned)((N + 2 * kWarps - 1) / (2 * kWarps)), kBlock, smem, s>>>(a, w, out, m, N, K, scale);
  return last_err();
}
}  // namespace
}  // namespace dinfer

extern "C" {
int mlp_fp8_norm_quant(const void* x, const void* w, void* q, int64_t m, int64_t h, float eps, float a_inv,
                       cudaStream_t stream) {
  using dinfer::bf16;
  if (m <= 0) return 0;
  if (h <= 0 || !(a_inv > 0.f)) return (int)cudaErrorInvalidValue;
  dinfer::norm_quant_kernel<<<(unsigned)m, 256, 0, stream>>>((const bf16*)x, (const bf16*)w, (uint8_t*)q, (int)h, eps, a_inv);
  return dinfer::last_err();
}
int mlp_fp8_gateup_act(const void* a_q, const void* gate_w, const void* up_w, void* act_q, int64_t m, int64_t inter,
                       int64_t k, float gate_scale, float up_scale, float dn_a_inv, cudaStream_t stream) {
  using namespace dinfer;
  if (m <= 0) return 0;
  if (m > kMaxM || inter <= 0 || k <= 0 || (k % 16) != 0) return (int)cudaErrorInvalidValue;
  const uint8_t* a = (const uint8_t*)a_q;
  const uint8_t* g = (const uint8_t*)gate_w;
  const uint8_t* u = (const uint8_t*)up_w;
  uint8_t* o = (uint8_t*)act_q;
  if (m == 1) return launch_gateup<1>(a, g, u, o, (int)m, (int)inter, (int)k, gate_scale, up_scale, dn_a_inv, stream);
  if (m == 2) return launch_gateup<2>(a, g, u, o, (int)m, (int)inter, (int)k, gate_scale, up_scale, dn_a_inv, stream);
  if (m <= 4) return launch_gateup<4>(a, g, u, o, (int)m, (int)inter, (int)k, gate_scale, up_scale, dn_a_inv, stream);
  if (m <= 8) return launch_gateup<8>(a, g, u, o, (int)m, (int)inter, (int)k, gate_scale, up_scale, dn_a_inv, stream);
  return launch_gateup<16>(a, g, u, o, (int)m, (int)inter, (int)k, gate_scale, up_scale, dn_a_inv, stream);
}
int mlp_fp8_gemv(const void* a_q, const void* w, void* out, int64_t m, int64_t n, int64_t k, float scale,
                 cudaStream_t stream) {
  using namespace dinfer;
  if (m <= 0) return 0;
  if (m > kMaxM || n <= 0 || k <= 0 || (k % 16) != 0) return (int)cudaErrorInvalidValue;
  const uint8_t* a = (const uint8_t*)a_q;
  const uint8_t* ww = (const uint8_t*)w;
  bf16* o = (bf16*)out;
  if (m == 1) return launch_gemv<1>(a, ww, o, (int)m, (int)n, (int)k, scale, stream);
  if (m == 2) return launch_gemv<2>(a, ww, o, (int)m, (int)n, (int)k, scale, stream);
  if (m <= 4) return launch_gemv<4>(a, ww, o, (int)m, (int)n, (int)k, scale, stream);
  if (m <= 8) return launch_gemv<8>(a, ww, o, (int)m, (int)n, (int)k, scale, stream);
  return launch_gemv<16>(a, ww, o, (int)m, (int)n, (int)k, scale, stream);
}
}  // extern "C"
