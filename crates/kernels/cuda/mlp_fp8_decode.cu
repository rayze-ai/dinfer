// decode（M ≤ 16）dense MLP 的带宽型 fp8 路径：norm→fp8 / gate‖up GEMV + gelu·mul→fp8 / down GEMV。
//
// 为什么不用 persistent mega（mega_mlp_sm120.cu）：decode 时 M=1，三个 GEMM 都是纯访存；mega 的阶段链
// 1 CTA/SM 自旋等依赖，nsys 实测 86.9 us/层 vs 权重 17.8 MB 的带宽下限 10 us（loop kaon3-decode-opt D60）。
// 这里每个 warp 负责权重的 2 行（gate‖up 同行号 / down 相邻两行，共用一次 A 读），lane 一次 16 B 向量读、
// K 维流式扫完；A（fp8 → bf16 精确展开）整体驻共享内存被 block 内所有 warp 复用；f32 FMA 累加、warp shuffle 归约。
// 数值口径与 mega 一致：fp8 输入、f32 累加、epilogue 乘 (w_scale·a_scale) 后舍 bf16、gelu 先舍 bf16 再乘 up 再舍、
// fp8 用 SATFINITE。累加顺序不同于 mma，非逐位，差在 bf16 ulp 级（Rust 测试 mlp_fp8_decode_parity）。
// 结构常量：每 block warp 数（host 按 SM 数/形状选，kMaxWarps 上限）、M_TILE 模板档位 {1,2,4,8,16}（决定共享内存 = M_TILE·K·2 B）。
// 没有卡型调优数值。所有 arch 共用（fp8 转换用 cuda_fp8.h）。限制：K % 16 == 0。
#include <cuda_fp8.h>

#include "common.cuh"
#include "kernels.h"

namespace dinfer {
namespace {
constexpr int kMaxM = 16;
constexpr int kMaxWarps = 16;  // block 最多 16 warp（launch_bounds 上限）
// 每 block 的 warp 数由 host 按 M_TILE / N 选（见 pick_warps），kernel 用 blockDim.x 读。
// 为什么可变：一次 A（M_TILE·K bf16）装进 smem 的流量要被 block 内所有 warp 的权重行摊；4 warp 只摊 8 行时
// m ≥ 8 的 A 装载反超权重流量（loop kaon3-decode-opt D62）。

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
// A8（loop kaon3-decode-opt D176）：A 以 e4m3 原样驻留 smem（M_TILE·K 字节，是 bf16 版的一半），dot2_a8 里按 16 元素
// 一次 `e4m3x16_to_f32` 解包。e4m3→f32 与 e4m3→bf16→f32 值完全相同，fma 顺序同 → 与 bf16 驻留版逐位同。
// 为什么：M_TILE=8×K=4096 的 o_proj（sliding 层）bf16 驻留要 64 KB > 48 KB 门槛，退回 cuBLASLt split-K（M=8 时 11.7 us ×30 call/步）。
template <int M_TILE>
__device__ __forceinline__ void load_a8(const uint8_t* __restrict__ a_q, uint8_t* sa, int m, int K) {
  const int nvec = K / 16;
  for (int idx = threadIdx.x; idx < M_TILE * nvec; idx += blockDim.x) {
    const int r = idx / nvec, c = idx % nvec;
    if (r < m) *reinterpret_cast<uint4*>(sa + (long)r * K + c * 16) = *reinterpret_cast<const uint4*>(a_q + (long)r * K + c * 16);
  }
  __syncthreads();
}
template <int M_TILE>
__device__ __forceinline__ void load_a(const uint8_t* __restrict__ a_q, bf16* sa, int m, int K) {
  const int nvec = K / 16;
  for (int idx = threadIdx.x; idx < M_TILE * nvec; idx += blockDim.x) {
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
// B20（loop kaon3-decode-opt D72）：先把本 lane 的全部权重向量一次性读进寄存器（≤ kMaxVecPerLane 个/行），再转换+FMA。
// 原版每轮迭代读 2 个 uint4 就等它回来，K=2816 时 5.5 轮串行 HBM 往返 ≈ 延迟绑定（qkv 16.5 us vs 带宽下限 4.8）。
// 累加顺序与原版完全相同（同 c 升序）→ 逐位同。K ≤ 16·32·kMaxVecPerLane = 4096。
constexpr int kMaxVecPerLane = 8;
// B42b（loop kaon3-decode-opt D185）：权重预取与点积拆开——kernel 先把本 warp 两行权重的 __ldg 发出去（寄存器在飞），
// 再做 load_a（smem 装载 + barrier），让 DRAM 流不等 prologue。算术序不变 → 逐位同。
__device__ __forceinline__ void prefetch2(int K, const uint8_t* __restrict__ w0, const uint8_t* __restrict__ w1,
                                          uint4 (&r0)[kMaxVecPerLane], uint4 (&r1)[kMaxVecPerLane], int lane) {
  const int nvec = K / 16;
#pragma unroll
  for (int j = 0; j < kMaxVecPerLane; ++j) {
    const int c = lane + j * 32;
    if (c < nvec) {
      r0[j] = __ldg(reinterpret_cast<const uint4*>(w0 + c * 16));
      r1[j] = __ldg(reinterpret_cast<const uint4*>(w1 + c * 16));
    }
  }
}
template <int M_TILE>
__device__ __forceinline__ void dot2(const bf16* sa, int K, const uint4 (&r0)[kMaxVecPerLane],
                                     const uint4 (&r1)[kMaxVecPerLane], float (&acc)[2][M_TILE], int lane) {
  const int nvec = K / 16;
#pragma unroll
  for (int j = 0; j < kMaxVecPerLane; ++j) {
    const int c = lane + j * 32;
    if (c < nvec) {
      float wf[2][16];
      e4m3x16_to_f32(r0[j], wf[0]);
      e4m3x16_to_f32(r1[j], wf[1]);
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
// A8 版 dot2：A 从 smem 以 e4m3 读（见 load_a8）。累加序与 dot2 完全相同。
template <int M_TILE>
__device__ __forceinline__ void dot2_a8(const uint8_t* sa, int K, const uint4 (&r0)[kMaxVecPerLane],
                                        const uint4 (&r1)[kMaxVecPerLane], float (&acc)[2][M_TILE], int lane) {
  const int nvec = K / 16;
#pragma unroll
  for (int j = 0; j < kMaxVecPerLane; ++j) {
    const int c = lane + j * 32;
    if (c < nvec) {
      float wf[2][16];
      e4m3x16_to_f32(r0[j], wf[0]);
      e4m3x16_to_f32(r1[j], wf[1]);
#pragma unroll
      for (int r = 0; r < M_TILE; ++r) {
        float af[16];
        e4m3x16_to_f32(*reinterpret_cast<const uint4*>(sa + (long)r * K + c * 16), af);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float2 x = make_float2(af[2 * i], af[2 * i + 1]);
          const float2 y = make_float2(af[8 + 2 * i], af[8 + 2 * i + 1]);
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

// ---- gate‖up GEMV + gelu·mul → fp8。grid = ceil(I / warps)，每 warp 一个行号 n（gate 行 n + up 行 n）。
template <int M_TILE, bool A8 = false>
__global__ void __launch_bounds__(kMaxWarps * 32) gateup_act_kernel(const uint8_t* __restrict__ a_q,
                                                            const uint8_t* __restrict__ gw,
                                                            const uint8_t* __restrict__ uw,
                                                            uint8_t* __restrict__ act_q, int m, int I, int K,
                                                            float gs, float us, float dn_a_inv) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  bf16* sa = reinterpret_cast<bf16*>(smem_raw);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = blockIdx.x * (blockDim.x >> 5) + warp;
  const bool active = n < I;
  uint4 r0[kMaxVecPerLane], r1[kMaxVecPerLane];
  if (active) prefetch2(K, gw + (long)n * K, uw + (long)n * K, r0, r1, lane);  // B42b：先发权重读
  if constexpr (A8) load_a8<M_TILE>(a_q, smem_raw, m, K); else load_a<M_TILE>(a_q, sa, m, K);
  if (!active) return;  // load_a 已过 barrier，之后无 barrier
  float acc[2][M_TILE];
#pragma unroll
  for (int r = 0; r < M_TILE; ++r) acc[0][r] = acc[1][r] = 0.f;
  if constexpr (A8) dot2_a8<M_TILE>(smem_raw, K, r0, r1, acc, lane);
  else dot2<M_TILE>(sa, K, r0, r1, acc, lane);
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

// ---- 通用 fp8 GEMV：out[r][n] = bf16(Σ_k a_q[r][k]·w[n][k] · scale)。grid = ceil(N / (2·warps))，每 warp 两行。
template <int M_TILE, bool A8 = false>
__global__ void __launch_bounds__(kMaxWarps * 32) gemv_kernel(const uint8_t* __restrict__ a_q, const uint8_t* __restrict__ w,
                                                      bf16* __restrict__ out, int m, int N, int K, float scale) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  bf16* sa = reinterpret_cast<bf16*>(smem_raw);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 2;
  const bool active = n0 < N;
  const bool two = n0 + 1 < N;
  uint4 r0[kMaxVecPerLane], r1[kMaxVecPerLane];
  if (active) prefetch2(K, w + (long)n0 * K, w + (long)(two ? n0 + 1 : n0) * K, r0, r1, lane);  // B42b：先发权重读
  if constexpr (A8) load_a8<M_TILE>(a_q, smem_raw, m, K); else load_a<M_TILE>(a_q, sa, m, K);
  if (!active) return;
  float acc[2][M_TILE];
#pragma unroll
  for (int r = 0; r < M_TILE; ++r) acc[0][r] = acc[1][r] = 0.f;
  if constexpr (A8) dot2_a8<M_TILE>(smem_raw, K, r0, r1, acc, lane);
  else dot2<M_TILE>(sa, K, r0, r1, acc, lane);
#pragma unroll
  for (int r = 0; r < M_TILE; ++r) {
    if (lane == r && r < m) {
      out[(long)r * N + n0] = to_bf16(acc[0][r] * scale);
      if (two) out[(long)r * N + n0 + 1] = to_bf16(acc[1][r] * scale);
    }
  }
}

// host 选每 block warp 数：A 装载摊到尽量多行（16 warp），但 M_TILE=16 时寄存器压力大用 8；
// 并保证 grid 至少 ~2×SM 数的 block（小 N 不丢并行度）。不是卡型调优数值：按 SM 数与形状算。
int pick_warps(int m_tile, long rows_total, int rows_per_warp) {
  // B31 探针：FP8_GEMV_WARPS=1/2/4/8/16 强制每 block warp 数（微基准扫参用；默认按 SM 数自选）。
  static const int force = []{ const char* v = getenv("FP8_GEMV_WARPS"); const int n = v ? atoi(v) : 0; return (n == 1 || n == 2 || n == 4 || n == 8 || n == 16) ? n : 0; }();
  if (force) return force;
  static int sms = 0;
  if (!sms) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    if (sms <= 0) sms = 1;
  }
  // D64：warp 多会把 block 数压到 < SM 数，权重流读并行度不够，c1 反慢 1.8 ms；小 M 下 block 数比 A 复用重要。
  // 上限 4 warp，且要求 block ≥ 4×SM。
  // B31b（loop kaon3-decode-opt D155 gemvsweep）：M_TILE ≥ 4 时 smem 里的 A（M_TILE·K bf16，33–45 KB）要被足够多的 warp 摊，
  // 每 block 8 warp 比自选的 2 warp 快 1.8×（m=8 dense_down 18.5 → 10.3 us，qkv 23.8 → 12.3；16 warp 与 8 同）。
  // m=1/2 仍按 "block 数 ≥ 4×SM" 自选（m=1 已在带宽下限）。
  if (m_tile >= 4) return 8;
  int w = 4;
  const long warps_total = (rows_total + rows_per_warp - 1) / rows_per_warp;
  while (w > 1 && warps_total / w < 4L * sms) w >>= 1;
  return w;
}
// A8 开关：`FP8_GEMV_A8=0` 回 bf16 驻留。Rust 侧 `fp8_gemv_a_bytes()` 同步此口径算门槛。
static bool a8_enabled() {
  static const bool on = []{ const char* v = getenv("FP8_GEMV_A8"); return !v || v[0] != '0'; }();
  return on;
}
template <int M_TILE, bool A8>
int launch_gateup_t(const uint8_t* a, const uint8_t* g, const uint8_t* u, uint8_t* act, int m, int I, int K, float gs,
                    float us, float dinv, cudaStream_t s) {
  const int smem = M_TILE * K * (A8 ? 1 : 2);
  const int warps = pick_warps(M_TILE, I, 1);
  static int attr_smem = 0;  // 每个模板实例一次（host 侧属性，非流操作；graph 捕获时也只在首次调用）
  if (smem > 48 * 1024 && smem > attr_smem) {
    if (const int rc = (int)cudaFuncSetAttribute(gateup_act_kernel<M_TILE, A8>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem))
      return rc;
    attr_smem = smem;
  }
  gateup_act_kernel<M_TILE, A8><<<(unsigned)((I + warps - 1) / warps), warps * 32, smem, s>>>(a, g, u, act, m, I, K, gs, us, dinv);
  return last_err();
}
template <int M_TILE>
int launch_gateup(const uint8_t* a, const uint8_t* g, const uint8_t* u, uint8_t* act, int m, int I, int K, float gs,
                  float us, float dinv, cudaStream_t s) {
  return a8_enabled() ? launch_gateup_t<M_TILE, true>(a, g, u, act, m, I, K, gs, us, dinv, s)
                      : launch_gateup_t<M_TILE, false>(a, g, u, act, m, I, K, gs, us, dinv, s);
}
template <int M_TILE, bool A8>
int launch_gemv_t(const uint8_t* a, const uint8_t* w, bf16* out, int m, int N, int K, float scale, cudaStream_t s) {
  const int smem = M_TILE * K * (A8 ? 1 : 2);
  const int warps = pick_warps(M_TILE, N, 2);
  static int attr_smem = 0;  // 每个模板实例一次（host 侧属性，非流操作；graph 捕获时也只在首次调用）
  if (smem > 48 * 1024 && smem > attr_smem) {
    if (const int rc = (int)cudaFuncSetAttribute(gemv_kernel<M_TILE, A8>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem))
      return rc;
    attr_smem = smem;
  }
  gemv_kernel<M_TILE, A8><<<(unsigned)((N + 2 * warps - 1) / (2 * warps)), warps * 32, smem, s>>>(a, w, out, m, N, K, scale);
  return last_err();
}
template <int M_TILE>
int launch_gemv(const uint8_t* a, const uint8_t* w, bf16* out, int m, int N, int K, float scale, cudaStream_t s) {
  return a8_enabled() ? launch_gemv_t<M_TILE, true>(a, w, out, m, N, K, scale, s)
                      : launch_gemv_t<M_TILE, false>(a, w, out, m, N, K, scale, s);
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
  if (m > kMaxM || inter <= 0 || k <= 0 || (k % 16) != 0 || k > 16 * 32 * kMaxVecPerLane) return (int)cudaErrorInvalidValue;
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
// A8 口径查询：每个 A 元素在 smem 里占几字节（1 = e4m3 驻留，2 = bf16 驻留）。Rust 侧据此算 M_TILE·K·bytes ≤ 门槛。
int mlp_fp8_gemv_a_bytes() { return dinfer::a8_enabled() ? 1 : 2; }
// 构建标记（bN 链用 strings 验二进制是否含本版 kernel）。
const char* mlp_fp8_build_tag() { return "fp8gemv_b42b_prefetch"; }
int mlp_fp8_gemv(const void* a_q, const void* w, void* out, int64_t m, int64_t n, int64_t k, float scale,
                 cudaStream_t stream) {
  using namespace dinfer;
  if (m <= 0) return 0;
  if (m > kMaxM || n <= 0 || k <= 0 || (k % 16) != 0 || k > 16 * 32 * kMaxVecPerLane) return (int)cudaErrorInvalidValue;
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
