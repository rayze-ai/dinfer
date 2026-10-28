// MoE decode（每专家行数小）的带宽型 NVFP4 grouped GEMV：gate‖up + gelu·mul（写 bf16）/ down（写 bf16）。
//
// 为什么不用 moe_grouped_sm120.cu 的 mma tile：decode c1 时每专家 1 行，BM=32 的 tile 装 1 行、每专家只 11/44 个
// CTA、每 K 步 __syncthreads，nsys 50 us/层 vs 权重 25 MB 的带宽下限 14 us（loop kaon3-decode-opt B7）。
// 这里 grid=(N 方向, 专家)，block 内按专家行段以 M_TILE 行为 chunk 循环：chunk 的 A（fp4 → bf16 精确）+ A 块缩放
// （e4m3 字节）驻共享内存；一 warp 负责 NR 行权重（gate‖up 同行号 2 行 / down 相邻 4 行），lane 一次读一个
// 16 元素块（8 B packed + 1 B scale），块内 16 FMA 后乘 (sa·sb) 累加 —— 与 mma block_scale 口径同式（单项
// fp4·fp4·e4m3·e4m3 ≤ 10 bit 尾数在 f32 精确，差只在累加序，bf16 ulp 级）；warp shuffle 归约。
// 布局与 moe_grouped_sm120.cu 同：packed [rows, K/2]（低半字节=偶数 k）、scale [rows, K/16] e4m3、alpha[e] = global·a_scale。
// 结构常量：每 block warp 数（host 按 SM 数选，kMaxWarps 上限）/ M_TILE 档位 {1,2,4,8}；无卡型调优数值；所有 arch 共用。限制：K % 64 == 0。
#include "common.cuh"
#include "kernels.h"

namespace dinfer {
namespace {
constexpr int kMaxWarps = 16;  // 每 block warp 数由 host 选（pick_warps），kernel 读 blockDim.x

__device__ __forceinline__ float e2m1_f(unsigned n) {
  // 0,0.5,1,1.5,2,3,4,6 + 符号位 3
  const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  const float v = mag[n & 7u];
  return (n & 8u) ? -v : v;
}
__device__ __forceinline__ float e4m3_f(unsigned b) {
  const unsigned s = (b >> 7) & 1u, e = (b >> 3) & 15u, m = b & 7u;
  float v;
  if (e == 0) v = m * (1.f / 8.f) * exp2f(-6.f);
  else if (e == 15 && m == 7) v = 448.f;
  else v = (1.f + m * (1.f / 8.f)) * exp2f((float)((int)e - 7));
  return s ? -v : v;
}
// 8 字节 packed（16 个 fp4，低半字节=偶 k）→ 16 个 f32
__device__ __forceinline__ void fp4x16_to_f32(uint2 p, float* f) {
  const uint32_t w[2] = {p.x, p.y};
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const unsigned byte = (w[i] >> (8 * j)) & 0xffu;
      f[i * 8 + 2 * j] = e2m1_f(byte & 0xfu);
      f[i * 8 + 2 * j + 1] = e2m1_f(byte >> 4);
    }
  }
}
__device__ __forceinline__ float gelu_tanh_f32(float x) {
  constexpr float k0 = 0.7978845608028654f;
  return 0.5f * x * (1.f + tanhf(k0 * (x + 0.044715f * x * x * x)));
}

// chunk 装载：行 [r0, r0+mc) 的 A packed → smem bf16 [M_TILE][K]，A 块缩放字节 → smem [M_TILE][nblk]。末尾 barrier。
template <int M_TILE>
__device__ __forceinline__ void load_chunk(const uint8_t* __restrict__ aq, const uint8_t* __restrict__ asc, int r0,
                                           int mc, int K, bf16* sa, uint8_t* ssc) {
  const int nblk = K / 16;
  for (int idx = threadIdx.x; idx < M_TILE * nblk; idx += blockDim.x) {
    const int r = idx / nblk, b = idx % nblk;
    if (r < mc) {
      float f[16];
      fp4x16_to_f32(__ldg(reinterpret_cast<const uint2*>(aq + (long)(r0 + r) * (K / 2) + b * 8)), f);
      uint4 o[2];
      __nv_bfloat162* ob = reinterpret_cast<__nv_bfloat162*>(o);
#pragma unroll
      for (int i = 0; i < 8; ++i) ob[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
      uint4* dst = reinterpret_cast<uint4*>(sa + (long)r * K + b * 16);
      dst[0] = o[0];
      dst[1] = o[1];
      ssc[r * nblk + b] = asc[(long)(r0 + r) * nblk + b];
    }
  }
  __syncthreads();
}

// 一 warp：NR 行权重（packed 行指针 wr[j]、scale 行指针 sr[j]）× chunk 的 M_TILE 行；acc 归约后每 lane 持全和。
template <int M_TILE, int NR>
__device__ __forceinline__ void dot_fp4(const bf16* sa, const uint8_t* ssc, int K, const uint8_t* const* wr,
                                        const uint8_t* const* sr, float (&acc)[NR][M_TILE], int lane) {
  const int nblk = K / 16;
  for (int b = lane; b < nblk; b += 32) {
    float wf[NR][16], ws[NR];
#pragma unroll
    for (int j = 0; j < NR; ++j) {
      fp4x16_to_f32(__ldg(reinterpret_cast<const uint2*>(wr[j] + b * 8)), wf[j]);
      ws[j] = e4m3_f(__ldg(sr[j] + b));
    }
#pragma unroll
    for (int r = 0; r < M_TILE; ++r) {
      const uint4* ap = reinterpret_cast<const uint4*>(sa + (long)r * K + b * 16);
      const uint4 a0 = ap[0], a1 = ap[1];
      const __nv_bfloat162* b0 = reinterpret_cast<const __nv_bfloat162*>(&a0);
      const __nv_bfloat162* b1 = reinterpret_cast<const __nv_bfloat162*>(&a1);
      float af[16];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 x = __bfloat1622float2(b0[i]);
        const float2 y = __bfloat1622float2(b1[i]);
        af[2 * i] = x.x; af[2 * i + 1] = x.y; af[8 + 2 * i] = y.x; af[8 + 2 * i + 1] = y.y;
      }
      const float as = e4m3_f(ssc[r * nblk + b]);
#pragma unroll
      for (int j = 0; j < NR; ++j) {
        float part = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) part = fmaf(wf[j][i], af[i], part);
        acc[j][r] = fmaf(part, ws[j] * as, acc[j][r]);
      }
    }
  }
#pragma unroll
  for (int j = 0; j < NR; ++j) {
#pragma unroll
    for (int r = 0; r < M_TILE; ++r) {
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) acc[j][r] += __shfl_xor_sync(0xffffffffu, acc[j][r], off);
    }
  }
}

// ---- gate‖up + gelu·mul：grid=(ceil(I/warps), E)；out[R, I] bf16 = bf16(bf16(gelu(bf16(g·αg)))·bf16(u·αu))。
template <int M_TILE>
__global__ void __launch_bounds__(kMaxWarps * 32) gateup_kernel(const uint8_t* __restrict__ aq, const uint8_t* __restrict__ asc,
                                                        const uint8_t* const* __restrict__ gptr,
                                                        const uint8_t* const* __restrict__ gsptr,
                                                        const float* __restrict__ alpha_g,
                                                        const uint8_t* const* __restrict__ uptr,
                                                        const uint8_t* const* __restrict__ usptr,
                                                        const float* __restrict__ alpha_u,
                                                        const int32_t* __restrict__ offsets, bf16* __restrict__ out,
                                                        int I, int K) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  const int e = blockIdx.y;
  const int row_lo = offsets[e], row_hi = offsets[e + 1];
  if (row_lo >= row_hi) return;  // 空专家：整块退（无 barrier 之前）
  bf16* sa = reinterpret_cast<bf16*>(smem_raw);
  uint8_t* ssc = smem_raw + (long)M_TILE * K * 2;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = blockIdx.x * (blockDim.x >> 5) + warp;
  const uint8_t* wr[2] = {gptr[e] + (long)n * (K / 2), uptr[e] + (long)n * (K / 2)};
  const uint8_t* sr[2] = {gsptr[e] + (long)n * (K / 16), usptr[e] + (long)n * (K / 16)};
  const float ag = alpha_g[e], au = alpha_u[e];
  for (int c0 = row_lo; c0 < row_hi; c0 += M_TILE) {
    const int mc = min(M_TILE, row_hi - c0);
    load_chunk<M_TILE>(aq, asc, c0, mc, K, sa, ssc);
    if (n < I) {
      float acc[2][M_TILE];
#pragma unroll
      for (int r = 0; r < M_TILE; ++r) acc[0][r] = acc[1][r] = 0.f;
      dot_fp4<M_TILE, 2>(sa, ssc, K, wr, sr, acc, lane);
#pragma unroll
      for (int r = 0; r < M_TILE; ++r) {
        if (lane == r && r < mc) {
          const bf16 g = to_bf16(acc[0][r] * ag);
          const bf16 u = to_bf16(acc[1][r] * au);
          const float y = to_f32(to_bf16(gelu_tanh_f32(to_f32(g))));
          out[(long)(c0 + r) * I + n] = to_bf16(y * to_f32(u));
        }
      }
    }
    __syncthreads();  // 下一 chunk 覆盖 smem
  }
}

// ---- down（通用 grouped GEMV）：grid=(ceil(N/(4·warps)), E)，一 warp 4 行；out[R, N] bf16 = bf16(acc·α)。
template <int M_TILE>
__global__ void __launch_bounds__(kMaxWarps * 32) gemv_kernel(const uint8_t* __restrict__ aq, const uint8_t* __restrict__ asc,
                                                      const uint8_t* const* __restrict__ bptr,
                                                      const uint8_t* const* __restrict__ sptr,
                                                      const float* __restrict__ alpha,
                                                      const int32_t* __restrict__ offsets, bf16* __restrict__ out, int N,
                                                      int K) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  const int e = blockIdx.y;
  const int row_lo = offsets[e], row_hi = offsets[e + 1];
  if (row_lo >= row_hi) return;
  bf16* sa = reinterpret_cast<bf16*>(smem_raw);
  uint8_t* ssc = smem_raw + (long)M_TILE * K * 2;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 4;
  const uint8_t* wr[4];
  const uint8_t* sr[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int nj = min(n0 + j, N - 1);  // 越界行夹到最后一行（结果不写）
    wr[j] = bptr[e] + (long)nj * (K / 2);
    sr[j] = sptr[e] + (long)nj * (K / 16);
  }
  const float al = alpha[e];
  for (int c0 = row_lo; c0 < row_hi; c0 += M_TILE) {
    const int mc = min(M_TILE, row_hi - c0);
    load_chunk<M_TILE>(aq, asc, c0, mc, K, sa, ssc);
    if (n0 < N) {
      float acc[4][M_TILE];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int r = 0; r < M_TILE; ++r) acc[j][r] = 0.f;
      dot_fp4<M_TILE, 4>(sa, ssc, K, wr, sr, acc, lane);
#pragma unroll
      for (int r = 0; r < M_TILE; ++r) {
        if (lane == r && r < mc) {
#pragma unroll
          for (int j = 0; j < 4; ++j)
            if (n0 + j < N) out[(long)(c0 + r) * N + n0 + j] = to_bf16(acc[j][r] * al);
        }
      }
    }
    __syncthreads();
  }
}

// host 选每 block warp 数（同 mlp_fp8_decode.cu）：按 SM 数与（活跃专家数估 × 行数）算，不是卡型调优数值。
// 活跃专家数 host 不知道（offsets 在 device），用 min(rows, E) 估。
int pick_warps(int m_tile, long rows_total_per_e, long active_e_est, int rows_per_warp) {
  static int sms = 0;
  if (!sms) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    if (sms <= 0) sms = 1;
  }
  // D64：warp 多会把 block 数压到 < SM 数，权重流读并行度不够，c1 反慢 1.8 ms；小 M 下 block 数比 A 复用重要。
  // 上限 4 warp，且要求 block ≥ 4×SM。
  int w = 4;
  const long warps_total = (rows_total_per_e + rows_per_warp - 1) / rows_per_warp * (active_e_est > 0 ? active_e_est : 1);
  while (w > 1 && warps_total / w < 4L * sms) w >>= 1;
  return w;
}
template <int M_TILE>
int launch_gateup(const uint8_t* aq, const uint8_t* asc, const uint8_t* const* gp, const uint8_t* const* gs,
                  const float* ag, const uint8_t* const* up, const uint8_t* const* us, const float* au,
                  const int32_t* off, bf16* out, int I, int K, int E, long rows, cudaStream_t s) {
  const int smem = M_TILE * K * 2 + M_TILE * (K / 16);
  const int warps = pick_warps(M_TILE, I, rows < E ? rows : E, 1);
  static int attr_smem = 0;
  if (smem > 48 * 1024 && smem > attr_smem) {
    if (const int rc = (int)cudaFuncSetAttribute(gateup_kernel<M_TILE>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem))
      return rc;
    attr_smem = smem;
  }
  const dim3 grid((unsigned)((I + warps - 1) / warps), (unsigned)E);
  gateup_kernel<M_TILE><<<grid, warps * 32, smem, s>>>(aq, asc, gp, gs, ag, up, us, au, off, out, I, K);
  return last_err();
}
template <int M_TILE>
int launch_gemv(const uint8_t* aq, const uint8_t* asc, const uint8_t* const* bp, const uint8_t* const* sp,
                const float* al, const int32_t* off, bf16* out, int N, int K, int E, long rows, cudaStream_t s) {
  const int smem = M_TILE * K * 2 + M_TILE * (K / 16);
  const int warps = pick_warps(M_TILE, N, rows < E ? rows : E, 4);
  static int attr_smem = 0;
  if (smem > 48 * 1024 && smem > attr_smem) {
    if (const int rc = (int)cudaFuncSetAttribute(gemv_kernel<M_TILE>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem))
      return rc;
    attr_smem = smem;
  }
  const dim3 grid((unsigned)((N + 4 * warps - 1) / (4 * warps)), (unsigned)E);
  gemv_kernel<M_TILE><<<grid, warps * 32, smem, s>>>(aq, asc, bp, sp, al, off, out, N, K);
  return last_err();
}
}  // namespace
}  // namespace dinfer

extern "C" {
// rows = 总行数（决定 M_TILE 档位：每专家行数 ≤ rows）；每专家行段来自 offsets（device）。
int moe_fp4_gateup_act(const void* aq, const void* asc, const void* const* gate_ptr, const void* const* gate_sptr,
                       const float* alpha_gate, const void* const* up_ptr, const void* const* up_sptr,
                       const float* alpha_up, const int32_t* offsets, void* out, int64_t rows, int64_t inter, int64_t k,
                       int64_t num_experts, cudaStream_t stream) {
  using namespace dinfer;
  if (rows <= 0 || num_experts <= 0) return 0;
  if (inter <= 0 || k <= 0 || (k % 64) != 0) return (int)cudaErrorInvalidValue;
  const uint8_t* a = (const uint8_t*)aq;
  const uint8_t* as = (const uint8_t*)asc;
  auto gp = (const uint8_t* const*)gate_ptr;
  auto gs = (const uint8_t* const*)gate_sptr;
  auto up = (const uint8_t* const*)up_ptr;
  auto us = (const uint8_t* const*)up_sptr;
  bf16* o = (bf16*)out;
  const int I = (int)inter, K = (int)k, E = (int)num_experts;
  if (rows == 1) return launch_gateup<1>(a, as, gp, gs, alpha_gate, up, us, alpha_up, offsets, o, I, K, E, rows, stream);
  if (rows == 2) return launch_gateup<2>(a, as, gp, gs, alpha_gate, up, us, alpha_up, offsets, o, I, K, E, rows, stream);
  if (rows <= 4) return launch_gateup<4>(a, as, gp, gs, alpha_gate, up, us, alpha_up, offsets, o, I, K, E, rows, stream);
  return launch_gateup<8>(a, as, gp, gs, alpha_gate, up, us, alpha_up, offsets, o, I, K, E, rows, stream);
}
int moe_fp4_gemv(const void* aq, const void* asc, const void* const* bptr, const void* const* sptr, const float* alpha,
                 const int32_t* offsets, void* out, int64_t rows, int64_t n, int64_t k, int64_t num_experts,
                 cudaStream_t stream) {
  using namespace dinfer;
  if (rows <= 0 || num_experts <= 0) return 0;
  if (n <= 0 || k <= 0 || (k % 64) != 0) return (int)cudaErrorInvalidValue;
  const uint8_t* a = (const uint8_t*)aq;
  const uint8_t* as = (const uint8_t*)asc;
  auto bp = (const uint8_t* const*)bptr;
  auto sp = (const uint8_t* const*)sptr;
  bf16* o = (bf16*)out;
  const int N = (int)n, K = (int)k, E = (int)num_experts;
  if (rows == 1) return launch_gemv<1>(a, as, bp, sp, alpha, offsets, o, N, K, E, rows, stream);
  if (rows == 2) return launch_gemv<2>(a, as, bp, sp, alpha, offsets, o, N, K, E, rows, stream);
  if (rows <= 4) return launch_gemv<4>(a, as, bp, sp, alpha, offsets, o, N, K, E, rows, stream);
  return launch_gemv<8>(a, as, bp, sp, alpha, offsets, o, N, K, E, rows, stream);
}
}  // extern "C"
