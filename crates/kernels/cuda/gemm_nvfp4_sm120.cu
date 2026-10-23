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
#include <cstdio>
#include <cstdlib>
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
    v = m * (1.f / 8.f) * exp2f(-6.f);  // 次正规：m/8 * 2^-6（实测镜像：0x23 → 0.171875）
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
__device__ __forceinline__ float rcp_approx(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
  return b;
}

/// gs = 激活全局 scale（vLLM 的 a1_gscale = 1/input_scale；不用时 1）。块 scale 与输出缩放逐式照抄
/// vLLM `nvfp4_utils.cuh` cvt_warp_fp16_to_fp4 的 NVFP4 分支（含 rcp.approx.ftz），保证与基线同一套量化网格。
__device__ __forceinline__ void quant16(const bf16* src, bool valid, uint2* out8, uint8_t* out_s, float gs, int exact_rq) {
  float x[16];
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    x[i] = valid ? to_f32(src[i]) : 0.f;
    amax = fmaxf(amax, fabsf(x[i]));
  }
  uint8_t sb8 = 0;  // 全零块：scale 字节 0、输出 0（与 vLLM 一致；硬件 0·0 = 0）
  float inv = 0.f;
  if (amax > 0.f) {
    // `exact_rq`：块缩放用**精确除法** `amax / 6`（对照参照），否则 `amax * rcp.approx(1/6)`（历史口径，1 ulp 偏差）。
    // 由 host 按 `NQ4_EXACT_RCP` 传入。A/B 见 docs/loops/kaon3-compile.md 第 8 轮。
    const float sfv = gs * (exact_rq ? __fdiv_rn(amax, 6.f) : (amax * rcp_approx(6.f)));
    sb8 = __nv_cvt_float_to_fp8(sfv, __NV_SATFINITE, __NV_E4M3);
    const float sfd = e4m3_f(sb8);
    inv = sfd != 0.f ? rcp_approx(sfd * rcp_approx(gs)) : 0.f;
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

// A 的预量化：bf16 [m,k] → packed [m,k/2] + 块缩放 [m,k/16]（e4m3）。
// 一个 16-元素块一个线程；权重侧同一套口径（scale = amax/6 的 e4m3，反量化后做 e2m1 就近舍入）。
__global__ void __launch_bounds__(256) quantize_a_kernel(const bf16* __restrict__ A, uint8_t* __restrict__ P,
                                                         uint8_t* __restrict__ S, int rows, int K, float gs, int exact_rq) {
  const int nb = K / 16;
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)rows * nb) return;
  const int r = (int)(idx / nb), bi = (int)(idx % nb);
  uint2 pk;
  uint8_t sb8;
  quant16(A + (size_t)r * K + bi * 16, true, &pk, &sb8, gs, exact_rq);
  *reinterpret_cast<uint2*>(P + (size_t)r * (K / 2) + bi * 8) = pk;
  S[(size_t)r * nb + bi] = sb8;
}

template <int BK>
__global__ void __launch_bounds__(256) gemm_nvfp4_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Sc,
                                                         const uint8_t* __restrict__ B, const uint8_t* __restrict__ S,
                                                         bf16* __restrict__ D, int M, int N, int K, float global,
                                                         int swz) {
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
      // A 已由 quantize_a 预量化成 [m, k/2] 字节（每行 K/2 字节，uint4 跨度同 B），这里只做 smem 搬运：
      // 与探针同粒度（每线程一个 uint4），RSA 含 16 B padding。
      const uint4* sa_src = reinterpret_cast<const uint4*>(A + static_cast<size_t>(m_blk) * (K / 2) + kb * (BK / 2));
      for (int i = tid; i < (BM * BK / 2) / 16; i += blockDim.x) {
        const int r = i / rb, c = i % rb;
        const uint4 z = make_uint4(0x11, 0x11, 0x11, 0x11);
        *reinterpret_cast<uint4*>(&sA[r * RSA + c * 16]) =
            (m_blk + r < M) ? sa_src[static_cast<size_t>(r) * (K / 32) + c] : z;
      }
      // A 的块缩放。
      //   `swz==0`：自然布局 `[M, K/16]`（我们自己量化的输出，行主序）；
      //   `swz==1`：**cutlass 的 padded+swizzled 布局**（`[round_up(M,128), round_up(K/16,4)]`），
      //     可以直接吃参照的 `a_s`。映射与 `vllm/.../nvfp4_utils.py::swizzle_blockscale` 一致：
      //       padded.reshape(g, a4, r32, kO, k4).transpose(0,1,4,3,2,5) → flat
      //     读第 (row, kb*NB16 + q) 个 16-元素块的 scale 时：
      //       g  = row / 128, rIn128 = row % 128, a4 = rIn128 / 32, r32 = rIn128 % 32
      //       col = kb*NB16 + q;  kO = col / 4, k4 = col % 4
      //       线性下标 = ((((g*(Mp/128) + a4) * (Kp/4) + kO) * 32 + r32) * 4 + k4)
      if (swz) {
        // 正确的维序（用随机数据枚举验证）：平铺索引 = ((((g*(Kp/4) + kO)*32 + r32)*4 + a4)*4 + k4)
        // 其中 row → g = row/128, r32 = row%32, a4 = (row%128)/32；col → kO = col/4, k4 = col%4
        const int Kp = ((K / 16) + 3) / 4 * 4;
        const int Mp = (M + 127) / 128 * 128;
        const long gp = (long)(Mp / 128);
        for (int i = tid; i < BM * NQ4; i += blockDim.x) {
          const int r = i / NQ4, q = i % NQ4;
          const int row = m_blk + r;
          if (row >= M) { sfa[r * NQ4 + q] = 0x38383838u; continue; }
          const int g = row / 128, rIn = row % 128, r32 = rIn % 32, a4 = rIn / 32;
          uint32_t v = 0;
          for (int t = 0; t < 4; ++t) {
            const int col = kb * NB16 + q * 4 + t;
            const int kO = col / 4, k4 = col % 4;
            const long off = ((((long)g * (Kp / 4) + kO) * 32 + r32) * 4 + a4) * 4 + k4;
            v |= (uint32_t)Sc[off] << (8 * t);
          }
          sfa[r * NQ4 + q] = v;
        }
        __syncthreads();
        if (tid == 0 && kb == 0) {
          printf("[swz] swz=%d sfa[0..4]=%08x %08x %08x %08x\n", swz, sfa[0], sfa[1], sfa[2], sfa[3]);
        }
        __syncthreads();
      } else {
        for (int i = tid; i < BM * NQ4; i += blockDim.x) {
          const int r = i / NQ4, q = i % NQ4;
          sfa[r * NQ4 + q] =
              (m_blk + r < M)
                  ? *reinterpret_cast<const uint32_t*>(Sc + static_cast<size_t>(m_blk + r) * (K / 16) + kb * NB16 + q * 4)
                  : 0x38383838u;
        }
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
      if (tid == 0 && kb == 0 && b == 0 && swz) {
        printf("[use] swz=%d sa=%08x sfa[0]=%08x sfa[1]=%08x\n", swz, sa, sfa[0], sfa[1]);
      }
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

// a_gscale = 激活全局 scale（vLLM a1_gscale = 1/input_scale）：只影响 A 的量化网格；global 是最终输出乘子
// （调用方给 w_global * input_scale，即 vLLM 的 alpha）。
// **A 已预量化**的入口：`aq` = `[m, k/2]` 的 fp4 字节，`as` = A 的块缩放。
// `swz`：`as` 是 cutlass 的 padded+swizzled 布局（1）还是自然 `[m, k/16]`（0）。
// 用途：把"量化"与"GEMM"隔离后逐位对拍（参照的 `a_q/a_s` 可以直接喂进来）。
extern "C" int gemm_nvfp4_prequant_bf16(const void* aq, const void* as, const void* packed, const void* scale,
                                        float global, void* out, int64_t m, int64_t n, int64_t k, int swz,
                                        cudaStream_t stream) {
  fprintf(stderr, "[nvfp4] prequant ENTER m=%lld n=%lld k=%lld swz=%d\n", (long long)m, (long long)n, (long long)k, swz);
  if (m < 0 || n < 0 || k <= 0) return static_cast<int>(cudaErrorInvalidValue);
  if (m == 0 || n == 0) return 0;
  if (k % 64 != 0) return static_cast<int>(cudaErrorInvalidValue);
  const dim3 grid((unsigned)((n + BN - 1) / BN), (unsigned)((m + BM - 1) / BM));
  const uint8_t* pa = static_cast<const uint8_t*>(aq);
  const uint8_t* psa = static_cast<const uint8_t*>(as);
  if (k % 128 == 0) {
    gemm_nvfp4_kernel<128><<<grid, 256, 0, stream>>>(pa, psa, static_cast<const uint8_t*>(packed),
                                                     static_cast<const uint8_t*>(scale), static_cast<bf16*>(out),
                                                     (int)m, (int)n, (int)k, global, swz);
  } else {
    gemm_nvfp4_kernel<64><<<grid, 256, 0, stream>>>(pa, psa, static_cast<const uint8_t*>(packed),
                                                    static_cast<const uint8_t*>(scale), static_cast<bf16*>(out),
                                                    (int)m, (int)n, (int)k, global, swz);
  }
  return gemm_last_err();
}

int gemm_nvfp4_gs_bf16(const void* a, float a_gscale, const void* packed, const void* scale, float global, void* out,
                       int64_t m, int64_t n, int64_t k, cudaStream_t stream) {
  if (!(a_gscale > 0.f)) return static_cast<int>(cudaErrorInvalidValue);
  static void* scratch = nullptr;
  static size_t scratch_cap = 0;
  static cudaStream_t scratch_stream = nullptr;
  if (scratch_stream != nullptr && scratch_stream != stream) {
    return static_cast<int>(cudaErrorInvalidValue);  // 单流语义：换流要重新分配，调用方保证同一 stream
  }
  scratch_stream = stream;
  if (m < 0 || n < 0 || k < 0 || k == 0) return static_cast<int>(cudaErrorInvalidValue);
  if (m == 0 || n == 0) return 0;
  // A 只量化一次（就地按 block 重量化会重复 N/BN 次，4096³ 下 64×，实测正是 3.4× 慢的根因）
  const size_t need = (size_t)m * (size_t)k / 2 + (size_t)m * (size_t)k / 16;
  if (need > scratch_cap) {
    if (scratch != nullptr) cudaFree(scratch);
    if (cudaMalloc(&scratch, need) != cudaSuccess) { scratch = nullptr; scratch_cap = 0; return gemm_last_err(); }
    scratch_cap = need;
  }
  uint8_t* pa = static_cast<uint8_t*>(scratch);
  uint8_t* psa = pa + (size_t)m * (size_t)k / 2;
  // `NQ4_EXACT_RCP=1`：块缩放改精确除法 `amax/6`（A/B 用，见 docs/loops/kaon3-compile.md 第 8 轮）
  static const int exact_rq = [] {
    const char* e2 = getenv("NQ4_EXACT_RCP");
    return (e2 && e2[0] == '1') ? 1 : 0;
  }();
  {
    const long blocks = (long)m * (k / 16);
    const unsigned g = (unsigned)((blocks + 255) / 256);
    quantize_a_kernel<<<g, 256, 0, stream>>>(static_cast<const bf16*>(a), pa, psa, (int)m, (int)k, a_gscale, exact_rq);
    if (const int e = gemm_last_err(); e != 0) return e;
  }
  // 块缩放的 k-block 是 16，mma 是 k64：BK 取 128（K%128==0）否则 64（调用方保证 K%64==0）
  const dim3 grid((unsigned)((n + BN - 1) / BN), (unsigned)((m + BM - 1) / BM));
  // `NQ4_SWIZZLED=1`：A 的块缩放按 cutlass 的 padded+swizzled 布局读（吃参照的 a_s 时用）
  static const int swz = [] {
    const char* e2 = getenv("NQ4_SWIZZLED");
    return (e2 && e2[0] == '1') ? 1 : 0;
  }();
  if (k % 128 == 0) {
    gemm_nvfp4_kernel<128><<<grid, 256, 0, stream>>>(pa, psa, static_cast<const uint8_t*>(packed),
                                                     static_cast<const uint8_t*>(scale), static_cast<bf16*>(out),
                                                     (int)m, (int)n, (int)k, global, swz);
  } else if (k % 64 == 0) {
    gemm_nvfp4_kernel<64><<<grid, 256, 0, stream>>>(pa, psa, static_cast<const uint8_t*>(packed),
                                                    static_cast<const uint8_t*>(scale), static_cast<bf16*>(out),
                                                    (int)m, (int)n, (int)k, global, swz);
  } else {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return gemm_last_err();
}

int gemm_nvfp4_bf16(const void* a, const void* packed, const void* scale, float global, void* out, int64_t m, int64_t n,
                    int64_t k, cudaStream_t stream) {
  return gemm_nvfp4_gs_bf16(a, 1.f, packed, scale, global, out, m, n, k, stream);
}

}  // extern "C"
