// FP8 per-tensor GEMM（sm_120 专用）：out[m,n] = (e4m3(a[m,k] / a_scale) · w[n,k]^T) * (a_scale * w_scale)
//   a : bf16 激活 [m,k]，载入时转 e4m3（饱和到 ±448，bf16 超出即钳）
//   w : [n,k] 的 e4m3 字节（每字节一个 fp8），w_scale 是 per-tensor 标量
// 结构照 models/kaon3/cuda/fp8_v2.cu（实测 292 TF/s）：block 64x128x64、8 warp = 2(M)x4(N)、
// 每 warp 32x32（2x4 个 mma tile）、ldmatrix.x4(A)/.x2(B)、smem padding 16 B、cp.async.cg 双缓冲。
// A 侧要 bf16→e4m3 转换，走"寄存器转换 + 普通 STS"；B 侧是字节流，仍是 cp.async。
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

#include "common.cuh"

using namespace dinfer;  // bf16 / to_f32 / to_bf16（common.cuh 放在 namespace 里）

namespace {

constexpr int BM = 64, BN = 128, BK = 64, PAD = 16, RS = BK + PAD;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void ldm_x4(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ void ldm_x2(uint32_t (&r)[2], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(a));
}
__device__ __forceinline__ void mma_fp8(const uint32_t* a, const uint32_t* b, float* d) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
               "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void cp16(uint32_t s, const void* g, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(g), "r"(bytes));
}

/// bf16 → e4m3 字节（每线程 16 个：2 个 uint4 读，1 个 uint4 写）。
/// 对拍参考侧要逐位复现：饱和 ±448 + 就近偶数舍入，与 __nv_cvt_float_to_fp8 一致。
__device__ __forceinline__ uint4 cvt16(const bf16* src, bool valid, float inv) {
  uint32_t w[4];
  if (!valid) {
    w[0] = w[1] = w[2] = w[3] = 0u;  // e4m3 的 +0 编码是 0x00
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint32_t v = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float x = to_f32(src[i * 4 + j]) * inv;  // 静态激活 scale（vLLM：x * (1/input_scale)）
        v |= static_cast<uint32_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3)) << (8 * j);
      }
      w[i] = v;
    }
  }
  return make_uint4(w[0], w[1], w[2], w[3]);
}

// A 已是 e4m3 字节（预量化）：跳过 cvt16，直接读 16 B。与 bf16 版本共用同一套 smem / mma / 累加顺序，
// 因此两者的输出在"同一个 A 的数值"上逐位相同——这是与 FlashInfer 逐位对拍所需的入口。
__global__ void __launch_bounds__(256) gemm_fp8_pre_kernel(const uint8_t* __restrict__ A,
                                                           const uint8_t* __restrict__ B, bf16* __restrict__ D,
                                                           int M, int N, int K, float wscale) {
  __shared__ uint8_t sA[2][BM * RS];
  __shared__ uint8_t sB[2][BN * RS];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 2, wn = warp & 3;
  const int m_blk = blockIdx.y * BM, n_blk = blockIdx.x * BN;
  float acc[2][4][4] = {};
  const int nk = (K + BK - 1) / BK;

  // A：每线程 16 个 e4m3 字节（row = tid>>2，chunk = tid&3）
  const int a_row = tid >> 2, a_ch = tid & 3;
  auto conv_a = [&](int kb) -> uint4 {
    const int gm = m_blk + a_row;
    const int gk = kb * BK + a_ch * 16;
    // 越界补 0：e4m3 的 +0 编码是 0x00，与 bf16 版本 valid=false 时的填法一致
    if (gm < M && gk + 15 < K) {
      return *reinterpret_cast<const uint4*>(A + static_cast<size_t>(gm) * K + gk);
    }
    return make_uint4(0u, 0u, 0u, 0u);
  };
  auto store_a = [&](int st, const uint4& v) {
    *reinterpret_cast<uint4*>(&sA[st][a_row * RS + a_ch * 16]) = v;
  };
  // B：每线程 2 个 16 B 块
  auto load_b = [&](int st, int kb) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int u = tid + i * 256, row = u >> 2, ch = u & 3;
      const int gn = n_blk + row, gk = kb * BK + ch * 16;
      const uint8_t* g = B + static_cast<size_t>(gn < N ? gn : N - 1) * K + gk;
      cp16(smem_u32(&sB[st][row * RS + ch * 16]), g, (gn < N && gk + 15 < K) ? 16 : 0);
    }
    asm volatile("cp.async.commit_group;\n");
  };

  load_b(0, 0);
  store_a(0, conv_a(0));
  for (int kb = 0; kb < nk; ++kb) {
    const int cur = kb & 1, nxt = cur ^ 1;
    // 下一块：B 走 cp.async 异步预取，A 在寄存器里转好，等本轮算完再落地
    uint4 anext = make_uint4(0, 0, 0, 0);
    if (kb + 1 < nk) {
      load_b(nxt, kb + 1);
      anext = conv_a(kb + 1);
    } else {
      asm volatile("cp.async.commit_group;\n");
    }
    asm volatile("cp.async.wait_group 1;\n");
    __syncthreads();
    // 上一轮 __syncthreads 之后没人再读 sA[nxt] / sB[nxt]，安全覆盖
    if (kb + 1 < nk) store_a(nxt, anext);
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const int kb0 = ks * 32;
      uint32_t a[2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        const int r = lane & 15, c = (lane >= 16) ? 16 : 0;
        ldm_x4(a[mt], smem_u32(&sA[cur][(wm * 32 + mt * 16 + r) * RS + kb0 + c]));
      }
      uint32_t b[4][2];
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int rb = lane & 7, cb = (lane & 8) ? 16 : 0;
        ldm_x2(b[nt], smem_u32(&sB[cur][(wn * 32 + nt * 8 + rb) * RS + kb0 + cb]));
      }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) mma_fp8(a[mt], b[nt], acc[mt][nt]);
    }
    __syncthreads();
  }
  const int r0 = lane >> 2, cg = (lane & 3) * 2;
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int g0 = m_blk + wm * 32 + mt * 16 + r0, g1 = g0 + 8, gn = n_blk + wn * 32 + nt * 8 + cg;
      if (g0 < M) {
        if (gn < N) D[static_cast<size_t>(g0) * N + gn] = to_bf16(acc[mt][nt][0] * wscale);
        if (gn + 1 < N) D[static_cast<size_t>(g0) * N + gn + 1] = to_bf16(acc[mt][nt][1] * wscale);
      }
      if (g1 < M) {
        if (gn < N) D[static_cast<size_t>(g1) * N + gn] = to_bf16(acc[mt][nt][2] * wscale);
        if (gn + 1 < N) D[static_cast<size_t>(g1) * N + gn + 1] = to_bf16(acc[mt][nt][3] * wscale);
      }
    }
}

int gemm_last_err() {
  const cudaError_t e = cudaGetLastError();
  return static_cast<int>(e);
}

}  // namespace

extern "C" {

// 入口：返回 cudaError_t 数值（0 = 成功），调用方非 0 即报错。
// a_scale = 静态 per-tensor 激活 scale（checkpoint 的 input_scale）：A 先乘 1/a_scale 再饱和转 e4m3，输出乘 a_scale*w_scale。
// 不用它（a_scale=1）时超过 448 的激活被截断、小于 2^-6 的落进次正规区；Kaon-V3 的 input_scale 实测 0.03 到 3.9。
int gemm_fp8_static_bf16(const void* a, float a_scale, const void* w, float w_scale, void* out, int64_t m, int64_t n,
                         int64_t k, cudaStream_t stream) {
  if (!(a_scale > 0.f)) return static_cast<int>(cudaErrorInvalidValue);
  if (m < 0 || n < 0 || k < 0) return static_cast<int>(cudaErrorInvalidValue);
  if (m == 0 || n == 0) return 0;
  // 形状由 host 侧校验过；这里只挡会越界读的（K 方向按 ceil 分块补零，任意 K 都安全）
  if (k == 0) return static_cast<int>(cudaErrorInvalidValue);
  if (n > 0x7fffffff || m > 0x7fffffff || k > 0x7fffffff) return static_cast<int>(cudaErrorInvalidValue);
  const dim3 grid((unsigned)((n + BN - 1) / BN), (unsigned)((m + BM - 1) / BM));
  gemm_fp8_kernel<<<grid, 256, 0, stream>>>(static_cast<const bf16*>(a), static_cast<const uint8_t*>(w),
                                            static_cast<bf16*>(out), (int)m, (int)n, (int)k, w_scale * a_scale,
                                            1.f / a_scale);
  return gemm_last_err();
}

int gemm_fp8_bf16(const void* a, const void* w, float w_scale, void* out, int64_t m, int64_t n, int64_t k,
                  cudaStream_t stream) {
  return gemm_fp8_static_bf16(a, 1.f, w, w_scale, out, m, n, k, stream);
}

// 预量化入口：a 已是 e4m3 字节 [m,k]（调用方做过 x/a_scale + RNE），输出仍需乘 a_scale*w_scale，
// 所以 a_scale 以普通 float 参数传入（不参与量化）。用于与外部实现（vLLM/FlashInfer）逐位对拍。
int gemm_fp8_prequant_bf16(const void* a, float a_scale, const void* w, float w_scale, void* out, int64_t m,
                           int64_t n, int64_t k, cudaStream_t stream) {
  if (m < 0 || n < 0 || k < 0) return static_cast<int>(cudaErrorInvalidValue);
  if (m == 0 || n == 0) return 0;
  if (k == 0) return static_cast<int>(cudaErrorInvalidValue);
  if (n > 0x7fffffff || m > 0x7fffffff || k > 0x7fffffff) return static_cast<int>(cudaErrorInvalidValue);
  const dim3 grid((unsigned)((n + BN - 1) / BN), (unsigned)((m + BM - 1) / BM));
  gemm_fp8_pre_kernel<<<grid, 256, 0, stream>>>(static_cast<const uint8_t*>(a), static_cast<const uint8_t*>(w),
                                                static_cast<bf16*>(out), (int)m, (int)n, (int)k,
                                                w_scale * a_scale);
  return gemm_last_err();
}

}  // extern "C"
