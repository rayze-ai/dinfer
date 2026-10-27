// 共享 device 原语：fp8 mma/ldmatrix、fp8_gemm_tile（含 split-K）、rmsnorm_row_warp、gelu·mul(round2)。
// 从 mega_mlp_sm120.cu 抽出（kaon3-mega B2），供 dense MLP 与整层 megakernel 复用 —— 两边走同一份数值，保证逐位一致。
#pragma once
#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include "common.cuh"
#include "mega_common.cuh"

namespace dinfer {
namespace mega_mlp {

using mega::kThreads;
using mega::kWarps;

// ---------------------------------------------------------------- fp8 原语（照 gemm_fp8_sm120.cu）
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
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
__device__ __forceinline__ float gelu_tanh(float x) {
  constexpr float k0 = 0.7978845608028654f;
  return 0.5f * x * (1.f + tanhf(k0 * (x + 0.044715f * x * x * x)));
}
/// vLLM `act_and_mul` 口径：gelu 先舍 bf16，再与 up 相乘、再舍 bf16。
__device__ __forceinline__ bf16 gelu_mul_round2(bf16 g, bf16 u) {
  const float y = to_f32(to_bf16(gelu_tanh(to_f32(g))));
  return to_bf16(y * to_f32(u));
}

/// 一行 rmsnorm（一 warp 一行）：out = bf16(x·rsqrt(mean(x²)+eps)·w)。
/// 与 `rmsnorm_into`/`norm.cu` 同口径（f32 平方和、末尾舍一次）。
__device__ __forceinline__ void rmsnorm_row_warp(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                                 bf16* __restrict__ o, int H, float eps, int lane, float* red,
                                                 int widx, bool store) {
  float ss = 0.f;
  for (int i = lane; i < H; i += 32) {
    const float v = to_f32(x[i]);
    ss += v * v;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
  if (lane == 0) red[widx] = ss;
  __syncthreads();
  const float inv = rsqrtf(red[widx] / (float)H + eps);
  if (!store) return;  // 占位 warp（行号 ≥ m）：只参与 barrier，不写
  for (int i = lane; i < H; i += 32) o[i] = to_bf16(to_f32(x[i]) * inv * to_f32(w[i]));
}

// ---------------------------------------------------------------- fp8 GEMM tile（M≤16，一个 N tile；可选 split-K）
// 片段寻址严格照 gemm_fp8_sm120.cu：A 用 ldm_x4（m16k32，每 lane 4×u32），B 用 ldm_x2（n8k32，每 lane 2×u32）。
// BN：本 tile 的 N 行数；8 warp 分 N 方向。
// split-K（splits>1）：本 CTA 只算 K 的 [k0, k1) 段（按 BK 块均分），f32 partial 写 `partial[split][m][N]`，
//   每 tile 一个 `sem`，末到 CTA（atomicAdd==splits-1）把 splits 份 partial f32 求和、乘 wscale、写 bf16。
//   splits==1 时走原路径（不碰 partial/sem），**与改动前逐位一致**。
template <int BN>
__device__ __forceinline__ void fp8_gemm_tile(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                                              bf16* __restrict__ D, int m, int N, int K, int n_blk, float wscale,
                                              uint8_t* smem, int split, int splits, float* __restrict__ partial,
                                              int* __restrict__ sem) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
#if defined(MEGA_DIAG_NOGEMM)  // 诊断开关：短路 GEMM（二分崩溃来源）
  // 诊断模式：不碰 smem、不算，只写 0，用以二分定位 SIGSEGV 来源
  const int nrow0d = warp * (BN / kWarps);
  const int r0d = lane >> 2, r1d = r0d + 8, cgd = (lane & 3) * 2;
  for (int i = 0; i < 2; ++i) {
    const int g0 = n_blk * BN + nrow0d + cgd + i * 8;
    if (g0 < N) {
      if (r0d < m) D[(long)r0d * N + g0] = to_bf16(0.f);
      if (r1d < m) D[(long)r1d * N + g0] = to_bf16(0.f);
    }
    if (g0 + 1 < N) {
      if (r0d < m) D[(long)r0d * N + g0 + 1] = to_bf16(0.f);
      if (r1d < m) D[(long)r1d * N + g0 + 1] = to_bf16(0.f);
    }
  }
  return;
#endif
  constexpr int BK = 128;
  constexpr int RS = BK + 16;
  // 双缓冲软件流水：两份 sA/sB（buf 0/1），算 kb 时预取 kb+1，藏访存延迟。smem 布局 = [sA0 sB0 sA1 sB1]。
  constexpr int TILE_SMEM = (16 + BN) * RS;  // 单份 = sA(16×RS)+sB(BN×RS)
  uint8_t* sA_buf[2] = {smem, smem + TILE_SMEM};
  uint8_t* sB_buf[2] = {smem + 16 * RS, smem + TILE_SMEM + 16 * RS};

  float acc[2][4] = {};  // 本 warp 的 2 个 n8 片段 × 4 累加器
  const int nk = (K + BK - 1) / BK;
  // split-K：按 BK 块把 nk 均分给 splits 份，本 CTA 只跑 [kb_lo, kb_hi)。splits==1 → 全 K。
  const int per = (nk + splits - 1) / splits;
  const int kb_lo = split * per;
  const int kb_hi = min(kb_lo + per, nk);
  // A：只有 16 行 × (BK/16) 个 16B 块 = 16*8 = 128 个槽（BK=128）。256 线程只需前 128 个参与，
  // 否则 a_row = tid>>2 会到 63 行 → 写出 sA（16 行）→ 段错误。
  constexpr int A_CH = BK / 16;                    // 每行块数
  constexpr int A_SLOTS = 16 * A_CH;               // 总槽数
  constexpr int B_PER_T = BN * A_CH / kThreads;

  // 发出 kb 的 B（cp.async）+ A（普通写）到 buf。只 active（前 A_SLOTS 线程写 A；全员发 B）。
  auto load_b = [&](int kb, uint8_t* sB) {
#pragma unroll
    for (int i = 0; i < B_PER_T; ++i) {
      const int u = tid + i * kThreads, row = u / A_CH, ch = u % A_CH;
      const int gn = n_blk * BN + row, gk = kb * BK + ch * 16;
      const uint8_t* g = B + (long)(gn < N ? gn : N - 1) * K + gk;
      mega::cp_async16_zfill(sB + row * RS + ch * 16, g, (gn < N && gk + 15 < K) ? 16 : 0);
    }
  };
  auto store_a = [&](int kb, uint8_t* sA) {
    if (tid < A_SLOTS) {
      const int ar_ = tid / A_CH, ac_ = tid % A_CH;
      const int gm = ar_, gk = kb * BK + ac_ * 16;
      const bool ok = gm < m && gk + 15 < K;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (ok) v = *reinterpret_cast<const uint4*>(A + (long)gm * K + gk);
      *reinterpret_cast<uint4*>(sA + ar_ * RS + ac_ * 16) = v;
    }
  };

  const int nrow0 = warp * (BN / kWarps);  // 每 warp 负责 BN/8 行 N
  const int nkb = kb_hi - kb_lo;
  if (nkb > 0) {
    // 序幕：发出第一块到 buf0。
    load_b(kb_lo, sB_buf[0]);
    store_a(kb_lo, sA_buf[0]);
    asm volatile("cp.async.commit_group;\n");
  }
  for (int j = 0; j < nkb; ++j) {
    const int kb = kb_lo + j;
    const int cur = j & 1, nxt = (j + 1) & 1;
    // 先发出下一块（kb+1）到另一 buf——它的访存将与本块 mma 重叠。
    if (j + 1 < nkb) {
      load_b(kb + 1, sB_buf[nxt]);
      store_a(kb + 1, sA_buf[nxt]);
      asm volatile("cp.async.commit_group;\n");
      asm volatile("cp.async.wait_group 1;\n");  // 等到只剩 1 组在飞（= 本块已落地，下一块可仍在飞）
    } else {
      asm volatile("cp.async.wait_group 0;\n");   // 最后一块：等全部落地
    }
    __syncthreads();
    uint8_t* sA = sA_buf[cur];
    uint8_t* sB = sB_buf[cur];
#pragma unroll
    for (int kc = 0; kc < BK / 32; ++kc) {
      const int kb0 = kc * 32;
      uint32_t a[4], b0[2], b1[2];
      const int ar = lane & 15, ac = (lane >= 16) ? 16 : 0;
      ldm_x4(a, smem_u32(sA + ar * RS + kb0 + ac));
      const int br = lane & 7, bc = (lane & 8) ? 16 : 0;
      ldm_x2(b0, smem_u32(sB + (nrow0 + br) * RS + kb0 + bc));
      ldm_x2(b1, smem_u32(sB + (nrow0 + 8 + br) * RS + kb0 + bc));
      mma_fp8(a, b0, acc[0]);
      mma_fp8(a, b1, acc[1]);
    }
    __syncthreads();
  }

  // 尾声：r0=lane>>2 / r1=r0+8，列 = nrow0 + (lane&3)*2 + {0,1}（第 i 片段 +8）
  const int r0 = lane >> 2, r1 = r0 + 8, cg = (lane & 3) * 2;
  if (splits == 1) {
    // 原路径：直接乘 wscale 写 bf16（**与改动前逐位一致**）
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int ncol = nrow0 + cg + i * 8;  // 该片段的起始列（偶）
      const int g0 = n_blk * BN + ncol, g1 = g0 + 1;
      if (g0 < N) {
        if (r0 < m) D[(long)r0 * N + g0] = to_bf16(acc[i][0] * wscale);
        if (r1 < m) D[(long)r1 * N + g0] = to_bf16(acc[i][2] * wscale);
      }
      if (g1 < N) {
        if (r0 < m) D[(long)r0 * N + g1] = to_bf16(acc[i][1] * wscale);
        if (r1 < m) D[(long)r1 * N + g1] = to_bf16(acc[i][3] * wscale);
      }
    }
    return;
  }
  // split-K：本 split 的 f32 partial 写 partial[split][m][N]（不乘 wscale，归约后统一乘）
  float* ps = partial + (long)split * m * N;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int ncol = nrow0 + cg + i * 8;
    const int g0 = n_blk * BN + ncol, g1 = g0 + 1;
    if (g0 < N) {
      if (r0 < m) ps[(long)r0 * N + g0] = acc[i][0];
      if (r1 < m) ps[(long)r1 * N + g0] = acc[i][2];
    }
    if (g1 < N) {
      if (r0 < m) ps[(long)r0 * N + g1] = acc[i][1];
      if (r1 < m) ps[(long)r1 * N + g1] = acc[i][3];
    }
  }
  // 末到 CTA 做归约：threadfence 保 partial 全局可见 → atomicAdd 到本 tile 的 sem
  __shared__ int s_last;
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    s_last = (atomicAdd(sem + n_blk, 1) == splits - 1) ? 1 : 0;
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  // 本 tile 的 N 列范围 [n0, n1)，按 split 顺序求和（0..splits-1）→ 乘 wscale → bf16
  const int n0 = n_blk * BN, n1 = min(n0 + BN, N);
  const int ncols = n1 - n0;
  for (int idx = tid; idx < m * ncols; idx += kThreads) {
    const int r = idx / ncols, c = idx - r * ncols;
    float s = 0.f;
    for (int sp = 0; sp < splits; ++sp) s += __ldcg(partial + (long)sp * m * N + (long)r * N + n0 + c);
    D[(long)r * N + n0 + c] = to_bf16(s * wscale);
  }
}

}  // namespace mega_mlp
}  // namespace dinfer
