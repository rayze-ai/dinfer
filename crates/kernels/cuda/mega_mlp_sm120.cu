// 模型无关的 dense MLP 支路 persistent megakernel（sm120，单卡）。
//
// 范围（第一版，最小可验证闭环）：把 dense MLP 支路的多个独立 kernel 压成**一次 launch**：
//   pre_ff_norm(rmsnorm) → gate GEMM(fp8) ‖ up GEMM(fp8) → gelu_tanh·mul(round2) → down GEMM(fp8) → post_ff_norm_1(rmsnorm)
// attention / MoE 仍走现有逐 kernel 路径（不在本 kernel 内）。
//
// **严格对齐 layer.rs 的真实语义**（L852-869）：
//   - `dense_gate` 与 `dense_up` 是**两个独立的 QLinear**（各自权重、各自 per-tensor scale），不是合并 [2I,H]；
//   - gelu 走 `gelu_tanh_mul_round2_bf16` 口径（gelu 先舍 bf16 再乘 up，vLLM act_and_mul 行为）；
//   - `dense_down` 吃 gelu 的 bf16 输出。
//
// 结构照 `crates/kernels/cuda/mega_sm100.cu` 的 persistent 框架；公共件在 `mega_common.cuh`。
// fp8 mma/ldmatrix 片段寻址照 `gemm_fp8_sm120.cu`（已验证正确的那份）。
#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "arch.cuh"
#include "common.cuh"
#include "kernels.h"
#include "mega_common.cuh"

namespace dinfer {
namespace mega_mlp {

using mega::kThreads;
using mega::kWarps;

enum Stage : int {
  S_NORM = 0,   // pre_ff_norm
  S_GATE = 1,   // dense_gate fp8 GEMM → gate_buf
  S_UP = 2,     // dense_up   fp8 GEMM → up_buf
  S_ACT = 3,    // gelu(gate)·up（round2）+ 量化 → act(fp8)
  S_DOWN = 4,   // dense_down fp8 GEMM → down_out
  S_POST = 5,   // post_ff_norm_1
  S_COUNT = 6,
};

/// host 填的一次 decode 步全部指针/形状（单卡，无卡间通信）。
struct Params {
  const bf16* x;      // [m, H] 输入残余流
  bf16* out;          // [m, H] 输出（post_ff_norm_1）
  const bf16* pre_w;  // [H]
  const bf16* post_w;  // [H]
  // gate: fp8 [I, H]；up: fp8 [I, H]；分别有自己的 per-tensor scale
  const uint8_t* gate_w;
  float gate_wscale;
  float gate_a_inv;
  const uint8_t* up_w;
  float up_wscale;
  float up_a_inv;
  // down: fp8 [H, I]
  const uint8_t* dn_w;
  float dn_wscale;
  float dn_a_inv;
  // 中间缓冲（调用方分配）
  bf16* norm1;    // [m, H]  bf16（对拍用）
  uint8_t* norm1_q;  // [m, H]  fp8（喂 gate/up 的 A）
  bf16* gate_buf;  // [m, I]
  bf16* up_buf;    // [m, I]
  bf16* act_bf;    // [m, I]  gelu·mul 结果
  uint8_t* act;    // [m, I]  量化后（喂 down）
  bf16* down_out;  // [m, H]
  int m, H, I;
  float eps;
  int pipe;
  int prof;
  const int* dep;
  const int* count;
  const int* start;
  int* ready;
  // split-K（每 GEMM 各一套；splits==1 时忽略 partial/sem）
  int sp_gu;        // gate/up 的 split 份数
  int sp_dn;        // down 的 split 份数
  float* part_gate; // [sp_gu][m][I]
  float* part_up;   // [sp_gu][m][I]
  float* part_dn;   // [sp_dn][m][H]
  int* sem_gate;    // [n_gu]
  int* sem_up;      // [n_gu]
  int* sem_dn;      // [n_dn]
};

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
  uint8_t* sA = smem;             // m(≤16) 行 × RS
  uint8_t* sB = smem + 16 * RS;   // BN 行 × RS

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

  auto load_b = [&](int kb) {
#pragma unroll
    for (int i = 0; i < B_PER_T; ++i) {
      const int u = tid + i * kThreads, row = u / A_CH, ch = u % A_CH;
      const int gn = n_blk * BN + row, gk = kb * BK + ch * 16;
      const uint8_t* g = B + (long)(gn < N ? gn : N - 1) * K + gk;
      mega::cp_async16_zfill(sB + row * RS + ch * 16, g, (gn < N && gk + 15 < K) ? 16 : 0);
    }
  };
  // **不能早 return**：后面有 __syncthreads()，早退会让部分线程跳过 barrier → 死锁/段错。
  auto store_a = [&](int kb) {
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
  for (int kb = kb_lo; kb < kb_hi; ++kb) {
    load_b(kb);
    store_a(kb);
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();
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

// ---------------------------------------------------------------- Dispatch
struct MlpDenseDispatch {
  using Params = mega_mlp::Params;
  static constexpr int kBN = 128;  // 每 warp 16 行 N（b0 8 + b1 8）→ 8 warp × 16 = 128

  __device__ void run(Params& p, int shard, int stage, uint8_t* smem) const {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    switch (stage) {
      case S_NORM: {
        __shared__ float red[kWarps];
        {
          const bool live = warp < p.m;
          const int row = live ? warp : 0;
          bf16* n1 = p.norm1 + (long)row * p.H;
          rmsnorm_row_warp(p.x + (long)row * p.H, p.pre_w, n1, p.H, p.eps, lane, red, warp, live);
          if (live) {
            // 同时产出喂 fp8 GEMM 的量化副本
            uint8_t* q = p.norm1_q + (long)row * p.H;
            for (int i = lane; i < p.H; i += 32) {
              const float v = to_f32(n1[i]) * p.gate_a_inv;
              q[i] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
            }
          }
        }
        return;
      }
      case S_GATE: {
        // 口径（照 gemm_fp8_sm120.cu）：out = (A/a_scale · W) * (a_scale * w_scale)。
        // A 已按 a_inv=1/a_scale 量化，故 epilogue 需乘 (w_scale * a_scale) = w_scale / a_inv。
        // split-K：shard = tile·sp_gu + split
        const int tile = shard / p.sp_gu, split = shard - tile * p.sp_gu;
        fp8_gemm_tile<kBN>(p.norm1_q, p.gate_w, p.gate_buf, p.m, p.I, p.H, tile,
                           p.gate_wscale / p.gate_a_inv, smem, split, p.sp_gu, p.part_gate, p.sem_gate);
        return;
      }
      case S_UP: {
        const int tile = shard / p.sp_gu, split = shard - tile * p.sp_gu;
        fp8_gemm_tile<kBN>(p.norm1_q, p.up_w, p.up_buf, p.m, p.I, p.H, tile, p.up_wscale / p.up_a_inv, smem,
                           split, p.sp_gu, p.part_up, p.sem_up);
        return;
      }
      case S_ACT: {
        const long n = (long)p.m * p.I;
        for (long i = tid; i < n; i += kThreads) p.act_bf[i] = gelu_mul_round2(p.gate_buf[i], p.up_buf[i]);
        __syncthreads();
        for (long i = tid; i < n; i += kThreads) {
          const float v = to_f32(p.act_bf[i]) * p.dn_a_inv;
          p.act[i] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
        }
        return;
      }
      case S_DOWN: {
        const int tile = shard / p.sp_dn, split = shard - tile * p.sp_dn;
        fp8_gemm_tile<kBN>(p.act, p.dn_w, p.down_out, p.m, p.H, p.I, tile, p.dn_wscale / p.dn_a_inv, smem,
                           split, p.sp_dn, p.part_dn, p.sem_dn);
        return;
      }
      case S_POST: {
        __shared__ float red[kWarps];
        {
          const bool live = warp < p.m;
          const int row = live ? warp : 0;
          rmsnorm_row_warp(p.down_out + (long)row * p.H, p.post_w, p.out + (long)row * p.H, p.H, p.eps, lane, red,
                           warp, live);
        }
        return;
      }
    }
  }
};

}  // namespace mega_mlp
}  // namespace dinfer

// ---------------------------------------------------------------- host launch（C ABI）
extern "C" {

/// 一次 decode 步的 dense MLP 支路：pre_norm → gate ‖ up → gelu·mul → down → post_norm。
/// `dep/count/start/ready` 由调用方给：count[S_COUNT] 每阶段任务数、start 为前缀和、ready 长度 S_COUNT（**须先清零**）。
/// `n_gu` = gate/up 各自分几个 N tile（= ceil(I / BN)）；`n_dn` = down 的 N tile（= ceil(H / BN)）。
/// 返回 cudaError_t 数值（0 = 成功）。
int mega_mlp_dense_branch(const void* x, void* out, const void* pre_w, const void* post_w, const void* gate_w,
                         float gate_wscale, float gate_a_inv, const void* up_w, float up_wscale, float up_a_inv,
                         const void* dn_w, float dn_wscale, float dn_a_inv, void* norm1, void* norm1_q, void* gate_buf, void* up_buf,
                         void* act_bf, void* act, void* down_out, int m, int H, int I, float eps, int n_gu, int n_dn,
                         int* dep, int* count, int* start, int* ready, int* dep2, int pipe, int prof,
                         unsigned long long* wait_cyc, unsigned long long* exec_cyc,
                         int sp_gu, int sp_dn, void* part_gate, void* part_up, void* part_dn,
                         int* sem_gate, int* sem_up, int* sem_dn, int ctas, cudaStream_t stream) {
  using namespace dinfer;
  using namespace dinfer::mega_mlp;
  if (m <= 0 || m > 8 || H <= 0 || I <= 0) return (int)cudaErrorInvalidValue;
  if (sp_gu < 1) sp_gu = 1;
  if (sp_dn < 1) sp_dn = 1;
  constexpr int kBN = MlpDenseDispatch::kBN;
  // smem：sA(16×RS) + sB(BN×RS)
  const size_t smem = (size_t)(16 + kBN) * (128 + 16);

  // split-K 的 sem 须每次 launch 前清零（partial 不用：每 split 全写自己的区）。
  if (sp_gu > 1) {
    if (cudaMemsetAsync(sem_gate, 0, sizeof(int) * n_gu, stream) != cudaSuccess ||
        cudaMemsetAsync(sem_up, 0, sizeof(int) * n_gu, stream) != cudaSuccess)
      return (int)cudaGetLastError();
  }
  if (sp_dn > 1) {
    if (cudaMemsetAsync(sem_dn, 0, sizeof(int) * n_dn, stream) != cudaSuccess) return (int)cudaGetLastError();
  }

  // **注意**：dep/count/start/ready 是**设备指针**（Rust 侧 st.ptr_mut 传入）。表先在 host 本地算，
  // 再 cudaMemcpy 拷过去；直接写 *dep 会往设备内存解引用 → SIGSEGV。
  int h_count[S_COUNT], h_dep[S_COUNT], h_start[S_COUNT];
  h_count[S_NORM] = 1;
  h_count[S_GATE] = n_gu * sp_gu;  // 每 tile 切 sp_gu 份 split-K
  h_count[S_UP] = n_gu * sp_gu;
  h_count[S_ACT] = 1;
  h_count[S_DOWN] = n_dn * sp_dn;
  h_count[S_POST] = 1;
  int h_dep2[S_COUNT];
  for (int i = 0; i < S_COUNT; ++i) h_dep2[i] = -1;
  h_dep[S_NORM] = -1;
  h_dep[S_GATE] = S_NORM;   // gate / up 都只依赖 norm → **并行**（解开原 gate→up 串行）
  h_dep[S_UP] = S_NORM;
  h_dep[S_ACT] = S_GATE;    // act 等 gate(dep) 与 up(dep2) 都完成
  h_dep2[S_ACT] = S_UP;
  h_dep[S_DOWN] = S_ACT;
  h_dep[S_POST] = S_DOWN;
  int acc = 0;
  for (int s = 0; s < S_COUNT; ++s) {
    h_start[s] = acc;
    acc += h_count[s];
  }
  if (cudaMemcpyAsync(count, h_count, sizeof(h_count), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemcpyAsync(dep, h_dep, sizeof(h_dep), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemcpyAsync(start, h_start, sizeof(h_start), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      (dep2 && cudaMemcpyAsync(dep2, h_dep2, sizeof(h_dep2), cudaMemcpyHostToDevice, stream) != cudaSuccess))
    return (int)cudaGetLastError();

  Params p{};
  p.x = static_cast<const bf16*>(x);
  p.out = static_cast<bf16*>(out);
  p.pre_w = static_cast<const bf16*>(pre_w);
  p.post_w = static_cast<const bf16*>(post_w);
  p.gate_w = static_cast<const uint8_t*>(gate_w);
  p.gate_wscale = gate_wscale;
  p.gate_a_inv = gate_a_inv;
  p.up_w = static_cast<const uint8_t*>(up_w);
  p.up_wscale = up_wscale;
  p.up_a_inv = up_a_inv;
  p.dn_w = static_cast<const uint8_t*>(dn_w);
  p.dn_wscale = dn_wscale;
  p.dn_a_inv = dn_a_inv;
  p.norm1 = static_cast<bf16*>(norm1);
  p.norm1_q = static_cast<uint8_t*>(norm1_q);
  p.gate_buf = static_cast<bf16*>(gate_buf);
  p.up_buf = static_cast<bf16*>(up_buf);
  p.act_bf = static_cast<bf16*>(act_bf);
  p.act = static_cast<uint8_t*>(act);
  p.down_out = static_cast<bf16*>(down_out);
  p.m = m;
  p.H = H;
  p.I = I;
  p.eps = eps;
  p.pipe = pipe > 0 ? pipe : 2;
  p.prof = prof;
  p.dep = dep;
  p.count = count;
  p.start = start;
  p.ready = ready;
  p.sp_gu = sp_gu;
  p.sp_dn = sp_dn;
  p.part_gate = static_cast<float*>(part_gate);
  p.part_up = static_cast<float*>(part_up);
  p.part_dn = static_cast<float*>(part_dn);
  p.sem_gate = sem_gate;
  p.sem_up = sem_up;
  p.sem_dn = sem_dn;

  int dev = 0, sms = 1, max_smem = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  if ((int)smem > max_smem) return (int)cudaErrorInvalidValue;

  // ctas_per_sm：ctas>=2 用 OCC=2 变体（launch_bounds 放宽寄存器上限，允许 2 CTA 常驻），否则 OCC=1。
  // 实际常驻数再受 occupancy 查询约束（smem / 寄存器）。split-K 后任务数 > SM，多一波常驻能减少自旋空等。
  const bool want2 = ctas >= 2;
  auto kern1 = mega::mega_kernel<MlpDenseDispatch, 1>;
  auto kern2 = mega::mega_kernel<MlpDenseDispatch, 2>;
  const void* kptr = want2 ? (const void*)kern2 : (const void*)kern1;
  static thread_local int smem_set1 = 0, smem_set2 = 0;
  int& smem_set = want2 ? smem_set2 : smem_set1;
  if (smem_set < (int)smem) {
    if (const int rc = (int)cudaFuncSetAttribute(kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem))
      return rc;
    smem_set = (int)smem;
  }
  int occ = 0;
  if (const int rc = (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kptr, mega::kThreads, smem)) return rc;
  if (occ <= 0) return (int)cudaErrorInvalidValue;
  const int occ_cap = want2 ? 2 : 1;
  if (occ > occ_cap) occ = occ_cap;
  int grid = sms * occ;
  if (grid > acc) grid = acc;
  const mega::StagePlan sp{dep, count, start, ready, prof, wait_cyc, exec_cyc, dep2};
  if (want2)
    kern2<<<grid, mega::kThreads, smem, stream>>>(sp, p, MlpDenseDispatch{}, acc, S_COUNT, 0);
  else
    kern1<<<grid, mega::kThreads, smem, stream>>>(sp, p, MlpDenseDispatch{}, acc, S_COUNT, 0);
  return (int)cudaGetLastError();
}

}  // extern "C"

