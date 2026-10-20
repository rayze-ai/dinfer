// KDA chunk 形式 prefill（Kimi Delta Attention）。模型无关：只按形状参数，BT 由 host 选（模板参数）。
//
// 规格与代数口径见 docs/loops/k3-kda.md「v2 规格」。与 fla 的 chunk_gated_delta_rule 同式，
// 已用 scripts/verify/kda_chunk_probe.py 与逐 token 顺序递归核对：
//   gc   = 块内 inclusive cumsum(gate) · 1/ln2            （下游一律 exp2）
//   A[i,j]  = β_i (k_i·k_j) exp2(gc_i − gc_j),  j <  i     （严格下三角，只用它求 M）
//   Aq[i,j] = (q_i·k_j)      exp2(gc_i − gc_j),  j <= i    （含对角线）
//   M = (I+A)⁻¹（单位下三角，前代）；u = M(β⊙v)；w = M(β⊙k⊙exp2(gc))；kg = k·exp2(gc_last − gc)
//   v' = u − w·h_prev；o = (q⊙exp2(gc))·h_prev + Aq·v'；h_next = h_prev·exp2(gc_last) + v'ᵀ·kg
//
// 三个 kernel（对应 fla 的 prep / wy / (h + o) 三段）：
//   chunk_prep  q/k 归一化（q 乘 scale）、beta、块内 cumsum → gc
//   chunk_wy    A/Aq → M → u / w / kg（A 只在本 kernel 内用；Aq 写回给输出段）
//   chunk_ho    跨 chunk 串行扫 h + 立即算 o（省掉 h 的 [NC,H,V,K] 全局中间量）
//
// 数值口径：全部 fp32（`docs/loops/k3-kda.md` 精度预算里最严的一档）；bf16 mma 是 v3 的提速项。
#include "common.cuh"
#include "kernels.h"

#include <stdint.h>

using namespace dinfer;

namespace {

// 数学常量（不是调优参数）：1/ln2，把自然对数空间的 cumsum 变成 exp2 的指数。
constexpr float kRcpLn2 = 1.4426950408889634f;

__device__ __forceinline__ float sigmoid_f(float x) { return 1.f / (1.f + __expf(-x)); }

// ================================================================ prep
// 每 block 一个 (序列, 头)。线程按 head_dim 并行，时间维串行，块内 cumsum 在寄存器里累加。
__global__ void prep_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k,
                            const bf16* __restrict__ raw_g, const float* __restrict__ raw_beta,
                            const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                            bf16* __restrict__ qn, bf16* __restrict__ kn, float* __restrict__ beta,
                            float* __restrict__ gc, const int* __restrict__ cu_seqlens, int heads, int K, int BT,
                            float scale, float lower_bound, int use_lower_bound) {
  const int h = blockIdx.x;
  const int seq = blockIdx.y;
  const int half = blockIdx.z;  // 0 = 归一化 q（并写 beta/gc），1 = 归一化 k；两块并行各做一次归约
  const int tid = threadIdx.x;
  const int bos = cu_seqlens[seq], eos = cu_seqlens[seq + 1];
  const int len = eos - bos;
  if (len <= 0) return;

  const float a = __expf(a_log[h]);
  const int stride = heads * K;
  float acc = 0.f;  // 本线程负责的单个 k 维在块内的 log-gate 累加

  for (int t = 0; t < len; ++t) {
    const long row = (long)(bos + t) * stride + (long)h * K;
    float v = 0.f;
    if (tid < K) v = to_f32(half ? k[row + tid] : q[row + tid]);
    const float ss = block_reduce_sum<256>(v * v);
    __syncthreads();
    if (tid < K) {
      const float n = v / sqrtf(ss + 1e-6f);
      if (half) {
        kn[row + tid] = to_bf16(n);
      } else {
        qn[row + tid] = to_bf16(n * scale);
        const float xv = to_f32(raw_g[row + tid]) + dt_bias[(long)h * K + tid];
        acc += use_lower_bound ? lower_bound * sigmoid_f(a * xv)
                               : -a * (xv > 20.f ? xv : __logf(1.f + __expf(xv)));
        gc[row + tid] = acc * kRcpLn2;
      }
    }
    if (tid == 0 && half == 0) beta[(long)(bos + t) * heads + h] = sigmoid_f(raw_beta[(long)(bos + t) * heads + h]);
    if ((t + 1) % BT == 0) acc = 0.f;  // 块边界重置（块内 cumsum）
  }
}

// ================================================================ wy
// 每 block 一个 (chunk, 头)，256 线程。K 分两半累加到寄存器，A/Aq 各 16 个/线程。
// 求解 M 用单位下三角前代；u/w/kg 用 M 左乘。
template <int BT, int BC, int BK>
__global__ void wy_kernel(const bf16* __restrict__ qn, const bf16* __restrict__ kn, const bf16* __restrict__ v,
                          const float* __restrict__ gc, const float* __restrict__ beta, float* __restrict__ aq_out,
                          float* __restrict__ u_out, float* __restrict__ w_out, float* __restrict__ kg_out,
                          const int* __restrict__ cu_seqlens, const int* __restrict__ chunk_tab, int heads, int K,
                          int V) {
  const int c = blockIdx.x;
  const int h = blockIdx.y;
  const int tid = threadIdx.x;
  const int nb = blockDim.x;
  constexpr int PAIRS = BT * BT / 256;  // 每线程负责的 (i,j) 对数
  static_assert(BT * BT % 256 == 0, "BT*BT 必须是线程数的整数倍");

  const int seq = chunk_tab[2 * c], ci = chunk_tab[2 * c + 1];
  const int bos = cu_seqlens[seq], eos = cu_seqlens[seq + 1];
  const int len = min(BT, eos - bos - ci * BT);
  const long row0 = bos + (long)ci * BT;
  const int stride = heads * K;
  const long hoff = (long)h * K;

  extern __shared__ float sm[];
  float* sk = sm;              // [BT][BK] 本 K 半段的 kn
  float* sg = sk + BT * BK;    // [BT][BK] 本 K 半段的 gc
  float* sq = sg + BT * BK;    // [BT][BK] 本 K 半段的 qn
  float* sA = sq + BT * BK;    // [BT][BT]  A（求 M 用）
  float* sM = sA + BT * BT;    // [BT][BT]  M / 后续 u、w 的左乘矩阵
  float* sfi = sM + BT * BT;   // [BC][BK] 行衰减 exp2(gc_i - gc_ref)
  // 列衰减 exp2(gc_ref - gc_j) 需要 [BT][BK]，与 sA（[BT][BT]，同为 BT*BT 个 float）同尺寸；
  // A 只在 A/Aq 循环结束后才写，这里直接借用 sA 当 sfj，省 16 KB shared（sm86 上限 99 KB）。
  float* sfj = sA;

  // ---- 1) A / Aq：按 K 分半累加到寄存器。
  // 衰减因子按子块预计算（fi 每组 BC 行、fj 全 BT 行），内层只剩乘加——
  // 与 fla `chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter` 同构，避免逐 (i,j,k) 做 exp2。
  // 每线程负责 PAIRS 个 pair（n 为全局序号，寄存器跨 K 分片累加）。
  // 256 线程下每个 n 覆盖 256/BT 行，子块 i0 对应 n ∈ [i0/RPN, (i0+BC)/RPN)。
  constexpr int RPN = 256 / BT;               // 每个 n 覆盖的行数
  constexpr int NPS = BC / RPN;               // 每个子块涉及的 n 个数
  float accA[PAIRS], accQ[PAIRS];
  for (int n = 0; n < PAIRS; ++n) { accA[n] = 0.f; accQ[n] = 0.f; }
  for (int k0 = 0; k0 < K; k0 += BK) {
    const int bw = min(BK, K - k0);
    for (int idx = tid; idx < BT * BK; idx += nb) {
      const int j = idx / BK, kk = idx % BK;
      const bool ok = j < len && kk < bw;
      const long r = (long)(row0 + j) * stride + hoff + k0 + kk;
      sk[idx] = ok ? to_f32(kn[r]) : 0.f;
      sq[idx] = ok ? to_f32(qn[r]) : 0.f;
      sg[idx] = ok ? gc[r] : 0.f;
    }
    __syncthreads();
    for (int i0 = 0; i0 < BT; i0 += BC) {
      // 参考行取子块首行（不足 len 时退到最后一个有效行，保证指数不爆）
      const int ref = min(i0, len - 1);
      for (int idx = tid; idx < BC * BK; idx += nb) {
        const int i = idx / BK, kk = idx % BK;
        const int ti = i0 + i;
        sfi[idx] = (ti < len && kk < bw) ? exp2f(sg[ti * BK + kk] - sg[ref * BK + kk]) : 0.f;
      }
      for (int idx = tid; idx < BT * BK; idx += nb) {
        const int j = idx / BK, kk = idx % BK;
        sfj[idx] = (j < len && kk < bw) ? exp2f(sg[ref * BK + kk] - sg[j * BK + kk]) : 0.f;
      }
      __syncthreads();
      const int n0 = i0 / RPN;
      for (int nn = 0; nn < NPS; ++nn) {
        const int n = n0 + nn;
        const int p = tid + n * 256;
        const int i = p / BT, j = p % BT;
        if (i >= len) continue;
        const bool useA = j < i, useQ = j <= i;
        if (!useA && !useQ) continue;
        const float* fi = sfi + (i - i0) * BK;
        const float* fj = sfj + j * BK;
        const float* ki = sk + i * BK;
        const float* qi = sq + i * BK;
        const float* kj = sk + j * BK;
        float aA = 0.f, aQ = 0.f;
        for (int kk = 0; kk < BK; ++kk) {
          const float b = kj[kk] * fj[kk];
          if (useA) aA += ki[kk] * fi[kk] * b;
          if (useQ) aQ += qi[kk] * fi[kk] * b;
        }
        accA[n] += aA;
        accQ[n] += aQ;
      }
      __syncthreads();
    }
  }
  // β_i 加权（A 用），Aq 直接写回；A 同 delta 规则写共享
  for (int n = 0; n < PAIRS; ++n) {
    const int p = tid + n * 256;
    const int i = p / BT;
    const float b = (i < len) ? beta[(long)(row0 + i) * heads + h] : 0.f;
    sA[p] = accA[n] * b;
    aq_out[(long)c * heads * BT * BT + (long)h * BT * BT + p] = accQ[n];
  }
  __syncthreads();

  // ---- 2) M = (I+A)⁻¹：单位下三角前代（按列并行、按行串行）
  for (int idx = tid; idx < BT * BT; idx += nb) sM[idx] = 0.f;
  __syncthreads();
  for (int i = 0; i < BT; ++i) {
    for (int j = tid; j < BT; j += nb) {
      float acc = (j == i) ? 1.f : 0.f;
      for (int jj = 0; jj < i; ++jj) acc -= sA[i * BT + jj] * sM[jj * BT + j];
      sM[i * BT + j] = acc;
    }
    __syncthreads();
  }

  // ---- 3) u = M·(β⊙v)：V 分块，逐块临时放 β⊙v
  for (int v0 = 0; v0 < V; v0 += BK) {
    const int bw = min(BK, V - v0);
    for (int idx = tid; idx < BT * BK; idx += nb) {
      const int i = idx / BK, j = idx % BK;
      float x = 0.f;
      if (i < len && j < bw) x = beta[(long)(row0 + i) * heads + h] *
                                   to_f32(v[(long)(row0 + i) * heads * V + (long)h * V + v0 + j]);
      sk[idx] = x;  // 复用 sk 作临时
    }
    __syncthreads();
    // 每个线程负责同一 i 的 4 个连续 j：sM[i][t] 读一次喂 4 次 FMA，sk[t][j] 也只在寄存器里放一次。
    for (int idx = tid; idx < BT * (BK / 4); idx += nb) {
      const int i = idx / (BK / 4), j0 = (idx % (BK / 4)) * 4;
      if (i >= len || j0 >= bw) continue;
      const float* mr = sM + i * BT;
      float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
      const int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
      const bool ok1 = j1 < bw, ok2 = j2 < bw, ok3 = j3 < bw;
      for (int t = 0; t < BT; ++t) {
        const float m = mr[t];
        const float* sr = sk + t * BK + j0;
        a0 += m * sr[0];
        if (ok1) a1 += m * sr[1];
        if (ok2) a2 += m * sr[2];
        if (ok3) a3 += m * sr[3];
      }
      const long o = (long)(row0 + i) * heads * V + (long)h * V + v0 + j0;
      u_out[o] = a0;
      if (ok1) u_out[o + 1] = a1;
      if (ok2) u_out[o + 2] = a2;
      if (ok3) u_out[o + 3] = a3;
    }
    __syncthreads();
  }

  // ---- 4) w = M·(β⊙k⊙exp2(gc))、kg = k·exp2(gc_last − gc)
  const float* glast_row = gc + (long)(row0 + len - 1) * stride + hoff;
  for (int k0 = 0; k0 < K; k0 += BK) {
    const int bw = min(BK, K - k0);
    for (int idx = tid; idx < BT * BK; idx += nb) {
      const int i = idx / BK, j = idx % BK;
      float x = 0.f;
      if (i < len && j < bw) {
        const long r = (long)(row0 + i) * stride + hoff + k0 + j;
        x = beta[(long)(row0 + i) * heads + h] * to_f32(kn[r]) * exp2f(gc[r]);
      }
      sk[idx] = x;
    }
    __syncthreads();
    for (int idx = tid; idx < BT * (BK / 4); idx += nb) {
      const int i = idx / (BK / 4), j0 = (idx % (BK / 4)) * 4;
      if (i >= len || j0 >= bw) continue;
      const float* mr = sM + i * BT;
      float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
      const int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
      const bool ok1 = j1 < bw, ok2 = j2 < bw, ok3 = j3 < bw;
      for (int t = 0; t < BT; ++t) {
        const float m = mr[t];
        const float* sr = sk + t * BK + j0;
        a0 += m * sr[0];
        if (ok1) a1 += m * sr[1];
        if (ok2) a2 += m * sr[2];
        if (ok3) a3 += m * sr[3];
      }
      const long r = (long)(row0 + i) * stride + hoff + k0 + j0;
      w_out[r] = a0;
      kg_out[r] = to_f32(kn[r]) * exp2f(glast_row[k0 + j0] - gc[r]);
      if (ok1) {
        w_out[r + 1] = a1;
        kg_out[r + 1] = to_f32(kn[r + 1]) * exp2f(glast_row[k0 + j1] - gc[r + 1]);
      }
      if (ok2) {
        w_out[r + 2] = a2;
        kg_out[r + 2] = to_f32(kn[r + 2]) * exp2f(glast_row[k0 + j2] - gc[r + 2]);
      }
      if (ok3) {
        w_out[r + 3] = a3;
        kg_out[r + 3] = to_f32(kn[r + 3]) * exp2f(glast_row[k0 + j3] - gc[r + 3]);
      }
    }
    __syncthreads();
  }
}

// ================================================================ ho
// 每 block 一个 (序列, 头, V 分片)。块在时间上串行；h 留 fp32 共享。
// 共享布局全部把 K 维补 1 列（KPAD = K+1）避免按列访问时的 bank 冲突。
// 顺序：qg·h → w·h（v'）→ o += Aq·v' → h 更新（kg），三段复用同一 [BT][KPAD] 缓冲。
template <int BT, int BV, int K>
__global__ void ho_kernel(const bf16* __restrict__ qn, const float* __restrict__ gc, const float* __restrict__ aq,
                          const float* __restrict__ u, const float* __restrict__ w, const float* __restrict__ kg,
                          bf16* __restrict__ out, float* __restrict__ state, const int* __restrict__ cu_seqlens,
                          const int* __restrict__ cu_nt, const int* __restrict__ chunk_off,
                          const int* __restrict__ state_indices, int64_t state_stride_tok, int heads, int V) {
  const int v0 = blockIdx.x * BV;
  const int h = blockIdx.y;
  const int seq = blockIdx.z;
  const int tid = threadIdx.x;
  const int nb = blockDim.x;
  constexpr int KPAD = K + 1;

  const int slot = state_indices ? state_indices[seq] : seq;
  const int bos = cu_seqlens[seq], eos = cu_seqlens[seq + 1];
  const int len_seq = eos - bos;
  if (len_seq <= 0 || slot < 0) return;
  const int nt = cu_nt[seq];
  const int stride = heads * K;
  const long hoff = (long)h * K;

  extern __shared__ float sm[];
  float* sh = sm;                     // [BV][KPAD] 状态
  float* sqg = sh + BV * KPAD;        // [BT][KPAD] 复用：qg → w → kg
  float* sv = sqg + BT * KPAD;        // [BT][BV]   v'
  float* so = sv + BT * BV;           // [BT][BV]   qg·h

  // 初态
  const float* h_in = state + (long)slot * state_stride_tok + (long)h * V * K;
  for (int idx = tid; idx < BV * KPAD; idx += nb) {
    const int vv = idx / KPAD, kk = idx % KPAD;
    sh[idx] = (kk < K) ? h_in[(long)(v0 + vv) * K + kk] : 0.f;
  }
  __syncthreads();

  for (int ci = 0; ci < nt; ++ci) {
    const int c = chunk_off[seq] + ci;
    const int len = min(BT, len_seq - ci * BT);
    const long row0 = bos + (long)ci * BT;
    const float* glast_row = gc + (long)(row0 + len - 1) * stride + hoff;
    const float* aqc = aq + (long)c * heads * BT * BT + (long)h * BT * BT;

    // ---- (1) qg·h：o 的第一项
    for (int idx = tid; idx < BT * KPAD; idx += nb) {
      const int i = idx / KPAD, kk = idx % KPAD;
      sqg[idx] = (i < len && kk < K)
                     ? to_f32(qn[(long)(row0 + i) * stride + hoff + kk]) *
                           exp2f(gc[(long)(row0 + i) * stride + hoff + kk])
                     : 0.f;
    }
    __syncthreads();
    for (int idx = tid; idx < BT * (BV / 2); idx += nb) {
      const int i = idx / (BV / 2), vv = idx % (BV / 2);
      const float* qr = sqg + i * KPAD;
      const float* h0 = sh + vv * KPAD;
      const float* h1 = sh + (vv + BV / 2) * KPAD;
      float a0 = 0.f, a1 = 0.f;
      #pragma unroll 8
      for (int kk = 0; kk < K; ++kk) {
        const float qv = qr[kk];
        a0 += qv * h0[kk];
        a1 += qv * h1[kk];
      }
      so[i * BV + vv] = a0;
      so[i * BV + vv + BV / 2] = a1;
    }
    __syncthreads();
    // ---- (2) v' = u − w·h（h 仍是更新前）
    for (int idx = tid; idx < BT * KPAD; idx += nb) {
      const int i = idx / KPAD, kk = idx % KPAD;
      sqg[idx] = (i < len && kk < K) ? w[(long)(row0 + i) * stride + hoff + kk] : 0.f;
    }
    __syncthreads();
    for (int idx = tid; idx < BT * (BV / 2); idx += nb) {
      const int i = idx / (BV / 2), vv = idx % (BV / 2);
      const float* wr = sqg + i * KPAD;
      const float* h0 = sh + vv * KPAD;
      const float* h1 = sh + (vv + BV / 2) * KPAD;
      float b0 = 0.f, b1 = 0.f;
      #pragma unroll 8
      for (int kk = 0; kk < K; ++kk) {
        const float wv = wr[kk];
        b0 += wv * h0[kk];
        b1 += wv * h1[kk];
      }
      const int p0 = i * BV + vv, p1 = p0 + BV / 2;
      const long ub = (long)(row0 + i) * heads * V + (long)h * V + v0;
      const float u0 = (i < len) ? u[ub + vv] : 0.f;
      const float u1 = (i < len) ? u[ub + vv + BV / 2] : 0.f;
      sv[p0] = u0 - b0;
      sv[p1] = u1 - b1;
    }
    __syncthreads();
    // ---- (3) o += Aq·v'，并写回
    for (int idx = tid; idx < BT * BV; idx += nb) {
      const int i = idx / BV, vv = idx % BV;
      if (i >= len) continue;
      float acc = so[idx];
      #pragma unroll 8
      for (int j = 0; j <= i; ++j) acc += aqc[i * BT + j] * sv[j * BV + vv];
      out[(long)(row0 + i) * heads * V + (long)h * V + v0 + vv] = to_bf16(acc);
    }
    __syncthreads();
    // ---- (4) h ← h·exp2(gc_last) + v'ᵀ·kg（kg 复用 sqg）
    for (int idx = tid; idx < BT * KPAD; idx += nb) {
      const int i = idx / KPAD, kk = idx % KPAD;
      sqg[idx] = (i < len && kk < K) ? kg[(long)(row0 + i) * stride + hoff + kk] : 0.f;
    }
    __syncthreads();
    for (int idx = tid; idx < BV * KPAD; idx += nb) {
      const int vv = idx / KPAD, kk = idx % KPAD;
      if (kk >= K) continue;
      float acc = sh[idx] * exp2f(glast_row[kk]);
      const float* kr = sqg + kk;
      const float* vr = sv + vv;
      #pragma unroll 8
      for (int i = 0; i < len; ++i) acc += vr[i * BV] * kr[i * KPAD];
      sh[idx] = acc;
    }
    __syncthreads();
  }

  // 末态回写
  float* h_out = state + (long)slot * state_stride_tok + (long)h * V * K;
  for (int idx = tid; idx < BV * KPAD; idx += nb) {
    const int vv = idx / KPAD, kk = idx % KPAD;
    if (kk < K) h_out[(long)(v0 + vv) * K + kk] = sh[idx];
  }
}

// 动态 shared 上限按 (kernel, 字节数) 记忆；换了特化必须重设，否则前一次的属性不生效。
thread_local int smem_cache[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};  // 按设备生效 → 每 rank 线程一份（pitfalls 2026-10-20）
int set_smem(const void* fn, int bytes) {
  const intptr_t key = (intptr_t)fn;
  int slot = 0;
  for (; slot < 4; ++slot) {
    if (smem_cache[slot][0] == key) {
      if (smem_cache[slot][1] >= bytes) return 0;
      break;
    }
  }
  const int rc = err(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  if (rc != 0) return rc;
  slot = (slot == 4) ? (int)((key >> 4) & 3) : slot;
  smem_cache[slot][0] = (int)key;
  smem_cache[slot][1] = bytes;
  return 0;
}

}  // namespace

extern "C" {

int kda_chunk_prep(const void* q, const void* k, const void* raw_g, const float* raw_beta, const float* a_log,
                   const float* dt_bias, void* qn, void* kn, float* beta, float* gc, const int* cu_seqlens,
                   int64_t num_seqs, int64_t heads, int64_t head_dim, int64_t bt, float scale, float lower_bound,
                   int use_lower_bound, cudaStream_t stream) {
  if (num_seqs <= 0 || heads <= 0 || head_dim <= 0 || bt <= 0 || head_dim > 256) return err(cudaErrorInvalidValue);
  dim3 grid((unsigned)heads, (unsigned)num_seqs, 2);
  prep_kernel<<<grid, 256, 0, stream>>>(static_cast<const bf16*>(q), static_cast<const bf16*>(k),
                                        static_cast<const bf16*>(raw_g), raw_beta, a_log, dt_bias,
                                        static_cast<bf16*>(qn), static_cast<bf16*>(kn), beta, gc, cu_seqlens,
                                        (int)heads, (int)head_dim, (int)bt, scale, lower_bound, use_lower_bound);
  return last_err();
}

int kda_chunk_wy(const void* qn, const void* kn, const void* v, const float* gc, const float* beta, float* aq,
                 float* u, float* w, float* kg, const int* cu_seqlens, const int* chunk_tab, int64_t num_chunks,
                 int64_t heads, int64_t head_dim, int64_t bt, cudaStream_t stream) {
  if (num_chunks <= 0 || heads <= 0) return err(cudaErrorInvalidValue);
  const int K = (int)head_dim;
  const int V = K;  // KDA 的 V == K == head_dim
  dim3 grid((unsigned)num_chunks, (unsigned)heads);
  if (bt == 64 && K % 64 == 0) {
    constexpr int BT = 64, BC = 16, BK = 64;
    const int smem = (int)((3 * BT * BK + 2 * BT * BT + BC * BK) * sizeof(float));
    if (const int rc = set_smem(reinterpret_cast<const void*>(&wy_kernel<BT, BC, BK>), smem)) return rc;
    wy_kernel<BT, BC, BK><<<grid, 256, smem, stream>>>(static_cast<const bf16*>(qn), static_cast<const bf16*>(kn),
                                                   static_cast<const bf16*>(v), gc, beta, aq, u, w, kg, cu_seqlens,
                                                   chunk_tab, (int)heads, K, V);
  } else if (bt == 64) {
    constexpr int BT = 64, BC = 16, BK = 32;
    const int smem = (int)((3 * BT * BK + 2 * BT * BT + BC * BK) * sizeof(float));
    if (const int rc = set_smem(reinterpret_cast<const void*>(&wy_kernel<BT, BC, BK>), smem)) return rc;
    wy_kernel<BT, BC, BK><<<grid, 256, smem, stream>>>(static_cast<const bf16*>(qn), static_cast<const bf16*>(kn),
                                                   static_cast<const bf16*>(v), gc, beta, aq, u, w, kg, cu_seqlens,
                                                   chunk_tab, (int)heads, K, V);
  } else {
    return err(cudaErrorInvalidValue);
  }
  return last_err();
}

int kda_chunk_ho(const void* qn, const float* gc, const float* aq, const float* u, const float* w,
                 const float* kg, void* out, void* state, const int* cu_seqlens, const int* cu_nt,
                 const int* chunk_off, const int* state_indices, int64_t state_stride_tok, int64_t num_seqs,
                 int64_t heads, int64_t head_dim, int64_t bt, cudaStream_t stream) {
  if (num_seqs <= 0 || heads <= 0 || head_dim <= 0) return err(cudaErrorInvalidValue);
  const int K = (int)head_dim;
  const int V = K;
  if (bt != 64) return err(cudaErrorInvalidValue);
  constexpr int BT = 64, BV = 16;
  dim3 grid((unsigned)(V / BV), (unsigned)heads, (unsigned)num_seqs);
  // K 是模板参数（内层可全展开）；只有 head_dim=128 的特化，其它尺寸用通用版（同样逻辑，不展开）。
#define LAUNCH_HO(KK)                                                                                  \
  {                                                                                                    \
    const int smem = (int)((BV * (KK + 1) + BT * (KK + 1) + 2 * BT * BV) * sizeof(float));              \
    if (const int rc = set_smem(reinterpret_cast<const void*>(&ho_kernel<BT, BV, KK>), smem)) return rc; \
    ho_kernel<BT, BV, KK><<<grid, 256, smem, stream>>>(                                                \
        static_cast<const bf16*>(qn), gc, aq, u, w, kg, static_cast<bf16*>(out),                       \
        static_cast<float*>(state), cu_seqlens, cu_nt, chunk_off, state_indices, state_stride_tok,     \
        (int)heads, V);                                                                                \
  }
  if (K == 128) {
    LAUNCH_HO(128)
  } else if (K == 64) {
    LAUNCH_HO(64)
  } else if (K == 256) {
    LAUNCH_HO(256)
  } else {
    return err(cudaErrorInvalidValue);
  }
#undef LAUNCH_HO
  return last_err();
}

}  // extern "C"
