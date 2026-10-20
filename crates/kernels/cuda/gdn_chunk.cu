// GDN（Gated DeltaNet）chunk 形式 prefill。模型无关：只按形状参数，BT 由 host 选（模板参数）。
//
// 与 kda_chunk.cu 同一代数（fla `chunk_gated_delta_rule`），差异（docs/loops/qwen35-perf.md 第 1 轮）：
//   ① gate 是逐 v 头标量 g[t,hv]（KDA 逐 k 维）：gc 变 [T, HV]，exp2 因子不再按维；
//   ② V 与 K 分开（接口不假设相等）；③ q/k 头是 h = hv / group（GQA 式共享）。
// 唯一数值口径是 gdn.cu 的 recurrent_kernel（逐 token 顺序递归），本文件必须与之等价：
//   gc   = 块内 inclusive cumsum(g) · 1/ln2                  （下游一律 exp2）
//   A[i,j]  = β_i (k_i·k_j) exp2(gc_i − gc_j),  j <  i      （严格下三角，只用它求 M）
//   Aq[i,j] = (q_i·k_j)     exp2(gc_i − gc_j),  j <= i      （含对角线）
//   M = (I+A)⁻¹；u = M(β⊙v)；w = M(β⊙k⊙exp2(gc))；kg = k·exp2(gc_last − gc)
//   v' = u − w·h_prev；o = (q⊙exp2(gc))·h_prev + Aq·v'；h_next = h_prev·exp2(gc_last) + v'ᵀ·kg
// 输入沿用 gdn_recurrent 的 mixed_qkv [T, 2·H·K + HV·V]（q|k|v 平铺）与 gdn_gate 的 g/beta [T, HV]；
// 状态布局 [slot, HV, V, K] f32（与 gdn.cu 一致）。
//
// 三个 kernel：
//   chunk_prep   q/k 的 l2norm（q 乘 scale）→ qn/kn [T,H,K] bf16；块内 cumsum(g) → gc [T,HV]
//   chunk_wy     A/Aq → M → u / w / kg（A 只在本 kernel 内用；Aq 写回给输出段）
//   chunk_ho     跨 chunk 串行扫 h + 立即算 o（融合版）；r9 起另有两段版 h-pass / o-pass（见 ho_hpass_kernel 注释）
// 数值口径：老版全 fp32；mma 版 bf16 乘 fp32 累加（与 fla 同口径）。
#include "common.cuh"
#include "kernels.h"

#include <stdint.h>

using namespace dinfer;

namespace {

// 数学常量（不是调优参数）：1/ln2，把自然对数空间的 cumsum 变成 exp2 的指数。
constexpr float kRcpLn2 = 1.4426950408889634f;

// ================================================================ prep
// (a) q/k 归一化：一个 warp 一个 (token, 头)；K <= 256 时每 lane 最多 8 个元素。
__global__ void norm_kernel(const bf16* __restrict__ mixed, bf16* __restrict__ qn, bf16* __restrict__ kn,
                            int total, int H, int K, long qkv_row, float scale) {
  const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;  // 全局 warp 号
  const int lane = threadIdx.x & 31;
  const int t = gw / (2 * H);
  if (t >= total) return;
  const int rem = gw % (2 * H);
  const int which = rem / H;  // 0 = q, 1 = k
  const int h = rem % H;
  const bf16* src = mixed + (long)t * qkv_row + (long)which * H * K + (long)h * K;
  bf16* dst = (which ? kn : qn) + ((long)t * H + h) * K;
  float v[8];
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int d = lane + i * 32;
    v[i] = (d < K) ? to_f32(src[d]) : 0.f;
    ss += v[i] * v[i];
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float r = (which ? 1.f : scale) / sqrtf(ss + 1e-6f);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int d = lane + i * 32;
    if (d < K) dst[d] = to_bf16(v[i] * r);
  }
}

// (b) 块内 cumsum：一个线程一个 (chunk, v 头)，时间维串行（BT 步）。
__global__ void cumsum_kernel(const float* __restrict__ g, float* __restrict__ gc,
                              const int* __restrict__ cu_seqlens, const int* __restrict__ chunk_tab,
                              int num_chunks, int HV, int BT) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_chunks * HV) return;
  const int c = idx / HV, hv = idx % HV;
  const int seq = chunk_tab[2 * c], ci = chunk_tab[2 * c + 1];
  const int bos = cu_seqlens[seq], eos = cu_seqlens[seq + 1];
  const int len = min(BT, eos - bos - ci * BT);
  const long row0 = bos + (long)ci * BT;
  float acc = 0.f;
  for (int t = 0; t < len; ++t) {
    acc += g[(row0 + t) * HV + hv];
    gc[(row0 + t) * HV + hv] = acc * kRcpLn2;
  }
}

// ================================================================ wy
// 每 block 一个 (chunk, v 头)，256 线程。K 按 BK 分段累加到寄存器，A/Aq 各 PAIRS 个/线程。
// 求 M 用单位下三角前代；u/w 用 M 左乘。衰减因子是标量，A/Aq 累完点积后一次乘 exp2(gc_i − gc_j)。
template <int BT, int BK>
__global__ void wy_kernel(const bf16* __restrict__ qn, const bf16* __restrict__ kn, const bf16* __restrict__ mixed,
                          const float* __restrict__ gc, const float* __restrict__ beta, float* __restrict__ aq_out,
                          float* __restrict__ u_out, float* __restrict__ w_out, float* __restrict__ kg_out,
                          const int* __restrict__ cu_seqlens, const int* __restrict__ chunk_tab, int H, int HV,
                          int K, int V, int group) {
  const int c = blockIdx.x;
  const int hv = blockIdx.y;
  const int h = hv / group;
  const int tid = threadIdx.x;
  const int nb = blockDim.x;
  constexpr int PAIRS = BT * BT / 256;
  static_assert(BT * BT % 256 == 0, "BT*BT 必须是线程数的整数倍");

  const int seq = chunk_tab[2 * c], ci = chunk_tab[2 * c + 1];
  const int bos = cu_seqlens[seq], eos = cu_seqlens[seq + 1];
  const int len = min(BT, eos - bos - ci * BT);
  const long row0 = bos + (long)ci * BT;
  const long qkv_row = 2L * H * K + (long)HV * V;
  const long kstride = (long)H * K;  // qn/kn 的每 token 宽度

  extern __shared__ float sm[];
  float* sk = sm;             // [BT][BK] kn 分段 → 之后复用作 β⊙v / β⊙k⊙exp2(gc) 临时
  float* sq = sk + BT * BK;   // [BT][BK] qn 分段
  float* sA = sq + BT * BK;   // [BT][BT] A
  float* sM = sA + BT * BT;   // [BT][BT] M
  float* sgc = sM + BT * BT;  // [BT] 本块 gc
  float* sb = sgc + BT;       // [BT] 本块 beta

  for (int i = tid; i < BT; i += nb) {
    const bool ok = i < len;
    sgc[i] = ok ? gc[(row0 + i) * HV + hv] : 0.f;
    sb[i] = ok ? beta[(row0 + i) * HV + hv] : 0.f;
  }

  // ---- 1) 点积 q_i·k_j / k_i·k_j：按 K 分段累加到寄存器
  float accA[PAIRS], accQ[PAIRS];
  for (int n = 0; n < PAIRS; ++n) { accA[n] = 0.f; accQ[n] = 0.f; }
  for (int k0 = 0; k0 < K; k0 += BK) {
    const int bw = min(BK, K - k0);
    __syncthreads();  // 上一轮的读完
    for (int idx = tid; idx < BT * BK; idx += nb) {
      const int j = idx / BK, kk = idx % BK;
      const bool ok = j < len && kk < bw;
      const long r = (row0 + j) * kstride + (long)h * K + k0 + kk;
      sk[idx] = ok ? to_f32(kn[r]) : 0.f;
      sq[idx] = ok ? to_f32(qn[r]) : 0.f;
    }
    __syncthreads();
    for (int n = 0; n < PAIRS; ++n) {
      const int p = tid + n * 256;
      const int i = p / BT, j = p % BT;
      if (i >= len || j > i) continue;
      const float* ki = sk + i * BK;
      const float* qi = sq + i * BK;
      const float* kj = sk + j * BK;
      float aA = 0.f, aQ = 0.f;
#pragma unroll 8
      for (int kk = 0; kk < BK; ++kk) {
        const float b = kj[kk];
        aA += ki[kk] * b;
        aQ += qi[kk] * b;
      }
      accA[n] += aA;
      accQ[n] += aQ;
    }
  }
  __syncthreads();
  // 衰减 + β；A 严格下三角，Aq 含对角线
  for (int n = 0; n < PAIRS; ++n) {
    const int p = tid + n * 256;
    const int i = p / BT, j = p % BT;
    float a = 0.f, q = 0.f;
    if (i < len && j <= i) {
      const float decay = exp2f(sgc[i] - sgc[j]);
      q = accQ[n] * decay;
      a = (j < i) ? accA[n] * decay * sb[i] : 0.f;
    }
    sA[p] = a;
    aq_out[((long)c * HV + hv) * BT * BT + p] = q;
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

  // ---- 3) u = M·(β⊙v)：V 分块，逐块临时放 β⊙v（复用 sk）
  const bf16* vbase = mixed + 2L * H * K + (long)hv * V;
  for (int v0 = 0; v0 < V; v0 += BK) {
    const int bw = min(BK, V - v0);
    for (int idx = tid; idx < BT * BK; idx += nb) {
      const int i = idx / BK, j = idx % BK;
      float x = 0.f;
      if (i < len && j < bw) x = sb[i] * to_f32(vbase[(row0 + i) * qkv_row + v0 + j]);
      sk[idx] = x;
    }
    __syncthreads();
    // 每个线程负责同一 i 的 4 个连续 j：sM[i][t] 读一次喂 4 次 FMA。
    for (int idx = tid; idx < BT * (BK / 4); idx += nb) {
      const int i = idx / (BK / 4), j0 = (idx % (BK / 4)) * 4;
      if (i >= len || j0 >= bw) continue;
      const float* mr = sM + i * BT;
      float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
      const bool ok1 = j0 + 1 < bw, ok2 = j0 + 2 < bw, ok3 = j0 + 3 < bw;
      for (int t = 0; t < BT; ++t) {
        const float m = mr[t];
        const float* sr = sk + t * BK + j0;
        a0 += m * sr[0];
        if (ok1) a1 += m * sr[1];
        if (ok2) a2 += m * sr[2];
        if (ok3) a3 += m * sr[3];
      }
      const long o = ((row0 + i) * HV + hv) * V + v0 + j0;
      u_out[o] = a0;
      if (ok1) u_out[o + 1] = a1;
      if (ok2) u_out[o + 2] = a2;
      if (ok3) u_out[o + 3] = a3;
    }
    __syncthreads();
  }

  // ---- 4) w = M·(β⊙k⊙exp2(gc))、kg = k·exp2(gc_last − gc)（都按 v 头写，布局 [T, HV, K]）
  const float glast = sgc[len - 1];
  for (int k0 = 0; k0 < K; k0 += BK) {
    const int bw = min(BK, K - k0);
    for (int idx = tid; idx < BT * BK; idx += nb) {
      const int i = idx / BK, j = idx % BK;
      float x = 0.f;
      if (i < len && j < bw) {
        const long r = (row0 + i) * kstride + (long)h * K + k0 + j;
        x = sb[i] * to_f32(kn[r]) * exp2f(sgc[i]);
      }
      sk[idx] = x;
    }
    __syncthreads();
    for (int idx = tid; idx < BT * (BK / 4); idx += nb) {
      const int i = idx / (BK / 4), j0 = (idx % (BK / 4)) * 4;
      if (i >= len || j0 >= bw) continue;
      const float* mr = sM + i * BT;
      float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
      const bool ok1 = j0 + 1 < bw, ok2 = j0 + 2 < bw, ok3 = j0 + 3 < bw;
      for (int t = 0; t < BT; ++t) {
        const float m = mr[t];
        const float* sr = sk + t * BK + j0;
        a0 += m * sr[0];
        if (ok1) a1 += m * sr[1];
        if (ok2) a2 += m * sr[2];
        if (ok3) a3 += m * sr[3];
      }
      const float dk = exp2f(glast - sgc[i]);
      const long rk = (row0 + i) * kstride + (long)h * K + k0 + j0;  // kn 读
      const long rw = ((row0 + i) * HV + hv) * K + k0 + j0;         // w/kg 写
      w_out[rw] = a0;
      kg_out[rw] = to_f32(kn[rk]) * dk;
      if (ok1) { w_out[rw + 1] = a1; kg_out[rw + 1] = to_f32(kn[rk + 1]) * dk; }
      if (ok2) { w_out[rw + 2] = a2; kg_out[rw + 2] = to_f32(kn[rk + 2]) * dk; }
      if (ok3) { w_out[rw + 3] = a3; kg_out[rw + 3] = to_f32(kn[rk + 3]) * dk; }
    }
    __syncthreads();
  }
}

// ================================================================ ho
// 每 block 一个 (序列, v 头, V 分片)。块在时间上串行；h 留 fp32 共享。
// 共享布局把 K 维补 1 列（KPAD = K+1）避免按列访问时的 bank 冲突。
// 顺序：qg·h → w·h（v'）→ o += Aq·v' → h 更新（kg），三段复用同一 [BT][KPAD] 缓冲。
template <int BT, int BV, int K>
__global__ void ho_kernel(const bf16* __restrict__ qn, const float* __restrict__ gc, const float* __restrict__ aq,
                          const float* __restrict__ u, const float* __restrict__ w, const float* __restrict__ kg,
                          bf16* __restrict__ out, float* __restrict__ state, const int* __restrict__ cu_seqlens,
                          const int* __restrict__ cu_nt, const int* __restrict__ chunk_off,
                          const int* __restrict__ state_indices, int64_t state_stride_tok, int H, int HV, int V,
                          int group) {
  const int v0 = blockIdx.x * BV;
  const int hv = blockIdx.y;
  const int h = hv / group;
  const int seq = blockIdx.z;
  const int tid = threadIdx.x;
  const int nb = blockDim.x;
  constexpr int KPAD = K + 1;

  const int slot = state_indices ? state_indices[seq] : seq;
  const int bos = cu_seqlens[seq], eos = cu_seqlens[seq + 1];
  const int len_seq = eos - bos;
  if (len_seq <= 0 || slot < 0) return;
  const int nt = cu_nt[seq];
  const long kstride = (long)H * K;

  extern __shared__ float sm[];
  float* sh = sm;               // [BV][KPAD] 状态
  float* sqg = sh + BV * KPAD;  // [BT][KPAD] 复用：qg → w → kg
  float* sv = sqg + BT * KPAD;  // [BT][BV]   v'
  float* so = sv + BT * BV;     // [BT][BV]   qg·h
  float* sdec = so + BT * BV;   // [BT]       exp2(gc)

  float* hslot = state + (long)slot * state_stride_tok + (long)hv * V * K;
  for (int idx = tid; idx < BV * KPAD; idx += nb) {
    const int vv = idx / KPAD, kk = idx % KPAD;
    sh[idx] = (kk < K) ? hslot[(long)(v0 + vv) * K + kk] : 0.f;
  }
  __syncthreads();

  for (int ci = 0; ci < nt; ++ci) {
    const int c = chunk_off[seq] + ci;
    const int len = min(BT, len_seq - ci * BT);
    const long row0 = bos + (long)ci * BT;
    const float* aqc = aq + ((long)c * HV + hv) * BT * BT;

    for (int i = tid; i < BT; i += nb) sdec[i] = (i < len) ? exp2f(gc[(row0 + i) * HV + hv]) : 0.f;
    __syncthreads();
    // ---- (1) qg·h：o 的第一项
    for (int idx = tid; idx < BT * KPAD; idx += nb) {
      const int i = idx / KPAD, kk = idx % KPAD;
      sqg[idx] = (i < len && kk < K) ? to_f32(qn[(row0 + i) * kstride + (long)h * K + kk]) * sdec[i] : 0.f;
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
      sqg[idx] = (i < len && kk < K) ? w[((row0 + i) * HV + hv) * K + kk] : 0.f;
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
      const long ub = ((row0 + i) * HV + hv) * V + v0;
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
      out[((row0 + i) * HV + hv) * V + v0 + vv] = to_bf16(acc);
    }
    __syncthreads();
    // ---- (4) h ← h·exp2(gc_last) + v'ᵀ·kg（kg 复用 sqg）
    for (int idx = tid; idx < BT * KPAD; idx += nb) {
      const int i = idx / KPAD, kk = idx % KPAD;
      sqg[idx] = (i < len && kk < K) ? kg[((row0 + i) * HV + hv) * K + kk] : 0.f;
    }
    __syncthreads();
    const float glast = sdec[len - 1];
    for (int idx = tid; idx < BV * KPAD; idx += nb) {
      const int vv = idx / KPAD, kk = idx % KPAD;
      if (kk >= K) continue;
      float acc = sh[idx] * glast;
      const float* kr = sqg + kk;
      const float* vr = sv + vv;
#pragma unroll 8
      for (int i = 0; i < len; ++i) acc += vr[i * BV] * kr[i * KPAD];
      sh[idx] = acc;
    }
    __syncthreads();
  }

  // 末态回写
  for (int idx = tid; idx < BV * KPAD; idx += nb) {
    const int vv = idx / KPAD, kk = idx % KPAD;
    if (kk < K) hslot[(long)(v0 + vv) * K + kk] = sh[idx];
  }
}

// ---- mma.sync m16n8k16 bf16 / ldmatrix 封装（wy / ho tensor core 版共用）
__device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, const void* p) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf16x2(float x, float y) {
  __nv_bfloat162 v = __floats2bfloat162_rn(x, y);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ uint32_t lds_u32(const bf16* p) { return *reinterpret_cast<const uint32_t*>(p); }
__device__ __forceinline__ void sts_u32(bf16* p, uint32_t v) { *reinterpret_cast<uint32_t*>(p) = v; }

// ================================================================ wy（第 5A 轮：tensor core 版）
// 每 block 一个 (chunk, v 头)，128 线程 = 4 warp，每 warp 固定持 16 行（token）tile。
//   1) A' = k·kᵀ、Aq' = q·kᵀ：mma（A 操作数 ldmatrix 行主序，B 操作数 = kn [j][k] 直接 32 位读）→ 乘衰减 / β 写 fp32 sM（A）与 aq_out
//   2) M = (I+A)⁻¹：fp32；TRI=1 分块三角逆（16×16 对角块各 warp 前代 + 非对角块 fp32 乘，r9 第 3 轮），TRI=0 64 步前代（第 5A 轮）
//   3) u = M·(β⊙v)、4) w = M·(β⊙k⊙exp2(gc))：按 64 列一片把右操作数转 bf16 进 stage [BT][64+8]，mma（B 走 ldmatrix.trans）
//   kg = k·exp2(gc_last − gc) 在 4) 的装载里顺手写（f32）。
// 数值口径：q/k 本来就是 bf16（prep 产出），1) 无额外损失；M 与 β⊙v / β⊙k 过 bf16（fla 同口径），fp32 累加。
constexpr int kWyWarps = 4;
constexpr int kWyStageCols = 64;  // u / w 阶段按 64 列一片
template <int BT, int K>
struct WySmem {
  static constexpr int KS = K + 8, TS = BT + 8, CS = kWyStageCols + 8;
  static constexpr int off_k = 0;                              // [BT][KS] bf16 kn；1) 之后 sK∪sQ 复用为 sM f32 [BT][BT]（r9 第 3 轮）
  static constexpr int off_q = off_k + BT * KS;                // [BT][KS] bf16 qn；2) 之后复用为 stage [BT][CS]
  static constexpr int q_elems = (BT * KS > BT * CS) ? BT * KS : BT * CS;
  static constexpr int off_m16 = off_q + q_elems;              // [BT][TS] bf16 M 影子
  static constexpr int elems16 = off_m16 + BT * TS;
  static_assert((BT * KS + q_elems) * 2 >= BT * BT * 4, "sK∪sQ 要装得下 sM f32");
  static constexpr int bytes = elems16 * 2 + 2 * BT * 4 + 16 * 16 * 4;  // + sgc / sb + sT（分块逆的 16×16 f32 临时）
};

template <int BT, int K, int TRI>
__global__ void __launch_bounds__(kWyWarps * 32) wy_mma_kernel(
    const bf16* __restrict__ qn, const bf16* __restrict__ kn, const bf16* __restrict__ mixed, const float* __restrict__ gc,
    const float* __restrict__ beta, float* __restrict__ aq_out, float* __restrict__ u_out, float* __restrict__ w_out,
    float* __restrict__ kg_out, const int* __restrict__ cu_seqlens, const int* __restrict__ chunk_tab, int H, int HV,
    int K_rt, int V, int group) {
  using S = WySmem<BT, K>;
  constexpr int KS = S::KS, TS = S::TS, CS = S::CS, NW = kWyWarps;
  constexpr int NT_T = BT / 8, KSTEP_K = K / 16, KSTEP_T = BT / 16;
  static_assert(BT / 16 == NW, "BT=64 → 4 warp 各一个 16 行 tile");
  (void)K_rt;

  const int c = blockIdx.x, hv = blockIdx.y, h = hv / group;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  const int seq = chunk_tab[2 * c], ci = chunk_tab[2 * c + 1];
  const int bos = cu_seqlens[seq], eos = cu_seqlens[seq + 1];
  const int len = min(BT, eos - bos - ci * BT);
  const long row0 = bos + (long)ci * BT;
  const long qkv_row = 2L * H * K + (long)HV * V;
  const long kstride = (long)H * K;

  extern __shared__ __align__(16) unsigned char smraw[];
  bf16* sm16 = reinterpret_cast<bf16*>(smraw);
  bf16* sK = sm16 + S::off_k;
  bf16* sQ = sm16 + S::off_q;
  bf16* stage = sQ;
  bf16* sM16 = sm16 + S::off_m16;
  float* sM = reinterpret_cast<float*>(sm16 + S::off_k);  // 1) 之后才能写（sK / sQ 读完）
  float* sgc = reinterpret_cast<float*>(sm16 + S::elems16);
  float* sb = sgc + BT;
  float* sT = sb + BT;  // [16][16] f32

  for (int i = tid; i < BT; i += NW * 32) {
    const bool ok = i < len;
    sgc[i] = ok ? gc[(row0 + i) * HV + hv] : 0.f;
    sb[i] = ok ? beta[(row0 + i) * HV + hv] : 0.f;
  }
  for (int idx = tid; idx < BT * (K / 8); idx += NW * 32) {
    const int i = idx / (K / 8), k8 = (idx % (K / 8)) * 8;
    uint4 kv = make_uint4(0, 0, 0, 0), qv = kv;
    if (i < len) {
      const long r = (row0 + i) * kstride + (long)h * K + k8;
      kv = *reinterpret_cast<const uint4*>(kn + r);
      qv = *reinterpret_cast<const uint4*>(qn + r);
    }
    *reinterpret_cast<uint4*>(sK + i * KS + k8) = kv;
    *reinterpret_cast<uint4*>(sQ + i * KS + k8) = qv;
  }
  __syncthreads();

  // ---- 1) 点积
  {
    float accA[NT_T][4], accQ[NT_T][4];
#pragma unroll
    for (int n = 0; n < NT_T; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) { accA[n][e] = 0.f; accQ[n][e] = 0.f; }
#pragma unroll
    for (int ks = 0; ks < KSTEP_K; ++ks) {
      uint32_t ak[4], aq[4];
      ldsm_x4(ak, sK + (warp * 16 + (lane & 15)) * KS + ks * 16 + (lane >> 4) * 8);
      ldsm_x4(aq, sQ + (warp * 16 + (lane & 15)) * KS + ks * 16 + (lane >> 4) * 8);
#pragma unroll
      for (int n = 0; n < NT_T; ++n) {
        const bf16* kb = sK + (n * 8 + g) * KS + ks * 16 + 2 * t4;
        const uint32_t b0 = lds_u32(kb), b1 = lds_u32(kb + 8);
        mma16816(accA[n], ak, b0, b1);
        mma16816(accQ[n], aq, b0, b1);
      }
    }
    const int i0 = warp * 16 + g, i1 = i0 + 8;
    float* aqc = aq_out + ((long)c * HV + hv) * BT * BT;
    __syncthreads();  // sM 覆盖 sK / sQ：所有 warp 的点积读完
#pragma unroll
    for (int n = 0; n < NT_T; ++n) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = (e < 2) ? i0 : i1, j = n * 8 + 2 * t4 + (e & 1);
        float a = 0.f, q = 0.f;
        if (i < len && j <= i) {
          const float decay = exp2f(sgc[i] - sgc[j]);
          q = accQ[n][e] * decay;
          a = (j < i) ? accA[n][e] * decay * sb[i] : 0.f;
        }
        sM[i * BT + j] = a;
        aqc[i * BT + j] = q;
      }
    }
  }
  __syncthreads();

  // ---- 2) M = (I+A)⁻¹
  if (TRI == 0) {
    // 64 步原地前代（第 5A 轮）：线程 tid < BT 各持一列，每步两次 __syncthreads
    static_assert(BT <= kWyWarps * 32, "前代每列一个线程");
    for (int i = 0; i < BT; ++i) {
      float acc = 0.f;
      if (tid < BT) {
        acc = (tid == i) ? 1.f : 0.f;
        for (int jj = 0; jj < i; ++jj) acc -= sM[i * BT + jj] * sM[jj * BT + tid];
      }
      __syncthreads();  // 行 i 的 A 值全部读完才能原地覆盖
      if (tid < BT) {
        sM[i * BT + tid] = acc;
        sM16[i * TS + tid] = to_bf16(acc);
      }
      __syncthreads();
    }
  } else {
    // 分块三角逆（r9 第 3 轮，结构同 fla solve_tril）：把 I+A 按 16×16 分成 NB×NB 块（单位下三角块矩阵）。
    //   对角块 M_ii = (I+A_ii)⁻¹：每 warp 一块，lane < 16 各持一列，16 步前代只需 __syncwarp；4 块并行。
    //   非对角块按列 j 升序、行 i 升序原地算：M_ij = −M_ii·(Σ_{j≤k<i} A_ik·M_kj)。按列走时读到的 A_ik（k > j）
    //   还是原值、M_kj（k < i）已算好，所以能原地。乘法是 16³ fp32 FMA（每线程 32 次），S 经 sT 中转。
    //   数值：全 fp32（与 64 步版同口径，只是结合顺序不同）；最后统一写 bf16 影子。
    constexpr int NB = BT / 16;
    static_assert(NB == NW, "对角块数 = warp 数");
    {
      const int r0 = warp * 16;
      for (int i = 0; i < 16; ++i) {
        float acc = 0.f;
        if (lane < 16) {
          acc = (lane == i) ? 1.f : 0.f;
          for (int jj = 0; jj < i; ++jj) acc -= sM[(r0 + i) * BT + r0 + jj] * sM[(r0 + jj) * BT + r0 + lane];
        }
        __syncwarp();
        if (lane < 16) sM[(r0 + i) * BT + r0 + lane] = acc;
        __syncwarp();
      }
    }
    __syncthreads();
    // 每线程 2 个输出元素：e ∈ {0,1}，(r, cc) = ((tid·2+e) / 16, (tid·2+e) % 16)
    static_assert(16 * 16 == 2 * kWyWarps * 32, "16×16 块 = 每线程 2 元素");
    const int r_a = (tid * 2) / 16, c_a = (tid * 2) % 16, c_b = c_a + 1;
    for (int j = 0; j < NB; ++j) {
      for (int i = j + 1; i < NB; ++i) {
        // S = Σ_{k=j}^{i-1} A_ik · M_kj  （k = j 时 A_ij 是原值；k > j 时 M_kj 已就位）
        float s0 = 0.f, s1 = 0.f;
        for (int k = j; k < i; ++k) {
          const float* arow = sM + (i * 16 + r_a) * BT + k * 16;
          const float* mcol = sM + (k * 16) * BT + j * 16;
#pragma unroll
          for (int t = 0; t < 16; ++t) {
            const float a = arow[t];
            s0 += a * mcol[t * BT + c_a];
            s1 += a * mcol[t * BT + c_b];
          }
        }
        sT[r_a * 16 + c_a] = s0;
        sT[r_a * 16 + c_b] = s1;
        __syncthreads();
        // M_ij = −M_ii · S
        float m0 = 0.f, m1 = 0.f;
        {
          const float* mrow = sM + (i * 16 + r_a) * BT + i * 16;
#pragma unroll
          for (int t = 0; t < 16; ++t) {
            const float m = mrow[t];
            m0 += m * sT[t * 16 + c_a];
            m1 += m * sT[t * 16 + c_b];
          }
        }
        __syncthreads();  // sT 读完、且本块 A_ij 的读（上面 S 的 k=j 项）已结束
        sM[(i * 16 + r_a) * BT + j * 16 + c_a] = -m0;
        sM[(i * 16 + r_a) * BT + j * 16 + c_b] = -m1;
        __syncthreads();
      }
    }
    // bf16 影子（含上三角 0）
    for (int idx = tid; idx < BT * BT; idx += NW * 32) {
      const int i = idx / BT, j = idx % BT;
      sM16[i * TS + j] = to_bf16(sM[idx]);
    }
    __syncthreads();
  }

  // ---- 3) u = M·(β⊙v)、4) w = M·(β⊙k⊙exp2(gc)) + kg
  const bf16* vbase = mixed + 2L * H * K + (long)hv * V;
  const float glast = sgc[len - 1];
  for (int pass = 0; pass < 2; ++pass) {
    const int cols = pass == 0 ? V : K;
    for (int c0 = 0; c0 < cols; c0 += kWyStageCols) {
      const int bw = min(kWyStageCols, cols - c0);
      __syncthreads();  // 上一片 stage 读完
      for (int idx = tid; idx < BT * (kWyStageCols / 2); idx += NW * 32) {
        const int i = idx / (kWyStageCols / 2), j = (idx % (kWyStageCols / 2)) * 2;
        float x0 = 0.f, x1 = 0.f;
        if (i < len && j < bw) {
          if (pass == 0) {
            const bf16* vp = vbase + (row0 + i) * qkv_row + c0 + j;
            x0 = sb[i] * to_f32(vp[0]);
            x1 = (j + 1 < bw) ? sb[i] * to_f32(vp[1]) : 0.f;
          } else {
            const long r = (row0 + i) * kstride + (long)h * K + c0 + j;
            const float k0 = to_f32(kn[r]), k1 = (j + 1 < bw) ? to_f32(kn[r + 1]) : 0.f;
            const float e = exp2f(sgc[i]);
            x0 = sb[i] * k0 * e;
            x1 = sb[i] * k1 * e;
            const float dk = exp2f(glast - sgc[i]);
            float* kgp = kg_out + ((row0 + i) * HV + hv) * K + c0 + j;
            kgp[0] = k0 * dk;
            if (j + 1 < bw) kgp[1] = k1 * dk;
          }
        }
        sts_u32(stage + i * CS + j, pack_bf16x2(x0, x1));
      }
      __syncthreads();
      float acc[kWyStageCols / 8][4];
#pragma unroll
      for (int n = 0; n < kWyStageCols / 8; ++n) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f; }
#pragma unroll
      for (int ks = 0; ks < KSTEP_T; ++ks) {
        uint32_t a[4];
        ldsm_x4(a, sM16 + (warp * 16 + (lane & 15)) * TS + ks * 16 + (lane >> 4) * 8);
#pragma unroll
        for (int n = 0; n < kWyStageCols / 8; n += 2) {
          uint32_t b[4];
          ldsm_x4_t(b, stage + (ks * 16 + (lane & 15)) * CS + n * 8 + (lane >> 4) * 8);
          mma16816(acc[n], a, b[0], b[1]);
          mma16816(acc[n + 1], a, b[2], b[3]);
        }
      }
      const int i0 = warp * 16 + g, i1 = i0 + 8;
      float* outp = pass == 0 ? u_out : w_out;
      const int ocols = pass == 0 ? V : K;
#pragma unroll
      for (int n = 0; n < kWyStageCols / 8; ++n) {
        const int j = c0 + n * 8 + 2 * t4;
        if (j >= c0 + bw) continue;
        if (i0 < len) *reinterpret_cast<float2*>(outp + ((row0 + i0) * HV + hv) * ocols + j) = make_float2(acc[n][0], acc[n][1]);
        if (i1 < len) *reinterpret_cast<float2*>(outp + ((row0 + i1) * HV + hv) * ocols + j) = make_float2(acc[n][2], acc[n][3]);
      }
    }
  }
}

// ================================================================ ho（第 5A 轮：tensor core 版）
// 每 block 一个 (序列, v 头, BV 行 V 分片)，128 线程 = 4 warp，块在时间上串行。四个矩阵乘全走 mma.sync m16n8k16
// bf16×bf16→fp32（与 fla Triton 版同口径：h 状态 fp32 常驻，参与乘法时转 bf16；u/w/kg/Aq 在 wy 侧是 fp32，装进 smem 时转 bf16）。
//   (1) o1 = q·hᵀ        [BT×K]·[K×BV]   A = sq（bf16 原样），B = sh16 按 [v][k] 存 = col-major B，直接 32 位读
//   (2) wh = w·hᵀ        同上，A = sw；v' = u − wh，按 [v][i] 转置写 bf16 smem（svT）
//   (3) o  = o1·dec + Aq·v'  A = sAq [i][j]，B = svT [v][j] col-major，直接读
//   (4) h  = h·glast + v'ᵀ·kg  A = svT [v][j]（行 = v，k 维 = j），B = skg [j][k] 行主序 → ldmatrix.trans
// h 的 fp32 主本留在寄存器（每 warp 固定持 (BV/16)·(K/8)/NW 个 16×8 tile，跨 chunk 不动），smem 只放 bf16 影子。
// 行跨度 +8 个 bf16（16 B）避免 ldmatrix / 32 位列读的 bank 冲突。BV 由 host 给（ArchProfile::gdn_chunk_ho_bv）。
constexpr int kHoWarps = 4;
template <int BT, int BV, int K>
struct HoSmem {
  static constexpr int KS = K + 8;    // [BT][KS] / [BV][KS]
  static constexpr int TS = BT + 8;   // [BT][TS] / [BV][TS]
  static constexpr int off_h16 = 0;                       // [BV][KS]  bf16
  static constexpr int off_q = off_h16 + BV * KS;         // [BT][KS]
  static constexpr int off_w = off_q + BT * KS;           // [BT][KS]
  static constexpr int off_kg = off_w + BT * KS;          // [BT][KS]
  static constexpr int off_aq = off_kg + BT * KS;         // [BT][TS]
  static constexpr int off_vT = off_aq + BT * TS;         // [BV][TS]
  static constexpr int elems = off_vT + BV * TS;
  static constexpr int bytes = elems * 2 + 2 * BT * 4;   // + sdec[BT] + srow[BT] f32
};

template <int BT, int BV, int K, int PF>
__global__ void __launch_bounds__(kHoWarps * 32) ho_mma_kernel(
    const bf16* __restrict__ qn, const float* __restrict__ gc, const float* __restrict__ aq, const float* __restrict__ u,
    const float* __restrict__ w, const float* __restrict__ kg, bf16* __restrict__ out, float* __restrict__ state,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_nt, const int* __restrict__ chunk_off,
    const int* __restrict__ state_indices, int64_t state_stride_tok, int H, int HV, int V, int group) {
  using S = HoSmem<BT, BV, K>;
  constexpr int KS = S::KS, TS = S::TS, NW = kHoWarps;
  constexpr int MT_T = BT / 16, MT_V = BV / 16;        // 行 tile 数（o 的行 = token；h 的行 = v）
  constexpr int NT_V = BV / 8, NT_K = K / 8;           // 列 tile 数
  constexpr int KSTEP_K = K / 16, KSTEP_T = BT / 16;   // k 维步数
  static_assert(BT % 16 == 0 && BV % 16 == 0 && K % 16 == 0 && MT_T == NW, "BT=64 → 4 warp 各持一个 16 行 tile");
  static_assert((MT_V * NT_K) % (2 * NW) == 0, "h tile 对要能均分到 warp");
  constexpr int H_TILES = MT_V * NT_K / NW;            // 每 warp 持的 h tile 数（成对：相邻两个 k 列 tile）

  const int v0 = blockIdx.x * BV;
  const int hv = blockIdx.y;
  const int h = hv / group;
  const int seq = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t4 = lane & 3;
  // h tile 编号 hi → (mt, ntk)：对号 pr = hi/2，全局对号 gp = warp + pr·NW，mt = gp % MT_V，ntk = (gp / MT_V)·2 + (hi&1)
  // 相邻 ntk 成对，(4) 阶段一条 ldmatrix.x4.trans 同时取两列 tile 的 B。
  auto h_tile = [&](int hi, int& mt, int& ntk) {
    const int gp = warp + (hi >> 1) * NW;
    mt = gp % MT_V;
    ntk = (gp / MT_V) * 2 + (hi & 1);
  };

  const int slot = state_indices ? state_indices[seq] : seq;
  const int bos = cu_seqlens[seq], eos = cu_seqlens[seq + 1];
  const int len_seq = eos - bos;
  if (len_seq <= 0 || slot < 0) return;
  const int nt = cu_nt[seq];
  const long kstride = (long)H * K;

  extern __shared__ __align__(16) unsigned char smraw[];
  bf16* sm16 = reinterpret_cast<bf16*>(smraw);
  bf16* sh16 = sm16 + S::off_h16;
  bf16* sq = sm16 + S::off_q;
  bf16* sw = sm16 + S::off_w;
  bf16* skg = sm16 + S::off_kg;
  bf16* sAq = sm16 + S::off_aq;
  bf16* svT = sm16 + S::off_vT;
  float* sdec = reinterpret_cast<float*>(sm16 + S::elems);  // [BT] exp2(gc)


  float* hslot = state + (long)slot * state_stride_tok + (long)hv * V * K;

  // h 主本 → 寄存器 tile。tile 号 hi ∈ [0, H_TILES)：全局 tile 序号 = warp + hi·NW，拆成 (mt, ntk)。
  float hacc[H_TILES][4];
#pragma unroll
  for (int hi = 0; hi < H_TILES; ++hi) {
    int mt, ntk;
    h_tile(hi, mt, ntk);
    const int vr = v0 + mt * 16 + g, kc = ntk * 8 + 2 * t4;
    const float2 a = *reinterpret_cast<const float2*>(hslot + (long)vr * K + kc);
    const float2 b = *reinterpret_cast<const float2*>(hslot + (long)(vr + 8) * K + kc);
    hacc[hi][0] = a.x; hacc[hi][1] = a.y; hacc[hi][2] = b.x; hacc[hi][3] = b.y;
    sts_u32(sh16 + (mt * 16 + g) * KS + kc, pack_bf16x2(a.x, a.y));
    sts_u32(sh16 + (mt * 16 + g + 8) * KS + kc, pack_bf16x2(b.x, b.y));
  }
  __syncthreads();

  // ---- 装载（r9 第 1 轮：寄存器预取）。chunk ci 的 q / w / kg / Aq / dec 先进寄存器（fetch），等本 chunk 四段算完、
  // smem 不再被读时再转 bf16 写进 smem（stage）。chunk ci+1 的 fetch 在 ci 的计算前发出，全局访存延迟与 mma 重叠；
  // 不加第二份 smem（本 kernel 已 1 block/SM）。每线程向量数由 BT / K / 线程数编译期算出，不是调优数值。
  constexpr int QV = BT * (K / 8) / (NW * 32);   // q / w / kg 的 16 B 向量数（8 个 bf16 或 8 个 f32）
  constexpr int AV = BT * (BT / 8) / (NW * 32);  // Aq 向量数
  constexpr int DV = (BT + NW * 32 - 1) / (NW * 32);
  static_assert(BT * (K / 8) % (NW * 32) == 0 && BT * (BT / 8) % (NW * 32) == 0, "装载向量数要整除线程数");
  uint4 rq[QV];
  float4 rw[QV][2], rg[QV][2], ra[AV][2];
  float rd[DV];
  auto fetch = [&](int ci) {
    const int c = chunk_off[seq] + ci;
    const int len = min(BT, len_seq - ci * BT);
    const long row0 = bos + (long)ci * BT;
    const float* aqc = aq + ((long)c * HV + hv) * BT * BT;
#pragma unroll
    for (int d = 0; d < DV; ++d) {
      const int i = tid + d * NW * 32;
      rd[d] = (i < BT && i < len) ? exp2f(gc[(row0 + i) * HV + hv]) : 0.f;
    }
#pragma unroll
    for (int vv = 0; vv < QV; ++vv) {
      const int idx = tid + vv * NW * 32;
      const int i = idx / (K / 8), k8 = (idx % (K / 8)) * 8;
      rq[vv] = make_uint4(0, 0, 0, 0);
      rw[vv][0] = rw[vv][1] = rg[vv][0] = rg[vv][1] = make_float4(0.f, 0.f, 0.f, 0.f);
      if (i < len) {
        rq[vv] = *reinterpret_cast<const uint4*>(qn + (row0 + i) * kstride + (long)h * K + k8);
        const float* wr = w + ((row0 + i) * HV + hv) * K + k8;
        const float* gr = kg + ((row0 + i) * HV + hv) * K + k8;
        rw[vv][0] = *reinterpret_cast<const float4*>(wr);
        rw[vv][1] = *reinterpret_cast<const float4*>(wr + 4);
        rg[vv][0] = *reinterpret_cast<const float4*>(gr);
        rg[vv][1] = *reinterpret_cast<const float4*>(gr + 4);
      }
    }
#pragma unroll
    for (int vv = 0; vv < AV; ++vv) {
      const int idx = tid + vv * NW * 32;
      const int i = idx / (BT / 8), j8 = (idx % (BT / 8)) * 8;
      ra[vv][0] = *reinterpret_cast<const float4*>(aqc + i * BT + j8);
      ra[vv][1] = *reinterpret_cast<const float4*>(aqc + i * BT + j8 + 4);
    }
  };
  auto stage = [&]() {
#pragma unroll
    for (int d = 0; d < DV; ++d) {
      const int i = tid + d * NW * 32;
      if (i < BT) sdec[i] = rd[d];
    }
#pragma unroll
    for (int vv = 0; vv < QV; ++vv) {
      const int idx = tid + vv * NW * 32;
      const int i = idx / (K / 8), k8 = (idx % (K / 8)) * 8;
      *reinterpret_cast<uint4*>(sq + i * KS + k8) = rq[vv];
      *reinterpret_cast<uint4*>(sw + i * KS + k8) =
          make_uint4(pack_bf16x2(rw[vv][0].x, rw[vv][0].y), pack_bf16x2(rw[vv][0].z, rw[vv][0].w),
                     pack_bf16x2(rw[vv][1].x, rw[vv][1].y), pack_bf16x2(rw[vv][1].z, rw[vv][1].w));
      *reinterpret_cast<uint4*>(skg + i * KS + k8) =
          make_uint4(pack_bf16x2(rg[vv][0].x, rg[vv][0].y), pack_bf16x2(rg[vv][0].z, rg[vv][0].w),
                     pack_bf16x2(rg[vv][1].x, rg[vv][1].y), pack_bf16x2(rg[vv][1].z, rg[vv][1].w));
    }
#pragma unroll
    for (int vv = 0; vv < AV; ++vv) {
      const int idx = tid + vv * NW * 32;
      const int i = idx / (BT / 8), j8 = (idx % (BT / 8)) * 8;
      *reinterpret_cast<uint4*>(sAq + i * TS + j8) =
          make_uint4(pack_bf16x2(ra[vv][0].x, ra[vv][0].y), pack_bf16x2(ra[vv][0].z, ra[vv][0].w),
                     pack_bf16x2(ra[vv][1].x, ra[vv][1].y), pack_bf16x2(ra[vv][1].z, ra[vv][1].w));
    }
  };
  // PF = 1：寄存器预取（下一 chunk 的全局读与本 chunk 的 mma 重叠，代价是每线程多 ~130 个寄存器，BV=32 时溢出）；
  // PF = 0：每 chunk 同步装载（第 5A 轮行为）。由 host 按 BV / 卡选（ArchProfile::gdn_chunk_ho_prefetch）。
  if (PF && nt > 0) {
    fetch(0);
    stage();
  }
  __syncthreads();

  for (int ci = 0; ci < nt; ++ci) {
    const int len = min(BT, len_seq - ci * BT);
    const long row0 = bos + (long)ci * BT;
    const bool has_next = PF && ci + 1 < nt;
    if (PF) {
      if (has_next) fetch(ci + 1);
    } else {
      fetch(ci);
      stage();
      __syncthreads();
    }

    // ---- (2) wh = w·hᵀ → v' = u − wh，转置写 svT。warp 持 token tile mt = warp，全部 NT_V 列 tile。
    {
      float wacc[NT_V][4];
#pragma unroll
      for (int n = 0; n < NT_V; ++n) { wacc[n][0] = wacc[n][1] = wacc[n][2] = wacc[n][3] = 0.f; }
      const bf16* a_base = sw + (warp * 16) * KS;
#pragma unroll
      for (int ks = 0; ks < KSTEP_K; ++ks) {
        uint32_t a[4];
        ldsm_x4(a, a_base + (lane & 15) * KS + ks * 16 + (lane >> 4) * 8);
#pragma unroll
        for (int n = 0; n < NT_V; ++n) {
          const bf16* hb = sh16 + (n * 8 + g) * KS + ks * 16 + 2 * t4;
          mma16816(wacc[n], a, lds_u32(hb), lds_u32(hb + 8));
        }
      }
      const int i0 = warp * 16 + g, i1 = i0 + 8;
#pragma unroll
      for (int n = 0; n < NT_V; ++n) {
        const int vv = n * 8 + 2 * t4;
        float u00 = 0.f, u01 = 0.f, u10 = 0.f, u11 = 0.f;
        if (i0 < len) {
          const float2 x = *reinterpret_cast<const float2*>(u + ((row0 + i0) * HV + hv) * V + v0 + vv);
          u00 = x.x; u01 = x.y;
        }
        if (i1 < len) {
          const float2 x = *reinterpret_cast<const float2*>(u + ((row0 + i1) * HV + hv) * V + v0 + vv);
          u10 = x.x; u11 = x.y;
        }
        // 行 ≥ len 的 v' 置 0（后面 Aq / kg 对应行也是 0，但 v' 本身别带垃圾）
        const float v00 = (i0 < len) ? u00 - wacc[n][0] : 0.f, v01 = (i0 < len) ? u01 - wacc[n][1] : 0.f;
        const float v10 = (i1 < len) ? u10 - wacc[n][2] : 0.f, v11 = (i1 < len) ? u11 - wacc[n][3] : 0.f;
        svT[vv * TS + i0] = to_bf16(v00);
        svT[(vv + 1) * TS + i0] = to_bf16(v01);
        svT[vv * TS + i1] = to_bf16(v10);
        svT[(vv + 1) * TS + i1] = to_bf16(v11);
      }
    }
    __syncthreads();

    // ---- (1)+(3) o = (q·hᵀ)·dec + Aq·v'，warp 持 token tile mt = warp
    {
      float oacc[NT_V][4];
#pragma unroll
      for (int n = 0; n < NT_V; ++n) { oacc[n][0] = oacc[n][1] = oacc[n][2] = oacc[n][3] = 0.f; }
      const bf16* a_base = sq + (warp * 16) * KS;
#pragma unroll
      for (int ks = 0; ks < KSTEP_K; ++ks) {
        uint32_t a[4];
        ldsm_x4(a, a_base + (lane & 15) * KS + ks * 16 + (lane >> 4) * 8);
#pragma unroll
        for (int n = 0; n < NT_V; ++n) {
          const bf16* hb = sh16 + (n * 8 + g) * KS + ks * 16 + 2 * t4;
          mma16816(oacc[n], a, lds_u32(hb), lds_u32(hb + 8));
        }
      }
      const int i0 = warp * 16 + g, i1 = i0 + 8;
      const float d0 = sdec[i0], d1 = sdec[i1];
#pragma unroll
      for (int n = 0; n < NT_V; ++n) {
        oacc[n][0] *= d0; oacc[n][1] *= d0; oacc[n][2] *= d1; oacc[n][3] *= d1;
      }
      const bf16* aq_base = sAq + (warp * 16) * TS;
#pragma unroll
      for (int ks = 0; ks < KSTEP_T; ++ks) {
        uint32_t a[4];
        ldsm_x4(a, aq_base + (lane & 15) * TS + ks * 16 + (lane >> 4) * 8);
#pragma unroll
        for (int n = 0; n < NT_V; ++n) {
          const bf16* vb = svT + (n * 8 + g) * TS + ks * 16 + 2 * t4;
          mma16816(oacc[n], a, lds_u32(vb), lds_u32(vb + 8));
        }
      }
#pragma unroll
      for (int n = 0; n < NT_V; ++n) {
        const int vv = v0 + n * 8 + 2 * t4;
        if (i0 < len)
          *reinterpret_cast<uint32_t*>(out + ((row0 + i0) * HV + hv) * V + vv) = pack_bf16x2(oacc[n][0], oacc[n][1]);
        if (i1 < len)
          *reinterpret_cast<uint32_t*>(out + ((row0 + i1) * HV + hv) * V + vv) = pack_bf16x2(oacc[n][2], oacc[n][3]);
      }
    }

    // ---- (4) h = h·glast + v'ᵀ·kg：每 warp 更新自己持的 h tile（A = svT 行 tile mt，B = skg 列 tile ntk，k 维 = token）
    {
      const float glast = sdec[len - 1];
#pragma unroll
      for (int hi = 0; hi < H_TILES; ++hi) {
        hacc[hi][0] *= glast; hacc[hi][1] *= glast; hacc[hi][2] *= glast; hacc[hi][3] *= glast;
      }
#pragma unroll
      for (int ks = 0; ks < KSTEP_T; ++ks) {
        // 每个 mt 一份 A（最多 MT_V 个），每个 ntk 一份 B
        uint32_t afr[MT_V][4];
#pragma unroll
        for (int mt = 0; mt < MT_V; ++mt)
          ldsm_x4(afr[mt], svT + (mt * 16 + (lane & 15)) * TS + ks * 16 + (lane >> 4) * 8);
#pragma unroll
        for (int hi = 0; hi < H_TILES; hi += 2) {
          int mt, ntk;
          h_tile(hi, mt, ntk);
          // B[k=j][n] = skg[j][ntk·8 + n]，行主序 → ldmatrix.x4.trans：lane&15 选 j 行，lane>>4 选列 tile ntk / ntk+1
          uint32_t b[4];
          ldsm_x4_t(b, skg + (ks * 16 + (lane & 15)) * KS + ntk * 8 + (lane >> 4) * 8);
          mma16816(hacc[hi], afr[mt], b[0], b[1]);
          mma16816(hacc[hi + 1], afr[mt], b[2], b[3]);
        }
      }
    }
    __syncthreads();  // svT / sh16 读完
    // sh16 ← 新 h（bf16 影子）
#pragma unroll
    for (int hi = 0; hi < H_TILES; ++hi) {
      int mt, ntk;
      h_tile(hi, mt, ntk);
      const int kc = ntk * 8 + 2 * t4;
      sts_u32(sh16 + (mt * 16 + g) * KS + kc, pack_bf16x2(hacc[hi][0], hacc[hi][1]));
      sts_u32(sh16 + (mt * 16 + g + 8) * KS + kc, pack_bf16x2(hacc[hi][2], hacc[hi][3]));
    }
    if (has_next) stage();  // sq / sw / skg / sAq / sdec 在 (4) 之前已读完
    __syncthreads();
  }

  // 末态回写 fp32
#pragma unroll
  for (int hi = 0; hi < H_TILES; ++hi) {
    int mt, ntk;
    h_tile(hi, mt, ntk);
    const int vr = v0 + mt * 16 + g, kc = ntk * 8 + 2 * t4;
    *reinterpret_cast<float2*>(hslot + (long)vr * K + kc) = make_float2(hacc[hi][0], hacc[hi][1]);
    *reinterpret_cast<float2*>(hslot + (long)(vr + 8) * K + kc) = make_float2(hacc[hi][2], hacc[hi][3]);
  }
}

// ================================================================ ho 两段版（r9 第 2 轮：h-pass + o-pass）
// 融合版每 block 串行走整条序列的所有 chunk，单长序列只有 HV·V/BV 个 block（27B 形状 192 / 384），铺不满大卡且每 chunk
// 四个矩阵乘串在一条延迟链上。拆成：
//   h-pass  grid (V/BV, HV, nseq)：只做 v' = u − w·hᵀ 与 h ← h·glast + v'ᵀ·kg（两个乘法），每 chunk 把入口 h（bf16）写到
//           h16 [num_chunks, HV, V, K]、v'（bf16）写到 v16 [T, HV, V]；smem 只放 h 影子 + w / kg（比融合版少 q / Aq），
//           占用翻倍；仍串行但算量减半。
//   o-pass  grid (num_chunks, HV)：全并行 o = (q·h_cᵀ)·dec + Aq·v'_c，每 block 读 h16 一块 + q / Aq / v'，纯 mma。
// 数值口径与融合版相同：h / v' / Aq / q 参与 mma 时均为 bf16，fp32 累加（parity 容差同 2e-3 / 1e-2）。
template <int BT, int BV, int K>
struct HpSmem {
  static constexpr int KS = K + 8, TS = BT + 8;
  static constexpr int off_h16 = 0;                  // [BV][KS]
  static constexpr int off_w = off_h16 + BV * KS;    // [BT][KS]
  static constexpr int off_kg = off_w + BT * KS;     // [BT][KS]
  static constexpr int off_vT = off_kg + BT * KS;    // [BV][TS]
  static constexpr int elems = off_vT + BV * TS;
  static constexpr int bytes = elems * 2 + BT * 4;   // + sdec[BT]
};

template <int BT, int BV, int K, int PF>
__global__ void __launch_bounds__(kHoWarps * 32) ho_hpass_kernel(
    const float* __restrict__ gc, const float* __restrict__ u, const float* __restrict__ w, const float* __restrict__ kg,
    bf16* __restrict__ h16, bf16* __restrict__ v16, float* __restrict__ state, const int* __restrict__ cu_seqlens,
    const int* __restrict__ cu_nt, const int* __restrict__ chunk_off, const int* __restrict__ state_indices,
    int64_t state_stride_tok, int HV, int V) {
  using S = HpSmem<BT, BV, K>;
  constexpr int KS = S::KS, TS = S::TS, NW = kHoWarps;
  constexpr int MT_V = BV / 16, NT_V = BV / 8, NT_K = K / 8, KSTEP_K = K / 16, KSTEP_T = BT / 16;
  static_assert(BT % 16 == 0 && BV % 16 == 0 && K % 16 == 0 && BT / 16 == NW, "BT=64 → 4 warp 各持一个 16 行 tile");
  static_assert((MT_V * NT_K) % (2 * NW) == 0, "h tile 对要能均分到 warp");
  constexpr int H_TILES = MT_V * NT_K / NW;
  constexpr int QV = BT * (K / 8) / (NW * 32);
  constexpr int DV = (BT + NW * 32 - 1) / (NW * 32);
  static_assert(BT * (K / 8) % (NW * 32) == 0, "装载向量数要整除线程数");

  const int v0 = blockIdx.x * BV, hv = blockIdx.y, seq = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  auto h_tile = [&](int hi, int& mt, int& ntk) {
    const int gp = warp + (hi >> 1) * NW;
    mt = gp % MT_V;
    ntk = (gp / MT_V) * 2 + (hi & 1);
  };
  const int slot = state_indices ? state_indices[seq] : seq;
  const int bos = cu_seqlens[seq], eos = cu_seqlens[seq + 1];
  const int len_seq = eos - bos;
  if (len_seq <= 0 || slot < 0) return;
  const int nt = cu_nt[seq];

  extern __shared__ __align__(16) unsigned char smraw[];
  bf16* sm16 = reinterpret_cast<bf16*>(smraw);
  bf16* sh16 = sm16 + S::off_h16;
  bf16* sw = sm16 + S::off_w;
  bf16* skg = sm16 + S::off_kg;
  bf16* svT = sm16 + S::off_vT;
  float* sdec = reinterpret_cast<float*>(sm16 + S::elems);
  float* hslot = state + (long)slot * state_stride_tok + (long)hv * V * K;

  float hacc[H_TILES][4];
#pragma unroll
  for (int hi = 0; hi < H_TILES; ++hi) {
    int mt, ntk;
    h_tile(hi, mt, ntk);
    const int vr = v0 + mt * 16 + g, kc = ntk * 8 + 2 * t4;
    const float2 a = *reinterpret_cast<const float2*>(hslot + (long)vr * K + kc);
    const float2 b = *reinterpret_cast<const float2*>(hslot + (long)(vr + 8) * K + kc);
    hacc[hi][0] = a.x; hacc[hi][1] = a.y; hacc[hi][2] = b.x; hacc[hi][3] = b.y;
    sts_u32(sh16 + (mt * 16 + g) * KS + kc, pack_bf16x2(a.x, a.y));
    sts_u32(sh16 + (mt * 16 + g + 8) * KS + kc, pack_bf16x2(b.x, b.y));
  }

  float4 rw[QV][2], rg[QV][2];
  float rd[DV];
  auto fetch = [&](int ci) {
    const int len = min(BT, len_seq - ci * BT);
    const long row0 = bos + (long)ci * BT;
#pragma unroll
    for (int d = 0; d < DV; ++d) {
      const int i = tid + d * NW * 32;
      rd[d] = (i < BT && i < len) ? exp2f(gc[(row0 + i) * HV + hv]) : 0.f;
    }
#pragma unroll
    for (int vv = 0; vv < QV; ++vv) {
      const int idx = tid + vv * NW * 32;
      const int i = idx / (K / 8), k8 = (idx % (K / 8)) * 8;
      rw[vv][0] = rw[vv][1] = rg[vv][0] = rg[vv][1] = make_float4(0.f, 0.f, 0.f, 0.f);
      if (i < len) {
        const float* wr = w + ((row0 + i) * HV + hv) * K + k8;
        const float* gr = kg + ((row0 + i) * HV + hv) * K + k8;
        rw[vv][0] = *reinterpret_cast<const float4*>(wr);
        rw[vv][1] = *reinterpret_cast<const float4*>(wr + 4);
        rg[vv][0] = *reinterpret_cast<const float4*>(gr);
        rg[vv][1] = *reinterpret_cast<const float4*>(gr + 4);
      }
    }
  };
  auto stage = [&]() {
#pragma unroll
    for (int d = 0; d < DV; ++d) {
      const int i = tid + d * NW * 32;
      if (i < BT) sdec[i] = rd[d];
    }
#pragma unroll
    for (int vv = 0; vv < QV; ++vv) {
      const int idx = tid + vv * NW * 32;
      const int i = idx / (K / 8), k8 = (idx % (K / 8)) * 8;
      *reinterpret_cast<uint4*>(sw + i * KS + k8) =
          make_uint4(pack_bf16x2(rw[vv][0].x, rw[vv][0].y), pack_bf16x2(rw[vv][0].z, rw[vv][0].w),
                     pack_bf16x2(rw[vv][1].x, rw[vv][1].y), pack_bf16x2(rw[vv][1].z, rw[vv][1].w));
      *reinterpret_cast<uint4*>(skg + i * KS + k8) =
          make_uint4(pack_bf16x2(rg[vv][0].x, rg[vv][0].y), pack_bf16x2(rg[vv][0].z, rg[vv][0].w),
                     pack_bf16x2(rg[vv][1].x, rg[vv][1].y), pack_bf16x2(rg[vv][1].z, rg[vv][1].w));
    }
  };
  if (PF && nt > 0) {
    fetch(0);
    stage();
  }
  __syncthreads();

  for (int ci = 0; ci < nt; ++ci) {
    const int c = chunk_off[seq] + ci;
    const int len = min(BT, len_seq - ci * BT);
    const long row0 = bos + (long)ci * BT;
    const bool has_next = PF && ci + 1 < nt;
    if (PF) {
      if (has_next) fetch(ci + 1);
    } else {
      fetch(ci);
      stage();
      __syncthreads();
    }

    // 入口 h（bf16 影子）→ h16[c, hv, v0.., :]：16 B 向量拷
    {
      bf16* hc = h16 + (((long)c * HV + hv) * V + v0) * K;
      for (int idx = tid; idx < BV * (K / 8); idx += NW * 32) {
        const int vr = idx / (K / 8), k8 = (idx % (K / 8)) * 8;
        *reinterpret_cast<uint4*>(hc + (long)vr * K + k8) = *reinterpret_cast<const uint4*>(sh16 + vr * KS + k8);
      }
    }

    // v' = u − w·hᵀ → svT（bf16 转置）+ v16
    {
      float wacc[NT_V][4];
#pragma unroll
      for (int n = 0; n < NT_V; ++n) { wacc[n][0] = wacc[n][1] = wacc[n][2] = wacc[n][3] = 0.f; }
      const bf16* a_base = sw + (warp * 16) * KS;
#pragma unroll
      for (int ks = 0; ks < KSTEP_K; ++ks) {
        uint32_t a[4];
        ldsm_x4(a, a_base + (lane & 15) * KS + ks * 16 + (lane >> 4) * 8);
#pragma unroll
        for (int n = 0; n < NT_V; ++n) {
          const bf16* hb = sh16 + (n * 8 + g) * KS + ks * 16 + 2 * t4;
          mma16816(wacc[n], a, lds_u32(hb), lds_u32(hb + 8));
        }
      }
      const int i0 = warp * 16 + g, i1 = i0 + 8;
#pragma unroll
      for (int n = 0; n < NT_V; ++n) {
        const int vv = n * 8 + 2 * t4;
        float v00 = 0.f, v01 = 0.f, v10 = 0.f, v11 = 0.f;
        if (i0 < len) {
          const float2 x = *reinterpret_cast<const float2*>(u + ((row0 + i0) * HV + hv) * V + v0 + vv);
          v00 = x.x - wacc[n][0]; v01 = x.y - wacc[n][1];
          *reinterpret_cast<uint32_t*>(v16 + ((row0 + i0) * HV + hv) * V + v0 + vv) = pack_bf16x2(v00, v01);
        }
        if (i1 < len) {
          const float2 x = *reinterpret_cast<const float2*>(u + ((row0 + i1) * HV + hv) * V + v0 + vv);
          v10 = x.x - wacc[n][2]; v11 = x.y - wacc[n][3];
          *reinterpret_cast<uint32_t*>(v16 + ((row0 + i1) * HV + hv) * V + v0 + vv) = pack_bf16x2(v10, v11);
        }
        svT[vv * TS + i0] = to_bf16(v00);
        svT[(vv + 1) * TS + i0] = to_bf16(v01);
        svT[vv * TS + i1] = to_bf16(v10);
        svT[(vv + 1) * TS + i1] = to_bf16(v11);
      }
    }
    __syncthreads();

    // h = h·glast + v'ᵀ·kg
    {
      const float glast = sdec[len - 1];
#pragma unroll
      for (int hi = 0; hi < H_TILES; ++hi) {
        hacc[hi][0] *= glast; hacc[hi][1] *= glast; hacc[hi][2] *= glast; hacc[hi][3] *= glast;
      }
#pragma unroll
      for (int ks = 0; ks < KSTEP_T; ++ks) {
        uint32_t afr[MT_V][4];
#pragma unroll
        for (int mt = 0; mt < MT_V; ++mt)
          ldsm_x4(afr[mt], svT + (mt * 16 + (lane & 15)) * TS + ks * 16 + (lane >> 4) * 8);
#pragma unroll
        for (int hi = 0; hi < H_TILES; hi += 2) {
          int mt, ntk;
          h_tile(hi, mt, ntk);
          uint32_t b[4];
          ldsm_x4_t(b, skg + (ks * 16 + (lane & 15)) * KS + ntk * 8 + (lane >> 4) * 8);
          mma16816(hacc[hi], afr[mt], b[0], b[1]);
          mma16816(hacc[hi + 1], afr[mt], b[2], b[3]);
        }
      }
    }
    __syncthreads();  // svT / sh16 / sw / skg 读完
#pragma unroll
    for (int hi = 0; hi < H_TILES; ++hi) {
      int mt, ntk;
      h_tile(hi, mt, ntk);
      const int kc = ntk * 8 + 2 * t4;
      sts_u32(sh16 + (mt * 16 + g) * KS + kc, pack_bf16x2(hacc[hi][0], hacc[hi][1]));
      sts_u32(sh16 + (mt * 16 + g + 8) * KS + kc, pack_bf16x2(hacc[hi][2], hacc[hi][3]));
    }
    if (has_next) stage();
    __syncthreads();
  }

#pragma unroll
  for (int hi = 0; hi < H_TILES; ++hi) {
    int mt, ntk;
    h_tile(hi, mt, ntk);
    const int vr = v0 + mt * 16 + g, kc = ntk * 8 + 2 * t4;
    *reinterpret_cast<float2*>(hslot + (long)vr * K + kc) = make_float2(hacc[hi][0], hacc[hi][1]);
    *reinterpret_cast<float2*>(hslot + (long)(vr + 8) * K + kc) = make_float2(hacc[hi][2], hacc[hi][3]);
  }
}

// o-pass：每 block 一个 (chunk, v 头, VT 列 V 分片)，每 warp 持 16 行 token × 64 列 v（VT=128 → 8 warp；VT ≤ 64 → 4 warp）。
// VT 由 host 按 V 选（ArchProfile::gdn_chunk_opass_vt）。3090 实测（r9 第 4 轮）VT=128（78 KB smem，1 block/SM）比 VT=64
// （49 KB，2 block/SM）快 10%：q / Aq 只装一次 > occupancy。
template <int BT, int K, int VT>
struct OpSmem {
  static constexpr int KS = K + 8, TS = BT + 8, VS = VT + 8;
  static constexpr int off_h = 0;                  // [VT][KS] bf16 h_c 分片（[v][k]）
  static constexpr int off_q = off_h + VT * KS;    // [BT][KS]
  static constexpr int off_aq = off_q + BT * KS;   // [BT][TS]
  static constexpr int off_v = off_aq + BT * TS;   // [BT][VS]  v' 分片（行主序，B 操作数走 ldmatrix.trans）
  static constexpr int elems = off_v + BT * VS;
  static constexpr int bytes = elems * 2 + BT * 4;  // + sdec
  static constexpr int VW = (VT <= 64) ? VT : 64;     // 每 warp 持的 v 列数（mma 累加器 ≤ 8 个 tile）
  static constexpr int NWARPS = (BT / 16) * (VT / VW);  // VT=128 → 8 warp（两个 64 列组），VT ≤ 64 → 4 warp
};

template <int BT, int K, int VT>
__global__ void __launch_bounds__(OpSmem<BT, K, VT>::NWARPS * 32) ho_opass_kernel(
    const bf16* __restrict__ qn, const float* __restrict__ gc, const float* __restrict__ aq, const bf16* __restrict__ h16,
    const bf16* __restrict__ v16, bf16* __restrict__ out, const int* __restrict__ cu_seqlens,
    const int* __restrict__ chunk_tab, int H, int HV, int V, int group) {
  using S = OpSmem<BT, K, VT>;
  constexpr int KS = S::KS, TS = S::TS, VS = S::VS, NW = S::NWARPS, VW = S::VW;
  constexpr int NT = VW / 8, KSTEP_K = K / 16, KSTEP_T = BT / 16;
  static_assert(VT % VW == 0 && VW % 16 == 0 && BT % 16 == 0 && K % 16 == 0 && NT % 2 == 0, "tile 形状");

  const int c = blockIdx.x, hv = blockIdx.y, v0 = blockIdx.z * VT, h = hv / group;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  const int mt = warp % (BT / 16), vb = (warp / (BT / 16)) * VW;  // 行 tile / 本 warp 的 v 列起点（块内）
  const int seq = chunk_tab[2 * c], ci = chunk_tab[2 * c + 1];
  const int bos = cu_seqlens[seq], eos = cu_seqlens[seq + 1];
  const int len = min(BT, eos - bos - ci * BT);
  const long row0 = bos + (long)ci * BT;
  const long kstride = (long)H * K;

  extern __shared__ __align__(16) unsigned char smraw[];
  bf16* sm16 = reinterpret_cast<bf16*>(smraw);
  bf16* sh = sm16 + S::off_h;
  bf16* sq = sm16 + S::off_q;
  bf16* sAq = sm16 + S::off_aq;
  bf16* sv = sm16 + S::off_v;
  float* sdec = reinterpret_cast<float*>(sm16 + S::elems);

  for (int i = tid; i < BT; i += NW * 32) sdec[i] = (i < len) ? exp2f(gc[(row0 + i) * HV + hv]) : 0.f;
  {
    const bf16* hc = h16 + (((long)c * HV + hv) * V + v0) * K;
    for (int idx = tid; idx < VT * (K / 8); idx += NW * 32) {
      const int vr = idx / (K / 8), k8 = (idx % (K / 8)) * 8;
      *reinterpret_cast<uint4*>(sh + vr * KS + k8) = *reinterpret_cast<const uint4*>(hc + (long)vr * K + k8);
    }
  }
  for (int idx = tid; idx < BT * (K / 8); idx += NW * 32) {
    const int i = idx / (K / 8), k8 = (idx % (K / 8)) * 8;
    uint4 qv = make_uint4(0, 0, 0, 0);
    if (i < len) qv = *reinterpret_cast<const uint4*>(qn + (row0 + i) * kstride + (long)h * K + k8);
    *reinterpret_cast<uint4*>(sq + i * KS + k8) = qv;
  }
  for (int idx = tid; idx < BT * (VT / 8); idx += NW * 32) {
    const int i = idx / (VT / 8), v8 = (idx % (VT / 8)) * 8;
    uint4 vv = make_uint4(0, 0, 0, 0);
    if (i < len) vv = *reinterpret_cast<const uint4*>(v16 + ((row0 + i) * HV + hv) * V + v0 + v8);
    *reinterpret_cast<uint4*>(sv + i * VS + v8) = vv;
  }
  {
    const float* aqc = aq + ((long)c * HV + hv) * BT * BT;
    for (int idx = tid; idx < BT * (BT / 8); idx += NW * 32) {
      const int i = idx / (BT / 8), j8 = (idx % (BT / 8)) * 8;
      const float4 a0 = *reinterpret_cast<const float4*>(aqc + i * BT + j8);
      const float4 a1 = *reinterpret_cast<const float4*>(aqc + i * BT + j8 + 4);
      *reinterpret_cast<uint4*>(sAq + i * TS + j8) =
          make_uint4(pack_bf16x2(a0.x, a0.y), pack_bf16x2(a0.z, a0.w), pack_bf16x2(a1.x, a1.y), pack_bf16x2(a1.z, a1.w));
    }
  }
  __syncthreads();

  float oacc[NT][4];
#pragma unroll
  for (int n = 0; n < NT; ++n) { oacc[n][0] = oacc[n][1] = oacc[n][2] = oacc[n][3] = 0.f; }
  // o1 = q·hᵀ：A = sq 行 tile mt，B = sh [v][k]（col-major B，直接 32 位读）
#pragma unroll
  for (int ks = 0; ks < KSTEP_K; ++ks) {
    uint32_t a[4];
    ldsm_x4(a, sq + (mt * 16 + (lane & 15)) * KS + ks * 16 + (lane >> 4) * 8);
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const bf16* hb = sh + (vb + n * 8 + g) * KS + ks * 16 + 2 * t4;
      mma16816(oacc[n], a, lds_u32(hb), lds_u32(hb + 8));
    }
  }
  const int i0 = mt * 16 + g, i1 = i0 + 8;
  const float d0 = sdec[i0], d1 = sdec[i1];
#pragma unroll
  for (int n = 0; n < NT; ++n) {
    oacc[n][0] *= d0; oacc[n][1] *= d0; oacc[n][2] *= d1; oacc[n][3] *= d1;
  }
  // + Aq·v'：A = sAq 行 tile mt，B = sv [j][v] 行主序 → ldmatrix.trans
#pragma unroll
  for (int ks = 0; ks < KSTEP_T; ++ks) {
    uint32_t a[4];
    ldsm_x4(a, sAq + (mt * 16 + (lane & 15)) * TS + ks * 16 + (lane >> 4) * 8);
#pragma unroll
    for (int n = 0; n < NT; n += 2) {
      uint32_t b[4];
      ldsm_x4_t(b, sv + (ks * 16 + (lane & 15)) * VS + vb + n * 8 + (lane >> 4) * 8);
      mma16816(oacc[n], a, b[0], b[1]);
      mma16816(oacc[n + 1], a, b[2], b[3]);
    }
  }
#pragma unroll
  for (int n = 0; n < NT; ++n) {
    const int vv = v0 + vb + n * 8 + 2 * t4;
    if (i0 < len)
      *reinterpret_cast<uint32_t*>(out + ((row0 + i0) * HV + hv) * V + vv) = pack_bf16x2(oacc[n][0], oacc[n][1]);
    if (i1 < len)
      *reinterpret_cast<uint32_t*>(out + ((row0 + i1) * HV + hv) * V + vv) = pack_bf16x2(oacc[n][2], oacc[n][3]);
  }
}

// 动态 shared 上限按 (kernel, 字节数) 记忆；换了特化必须重设，否则前一次的属性不生效。
constexpr int kSmemSlots = 16;
thread_local int smem_cache[kSmemSlots][2] = {};  // 按设备生效 → 每 rank 线程一份（pitfalls 2026-10-20）
int set_smem(const void* fn, int bytes) {
  const intptr_t key = (intptr_t)fn;
  int slot = 0;
  for (; slot < kSmemSlots; ++slot) {
    if (smem_cache[slot][0] == 0 || smem_cache[slot][0] == (int)key) {
      if (smem_cache[slot][1] >= bytes) return 0;
      break;
    }
  }
  const int rc = err(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  if (rc != 0) return rc;
  slot = (slot == kSmemSlots) ? (int)((key >> 4) & (kSmemSlots - 1)) : slot;
  smem_cache[slot][0] = (int)key;
  smem_cache[slot][1] = bytes;
  return 0;
}

}  // namespace

extern "C" {

int gdn_chunk_prep(const void* mixed_qkv, const float* g, void* qn, void* kn, float* gc, const int* cu_seqlens,
                   const int* chunk_tab, int64_t total, int64_t num_chunks, int64_t H, int64_t HV, int64_t K,
                   int64_t V, int64_t bt, float scale, cudaStream_t stream) {
  if (total <= 0 || num_chunks <= 0 || H <= 0 || HV <= 0 || K <= 0 || K > 256 || bt <= 0)
    return err(cudaErrorInvalidValue);
  const long qkv_row = 2L * H * K + (long)HV * V;
  const long warps = (long)total * 2 * H;
  const long blocks = (warps * 32 + 255) / 256;
  norm_kernel<<<(unsigned)blocks, 256, 0, stream>>>(static_cast<const bf16*>(mixed_qkv), static_cast<bf16*>(qn),
                                                    static_cast<bf16*>(kn), (int)total, (int)H, (int)K, qkv_row,
                                                    scale);
  const long n2 = num_chunks * HV;
  cumsum_kernel<<<(unsigned)((n2 + 255) / 256), 256, 0, stream>>>(g, gc, cu_seqlens, chunk_tab, (int)num_chunks,
                                                                   (int)HV, (int)bt);
  return last_err();
}

int gdn_chunk_wy(const void* qn, const void* kn, const void* mixed_qkv, const float* gc, const float* beta,
                 float* aq, float* u, float* w, float* kg, const int* cu_seqlens, const int* chunk_tab,
                 int64_t num_chunks, int64_t H, int64_t HV, int64_t K, int64_t V, int64_t bt, int64_t wy_mode,
                 cudaStream_t stream) {
  if (num_chunks <= 0 || H <= 0 || HV <= 0 || HV % H != 0 || K <= 0 || V <= 0) return err(cudaErrorInvalidValue);
  const int group = (int)(HV / H);
  dim3 grid((unsigned)num_chunks, (unsigned)HV);
  if (bt != 64) return err(cudaErrorInvalidValue);
  constexpr int BT = 64;
  // wy_mode 1 → tensor core 版 + 分块三角逆（r9 第 3 轮）；2 → tensor core 版 + 64 步前代（第 5A 轮）；0 → 老 fp32 版
  if (wy_mode == 1 || wy_mode == 2) {
    if (V % 8 != 0) return err(cudaErrorInvalidValue);
#define LAUNCH_WY_MMA1(KK, TRI)                                                                                 \
  {                                                                                                             \
    constexpr int smem = WySmem<BT, KK>::bytes;                                                                 \
    if (const int rc = set_smem(reinterpret_cast<const void*>(&wy_mma_kernel<BT, KK, TRI>), smem)) return rc;   \
    wy_mma_kernel<BT, KK, TRI><<<grid, kWyWarps * 32, smem, stream>>>(                                          \
        static_cast<const bf16*>(qn), static_cast<const bf16*>(kn), static_cast<const bf16*>(mixed_qkv), gc,    \
        beta, aq, u, w, kg, cu_seqlens, chunk_tab, (int)H, (int)HV, (int)K, (int)V, group);                     \
  }
#define LAUNCH_WY_MMA(KK) \
  { if (wy_mode == 1) LAUNCH_WY_MMA1(KK, 1) else LAUNCH_WY_MMA1(KK, 0) }
    if (K == 128) { LAUNCH_WY_MMA(128) } else if (K == 64) { LAUNCH_WY_MMA(64) } else if (K == 256) { LAUNCH_WY_MMA(256) } else return err(cudaErrorInvalidValue);
#undef LAUNCH_WY_MMA
#undef LAUNCH_WY_MMA1
    return last_err();
  }
#define LAUNCH_WY(BK)                                                                                   \
  {                                                                                                     \
    const int smem = (int)((2 * BT * BK + 2 * BT * BT + 2 * BT) * sizeof(float));                       \
    if (const int rc = set_smem(reinterpret_cast<const void*>(&wy_kernel<BT, BK>), smem)) return rc;   \
    wy_kernel<BT, BK><<<grid, 256, smem, stream>>>(                                                     \
        static_cast<const bf16*>(qn), static_cast<const bf16*>(kn), static_cast<const bf16*>(mixed_qkv), \
        gc, beta, aq, u, w, kg, cu_seqlens, chunk_tab, (int)H, (int)HV, (int)K, (int)V, group);         \
  }
  if (K % 64 == 0 && V % 64 == 0) {
    LAUNCH_WY(64)
  } else {
    LAUNCH_WY(32)
  }
#undef LAUNCH_WY
  return last_err();
}

// 融合版 ho（h 递推 + o 同 block）：ho_bv ∈ {16,32} tensor core 版，0 老 fp32 版；prefetch 0/1 选 PF 特化。
static int ho_fused(const void* qn, const float* gc, const float* aq, const float* u, const float* w, const float* kg,
                    void* out, void* state, const int* cu_seqlens, const int* cu_nt, const int* chunk_off,
                    const int* state_indices, int64_t state_stride_tok, int64_t num_seqs, int64_t H, int64_t HV,
                    int64_t K, int64_t V, int64_t bt, int64_t ho_bv, int64_t prefetch, cudaStream_t stream) {
  if (num_seqs <= 0 || H <= 0 || HV <= 0 || HV % H != 0 || K <= 0 || V <= 0) return err(cudaErrorInvalidValue);
  if (bt != 64) return err(cudaErrorInvalidValue);
  constexpr int BT = 64;
  const int group = (int)(HV / H);
  // ho_bv ∈ {16, 32} → tensor core 版（K ∈ {64,128,256}，V 整除 ho_bv）；0 → 老 fp32 版
  if (ho_bv == 16 || ho_bv == 32) {
    if (V % ho_bv != 0) return err(cudaErrorInvalidValue);
    dim3 grid((unsigned)(V / ho_bv), (unsigned)HV, (unsigned)num_seqs);
#define LAUNCH_HO_MMA1(BVV, KK, PF)                                                                               \
  {                                                                                                               \
    constexpr int smem = HoSmem<BT, BVV, KK>::bytes;                                                              \
    if (const int rc = set_smem(reinterpret_cast<const void*>(&ho_mma_kernel<BT, BVV, KK, PF>), smem)) return rc; \
    ho_mma_kernel<BT, BVV, KK, PF><<<grid, kHoWarps * 32, smem, stream>>>(                                        \
        static_cast<const bf16*>(qn), gc, aq, u, w, kg, static_cast<bf16*>(out), static_cast<float*>(state),      \
        cu_seqlens, cu_nt, chunk_off, state_indices, state_stride_tok, (int)H, (int)HV, (int)V, group);           \
  }
#define LAUNCH_HO_MMA(BVV, KK) \
  { if (prefetch) LAUNCH_HO_MMA1(BVV, KK, 1) else LAUNCH_HO_MMA1(BVV, KK, 0) }
    if (ho_bv == 32) {
      if (K == 128) { LAUNCH_HO_MMA(32, 128) } else if (K == 64) { LAUNCH_HO_MMA(32, 64) } else if (K == 256) { LAUNCH_HO_MMA(32, 256) } else return err(cudaErrorInvalidValue);
    } else {
      if (K == 128) { LAUNCH_HO_MMA(16, 128) } else if (K == 64) { LAUNCH_HO_MMA(16, 64) } else if (K == 256) { LAUNCH_HO_MMA(16, 256) } else return err(cudaErrorInvalidValue);
    }
#undef LAUNCH_HO_MMA
#undef LAUNCH_HO_MMA1
    return last_err();
  }
  constexpr int BV = 16;
  if (V % BV != 0) return err(cudaErrorInvalidValue);
  dim3 grid((unsigned)(V / BV), (unsigned)HV, (unsigned)num_seqs);
  // K 是模板参数（内层可全展开）；只有 64 / 128 / 256 三档特化。
#define LAUNCH_HO(KK)                                                                                     \
  {                                                                                                       \
    const int smem = (int)((BV * (KK + 1) + BT * (KK + 1) + 2 * BT * BV + BT) * sizeof(float));           \
    if (const int rc = set_smem(reinterpret_cast<const void*>(&ho_kernel<BT, BV, KK>), smem)) return rc;  \
    ho_kernel<BT, BV, KK><<<grid, 256, smem, stream>>>(                                                   \
        static_cast<const bf16*>(qn), gc, aq, u, w, kg, static_cast<bf16*>(out), static_cast<float*>(state), \
        cu_seqlens, cu_nt, chunk_off, state_indices, state_stride_tok, (int)H, (int)HV, (int)V, group);   \
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

// 两段版：h-pass（BV ∈ {16,32}，prefetch 0/1）→ o-pass（(K, V) 特化：qwen35 128/128，另给 64/64、128/64）。
static int ho_split(const void* qn, const float* gc, const float* aq, const float* u, const float* w, const float* kg,
                    bf16* h16, bf16* v16, void* out, void* state, const int* cu_seqlens, const int* cu_nt,
                    const int* chunk_off, const int* chunk_tab, const int* state_indices, int64_t state_stride_tok,
                    int64_t num_seqs, int64_t num_chunks, int64_t H, int64_t HV, int64_t K, int64_t V, int64_t bt,
                    int64_t ho_bv, int64_t prefetch, int64_t opass_vt, cudaStream_t stream) {
  if (num_seqs <= 0 || H <= 0 || HV <= 0 || HV % H != 0 || K <= 0 || V <= 0) return err(cudaErrorInvalidValue);
  if (bt != 64 || !(ho_bv == 16 || ho_bv == 32) || V % ho_bv != 0) return err(cudaErrorInvalidValue);
  constexpr int BT = 64;
  const int group = (int)(HV / H);
  {
    dim3 grid((unsigned)(V / ho_bv), (unsigned)HV, (unsigned)num_seqs);
#define LAUNCH_HP1(BVV, KK, PF)                                                                                   \
  {                                                                                                               \
    constexpr int smem = HpSmem<BT, BVV, KK>::bytes;                                                              \
    if (const int rc = set_smem(reinterpret_cast<const void*>(&ho_hpass_kernel<BT, BVV, KK, PF>), smem)) return rc; \
    ho_hpass_kernel<BT, BVV, KK, PF><<<grid, kHoWarps * 32, smem, stream>>>(                                      \
        gc, u, w, kg, h16, v16, static_cast<float*>(state), cu_seqlens, cu_nt, chunk_off, state_indices,          \
        state_stride_tok, (int)HV, (int)V);                                                                       \
  }
#define LAUNCH_HP(BVV, KK) \
  { if (prefetch) LAUNCH_HP1(BVV, KK, 1) else LAUNCH_HP1(BVV, KK, 0) }
    if (ho_bv == 32) {
      if (K == 128) { LAUNCH_HP(32, 128) } else if (K == 64) { LAUNCH_HP(32, 64) } else if (K == 256) { LAUNCH_HP(32, 256) } else return err(cudaErrorInvalidValue);
    } else {
      if (K == 128) { LAUNCH_HP(16, 128) } else if (K == 64) { LAUNCH_HP(16, 64) } else if (K == 256) { LAUNCH_HP(16, 256) } else return err(cudaErrorInvalidValue);
    }
#undef LAUNCH_HP
#undef LAUNCH_HP1
    if (const int rc = last_err()) return rc;
  }
  {
    // o-pass 的 V 分片 VT：host 给（opass_vt ∈ {64, 32, 16}，整除 V）
    if (!(opass_vt == 128 || opass_vt == 64 || opass_vt == 32 || opass_vt == 16) || V % opass_vt != 0) return err(cudaErrorInvalidValue);
    dim3 grid((unsigned)num_chunks, (unsigned)HV, (unsigned)(V / opass_vt));
#define LAUNCH_OP(KK, VT)                                                                                     \
  {                                                                                                           \
    using S = OpSmem<BT, KK, VT>;                                                                             \
    if (const int rc = set_smem(reinterpret_cast<const void*>(&ho_opass_kernel<BT, KK, VT>), S::bytes)) return rc; \
    ho_opass_kernel<BT, KK, VT><<<grid, S::NWARPS * 32, S::bytes, stream>>>(                                   \
        static_cast<const bf16*>(qn), gc, aq, h16, v16, static_cast<bf16*>(out), cu_seqlens, chunk_tab, (int)H, \
        (int)HV, (int)V, group);                                                                              \
  }
#define LAUNCH_OP_K(KK) \
  { if (opass_vt == 128) LAUNCH_OP(KK, 128) else if (opass_vt == 64) LAUNCH_OP(KK, 64) else if (opass_vt == 32) LAUNCH_OP(KK, 32) else LAUNCH_OP(KK, 16) }
    if (K == 128) { LAUNCH_OP_K(128) } else if (K == 64) { LAUNCH_OP_K(64) } else if (K == 256) { LAUNCH_OP_K(256) } else return err(cudaErrorInvalidValue);
#undef LAUNCH_OP_K
#undef LAUNCH_OP
  }
  return last_err();
}

int gdn_chunk_ho(const void* qn, const float* gc, const float* aq, const float* u, const float* w, const float* kg,
                 void* out, void* state, const int* cu_seqlens, const int* cu_nt, const int* chunk_off,
                 const int* state_indices, int64_t state_stride_tok, int64_t num_seqs, int64_t H, int64_t HV,
                 int64_t K, int64_t V, int64_t bt, int64_t ho_bv, cudaStream_t stream) {
  return ho_fused(qn, gc, aq, u, w, kg, out, state, cu_seqlens, cu_nt, chunk_off, state_indices, state_stride_tok,
                  num_seqs, H, HV, K, V, bt, ho_bv, 0, stream);
}

// r9：prefetch = 寄存器预取（0/1）；split ≠ 0 走 h-pass + o-pass 两段（第 2 轮，h16 / v16 为中间缓冲），值 > 1 即 o-pass 的 V 分片 VT。
int gdn_chunk_ho_v2(const void* qn, const float* gc, const float* aq, const float* u, const float* w, const float* kg,
                    void* h16, void* v16, void* out, void* state, const int* cu_seqlens, const int* cu_nt,
                    const int* chunk_off, const int* chunk_tab, const int* state_indices, int64_t state_stride_tok,
                    int64_t num_seqs, int64_t num_chunks, int64_t H, int64_t HV, int64_t K, int64_t V, int64_t bt,
                    int64_t ho_bv, int64_t prefetch, int64_t split, cudaStream_t stream) {
  if (split != 0) {
    if (!h16 || !v16 || !chunk_tab || num_chunks <= 0) return err(cudaErrorInvalidValue);
    // split > 1 = o-pass 的 V 分片 VT（64 / 32 / 16）；split == 1 兼容旧语义 = 整 V 一片（≤ 64）或 64
    const int64_t vt = split > 1 ? split : (V <= 64 ? V : 64);
    return ho_split(qn, gc, aq, u, w, kg, static_cast<bf16*>(h16), static_cast<bf16*>(v16), out, state, cu_seqlens,
                    cu_nt, chunk_off, chunk_tab, state_indices, state_stride_tok, num_seqs, num_chunks, H, HV, K, V,
                    bt, ho_bv, prefetch, vt, stream);
  }
  return ho_fused(qn, gc, aq, u, w, kg, out, state, cu_seqlens, cu_nt, chunk_off, state_indices, state_stride_tok,
                  num_seqs, H, HV, K, V, bt, ho_bv, prefetch, stream);
}

}  // extern "C"
