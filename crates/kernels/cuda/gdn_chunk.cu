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
//   chunk_ho     跨 chunk 串行扫 h + 立即算 o
// 数值口径：全部 fp32 累加（先求对；bf16 mma 以后再说）。
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

// 动态 shared 上限按 (kernel, 字节数) 记忆；换了特化必须重设，否则前一次的属性不生效。
int smem_cache[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
int set_smem(const void* fn, int bytes) {
  const intptr_t key = (intptr_t)fn;
  int slot = 0;
  for (; slot < 4; ++slot) {
    if (smem_cache[slot][0] == (int)key) {
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
                 int64_t num_chunks, int64_t H, int64_t HV, int64_t K, int64_t V, int64_t bt, cudaStream_t stream) {
  if (num_chunks <= 0 || H <= 0 || HV <= 0 || HV % H != 0 || K <= 0 || V <= 0) return err(cudaErrorInvalidValue);
  const int group = (int)(HV / H);
  dim3 grid((unsigned)num_chunks, (unsigned)HV);
  if (bt != 64) return err(cudaErrorInvalidValue);
  constexpr int BT = 64;
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

int gdn_chunk_ho(const void* qn, const float* gc, const float* aq, const float* u, const float* w, const float* kg,
                 void* out, void* state, const int* cu_seqlens, const int* cu_nt, const int* chunk_off,
                 const int* state_indices, int64_t state_stride_tok, int64_t num_seqs, int64_t H, int64_t HV,
                 int64_t K, int64_t V, int64_t bt, cudaStream_t stream) {
  if (num_seqs <= 0 || H <= 0 || HV <= 0 || HV % H != 0 || K <= 0 || V <= 0) return err(cudaErrorInvalidValue);
  if (bt != 64) return err(cudaErrorInvalidValue);
  constexpr int BT = 64, BV = 16;
  if (V % BV != 0) return err(cudaErrorInvalidValue);
  const int group = (int)(HV / H);
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

}  // extern "C"
