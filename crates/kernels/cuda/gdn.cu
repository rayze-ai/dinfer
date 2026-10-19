// GDN（Gated DeltaNet）算子。模型无关：只按形状参数，不含模型名 / 层号 / 调优常量。
//
// 数学口径（与 vLLM `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` +
// `third_party/flash_linear_attention/ops/{fused_recurrent,chunk,l2norm}.py` 一致；
// 与同族的 KDA 有三处差异：dt_bias 是逐 v 头的标量、gate 恒为 -exp(A_log)·softplus、
// 输出门是 silu(z)（K3 是 sigmoid），所以单独一份文件）：
//   1. 短卷积（depthwise causal conv1d, width=W，bias=False，默认 siLU）
//        y[t,d] = silu( Σ_j w[d,j] · x[t-W+1+j,d] )
//      宽度维按时间从旧到新；越界部分取 conv_state 或 0。
//   2. gate / beta（**逐 v 头标量**，不是逐 k 维）
//        A = exp(A_log[hv]);  x = a[t,hv] + dt_bias[hv]
//        g = -A · softplus(x)          softplus 用阈值 20 的稳定写法
//        beta = sigmoid(b[t,hv])
//   3. q、k 在 kernel 内做 l2 归一化：q /= sqrt(Σq² + 1e-6)，再乘 scale = K^-0.5；k 同（不乘 scale）
//   4. 递归核心（逐 token，S 为 fp32 [V, K]）
//        S *= exp(g)[None, :]
//        v' = (v - S·k) · beta
//        S += v'[:, None] · k
//        o  = S·q
//      v 头 hv 用 k 头 h = hv / group（group = HV / H，vLLM 同式）。
//   5. 输出门 + RMSNorm（norm_before_gate）：y = rmsnorm(o, w, eps) · act(z)，act ∈ {silu, sigmoid}
//
// 本文件给的是**顺序递归**实现：prefill 与 decode 同一条递推路径（varlen + 状态槽），
// 慢但对拍精确；块级并行的 chunk 形式（BT=64）留 v2，见 docs/loops/qwen35-gdn.md。

#include "common.cuh"

namespace dinfer {

namespace {

__device__ __forceinline__ float sigmoid_f(float x) { return 1.f / (1.f + __expf(-x)); }

// softplus，阈值 20（fp32 下 exp(20) 还没溢出，但与 Triton 参考保持同一分支以免对拍漂移）
__device__ __forceinline__ float softplus_f(float x) {
  return x > 20.f ? x : __logf(1.f + __expf(x));
}

__device__ __forceinline__ float act_f(float x, int act) {  // 0 = silu, 1 = sigmoid
  return act == 1 ? sigmoid_f(x) : x * sigmoid_f(x);
}

// ---------------------------------------------------------------- conv1d prefill
// 老版（每 (seq, 通道) 一线程串行扫 T）：T=1100 时每步一次 L2 往返，串行链 ~0.9 ms/层；只作回退（CONV_PREFILL=legacy）。
// conv_state 布局 [slots, dim, state_len]，列 0 最旧、最后一列最新。
__global__ void conv1d_prefill_kernel(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                      bf16* __restrict__ out, bf16* __restrict__ conv_state,
                                      const int* __restrict__ cu_seqlens,
                                      const int* __restrict__ state_indices,
                                      const int* __restrict__ has_initial, int dim, int width,
                                      int state_len, int dim_per_block, int act) {
  const int seq = blockIdx.x;
  const int d = blockIdx.y * dim_per_block + threadIdx.x;
  if (d >= dim) return;
  const int slot = state_indices ? state_indices[seq] : seq;
  if (slot < 0) return;

  const int bos = cu_seqlens[seq], eos = cu_seqlens[seq + 1];
  const int len = eos - bos;

  float ww[8];
  for (int j = 0; j < width; ++j) ww[j] = to_f32(w[d * width + j]);

  // 历史窗口：[0] 最旧 … [W-2] 最新
  float hist[8];
  for (int j = 0; j < width - 1; ++j) hist[j] = 0.f;
  if (conv_state && state_len > 0 && (!has_initial || has_initial[seq])) {
    const bf16* st = conv_state + ((long)slot * dim + d) * state_len;
    for (int j = 0; j < state_len; ++j) hist[j] = to_f32(st[j]);
  }

  for (int t = 0; t < len; ++t) {
    const float xt = to_f32(x[((long)(bos + t)) * dim + d]);
    float acc = ww[width - 1] * xt;
    for (int j = 0; j < width - 1; ++j) acc += ww[j] * hist[j];
    out[((long)(bos + t)) * dim + d] = to_bf16(act_f(acc, act));
    if (width > 1) {
      for (int j = 0; j < width - 2; ++j) hist[j] = hist[j + 1];
      hist[width - 2] = xt;
    }
  }

  // 回写最后 W-1 个原始输入
  if (conv_state && state_len > 0) {
    bf16* st = conv_state + ((long)slot * dim + d) * state_len;
    for (int j = 0; j < state_len; ++j) st[j] = to_bf16(hist[j]);
  }
}

// 新版（第 5A 轮）：按 token 平铺分块。每 block 一个 (token 块 rows_per_block 行, 通道块 256·EPT 通道)，
// 线程持 EPT 个连续通道（EPT=8 → 16 B 向量访存；dim 不是 8 的倍数时 EPT=1）。块内行串行、块间并行。
// 块首的 W-1 个历史值直接从 x 读（同序列的前几行），序列首行的历史从 conv_state 读。
// 跨 block 的读写顺序：旧状态只被"包含序列首行 bos 的 block"读，新状态也只由它写（新状态 = 序列最后 W-1 个输入，
// 从只读的 x 取，不足用旧状态补），同一线程先读后写，无竞争。为了让别的 block 永不读 state，
// 包含 bos 的 block 额外接管 [bos, bos+W-1) 这几行（即使越过本块行界），后续 block 从 max(r0, bos+W-1) 起算，
// 其 halo 全部落在 x 内。要求 rows_per_block ≥ W-1（host 保证）。
// 调优数值（rows_per_block）由 host 给（ArchProfile::gdn_conv_rows），不在 kernel 内。
__device__ __forceinline__ void ld_bf16x8(float* f, const bf16* p) {
  const uint4 raw = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 t = __bfloat1622float2(h[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}
__device__ __forceinline__ void st_bf16x8(bf16* p, const float* f) {
  uint4 raw;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
  *reinterpret_cast<uint4*>(p) = raw;
}

template <int EPT, int W>
__global__ void __launch_bounds__(256) conv1d_prefill_blocked_kernel(
    const bf16* __restrict__ x, const bf16* __restrict__ w, bf16* __restrict__ out, bf16* __restrict__ conv_state,
    const int* __restrict__ cu_seqlens, const int* __restrict__ state_indices, const int* __restrict__ has_initial,
    int num_seqs, int dim, int total, int rows_per_block, int act) {
  constexpr int H = W - 1;  // 历史长度 = state_len
  static_assert(H >= 1, "width >= 2");
  const int d0 = (blockIdx.y * 256 + threadIdx.x) * EPT;
  if (d0 >= dim) return;
  const int r0 = blockIdx.x * rows_per_block;
  const int r1 = min(total, r0 + rows_per_block);
  if (r0 >= r1) return;

  float ww[EPT][W];
#pragma unroll
  for (int e = 0; e < EPT; ++e)
#pragma unroll
    for (int j = 0; j < W; ++j) ww[e][j] = to_f32(w[(long)(d0 + e) * W + j]);

  auto load_x = [&](float* f, int row) {
    if (EPT == 8) {
      ld_bf16x8(f, x + (long)row * dim + d0);
    } else {
#pragma unroll
      for (int e = 0; e < EPT; ++e) f[e] = to_f32(x[(long)row * dim + d0 + e]);
    }
  };

  // r0 所在序列（cu[seq+1] > r0 的最小 seq）
  int seq;
  {
    int lo = 0, hi = num_seqs - 1;
    while (lo < hi) {
      const int mid = (lo + hi) >> 1;
      if (cu_seqlens[mid + 1] <= r0) lo = mid + 1; else hi = mid;
    }
    seq = lo;
  }

  float hist[EPT][H];  // [e][0] 最旧 … [e][H-1] 最新
  for (; seq < num_seqs; ++seq) {
    const int bos = cu_seqlens[seq], eos = cu_seqlens[seq + 1];
    if (bos >= r1) break;
    if (eos <= bos) continue;
    const bool owns_head = bos >= r0;
    const int start = owns_head ? bos : max(r0, bos + H);
    const int end = owns_head ? min(eos, max(r1, bos + H)) : min(eos, r1);
    if (start >= end) continue;

    if (owns_head) {
      const int slot = state_indices ? state_indices[seq] : seq;
      const bool init = conv_state && slot >= 0 && (!has_initial || has_initial[seq]);
#pragma unroll
      for (int e = 0; e < EPT; ++e) {
        const bf16* st = conv_state + ((long)slot * dim + d0 + e) * H;
#pragma unroll
        for (int j = 0; j < H; ++j) hist[e][j] = init ? to_f32(st[j]) : 0.f;
      }
      // 新状态 = 最后 H 个输入（位置 eos-H+j；落在序列前的用旧状态列 j+len 补），先读后写同线程
      if (conv_state && slot >= 0) {
        const int len = eos - bos;
        float ns[EPT][H];
#pragma unroll
        for (int j = 0; j < H; ++j) {
          const int p = eos - H + j;
          if (p >= bos) {
            float f[EPT];
            load_x(f, p);
#pragma unroll
            for (int e = 0; e < EPT; ++e) ns[e][j] = f[e];
          } else {
#pragma unroll
            for (int e = 0; e < EPT; ++e) ns[e][j] = hist[e][j + len];
          }
        }
#pragma unroll
        for (int e = 0; e < EPT; ++e) {
          bf16* st = conv_state + ((long)slot * dim + d0 + e) * H;
#pragma unroll
          for (int j = 0; j < H; ++j) st[j] = to_bf16(ns[e][j]);
        }
      }
    } else {
      // start ≥ bos + H：历史全在 x 内
#pragma unroll
      for (int j = 0; j < H; ++j) {
        float f[EPT];
        load_x(f, start - H + j);
#pragma unroll
        for (int e = 0; e < EPT; ++e) hist[e][j] = f[e];
      }
    }

    for (int r = start; r < end; ++r) {
      float xt[EPT], o[EPT];
      load_x(xt, r);
#pragma unroll
      for (int e = 0; e < EPT; ++e) {
        float acc = ww[e][W - 1] * xt[e];
#pragma unroll
        for (int j = 0; j < H; ++j) acc += ww[e][j] * hist[e][j];
        o[e] = act_f(acc, act);
#pragma unroll
        for (int j = 0; j + 1 < H; ++j) hist[e][j] = hist[e][j + 1];
        hist[e][H - 1] = xt[e];
      }
      if (EPT == 8) {
        st_bf16x8(out + (long)r * dim + d0, o);
      } else {
#pragma unroll
        for (int e = 0; e < EPT; ++e) out[(long)r * dim + d0 + e] = to_bf16(o[e]);
      }
    }
  }
}

// ---------------------------------------------------------------- conv1d update (decode, 1 token)
__global__ void conv1d_update_kernel(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                     bf16* __restrict__ out, bf16* __restrict__ conv_state,
                                     const int* __restrict__ state_indices, int batch, int dim,
                                     int width, int state_len, int act) {
  const int b = blockIdx.x;
  const int d = blockIdx.y * blockDim.x + threadIdx.x;
  if (d >= dim) return;
  const int slot = state_indices ? state_indices[b] : b;
  if (slot < 0) return;

  bf16* st = conv_state + ((long)slot * dim + d) * state_len;
  float acc = 0.f;
  for (int j = 0; j < width - 1; ++j) acc += to_f32(w[d * width + j]) * to_f32(st[j]);
  const float xt = to_f32(x[(long)b * dim + d]);
  acc += to_f32(w[d * width + width - 1]) * xt;
  out[(long)b * dim + d] = to_bf16(act_f(acc, act));
  for (int j = 0; j + 1 < state_len; ++j) st[j] = st[j + 1];
  st[state_len - 1] = to_bf16(xt);
}

// ---------------------------------------------------------------- gate / beta（逐 v 头标量）
// g[t,hv] = -exp(A_log[hv]) · softplus(a[t,hv] + dt_bias[hv]);  beta[t,hv] = sigmoid(b[t,hv])
__global__ void gate_kernel(const bf16* __restrict__ a, const bf16* __restrict__ b,
                            const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                            float* __restrict__ g_out, float* __restrict__ beta_out, long total,
                            int heads, int do_beta) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int hv = (int)(i % heads);
  const float A = __expf(a_log[hv]);
  const float x = to_f32(a[i]) + dt_bias[hv];
  g_out[i] = -A * softplus_f(x);
  if (do_beta) beta_out[i] = sigmoid_f(to_f32(b[i]));
}

// ---------------------------------------------------------------- 递归核心（prefill + decode）
// 每 block 一个 (序列, v 头)。S 为 [V, K] fp32 放动态 shared：一次 load、整段循环、一次 store。
// 共享内存 = (V/vsplit·K + 2·K) · 4 字节；V=K=128、vsplit=1 时 66 KB。grid.z = vsplit（host 给，须整除 V）。
// FUSE_GATE=1：g/beta 在 kernel 内由 raw a/b + A_log/dt_bias 现算（decode 的 fused 路径）；
// FUSE_GATE=0：读调用方预算好的 g/beta（prefill 路径，与 vLLM fused_post_conv_prep 对应）。
template <int BLOCK, bool FUSE_GATE>
__global__ void recurrent_kernel(const bf16* __restrict__ mixed, const float* __restrict__ g_in,
                                 const float* __restrict__ beta_in,
                                 const bf16* __restrict__ a_raw, const bf16* __restrict__ b_raw,
                                 const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                                 bf16* __restrict__ out, float* __restrict__ state,
                                 const int* __restrict__ cu_seqlens,
                                 const int* __restrict__ state_indices, int H, int HV, int K, int V,
                                 int group, float scale, long state_stride_tok) {
  extern __shared__ float smem[];
  // V 维切 gridDim.z 份：本 block 只负责状态的 [v0, v0+Vb) 行（各行递推互不依赖；q/k 归一化每块重复算，很便宜）。
  const int Vb = V / (int)gridDim.z;
  const int v0 = (int)blockIdx.z * Vb;
  float* S = smem;             // [Vb, K]
  float* kbuf = S + (long)Vb * K;
  float* qbuf = kbuf + K;
  const int tid = threadIdx.x;

  const int seq = blockIdx.x;
  const int hv = blockIdx.y;
  const int h = hv / group;
  const int slot = state_indices ? state_indices[seq] : seq;
  const int bos = cu_seqlens[seq], eos = cu_seqlens[seq + 1];
  const int len = eos - bos;

  const long qkv_row = 2L * H * K + (long)HV * V;  // mixed_qkv 的每 token 宽度
  float* Sg = state + (long)slot * state_stride_tok + (long)hv * V * K + (long)v0 * K;
  if (slot < 0 || len <= 0) return;
  for (int i = tid; i < Vb * K; i += BLOCK) S[i] = Sg[i];
  __syncthreads();

  // 该 v 头对应的 k 头偏移
  const bf16* qp = mixed + (long)h * K;
  const bf16* kp = mixed + (long)H * K + (long)h * K;
  const bf16* vp = mixed + 2L * H * K + (long)hv * V;

  for (int t = 0; t < len; ++t) {
    const long row = bos + t;
    const float gt = FUSE_GATE ? 0.f : g_in[row * HV + hv];
    const float bt = FUSE_GATE ? 0.f : beta_in[row * HV + hv];
    if (tid < K) {
      const float qv = to_f32(qp[row * qkv_row + tid]);
      const float kv = to_f32(kp[row * qkv_row + tid]);
      kbuf[tid] = kv;
      qbuf[tid] = qv;
    }
    // l2 归一化（q 再乘 scale）
    const float qsq = (tid < K) ? qbuf[tid] * qbuf[tid] : 0.f;
    const float ksq = (tid < K) ? kbuf[tid] * kbuf[tid] : 0.f;
    const float qs = block_reduce_sum<BLOCK>(qsq);
    const float ks = block_reduce_sum<BLOCK>(ksq);
    if (tid < K) {
      qbuf[tid] = qbuf[tid] / sqrtf(qs + 1e-6f) * scale;
      kbuf[tid] = kbuf[tid] / sqrtf(ks + 1e-6f);
    }
    __syncthreads();

    float g, beta;
    if (FUSE_GATE) {
      const float A = __expf(a_log[hv]);
      g = -A * softplus_f(to_f32(a_raw[row * HV + hv]) + dt_bias[hv]);
      beta = sigmoid_f(to_f32(b_raw[row * HV + hv]));
    } else {
      g = gt;
      beta = bt;
    }
    const float eg = __expf(g);

    for (int vv = tid; vv < Vb; vv += BLOCK) {
      float* r = S + (long)vv * K;
      float p = 0.f;
      for (int kk = 0; kk < K; ++kk) {
        r[kk] *= eg;
        p += r[kk] * kbuf[kk];
      }
      const float vin = to_f32(vp[row * qkv_row + v0 + vv]);
      const float vprime = (vin - p) * beta;
      float o = 0.f;
      for (int kk = 0; kk < K; ++kk) {
        r[kk] += vprime * kbuf[kk];
        o += r[kk] * qbuf[kk];
      }
      out[(row * HV + hv) * V + v0 + vv] = to_bf16(o);
    }
    __syncthreads();  // 下一 token 会覆盖 kbuf/qbuf
  }

  for (int i = tid; i < Vb * K; i += BLOCK) Sg[i] = S[i];
}

// ---------------------------------------------------------------- 递归核心：decode 专用（寄存器版）
// 老 recurrent_kernel 把整个 [V, K] 状态装进 shared，再按 v 行逐线程串行扫 K 维两趟：decode（len=1）时
// 整个 kernel 只是"读一遍状态 + 两个 K 维点积 + 写一遍状态"，却被 block 归约 / 同步和串行 K 循环拖成
// 延迟绑定，B=32 时不随 batch 扩展（docs/loops/qwen35-perf.md 第 3g 行）。本版：
//   - 每 block 一个 (序列, v 头, V 块 BV)；kDecodeWarps 个 warp，每 warp 负责 ROWS = BV/kDecodeWarps 行状态；
//   - 每行 [K] f32 按 lane 切成 KPL = K/32 个连续元素（KPL=4 → 16 B 向量读写），全程在寄存器，不经 shared；
//   - K 维点积 = 每 lane KPL 次乘加 + warp shuffle 归约；q/k 的 l2 归一化每 warp 重复算（K 维一次 warp 归约，
//     远比 block 级 __syncthreads 便宜），没有任何 block 级同步；
//   - 支持任意 len（逐 token 循环，状态留在寄存器），所以 host 把 prefill 尾段误送进来也不影响正确性。
// BV 由 host 给（`ArchProfile::gdn_decode_bv`），kDecodeWarps 是结构常量（不是按卡调的数值）。
//
// 第 7 轮（2026-10-20）两个可选变体，模板开关、host 由 `decode_variant` 选（`ArchProfile::gdn_decode_variant`，
// `GDN_DECODE_VARIANT=base|tr|pipe|pipe_base` 覆盖）：
//   TR（转置归约）：每 token 只做两次"多值 warp 归约"（蝶式折半：N 个部分和 → log2 N 轮共 ~N 次 shuffle，
//     而不是 N 次 × 5 shuffle）。第一次归约 {q·q, k·k, k·q}；第二次把 ROWS 行的 {S·k, S·q} 一次归约（2·ROWS ≤ 32 个值），
//     行 i 的结果落在 lane i·S（p）和 lane 16+i·S（o₀）。数学上用 S_new = eg·S_old + v'·k 展开：
//     p = eg·(S_old·k)，o = eg·(S_old·q) + v'·(k·q)，S 只扫一遍（一次 FMA 遍）。shuffle 数 ROWS=16 时 170 → ~57。
//   PIPE（持久 block + cp.async 流水）：grid 只铺一波（occupancy × SM 数），每 block 顺序吃多个 (seq, v 头, V 块)
//     tile；算当前 tile 时下一 tile 的状态已在用 cp.async 进 smem（单缓冲：每 lane 先把自己那 16 B 取进寄存器，
//     再对同一位置发下一 tile 的 cp.async，不需要 block 同步）。目的：B300 这类 HBM 延迟/带宽比高的卡上，
//     "读 → 算 → 写"锁步的 block 波之间 HBM 空转，流水把下一块的读藏在当前块的算 + 写后面。
constexpr int kDecodeWarps = 4;

template <int KPL> struct DecodeVec;
template <> struct DecodeVec<4> { using type = float4; };
template <> struct DecodeVec<2> { using type = float2; };

__device__ __forceinline__ float warp_sum_f(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// 多值 warp 归约（蝶式折半）：输入每 lane 持 N 个部分和 v[0..N)（N 为 2 的幂，≤ 32），
// 返回后 lane L 持有第 (L / (32/N)) 个值的全和（每个值由 32/N 个连续 lane 冗余持有）。
// 共 N-1 + log2(32/N) 次 shuffle（N=32：31；N=16：19；N=4：6），远少于 N 次独立 warp_sum 的 5N。
// 全部用模板递归展开：数组下标都是编译期常量，v[] 留在寄存器（循环变量版会把 v[] 压进 local memory）。
template <int N, int HALF>
struct WarpSumMulti {
  static __device__ __forceinline__ float run(float (&v)[N], int lane) {
    constexpr int HN = N / 2;
    const bool upper = (lane & HALF) != 0;
    float w[HN];
#pragma unroll
    for (int i = 0; i < HN; ++i) {
      const float keep = upper ? v[i + HN] : v[i];
      const float send = upper ? v[i] : v[i + HN];
      w[i] = keep + __shfl_xor_sync(0xffffffffu, send, HALF);
    }
    return WarpSumMulti<HN, HALF / 2>::run(w, lane);
  }
};
template <int HALF>
struct WarpSumMulti<1, HALF> {
  static __device__ __forceinline__ float run(float (&v)[1], int) {
    float r = v[0];
#pragma unroll
    for (int o = HALF; o > 0; o >>= 1) r += __shfl_xor_sync(0xffffffffu, r, o);
    return r;
  }
};
template <int N>
__device__ __forceinline__ float warp_sum_multi(float (&v)[N], int lane) {
  static_assert(N >= 1 && N <= 32 && (N & (N - 1)) == 0, "N 须为 2 的幂且 ≤ 32");
  return WarpSumMulti<N, 16>::run(v, lane);
}

template <int KPL>
__device__ __forceinline__ void vec_load(float (&r)[KPL], const float* p) {
  using Vec = typename DecodeVec<KPL>::type;
  const Vec v = *reinterpret_cast<const Vec*>(p);
  if constexpr (KPL == 4) { r[0] = v.x; r[1] = v.y; r[2] = v.z; r[3] = v.w; }
  else { r[0] = v.x; r[1] = v.y; }
}

template <int KPL>
__device__ __forceinline__ void vec_store(float* p, const float (&r)[KPL]) {
  using Vec = typename DecodeVec<KPL>::type;
  Vec v;
  if constexpr (KPL == 4) { v.x = r[0]; v.y = r[1]; v.z = r[2]; v.w = r[3]; }
  else { v.x = r[0]; v.y = r[1]; }
  *reinterpret_cast<Vec*>(p) = v;
}

__device__ __forceinline__ void cp_async16_smem(void* smem, const void* gmem) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit_g() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all_g() { asm volatile("cp.async.wait_group 0;\n" ::); }

struct DecodeArgs {
  const bf16* mixed;
  const float* g_in;
  const float* beta_in;
  const bf16* a_raw;
  const bf16* b_raw;
  const float* a_log;
  const float* dt_bias;
  bf16* out;
  float* state;
  const int* cu_seqlens;
  const int* state_indices;
  int H, HV, K, V, group;
  float scale;
  long state_stride_tok;
};

// 一个 warp 对一个 (seq, v 头, ROWS 行) 的全部 token 递推；状态 r 进来是旧值、出去是新值（调用方负责装 / 存）。
template <int ROWS, int KPL, bool FUSE_GATE, bool TR>
__device__ __forceinline__ void decode_tile_body(const DecodeArgs& a, float (&r)[ROWS][KPL], int seq, int hv,
                                                 int v0, int bos, int len, int lane) {
  // TR：2·ROWS 个值一次归约后，每个值由 S = 32/(2·ROWS) 个连续 lane 持有；lane < 16 持 p(行 my)，lane ≥ 16 持 o₀(行 my)
  constexpr int S = 16 / ROWS;
  static_assert(!TR || (2 * ROWS <= 32 && 16 % ROWS == 0), "TR 要求 ROWS ∈ {4, 8, 16}");
  const int h = hv / a.group;
  const long qkv_row = 2L * a.H * a.K + (long)a.HV * a.V;
  const bf16* qp = a.mixed + (long)h * a.K + lane * KPL;
  const bf16* kp = a.mixed + (long)a.H * a.K + (long)h * a.K + lane * KPL;
  const bf16* vp = a.mixed + 2L * a.H * a.K + (long)hv * a.V + v0;
  const int my = TR ? (lane & 15) / S : 0;  // TR：本 lane 对应的行号

  for (int t = 0; t < len; ++t) {
    const long row = bos + t;
    float q[KPL], k[KPL];
    float qsq = 0.f, ksq = 0.f, kq = 0.f;
#pragma unroll
    for (int j = 0; j < KPL; ++j) {
      q[j] = to_f32(qp[row * qkv_row + j]);
      k[j] = to_f32(kp[row * qkv_row + j]);
      qsq += q[j] * q[j];
      ksq += k[j] * k[j];
      kq += k[j] * q[j];
    }
    if (TR) {
      float s3[4] = {qsq, ksq, kq, 0.f};
      const float mine = warp_sum_multi<4>(s3, lane);  // lane 0-7 qsq，8-15 ksq，16-23 kq
      qsq = __shfl_sync(0xffffffffu, mine, 0);
      ksq = __shfl_sync(0xffffffffu, mine, 8);
      kq = __shfl_sync(0xffffffffu, mine, 16);
    } else {
      qsq = warp_sum_f(qsq);
      ksq = warp_sum_f(ksq);
    }
    const float qn = 1.f / sqrtf(qsq + 1e-6f) * a.scale;
    const float kn = 1.f / sqrtf(ksq + 1e-6f);
#pragma unroll
    for (int j = 0; j < KPL; ++j) {
      q[j] *= qn;
      k[j] *= kn;
    }

    float g, beta;
    if (FUSE_GATE) {
      const float A = __expf(a.a_log[hv]);
      g = -A * softplus_f(to_f32(a.a_raw[row * a.HV + hv]) + a.dt_bias[hv]);
      beta = sigmoid_f(to_f32(a.b_raw[row * a.HV + hv]));
    } else {
      g = a.g_in[row * a.HV + hv];
      beta = a.beta_in[row * a.HV + hv];
    }
    const float eg = __expf(g);

    if constexpr (TR) {
      kq *= qn * kn;
      // 一遍扫状态：行 i 的 S_old·k → pr[i]，S_old·q → pr[ROWS+i]
      float pr[2 * ROWS];
#pragma unroll
      for (int i = 0; i < ROWS; ++i) {
        float p = 0.f, o = 0.f;
#pragma unroll
        for (int j = 0; j < KPL; ++j) {
          p += r[i][j] * k[j];
          o += r[i][j] * q[j];
        }
        pr[i] = p;
        pr[ROWS + i] = o;
      }
      const float mine = warp_sum_multi<2 * ROWS>(pr, lane);  // lane < 16：p(my)；lane ≥ 16：o₀(my)
      // 低半 lane 算 v'，高半 lane 从 xor 16 取 v' 后算 o
      const float vin = to_f32(vp[row * qkv_row + my]);  // 行 my 的 v（lane ≥ 16 也读，便宜）
      const float vprime_lo = (vin - eg * mine) * beta;
      const float vprime = __shfl_xor_sync(0xffffffffu, vprime_lo, 16);  // 高半拿到低半的 v'
      if (lane >= 16 && (lane & (S - 1)) == 0) {
        a.out[(row * a.HV + hv) * a.V + v0 + my] = to_bf16(eg * mine + vprime * kq);
      }
      // 状态更新：S = eg·S + v'_i·k，v'_i 从 lane i·S 广播
#pragma unroll
      for (int i = 0; i < ROWS; ++i) {
        const float vpi = __shfl_sync(0xffffffffu, vprime_lo, i * S);
#pragma unroll
        for (int j = 0; j < KPL; ++j) r[i][j] = fmaf(eg, r[i][j], vpi * k[j]);
      }
    } else {
      float o_mine = 0.f;  // lane i 保存第 i 行的输出，最后 ROWS 个 lane 连续写出
#pragma unroll
      for (int i = 0; i < ROWS; ++i) {
        float p = 0.f;
#pragma unroll
        for (int j = 0; j < KPL; ++j) {
          r[i][j] *= eg;
          p += r[i][j] * k[j];
        }
        p = warp_sum_f(p);
        const float vin = to_f32(vp[row * qkv_row + i]);  // 全 warp 同址，广播
        const float vprime = (vin - p) * beta;
        float o = 0.f;
#pragma unroll
        for (int j = 0; j < KPL; ++j) {
          r[i][j] += vprime * k[j];
          o += r[i][j] * q[j];
        }
        o = warp_sum_f(o);
        if (lane == i) o_mine = o;
      }
      if (lane < ROWS) a.out[(row * a.HV + hv) * a.V + v0 + lane] = to_bf16(o_mine);
    }
  }
}

// 非流水：grid = (B, HV, V/BV)，每 block 一个 tile，状态直接 global → 寄存器 → global。
template <int BV, int KPL, bool FUSE_GATE, bool TR>
__global__ void __launch_bounds__(kDecodeWarps * 32) recurrent_decode_kernel(DecodeArgs a) {
  constexpr int ROWS = BV / kDecodeWarps;
  static_assert(BV % kDecodeWarps == 0, "BV 必须是 warp 数的整数倍");
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int seq = blockIdx.x;
  const int hv = blockIdx.y;
  const int v0 = (int)blockIdx.z * BV + warp * ROWS;  // 本 warp 的首行
  const int slot = a.state_indices ? a.state_indices[seq] : seq;
  const int bos = a.cu_seqlens[seq], eos = a.cu_seqlens[seq + 1];
  const int len = eos - bos;
  if (slot < 0 || len <= 0) return;

  float* Sg = a.state + (long)slot * a.state_stride_tok + (long)hv * a.V * a.K + (long)v0 * a.K + lane * KPL;
  // 状态进寄存器：ROWS 个 16 B 读一次性发出（访存并行度）
  float r[ROWS][KPL];
#pragma unroll
  for (int i = 0; i < ROWS; ++i) vec_load<KPL>(r[i], Sg + (long)i * a.K);
  decode_tile_body<ROWS, KPL, FUSE_GATE, TR>(a, r, seq, hv, v0, bos, len, lane);
#pragma unroll
  for (int i = 0; i < ROWS; ++i) vec_store<KPL>(Sg + (long)i * a.K, r[i]);
}

// 流水：持久 block，tile = blockIdx.x + k·gridDim.x（tile → (seq, hv, vblk)），下一 tile 的状态 cp.async 进 smem。
// 每 lane 只搬 / 只读自己那 ROWS × KPL 个 float（smem 位置与 lane 一一对应），所以单缓冲、不需要 block 同步：
// 先把当前 tile 从 smem 取进寄存器（wait_group 0 + __syncwarp），再对同一位置发下一 tile 的 cp.async。
template <int BV, int KPL, bool FUSE_GATE, bool TR>
__global__ void __launch_bounds__(kDecodeWarps * 32) recurrent_decode_pipe_kernel(DecodeArgs a, int num_tiles,
                                                                                   int vblocks) {
  constexpr int ROWS = BV / kDecodeWarps;
  static_assert(BV % kDecodeWarps == 0, "BV 必须是 warp 数的整数倍");
  __shared__ __align__(16) float sbuf[kDecodeWarps][ROWS][32 * KPL];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  float(&sw)[ROWS][32 * KPL] = sbuf[warp];

  struct Tile {
    int seq, hv, v0, bos, len;
    float* Sg;
    bool valid;
  };
  auto decode_tile = [&](int tile) {
    Tile t;
    const int seq = tile / (a.HV * vblocks);
    const int rem = tile - seq * (a.HV * vblocks);
    t.seq = seq;
    t.hv = rem / vblocks;
    t.v0 = (rem - t.hv * vblocks) * BV + warp * ROWS;
    const int slot = a.state_indices ? a.state_indices[seq] : seq;
    t.bos = a.cu_seqlens[seq];
    t.len = a.cu_seqlens[seq + 1] - t.bos;
    t.valid = slot >= 0 && t.len > 0;
    t.Sg = a.state + (long)slot * a.state_stride_tok + (long)t.hv * a.V * a.K + (long)t.v0 * a.K + lane * KPL;
    return t;
  };
  auto issue = [&](const Tile& t) {
    if (!t.valid) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
      if constexpr (KPL == 4) {
        cp_async16_smem(&sw[i][lane * KPL], t.Sg + (long)i * a.K);
      } else {
        // 8 B：cp.async.cg 只支持 16 B，直接 ld/st（K=64 时才走这里）
        *reinterpret_cast<float2*>(&sw[i][lane * KPL]) = *reinterpret_cast<const float2*>(t.Sg + (long)i * a.K);
      }
    }
    if constexpr (KPL == 4) cp_async_commit_g();
  };

  int tile = blockIdx.x;
  if (tile >= num_tiles) return;
  Tile cur = decode_tile(tile);
  issue(cur);
  for (; tile < num_tiles; tile += gridDim.x) {
    const int next_tile = tile + gridDim.x;
    const bool has_next = next_tile < num_tiles;
    Tile nxt;
    if (has_next) nxt = decode_tile(next_tile);
    float r[ROWS][KPL];
    if (cur.valid) {
      if constexpr (KPL == 4) cp_async_wait_all_g();
      __syncwarp();
#pragma unroll
      for (int i = 0; i < ROWS; ++i) vec_load<KPL>(r[i], &sw[i][lane * KPL]);
      __syncwarp();
    }
    if (has_next) issue(nxt);  // 下一 tile 的读与当前 tile 的算 / 写重叠
    if (cur.valid) {
      decode_tile_body<ROWS, KPL, FUSE_GATE, TR>(a, r, cur.seq, cur.hv, cur.v0, cur.bos, cur.len, lane);
#pragma unroll
      for (int i = 0; i < ROWS; ++i) vec_store<KPL>(cur.Sg + (long)i * a.K, r[i]);
    }
    if (has_next) cur = nxt;
  }
}

// 持久 kernel 的 grid：occupancy × SM 数（按实例查一次，缓存）；decode_grid > 0 时 host 覆盖。
template <typename Kern>
int persistent_grid(Kern kern, int num_tiles, int decode_grid) {
  if (decode_grid > 0) return decode_grid < num_tiles ? decode_grid : num_tiles;
  static int cached = 0;  // 每个模板实例一份
  if (cached == 0) {
    int dev = 0, sms = 1, per_sm = 1;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, kDecodeWarps * 32, 0);
    cached = (sms > 0 ? sms : 1) * (per_sm > 0 ? per_sm : 1);
  }
  return cached < num_tiles ? cached : num_tiles;
}

template <int BV, int KPL, bool FUSE_GATE, bool TR>
int launch_recurrent_decode_tr(bool pipe, int num_seqs, const DecodeArgs& a, int decode_grid, cudaStream_t stream) {
  const int vblocks = a.V / BV;
  if (pipe) {
    const int num_tiles = num_seqs * a.HV * vblocks;
    auto kern = recurrent_decode_pipe_kernel<BV, KPL, FUSE_GATE, TR>;
    const int grid = persistent_grid(kern, num_tiles, decode_grid);
    kern<<<grid, kDecodeWarps * 32, 0, stream>>>(a, num_tiles, vblocks);
  } else {
    dim3 grid((unsigned)num_seqs, (unsigned)a.HV, (unsigned)vblocks);
    recurrent_decode_kernel<BV, KPL, FUSE_GATE, TR><<<grid, kDecodeWarps * 32, 0, stream>>>(a);
  }
  return last_err();
}

// decode_variant：0 base（第 4 轮 kernel），1 tr，2 pipe（流水 + tr），3 pipe_base（流水 + base 归约）
template <int BV, int KPL>
int launch_recurrent_decode(bool fused, int variant, int num_seqs, const DecodeArgs& a, int decode_grid,
                            cudaStream_t stream) {
  const bool pipe = variant == 2 || variant == 3;
  const bool tr = variant == 1 || variant == 2;
  if (fused) {
    return tr ? launch_recurrent_decode_tr<BV, KPL, true, true>(pipe, num_seqs, a, decode_grid, stream)
              : launch_recurrent_decode_tr<BV, KPL, true, false>(pipe, num_seqs, a, decode_grid, stream);
  }
  return tr ? launch_recurrent_decode_tr<BV, KPL, false, true>(pipe, num_seqs, a, decode_grid, stream)
            : launch_recurrent_decode_tr<BV, KPL, false, false>(pipe, num_seqs, a, decode_grid, stream);
}

template <int KPL>
int launch_recurrent_decode_bv(int bv, bool fused, int variant, int num_seqs, const DecodeArgs& a, int decode_grid,
                               cudaStream_t stream) {
  if (variant < 0 || variant > 3) return (int)cudaErrorInvalidValue;
  switch (bv) {
    case 16: return launch_recurrent_decode<16, KPL>(fused, variant, num_seqs, a, decode_grid, stream);
    case 32: return launch_recurrent_decode<32, KPL>(fused, variant, num_seqs, a, decode_grid, stream);
    case 64: return launch_recurrent_decode<64, KPL>(fused, variant, num_seqs, a, decode_grid, stream);
    default: return (int)cudaErrorInvalidValue;
  }
}

// ---------------------------------------------------------------- 输出门 + RMSNorm
// y[t,hv,d] = rmsnorm(o[t,hv,:], w, eps) · act(z[t,hv,d])，一个 (token, v 头) 一行。
__global__ void gated_rmsnorm_kernel(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                     const bf16* __restrict__ z, bf16* __restrict__ out,
                                     int head_dim, float eps, int act) {
  const long row = blockIdx.x;
  const bf16* xr = x + row * head_dim;
  float ss = 0.f;
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    const float v = to_f32(xr[d]);
    ss += v * v;
  }
  const float s = block_reduce_sum<256>(ss);
  const float rstd = rsqrtf(s / head_dim + eps);
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    const float v = to_f32(xr[d]) * rstd * to_f32(w[d]) * act_f(to_f32(z[row * head_dim + d]), act);
    out[row * head_dim + d] = to_bf16(v);
  }
}

// 第 7 轮：一 warp 一 (token, v 头) 行、kGatedWarps 个 warp 一 block。老版一 block 256 线程算 128 列（一半线程空转）
// + 两次 __syncthreads + x 读两遍；本版每 lane 持 DPL = head_dim/32 个元素（x / z / w 各一次合并读，一起发出），
// 只有 warp shuffle。head_dim ∈ {32, 64, 128, 256}（DPL ∈ {1,2,4,8}），其它回老 kernel。
constexpr int kGatedWarps = 8;

template <int DPL>
__global__ void __launch_bounds__(kGatedWarps * 32) gated_rmsnorm_warp_kernel(const bf16* __restrict__ x,
                                                                              const bf16* __restrict__ w,
                                                                              const bf16* __restrict__ z,
                                                                              bf16* __restrict__ out, long rows,
                                                                              float eps, int act) {
  constexpr int D = DPL * 32;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const long row = (long)blockIdx.x * kGatedWarps + warp;
  if (row >= rows) return;
  const bf16* xr = x + row * D;
  const bf16* zr = z + row * D;
  float xv[DPL], zv[DPL], wv[DPL];
  float ss = 0.f;
#pragma unroll
  for (int c = 0; c < DPL; ++c) {
    const int d = lane + 32 * c;
    xv[c] = to_f32(xr[d]);
    zv[c] = to_f32(zr[d]);
    wv[c] = to_f32(w[d]);
    ss += xv[c] * xv[c];
  }
  ss = warp_sum_f(ss);
  const float rstd = rsqrtf(ss / D + eps);
#pragma unroll
  for (int c = 0; c < DPL; ++c) {
    const int d = lane + 32 * c;
    out[row * D + d] = to_bf16(xv[c] * rstd * wv[c] * act_f(zv[c], act));
  }
}

int set_smem(const void* fn, int bytes) {
  return err(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

constexpr int kBlock = 256;

}  // namespace

extern "C" {

int gdn_conv1d_prefill(const void* x, const void* w, void* out, void* conv_state,
                       const int* cu_seqlens, const int* state_indices, const int* has_initial,
                       int64_t num_seqs, int64_t dim, int64_t width, int64_t state_len, int act,
                       int64_t total, int64_t rows_per_block, cudaStream_t stream) {
  if (num_seqs <= 0) return 0;
  const int per_block = 256;
  // rows_per_block == 0 → 老版（每 (seq, 通道) 串行扫 T）
  if (rows_per_block < width - 1 || rows_per_block <= 0 || total <= 0 || width < 2 || width > 4 ||
      state_len != width - 1) {
    dim3 grid((unsigned)num_seqs, (unsigned)((dim + per_block - 1) / per_block));
    conv1d_prefill_kernel<<<grid, per_block, 0, stream>>>(
        (const bf16*)x, (const bf16*)w, (bf16*)out, (bf16*)conv_state, cu_seqlens, state_indices,
        has_initial, (int)dim, (int)width, (int)state_len, per_block, act);
    return last_err();
  }
  const int ept = (dim % 8 == 0) ? 8 : 1;
  const long dim_blocks = (dim + (long)per_block * ept - 1) / ((long)per_block * ept);
  dim3 grid((unsigned)((total + rows_per_block - 1) / rows_per_block), (unsigned)dim_blocks);
#define LAUNCH_CONV(E, W)                                                                                    \
  conv1d_prefill_blocked_kernel<E, W><<<grid, per_block, 0, stream>>>(                                       \
      (const bf16*)x, (const bf16*)w, (bf16*)out, (bf16*)conv_state, cu_seqlens, state_indices, has_initial, \
      (int)num_seqs, (int)dim, (int)total, (int)rows_per_block, act);
  if (ept == 8) {
    if (width == 4) { LAUNCH_CONV(8, 4) } else if (width == 3) { LAUNCH_CONV(8, 3) } else { LAUNCH_CONV(8, 2) }
  } else {
    if (width == 4) { LAUNCH_CONV(1, 4) } else if (width == 3) { LAUNCH_CONV(1, 3) } else { LAUNCH_CONV(1, 2) }
  }
#undef LAUNCH_CONV
  return last_err();
}

int gdn_conv1d_update(const void* x, const void* w, void* out, void* conv_state,
                      const int* state_indices, int64_t batch, int64_t dim, int64_t width,
                      int64_t state_len, int act, cudaStream_t stream) {
  if (batch <= 0) return 0;
  const int per_block = 256;
  dim3 grid((unsigned)batch, (unsigned)((dim + per_block - 1) / per_block));
  conv1d_update_kernel<<<grid, per_block, 0, stream>>>((const bf16*)x, (const bf16*)w, (bf16*)out,
                                            (bf16*)conv_state, state_indices, (int)batch, (int)dim,
                                            (int)width, (int)state_len, act);
  return last_err();
}

int gdn_gate(const void* a, const void* b, const float* a_log, const float* dt_bias, float* g_out,
             float* beta_out, int64_t total, int64_t heads, int do_beta, cudaStream_t stream) {
  if (total <= 0) return 0;
  const long blocks = (long)((total + kBlock - 1) / kBlock);
  gate_kernel<<<(unsigned)blocks, kBlock, 0, stream>>>((const bf16*)a, (const bf16*)b, a_log, dt_bias, g_out,
                                            beta_out, (long)total, (int)heads, do_beta);
  return last_err();
}

// fused_gate = 0：读 g/beta（prefill）；= 1：kernel 内由 raw a/b 现算（decode）。
// decode_bv > 0：走寄存器版 recurrent_decode_kernel（每 block 一个 (seq, v 头, BV 行)；BV ∈ {16, 32, 64} 且整除 V，
//   K ∈ {64, 128}）；= 0：走老 shared 版（vsplit 生效）。
// decode_variant：0 base / 1 tr / 2 pipe / 3 pipe_base（见 kernel 注释）；decode_grid：pipe 的持久 block 数（0 = occupancy × SM）。
int gdn_recurrent(const void* mixed_qkv, const float* g, const float* beta, const void* a_raw,
                  const void* b_raw, const float* a_log, const float* dt_bias, void* out,
                  void* state, const int* cu_seqlens, const int* state_indices, int64_t num_seqs,
                  int64_t H, int64_t HV, int64_t K, int64_t V, int64_t group, float scale,
                  int64_t state_stride_tok, int fused_gate, int64_t vsplit, int64_t decode_bv,
                  int decode_variant, int64_t decode_grid, cudaStream_t stream) {
  if (num_seqs <= 0) return 0;
  if (decode_bv > 0) {
    if (V % decode_bv != 0 || (K != 64 && K != 128)) return (int)cudaErrorInvalidValue;
    DecodeArgs a;
    a.mixed = (const bf16*)mixed_qkv;
    a.g_in = g;
    a.beta_in = beta;
    a.a_raw = (const bf16*)a_raw;
    a.b_raw = (const bf16*)b_raw;
    a.a_log = a_log;
    a.dt_bias = dt_bias;
    a.out = (bf16*)out;
    a.state = (float*)state;
    a.cu_seqlens = cu_seqlens;
    a.state_indices = state_indices;
    a.H = (int)H;
    a.HV = (int)HV;
    a.K = (int)K;
    a.V = (int)V;
    a.group = (int)group;
    a.scale = scale;
    a.state_stride_tok = (long)state_stride_tok;
    const bool fused = fused_gate != 0;
    if (K == 128) {
      return launch_recurrent_decode_bv<4>((int)decode_bv, fused, decode_variant, (int)num_seqs, a, (int)decode_grid,
                                           stream);
    }
    return launch_recurrent_decode_bv<2>((int)decode_bv, fused, decode_variant, (int)num_seqs, a, (int)decode_grid,
                                         stream);
  }
  // vsplit：V 维切几个 block（host 给，须整除 V；1 = 不切）。decode 时 (seq, v 头) 只有几十个 block，切开才铺满 SM。
  if (vsplit <= 0 || V % vsplit != 0) return (int)cudaErrorInvalidValue;
  const int smem = (int)((V / vsplit * K + 2 * K) * 4);
  dim3 grid((unsigned)num_seqs, (unsigned)HV, (unsigned)vsplit);
  // cudaFuncSetAttribute 不能在 stream capture 期间调（CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED）：
  // 每个模板实例只在首次（capture 前的 eager 预热）设一次；smem 只随 (V, K) 变，同一模型内恒定。
  static int smem_set_fused = 0, smem_set_plain = 0;
  if (fused_gate) {
    if (smem_set_fused < smem) {
      if (const int rc = set_smem((const void*)recurrent_kernel<kBlock, true>, smem)) return rc;
      smem_set_fused = smem;
    }
    recurrent_kernel<kBlock, true><<<grid, kBlock, smem, stream>>>(
        (const bf16*)mixed_qkv, g, beta, (const bf16*)a_raw, (const bf16*)b_raw, a_log, dt_bias,
        (bf16*)out, (float*)state, cu_seqlens, state_indices, (int)H, (int)HV, (int)K, (int)V,
        (int)group, scale, (long)state_stride_tok);
  } else {
    if (smem_set_plain < smem) {
      if (const int rc = set_smem((const void*)recurrent_kernel<kBlock, false>, smem)) return rc;
      smem_set_plain = smem;
    }
    recurrent_kernel<kBlock, false><<<grid, kBlock, smem, stream>>>(
        (const bf16*)mixed_qkv, g, beta, (const bf16*)a_raw, (const bf16*)b_raw, a_log, dt_bias,
        (bf16*)out, (float*)state, cu_seqlens, state_indices, (int)H, (int)HV, (int)K, (int)V,
        (int)group, scale, (long)state_stride_tok);
  }
  return last_err();
}

int gdn_gated_rmsnorm(const void* x, const void* w, const void* z, void* out, int64_t rows,
                      int64_t head_dim, float eps, int act, cudaStream_t stream) {
  if (rows <= 0) return 0;
  const unsigned wgrid = (unsigned)((rows + kGatedWarps - 1) / kGatedWarps);
#define GATED_WARP_CASE(DPL)                                                                                   \
  case DPL * 32:                                                                                               \
    gated_rmsnorm_warp_kernel<DPL><<<wgrid, kGatedWarps * 32, 0, stream>>>((const bf16*)x, (const bf16*)w,     \
                                                                           (const bf16*)z, (bf16*)out, (long)rows, \
                                                                           eps, act);                          \
    return last_err();
  switch (head_dim) {
    GATED_WARP_CASE(1)
    GATED_WARP_CASE(2)
    GATED_WARP_CASE(4)
    GATED_WARP_CASE(8)
    default: break;
  }
#undef GATED_WARP_CASE
  gated_rmsnorm_kernel<<<(unsigned)rows, 256, 0, stream>>>((const bf16*)x, (const bf16*)w, (const bf16*)z,
                                                (bf16*)out, (int)head_dim, eps, act);
  return last_err();
}

}  // namespace
}  // extern "C"
