// KDA（Kimi Delta Attention）算子。模型无关：只按形状参数，不含任何模型名 / 层号 / 调优常量。
//
// 数学口径（与 vLLM third_party/flash_linear_attention/ops/kda.py 及
// models/kimi_k3/nvidia/ops/third_party/kda/fused_recurrent.py 的 packed decode 一致）：
//   1. 短卷积（depthwise causal conv1d, width=W, siLU）
//        y[t, d] = silu( Σ_{j<W} w[d, j] · x[t-W+1+j, d] )
//      权重宽度维按时间从旧到新（w[...,0] 乘最早的输入）；越界部分取 conv_state 或 0。
//   2. gate（逐 k 维，per-head 标量 A_log）
//        a = exp(A_log[h]); x = g[t,h,k] + dt_bias[h,k]
//        gate = lower_bound · sigmoid(a·x)          （有 lower_bound，K3 用这条）
//             = -a · softplus(x)                    （没有时）
//       K3 的 gate_lower_bound = -5，所以 gate ≤ 0，exp(gate) 单调衰减。
//   3. beta = sigmoid(raw_beta[t,h])（逐头标量）
//   4. q,k 在 kernel 内做 l2 归一化：q /= sqrt(Σq² + 1e-6)，再乘 scale = head_dim^-0.5
//   5. 递归核心（逐 token，S 为 fp32 [V, K]）
//        S *= exp(gate)[None, :]
//        v' = (v - S·k) · beta
//        S += v'[:, None] · k
//        o  = S·q
//   6. 输出门：o = RMSNorm(o, w, eps) · sigmoid(g2)
//
// 这里给的是**顺序递归**实现：prefill 与 decode 走同一条路径（varlen + 状态槽），
// 慢但对拍精确；块级并行（chunk 形式，BT=64）留 v2，见 docs/loops/k3-kda.md。
#include "common.cuh"
#include "kernels.h"

using namespace dinfer;

namespace {

__device__ __forceinline__ float sigmoid_f(float x) { return 1.f / (1.f + __expf(-x)); }

// ---------------------------------------------------------------- conv1d prefill
// 每 block 一个 (序列, 通道块)。线程按通道并行，时间维串行，寄存器窗口保存 W-1 个历史值。
// conv_state 布局 [slots, dim, W-1]，列 0 最旧、列 W-2 最新（与 vLLM 一致）。
__global__ void conv1d_prefill_kernel(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                      bf16* __restrict__ out, bf16* __restrict__ conv_state,
                                      const int* __restrict__ cu_seqlens,
                                      const int* __restrict__ state_indices,
                                      const int* __restrict__ has_initial, int dim, int width,
                                      int state_len, int dim_per_block) {
  const int seq = blockIdx.x;
  const int d = blockIdx.y * dim_per_block + threadIdx.x;
  if (d >= dim) return;
  const int slot = state_indices ? state_indices[seq] : seq;
  if (slot < 0) return;

  const int bos = cu_seqlens[seq], eos = cu_seqlens[seq + 1];
  const int len = eos - bos;

  // 权重窗口（fp32），w[0] 乘最旧的输入
  float ww[8];
  for (int j = 0; j < width; ++j) ww[j] = to_f32(w[d * width + j]);

  // 历史值窗口：[0] 最旧 … [W-2] 最新
  float hist[8];
  for (int j = 0; j < width - 1; ++j) hist[j] = 0.f;
  if (conv_state && state_len > 0 && (!has_initial || has_initial[seq])) {
    const bf16* st = conv_state + ((long)slot * dim + d) * state_len;
    for (int j = 0; j < state_len; ++j) hist[j] = to_f32(st[j]);
  }

  for (int t = 0; t < len; ++t) {
    const float xt = to_f32(x[((long)(bos + t)) * dim + d]);
    // y = Σ_j w[j] · x[t-W+1+j]；最后一个 j 对应 xt 自己
    float acc = ww[width - 1] * xt;
    for (int j = 0; j < width - 1; ++j) acc += ww[j] * hist[j];
    out[((long)(bos + t)) * dim + d] = to_bf16(silu(acc));
    if (width > 1) {
      for (int j = 0; j < width - 2; ++j) hist[j] = hist[j + 1];
      hist[width - 2] = xt;
    }
  }

  // 回写最后 W-1 个原始输入；序列短于窗口时前面补 0
  if (conv_state && state_len > 0) {
    bf16* st = conv_state + ((long)slot * dim + d) * state_len;
    for (int j = 0; j < state_len; ++j) st[j] = to_bf16(hist[j]);
  }
}

// ---------------------------------------------------------------- conv1d update (decode, 1 token)
__global__ void conv1d_update_kernel(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                     bf16* __restrict__ out, bf16* __restrict__ conv_state,
                                     const int* __restrict__ state_indices, int batch, int dim,
                                     int width, int state_len) {
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
  out[(long)b * dim + d] = to_bf16(silu(acc));
  // 左移并追加新 token
  for (int j = 0; j + 1 < state_len; ++j) st[j] = st[j + 1];
  st[state_len - 1] = to_bf16(xt);
}

// ---------------------------------------------------------------- gate
// out[t,h,k] = lower_bound·sigmoid(exp(A_log[h])·(g[t,h,k] + dt_bias[h,k]))  （use_lower_bound）
//            = -exp(A_log[h])·softplus(g[t,h,k] + dt_bias[h,k])              （否则）
__global__ void gate_kernel(const bf16* __restrict__ g, const float* __restrict__ a_log,
                            const float* __restrict__ dt_bias, float* __restrict__ out, long total,
                            int head_dim, int heads, float lower_bound, int use_lower_bound) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int h = (int)((i / head_dim) % heads);
  const int k = (int)(i % head_dim);
  const float a = __expf(a_log[h]);
  const float x = to_f32(g[i]) + dt_bias[(long)h * head_dim + k];
  float y;
  if (use_lower_bound) {
    y = lower_bound * sigmoid_f(a * x);
  } else {
    // softplus 的数值稳定形式
    y = -a * (x > 20.f ? x : __logf(1.f + __expf(x)));
  }
  out[i] = y;
}

// ---------------------------------------------------------------- fused recurrent (prefill + decode)
// 每 block 一个 (序列, 头)。状态 [V,K] fp32 放动态 shared：一次 load、整段循环、一次 store。
// 共享内存用量 = (V·K + 3·K) · 4 字节；V=K=128 时 67 KB。
__global__ void recurrent_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k,
                                 const bf16* __restrict__ v, const bf16* __restrict__ raw_g,
                                 const float* __restrict__ raw_beta, const float* __restrict__ a_log,
                                 const float* __restrict__ dt_bias, bf16* __restrict__ out,
                                 float* __restrict__ state, const int* __restrict__ cu_seqlens,
                                 const int* __restrict__ state_indices, int heads, int K, int V,
                                 float scale, float lower_bound, int use_lower_bound,
                                 long state_stride_tok) {
  extern __shared__ float smem[];
  float* S = smem;            // [V, K]
  float* kbuf = S + (long)V * K;
  float* qbuf = kbuf + K;
  float* eg = qbuf + K;       // exp(gate)

  const int seq = blockIdx.x;
  const int h = blockIdx.y;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  const int slot = state_indices ? state_indices[seq] : seq;
  const int bos = cu_seqlens[seq], eos = cu_seqlens[seq + 1];
  const int len = eos - bos;

  float* Sg = state + (long)slot * state_stride_tok + (long)h * V * K;
  if (slot < 0 || len <= 0) {
    // 无效槽：输出清零即可（调用方不会读它）
    for (int i = tid; i < len; i += nthreads)
      for (int d = tid; d < V; d += nthreads) out[((long)(bos + i) * heads + h) * V + d] = to_bf16(0.f);
    return;
  }
  for (int i = tid; i < V * K; i += nthreads) S[i] = Sg[i];
  __syncthreads();

  const float a = __expf(a_log[h]);
  for (int t = 0; t < len; ++t) {
    const long row = ((long)(bos + t) * heads + h);
    if (tid < K) {
      const float qv = to_f32(q[row * K + tid]);
      const float kv = to_f32(k[row * K + tid]);
      kbuf[tid] = kv;
      qbuf[tid] = qv;  // 归一化在下面统一做
      const float x = to_f32(raw_g[row * K + tid]) + dt_bias[(long)h * K + tid];
      const float gate = use_lower_bound ? lower_bound * sigmoid_f(a * x)
                                         : -a * (x > 20.f ? x : __logf(1.f + __expf(x)));
      eg[tid] = __expf(gate);
    }
    // l2 归一化（q 再乘 scale）
    float qsq = (tid < K) ? qbuf[tid] * qbuf[tid] : 0.f;
    float ksq = (tid < K) ? kbuf[tid] * kbuf[tid] : 0.f;
    const float qs = block_reduce_sum<256>(qsq);
    // block_reduce_sum 内部复用同一块静态 shared 作中转，且返回值是最后一步读 smem[0]；
    // 紧接的第二次调用会立刻写 smem，必须在两次之间加屏障，否则先到的 warp 拿到后一次的值。
    __syncthreads();
    const float ks = block_reduce_sum<256>(ksq);
    if (tid < K) {
      qbuf[tid] = qbuf[tid] / sqrtf(qs + 1e-6f) * scale;
      kbuf[tid] = kbuf[tid] / sqrtf(ks + 1e-6f);
    }
    __syncthreads();

    const float beta = sigmoid_f(raw_beta[(long)(bos + t) * heads + h]);
    for (int vv = tid; vv < V; vv += nthreads) {
      float* r = S + (long)vv * K;
      float p = 0.f;
      for (int kk = 0; kk < K; ++kk) {
        r[kk] *= eg[kk];
        p += r[kk] * kbuf[kk];
      }
      const float vin = to_f32(v[(row)*V + vv]);
      const float vp = (vin - p) * beta;
      float o = 0.f;
      for (int kk = 0; kk < K; ++kk) {
        r[kk] += vp * kbuf[kk];
        o += r[kk] * qbuf[kk];
      }
      out[row * V + vv] = to_bf16(o);
    }
    __syncthreads();  // 下一 token 会覆盖 kbuf/qbuf/eg
  }

  for (int i = tid; i < V * K; i += nthreads) Sg[i] = S[i];
}

// ---------------------------------------------------------------- gated RMSNorm（输出门）
// y[t,h,d] = rmsnorm(o[t,h,d], w[d], eps) · sigmoid(g2[t,h,d])
__global__ void gated_rmsnorm_kernel(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                     const bf16* __restrict__ gate, bf16* __restrict__ out,
                                     int head_dim, float eps) {
  const long row = blockIdx.x;  // 一个 (token, head)
  const int tid = threadIdx.x;
  const bf16* xr = x + row * head_dim;
  float ss = 0.f;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    const float v = to_f32(xr[d]);
    ss += v * v;
  }
  const float s = block_reduce_sum<256>(ss);
  const float rstd = rsqrtf(s / head_dim + eps);
  for (int d = tid; d < head_dim; d += blockDim.x) {
    const float y = to_f32(xr[d]) * rstd * to_f32(w[d]) * sigmoid_f(to_f32(gate[row * head_dim + d]));
    out[row * head_dim + d] = to_bf16(y);
  }
}

int set_smem(const void* fn, int bytes) {
  return err(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

}  // namespace

extern "C" {

int kda_conv1d_prefill(const void* x, const void* w, void* out, void* conv_state, const int* cu_seqlens,
                       const int* state_indices, const int* has_initial, int64_t num_seqs, int64_t dim,
                       int64_t width, int64_t state_len) {
  if (num_seqs <= 0 || dim <= 0 || width <= 0 || width > 8) return err(cudaErrorInvalidValue);
  const int dim_per_block = 128;
  dim3 grid((unsigned)num_seqs, (unsigned)((dim + dim_per_block - 1) / dim_per_block));
  conv1d_prefill_kernel<<<grid, dim_per_block, 0, nullptr>>>(static_cast<const bf16*>(x),
                                                             static_cast<const bf16*>(w),
                                                             static_cast<bf16*>(out),
                                                             static_cast<bf16*>(conv_state), cu_seqlens,
                                                             state_indices, has_initial, (int)dim,
                                                             (int)width, (int)state_len, dim_per_block);
  return last_err();
}

int kda_conv1d_update(const void* x, const void* w, void* out, void* conv_state, const int* state_indices,
                      int64_t batch, int64_t dim, int64_t width, int64_t state_len) {
  if (batch <= 0 || dim <= 0 || width <= 0 || width > 8) return err(cudaErrorInvalidValue);
  const int nb = 256;
  dim3 grid((unsigned)batch, (unsigned)((dim + nb - 1) / nb));
  conv1d_update_kernel<<<grid, nb, 0, nullptr>>>(static_cast<const bf16*>(x), static_cast<const bf16*>(w),
                                                 static_cast<bf16*>(out), static_cast<bf16*>(conv_state),
                                                 state_indices, (int)batch, (int)dim, (int)width,
                                                 (int)state_len);
  return last_err();
}

int kda_gate(const void* g, const float* a_log, const float* dt_bias, float* out, int64_t tokens,
             int64_t heads, int64_t head_dim, float lower_bound, int use_lower_bound) {
  if (tokens <= 0 || heads <= 0 || head_dim <= 0) return err(cudaErrorInvalidValue);
  const long total = (long)tokens * heads * head_dim;
  const int nb = 256;
  gate_kernel<<<(unsigned)((total + nb - 1) / nb), nb, 0, nullptr>>>(
      static_cast<const bf16*>(g), a_log, dt_bias, out, total, (int)head_dim, (int)heads, lower_bound,
      use_lower_bound);
  return last_err();
}

int kda_recurrent(const void* q, const void* k, const void* v, const void* raw_g, const float* raw_beta,
                  const float* a_log, const float* dt_bias, void* out, void* state, const int* cu_seqlens,
                  const int* state_indices, int64_t num_seqs, int64_t heads, int64_t head_dim,
                  int64_t state_stride_tok, float scale, float lower_bound, int use_lower_bound) {
  if (num_seqs <= 0 || heads <= 0 || head_dim <= 0) return err(cudaErrorInvalidValue);
  const int smem = (int)((head_dim * head_dim + 3 * head_dim) * sizeof(float));
  // thread_local：cudaFuncSetAttribute 按设备生效，一进程多线程 TP 每线程一张卡（pitfalls 2026-10-20 C-6 卡死）
  static thread_local int smem_set = 0;
  if (smem_set < smem) {
    const int rc = set_smem(reinterpret_cast<const void*>(&recurrent_kernel), smem);
    if (rc != 0) return rc;
    smem_set = smem;
  }
  const int nb = 256;
  dim3 grid((unsigned)num_seqs, (unsigned)heads);
  recurrent_kernel<<<grid, nb, smem, nullptr>>>(static_cast<const bf16*>(q), static_cast<const bf16*>(k),
                                                static_cast<const bf16*>(v), static_cast<const bf16*>(raw_g),
                                                raw_beta, a_log, dt_bias, static_cast<bf16*>(out),
                                                static_cast<float*>(state), cu_seqlens, state_indices,
                                                (int)heads, (int)head_dim, (int)head_dim, scale,
                                                lower_bound, use_lower_bound, state_stride_tok);
  return last_err();
}

int kda_gated_rmsnorm(const void* x, const void* w, const void* gate, void* out, int64_t rows,
                      int64_t head_dim, float eps) {
  if (rows <= 0 || head_dim <= 0) return err(cudaErrorInvalidValue);
  gated_rmsnorm_kernel<<<(unsigned)rows, 256, 0, nullptr>>>(static_cast<const bf16*>(x),
                                                           static_cast<const bf16*>(w),
                                                           static_cast<const bf16*>(gate),
                                                           static_cast<bf16*>(out), (int)head_dim, eps);
  return last_err();
}

}  // extern "C"
