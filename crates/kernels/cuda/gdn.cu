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
// 每 block 一个 (序列, 通道块)。线程按通道并行，时间维串行，寄存器窗口保存 W-1 个历史值。
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
// 共享内存 = (V·K + 2·K) · 4 字节；V=K=128 时 66 KB。
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
  float* S = smem;             // [V, K]
  float* kbuf = S + (long)V * K;
  float* qbuf = kbuf + K;
  const int tid = threadIdx.x;

  const int seq = blockIdx.x;
  const int hv = blockIdx.y;
  const int h = hv / group;
  const int slot = state_indices ? state_indices[seq] : seq;
  const int bos = cu_seqlens[seq], eos = cu_seqlens[seq + 1];
  const int len = eos - bos;

  const long qkv_row = 2L * H * K + (long)HV * V;  // mixed_qkv 的每 token 宽度
  float* Sg = state + (long)slot * state_stride_tok + (long)hv * V * K;
  if (slot < 0 || len <= 0) return;
  for (int i = tid; i < V * K; i += BLOCK) S[i] = Sg[i];
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

    for (int vv = tid; vv < V; vv += BLOCK) {
      float* r = S + (long)vv * K;
      float p = 0.f;
      for (int kk = 0; kk < K; ++kk) {
        r[kk] *= eg;
        p += r[kk] * kbuf[kk];
      }
      const float vin = to_f32(vp[row * qkv_row + vv]);
      const float vprime = (vin - p) * beta;
      float o = 0.f;
      for (int kk = 0; kk < K; ++kk) {
        r[kk] += vprime * kbuf[kk];
        o += r[kk] * qbuf[kk];
      }
      out[(row * HV + hv) * V + vv] = to_bf16(o);
    }
    __syncthreads();  // 下一 token 会覆盖 kbuf/qbuf
  }

  for (int i = tid; i < V * K; i += BLOCK) Sg[i] = S[i];
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

int set_smem(const void* fn, int bytes) {
  return err(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

constexpr int kBlock = 256;

}  // namespace

extern "C" {

int gdn_conv1d_prefill(const void* x, const void* w, void* out, void* conv_state,
                       const int* cu_seqlens, const int* state_indices, const int* has_initial,
                       int64_t num_seqs, int64_t dim, int64_t width, int64_t state_len, int act) {
  if (num_seqs <= 0) return 0;
  const int per_block = 256;
  dim3 grid((unsigned)num_seqs, (unsigned)((dim + per_block - 1) / per_block));
  conv1d_prefill_kernel<<<grid, per_block>>>(
      (const bf16*)x, (const bf16*)w, (bf16*)out, (bf16*)conv_state, cu_seqlens, state_indices,
      has_initial, (int)dim, (int)width, (int)state_len, per_block, act);
  return last_err();
}

int gdn_conv1d_update(const void* x, const void* w, void* out, void* conv_state,
                      const int* state_indices, int64_t batch, int64_t dim, int64_t width,
                      int64_t state_len, int act) {
  if (batch <= 0) return 0;
  const int per_block = 256;
  dim3 grid((unsigned)batch, (unsigned)((dim + per_block - 1) / per_block));
  conv1d_update_kernel<<<grid, per_block>>>((const bf16*)x, (const bf16*)w, (bf16*)out,
                                            (bf16*)conv_state, state_indices, (int)batch, (int)dim,
                                            (int)width, (int)state_len, act);
  return last_err();
}

int gdn_gate(const void* a, const void* b, const float* a_log, const float* dt_bias, float* g_out,
             float* beta_out, int64_t total, int64_t heads, int do_beta) {
  if (total <= 0) return 0;
  const long blocks = (long)((total + kBlock - 1) / kBlock);
  gate_kernel<<<(unsigned)blocks, kBlock>>>((const bf16*)a, (const bf16*)b, a_log, dt_bias, g_out,
                                            beta_out, (long)total, (int)heads, do_beta);
  return last_err();
}

// fused_gate = 0：读 g/beta（prefill）；= 1：kernel 内由 raw a/b 现算（decode）。
int gdn_recurrent(const void* mixed_qkv, const float* g, const float* beta, const void* a_raw,
                  const void* b_raw, const float* a_log, const float* dt_bias, void* out,
                  void* state, const int* cu_seqlens, const int* state_indices, int64_t num_seqs,
                  int64_t H, int64_t HV, int64_t K, int64_t V, int64_t group, float scale,
                  int64_t state_stride_tok, int fused_gate) {
  if (num_seqs <= 0) return 0;
  const int smem = (int)((V * K + 2 * K) * 4);
  dim3 grid((unsigned)num_seqs, (unsigned)HV);
  if (fused_gate) {
    set_smem((const void*)recurrent_kernel<kBlock, true>, smem);
    recurrent_kernel<kBlock, true><<<grid, kBlock, smem>>>(
        (const bf16*)mixed_qkv, g, beta, (const bf16*)a_raw, (const bf16*)b_raw, a_log, dt_bias,
        (bf16*)out, (float*)state, cu_seqlens, state_indices, (int)H, (int)HV, (int)K, (int)V,
        (int)group, scale, (long)state_stride_tok);
  } else {
    set_smem((const void*)recurrent_kernel<kBlock, false>, smem);
    recurrent_kernel<kBlock, false><<<grid, kBlock, smem>>>(
        (const bf16*)mixed_qkv, g, beta, (const bf16*)a_raw, (const bf16*)b_raw, a_log, dt_bias,
        (bf16*)out, (float*)state, cu_seqlens, state_indices, (int)H, (int)HV, (int)K, (int)V,
        (int)group, scale, (long)state_stride_tok);
  }
  return last_err();
}

int gdn_gated_rmsnorm(const void* x, const void* w, const void* z, void* out, int64_t rows,
                      int64_t head_dim, float eps, int act) {
  if (rows <= 0) return 0;
  gated_rmsnorm_kernel<<<(unsigned)rows, 256>>>((const bf16*)x, (const bf16*)w, (const bf16*)z,
                                                (bf16*)out, (int)head_dim, eps, act);
  return last_err();
}

}  // namespace
}  // extern "C"
