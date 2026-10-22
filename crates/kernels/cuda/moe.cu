// MoE 参考 kernel（sm80+）。不追求性能：每 token 一个 block 做路由；专家分组用 count + prefix + scatter。
//
//   moe_route_f32        logits[T,E] fp32 → topk_ids[T,K] i32, topk_w[T,K] f32
//                        语义 = HF DeepseekV3TopkRouter / vLLM grouped_topk（n_group=1）：
//                        scores = sigmoid(logits); choice = scores + bias; top-k by choice;
//                        weights = scores[chosen]（不含 bias）; if renorm: w /= sum(w)+1e-20; w *= scale
//   moe_bucket_bf16      按专家分桶：counts[E] → offsets[E+1]（exclusive scan），sorted_tok[T*K] / sorted_k[T*K]
//                        （两 kernel：count + fill；offsets 由 host 读 counts 做 scan 再传回，或 GPU 单 block scan）
//   moe_gather_bf16      按 sorted_tok 把 x[T,H] 行收集到 xg[T*K, H]
//   moe_scatter_add_bf16 out[tok] += w * y[i]（i 为 sorted 下标）
#include "common.cuh"
#include "kernels.h"

using namespace dinfer;

namespace {

constexpr int MAX_E = 1024;  // GLM 256 / K3 896
constexpr int MAX_K = 32;

__global__ void route_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                             int32_t* __restrict__ topk_ids, float* __restrict__ topk_w, int E, int K, int renorm,
                             float scale) {
  const int64_t t = blockIdx.x;
  __shared__ float s_score[MAX_E];
  __shared__ float s_choice[MAX_E];
  __shared__ int s_sel[MAX_K];
  for (int e = threadIdx.x; e < E; e += blockDim.x) {
    const float sc = 1.f / (1.f + __expf(-logits[t * E + e]));
    s_score[e] = sc;
    s_choice[e] = sc + (bias ? bias[e] : 0.f);
  }
  __syncthreads();
  // 单线程做 K 次 argmax（E ≤ 1024，K ≤ 32：参考实现足够）。相等取小下标，与 torch.topk 稳定序一致。
  if (threadIdx.x == 0) {
    float wsum = 0.f;
    for (int k = 0; k < K; ++k) {
      int best = -1;
      float bv = -INFINITY;
      for (int e = 0; e < E; ++e) {
        const float v = s_choice[e];
        if (v > bv) {
          bv = v;
          best = e;
        }
      }
      s_sel[k] = best;
      s_choice[best] = -INFINITY;
      wsum += s_score[best];
    }
    const float denom = renorm ? (wsum + 1e-20f) : 1.f;
    for (int k = 0; k < K; ++k) {
      topk_ids[t * K + k] = s_sel[k];
      topk_w[t * K + k] = s_score[s_sel[k]] / denom * scale;
    }
  }
}

// logits[t,e] = Σ_d f32(x[t,d]) · f32(w[e,d])，全程 fp32（HF 路由把 x 与 gate.weight 都 cast 到 fp32 再 linear）。
// 一个 block 算一个 (t, e)；参考版，不追求性能。
__global__ void gate_kernel(const bf16* __restrict__ x, const bf16* __restrict__ w, float* __restrict__ logits, int E,
                            int H) {
  const int64_t t = blockIdx.x / E;
  const int e = blockIdx.x % E;
  const bf16* xr = x + t * H;
  const bf16* wr = w + (int64_t)e * H;
  float acc = 0.f;
  for (int d = threadIdx.x; d < H; d += blockDim.x) acc += to_f32(xr[d]) * to_f32(wr[d]);
  acc = block_reduce_sum<128>(acc);
  if (threadIdx.x == 0) logits[t * E + e] = acc;
}

__global__ void count_kernel(const int32_t* __restrict__ topk_ids, int32_t* __restrict__ counts, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) atomicAdd(&counts[topk_ids[i]], 1);
}

// 单 block exclusive scan（E ≤ 1024）：offsets[0..E] ；并把 cursor[e] = offsets[e] 作为填充游标。
__global__ void scan_kernel(const int32_t* __restrict__ counts, int32_t* __restrict__ offsets,
                            int32_t* __restrict__ cursor, int E) {
  __shared__ int s[MAX_E];
  for (int e = threadIdx.x; e < E; e += blockDim.x) s[e] = counts[e];
  __syncthreads();
  if (threadIdx.x == 0) {
    int acc = 0;
    for (int e = 0; e < E; ++e) {
      offsets[e] = acc;
      cursor[e] = acc;
      acc += s[e];
    }
    offsets[E] = acc;
  }
}

// 把 (t, k) 对填到其专家的区间；区间内顺序不定（无需稳定：归约是加法）。
__global__ void fill_kernel(const int32_t* __restrict__ topk_ids, int32_t* __restrict__ cursor,
                            int32_t* __restrict__ sorted_tok, int32_t* __restrict__ sorted_k, int K, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    const int e = topk_ids[i];
    const int pos = atomicAdd(&cursor[e], 1);
    sorted_tok[pos] = (int32_t)(i / K);
    sorted_k[pos] = (int32_t)(i % K);
  }
}

__global__ void gather_kernel(const bf16* __restrict__ x, bf16* __restrict__ xg, const int32_t* __restrict__ sorted_tok,
                              int H) {
  const int64_t i = blockIdx.x;
  const bf16* src = x + (int64_t)sorted_tok[i] * H;
  bf16* dst = xg + i * H;
  for (int d = threadIdx.x; d < H; d += blockDim.x) dst[d] = src[d];
}

// out[sorted_tok[i], :] += topk_w[sorted_tok[i]*K + sorted_k[i]] * y[i, :]
// 同一 token 的 K 个专家可能并发写同一行 → 用 fp32 atomicAdd 到临时 fp32 累加器更稳；
// 参考版直接 bf16 读-加-写会有竞争。这里 out 用 fp32 累加器，最后 cast。
// mode：0 = 直接 `acc += f32(w) * f32(y)`（f32 权重，我们原口径）；
//   1 = vLLM `cutlass_moe` 合并口径：`w = bf16(f32(topk_w·per_expert_scale))`，每项 `bf16(w · f32(y))` 再累加。
// per_expert 为空指针时视为全 1（无 per-expert scale 的模型）。kernel 体内无 #if，口径由运行时参数选。
__global__ void scatter_add_kernel(const bf16* __restrict__ y, float* __restrict__ acc,
                                   const int32_t* __restrict__ sorted_tok, const int32_t* __restrict__ sorted_k,
                                   const float* __restrict__ topk_w, const float* __restrict__ per_expert,
                                   float w_scale, int K, int H, int mode) {
  const int64_t i = blockIdx.x;
  const int tok = sorted_tok[i];
  const int e = sorted_k[i];
  float w = topk_w[(int64_t)tok * K + e];
  if (per_expert) w *= per_expert[e] * w_scale;
  else w *= w_scale;
  const bf16* src = y + i * H;
  float* dst = acc + (int64_t)tok * H;
  for (int d = threadIdx.x; d < H; d += blockDim.x) {
    const float v = to_f32(src[d]);
    if (mode == 1) {
      const float wb = to_f32(to_bf16(w));
      atomicAdd(&dst[d], to_f32(to_bf16(wb * v)));
    } else {
      atomicAdd(&dst[d], w * v);
    }
  }
}

// out = bf16(acc) + out：先把 fp32 累加器 cast 成 bf16，再与 out（shared expert 输出）bf16 相加 → 与 HF
// `moe(...).type(bf16) + shared_experts(x)` 的舍入顺序一致。
__global__ void f32_add_to_bf16_kernel(const float* __restrict__ acc, bf16* __restrict__ out, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) out[i] = to_bf16(to_f32(to_bf16(acc[i])) + to_f32(out[i]));
}

inline int grid_for(int64_t n) {
  const int64_t want = (n + 255) / 256;
  return (int)(want > 4096 ? 4096 : want);
}

}  // namespace

extern "C" int moe_gate_f32(const void* x, const void* w, float* logits, int64_t tokens, int64_t num_experts,
                            int64_t hidden, cudaStream_t stream) {
  if (tokens <= 0) return 0;
  if (tokens * num_experts > 0x7fffffff) return err(cudaErrorInvalidValue);
  gate_kernel<<<(unsigned)(tokens * num_experts), 128, 0, stream>>>(static_cast<const bf16*>(x),
                                                                    static_cast<const bf16*>(w), logits,
                                                                    (int)num_experts, (int)hidden);
  return last_err();
}

extern "C" int moe_route_f32(const float* logits, const float* bias, int32_t* topk_ids, float* topk_w, int64_t tokens,
                             int64_t num_experts, int64_t topk, int renormalize, float scale, cudaStream_t stream) {
  if (tokens <= 0) return 0;
  if (num_experts <= 0 || num_experts > MAX_E || topk <= 0 || topk > MAX_K || topk > num_experts)
    return err(cudaErrorInvalidValue);
  route_kernel<<<(unsigned)tokens, 256, 0, stream>>>(logits, bias, topk_ids, topk_w, (int)num_experts, (int)topk,
                                                     renormalize, scale);
  return last_err();
}

// counts/offsets/cursor: 调用方提供的 [E] / [E+1] / [E] 缓冲（counts 需先清零）。
// ---------------------------------------------------------------------------
// Gemma4 路由：**全专家 softmax → 按原始 logits 取 top-k → top-k 上重新归一化 → 乘 per_expert_scale**。
// 与上面 route_kernel 的 sigmoid 语义完全不同（那是 DeepSeekV3 / GLM / K3 的口径），不能混用。
// 依据 vLLM `model_executor/models/gemma4.py`：
//   topk_ids = torch.topk(gating_output, k)[1]                 // 按原始 logits
//   router_probabilities = softmax(gating_output, dim=-1)      // 全专家
//   gate_weights = one_hot(topk_ids) * router_probabilities
//   dispatch_weights = gate_weights / sum(gate_weights)        // 只在 top-k 上
//   topk_weights = dispatch_weights.gather(1, topk_ids) * per_expert_scale[topk_ids]
__global__ void route_softmax_kernel(const float* __restrict__ logits, const int32_t* __restrict__ topk_ids,
                                     const float* __restrict__ per_expert_scale, float* __restrict__ topk_w, int E,
                                     int K) {
  const int64_t t = blockIdx.x;
  __shared__ float s_prob[MAX_E];
  __shared__ float s_sel[MAX_K];
  // 全专家 softmax（减最大值保证数值稳定，与 torch.softmax 同）
  __shared__ float s_max;
  if (threadIdx.x == 0) {
    float m = -INFINITY;
    for (int e = 0; e < E; ++e) m = fmaxf(m, logits[t * E + e]);
    s_max = m;
  }
  __syncthreads();
  float partial = 0.f;
  for (int e = threadIdx.x; e < E; e += blockDim.x) {
    const float p = __expf(logits[t * E + e] - s_max);
    s_prob[e] = p;
    partial += p;
  }
  // block 内归约求和
  __shared__ float s_red[32];
  {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1) partial += __shfl_down_sync(0xffffffff, partial, off);
    if (lane == 0) s_red[warp] = partial;
    __syncthreads();
    if (threadIdx.x == 0) {
      float sum = 0.f;
      const int nw = (blockDim.x + 31) >> 5;
      for (int i = 0; i < nw; ++i) sum += s_red[i];
      s_red[0] = sum > 0.f ? 1.f / sum : 1.f;
    }
    __syncthreads();
  }
  const float inv_sum = s_red[0];
  // top-k 的 id 由 moe_route 的前一步给出（按原始 logits 排序），这里只算权重
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    const int e = topk_ids[t * K + k];
    s_sel[k] = s_prob[e] * inv_sum;  // softmax 概率
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float wsum = 0.f;
    for (int k = 0; k < K; ++k) wsum += s_sel[k];
    const float inv_w = wsum > 0.f ? 1.f / wsum : 1.f;  // 只在 top-k 上重新归一化
    for (int k = 0; k < K; ++k) {
      const int e = topk_ids[t * K + k];
      topk_w[t * K + k] = s_sel[k] * inv_w * (per_expert_scale ? per_expert_scale[e] : 1.f);
    }
  }
}

// 按原始 logits 取 top-k（Gemma4 口径：不看 softmax 概率的大小关系，两者单调等价，但按 logits 可避免
// exp 下溢导致全 0 时排序不稳定）。
__global__ void route_topk_ids_kernel(const float* __restrict__ logits, int32_t* __restrict__ topk_ids, int E,
                                      int K) {
  const int64_t t = blockIdx.x;
  __shared__ float s_val[MAX_E];
  for (int e = threadIdx.x; e < E; e += blockDim.x) s_val[e] = logits[t * E + e];
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int k = 0; k < K; ++k) {
      int best = -1;
      float bv = -INFINITY;
      for (int e = 0; e < E; ++e) {
        const float v = s_val[e];
        if (v > bv) {
          bv = v;
          best = e;
        }
      }
      topk_ids[t * K + k] = best;
      s_val[best] = -INFINITY;
    }
  }
}

extern "C" int moe_route_softmax_f32(const float* logits, int32_t* topk_ids, const float* per_expert_scale,
                                     float* topk_w, int64_t tokens, int64_t num_experts, int64_t topk,
                                     cudaStream_t stream) {
  if (num_experts <= 0 || num_experts > MAX_E || topk <= 0 || topk > MAX_K || topk > num_experts) return -1;
  if (tokens <= 0) return 0;
  route_topk_ids_kernel<<<(unsigned)tokens, 256, 0, stream>>>(logits, topk_ids, (int)num_experts, (int)topk);
  route_softmax_kernel<<<(unsigned)tokens, 256, 0, stream>>>(logits, topk_ids, per_expert_scale, topk_w,
                                                             (int)num_experts, (int)topk);
  return 0;
}

extern "C" int moe_bucket(const int32_t* topk_ids, int32_t* counts, int32_t* offsets, int32_t* cursor,
                          int32_t* sorted_tok, int32_t* sorted_k, int64_t tokens, int64_t num_experts, int64_t topk,
                          cudaStream_t stream) {
  if (tokens <= 0) return 0;
  if (num_experts <= 0 || num_experts > MAX_E) return err(cudaErrorInvalidValue);
  const int64_t n = tokens * topk;
  if (cudaMemsetAsync(counts, 0, num_experts * sizeof(int32_t), stream) != cudaSuccess) return last_err();
  count_kernel<<<grid_for(n), 256, 0, stream>>>(topk_ids, counts, n);
  scan_kernel<<<1, 1024, 0, stream>>>(counts, offsets, cursor, (int)num_experts);
  fill_kernel<<<grid_for(n), 256, 0, stream>>>(topk_ids, cursor, sorted_tok, sorted_k, (int)topk, n);
  return last_err();
}

extern "C" int moe_gather_bf16(const void* x, void* xg, const int32_t* sorted_tok, int64_t n, int64_t hidden,
                               cudaStream_t stream) {
  if (n <= 0) return 0;
  const int threads = hidden >= 256 ? 256 : (int)(((hidden + 31) / 32) * 32);
  gather_kernel<<<(unsigned)n, threads, 0, stream>>>(static_cast<const bf16*>(x), static_cast<bf16*>(xg), sorted_tok,
                                                     (int)hidden);
  return last_err();
}

// acc: [tokens, hidden] fp32，需先清零；y: [n, hidden] bf16（n = tokens*topk 的排序后行）。
extern "C" int moe_scatter_add_f32(const void* y, float* acc, const int32_t* sorted_tok, const int32_t* sorted_k,
                                   const float* topk_w, const float* per_expert, float w_scale,
                                   int64_t n, int64_t topk, int64_t hidden, int mode, cudaStream_t stream) {
  if (n <= 0) return 0;
  const int threads = hidden >= 256 ? 256 : (int)(((hidden + 31) / 32) * 32);
  scatter_add_kernel<<<(unsigned)n, threads, 0, stream>>>(static_cast<const bf16*>(y), acc, sorted_tok, sorted_k,
                                                          topk_w, per_expert, w_scale, (int)topk, (int)hidden, mode);
  return last_err();
}

// out[:n] (bf16) += acc[:n] (fp32)
extern "C" int f32_add_to_bf16(const float* acc, void* out, int64_t n, cudaStream_t stream) {
  if (n <= 0) return 0;
  f32_add_to_bf16_kernel<<<grid_for(n), 256, 0, stream>>>(acc, static_cast<bf16*>(out), n);
  return last_err();
}
