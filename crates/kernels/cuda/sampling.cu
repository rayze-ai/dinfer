// 采样参考 kernel。每行一个 block；vocab 可到 16 万（K3 163840 / GLM 154880）。
// 目标是正确性与可读性；性能版（radix select、无全排序）留到 P10。
#include "common.cuh"
#include "dinfer_kernels.h"

using namespace dinfer;

namespace {

constexpr int BLOCK = 1024;

// ---- argmax ----
__global__ void argmax_kernel(const float* __restrict__ logits, int32_t* __restrict__ out, int vocab) {
  const float* row = logits + (int64_t)blockIdx.x * vocab;
  float best = -INFINITY;
  int besti = 0;
  for (int i = threadIdx.x; i < vocab; i += BLOCK) {
    const float v = row[i];
    if (v > best || (v == best && i < besti)) {
      best = v;
      besti = i;
    }
  }
  // block 内归约 (value, index)，相等取小 index。
  __shared__ float sv[BLOCK / 32];
  __shared__ int si[BLOCK / 32];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, besti, o);
    if (ov > best || (ov == best && oi < besti)) {
      best = ov;
      besti = oi;
    }
  }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) {
    sv[warp] = best;
    si[warp] = besti;
  }
  __syncthreads();
  if (warp == 0) {
    best = lane < BLOCK / 32 ? sv[lane] : -INFINITY;
    besti = lane < BLOCK / 32 ? si[lane] : 0x7fffffff;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best, o);
      const int oi = __shfl_xor_sync(0xffffffffu, besti, o);
      if (ov > best || (ov == best && oi < besti)) {
        best = ov;
        besti = oi;
      }
    }
    if (lane == 0) out[blockIdx.x] = besti;
  }
}

// ---- top-k / top-p 掩码（参考实现）----
// 思路：找到第 k 大的阈值 t_k（二分 + 计数），以及 top-p 的阈值 t_p（二分：使 sum(softmax(v>=t)) >= p 的最大 t）。
// 两者取 max 作为保留阈值；低于阈值的置 -inf。精度上与排序法一致到 float 比较。
__device__ float row_max(const float* row, int vocab) {
  float m = -INFINITY;
  for (int i = threadIdx.x; i < vocab; i += BLOCK) m = fmaxf(m, row[i]);
  return block_reduce_max<BLOCK>(m);
}

__device__ int count_ge(const float* row, int vocab, float t) {
  float c = 0.f;
  for (int i = threadIdx.x; i < vocab; i += BLOCK) c += row[i] >= t ? 1.f : 0.f;
  return (int)block_reduce_sum<BLOCK>(c);
}

// 只统计 v >= floor 的元素（floor 是 top-k 阈值），在该子集上算 softmax 质量。
__device__ float prob_mass_ge(const float* row, int vocab, float t, float m, float denom, float floor_) {
  float s = 0.f;
  for (int i = threadIdx.x; i < vocab; i += BLOCK) {
    const float v = row[i];
    if (v >= t && v >= floor_) s += __expf(v - m);
  }
  return block_reduce_sum<BLOCK>(s) / denom;
}

// 语义与 vLLM `apply_top_k_top_p_pytorch` 一致：先 top-k 置 -inf，再在剩余分布上做 top-p，
// 保留"按概率降序累计到 p 为止（含跨过 p 的那个）"。
__global__ void topk_topp_mask_kernel(float* __restrict__ logits, const int32_t* __restrict__ top_k,
                                      const float* __restrict__ top_p, int vocab) {
  float* row = logits + (int64_t)blockIdx.x * vocab;
  const int k = top_k[blockIdx.x];
  const float p = top_p[blockIdx.x];
  const bool do_k = k > 0 && k < vocab;
  const bool do_p = p < 1.f && p > 0.f;
  if (!do_k && !do_p) return;

  const float m = row_max(row, vocab);
  float lo_local = INFINITY;
  for (int i = threadIdx.x; i < vocab; i += BLOCK) {
    const float v = row[i];
    if (v > -INFINITY) lo_local = fminf(lo_local, v);
  }
  const float lo = -block_reduce_max<BLOCK>(-lo_local);
  const float hi = m;

  // 1) top-k 阈值：最大的 t 使 count(v >= t) >= k。
  float floor_ = -INFINITY;
  if (do_k) {
    float a = lo, b = hi;
    for (int it = 0; it < 48 && b > a; ++it) {
      const float mid = 0.5f * (a + b);
      if (count_ge(row, vocab, mid) >= k) a = mid; else b = mid;
    }
    floor_ = a;
  }
  // 2) top-p 阈值：在 v >= floor_ 的子集上，最大的 t 使 mass(v >= t) >= p。
  float thresh = floor_;
  if (do_p) {
    float d = 0.f;
    for (int i = threadIdx.x; i < vocab; i += BLOCK) {
      const float v = row[i];
      if (v >= floor_) d += __expf(v - m);
    }
    const float denom = block_reduce_sum<BLOCK>(d);
    float a = fmaxf(lo, floor_), b = hi;
    for (int it = 0; it < 48 && b > a; ++it) {
      const float mid = 0.5f * (a + b);
      if (prob_mass_ge(row, vocab, mid, m, denom, floor_) >= p) a = mid; else b = mid;
    }
    thresh = fmaxf(thresh, a);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < vocab; i += BLOCK) {
    if (row[i] < thresh) row[i] = -INFINITY;
  }
}

// ---- 温度 + softmax + 多项式采样 ----
// Philox4x32-10 单次调用产一个 [0,1) 随机数。
__device__ __forceinline__ uint32_t mulhilo(uint32_t a, uint32_t b, uint32_t* hi) {
  const uint64_t p = (uint64_t)a * b;
  *hi = (uint32_t)(p >> 32);
  return (uint32_t)p;
}

__device__ float philox_uniform(uint64_t seed, uint64_t offset) {
  uint32_t c0 = (uint32_t)offset, c1 = (uint32_t)(offset >> 32), c2 = 0, c3 = 0;
  uint32_t k0 = (uint32_t)seed, k1 = (uint32_t)(seed >> 32);
#pragma unroll
  for (int r = 0; r < 10; ++r) {
    uint32_t hi0, hi1;
    const uint32_t lo0 = mulhilo(0xD2511F53u, c0, &hi0);
    const uint32_t lo1 = mulhilo(0xCD9E8D57u, c2, &hi1);
    const uint32_t n0 = hi1 ^ c1 ^ k0, n1 = lo1, n2 = hi0 ^ c3 ^ k1, n3 = lo0;
    c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  return (c0 >> 8) * (1.f / 16777216.f);  // 24-bit mantissa → [0,1)
}

__global__ void sample_kernel(const float* __restrict__ logits, const float* __restrict__ temperature,
                              const uint64_t* __restrict__ seeds, const uint64_t* __restrict__ offsets,
                              int32_t* __restrict__ out, int vocab) {
  const int b = blockIdx.x;
  const float* row = logits + (int64_t)b * vocab;
  const float temp = temperature[b];
  __shared__ float s_target;
  __shared__ int s_choice;

  if (temp <= 1e-5f) {
    // greedy：复用 argmax 逻辑（简单版）
    float best = -INFINITY;
    int besti = 0;
    for (int i = threadIdx.x; i < vocab; i += BLOCK) {
      const float v = row[i];
      if (v > best || (v == best && i < besti)) { best = v; besti = i; }
    }
    __shared__ float sv[BLOCK / 32];
    __shared__ int si[BLOCK / 32];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best, o);
      const int oi = __shfl_xor_sync(0xffffffffu, besti, o);
      if (ov > best || (ov == best && oi < besti)) { best = ov; besti = oi; }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { sv[warp] = best; si[warp] = besti; }
    __syncthreads();
    if (warp == 0) {
      best = lane < BLOCK / 32 ? sv[lane] : -INFINITY;
      besti = lane < BLOCK / 32 ? si[lane] : 0x7fffffff;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, besti, o);
        if (ov > best || (ov == best && oi < besti)) { best = ov; besti = oi; }
      }
      if (lane == 0) out[b] = besti;
    }
    return;
  }

  const float inv_t = 1.f / temp;
  // max（缩放后）
  float m = -INFINITY;
  for (int i = threadIdx.x; i < vocab; i += BLOCK) m = fmaxf(m, row[i] * inv_t);
  m = block_reduce_max<BLOCK>(m);
  // 分母
  float d = 0.f;
  for (int i = threadIdx.x; i < vocab; i += BLOCK) {
    const float v = row[i];
    if (v > -INFINITY) d += __expf(v * inv_t - m);
  }
  const float denom = block_reduce_sum<BLOCK>(d);
  if (threadIdx.x == 0) {
    s_target = philox_uniform(seeds[b], offsets[b]) * denom;
    s_choice = 0x7fffffff;  // INT_MAX：atomicMin 取最小命中下标
  }
  __syncthreads();
  const float target = s_target;

  // 逆 CDF：按 vocab 顺序做分块前缀和。每轮 BLOCK 个元素：block 内 scan，找第一个 cum >= target。
  __shared__ float s_prefix[BLOCK];
  __shared__ float s_carry;
  if (threadIdx.x == 0) s_carry = 0.f;
  __syncthreads();
  for (int base = 0; base < vocab && s_choice == 0x7fffffff; base += BLOCK) {
    const int i = base + threadIdx.x;
    float p = 0.f;
    if (i < vocab) {
      const float v = row[i];
      p = v > -INFINITY ? __expf(v * inv_t - m) : 0.f;
    }
    // 简单 Hillis-Steele inclusive scan
    s_prefix[threadIdx.x] = p;
    __syncthreads();
    for (int o = 1; o < BLOCK; o <<= 1) {
      const float add = threadIdx.x >= o ? s_prefix[threadIdx.x - o] : 0.f;
      __syncthreads();
      s_prefix[threadIdx.x] += add;
      __syncthreads();
    }
    const float carry = s_carry;
    const float cum = carry + s_prefix[threadIdx.x];
    const float prev = carry + (threadIdx.x > 0 ? s_prefix[threadIdx.x - 1] : 0.f);
    if (i < vocab && p > 0.f && prev < target && cum >= target) {
      atomicMin(&s_choice, i);
    }
    __syncthreads();
    if (threadIdx.x == 0) s_carry = carry + s_prefix[BLOCK - 1];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    int c = s_choice;
    if (c == 0x7fffffff) {
      // 数值边界：target 略大于总和，取最后一个非零概率 token。
      for (int i = vocab - 1; i >= 0; --i) {
        if (row[i] > -INFINITY) { c = i; break; }
      }
      if (c < 0) c = 0;
    }
    out[b] = c;
  }
}


// ---- penalties（vLLM ops/penalties.py 语义）----
// 对每行：对出现在 prompt 或 output 的 token：repetition：logit>0 ? /r : *r（r 作用于 prompt∪output）；
// frequency：logit -= f · count_in_output；presence：logit -= p · [in_output]。
// token 历史以 CSR 给：ids[offsets[b]..offsets[b+1]] 前 prompt_len[b] 个是 prompt，其余是 output。
__global__ void penalties_kernel(float* __restrict__ logits, const int32_t* __restrict__ ids,
                                 const int32_t* __restrict__ offsets, const int32_t* __restrict__ prompt_len,
                                 const float* __restrict__ rep, const float* __restrict__ freq,
                                 const float* __restrict__ pres, int vocab) {
  const int b = blockIdx.x;
  const float r = rep[b], f = freq[b], pr = pres[b];
  if (r == 1.f && f == 0.f && pr == 0.f) return;
  float* row = logits + (int64_t)b * vocab;
  const int lo = offsets[b], hi = offsets[b + 1], np = prompt_len[b];
  // 每个线程负责若干 token id 的 "出现次数" 统计需要跨线程归约；参考版：用 shared 的 vocab 位图太大（16 万）→
  // 直接按历史 token 遍历，用 atomic 在 logits 上改。重复 token 的 repetition 只能应用一次 → 先标记。
  // 简化：历史长度 ≤ max_model_len（几千），每线程处理一段历史；重复 id 由 "第一次出现才做 rep" 保证：
  // 做法是对每个历史位置 i，检查 [lo, i) 内是否已有同 id（O(n^2) 参考版，n ≤ 4k 可接受）。
  for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    const int t = ids[i];
    if (t < 0 || t >= vocab) continue;
    bool first = true;
    for (int j = lo; j < i; ++j) if (ids[j] == t) { first = false; break; }
    if (!first) continue;
    // 出现在 output 的次数
    int cnt = 0;
    for (int j = lo + np; j < hi; ++j) if (ids[j] == t) ++cnt;
    float v = row[t];
    if (r != 1.f) v = v > 0.f ? v / r : v * r;
    if (cnt > 0) v -= f * cnt + pr;
    row[t] = v;
  }
}

// ---- min_p（vLLM apply_min_p）：p_i < min_p · max_i p_i 的置 -inf；在 temperature 之后、top_k/p 之前 ----
__global__ void min_p_mask_kernel(float* __restrict__ logits, const float* __restrict__ min_p, int vocab) {
  const int b = blockIdx.x;
  const float mp = min_p[b];
  if (mp <= 0.f) return;
  float* row = logits + (int64_t)b * vocab;
  const float m = row_max(row, vocab);
  float d = 0.f;
  for (int i = threadIdx.x; i < vocab; i += BLOCK) { const float v = row[i]; if (v > -INFINITY) d += __expf(v - m); }
  const float denom = block_reduce_sum<BLOCK>(d);
  const float thresh = mp * (1.f / denom);  // max prob = exp(0)/denom
  __syncthreads();
  for (int i = threadIdx.x; i < vocab; i += BLOCK) {
    const float v = row[i];
    if (v > -INFINITY && __expf(v - m) / denom < thresh) row[i] = -INFINITY;
  }
}

// ---- 温度缩放（原地）----
__global__ void temperature_kernel(float* __restrict__ logits, const float* __restrict__ temperature, int vocab) {
  const int b = blockIdx.x;
  const float t = temperature[b];
  if (t <= 1e-5f || t == 1.f) return;
  float* row = logits + (int64_t)b * vocab;
  const float inv = 1.f / t;
  for (int i = threadIdx.x; i < vocab; i += BLOCK) row[i] *= inv;
}

// ---- logprobs：对原始 logits 的 log_softmax，取 top-k（k ≤ 32）+ 指定 token 的 logprob 与 rank ----
// out_ids/out_lp: [batch, k+1]：第 0 列是 chosen，之后 k 个 top（降序）；rank[b] = 严格大于 chosen logprob 的个数。
constexpr int MAX_LP_K = 32;
__global__ void logprobs_kernel(const float* __restrict__ logits, const int32_t* __restrict__ chosen, int k,
                                int32_t* __restrict__ out_ids, float* __restrict__ out_lp, int32_t* __restrict__ rank,
                                int vocab) {
  const int b = blockIdx.x;
  const float* row = logits + (int64_t)b * vocab;
  const float m = row_max(row, vocab);
  float d = 0.f;
  for (int i = threadIdx.x; i < vocab; i += BLOCK) { const float v = row[i]; if (v > -INFINITY) d += __expf(v - m); }
  const float lse = m + __logf(block_reduce_sum<BLOCK>(d));
  const int c = chosen[b];
  const float c_lp = row[c] - lse;
  // rank
  int gt = 0;
  for (int i = threadIdx.x; i < vocab; i += BLOCK) if (row[i] > row[c]) ++gt;
  gt = (int)block_reduce_sum<BLOCK>((float)gt);
  if (threadIdx.x == 0) { out_ids[b * (k + 1)] = c; out_lp[b * (k + 1)] = c_lp; rank[b] = gt; }
  // top-k：k 轮 argmax（排除已选），参考版
  __shared__ int s_excl[MAX_LP_K];
  for (int j = 0; j < k; ++j) {
    float best = -INFINITY; int besti = 0x7fffffff;
    for (int i = threadIdx.x; i < vocab; i += BLOCK) {
      bool ex = false;
      for (int e = 0; e < j; ++e) if (s_excl[e] == i) { ex = true; break; }
      if (ex) continue;
      const float v = row[i];
      if (v > best || (v == best && i < besti)) { best = v; besti = i; }
    }
    __shared__ float sv[BLOCK / 32]; __shared__ int si[BLOCK / 32];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best, o); const int oi = __shfl_xor_sync(0xffffffffu, besti, o);
      if (ov > best || (ov == best && oi < besti)) { best = ov; besti = oi; }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { sv[warp] = best; si[warp] = besti; }
    __syncthreads();
    if (warp == 0) {
      best = lane < BLOCK / 32 ? sv[lane] : -INFINITY; besti = lane < BLOCK / 32 ? si[lane] : 0x7fffffff;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o); const int oi = __shfl_xor_sync(0xffffffffu, besti, o);
        if (ov > best || (ov == best && oi < besti)) { best = ov; besti = oi; }
      }
      if (lane == 0) { s_excl[j] = besti; out_ids[b * (k + 1) + 1 + j] = besti; out_lp[b * (k + 1) + 1 + j] = best - lse; }
    }
    __syncthreads();
  }
}

// 异步调度：把上一步采样结果 prev[gather_idx[i]] 写进本步 ids[i]（gather_idx[i] < 0 的行不动）。
__global__ void row_gather_i32_kernel(const int32_t* __restrict__ prev, const int32_t* __restrict__ gather_idx,
                                      int32_t* __restrict__ ids, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int g = gather_idx[i];
  if (g >= 0) ids[i] = prev[g];
}
}  // namespace

extern "C" int penalties_f32(float* logits, const int32_t* ids, const int32_t* offsets, const int32_t* prompt_len,
                             const float* rep, const float* freq, const float* pres, int64_t batch, int64_t vocab,
                             cudaStream_t stream) {
  if (batch <= 0) return 0;
  penalties_kernel<<<(unsigned)batch, 256, 0, stream>>>(logits, ids, offsets, prompt_len, rep, freq, pres, (int)vocab);
  return last_err();
}
extern "C" int min_p_mask_f32(float* logits, const float* min_p, int64_t batch, int64_t vocab, cudaStream_t stream) {
  if (batch <= 0) return 0;
  min_p_mask_kernel<<<(unsigned)batch, BLOCK, 0, stream>>>(logits, min_p, (int)vocab);
  return last_err();
}
extern "C" int temperature_f32(float* logits, const float* temperature, int64_t batch, int64_t vocab, cudaStream_t stream) {
  if (batch <= 0) return 0;
  temperature_kernel<<<(unsigned)batch, BLOCK, 0, stream>>>(logits, temperature, (int)vocab);
  return last_err();
}
extern "C" int logprobs_f32(const float* logits, const int32_t* chosen, int64_t k, int32_t* out_ids, float* out_lp,
                            int32_t* rank, int64_t batch, int64_t vocab, cudaStream_t stream) {
  if (batch <= 0) return 0;
  if (k < 0 || k > MAX_LP_K) return err(cudaErrorInvalidValue);
  logprobs_kernel<<<(unsigned)batch, BLOCK, 0, stream>>>(logits, chosen, (int)k, out_ids, out_lp, rank, (int)vocab);
  return last_err();
}

extern "C" int argmax_f32(const float* logits, int32_t* out, int64_t batch, int64_t vocab, cudaStream_t stream) {
  if (batch <= 0) return 0;
  if (vocab <= 0) return err(cudaErrorInvalidValue);
  argmax_kernel<<<(unsigned)batch, BLOCK, 0, stream>>>(logits, out, (int)vocab);
  return last_err();
}

extern "C" int topk_topp_mask_f32(float* logits, const int32_t* top_k, const float* top_p, int64_t batch,
                                  int64_t vocab, cudaStream_t stream) {
  if (batch <= 0) return 0;
  if (vocab <= 0) return err(cudaErrorInvalidValue);
  topk_topp_mask_kernel<<<(unsigned)batch, BLOCK, 0, stream>>>(logits, top_k, top_p, (int)vocab);
  return last_err();
}

extern "C" int sample_f32(const float* logits, const float* temperature, const uint64_t* seeds,
                          const uint64_t* offsets, int32_t* out, int64_t batch, int64_t vocab,
                          cudaStream_t stream) {
  if (batch <= 0) return 0;
  if (vocab <= 0) return err(cudaErrorInvalidValue);
  sample_kernel<<<(unsigned)batch, BLOCK, 0, stream>>>(logits, temperature, seeds, offsets, out, (int)vocab);
  return last_err();
}

extern "C" int row_gather_i32(const int32_t* prev, const int32_t* gather_idx, int32_t* ids, int64_t n,
                              cudaStream_t stream) {
  if (n <= 0) return 0;
  row_gather_i32_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(prev, gather_idx, ids, (int)n);
  return last_err();
}
