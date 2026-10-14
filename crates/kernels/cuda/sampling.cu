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

}  // namespace

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
