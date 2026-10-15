// GQA（分组多头）paged attention 参考 kernel（sm80+，bf16 in / fp32 累加）。Qwen3 等标准 dense 模型用。
// 结构与 mla.cu 同（每 (req, q_head) 一个 block，逐 query 行三遍扫 key：max / sum / 加权累加），差别：
//   - K/V 分开存：paged cache 布局 [num_blocks, block_size, kv_heads, 2, D]（每 slot 内按 kv_head 排，K 在 V 前）；
//   - q_head h 读 kv_head h / group（group = H / Hkv）；
//   - 无 top-k 稀疏。
//
// 三个入口：
//   gqa_cache_write_bf16   k, v: [T, Hkv, D] → cache（slot_mapping）
//   gqa_prefill_bf16       无历史的 causal prefill，直接读 k/v [T, Hkv, D]（不走 cache；cache 由 cache_write 另写）
//   gqa_paged_bf16         从 paged cache 读：decode（每请求 1 行）与带历史 chunked prefill（任意 query 长）统一路径
#include "common.cuh"
#include "dinfer_kernels.h"

using namespace dinfer;

namespace {

constexpr int GQ_BLOCK = 256;
constexpr int MAX_D = 256;

__device__ __forceinline__ int64_t slot_off(int64_t slot, int kv_heads, int D) { return slot * kv_heads * 2 * D; }

__global__ void gqa_cache_write_kernel(const bf16* __restrict__ k, const bf16* __restrict__ v, bf16* __restrict__ cache,
                                       const int64_t* __restrict__ slot_mapping, int kv_heads, int D, int block_size) {
  const int64_t t = blockIdx.x;
  const int64_t slot = slot_mapping[t];
  if (slot < 0) return;
  const int64_t blk = slot / block_size, off = slot % block_size;
  bf16* dst = cache + slot_off(blk * block_size + off, kv_heads, D);
  const bf16* ks = k + t * kv_heads * D;
  const bf16* vs = v + t * kv_heads * D;
  for (int i = threadIdx.x; i < kv_heads * D; i += blockDim.x) {
    const int h = i / D, d = i % D;
    dst[(h * 2 + 0) * D + d] = ks[i];
    dst[(h * 2 + 1) * D + d] = vs[i];
  }
}

// 一个 (req, q_head) block；q/k/v: [T, H|Hkv, D]；out: [T, H, D]。
__global__ void gqa_prefill_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
                                   bf16* __restrict__ out, const int32_t* __restrict__ cu_seqlens, int H, int Hkv, int D,
                                   float scale) {
  const int req = blockIdx.x, h = blockIdx.y;
  const int hk = h / (H / Hkv);
  const int start = cu_seqlens[req], end = cu_seqlens[req + 1];
  const int T = end - start;
  if (T <= 0) return;
  __shared__ float s_q[MAX_D];
  __shared__ float s_p[GQ_BLOCK];
  __shared__ float s_stat[2];
  for (int i = 0; i < T; ++i) {
    const bf16* qi = q + ((int64_t)(start + i) * H + h) * D;
    for (int d = threadIdx.x; d < D; d += GQ_BLOCK) s_q[d] = to_f32(qi[d]);
    __syncthreads();
    const int n = i + 1;
    auto kptr = [&](int j) { return k + ((int64_t)(start + j) * Hkv + hk) * D; };
    auto vptr = [&](int j) { return v + ((int64_t)(start + j) * Hkv + hk) * D; };
    float lmax = -INFINITY;
    for (int j = threadIdx.x; j < n; j += GQ_BLOCK) {
      const bf16* kj = kptr(j);
      float s = 0.f;
      for (int d = 0; d < D; ++d) s += s_q[d] * to_f32(kj[d]);
      lmax = fmaxf(lmax, s * scale);
    }
    const float m = block_reduce_max<GQ_BLOCK>(lmax);
    float lsum = 0.f;
    for (int j = threadIdx.x; j < n; j += GQ_BLOCK) {
      const bf16* kj = kptr(j);
      float s = 0.f;
      for (int d = 0; d < D; ++d) s += s_q[d] * to_f32(kj[d]);
      lsum += __expf(s * scale - m);
    }
    const float denom = block_reduce_sum<GQ_BLOCK>(lsum);
    if (threadIdx.x == 0) {
      s_stat[0] = m;
      s_stat[1] = denom;
    }
    __syncthreads();
    float acc[MAX_D / GQ_BLOCK + 1];
#pragma unroll
    for (int c = 0; c < MAX_D / GQ_BLOCK + 1; ++c) acc[c] = 0.f;
    for (int base = 0; base < n; base += GQ_BLOCK) {
      const int j = base + threadIdx.x;
      float p = 0.f;
      if (j < n) {
        const bf16* kj = kptr(j);
        float s = 0.f;
        for (int d = 0; d < D; ++d) s += s_q[d] * to_f32(kj[d]);
        p = __expf(s * scale - s_stat[0]) / s_stat[1];
      }
      s_p[threadIdx.x] = p;
      __syncthreads();
      const int cnt = min(GQ_BLOCK, n - base);
      for (int d = threadIdx.x, c = 0; d < D; d += GQ_BLOCK, ++c) {
        float a = 0.f;
        for (int jj = 0; jj < cnt; ++jj) a += s_p[jj] * to_f32(vptr(base + jj)[d]);
        acc[c] += a;
      }
      __syncthreads();
    }
    bf16* oi = out + ((int64_t)(start + i) * H + h) * D;
    for (int d = threadIdx.x, c = 0; d < D; d += GQ_BLOCK, ++c) oi[d] = to_bf16(acc[c]);
    __syncthreads();
  }
}

// 一个 (req, q_head) block；q: [Tq, H, D]（Tq = Σ query 长）；cache 见上；cu_seqlens_q: [B+1]；seq_lens: [B] = 本步后总长。
// query 行 i 的绝对位置 pos = seq_len - q_len + i，可见 key = [0, pos]。decode 即 q_len = 1。
__global__ void gqa_paged_kernel(const bf16* __restrict__ q, const bf16* __restrict__ cache,
                                 const int32_t* __restrict__ block_table, const int32_t* __restrict__ seq_lens,
                                 const int32_t* __restrict__ cu_seqlens_q, bf16* __restrict__ out, int H, int Hkv, int D,
                                 int block_size, int max_blocks, float scale) {
  const int b = blockIdx.x, h = blockIdx.y;
  const int hk = h / (H / Hkv);
  const int q0 = cu_seqlens_q[b], q1 = cu_seqlens_q[b + 1];
  const int q_len = q1 - q0;
  const int seq_len = seq_lens[b];
  if (q_len <= 0) return;
  const int hist = seq_len - q_len;
  __shared__ float s_q[MAX_D];
  __shared__ float s_p[GQ_BLOCK];
  __shared__ float s_stat[2];
  const int32_t* bt = block_table + (int64_t)b * max_blocks;
  auto kptr = [&](int j) {
    const int64_t blk = bt[j / block_size];
    return cache + slot_off(blk * block_size + (j % block_size), Hkv, D) + (hk * 2 + 0) * D;
  };
  auto vptr = [&](int j) {
    const int64_t blk = bt[j / block_size];
    return cache + slot_off(blk * block_size + (j % block_size), Hkv, D) + (hk * 2 + 1) * D;
  };
  for (int i = 0; i < q_len; ++i) {
    const int row = q0 + i;
    const int n = hist + i + 1;
    const bf16* qi = q + ((int64_t)row * H + h) * D;
    for (int d = threadIdx.x; d < D; d += GQ_BLOCK) s_q[d] = to_f32(qi[d]);
    __syncthreads();
    float lmax = -INFINITY;
    for (int j = threadIdx.x; j < n; j += GQ_BLOCK) {
      const bf16* kj = kptr(j);
      float s = 0.f;
      for (int d = 0; d < D; ++d) s += s_q[d] * to_f32(kj[d]);
      lmax = fmaxf(lmax, s * scale);
    }
    const float m = block_reduce_max<GQ_BLOCK>(lmax);
    float lsum = 0.f;
    for (int j = threadIdx.x; j < n; j += GQ_BLOCK) {
      const bf16* kj = kptr(j);
      float s = 0.f;
      for (int d = 0; d < D; ++d) s += s_q[d] * to_f32(kj[d]);
      lsum += __expf(s * scale - m);
    }
    const float denom = block_reduce_sum<GQ_BLOCK>(lsum);
    if (threadIdx.x == 0) {
      s_stat[0] = m;
      s_stat[1] = denom;
    }
    __syncthreads();
    float acc[MAX_D / GQ_BLOCK + 1];
#pragma unroll
    for (int c = 0; c < MAX_D / GQ_BLOCK + 1; ++c) acc[c] = 0.f;
    for (int base = 0; base < n; base += GQ_BLOCK) {
      const int j = base + threadIdx.x;
      float p = 0.f;
      if (j < n) {
        const bf16* kj = kptr(j);
        float s = 0.f;
        for (int d = 0; d < D; ++d) s += s_q[d] * to_f32(kj[d]);
        p = __expf(s * scale - s_stat[0]) / s_stat[1];
      }
      s_p[threadIdx.x] = p;
      __syncthreads();
      const int cnt = min(GQ_BLOCK, n - base);
      for (int d = threadIdx.x, c = 0; d < D; d += GQ_BLOCK, ++c) {
        float a = 0.f;
        for (int jj = 0; jj < cnt; ++jj) a += s_p[jj] * to_f32(vptr(base + jj)[d]);
        acc[c] += a;
      }
      __syncthreads();
    }
    bf16* oi = out + ((int64_t)row * H + h) * D;
    for (int d = threadIdx.x, c = 0; d < D; d += GQ_BLOCK, ++c) oi[d] = to_bf16(acc[c]);
    __syncthreads();
  }
}

}  // namespace

extern "C" int gqa_cache_write_bf16(const void* k, const void* v, void* cache, const int64_t* slot_mapping,
                                    int64_t tokens, int64_t kv_heads, int64_t head_dim, int64_t block_size,
                                    cudaStream_t stream) {
  if (tokens <= 0) return 0;
  if (kv_heads <= 0 || head_dim <= 0 || block_size <= 0) return err(cudaErrorInvalidValue);
  gqa_cache_write_kernel<<<(unsigned)tokens, 256, 0, stream>>>(static_cast<const bf16*>(k), static_cast<const bf16*>(v),
                                                               static_cast<bf16*>(cache), slot_mapping, (int)kv_heads,
                                                               (int)head_dim, (int)block_size);
  return last_err();
}

extern "C" int gqa_prefill_bf16(const void* q, const void* k, const void* v, void* out, const int32_t* cu_seqlens,
                                int64_t num_reqs, int64_t heads, int64_t kv_heads, int64_t head_dim, float scale,
                                cudaStream_t stream) {
  if (num_reqs <= 0) return 0;
  if (head_dim <= 0 || head_dim > MAX_D || heads <= 0 || kv_heads <= 0 || heads % kv_heads != 0)
    return err(cudaErrorInvalidValue);
  dim3 grid((unsigned)num_reqs, (unsigned)heads);
  gqa_prefill_kernel<<<grid, GQ_BLOCK, 0, stream>>>(static_cast<const bf16*>(q), static_cast<const bf16*>(k),
                                                    static_cast<const bf16*>(v), static_cast<bf16*>(out), cu_seqlens,
                                                    (int)heads, (int)kv_heads, (int)head_dim, scale);
  return last_err();
}

extern "C" int gqa_paged_bf16(const void* q, const void* cache, const int32_t* block_table, const int32_t* seq_lens,
                              const int32_t* cu_seqlens_q, void* out, int64_t num_reqs, int64_t heads, int64_t kv_heads,
                              int64_t head_dim, int64_t block_size, int64_t max_blocks, float scale,
                              cudaStream_t stream) {
  if (num_reqs <= 0) return 0;
  if (head_dim <= 0 || head_dim > MAX_D || heads <= 0 || kv_heads <= 0 || heads % kv_heads != 0 || block_size <= 0)
    return err(cudaErrorInvalidValue);
  dim3 grid((unsigned)num_reqs, (unsigned)heads);
  gqa_paged_kernel<<<grid, GQ_BLOCK, 0, stream>>>(static_cast<const bf16*>(q), static_cast<const bf16*>(cache),
                                                  block_table, seq_lens, cu_seqlens_q, static_cast<bf16*>(out),
                                                  (int)heads, (int)kv_heads, (int)head_dim, (int)block_size,
                                                  (int)max_blocks, scale);
  return last_err();
}
