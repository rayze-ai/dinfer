// MLA 参考 kernel（sm80+，bf16 in / fp32 累加）。目标是正确性与可读性；快路径（FlashMLA 等）P10 接。
//
// 三个入口：
//   mla_cache_write_bf16   把每 token 的 latent [kv_lora + rope] 写入 paged cache（slot_mapping）
//   mla_prefill_bf16       展开 K/V 的 causal attention（每 (req, head) 一个 block，逐 query 行扫描）
//   mla_decode_bf16        吸收后的 MQA decode：q_latent[heads, L+R] 对 paged latent cache，每 (req, head) 一个 block
//
// paged cache 布局：[num_blocks, block_size, L+R] bf16（与 vLLM concat_and_cache_mla 一致）。
#include "common.cuh"
#include "dinfer_kernels.h"

using namespace dinfer;

namespace {

// ---------- cache write ----------
__global__ void cache_write_kernel(const bf16* __restrict__ latent, bf16* __restrict__ cache,
                                   const int64_t* __restrict__ slot_mapping, int entry, int block_size) {
  const int64_t t = blockIdx.x;
  const int64_t slot = slot_mapping[t];
  if (slot < 0) return;
  const int64_t blk = slot / block_size, off = slot % block_size;
  const bf16* src = latent + t * entry;
  bf16* dst = cache + (blk * block_size + off) * entry;
  for (int i = threadIdx.x; i < entry; i += blockDim.x) dst[i] = src[i];
}

// ---------- prefill ----------
// q: [T, H, Dq]（Dq = nope + rope），k: [T, H, Dq]，v: [T, H, Dv]，out: [T, H, Dv]。
// 每 block 处理一个 (req, head)。线程按 key 维并行：每个 query 行 i，线程 j 负责 key j（j<=i），
// 两遍：先算 score 与 max/sum（block reduce），再加权累加 v（每线程负责若干 v 维）。
// 复杂度 O(T^2) 简单实现，T ≤ 几千可接受（tiny 对拍 / 开发期）。
constexpr int PF_BLOCK = 256;
constexpr int MAX_DQ = 320;  // 支持到 GLM 的 256 / K3 的 192
constexpr int MAX_DV = 256;

// topk_indices（可选，DSA 稀疏）：[T_total, topk] i32，每个 query 行只看列出的 key（请求内位置，-1 = 空）；
// 为空指针时 = 稠密 causal。gather 模式下 "第 jj 个可见 key" = topk_indices[row][jj]，不再是 j。
__global__ void prefill_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
                               bf16* __restrict__ out, const int32_t* __restrict__ cu_seqlens,
                               const int32_t* __restrict__ topk_indices, int topk, int H, int Dq, int Dv,
                               float scale) {
  const int req = blockIdx.x, h = blockIdx.y;
  const int start = cu_seqlens[req], end = cu_seqlens[req + 1];
  const int T = end - start;
  if (T <= 0) return;

  __shared__ float s_q[MAX_DQ];
  __shared__ float s_p[PF_BLOCK];  // 当前 chunk 的概率
  __shared__ int s_j[PF_BLOCK];    // 当前 chunk 的 key 位置（-1 = 跳过）
  __shared__ float s_stat[2];      // max, sum

  for (int i = 0; i < T; ++i) {
    // 载入 q_i
    const bf16* qi = q + ((int64_t)(start + i) * H + h) * Dq;
    for (int d = threadIdx.x; d < Dq; d += PF_BLOCK) s_q[d] = to_f32(qi[d]);
    __syncthreads();
    const int32_t* sel = topk_indices ? topk_indices + (int64_t)(start + i) * topk : nullptr;
    const int n = sel ? topk : i + 1;  // 候选数（gather 模式含 -1 空槽）
    auto key_pos = [&](int jj) -> int {
      const int j = sel ? sel[jj] : jj;
      return (j >= 0 && j <= i) ? j : -1;
    };

    // pass 1：score max
    float lmax = -INFINITY;
    for (int jj = threadIdx.x; jj < n; jj += PF_BLOCK) {
      const int j = key_pos(jj);
      if (j < 0) continue;
      const bf16* kj = k + ((int64_t)(start + j) * H + h) * Dq;
      float s = 0.f;
      for (int d = 0; d < Dq; ++d) s += s_q[d] * to_f32(kj[d]);
      lmax = fmaxf(lmax, s * scale);
    }
    const float m = block_reduce_max<PF_BLOCK>(lmax);
    // pass 2：sum of exp
    float lsum = 0.f;
    for (int jj = threadIdx.x; jj < n; jj += PF_BLOCK) {
      const int j = key_pos(jj);
      if (j < 0) continue;
      const bf16* kj = k + ((int64_t)(start + j) * H + h) * Dq;
      float s = 0.f;
      for (int d = 0; d < Dq; ++d) s += s_q[d] * to_f32(kj[d]);
      lsum += __expf(s * scale - m);
    }
    const float denom = block_reduce_sum<PF_BLOCK>(lsum);
    if (threadIdx.x == 0) {
      s_stat[0] = m;
      s_stat[1] = denom;
    }
    __syncthreads();

    // pass 3：按 chunk 重算 p_j 并累加 v。每线程负责 v 的若干维（Dv ≤ 256，PF_BLOCK=256 → 每线程 1 维）。
    float acc[MAX_DV / PF_BLOCK + 1];
#pragma unroll
    for (int c = 0; c < MAX_DV / PF_BLOCK + 1; ++c) acc[c] = 0.f;
    for (int base = 0; base < n; base += PF_BLOCK) {
      const int jj = base + threadIdx.x;
      float p = 0.f;
      int j = -1;
      if (jj < n) {
        j = key_pos(jj);
        if (j >= 0) {
          const bf16* kj = k + ((int64_t)(start + j) * H + h) * Dq;
          float s = 0.f;
          for (int d = 0; d < Dq; ++d) s += s_q[d] * to_f32(kj[d]);
          p = __expf(s * scale - s_stat[0]) / s_stat[1];
        }
      }
      s_p[threadIdx.x] = p;
      s_j[threadIdx.x] = j;
      __syncthreads();
      const int cnt = min(PF_BLOCK, n - base);
      for (int d = threadIdx.x, c = 0; d < Dv; d += PF_BLOCK, ++c) {
        float a = 0.f;
        for (int q2 = 0; q2 < cnt; ++q2) {
          if (s_j[q2] < 0) continue;
          const bf16* vj = v + ((int64_t)(start + s_j[q2]) * H + h) * Dv;
          a += s_p[q2] * to_f32(vj[d]);
        }
        acc[c] += a;
      }
      __syncthreads();
    }
    bf16* oi = out + ((int64_t)(start + i) * H + h) * Dv;
    for (int d = threadIdx.x, c = 0; d < Dv; d += PF_BLOCK, ++c) oi[d] = to_bf16(acc[c]);
    __syncthreads();
  }
}

// ---------- decode (absorbed MQA) ----------
// q: [B, H, E]（E = kv_lora + rope），cache: [num_blocks, block_size, E]，
// block_table: [B, max_blocks] i32，seq_lens: [B] i32，out: [B, H, kv_lora]（latent 空间，之后乘 W_UV）。
// 每 block 一个 (b, h)；线程按 key 并行做 online softmax（单 split，不分 kv_splits）。
constexpr int DC_BLOCK = 256;
constexpr int MAX_E = 640;  // kv_lora 512 + rope 64 = 576

__global__ void decode_kernel(const bf16* __restrict__ q, const bf16* __restrict__ cache,
                              const int32_t* __restrict__ block_table, const int32_t* __restrict__ seq_lens,
                              const int32_t* __restrict__ topk_indices, int topk, bf16* __restrict__ out, int H,
                              int E, int L, int block_size, int max_blocks, float scale) {
  const int b = blockIdx.x, h = blockIdx.y;
  const int seq_len = seq_lens[b];
  const int32_t* sel = topk_indices ? topk_indices + (int64_t)b * topk : nullptr;
  const int n = sel ? topk : seq_len;  // 候选数（gather 模式含 -1 空槽）
  __shared__ float s_q[MAX_E];
  __shared__ float s_p[DC_BLOCK];
  __shared__ int s_j[DC_BLOCK];
  __shared__ float s_stat[2];
  const bf16* qb = q + ((int64_t)b * H + h) * E;
  for (int d = threadIdx.x; d < E; d += DC_BLOCK) s_q[d] = to_f32(qb[d]);
  __syncthreads();
  const int32_t* bt = block_table + (int64_t)b * max_blocks;

  auto key_ptr = [&](int j) -> const bf16* {
    const int64_t blk = bt[j / block_size];
    return cache + (blk * block_size + (j % block_size)) * E;
  };
  auto key_pos = [&](int jj) -> int {
    const int j = sel ? sel[jj] : jj;
    return (j >= 0 && j < seq_len) ? j : -1;
  };

  // pass 1/2：max 与 sum
  float lmax = -INFINITY;
  for (int jj = threadIdx.x; jj < n; jj += DC_BLOCK) {
    const int j = key_pos(jj);
    if (j < 0) continue;
    const bf16* kj = key_ptr(j);
    float s = 0.f;
    for (int d = 0; d < E; ++d) s += s_q[d] * to_f32(kj[d]);
    lmax = fmaxf(lmax, s * scale);
  }
  const float m = block_reduce_max<DC_BLOCK>(lmax);
  float lsum = 0.f;
  for (int jj = threadIdx.x; jj < n; jj += DC_BLOCK) {
    const int j = key_pos(jj);
    if (j < 0) continue;
    const bf16* kj = key_ptr(j);
    float s = 0.f;
    for (int d = 0; d < E; ++d) s += s_q[d] * to_f32(kj[d]);
    lsum += __expf(s * scale - m);
  }
  const float denom = block_reduce_sum<DC_BLOCK>(lsum);
  if (threadIdx.x == 0) {
    s_stat[0] = m;
    s_stat[1] = denom;
  }
  __syncthreads();

  // pass 3：累加 v = latent[:L]
  float acc[MAX_E / DC_BLOCK + 1];
#pragma unroll
  for (int c = 0; c < MAX_E / DC_BLOCK + 1; ++c) acc[c] = 0.f;
  for (int base = 0; base < n; base += DC_BLOCK) {
    const int jj = base + threadIdx.x;
    float p = 0.f;
    int j = -1;
    if (jj < n) {
      j = key_pos(jj);
      if (j >= 0) {
        const bf16* kj = key_ptr(j);
        float s = 0.f;
        for (int d = 0; d < E; ++d) s += s_q[d] * to_f32(kj[d]);
        p = __expf(s * scale - s_stat[0]) / s_stat[1];
      }
    }
    s_p[threadIdx.x] = p;
    s_j[threadIdx.x] = j;
    __syncthreads();
    const int cnt = min(DC_BLOCK, n - base);
    for (int d = threadIdx.x, c = 0; d < L; d += DC_BLOCK, ++c) {
      float a = 0.f;
      for (int q2 = 0; q2 < cnt; ++q2) {
        if (s_j[q2] < 0) continue;
        a += s_p[q2] * to_f32(key_ptr(s_j[q2])[d]);
      }
      acc[c] += a;
    }
    __syncthreads();
  }
  bf16* ob = out + ((int64_t)b * H + h) * L;
  for (int d = threadIdx.x, c = 0; d < L; d += DC_BLOCK, ++c) ob[d] = to_bf16(acc[c]);
}

}  // namespace

extern "C" int mla_cache_write_bf16(const void* latent, void* cache, const int64_t* slot_mapping, int64_t tokens,
                                    int64_t entry, int64_t block_size, cudaStream_t stream) {
  if (tokens <= 0) return 0;
  if (entry <= 0 || block_size <= 0) return err(cudaErrorInvalidValue);
  const int threads = entry >= 256 ? 256 : (int)(((entry + 31) / 32) * 32);
  cache_write_kernel<<<(unsigned)tokens, threads, 0, stream>>>(static_cast<const bf16*>(latent),
                                                               static_cast<bf16*>(cache), slot_mapping, (int)entry,
                                                               (int)block_size);
  return last_err();
}

extern "C" int mla_prefill_bf16(const void* q, const void* k, const void* v, void* out, const int32_t* cu_seqlens,
                                const int32_t* topk_indices, int64_t topk, int64_t num_reqs, int64_t heads,
                                int64_t dq, int64_t dv, float scale, cudaStream_t stream) {
  if (num_reqs <= 0) return 0;
  if (dq <= 0 || dq > MAX_DQ || dv <= 0 || dv > MAX_DV || heads <= 0) return err(cudaErrorInvalidValue);
  if (topk_indices && topk <= 0) return err(cudaErrorInvalidValue);
  dim3 grid((unsigned)num_reqs, (unsigned)heads);
  prefill_kernel<<<grid, PF_BLOCK, 0, stream>>>(static_cast<const bf16*>(q), static_cast<const bf16*>(k),
                                                static_cast<const bf16*>(v), static_cast<bf16*>(out), cu_seqlens,
                                                topk_indices, (int)topk, (int)heads, (int)dq, (int)dv, scale);
  return last_err();
}

// ---------- paged prefill (absorbed MQA, arbitrary query length, history in cache) ----------
// q: [T, H, E]（latent 空间，T = Σ query 长），cache 同 decode；cu_seqlens_q: [B+1]；seq_lens: [B] = 每请求
// 本 step 结束后的总长（历史 + 本步）。query 行 i（0-based 于本请求）对应绝对位置 pos = seq_len - q_len + i，
// 可见 key = [0, pos]。本 step 的 K 已由 mla_cache_write 写入 cache，所以历史与新 token 统一从 cache 读。
// 每 block 一个 (req, head)；逐 query 行三遍扫 key（max / sum / 加权累加），与 decode_kernel 同结构。
__global__ void prefill_paged_kernel(const bf16* __restrict__ q, const bf16* __restrict__ cache,
                                     const int32_t* __restrict__ block_table, const int32_t* __restrict__ seq_lens,
                                     const int32_t* __restrict__ cu_seqlens_q,
                                     const int32_t* __restrict__ topk_indices, int topk, bf16* __restrict__ out,
                                     int H, int E, int L, int block_size, int max_blocks, float scale) {
  const int b = blockIdx.x, h = blockIdx.y;
  const int q0 = cu_seqlens_q[b], q1 = cu_seqlens_q[b + 1];
  const int q_len = q1 - q0;
  const int seq_len = seq_lens[b];
  if (q_len <= 0) return;
  const int hist = seq_len - q_len;
  __shared__ float s_q[MAX_E];
  __shared__ float s_p[DC_BLOCK];
  __shared__ int s_j[DC_BLOCK];
  __shared__ float s_stat[2];
  const int32_t* bt = block_table + (int64_t)b * max_blocks;
  auto key_ptr = [&](int j) -> const bf16* {
    const int64_t blk = bt[j / block_size];
    return cache + (blk * block_size + (j % block_size)) * E;
  };

  for (int i = 0; i < q_len; ++i) {
    const int vis = hist + i + 1;  // causal 可见 key 数
    const int32_t* sel = topk_indices ? topk_indices + (int64_t)(q0 + i) * topk : nullptr;
    const int n = sel ? topk : vis;
    auto key_pos = [&](int jj) -> int {
      const int j = sel ? sel[jj] : jj;
      return (j >= 0 && j < vis) ? j : -1;
    };
    const bf16* qb = q + ((int64_t)(q0 + i) * H + h) * E;
    for (int d = threadIdx.x; d < E; d += DC_BLOCK) s_q[d] = to_f32(qb[d]);
    __syncthreads();
    float lmax = -INFINITY;
    for (int jj = threadIdx.x; jj < n; jj += DC_BLOCK) {
      const int j = key_pos(jj);
      if (j < 0) continue;
      const bf16* kj = key_ptr(j);
      float s = 0.f;
      for (int d = 0; d < E; ++d) s += s_q[d] * to_f32(kj[d]);
      lmax = fmaxf(lmax, s * scale);
    }
    const float m = block_reduce_max<DC_BLOCK>(lmax);
    float lsum = 0.f;
    for (int jj = threadIdx.x; jj < n; jj += DC_BLOCK) {
      const int j = key_pos(jj);
      if (j < 0) continue;
      const bf16* kj = key_ptr(j);
      float s = 0.f;
      for (int d = 0; d < E; ++d) s += s_q[d] * to_f32(kj[d]);
      lsum += __expf(s * scale - m);
    }
    const float denom = block_reduce_sum<DC_BLOCK>(lsum);
    if (threadIdx.x == 0) {
      s_stat[0] = m;
      s_stat[1] = denom;
    }
    __syncthreads();
    float acc[MAX_E / DC_BLOCK + 1];
#pragma unroll
    for (int c = 0; c < MAX_E / DC_BLOCK + 1; ++c) acc[c] = 0.f;
    for (int base = 0; base < n; base += DC_BLOCK) {
      const int jj = base + threadIdx.x;
      float p = 0.f;
      int j = -1;
      if (jj < n) {
        j = key_pos(jj);
        if (j >= 0) {
          const bf16* kj = key_ptr(j);
          float s = 0.f;
          for (int d = 0; d < E; ++d) s += s_q[d] * to_f32(kj[d]);
          p = __expf(s * scale - s_stat[0]) / s_stat[1];
        }
      }
      s_p[threadIdx.x] = p;
      s_j[threadIdx.x] = j;
      __syncthreads();
      const int cnt = min(DC_BLOCK, n - base);
      for (int d = threadIdx.x, c = 0; d < L; d += DC_BLOCK, ++c) {
        float a = 0.f;
        for (int q2 = 0; q2 < cnt; ++q2) {
          if (s_j[q2] < 0) continue;
          a += s_p[q2] * to_f32(key_ptr(s_j[q2])[d]);
        }
        acc[c] += a;
      }
      __syncthreads();
    }
    bf16* ob = out + ((int64_t)(q0 + i) * H + h) * L;
    for (int d = threadIdx.x, c = 0; d < L; d += DC_BLOCK, ++c) ob[d] = to_bf16(acc[c]);
    __syncthreads();
  }
}

extern "C" int mla_prefill_paged_bf16(const void* q, const void* cache, const int32_t* block_table,
                                      const int32_t* seq_lens, const int32_t* cu_seqlens_q,
                                      const int32_t* topk_indices, int64_t topk, void* out, int64_t num_reqs,
                                      int64_t heads, int64_t entry, int64_t kv_lora, int64_t block_size,
                                      int64_t max_blocks, float scale, cudaStream_t stream) {
  if (num_reqs <= 0) return 0;
  if (entry <= 0 || entry > MAX_E || kv_lora <= 0 || kv_lora > entry || heads <= 0 || block_size <= 0)
    return err(cudaErrorInvalidValue);
  if (topk_indices && topk <= 0) return err(cudaErrorInvalidValue);
  dim3 grid((unsigned)num_reqs, (unsigned)heads);
  prefill_paged_kernel<<<grid, DC_BLOCK, 0, stream>>>(static_cast<const bf16*>(q), static_cast<const bf16*>(cache),
                                                      block_table, seq_lens, cu_seqlens_q, topk_indices, (int)topk,
                                                      static_cast<bf16*>(out), (int)heads, (int)entry, (int)kv_lora,
                                                      (int)block_size, (int)max_blocks, scale);
  return last_err();
}

extern "C" int mla_decode_bf16(const void* q, const void* cache, const int32_t* block_table, const int32_t* seq_lens,
                               const int32_t* topk_indices, int64_t topk, void* out, int64_t batch, int64_t heads,
                               int64_t entry, int64_t kv_lora, int64_t block_size, int64_t max_blocks, float scale,
                               cudaStream_t stream) {
  if (batch <= 0) return 0;
  if (entry <= 0 || entry > MAX_E || kv_lora <= 0 || kv_lora > entry || heads <= 0 || block_size <= 0)
    return err(cudaErrorInvalidValue);
  if (topk_indices && topk <= 0) return err(cudaErrorInvalidValue);
  dim3 grid((unsigned)batch, (unsigned)heads);
  decode_kernel<<<grid, DC_BLOCK, 0, stream>>>(static_cast<const bf16*>(q), static_cast<const bf16*>(cache),
                                               block_table, seq_lens, topk_indices, (int)topk,
                                               static_cast<bf16*>(out), (int)heads, (int)entry, (int)kv_lora,
                                               (int)block_size, (int)max_blocks, scale);
  return last_err();
}
