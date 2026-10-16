// GQA（分组多头）paged attention 参考 kernel（sm80+，bf16 in / fp32 累加）。Qwen3 等标准 dense 模型用。
// 结构与 mla.cu 同（每 (req, q_head) 一个 block，逐 query 行三遍扫 key：max / sum / 加权累加），差别：
//   - K/V 分开存：paged cache 布局 [num_blocks, block_size, kv_heads, 2, D]（每 slot 内按 kv_head 排，K 在 V 前）；
//   - q_head h 读 kv_head h / group（group = H / Hkv）；
//   - 无 top-k 稀疏。
//
// 三个入口：
//   gqa_cache_write_bf16   k, v: [T, Hkv, D] → cache（slot_mapping）
//   gqa_prefill_bf16       无历史的 causal prefill，直接读 k/v [T, Hkv, D]（不走 cache；cache 由 cache_write 另写）
//                          D=128 走 flash 风格 tile kernel（Q3-a，2026-10-16），其他 D 走旧参考实现
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
// prefill（无历史，causal）：flash 风格。grid = (ceil(max_seqlen / PF_WARPS), H, num_reqs)，block = PF_WARPS 个 warp，
// 每 warp 负责一个 query 行；K/V 按 PF_TILE 个 key 一片进 shared memory（K 转置存成 [D][TILE] 让 lane 读连续地址），
// 每 lane 算一个 key 的分数，warp 内 online softmax（m / l 随 tile 更新），PV 累加时 lane 持有 D/32 个维度、
// p_j 由 __shfl 广播。只扫到对角 tile（causal），fp32 累加、bf16 I/O。
// 复杂度 O(T²·D/2) 但并行度 = T/PF_WARPS × H × reqs 个 block（旧版 H × reqs 个 block 串行遍历 T 行）。
constexpr int PF_WARPS = 4;
constexpr int PF_TILE = 32;             // 每片 key 数 == warp 宽，一 lane 一 key
constexpr int PF_D = 128;               // Qwen3 head_dim；其他 D 走通用慢路径（见 host 分派）
constexpr int PF_DPL = PF_D / 32;       // 每 lane 持有的维度数

__global__ void __launch_bounds__(PF_WARPS * 32) gqa_prefill_flash_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v, bf16* __restrict__ out,
    const int32_t* __restrict__ cu_seqlens, int H, int Hkv, float scale) {
  const int req = blockIdx.z, h = blockIdx.y;
  const int hk = h / (H / Hkv);
  const int start = cu_seqlens[req], end = cu_seqlens[req + 1];
  const int T = end - start;
  const int q_base = blockIdx.x * PF_WARPS;
  if (q_base >= T) return;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int qi = q_base + warp;             // 本 warp 的 query 行（相对请求）
  const bool active = qi < T;

  __shared__ float sKT[PF_D][PF_TILE];      // 转置：sKT[d][j]
  __shared__ float sV[PF_TILE][PF_D];
  __shared__ float sQ[PF_WARPS][PF_D];

  // q → shared（每 warp 一行）
  if (active) {
    const bf16* qp = q + ((int64_t)(start + qi) * H + h) * PF_D;
    for (int d = lane; d < PF_D; d += 32) sQ[warp][d] = to_f32(qp[d]) * scale;
  }
  float m = -INFINITY, l = 0.f;
  float acc[PF_DPL];
#pragma unroll
  for (int c = 0; c < PF_DPL; ++c) acc[c] = 0.f;

  // 本 block 要扫到的最后一个 key = block 内最大 query 位置
  const int block_last = min(q_base + PF_WARPS - 1, T - 1);
  const int n_tiles = block_last / PF_TILE + 1;
  for (int tile = 0; tile < n_tiles; ++tile) {
    const int k0 = tile * PF_TILE;
    const int kn = min(PF_TILE, T - k0);   // 本片有效 key 数（请求末尾）
    __syncthreads();                       // 上一片用完
    // 装载 K（转置）与 V：PF_TILE × PF_D 个元素，PF_WARPS*32 个线程
    for (int e = threadIdx.x; e < PF_TILE * PF_D; e += PF_WARPS * 32) {
      const int j = e / PF_D, d = e % PF_D;
      float kv_k = 0.f, kv_v = 0.f;
      if (j < kn) {
        const int64_t off = ((int64_t)(start + k0 + j) * Hkv + hk) * PF_D + d;
        kv_k = to_f32(k[off]);
        kv_v = to_f32(v[off]);
      }
      sKT[d][j] = kv_k;
      sV[j][d] = kv_v;
    }
    __syncthreads();
    if (!active) continue;
    // 本 warp 在此片的有效 key 数（causal：key ≤ qi）
    const int valid = min(kn, qi - k0 + 1);
    if (valid <= 0) continue;
    // 1) 分数：lane j 对 key j
    float sc = -INFINITY;
    if (lane < valid) {
      float acc_s = 0.f;
#pragma unroll 8
      for (int d = 0; d < PF_D; ++d) acc_s += sQ[warp][d] * sKT[d][lane];
      sc = acc_s;
    }
    // 2) online softmax
    float tmax = sc;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, o));
    const float m_new = fmaxf(m, tmax);
    const float alpha = __expf(m - m_new);   // m=-inf 时为 0
    const float p = (lane < valid) ? __expf(sc - m_new) : 0.f;
    float psum = p;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) psum += __shfl_xor_sync(0xffffffffu, psum, o);
    l = l * alpha + psum;
#pragma unroll
    for (int c = 0; c < PF_DPL; ++c) acc[c] *= alpha;
    m = m_new;
    // 3) PV：lane 持有维度 d = lane + 32c；p_j 广播
    for (int j = 0; j < valid; ++j) {
      const float pj = __shfl_sync(0xffffffffu, p, j);
#pragma unroll
      for (int c = 0; c < PF_DPL; ++c) acc[c] += pj * sV[j][lane + 32 * c];
    }
  }
  if (!active) return;
  const float inv = 1.f / l;
  bf16* op = out + ((int64_t)(start + qi) * H + h) * PF_D;
#pragma unroll
  for (int c = 0; c < PF_DPL; ++c) op[lane + 32 * c] = to_bf16(acc[c] * inv);
}

// 通用慢路径（任意 D ≤ MAX_D）：旧参考实现，每 (req, q_head) 一个 block 逐 query 行三遍扫 key。对拍 fixture（D=32）走这里。
__global__ void gqa_prefill_ref_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
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
                                int64_t num_reqs, int64_t max_seqlen, int64_t heads, int64_t kv_heads, int64_t head_dim,
                                float scale, cudaStream_t stream) {
  if (num_reqs <= 0) return 0;
  if (heads <= 0 || kv_heads <= 0 || head_dim <= 0 || head_dim > MAX_D || heads % kv_heads != 0 || max_seqlen <= 0)
    return err(cudaErrorInvalidValue);
  if (head_dim == PF_D) {
    dim3 grid((unsigned)((max_seqlen + PF_WARPS - 1) / PF_WARPS), (unsigned)heads, (unsigned)num_reqs);
    gqa_prefill_flash_kernel<<<grid, PF_WARPS * 32, 0, stream>>>(
        static_cast<const bf16*>(q), static_cast<const bf16*>(k), static_cast<const bf16*>(v), static_cast<bf16*>(out),
        cu_seqlens, (int)heads, (int)kv_heads, scale);
  } else {
    dim3 grid((unsigned)num_reqs, (unsigned)heads);
    gqa_prefill_ref_kernel<<<grid, GQ_BLOCK, 0, stream>>>(static_cast<const bf16*>(q), static_cast<const bf16*>(k),
                                                          static_cast<const bf16*>(v), static_cast<bf16*>(out), cu_seqlens,
                                                          (int)heads, (int)kv_heads, (int)head_dim, scale);
  }
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
