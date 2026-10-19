// DSA（DeepSeek Sparse Attention）索引器参考 kernel（sm80+，不追求性能）。
//
//   layernorm_bf16        索引器 k_norm：LayerNorm（带 bias，eps 1e-6），fp32 计算，bf16 出
//   dsa_row_vis_i32       每个 query 行的 causal 可见 key 数
//   dsa_index_score_f32   score[row, j] = Σ_h w[row,h] · relu(q[row,h]·k[j] · scale)，k 从 paged 索引 cache 读
//   dsa_topk_i32          每行对 score[:vis] 取 top-k 下标（相等取小下标），不足填 -1
//
// 对照 HF `GlmMoeDsaIndexer.forward`（transformers 5.19，modeling_glm_moe_dsa.py:184-242）：
//   scores = relu(q.float() @ k.float()^T * softmax_scale)；index_scores = weights @ scores；causal mask；topk。
// 真机版（fp8 索引 cache、DeepGEMM fp8_mqa_logits、radix top-k）放 P10。
#include "common.cuh"
#include "kernels.h"

using namespace dinfer;

namespace {

constexpr int LN_BLOCK = 256;

__global__ void layernorm_kernel(const bf16* __restrict__ x, const bf16* __restrict__ w, const bf16* __restrict__ b,
                                 bf16* __restrict__ out, int hidden, float eps) {
  const int64_t row = blockIdx.x;
  const bf16* xr = x + row * hidden;
  bf16* orow = out + row * hidden;
  float sum = 0.f;
  for (int i = threadIdx.x; i < hidden; i += LN_BLOCK) sum += to_f32(xr[i]);
  const float mean = block_reduce_sum<LN_BLOCK>(sum) / hidden;
  float sq = 0.f;
  for (int i = threadIdx.x; i < hidden; i += LN_BLOCK) {
    const float d = to_f32(xr[i]) - mean;
    sq += d * d;
  }
  const float var = block_reduce_sum<LN_BLOCK>(sq) / hidden;
  const float inv = rsqrtf(var + eps);
  for (int i = threadIdx.x; i < hidden; i += LN_BLOCK)
    orow[i] = to_bf16((to_f32(xr[i]) - mean) * inv * to_f32(w[i]) + to_f32(b[i]));
}

__global__ void row_vis_kernel(const int32_t* __restrict__ seq_lens, const int32_t* __restrict__ cu_seqlens_q,
                               int32_t* __restrict__ vis) {
  const int b = blockIdx.x;
  const int q0 = cu_seqlens_q[b], q1 = cu_seqlens_q[b + 1];
  const int q_len = q1 - q0;
  const int hist = seq_lens[b] - q_len;
  for (int i = threadIdx.x; i < q_len; i += blockDim.x) vis[q0 + i] = hist + i + 1;
}

constexpr int SC_BLOCK = 256;
constexpr int MAX_IDX_H = 64;   // GLM 32
constexpr int MAX_IDX_D = 256;  // GLM 128

// 一个 block 处理一个 (req, query 行)：线程按 key 并行。head 逐个处理（q 的一个 head 放 shared，静态 shared
// 必须 ≤ 48 KB：H·D fp32 = 32·128·4 = 16 KB 对 GLM 可行，但上限 64·256 会超，故按 head 循环，shared 只放一个 head）。
// 每个线程为自己负责的 key 维护 fp32 累加器（寄存器）；key 多于 SC_BLOCK 时分轮。
__global__ void index_score_kernel(const bf16* __restrict__ q, const float* __restrict__ w,
                                   const bf16* __restrict__ k_cache, const int32_t* __restrict__ block_table,
                                   const int32_t* __restrict__ seq_lens, const int32_t* __restrict__ cu_seqlens_q,
                                   float* __restrict__ score, int H, int D, int block_size, int max_blocks,
                                   int max_seq, float scale) {
  const int b = blockIdx.x, i = blockIdx.y;
  const int q0 = cu_seqlens_q[b], q1 = cu_seqlens_q[b + 1];
  const int q_len = q1 - q0;
  if (i >= q_len) return;
  const int row = q0 + i;
  const int vis = seq_lens[b] - q_len + i + 1;
  __shared__ float s_q[MAX_IDX_D];
  const int32_t* bt = block_table + (int64_t)b * max_blocks;
  float* srow = score + (int64_t)row * max_seq;
  // 先清零（含 -inf 填充）
  for (int j = threadIdx.x; j < max_seq; j += SC_BLOCK) srow[j] = j < vis ? 0.f : -INFINITY;
  __syncthreads();
  for (int h = 0; h < H; ++h) {
    for (int d = threadIdx.x; d < D; d += SC_BLOCK) s_q[d] = to_f32(q[((int64_t)row * H + h) * D + d]);
    __syncthreads();
    const float wh = w[(int64_t)row * H + h];
    for (int j = threadIdx.x; j < vis; j += SC_BLOCK) {
      const int64_t blk = bt[j / block_size];
      const bf16* kj = k_cache + (blk * block_size + (j % block_size)) * D;
      float sdot = 0.f;
      for (int d = 0; d < D; ++d) sdot += s_q[d] * to_f32(kj[d]);
      sdot *= scale;
      srow[j] += wh * (sdot > 0.f ? sdot : 0.f);
    }
    __syncthreads();
  }
}

// 每行一个 block：K 轮 argmax（相等取小下标），已选置 -inf 后再找。参考版，K ≤ 2048、vis ≤ 几千可接受。
constexpr int TK_BLOCK = 256;
__global__ void topk_kernel(const float* __restrict__ score, const int32_t* __restrict__ vis, int32_t* __restrict__ out,
                            int max_seq, int topk) {
  const int row = blockIdx.x;
  const int n = min(vis[row], max_seq);
  const float* srow = score + (int64_t)row * max_seq;
  int32_t* orow = out + (int64_t)row * topk;
  extern __shared__ float s_val[];  // [max_seq]（动态 shared）
  for (int j = threadIdx.x; j < max_seq; j += TK_BLOCK) s_val[j] = j < n ? srow[j] : -INFINITY;
  __syncthreads();
  __shared__ float s_bv[TK_BLOCK / 32];
  __shared__ int s_bi[TK_BLOCK / 32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int k = 0; k < topk; ++k) {
    if (k >= n) {
      if (threadIdx.x == 0) orow[k] = -1;
      continue;
    }
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int j = threadIdx.x; j < n; j += TK_BLOCK) {
      const float v = s_val[j];
      if (v > bv || (v == bv && j < bi)) {
        bv = v;
        bi = j;
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) {
        bv = ov;
        bi = oi;
      }
    }
    if (lane == 0) {
      s_bv[warp] = bv;
      s_bi[warp] = bi;
    }
    __syncthreads();
    if (warp == 0) {
      bv = lane < TK_BLOCK / 32 ? s_bv[lane] : -INFINITY;
      bi = lane < TK_BLOCK / 32 ? s_bi[lane] : 0x7fffffff;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) {
          bv = ov;
          bi = oi;
        }
      }
      if (lane == 0) {
        orow[k] = bi;
        s_val[bi] = -INFINITY;
      }
    }
    __syncthreads();
  }
}

}  // namespace

extern "C" int layernorm_bf16(const void* x, const void* w, const void* b, void* out, int64_t tokens, int64_t hidden,
                              float eps, cudaStream_t stream) {
  if (tokens <= 0) return 0;
  if (hidden <= 0) return err(cudaErrorInvalidValue);
  layernorm_kernel<<<(unsigned)tokens, LN_BLOCK, 0, stream>>>(static_cast<const bf16*>(x), static_cast<const bf16*>(w),
                                                              static_cast<const bf16*>(b), static_cast<bf16*>(out),
                                                              (int)hidden, eps);
  return last_err();
}

extern "C" int dsa_row_vis_i32(const int32_t* seq_lens, const int32_t* cu_seqlens_q, int32_t* vis, int64_t num_reqs,
                               cudaStream_t stream) {
  if (num_reqs <= 0) return 0;
  row_vis_kernel<<<(unsigned)num_reqs, 128, 0, stream>>>(seq_lens, cu_seqlens_q, vis);
  return last_err();
}

// max_q_len：grid.y 用；调用方传本步任一请求的最大 query 长（Rust 侧用 total_q 上界即可）。
extern "C" int dsa_index_score_f32(const void* q, const float* w, const void* k_cache, const int32_t* block_table,
                                   const int32_t* seq_lens, const int32_t* cu_seqlens_q, float* score,
                                   int64_t num_reqs, int64_t heads, int64_t head_dim, int64_t block_size,
                                   int64_t max_blocks, int64_t max_seq, float scale, cudaStream_t stream) {
  if (num_reqs <= 0) return 0;
  if (heads <= 0 || heads > MAX_IDX_H || head_dim <= 0 || head_dim > MAX_IDX_D || block_size <= 0 || max_seq <= 0)
    return err(cudaErrorInvalidValue);
  // grid.y = max_seq 是 q_len 的上界（q_len ≤ seq_len ≤ max_seq）；多余的 block 立即返回。
  dim3 grid((unsigned)num_reqs, (unsigned)max_seq);
  index_score_kernel<<<grid, SC_BLOCK, 0, stream>>>(static_cast<const bf16*>(q), w, static_cast<const bf16*>(k_cache),
                                                    block_table, seq_lens, cu_seqlens_q, score, (int)heads,
                                                    (int)head_dim, (int)block_size, (int)max_blocks, (int)max_seq,
                                                    scale);
  return last_err();
}

extern "C" int dsa_topk_i32(const float* score, const int32_t* vis, int32_t* out, int64_t rows, int64_t max_seq,
                            int64_t topk, cudaStream_t stream) {
  if (rows <= 0) return 0;
  if (max_seq <= 0 || topk <= 0) return err(cudaErrorInvalidValue);
  const size_t smem = (size_t)max_seq * sizeof(float);
  if (smem > 48 * 1024) {
    if (cudaFuncSetAttribute(topk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) != cudaSuccess)
      return last_err();
  }
  topk_kernel<<<(unsigned)rows, TK_BLOCK, smem, stream>>>(score, vis, out, (int)max_seq, (int)topk);
  return last_err();
}
