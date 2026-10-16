// kernels 的 C ABI。所有入口：裸指针 + 维度 + cudaStream_t，返回 cudaError_t 数值（0 = OK）。
// bf16 参数一律用 `const void*` / `void*` 传（避免 Rust 侧依赖 cuda_bf16.h 的类型）。
// 命名：不带项目前缀（项目可能改名）；符号只通过静态库链接进 Rust，不对外导出。
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---- P0 自检 ----
int add_f32(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream);

// ---- 归一化 ----
// out[t,:] = x[t,:] * rsqrt(mean(x^2) + eps) * w   （bf16 in/out，fp32 累加）
int rmsnorm_bf16(const void* x, const void* w, void* out, int64_t tokens, int64_t hidden, float eps,
                 cudaStream_t stream);
// residual 融合：res[t,:] += x[t,:]; out[t,:] = rmsnorm(res[t,:])。res 原位更新。
int fused_add_rmsnorm_bf16(const void* x, void* res, const void* w, void* out, int64_t tokens, int64_t hidden,
                           float eps, cudaStream_t stream);

// ---- RoPE ----
// q: [tokens, q_heads, head_dim]，k: [tokens, k_heads, head_dim]，只对前 rope_dim 维做旋转。
// cos_sin: [max_pos, rope_dim] fp32，前半 cos 后半 sin（按 rope_dim/2 对）。
// neox=1：成对 (i, i+rope_dim/2)；neox=0（interleave）：成对 (2i, 2i+1)。
// positions: [tokens] i32。k 可为 nullptr（k_heads=0）。
int rope_bf16(void* q, void* k, const int32_t* positions, const float* cos_sin, int64_t tokens, int64_t q_heads,
              int64_t k_heads, int64_t head_dim, int64_t rope_dim, int neox, cudaStream_t stream);

// ---- 激活 ----
// x: [tokens, 2*inter]，out: [tokens, inter]，out = silu(x[:, :inter]) * x[:, inter:]
int silu_and_mul_bf16(const void* x, void* out, int64_t tokens, int64_t inter, cudaStream_t stream);

// ---- Embedding ----
int embedding_bf16(const int32_t* ids, const void* table, void* out, int64_t tokens, int64_t hidden, int64_t vocab,
                   cudaStream_t stream);

// ---- 采样 ----
// logits: [batch, vocab] fp32（就地修改允许）；out: [batch] i32。
// ---- 采样（sampling.cu）：vLLM 顺序 penalties → temperature → min_p → top_k/top_p → sample；logprobs 用原始 logits ----
int penalties_f32(float* logits, const int32_t* ids, const int32_t* offsets, const int32_t* prompt_len, const float* rep,
                  const float* freq, const float* pres, int64_t batch, int64_t vocab, cudaStream_t stream);
int min_p_mask_f32(float* logits, const float* min_p, int64_t batch, int64_t vocab, cudaStream_t stream);
int temperature_f32(float* logits, const float* temperature, int64_t batch, int64_t vocab, cudaStream_t stream);
int logprobs_f32(const float* logits, const int32_t* chosen, int64_t k, int32_t* out_ids, float* out_lp, int32_t* rank,
                 int64_t batch, int64_t vocab, cudaStream_t stream);
int argmax_f32(const float* logits, int32_t* out, int64_t batch, int64_t vocab, cudaStream_t stream);
// 对每行做 top-k / top-p 掩码（不满足的置 -inf）。top_k<=0 表示不限；top_p>=1 表示不限。
int topk_topp_mask_f32(float* logits, const int32_t* top_k, const float* top_p, int64_t batch, int64_t vocab,
                       cudaStream_t stream);
// 温度缩放 + softmax + 多项式采样（Philox，seed/offset 每行）。temperature<=0 的行走 argmax。
int sample_f32(const float* logits, const float* temperature, const uint64_t* seeds, const uint64_t* offsets,
               int32_t* out, int64_t batch, int64_t vocab, cudaStream_t stream);


// ---- MLA（参考版，见 mla.cu）----
// latent: [tokens, entry] bf16（entry = kv_lora + rope）→ paged cache [num_blocks, block_size, entry]；slot<0 跳过。
int mla_cache_write_bf16(const void* latent, void* cache, const int64_t* slot_mapping, int64_t tokens, int64_t entry,
                         int64_t block_size, cudaStream_t stream);
// causal attention（展开 K/V）：q/k: [T, heads, dq]，v: [T, heads, dv]，out: [T, heads, dv]，cu_seqlens: [num_reqs+1]。
// topk_indices（可为 NULL = 稠密 causal）：[T, topk] i32，DSA 稀疏时每 query 行只看列出的 key（请求内位置，-1 空槽）。
int mla_prefill_bf16(const void* q, const void* k, const void* v, void* out, const int32_t* cu_seqlens,
                     const int32_t* topk_indices, int64_t topk, int64_t num_reqs, int64_t heads, int64_t dq,
                     int64_t dv, float scale, cudaStream_t stream);
// 吸收后 MQA、任意 query 长、历史在 cache 中（chunked prefill / 混合 batch）：q: [T, heads, entry]，
// cu_seqlens_q: [num_reqs+1]，seq_lens: [num_reqs] = 历史 + 本步；out: [T, heads, kv_lora]。
int mla_prefill_paged_bf16(const void* q, const void* cache, const int32_t* block_table, const int32_t* seq_lens,
                           const int32_t* cu_seqlens_q, const int32_t* topk_indices, int64_t topk, void* out,
                           int64_t num_reqs, int64_t heads, int64_t entry, int64_t kv_lora, int64_t block_size,
                           int64_t max_blocks, float scale, cudaStream_t stream);
// 吸收后 MQA decode：q: [batch, heads, entry]，out: [batch, heads, kv_lora]（latent 空间）。
int mla_decode_bf16(const void* q, const void* cache, const int32_t* block_table, const int32_t* seq_lens,
                    const int32_t* topk_indices, int64_t topk, void* out, int64_t batch, int64_t heads, int64_t entry,
                    int64_t kv_lora, int64_t block_size, int64_t max_blocks, float scale, cudaStream_t stream);


// ---- GQA paged attention（gqa.cu）：cache [num_blocks, block_size, kv_heads, 2, D]（K 在 V 前）----
int gqa_cache_write_bf16(const void* k, const void* v, void* cache, const int64_t* slot_mapping, int64_t tokens,
                         int64_t kv_heads, int64_t head_dim, int64_t block_size, cudaStream_t stream);
// q_stride / kv_stride：每 token 行的元素跨度（0 = 紧凑 heads·D / kv_heads·D）；允许直接读 qkv GEMM 输出。
int gqa_prefill_bf16(const void* q, const void* k, const void* v, void* out, const int32_t* cu_seqlens, int64_t num_reqs,
                     int64_t max_seqlen, int64_t heads, int64_t kv_heads, int64_t head_dim, float scale, int64_t q_stride,
                     int64_t kv_stride, cudaStream_t stream);
int gqa_paged_bf16(const void* q, const void* cache, const int32_t* block_table, const int32_t* seq_lens,
                   const int32_t* cu_seqlens_q, void* out, int64_t num_reqs, int64_t max_q_len, int64_t heads,
                   int64_t kv_heads, int64_t head_dim, int64_t block_size, int64_t max_blocks, float scale,
                   int64_t q_stride, cudaStream_t stream);
int gqa_decode_bf16(const void* q, const void* cache, const int32_t* block_table, const int32_t* seq_lens, void* out,
                    float* scratch, int64_t num_reqs, int64_t heads, int64_t kv_heads, int64_t head_dim,
                    int64_t block_size, int64_t max_blocks, float scale, int64_t q_stride, cudaStream_t stream);
// Qwen3 QKV 后处理融合：q/k per-head RMSNorm + neox RoPE（原地）+ K/V 写 paged cache。D 必须 == 128。
int qkv_post_bf16(void* qkv, const void* wq, const void* wk, const int32_t* positions, const float* cos_sin, void* cache,
                  const int64_t* slot_mapping, int64_t tokens, int64_t heads, int64_t kv_heads, int64_t head_dim, float eps,
                  int64_t block_size, cudaStream_t stream);

// ---- 布局搬运（layout.cu）----
int copy_cols_bf16(const void* src, void* dst, int64_t tokens, int64_t src_cols, int64_t col0, int64_t dst_cols,
                   int64_t dcol0, int64_t width, cudaStream_t stream);
int bcast_kpe_bf16(const void* kpe, void* k, int64_t tokens, int64_t heads, int64_t qk, int64_t nope, int64_t rope,
                   cudaStream_t stream);
int copy_head_cols_bf16(const void* src, void* dst, int64_t tokens, int64_t heads, int64_t in_w, int64_t in_col0,
                        int64_t out_w, int64_t out_col0, int64_t width, cudaStream_t stream);
int gather_head_bf16(const void* src, void* dst, int64_t tokens, int64_t heads, int64_t w, int64_t head, int64_t col0,
                     int64_t width, cudaStream_t stream);
int scatter_head_bf16(const void* src, void* dst, int64_t tokens, int64_t heads, int64_t w, int64_t head, int64_t col0,
                      int64_t width, cudaStream_t stream);


// ---- 逐元素（elementwise.cu）----
// x[t,:] = 0 where pos[t] == 0（MTP）
int zero_rows_where_pos0_bf16(void* x, const int32_t* pos, int64_t tokens, int64_t cols, cudaStream_t stream);
int add_inplace_bf16(const void* x, void* y, int64_t n, cudaStream_t stream);  // y += x
int gather_rows_bf16(const void* src, void* dst, const int32_t* rows, int64_t num_rows, int64_t width,
                     cudaStream_t stream);  // dst[r] = src[rows[r]]
int bf16_to_f32(const void* x, float* y, int64_t n, cudaStream_t stream);
int scale_f32(float* x, int64_t n, float a, cudaStream_t stream);  // x[:n] *= a

// ---- MoE 参考（moe.cu）----
// logits[T,E] f32 = x[T,H] bf16 · w[E,H] bf16，fp32 累加（与 HF 路由的 fp32 linear 一致）
int moe_gate_f32(const void* x, const void* w, float* logits, int64_t tokens, int64_t num_experts, int64_t hidden,
                 cudaStream_t stream);
// logits[T,E] f32 → topk_ids[T,K] i32, topk_w[T,K] f32。语义 = vLLM grouped_topk(n_group=1, sigmoid, bias)。
int moe_route_f32(const float* logits, const float* bias, int32_t* topk_ids, float* topk_w, int64_t tokens,
                  int64_t num_experts, int64_t topk, int renormalize, float scale, cudaStream_t stream);
// 按专家分桶：counts[E]（内部清零）/ offsets[E+1] / cursor[E] / sorted_tok[T*K] / sorted_k[T*K]
int moe_bucket(const int32_t* topk_ids, int32_t* counts, int32_t* offsets, int32_t* cursor, int32_t* sorted_tok,
               int32_t* sorted_k, int64_t tokens, int64_t num_experts, int64_t topk, cudaStream_t stream);
int moe_gather_bf16(const void* x, void* xg, const int32_t* sorted_tok, int64_t n, int64_t hidden, cudaStream_t stream);
// acc[tok] += topk_w[tok,k] * y[i]（fp32 累加器，调用方清零）
int moe_scatter_add_f32(const void* y, float* acc, const int32_t* sorted_tok, const int32_t* sorted_k,
                        const float* topk_w, int64_t n, int64_t topk, int64_t hidden, cudaStream_t stream);
// out[:n] = bf16(bf16(acc) + out)
int f32_add_to_bf16(const float* acc, void* out, int64_t n, cudaStream_t stream);

// ---- DSA 索引器参考（dsa.cu）----
// LayerNorm（带 bias，fp32 计算，bf16 出）：out = (x - mean) / sqrt(var + eps) * w + b
int layernorm_bf16(const void* x, const void* w, const void* b, void* out, int64_t tokens, int64_t hidden, float eps,
                   cudaStream_t stream);
// 索引分数：score[row, j] = Σ_h w[row,h] · relu(q[row,h,:]·k_cache[j,:] · scale)，j ∈ [0, vis(row))，其余 -inf。
//   q: [T, H, D] bf16；w: [T, H] f32（已含 H^-0.5）；k_cache: paged [num_blocks, block_size, D] bf16；
//   seq_lens[b] = 请求本步结束后总长；cu_seqlens_q 划分 row → 请求；row i 的 vis = seq_len - q_len + i + 1；
//   score: [T, max_seq] f32（行步长 max_seq）。
int dsa_index_score_f32(const void* q, const float* w, const void* k_cache, const int32_t* block_table,
                        const int32_t* seq_lens, const int32_t* cu_seqlens_q, float* score, int64_t num_reqs,
                        int64_t heads, int64_t head_dim, int64_t block_size, int64_t max_blocks, int64_t max_seq,
                        float scale, cudaStream_t stream);
// 每行 top-k 下标（相等取小下标），不足 k 填 -1。vis[row] = 有效列数；out: [rows, topk] i32。
int dsa_topk_i32(const float* score, const int32_t* vis, int32_t* out, int64_t rows, int64_t max_seq, int64_t topk,
                 cudaStream_t stream);
// vis[row] = seq_lens[b] - q_len[b] + i + 1
int dsa_row_vis_i32(const int32_t* seq_lens, const int32_t* cu_seqlens_q, int32_t* vis, int64_t num_reqs,
                    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
