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
int argmax_f32(const float* logits, int32_t* out, int64_t batch, int64_t vocab, cudaStream_t stream);
// 对每行做 top-k / top-p 掩码（不满足的置 -inf）。top_k<=0 表示不限；top_p>=1 表示不限。
int topk_topp_mask_f32(float* logits, const int32_t* top_k, const float* top_p, int64_t batch, int64_t vocab,
                       cudaStream_t stream);
// 温度缩放 + softmax + 多项式采样（Philox，seed/offset 每行）。temperature<=0 的行走 argmax。
int sample_f32(const float* logits, const float* temperature, const uint64_t* seeds, const uint64_t* offsets,
               int32_t* out, int64_t batch, int64_t vocab, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
