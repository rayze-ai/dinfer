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
// layout：线程布局（0 一 block 256 线程一行 / 1 一 warp 一行 / 2 一 block 1024 线程一行），由 `ArchProfile::rmsnorm_layout` 给。
// mode：数值口径，0 = compile（默认；vLLM 未开 enforce_eager 时的实现），1 = eager BI。见 `RMSNORM_MODE`。
int gather_rmsnorm_bf16(const void* x, const int32_t* rows, const void* w, void* out, int64_t num_rows,
                        int64_t hidden, float eps, int layout, int mode, cudaStream_t stream);
int rmsnorm_bf16(const void* x, const void* w, void* out, int64_t tokens, int64_t hidden, float eps, int layout,
                 int mode, cudaStream_t stream);
// residual 融合：res[t,:] += x[t,:]; out[t,:] = rmsnorm(res[t,:])。res 原位更新。
int fused_add_rmsnorm_bf16(const void* x, void* res, const void* w, void* out, int64_t tokens, int64_t hidden,
                           float eps, int layout, int mode, cudaStream_t stream);
// norm + fp8 量化融合（vLLM compile 口径）：out[i] = fp8(norm(x)[i] * (1/a_scale))，h % 8 == 0 且 h <= 8192。
int rmsnorm_quant_fp8_bf16(const void* x, const void* w, void* out, int64_t tokens, int64_t hidden, float eps,
                           float a_scale, cudaStream_t stream);
// router 预处理融合（compile 口径）：out = bf16(norm(x, 无权重) · rs · scale)，只舍入一次。
int rmsnorm_scale2_bf16(const void* x, const void* scale, void* out, int64_t tokens, int64_t hidden, float eps,
                        float rs, int mode, cudaStream_t stream);

// ---- 小消息 one-shot all-reduce（allreduce.cu，对齐 vLLM custom_all_reduce；需 cuda-ipc / P2P）----
// Signal / RankData 的字节数（Rust 侧按此分配 ipc 区头部 / 设备 RankData）。
int64_t ar_signal_bytes(void);
int64_t ar_rank_data_bytes(void);
// out[0..n) = Σ_r mirror_r[0..n)（固定 rank 顺序累加，逐位一致）。rank_data：设备 RankData（ptrs[r] = rank r 镜像区）；
// signals[r]：rank r 的 Signal 设备地址（host 数组，长 world）。n 为 8 的倍数，world ∈ {2,4,8}。
int ar_oneshot_bf16(void* rank_data, const void* const* signals, void* out, int rank, int world, int64_t n,
                    cudaStream_t stream);
// res[t,:] += Σ_r mirror_r[t,:]; normed[t,:] = rmsnorm(res[t,:]) · w（all-reduce + 残差 + RMSNorm 一个 kernel）。
int ar_oneshot_add_rmsnorm_bf16(void* rank_data, const void* const* signals, void* res, const void* w, void* normed,
                                int rank, int world, int64_t tokens, int64_t hidden, float eps, cudaStream_t stream);

// ---- RoPE ----
// q: [tokens, q_heads, head_dim]，k: [tokens, k_heads, head_dim]，只对前 rope_dim 维做旋转。
// cos_sin: [max_pos, rope_dim] fp32，前半 cos 后半 sin（按 rope_dim/2 对）。
// neox=1：成对 (i, i+rope_dim/2)；neox=0（interleave）：成对 (2i, 2i+1)。
// positions: [tokens] i32。k 可为 nullptr（k_heads=0）。
int rope_bf16(void* q, void* k, const int32_t* positions, const float* cos_sin, int64_t tokens, int64_t q_heads,
              int64_t k_heads, int64_t head_dim, int64_t rope_dim, int neox, cudaStream_t stream);
// 同上，neox 配对改为 (i, i+span)，i < rope_dim/2；span = rope_dim/2 即 rope_bf16。
// Gemma4 proportional RoPE（零频补齐到整头）= rope_dim 个非零频旋转、span = head_dim/2。
int rope_span_bf16(void* q, void* k, const int32_t* positions, const float* cos_sin, int64_t tokens, int64_t q_heads,
                   int64_t k_heads, int64_t head_dim, int64_t rope_dim, int64_t span, int neox, cudaStream_t stream);

// ---- 激活 ----
// x: [tokens, 2*inter]，out: [tokens, inter]，out = silu(x[:, :inter]) * x[:, inter:]
int silu_and_mul_bf16(const void* x, void* out, int64_t tokens, int64_t inter, cudaStream_t stream);
// Gemma 系 gelu_tanh（双缓冲）：out = gelu_tanh(gate) * up。
// 逐行乘向量：x[r, j] *= w[j]。
int mul_row_bf16(void* x, const void* w, int64_t rows, int64_t cols, cudaStream_t stream);
// logit softcapping：x = tanh(x/c)·c（bf16 原地）。
int softcap_bf16(void* x, int64_t n, float c, cudaStream_t stream);
// mode：0 = 全 f32 一次舍入；1 = vLLM act_and_mul 口径（gelu 先舍入 bf16 再乘 up）。
int gelu_tanh_mul_bf16(const void* gate, const void* up, void* out, int64_t tokens, int64_t inter, int mode, cudaStream_t stream);
// 输出门（Qwen3.5 attn_output_gate）：x *= sigmoid(gate)，原地逐元素。n 必须是 8 的倍数。
int mul_sigmoid_bf16(void* x, const void* gate, int64_t n, cudaStream_t stream);
// 行式：x 紧凑 [rows, cols]，gate 是 [rows, gate_stride] 的列段（融合 qkv+gate 输出的 gate 段）；cols / gate_stride 是 8 的倍数。
int mul_sigmoid_rows_bf16(void* x, const void* gate, int64_t rows, int64_t cols, int64_t gate_stride, cudaStream_t stream);
// SituAndMul（Kimi-K3 situ 激活）：out = beta·tanh(g/beta)·sigmoid(g) · (linear_beta>0 ? linear_beta·tanh(u/linear_beta) : u)，
// g = x[:, :inter]，u = x[:, inter:]；beta / linear_beta 是模型参数（host 传）。
int situ_and_mul_bf16(const void* x, void* out, int64_t tokens, int64_t inter, float beta, float linear_beta,
                      cudaStream_t stream);

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
int row_gather_i32(const int32_t* prev, const int32_t* gather_idx, int32_t* ids, int64_t n, cudaStream_t stream);
// 对每行做 top-k / top-p 掩码（不满足的置 -inf）。top_k<=0 表示不限；top_p>=1 表示不限。
int topk_topp_mask_f32(float* logits, const int32_t* top_k, const float* top_p, int64_t batch, int64_t vocab,
                       cudaStream_t stream);
// 温度缩放 + softmax + 多项式采样（Philox，seed/offset 每行）。temperature<=0 的行走 argmax。
int sample_f32(const float* logits, const float* temperature, const uint64_t* seeds, const uint64_t* offsets,
               int32_t* out, int64_t batch, int64_t vocab, cudaStream_t stream);
// 受限解码 mask + logit_bias（原地）：mask_row[b]<0 无 mask，否则 mask_bits[mask_row[b]*mask_words..] bit=0 置 -inf；
// bias 为 CSR（offsets 长 batch+1；可全为 NULL = 无 bias）。无约束行零工作量。
int logits_mask_f32(float* logits, const int32_t* mask_row, const uint32_t* mask_bits, int64_t mask_words,
                    const int32_t* bias_offsets, const int32_t* bias_ids, const float* bias_vals, int64_t batch,
                    int64_t vocab, cudaStream_t stream);
int sample_fused_f32(float* work, const float* temperature, const float* min_p, const int32_t* top_k,
                     const float* top_p, const float* temp01, const uint64_t* seeds, const uint64_t* offsets,
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
// head_dim ∈ {128, 256} 走 tensor-core tile kernel；其他 D ≤ 256 走通用慢路径。
// cfg（r11a）：D=256 的 kernel 结构选择 = 版本·1000 + warp 数·100 + tile key 数（版本 0 = v6 Q 进 smem、1 = v7 warp 对分工、
// 2 = v6 + K/V 双缓冲、3 = v6 + SFU ex2.approx；
// NW ∈ {4,8} × TILE ∈ {32,64}），0 = v5 旧路径
// （4 warp、tile 32、Q 在寄存器）。数值由 Rust 侧 ArchProfile 按卡给；D=128 忽略 cfg。
// 进程级：P 以 e4m3 参与 PV（复刻 vLLM FP8 attention 数值），模型加载时设一次。
int gqa_prefill_bf16(const void* q, const void* k, const void* v, void* out, const int32_t* cu_seqlens, int64_t num_reqs,
                     int64_t max_seqlen, int64_t heads, int64_t kv_heads, int64_t head_dim, float scale, int64_t q_stride,
                     int64_t kv_stride, int cfg, cudaStream_t stream);
int gqa_paged_bf16(const void* q, const void* cache, const int32_t* block_table, const int32_t* seq_lens,
                   const int32_t* cu_seqlens_q, void* out, int64_t num_reqs, int64_t max_q_len, int64_t heads,
                   int64_t kv_heads, int64_t head_dim, int64_t block_size, int64_t max_blocks, float scale,
                   int64_t q_stride, int skip_single, int cfg, cudaStream_t stream);
// 滑窗变体（Kaon-V3 / Gemma4 的 25 个 sliding_attention 层）：window <= 0 = 全注意力，与上面的旧符号等价。
int gqa_prefill_win_bf16(const void* q, const void* k, const void* v, void* out, const int32_t* cu_seqlens,
                         int64_t num_reqs, int64_t max_seqlen, int64_t heads, int64_t kv_heads, int64_t head_dim,
                         float scale, int64_t q_stride, int64_t kv_stride, int cfg, int window, int p_fp8, cudaStream_t stream);
int gqa_paged_win_bf16(const void* q, const void* cache, const int32_t* block_table, const int32_t* seq_lens,
                       const int32_t* cu_seqlens_q, void* out, int64_t num_reqs, int64_t max_q_len, int64_t heads,
                       int64_t kv_heads, int64_t head_dim, int64_t block_size, int64_t max_blocks, float scale,
                       int64_t q_stride, int skip_single, int cfg, int window, int p_fp8, cudaStream_t stream);
int gqa_decode_bf16(const void* q, const void* cache, const int32_t* block_table, const int32_t* seq_lens, void* out,
                    float* scratch, int64_t num_reqs, int64_t heads, int64_t kv_heads, int64_t head_dim,
                    int64_t block_size, int64_t max_blocks, float scale, int64_t q_stride, const int32_t* cu_seqlens_q, cudaStream_t stream);
// decode v3（gqa_decode_v3.cu）：同 v2 的 partial 布局；splits==1 时 out_direct 非空直接写输出。GQA_DECODE_V3=1 时由 gqa_decode_bf16 转发。
// head_dim ∈ {128, 256}：D=128 支持 group 1/2/4；D=256 支持 group 1/2/4/6（Qwen3.5 全注意力层）。
int gqa_decode_v3_bf16(const void* q, const void* cache, const int32_t* block_table, const int32_t* seq_lens,
                       const int32_t* cu_seqlens_q, float* scratch, void* out_direct, int64_t num_reqs, int64_t heads,
                       int64_t kv_heads, int64_t head_dim, int64_t block_size, int64_t max_blocks, float scale,
                       int64_t q_stride, int64_t splits, int64_t keys_per_split, int window, cudaStream_t stream);
// 带滑窗的 decode 入口（window>0 只在 v3 / d512 路径合法；D ∈ {128, 256, 512}）。loop kaon3-decode-opt B3。
int gqa_decode_win_bf16(const void* q, const void* cache, const int32_t* block_table, const int32_t* seq_lens, void* out,
                        float* scratch, int64_t num_reqs, int64_t heads, int64_t kv_heads, int64_t head_dim,
                        int64_t block_size, int64_t max_blocks, float scale, int64_t q_stride, const int32_t* cu_seqlens_q,
                        int window, cudaStream_t stream);
// B1（kaon3-mega R36）：split partial → 输出的 merge（gqa_decode_merge_v2_kernel 的 C ABI 入口），persistent attn 对拍用。
int gqa_decode_merge_v2_launch(const float* scratch, const int32_t* cu_seqlens_q, const int32_t* seq_lens, void* out,
                               int num_reqs, int heads, int head_dim, int splits, int keys_per_split,
                               cudaStream_t stream);
// QKV 后处理融合：q/k per-head RMSNorm + neox RoPE（原地）+ K/V 写 paged cache。
// 旧入口（Qwen3）：整头 RoPE、普通 RMSNorm（weight 直接用）、无输出门。D 必须 == 128。
int qkv_post_bf16(void* qkv, const void* wq, const void* wk, const int32_t* positions, const float* cos_sin, void* cache,
                  const int64_t* slot_mapping, int64_t tokens, int64_t heads, int64_t kv_heads, int64_t head_dim, float eps,
                  int64_t block_size, cudaStream_t stream);
// 扩展入口（Qwen3.5 等）：正常化权重要加 norm_beta（zero-centered norm 用 1.0）、只旋前 rope_dim 维（partial RoPE）、
// 可选输出门。行布局 `[q(H·D) | gate(H·D) | k | v]`（qmul=2 时 gate 段本 kernel 不碰，attention 后由 mul_sigmoid 消费），
// qmul=1 时 `[q | k | v]`。head_dim ∈ {128, 256}；rope_dim ∈ {64, 128} 且必须 64 的倍数、<= head_dim。
int qkv_post_ext_bf16(void* qkv, const void* wq, const void* wk, const int32_t* positions, const float* cos_sin,
                      void* cache, const int64_t* slot_mapping, int64_t tokens, int64_t heads, int64_t kv_heads,
                      int64_t head_dim, int64_t rope_dim, float eps, float norm_beta, int64_t qmul, int64_t block_size,
                      cudaStream_t stream);

// ---- MXFP4 解包（mxfp4.cu）：compressed-tensors `mxfp4-pack-quantized` 的参考回退 ----
// packed u8 [rows, cols/2]（低半字节 = 偶数下标）+ scale u8 [rows, cols/group]（e8m0 = 2^(u8-127)）
// → bf16 [rows, cols]。原生 MXFP4 GEMM 留 TODO（docs/loops/k3-moe.md）。
int mxfp4_dequant_bf16(const void* packed, const void* scale, void* out, int64_t rows, int64_t cols,
                       int64_t group);

// ---- MXFP4 grouped GEMM（mxfp4_gemm_sm8x.cu；sm100 路径见 mxfp4_gemm_sm100.cu 空壳）----
// 分组语义同 moe.cu：专家 e 的 token 占 `[expert_off[e], expert_off[e+1])`；xg 由 moe_gather_bf16 收集。
// 权重：[E, N, K/2] u8 + [E, N, K/group] u8 e8m0。out: [tokens, N] bf16（按 xg 的行号写，即排序后的行）。
// tile 三元组已由 host 展平到 tile_e/tile_m/tile_n（长度 = num_tiles）。
int mxfp4_gemm_grouped(const void* xg, const void* wp, const void* ws, const int32_t* expert_off,
                       const int32_t* tile_e, const int32_t* tile_m, const int32_t* tile_n, void* out,
                       int64_t num_tiles, int64_t K, int64_t N, int64_t group, cudaStream_t stream);
// sm100 专用入口；未实现期间返回 cudaErrorNotSupported，host 回落到上面的 sm8x 入口。
int mxfp4_gemm_grouped_sm100(const void* xg, const void* wp, const void* ws, const int32_t* expert_off,
                             const int32_t* tile_e, const int32_t* tile_m, const int32_t* tile_n, void* out,
                             int64_t num_tiles, int64_t K, int64_t N, int64_t group, cudaStream_t stream);

// ---- KDA（kda.cu）：Kimi Delta Attention 的模型无关算子 ----
// 短卷积（depthwise causal conv1d, width<=8, 带 siLU）：
//   y[t,d] = silu(Σ_j w[d,j]·x[t-W+1+j,d])，越界部分取 conv_state（布局 [slots, dim, state_len]）。
//   prefill：x 为 varlen 拼接 [T, dim]，按 cu_seqlens 分段；state_indices/has_initial 可空。
int kda_conv1d_prefill(const void* x, const void* w, void* out, void* conv_state, const int* cu_seqlens,
                       const int* state_indices, const int* has_initial, int64_t num_seqs, int64_t dim,
                       int64_t width, int64_t state_len);
//   update（decode 单 token）：x [batch, dim]；conv_state 原地左移并追加。
int kda_conv1d_update(const void* x, const void* w, void* out, void* conv_state, const int* state_indices,
                      int64_t batch, int64_t dim, int64_t width, int64_t state_len);
// gate：out[t,h,k] = lower_bound·sigmoid(exp(A_log[h])·(g+dt_bias)) 或 -exp(A_log[h])·softplus(g+dt_bias)。
int kda_gate(const void* g, const float* a_log, const float* dt_bias, float* out, int64_t tokens,
             int64_t heads, int64_t head_dim, float lower_bound, int use_lower_bound);
// 递归核心（prefill 与 decode 同一路径）：q/k/v [T,heads,head_dim] bf16（v 用 head_dim 当 V），
// raw_g 同形 bf16，raw_beta [T,heads] f32，state [slots, heads, head_dim(V), head_dim(K)] f32。
// kda 输出 out [T,heads,head_dim] bf16。
int kda_recurrent(const void* q, const void* k, const void* v, const void* raw_g, const float* raw_beta,
                  const float* a_log, const float* dt_bias, void* out, void* state, const int* cu_seqlens,
                  const int* state_indices, int64_t num_seqs, int64_t heads, int64_t head_dim,
                  int64_t state_stride_tok, float scale, float lower_bound, int use_lower_bound);
// 输出门 + gated RMSNorm：y = rmsnorm(x, w, eps) · sigmoid(gate)，逐 (token,head) 一行。
int kda_gated_rmsnorm(const void* x, const void* w, const void* gate, void* out, int64_t rows,
                      int64_t head_dim, float eps);

// ---- KDA chunk 形式 prefill（kda_chunk.cu，规格见 docs/loops/k3-kda.md「v2 规格」）----
// 三段式，host 侧按顺序调用；中间量（qn/kn/beta/gc/aq/u/w/kg）由调用方按下面尺寸分配（全 fp32，bf16 除外）：
//   qn/kn            bf16 [T, heads, head_dim]     （q 已乘 scale，k 已 l2norm）
//   beta             f32  [T, heads]
//   gc               f32  [T, heads, head_dim]     （块内 inclusive cumsum(gate)·1/ln2）
//   chunk_tab        i32  [2, num_chunks]          （chunk → (序列, 块号)）
//   cu_nt            i32  [num_seqs]               （每序列的块数）
//   chunk_off        i32  [num_seqs+1]             （每序列首块在 chunk_tab 的下标）
//   aq               f32  [num_chunks, heads, BT, BT]
//   u/w/kg           f32  [T, heads, head_dim]
// bt 是模板参数，host 选（当前支持 64）。head_dim <= 256；ho 段的 BV 由实现按 head_dim 选。
int kda_chunk_prep(const void* q, const void* k, const void* raw_g, const float* raw_beta, const float* a_log,
                   const float* dt_bias, void* qn, void* kn, float* beta, float* gc, const int* cu_seqlens,
                   int64_t num_seqs, int64_t heads, int64_t head_dim, int64_t bt, float scale, float lower_bound,
                   int use_lower_bound, cudaStream_t stream);
int kda_chunk_wy(const void* qn, const void* kn, const void* v, const float* gc, const float* beta, float* aq,
                 float* u, float* w, float* kg, const int* cu_seqlens, const int* chunk_tab, int64_t num_chunks,
                 int64_t heads, int64_t head_dim, int64_t bt, cudaStream_t stream);
int kda_chunk_ho(const void* qn, const float* gc, const float* aq, const float* u, const float* w,
                 const float* kg, void* out, void* state, const int* cu_seqlens, const int* cu_nt,
                 const int* chunk_off, const int* state_indices, int64_t state_stride_tok, int64_t num_seqs,
                 int64_t heads, int64_t head_dim, int64_t bt, cudaStream_t stream);

// ---- GDN（gdn.cu）：Gated DeltaNet 的模型无关算子（Qwen3.5 线；与 KDA 同族，口径见 gdn.cu 顶部）----
// act：0 = siLU（Qwen3.5），1 = sigmoid（其它 GDN 变体）。
// 短卷积 y[t,d] = act(Σ_j w[d,j]·x[t-W+1+j,d])，w 布局 [dim, width]，宽度维从旧到新；
// conv_state 布局 [slots, dim, state_len]（state_len = width-1），列 0 最旧。
// total = token 总数（cu_seqlens 末值）；rows_per_block = 新版按 token 分块的块行数（0 = 老版每 (seq, 通道) 串行）。
int gdn_conv1d_prefill(const void* x, const void* w, void* out, void* conv_state,
                       const int* cu_seqlens, const int* state_indices, const int* has_initial,
                       int64_t num_seqs, int64_t dim, int64_t width, int64_t state_len, int act,
                       int64_t total, int64_t rows_per_block, int64_t x_stride, cudaStream_t stream);
// x 的行距 x_stride（紧凑 = dim；融合 in_proj 输出的 qkv 列段 = in_proj 行宽），out 紧凑 [T, dim]。
int gdn_conv1d_update(const void* x, const void* w, void* out, void* conv_state,
                      const int* state_indices, int64_t batch, int64_t dim, int64_t width,
                      int64_t state_len, int act, int64_t x_stride, cudaStream_t stream);
// 逐 v 头标量的 gate / beta：g = -exp(A_log[hv])·softplus(a+dt_bias)，beta = sigmoid(b)（f32 输出）。
int gdn_gate(const void* a, const void* b, const float* a_log, const float* dt_bias, float* g_out,
             float* beta_out, int64_t total, int64_t heads, int do_beta, int64_t ab_stride, cudaStream_t stream);
// 递归核心：mixed_qkv [T, 2·H·K + HV·V] bf16（q|k|v 平铺），out [T, HV, V] bf16，
// state [slots, HV, V, K] f32（原地更新）。fused_gate=1 时 g/beta 由 raw a/b 现算，忽略 g/beta 入口。
int gdn_recurrent(const void* mixed_qkv, const float* g, const float* beta, const void* a_raw,
                  const void* b_raw, const float* a_log, const float* dt_bias, void* out,
                  void* state, const int* cu_seqlens, const int* state_indices, int64_t num_seqs,
                  int64_t H, int64_t HV, int64_t K, int64_t V, int64_t group, float scale,
                  int64_t state_stride_tok, int fused_gate, int64_t vsplit, int64_t decode_bv,
                  int decode_variant, int64_t decode_grid, int64_t ab_stride, cudaStream_t stream);
// 输出门 + RMSNorm：y[t,hv,d] = rmsnorm(o[t,hv,:], w, eps) · act(z[t,hv,d])，w 为 [head_dim]。
// z 的行 = (token, v 头)：token 行距 z_stride（紧凑 = heads·head_dim），heads = 每 token 的头数（rows = T·heads）。
int gdn_gated_rmsnorm(const void* x, const void* w, const void* z, void* out, int64_t rows,
                      int64_t head_dim, float eps, int act, int64_t heads, int64_t z_stride, cudaStream_t stream);
// ---- GDN chunk 形式 prefill（gdn_chunk.cu，规格 docs/loops/qwen35-perf.md 第 1 轮）----
// 与 kda_chunk 同三段结构；输入沿用 gdn_recurrent 的 mixed_qkv 与 gdn_gate 的 g/beta。中间量由调用方分配：
//   qn/kn   bf16 [T, H, K]        （q 已乘 scale，k 已 l2norm）
//   gc      f32  [T, HV]          （块内 inclusive cumsum(g)·1/ln2）
//   aq      f32  [num_chunks, HV, BT, BT]
//   u       f32  [T, HV, V]；w/kg  f32 [T, HV, K]
//   chunk_tab / cu_nt / chunk_off 同 kda_chunk。bt 当前只支持 64；K ∈ {64,128,256}；V 是 16 的整数倍。
int gdn_chunk_prep(const void* mixed_qkv, const float* g, void* qn, void* kn, float* gc, const int* cu_seqlens,
                   const int* chunk_tab, int64_t total, int64_t num_chunks, int64_t H, int64_t HV, int64_t K,
                   int64_t V, int64_t bt, float scale, int do_cumsum, cudaStream_t stream);
/// gate + beta + 块内 cumsum 一次算完（prefill；替代 gdn_gate + cumsum 两次 launch）。
int gdn_gate_cumsum(const void* a, const void* b, const float* a_log, const float* dt_bias, float* g_out,
                    float* beta_out, float* gc_out, const int* cu_seqlens, const int* chunk_tab,
                    int64_t num_chunks, int64_t hv, int64_t bt, int do_beta, int64_t ab_stride,
                    cudaStream_t stream);
// r10：aq / u / w / kg 为 bf16（void* 传）。
int gdn_chunk_wy(const void* qn, const void* kn, const void* mixed_qkv, const float* gc, const float* beta,
                 void* aq, void* u, void* w, void* kg, const int* cu_seqlens, const int* chunk_tab,
                 int64_t num_chunks, int64_t H, int64_t HV, int64_t K, int64_t V, int64_t bt, int64_t wy_mode, cudaStream_t stream);
// r11d（docs/loops/qwen35-gdn-r11d.md）：ho 段的 tcgen05（UMMA）版。
// BV = 全 V（不切片），h 常驻全局 state（f32 主本），四个 GEMM 全走 tcgen05，A 操作数 M 补零到 128。
// 只有 sm_103 + K=V=128 时可用；返回 cudaErrorInvalidValue 表示形状不支持，调用方回退 gdn_chunk_ho。
int gdn_ho_umma(const void* qn, const float* gc, const void* aq, const void* u, const void* w, const void* kg,
                void* out, void* state, const int* cu_seqlens, const int* cu_nt, const int* state_indices,
                int64_t state_stride_tok, int64_t num_seqs, int64_t H, int64_t HV, int64_t K, int64_t V,
                cudaStream_t stream);

int gdn_chunk_ho(const void* qn, const float* gc, const void* aq, const void* u, const void* w, const void* kg,
                 void* out, void* state, const int* cu_seqlens, const int* cu_nt, const int* chunk_off,
                 const int* state_indices, int64_t state_stride_tok, int64_t num_seqs, int64_t H, int64_t HV,
                 int64_t K, int64_t V, int64_t bt, int64_t ho_bv, cudaStream_t stream);
// r9（docs/loops/qwen35-gdn-r9.md）：prefetch 0/1 = 寄存器预取下一 chunk；split ≠ 0 = h-pass + o-pass 两段（值 > 1 为 o-pass V 分片 VT ∈ {64,32,16}），
// h16 bf16 [num_chunks, HV, V, K]（每 chunk 入口状态）、v16 bf16 [T, HV, V]（v'）为中间缓冲（调用方分配）。
int gdn_chunk_ho_v2(const void* qn, const float* gc, const void* aq, const void* u, const void* w, const void* kg,
                    void* h16, void* v16, void* out, void* state, const int* cu_seqlens, const int* cu_nt,
                    const int* chunk_off, const int* chunk_tab, const int* state_indices, int64_t state_stride_tok,
                    int64_t num_seqs, int64_t num_chunks, int64_t H, int64_t HV, int64_t K, int64_t V, int64_t bt,
                    int64_t ho_bv, int64_t prefetch, int64_t split, cudaStream_t stream);

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
int kv_pack(const void* const* layer_ptrs, const int32_t* blocks, void* staging, int64_t n_blocks, int64_t num_layers,
            int64_t per_layer_bytes, int64_t block_size, int64_t kv_heads, cudaStream_t stream);
int kv_unpack(void* const* layer_ptrs, const int32_t* blocks, const void* staging, int64_t n_blocks, int64_t num_layers,
              int64_t per_layer_bytes, int64_t block_size, int64_t kv_heads, cudaStream_t stream);
int gather_rows_bf16(const void* src, void* dst, const int32_t* rows, int64_t num_rows, int64_t width,
                     cudaStream_t stream);  // dst[r] = src[rows[r]]
int bf16_to_f32(const void* x, float* y, int64_t n, cudaStream_t stream);
// [nb][rows][cols] bf16 → [rows][nb*cols] f32（all-gather 块布局 → 行布局 + 升精度，一趟）
int gather_blocks_bf16_to_f32(const void* x, float* y, int64_t nb, int64_t rows, int64_t cols, cudaStream_t stream);
int scale_f32(float* x, int64_t n, float a, cudaStream_t stream);
// 逐元素乘标量（bf16 原地）：x *= a。
int scale_bf16(void* x, int64_t n, float a, cudaStream_t stream);  // x[:n] *= a
// x[:n] = bf16(e4m3_satfinite(x / scale) * scale)：FP8（e4m3）KV cache 存取误差的模拟 / 量化口径。
int fp8_e4m3_roundtrip_bf16(void* x, int64_t n, float scale, cudaStream_t stream);

// ---- MoE 参考（moe.cu）----
// logits[T,E] f32 = x[T,H] bf16 · w[E,H] bf16，fp32 累加（与 HF 路由的 fp32 linear 一致）
int moe_gate_f32(const void* x, const void* w, float* logits, int64_t tokens, int64_t num_experts, int64_t hidden,
                 cudaStream_t stream);
// router 的 BI persistent matmul（复刻 vLLM `matmul_persistent`，见 docs/loops/kaon3-compile.md 第 5 轮）：
// out[T,E] f32 = x[T,H] bf16 · w[E,H]^T bf16，K 方向严格顺序累加、末尾舍 bf16 再转 f32。
int router_gemm_bi_f32(const void* x, const void* w, float* out, int64_t tokens, int64_t num_experts, int64_t hidden,
                       cudaStream_t stream);
// logits[T,E] f32 → topk_ids[T,K] i32, topk_w[T,K] f32。语义 = vLLM grouped_topk(n_group=1, sigmoid, bias)。
int moe_route_f32(const float* logits, const float* bias, int32_t* topk_ids, float* topk_w, int64_t tokens,
                  int64_t num_experts, int64_t topk, int renormalize, float scale, cudaStream_t stream);
// 按专家分桶：counts[E]（内部清零）/ offsets[E+1] / cursor[E] / sorted_tok[T*K] / sorted_k[T*K]
int moe_bucket(const int32_t* topk_ids, int32_t* counts, int32_t* offsets, int32_t* cursor, int32_t* sorted_tok,
               int32_t* sorted_k, int32_t* tile_off, int32_t* inv, int64_t tokens, int64_t num_experts, int64_t topk,
               cudaStream_t stream);
// B11：MoE 输出合并一次 launch（替代 memset+scatter_add+memset+f32_add_to_bf16），inv = 路由产出的 (tok,k)→sorted 位置。
int moe_combine_bf16(const void* y, const float* topk_w, const int32_t* inv, const void* per_expert,
                     const int32_t* topk_ids, float w_scale, void* out, int64_t tokens, int64_t topk, int64_t hidden,
                     int mode, cudaStream_t stream);
// B10：decode 小 batch（tokens ≤ 64）路由前处理一次 launch：top-k + 权重 + counts/offsets/cursor/tile_off + sorted + row2e。
int moe_route_bucket_small(const float* logits, const void* per_expert_scale, int32_t* topk_ids, float* topk_w,
                           int32_t* counts, int32_t* offsets, int32_t* cursor, int32_t* sorted_tok, int32_t* sorted_k,
                           int32_t* tile_off, int32_t* row2e, int32_t* inv, int64_t tokens, int64_t num_experts,
                           int64_t topk, cudaStream_t stream);
int moe_gather_bf16(const void* x, void* xg, const int32_t* sorted_tok, int64_t n, int64_t hidden, cudaStream_t stream);
// acc[tok] += topk_w[tok,k] * y[i]（fp32 累加器，调用方清零）
// MoE 输出合并：acc[tok] += w * y[i]（i 为 sorted 下标）。mode 0 = f32 权重/乘积；
// mode 1 = vLLM cutlass_moe 口径（w 先舍 bf16，乘积再舍 bf16）。per_expert 可空（视为全 1）。
int moe_scatter_add_f32(const void* y, float* acc, const int32_t* sorted_tok, const int32_t* sorted_k,
                        const float* topk_w, const void* per_expert, float w_scale,
                        int64_t n, int64_t topk, int64_t hidden, int mode, cudaStream_t stream);
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

// ---- sm_120 低精度 GEMM（gemm_fp8_sm120.cu / gemm_nvfp4_sm120.cu，编译期只在 CUDA_ARCHS 含 120 时存在）----
// out[m,n] = (a[m,k] bf16 · w[n,k] e4m3^T) * w_scale；返回 cudaError_t 数值（0 = 成功）。
// 带静态激活 scale：out = (e4m3(a / a_scale) · w^T) * (a_scale * w_scale)。
int gemm_fp8_static_bf16(const void* a, float a_scale, const void* w, float w_scale, void* out, int64_t m, int64_t n,
                         int64_t k, cudaStream_t stream);
int gemm_fp8_bf16(const void* a, const void* w, float w_scale, void* out, int64_t m, int64_t n, int64_t k,
                  cudaStream_t stream);
// out[m,n] = global * (a[m,k] bf16 现场量化 · packed[n,k/2] fp4^T)，块缩放 scale[n,k/16] e4m3；K % 64 == 0。
// 带激活全局 scale（vLLM a1_gscale）：A 按 gs 网格量化；global = w_global * input_scale。
int gemm_nvfp4_prequant_bf16(const void* aq, const void* a_s, const void* packed, const void* scale, float global,
                             void* out, int64_t m, int64_t n, int64_t k, int swz, cudaStream_t stream);
int gemm_nvfp4_gs_bf16(const void* a, float a_gscale, const void* packed, const void* scale, float global, void* out,
                       int64_t m, int64_t n, int64_t k, cudaStream_t stream);
int gemm_nvfp4_bf16(const void* a, const void* packed, const void* scale, float global, void* out, int64_t m, int64_t n,
                    int64_t k, cudaStream_t stream);

// decode dense MLP 带宽型 fp8 路径（mlp_fp8_decode.cu，M ≤ 16，K % 16 == 0）：
//   norm → fp8：q[m][h] = fp8(bf16(rmsnorm(x)·w) · a_inv)
//   gate‖up GEMV + gelu·mul → fp8：act_q[m][inter] = fp8(bf16(bf16(gelu(bf16(Σ a·g)·gate_scale))·bf16(Σ a·u·up_scale))·dn_a_inv)
//   通用 fp8 GEMV：out[m][n] = bf16(Σ a_q·w · scale)。scale = w_scale·a_scale。
int mlp_fp8_norm_quant(const void* x, const void* w, void* q, int64_t m, int64_t h, float eps, float a_inv,
                       cudaStream_t stream);
int mlp_fp8_gateup_act(const void* a_q, const void* gate_w, const void* up_w, void* act_q, int64_t m, int64_t inter,
                       int64_t k, float gate_scale, float up_scale, float dn_a_inv, cudaStream_t stream);
int mlp_fp8_gemv(const void* a_q, const void* w, void* out, int64_t m, int64_t n, int64_t k, float scale,
                 cudaStream_t stream);
// MoE decode 带宽型 NVFP4 grouped GEMV（moe_fp4_decode.cu，K % 64 == 0）：布局/alpha 与 moe_grouped_gemm 同。
//   gate‖up + gelu·mul → bf16 out[rows, inter]；down → bf16 out[rows, n]。rows = 总行数（选 M_TILE 档位）。
int moe_fp4_gateup_act(const void* aq, const void* asc, const void* const* gate_ptr, const void* const* gate_sptr,
                       const float* alpha_gate, const void* const* up_ptr, const void* const* up_sptr,
                       const float* alpha_up, const int32_t* offsets, void* out, int64_t rows, int64_t inter, int64_t k,
                       int64_t num_experts, cudaStream_t stream);
int moe_fp4_gemv(const void* aq, const void* asc, const void* const* bptr, const void* const* sptr, const float* alpha,
                 const int32_t* offsets, void* out, int64_t rows, int64_t n, int64_t k, int64_t num_experts,
                 cudaStream_t stream);
// 小 M bf16 GEMM（decode）：out[M,N] = a[M,K]·w[N,K]^T (+bias)；splits>1 时需 partial[splits,M,N] f32。见 gemm_small_m.cu。
int gemm_small_m_bf16(const void* a, const void* w, const void* bias, void* out, float* partial, int64_t m,
                      int64_t n, int64_t k, int64_t splits, cudaStream_t stream);
// tensor-core 版（r11b）：K % 128 == 0，warps ∈ {4,8}，stripes（每 warp 的 8 行 W 条数）∈ {1,2,4}；
// splits>1 时 partial[splits,M,N] f32 + counters[ceil(N/8)] int（初值 0，自复位），归约在 kernel 内由最后到达的 warp 做。
int gemm_small_m_mma_bf16(const void* a, const void* w, const void* bias, void* out, float* partial, int* counters,
                          int64_t m, int64_t n, int64_t k, int64_t splits, int64_t warps, int64_t stripes,
                          cudaStream_t stream);

// ---- Qwen3.5 GDN 层 decode 的单 persistent kernel（实验，mega_sm100.cu；只在 CUDA_ARCHS 含 100/103 时编入）。
// 全部指针为 device 指针；形状 / 调优数值全部由 host 给。counters 长度 = 12 + 四个 GEMV 的 tile 数之和（Rust `MegaScratch::new` 算），
// 首次使用前清零一次，之后 kernel 自复位；partial 为 f32，长度 ≥ max_splits · batch · max(N)。
// info（可空）回填 [grid, occupancy, smem_bytes]（host 侧 pinned 或 device 均可，launch 前同步写）。
typedef struct MegaLayerArgs {
  const void* res;           // [B, hidden] bf16，残差，原位更新
  const void* norm_w;        // [hidden]
  const void* post_norm_w;   // [hidden]
  const void* in_w;          // [in_row, hidden]  in_row = conv_dim + HV·V + 2·HV
  const void* conv_w;        // [conv_dim, width]
  const float* a_log;        // [HV]
  const float* dt_bias;      // [HV]
  const void* gnorm_w;       // [V]
  const void* out_w;         // [hidden, HV·V]
  const void* gate_up_w;     // [2·inter, hidden]
  const void* down_w;        // [hidden, inter]
  void* conv_state;          // [slots, conv_dim, width-1] bf16
  void* rec_state;           // [slots, HV, V, K] f32（state_stride_tok = HV·V·K）
  const int32_t* state_indices;  // [B] 或 NULL（恒等）
  void* xn; void* qkvzab; void* conv_out; void* o; void* n; void* h2; void* xn2; void* gu; void* act; void* d;  // 中间量
  float* partial;
  int32_t* counters;
  int32_t* info;
  unsigned long long* prof;  // 可空：[11][4] = {~min_start, max_end, Σwait, Σbusy}（ns），host 清零后读
  int64_t batch, hidden, inter, heads_k, heads_v, k_dim, v_dim, width, state_stride_tok;
  float eps, scale;
  int64_t groups, unroll, stages, ctas_per_sm;              // 结构参数（ArchProfile）
  int64_t splits_in, splits_out, splits_gu, splits_down;    // split-K 份数（host 按 SM 数算）
  int64_t gdn_hv;                                           // 每个 GDN 任务连续处理的 v 头数（host 按波数算）
  int64_t gdn_bufs;                                         // GDN 状态 tile smem 缓冲数（1 / 2，按 smem 预算）
} MegaLayerArgs;
int mega_gdn_layer_decode_sm100(const MegaLayerArgs* a, cudaStream_t stream);

// ---- tcgen05 工具链探针（umma_peak_sm100.cu，r12a；只在 CUDA_ARCHS 含 100/103 时编入）----
// M=128 N=256，K = 64 * k_iters；c: [128, 256] f32。
int umma_peak_launch(const void* a, const void* b, void* c, int64_t k_iters, int64_t blocks, int64_t check,
                     cudaStream_t stream);

// ---- D=256 全注意力 prefill 的 tcgen05 路径（gqa_fa_sm100.cu，r12a；只在 CUDA_ARCHS 含 100/103 时编入）----
// 返回 cudaErrorInvalidValue = 该形状不走这条路（调用方回退 v5 / v6）。
// prof 非空时写 [0]=等 QK [1]=softmax [2]=rescale [3]=PV+装载 [4]=tile 数（clock64 累加，仅 tid 0）。
int gqa_fa_umma_launch(int paged, int max_q, int heads, int num_reqs, const void* q, const void* k, const void* v,
                       const void* cache, const int32_t* bt, const int32_t* seq_lens, const int32_t* cu, void* out,
                       int H, int Hkv, float scale_log2, int q_stride, int kv_stride, int block_size, int max_blocks,
                       int skip_single, void* prof, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
