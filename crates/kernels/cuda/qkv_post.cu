// Qwen3 attention 的 QKV 后处理，一个 kernel 做完（Q3-e 融合，替代 copy_cols×4 + rmsnorm×2 + rope + cache_write）：
//   输入 qkv: [T, (H + 2·Hkv)·D] 即 qkv GEMM 的输出（行主，q | k | v 顺序）
//   1) q_h ← RMSNorm_D(q_h) · w_q；k_h ← RMSNorm_D(k_h) · w_k（per-head，归一化维 = D）
//   2) q_h, k_h ← neox RoPE（整个 head_dim，cos_sin[pos] 布局 [rope_dim]：前半 cos 后半 sin）
//   3) k_h, v_h → paged cache（slot_mapping[t] < 0 跳过）
//   结果 q/k 原地写回 qkv（attention kernel 按 stride (H+2Hkv)·D 直接读）。
// 线程组织：grid = T，block = 128（= D；一个 warp 处理一个 head 的 D 维中的 32 列，4 warp 一次处理 4 个 head），
// head 数 H + Hkv 个要 norm+rope，再 Hkv 个 v 只写 cache。D 必须 == 128（Qwen3）；其他 D 由 host 回退到旧多 kernel 路径。
#include "common.cuh"
#include "dinfer_kernels.h"

using namespace dinfer;

namespace {

constexpr int QP_D = 128;
constexpr int QP_WARPS = 8;  // B4 第 1 轮：4 → 8 warp/block，并按 grid.y 把头铺到多个 block（见 kernel 注释）

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__global__ void __launch_bounds__(QP_WARPS * 32) qkv_post_kernel(
    bf16* __restrict__ qkv, const bf16* __restrict__ wq, const bf16* __restrict__ wk, const int32_t* __restrict__ pos,
    const float* __restrict__ cos_sin, bf16* __restrict__ cache, const int64_t* __restrict__ slot_mapping, int H, int Hkv,
    float eps, int block_size) {
  const int64_t t = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row_w = (H + 2 * Hkv) * QP_D;
  bf16* row = qkv + t * row_w;
  const float* cs = cos_sin + (int64_t)pos[t] * QP_D;  // rope_dim == D：cos[0..64) sin[64..128)
  const int64_t slot = slot_mapping[t];
  bf16* cdst = slot < 0 ? nullptr : cache + slot * Hkv * 2 * QP_D;  // slot_off(slot) with D=128

  // B4 第 1 轮：原来 4 个 warp 轮流处理 H+Hkv 个头（Qwen3 24 头 → 每 warp 6 轮串行访存，B300 上 6.1 µs，
  // 带宽下限 0.5 µs）。现在每个 warp 只处理一个"工作项"：gh in [0, H+Hkv) 是 q/k 头的 norm+rope，
  // [H+Hkv, H+2Hkv) 是 v 头搬 cache；工作项按 grid.y 铺开，所有访存同时在飞。
  const int gh = blockIdx.y * QP_WARPS + warp;
  if (gh >= H + 2 * Hkv) return;
  if (gh < H + Hkv) {
    const int hh = gh;
    const bool is_q = hh < H;
    bf16* base = row + hh * QP_D;                  // q 段紧接 k 段，正好连续
    const bf16* w = is_q ? wq : wk;
    // lane 持有 4 个维度：d = lane, lane+32, lane+64, lane+96（4 次 64 B 合并读，一起发出）
    float x[4];
    float ss = 0.f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      x[c] = to_f32(base[lane + 32 * c]);
      ss += x[c] * x[c];
    }
    ss = warp_sum(ss);
    const float inv = rsqrtf(ss / QP_D + eps);
#pragma unroll
    for (int c = 0; c < 4; ++c) x[c] = x[c] * inv * to_f32(w[lane + 32 * c]);
    // neox rope：pair (i, i+64)；lane 持 d=lane (c=0) 与 d=lane+64 (c=2)，以及 d=lane+32 (c=1) 与 d=lane+96 (c=3)
    {
      const float c0 = cs[lane], s0 = cs[64 + lane];
      const float a = x[0], b = x[2];
      x[0] = a * c0 - b * s0;
      x[2] = b * c0 + a * s0;
      const float c1 = cs[lane + 32], s1 = cs[64 + lane + 32];
      const float a1 = x[1], b1 = x[3];
      x[1] = a1 * c1 - b1 * s1;
      x[3] = b1 * c1 + a1 * s1;
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) base[lane + 32 * c] = to_bf16(x[c]);
    if (!is_q && cdst) {
      const int hk = hh - H;
      bf16* kd = cdst + (hk * 2 + 0) * QP_D;
#pragma unroll
      for (int c = 0; c < 4; ++c) kd[lane + 32 * c] = to_bf16(x[c]);
    }
  } else if (cdst) {
    // v 头：只搬到 cache
    const int hk = gh - H - Hkv;
    const bf16* vs = row + (H + Hkv + hk) * QP_D;
    bf16* vd = cdst + (hk * 2 + 1) * QP_D;
#pragma unroll
    for (int c = 0; c < 4; ++c) vd[lane + 32 * c] = vs[lane + 32 * c];
  }
}

}  // namespace

extern "C" int qkv_post_bf16(void* qkv, const void* wq, const void* wk, const int32_t* positions, const float* cos_sin,
                             void* cache, const int64_t* slot_mapping, int64_t tokens, int64_t heads, int64_t kv_heads,
                             int64_t head_dim, float eps, int64_t block_size, cudaStream_t stream) {
  if (tokens <= 0) return 0;
  if (head_dim != QP_D || heads <= 0 || kv_heads <= 0 || block_size <= 0) return err(cudaErrorInvalidValue);
  const unsigned items = (unsigned)(heads + 2 * kv_heads);
  const dim3 grid((unsigned)tokens, (items + QP_WARPS - 1) / QP_WARPS);
  qkv_post_kernel<<<grid, QP_WARPS * 32, 0, stream>>>(
      static_cast<bf16*>(qkv), static_cast<const bf16*>(wq), static_cast<const bf16*>(wk), positions, cos_sin,
      static_cast<bf16*>(cache), slot_mapping, (int)heads, (int)kv_heads, eps, (int)block_size);
  return last_err();
}
