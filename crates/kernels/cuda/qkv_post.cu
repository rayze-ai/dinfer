// Qwen3 / Qwen3.5 attention 的 QKV 后处理，一个 kernel 做完（Q3-e 融合，替代 copy_cols×4 + rmsnorm×2 + rope + cache_write）：
//   输入 qkv: [T, (H·qmul + 2·Hkv)·D]，qmul = 1（无输出门）或 2（Qwen3.5：q 段后紧跟同宽 gate 段）
//     无门：q | k | v；有门：q | gate | k | v（gate 段本 kernel 不碰，由 attention 之后 mul_sigmoid 用）
//   1) q_h ← RMSNorm_D(q_h) · (w_q + beta)；k_h 同（beta=1 即 Qwen3.5 的 zero-centered 权重 1+w）
//   2) q_h, k_h 的前 RD 维 ← neox RoPE（pair (i, i+RD/2)，其余维只 norm 不旋转；RD=D 时即整头旋转）
//   3) k_h, v_h → paged cache [slot][Hkv][2][D]（slot_mapping[t] < 0 跳过）
//   结果 q/k 原地写回 qkv（attention kernel 按 stride (H·qmul+2Hkv)·D 直接读）。
// 线程组织：grid = (T, ceil((H+2Hkv)/8))，block = 8 warp；一个 warp 处理一个头（q 的 norm+rope / k 的 norm+rope+写 cache /
// v 的只写 cache）。lane 持有 d = lane + 32·c（c < D/32）。
// D ∈ {128, 256} 模板实例化；RD ∈ {64, 128}（必须 64 的倍数：一个 lane 的 cos/sin 常数覆盖 32 个维度）。
#include "common.cuh"
#include "kernels.h"

using namespace dinfer;

namespace {

constexpr int QP_WARPS = 8;  // B4 第 1 轮：4 → 8 warp/block，并按 grid.y 把头铺到多个 block（见 kernel 注释）

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

template <int D, int RD>
__global__ void __launch_bounds__(QP_WARPS * 32) qkv_post_kernel(
    bf16* __restrict__ qkv, const bf16* __restrict__ wq, const bf16* __restrict__ wk, const int32_t* __restrict__ pos,
    const float* __restrict__ cos_sin, bf16* __restrict__ cache, const int64_t* __restrict__ slot_mapping, int H, int Hkv,
    int qmul, float eps, float beta, int block_size) {
  constexpr int DPL = D / 32;       // 每 lane 持有的维度数
  constexpr int HALF = RD / 2;      // rope 前半部分维数
  constexpr int NP = HALF / 32;     // 每 lane 的 rope pair 数（partner 在 c + NP）
  static_assert(RD % 64 == 0 && RD <= D, "qkv_post: RD 必须是 64 的倍数且 <= D");
  const int64_t t = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row_w = (H * qmul + 2 * Hkv) * D;
  bf16* row = qkv + t * row_w;
  const float* cs = cos_sin + (int64_t)pos[t] * RD;  // [cos(HALF) | sin(HALF)]
  const int64_t slot = slot_mapping[t];
  bf16* cdst = slot < 0 ? nullptr : cache + slot * Hkv * 2 * D;

  // B4 第 1 轮：原来 4 个 warp 轮流处理 H+Hkv 个头（Qwen3 24 头 → 每 warp 6 轮串行访存，B300 上 6.1 µs，
  // 带宽下限 0.5 µs）。现在每个 warp 只处理一个"工作项"：gh in [0, H+Hkv) 是 q/k 头的 norm+rope，
  // [H+Hkv, H+2Hkv) 是 v 头搬 cache；工作项按 grid.y 铺开，所有访存同时在飞。
  const int gh = blockIdx.y * QP_WARPS + warp;
  if (gh >= H + 2 * Hkv) return;
  if (gh < H + Hkv) {
    const int hh = gh;
    const bool is_q = hh < H;
    bf16* base = is_q ? row + hh * D : row + (H * qmul + (hh - H)) * D;
    const bf16* w = is_q ? wq : wk;
    // lane 持有 D/32 个维度：d = lane + 32·c（D/32 次合并读，一起发出）
    float x[DPL];
    float ss = 0.f;
#pragma unroll
    for (int c = 0; c < DPL; ++c) {
      x[c] = to_f32(base[lane + 32 * c]);
      ss += x[c] * x[c];
    }
    ss = warp_sum(ss);
    const float inv = rsqrtf(ss / D + eps);
#pragma unroll
    for (int c = 0; c < DPL; ++c) x[c] = x[c] * inv * (to_f32(w[lane + 32 * c]) + beta);
    // neox rope：pair (i, i + RD/2)；lane 持 d=lane+32c，所以 pair 落在 (c, c+NP)
    if (NP > 0) {
#pragma unroll
      for (int j = 0; j < NP; ++j) {
        const float c0 = cs[lane + 32 * j], s0 = cs[HALF + lane + 32 * j];
        const float a = x[j], b = x[j + NP];
        x[j] = a * c0 - b * s0;
        x[j + NP] = b * c0 + a * s0;
      }
    }
#pragma unroll
    for (int c = 0; c < DPL; ++c) base[lane + 32 * c] = to_bf16(x[c]);
    if (!is_q && cdst) {
      const int hk = hh - H;
      bf16* kd = cdst + (hk * 2 + 0) * D;
#pragma unroll
      for (int c = 0; c < DPL; ++c) kd[lane + 32 * c] = to_bf16(x[c]);
    }
  } else if (cdst) {
    // v 头：只搬到 cache
    const int hk = gh - H - Hkv;
    const bf16* vs = row + (H * qmul + Hkv + hk) * D;
    bf16* vd = cdst + (hk * 2 + 1) * D;
#pragma unroll
    for (int c = 0; c < DPL; ++c) vd[lane + 32 * c] = vs[lane + 32 * c];
  }
}

template <int D>
int launch_qkv_post(int rd, bf16* qkv, const bf16* wq, const bf16* wk, const int32_t* positions, const float* cos_sin,
                    bf16* cache, const int64_t* slots, int64_t tokens, int heads, int kv_heads, int qmul, float eps,
                    float beta, int block_size, cudaStream_t stream) {
  const unsigned items = (unsigned)(heads + 2 * kv_heads);
  const dim3 grid((unsigned)tokens, (items + QP_WARPS - 1) / QP_WARPS);
  if (rd == D) {
    qkv_post_kernel<D, D><<<grid, QP_WARPS * 32, 0, stream>>>(qkv, wq, wk, positions, cos_sin, cache, slots, heads,
                                                              kv_heads, qmul, eps, beta, block_size);
  } else if (rd == 64) {
    qkv_post_kernel<D, 64><<<grid, QP_WARPS * 32, 0, stream>>>(qkv, wq, wk, positions, cos_sin, cache, slots, heads,
                                                               kv_heads, qmul, eps, beta, block_size);
  } else {
    return err(cudaErrorInvalidValue);
  }
  return last_err();
}

}  // namespace

extern "C" int qkv_post_ext_bf16(void* qkv, const void* wq, const void* wk, const int32_t* positions,
                                 const float* cos_sin, void* cache, const int64_t* slot_mapping, int64_t tokens,
                                 int64_t heads, int64_t kv_heads, int64_t head_dim, int64_t rope_dim, float eps,
                                 float norm_beta, int64_t qmul, int64_t block_size, cudaStream_t stream) {
  if (tokens <= 0) return 0;
  if (heads <= 0 || kv_heads <= 0 || block_size <= 0 || rope_dim <= 0 || rope_dim > head_dim || rope_dim % 64 != 0)
    return err(cudaErrorInvalidValue);
  if (qmul != 1 && qmul != 2) return err(cudaErrorInvalidValue);
  auto* q = static_cast<bf16*>(qkv);
  const auto* a = static_cast<const bf16*>(wq);
  const auto* b = static_cast<const bf16*>(wk);
  auto* c = static_cast<bf16*>(cache);
  const int H = (int)heads, HK = (int)kv_heads, qm = (int)qmul, bs = (int)block_size;
  if (head_dim == 128) return launch_qkv_post<128>((int)rope_dim, q, a, b, positions, cos_sin, c, slot_mapping,
                                                   tokens, H, HK, qm, eps, norm_beta, bs, stream);
  if (head_dim == 256) return launch_qkv_post<256>((int)rope_dim, q, a, b, positions, cos_sin, c, slot_mapping,
                                                   tokens, H, HK, qm, eps, norm_beta, bs, stream);
  return err(cudaErrorInvalidValue);
}

// 旧入口（Qwen3）：整头 RoPE、普通 RMSNorm（beta=0）、无输出门。等价于 qkv_post_ext_bf16(..., rope_dim=D, beta=0, qmul=1)。
extern "C" int qkv_post_bf16(void* qkv, const void* wq, const void* wk, const int32_t* positions, const float* cos_sin,
                             void* cache, const int64_t* slot_mapping, int64_t tokens, int64_t heads, int64_t kv_heads,
                             int64_t head_dim, float eps, int64_t block_size, cudaStream_t stream) {
  return qkv_post_ext_bf16(qkv, wq, wk, positions, cos_sin, cache, slot_mapping, tokens, heads, kv_heads, head_dim,
                           head_dim, eps, 0.f, 1, block_size, stream);
}
