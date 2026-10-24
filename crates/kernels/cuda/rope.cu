#include "common.cuh"
#include "kernels.h"

using namespace dinfer;

namespace {

// 一个线程处理一个 (token, head, pair)。
template <bool NEOX>
__global__ void rope_kernel(bf16* __restrict__ q, bf16* __restrict__ k, const int32_t* __restrict__ pos,
                            const float* __restrict__ cos_sin, int q_heads, int k_heads, int head_dim,
                            int rope_dim, int span) {
  const int64_t token = blockIdx.x;
  const int half = rope_dim / 2;
  const int total_heads = q_heads + k_heads;
  const float* cs = cos_sin + (int64_t)pos[token] * rope_dim;

  for (int idx = threadIdx.x; idx < total_heads * half; idx += blockDim.x) {
    const int h = idx / half;
    const int i = idx - h * half;
    bf16* base;
    if (h < q_heads) {
      base = q + (token * q_heads + h) * head_dim;
    } else {
      base = k + (token * k_heads + (h - q_heads)) * head_dim;
    }
    const float c = cs[i], s = cs[half + i];
    int a, b;
    if (NEOX) {
      a = i;
      b = i + span;  // 常规 = half；Gemma4 proportional = head_dim/2（零频补齐的整头旋转）
    } else {
      a = 2 * i;
      b = 2 * i + 1;
    }
    // vLLM `ApplyRotaryEmb`（`enable_fp32_compute=False`）：cos/sin 与 x 都是 bf16，torch 的 bf16
    // 二元 op 逐个 fp32 算再舍 bf16。逐位对齐就要复现这个"每步舍 bf16"，不能全程 fp32 只末尾舍一次
    // （Kaon-V3 loop kaon3-layer1 第 10 轮实测：全程 fp32 与参照差 15843/90112，max|d|=0.0625）。
    const float x0 = to_f32(base[a]), x1 = to_f32(base[b]);
    const float p0 = to_f32(to_bf16(x0 * c)), p1 = to_f32(to_bf16(x1 * s));
    const float p2 = to_f32(to_bf16(x1 * c)), p3 = to_f32(to_bf16(x0 * s));
    base[a] = to_bf16(p0 - p1);
    base[b] = to_bf16(p2 + p3);
  }
}

}  // namespace

extern "C" int rope_span_bf16(void* q, void* k, const int32_t* positions, const float* cos_sin, int64_t tokens,
                              int64_t q_heads, int64_t k_heads, int64_t head_dim, int64_t rope_dim, int64_t span,
                              int neox, cudaStream_t stream) {
  if (tokens <= 0) return 0;
  if (rope_dim <= 0 || (rope_dim & 1) || rope_dim > head_dim) return err(cudaErrorInvalidValue);
  if (neox && (span < rope_dim / 2 || span + rope_dim / 2 > head_dim)) return err(cudaErrorInvalidValue);
  if (k_heads > 0 && k == nullptr) return err(cudaErrorInvalidValue);
  const int work = (int)((q_heads + k_heads) * (rope_dim / 2));
  const int threads = work >= 256 ? 256 : ((work + 31) / 32) * 32;
  auto qp = static_cast<bf16*>(q);
  auto kp = static_cast<bf16*>(k);
  if (neox) {
    rope_kernel<true><<<(unsigned)tokens, threads, 0, stream>>>(qp, kp, positions, cos_sin, (int)q_heads,
                                                                (int)k_heads, (int)head_dim, (int)rope_dim, (int)span);
  } else {
    rope_kernel<false><<<(unsigned)tokens, threads, 0, stream>>>(qp, kp, positions, cos_sin, (int)q_heads,
                                                                 (int)k_heads, (int)head_dim, (int)rope_dim, 0);
  }
  return last_err();
}

extern "C" int rope_bf16(void* q, void* k, const int32_t* positions, const float* cos_sin, int64_t tokens,
                         int64_t q_heads, int64_t k_heads, int64_t head_dim, int64_t rope_dim, int neox,
                         cudaStream_t stream) {
  return rope_span_bf16(q, k, positions, cos_sin, tokens, q_heads, k_heads, head_dim, rope_dim, rope_dim / 2, neox,
                        stream);
}
