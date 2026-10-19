#include "common.cuh"
#include "kernels.h"

using namespace dinfer;

namespace {

// 一个线程处理一个 (token, head, pair)。
template <bool NEOX>
__global__ void rope_kernel(bf16* __restrict__ q, bf16* __restrict__ k, const int32_t* __restrict__ pos,
                            const float* __restrict__ cos_sin, int q_heads, int k_heads, int head_dim,
                            int rope_dim) {
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
      b = i + half;
    } else {
      a = 2 * i;
      b = 2 * i + 1;
    }
    const float x0 = to_f32(base[a]), x1 = to_f32(base[b]);
    base[a] = to_bf16(x0 * c - x1 * s);
    base[b] = to_bf16(x1 * c + x0 * s);
  }
}

}  // namespace

extern "C" int rope_bf16(void* q, void* k, const int32_t* positions, const float* cos_sin, int64_t tokens,
                         int64_t q_heads, int64_t k_heads, int64_t head_dim, int64_t rope_dim, int neox,
                         cudaStream_t stream) {
  if (tokens <= 0) return 0;
  if (rope_dim <= 0 || (rope_dim & 1) || rope_dim > head_dim) return err(cudaErrorInvalidValue);
  if (k_heads > 0 && k == nullptr) return err(cudaErrorInvalidValue);
  const int work = (int)((q_heads + k_heads) * (rope_dim / 2));
  const int threads = work >= 256 ? 256 : ((work + 31) / 32) * 32;
  auto qp = static_cast<bf16*>(q);
  auto kp = static_cast<bf16*>(k);
  if (neox) {
    rope_kernel<true><<<(unsigned)tokens, threads, 0, stream>>>(qp, kp, positions, cos_sin, (int)q_heads,
                                                                (int)k_heads, (int)head_dim, (int)rope_dim);
  } else {
    rope_kernel<false><<<(unsigned)tokens, threads, 0, stream>>>(qp, kp, positions, cos_sin, (int)q_heads,
                                                                 (int)k_heads, (int)head_dim, (int)rope_dim);
  }
  return last_err();
}
