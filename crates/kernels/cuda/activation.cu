#include "common.cuh"
#include "dinfer_kernels.h"

using namespace dinfer;

namespace {

__global__ void silu_and_mul_kernel(const bf16* __restrict__ x, bf16* __restrict__ out, int inter) {
  const int64_t row = blockIdx.x;
  const bf16* gate = x + row * 2 * inter;
  const bf16* up = gate + inter;
  bf16* o = out + row * inter;
  for (int i = threadIdx.x; i < inter; i += blockDim.x) {
    o[i] = to_bf16(silu(to_f32(gate[i])) * to_f32(up[i]));
  }
}

__global__ void embedding_kernel(const int32_t* __restrict__ ids, const bf16* __restrict__ table,
                                 bf16* __restrict__ out, int hidden, int vocab) {
  const int64_t t = blockIdx.x;
  int id = ids[t];
  if (id < 0 || id >= vocab) id = 0;  // 越界 id 兜底取 0（与 vLLM 占位 token 行为一致）
  const bf16* src = table + (int64_t)id * hidden;
  bf16* dst = out + t * hidden;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) dst[i] = src[i];
}

}  // namespace

extern "C" int silu_and_mul_bf16(const void* x, void* out, int64_t tokens, int64_t inter, cudaStream_t stream) {
  if (tokens <= 0) return 0;
  if (inter <= 0) return err(cudaErrorInvalidValue);
  const int threads = inter >= 256 ? 256 : (int)(((inter + 31) / 32) * 32);
  silu_and_mul_kernel<<<(unsigned)tokens, threads, 0, stream>>>(static_cast<const bf16*>(x),
                                                                static_cast<bf16*>(out), (int)inter);
  return last_err();
}

extern "C" int embedding_bf16(const int32_t* ids, const void* table, void* out, int64_t tokens, int64_t hidden,
                              int64_t vocab, cudaStream_t stream) {
  if (tokens <= 0) return 0;
  if (hidden <= 0 || vocab <= 0) return err(cudaErrorInvalidValue);
  const int threads = hidden >= 256 ? 256 : (int)(((hidden + 31) / 32) * 32);
  embedding_kernel<<<(unsigned)tokens, threads, 0, stream>>>(ids, static_cast<const bf16*>(table),
                                                             static_cast<bf16*>(out), (int)hidden, (int)vocab);
  return last_err();
}
