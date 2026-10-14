#include "dinfer_kernels.h"

namespace {
__global__ void add_f32_kernel(const float* __restrict__ a, const float* __restrict__ b,
                               float* __restrict__ out, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) out[i] = a[i] + b[i];
}
}  // namespace

extern "C" int add_f32(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream) {
  if (n <= 0) return 0;
  int threads = 256;
  int64_t want = (n + threads - 1) / threads;
  int blocks = (int)(want > 4096 ? 4096 : want);
  add_f32_kernel<<<blocks, threads, 0, stream>>>(a, b, out, n);
  return (int)cudaGetLastError();
}
