// dinfer kernels 的 C ABI。所有入口：裸指针 + 维度 + cudaStream_t，返回 cudaError_t 数值。
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// out[i] = a[i] + b[i]，fp32。P0 自检用。
int dinfer_add_f32(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
