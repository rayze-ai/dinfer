// 参考 kernel 共用工具。目标 sm80+（3090 / 4090 / H100），bf16 用 cuda_bf16.h 原生类型。
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace dinfer {

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float to_f32(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 to_bf16(float v) { return __float2bfloat16(v); }

// block 内全部线程求和（blockDim.x 为 32 的倍数，<= 1024）。
template <int BLOCK>
__device__ __forceinline__ float block_reduce_sum(float v) {
  static_assert(BLOCK % 32 == 0 && BLOCK <= 1024, "BLOCK");
  __shared__ float smem[BLOCK / 32];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  float r = 0.f;
  if (threadIdx.x < BLOCK / 32) r = smem[threadIdx.x];
  if (warp == 0) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) r += __shfl_xor_sync(0xffffffffu, r, o);
    if (lane == 0) smem[0] = r;
  }
  __syncthreads();
  return smem[0];
}

template <int BLOCK>
__device__ __forceinline__ float block_reduce_max(float v) {
  static_assert(BLOCK % 32 == 0 && BLOCK <= 1024, "BLOCK");
  __shared__ float smem[BLOCK / 32];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  float r = -INFINITY;
  if (threadIdx.x < BLOCK / 32) r = smem[threadIdx.x];
  if (warp == 0) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) r = fmaxf(r, __shfl_xor_sync(0xffffffffu, r, o));
    if (lane == 0) smem[0] = r;
  }
  __syncthreads();
  return smem[0];
}

__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

inline int err(cudaError_t e) { return static_cast<int>(e); }
inline int last_err() { return err(cudaGetLastError()); }

}  // namespace dinfer
