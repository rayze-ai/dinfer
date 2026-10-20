// r11b 探针：同一 W（34816×5120 bf16 = 356 MB，冷 L2 用 3 份轮转）上 ① 纯流式读 kernel（上限），② 不同 grid/线程配置。
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdint.h>
__global__ void stream_rows(const uint4* __restrict__ w, int n, int k8, int rows_per_block, float* sink) {
  // 每 block rows_per_block 行，每行 k8 个 16 B；lane 顺序读
  const int r0 = blockIdx.x * rows_per_block;
  uint4 acc = {0, 0, 0, 0};
  for (int r = 0; r < rows_per_block; ++r) {
    const uint4* row = w + (long)(r0 + r) * k8;
    for (int i = threadIdx.x; i < k8; i += blockDim.x) { uint4 v = __ldg(row + i); acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w; }
  }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678u) sink[0] = 1.f;
}
// 8 行 × 128 列 块状（模拟 mma 的 warp 读模式：每 lane 16 B、4 lane 一行 64 B、8 行）
__global__ void stream_mma_pattern(const __nv_bfloat16* __restrict__ w, int n, int k, int stripes, float* sink) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int n0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 8 * stripes;
  if (n0 >= n) return;
  uint32_t acc = 0;
  for (int c = 0; c < k; c += 128) {
    uint4 v[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r)
        if (r < stripes) v[r][j] = __ldg(reinterpret_cast<const uint4*>(w + (long)min(n0 + 8 * r + g, n - 1) * k + c + j * 32 + t * 8));
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r)
        if (r < stripes) acc ^= v[r][j].x ^ v[r][j].y ^ v[r][j].z ^ v[r][j].w;
  }
  if (acc == 0x12345678u) sink[0] = 1.f;
}
int main() {
  const int n = 34816, k = 5120; const size_t bytes = (size_t)n * k * 2; const int copies = 3;
  __nv_bfloat16* w[copies]; for (int i = 0; i < copies; ++i) { cudaMalloc(&w[i], bytes); cudaMemset(w[i], 1, bytes); }
  float* sink; cudaMalloc(&sink, 4);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  auto report = [&](const char* name, float ms, int iters) { float us = ms * 1e3f / iters; printf("%-40s %8.1f us %6.0f GB/s\n", name, us, bytes / us / 1e3); };
  // ① 流式：rows_per_block ∈ {1,2,4,8}，block 128/256 线程
  for (int rpb : {1, 2, 4, 8}) for (int bs : {128, 256}) {
    char nm[64]; snprintf(nm, 64, "stream_rows rpb=%d bs=%d grid=%d", rpb, bs, n / rpb);
    for (int i = 0; i < 2; ++i) stream_rows<<<n / rpb, bs>>>((const uint4*)w[i % copies], n, k / 8, rpb, sink);
    cudaEventRecord(a); const int it = 9; for (int i = 0; i < it; ++i) stream_rows<<<n / rpb, bs>>>((const uint4*)w[i % copies], n, k / 8, rpb, sink);
    cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); report(nm, ms, it);
  }
  // ② mma 读模式（无 mma）：stripes ∈ {1,2,4}，warps ∈ {4,8}
  for (int st : {1, 2, 4}) for (int wp : {4, 8}) {
    char nm[64]; snprintf(nm, 64, "mma_pattern stripes=%d warps=%d", st, wp);
    const int grid = (n + 8 * st * wp - 1) / (8 * st * wp);
    for (int i = 0; i < 2; ++i) stream_mma_pattern<<<grid, 32 * wp>>>(w[i % copies], n, k, st, sink);
    cudaEventRecord(a); const int it = 9; for (int i = 0; i < it; ++i) stream_mma_pattern<<<grid, 32 * wp>>>(w[i % copies], n, k, st, sink);
    cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); report(nm, ms, it);
  }
  cudaError_t e = cudaGetLastError(); if (e) printf("err %s\n", cudaGetErrorString(e));
  return 0;
}
