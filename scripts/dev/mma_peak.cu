// mma.sync m16n8k16 bf16 → f32 纯算力上限探针（r11a）：无访存，每 warp CH 条独立累加链。
// 用途：量 B300（sm_103）上 legacy mma.sync 的实际峰值，决定 FA2 结构 kernel 的物理上限（tcgen05 另算）。
// nvcc -O3 -arch=sm_103 -o mma_peak mma_peak.cu && ./mma_peak
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
template <int CH>
__global__ void __launch_bounds__(128) mma_loop(float* out, int iters) {
  uint32_t a[4] = {0x3f803f80u + threadIdx.x, 0x3f803f80u, 0x3f803f80u, 0x3f803f80u};
  uint32_t b0 = 0x3f803f80u, b1 = 0x3f803f80u + blockIdx.x;
  float c[CH][4];
#pragma unroll
  for (int i = 0; i < CH; ++i) c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0.f;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < CH; ++i)
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                   : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < CH; ++i) s += c[i][0] + c[i][1] + c[i][2] + c[i][3];
  if (s == 12345.f) out[threadIdx.x] = s;
}
template <int CH>
void run(int blocks_per_sm, int sm) {
  float* out; cudaMalloc(&out, 1024 * 4);
  const int iters = 4096, blocks = sm * blocks_per_sm;
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  mma_loop<CH><<<blocks, 128>>>(out, iters);
  cudaEventRecord(e0);
  for (int r = 0; r < 5; ++r) mma_loop<CH><<<blocks, 128>>>(out, iters);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 5;
  const double flop = (double)blocks * 4 /*warps*/ * iters * CH * (16.0 * 8 * 16 * 2);
  printf("CH=%2d blocks/SM=%d (%2d warp/SM): %.3f ms  %.1f TFLOP/s\n", CH, blocks_per_sm, blocks_per_sm * 4, ms, flop / ms / 1e9);
  cudaFree(out);
}
int main() {
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  int clk = 0; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
  printf("%s SM=%d clock=%d MHz\n", p.name, p.multiProcessorCount, clk / 1000);
  const int sm = p.multiProcessorCount;
  run<4>(1, sm); run<8>(1, sm); run<8>(2, sm); run<8>(4, sm); run<16>(2, sm); run<16>(4, sm); run<8>(8, sm); run<4>(8, sm);
  return 0;
}
