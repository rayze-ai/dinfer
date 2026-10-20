// r11b 探针 2：PDL（programmatic dependent launch）能否把相邻 kernel 的 ramp/tail 重叠掉。
// 64 个 W（各 169 MB，模拟 in_proj 一层一个）依次流式读；A：普通 launch；B：cudaLaunchKernelEx + programmaticStreamSerialization，
// kernel 开头 griddepcontrol.wait（cudaGridDependencySynchronize），末尾前 launch_dependents。
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdint.h>
__global__ void stream_k(const uint4* __restrict__ w, int k8, int rpb, float* sink, int pdl) {
  if (pdl) cudaGridDependencySynchronize();
  const int r0 = blockIdx.x * rpb;
  uint4 acc = {0, 0, 0, 0};
  for (int r = 0; r < rpb; ++r) {
    const uint4* row = w + (long)(r0 + r) * k8;
    for (int i = threadIdx.x; i < k8; i += blockDim.x) { uint4 v = __ldg(row + i); acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w; }
  }
  if (pdl) cudaTriggerProgrammaticLaunchCompletion();
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678u) sink[0] = 1.f;
}
int main() {
  const int n = 16480, k = 5120, L = 64; const size_t bytes = (size_t)n * k * 2;
  uint4* w[L]; for (int i = 0; i < L; ++i) { cudaMalloc(&w[i], bytes); cudaMemset(w[i], 1, bytes); }
  float* sink; cudaMalloc(&sink, 4);
  cudaStream_t s; cudaStreamCreate(&s);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  const int rpb = 2, bs = 128, grid = n / rpb;
  for (int mode = 0; mode < 2; ++mode) {
    for (int rep = 0; rep < 3; ++rep) {
      cudaEventRecord(a, s);
      for (int i = 0; i < L; ++i) {
        if (mode == 0) {
          stream_k<<<grid, bs, 0, s>>>(w[i], k / 8, rpb, sink, 0);
        } else {
          cudaLaunchConfig_t cfg = {}; cfg.gridDim = grid; cfg.blockDim = bs; cfg.stream = s;
          cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
          cfg.attrs = at; cfg.numAttrs = 1;
          cudaLaunchKernelEx(&cfg, stream_k, (const uint4*)w[i], k / 8, rpb, sink, 1);
        }
      }
      cudaEventRecord(b, s); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b);
      if (rep == 2) printf("%s: 64 × 169 MB  total %.1f us  per-kernel %.1f us  %.0f GB/s\n", mode ? "PDL   " : "plain ", ms * 1e3, ms * 1e3 / L, bytes * L / (ms * 1e3) / 1e3);
    }
  }
  // graph 捕获版（decode 实际走 graph）：两种模式各捕获一次重放
  for (int mode = 0; mode < 2; ++mode) {
    cudaGraph_t g; cudaGraphExec_t ge;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < L; ++i) {
      if (mode == 0) stream_k<<<grid, bs, 0, s>>>(w[i], k / 8, rpb, sink, 0);
      else {
        cudaLaunchConfig_t cfg = {}; cfg.gridDim = grid; cfg.blockDim = bs; cfg.stream = s;
        cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = at; cfg.numAttrs = 1;
        cudaLaunchKernelEx(&cfg, stream_k, (const uint4*)w[i], k / 8, rpb, sink, 1);
      }
    }
    cudaStreamEndCapture(s, &g); cudaGraphInstantiate(&ge, g, 0);
    for (int rep = 0; rep < 3; ++rep) {
      cudaEventRecord(a, s); cudaGraphLaunch(ge, s); cudaEventRecord(b, s); cudaEventSynchronize(b);
      float ms; cudaEventElapsedTime(&ms, a, b);
      if (rep == 2) printf("graph %s: total %.1f us  per-kernel %.1f us  %.0f GB/s\n", mode ? "PDL  " : "plain", ms * 1e3, ms * 1e3 / L, bytes * L / (ms * 1e3) / 1e3);
    }
  }
  cudaError_t e = cudaGetLastError(); if (e) printf("err %s\n", cudaGetErrorString(e));
  return 0;
}
