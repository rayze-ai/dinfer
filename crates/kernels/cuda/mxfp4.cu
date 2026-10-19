// MXFP4 专家权重的解包（compressed-tensors `mxfp4-pack-quantized`）。模型无关，只按形状参数。
//
// 格式（规格来源：K3 的 config.json quantization_config + vLLM `quantization/mxfp4.py:895` 的转换路径）：
//   weight_packed : u8 [rows, cols/2]，每字节两个 E2M1 值，**低半字节是偶数下标元素**（torch 的
//                   `float4_e2m1fn_x2` 打包约定）。
//   weight_scale  : u8 [rows, cols/group]，e8m0：值 = 2^(u8 - 127)（无符号指数，无尾数、无符号位）。
//   E2M1：4 位 = 1 符号 + 3 指数；幅值表 [0, .5, 1, 1.5, 2, 3, 4, 6]，指数偏差 1。
// 解出： w[r, c] = e2m1(nibble) · 2^(scale[r, c/group] - 127)
//
// 这是**参考路径**（正确性优先，bf16 回退用）：原生 MXFP4 GEMM 见 docs/loops/k3-moe.md 的 TODO。
#include "common.cuh"
#include "dinfer_kernels.h"

using namespace dinfer;

namespace {

// 幅值表：下标 = 3 位指数（e2m1 的偏置 1）
__constant__ float kE2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

__device__ __forceinline__ float e2m1(unsigned nibble) {
  const float mag = kE2M1[nibble & 7u];
  return (nibble & 8u) ? -mag : mag;
}

__device__ __forceinline__ float e8m0(unsigned s) {
  // 2^(s-127)：直接构造 fp32 位模式（指数 8 位 + 尾数 0）
  return __uint_as_float(s << 23);
}

__global__ void dequant_kernel(const uint8_t* __restrict__ packed, const uint8_t* __restrict__ scale,
                               bf16* __restrict__ out, long total, int cols, int group) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const long row = i / cols;
  const int col = (int)(i % cols);
  const uint8_t byte = packed[row * (cols / 2) + col / 2];
  const unsigned nib = (col & 1) ? (unsigned)(byte >> 4) : (unsigned)(byte & 0xFu);
  const float s = e8m0(scale[row * (cols / group) + col / group]);
  out[i] = to_bf16(e2m1(nib) * s);
}

}  // namespace

extern "C" {

int mxfp4_dequant_bf16(const void* packed, const void* scale, void* out, int64_t rows, int64_t cols,
                       int64_t group) {
  if (rows <= 0 || cols <= 0 || group <= 0) return err(cudaErrorInvalidValue);
  if (cols % 2 != 0 || cols % group != 0) return err(cudaErrorInvalidValue);
  const long total = rows * cols;
  const int nb = 256;
  dequant_kernel<<<(unsigned)((total + nb - 1) / nb), nb, 0, nullptr>>>(
      static_cast<const uint8_t*>(packed), static_cast<const uint8_t*>(scale), static_cast<bf16*>(out),
      total, (int)cols, (int)group);
  return last_err();
}

}  // extern "C"
