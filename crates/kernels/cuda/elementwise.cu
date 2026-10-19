// 逐元素小算子：原位加、按行 gather、bf16→f32。
#include "common.cuh"
#include "kernels.h"

using namespace dinfer;

namespace {

// KV 打包（host tier / PD）：把 n 个块的全部层打包成连续 staging，每块 [layer][kv_head][slot][K/V][D]（**head-major**，
// design §13.7 异构 tp：cache 内是 [slot][kv_head][2][D]，转成按头连续后，TP 下 rank r 的分片就是每层里连续的一段，
// 任意 tp_P / tp_D 组合可用 2D memcpy 按头段拼 / 拆）。layer_ptrs[L] 每层 cache 基址；blocks[n] 块号。
// grid = (n, L)，block = 256 线程，按 16 B 向量搬；row_bytes = 2 × D × 2（一个 slot 一个头的 K+V）。
__global__ void kv_pack_kernel(const uint8_t* const* __restrict__ layer_ptrs, const int32_t* __restrict__ blocks,
                               uint8_t* __restrict__ staging, int64_t per_layer_bytes, int num_layers, int block_size,
                               int kv_heads, int row16) {
  const int b = blockIdx.x, l = blockIdx.y;
  const uint8_t* src = layer_ptrs[l] + (int64_t)blocks[b] * per_layer_bytes;
  uint8_t* dst = staging + ((int64_t)b * num_layers + l) * per_layer_bytes;
  const int64_t n16 = per_layer_bytes / 16;
  for (int64_t i = threadIdx.x; i < n16; i += blockDim.x) {
    const int64_t pair = i / row16, v = i % row16;  // 读端 coalesced：pair = s × hk + h
    const int s = (int)(pair / kv_heads), h = (int)(pair % kv_heads);
    reinterpret_cast<uint4*>(dst)[((int64_t)h * block_size + s) * row16 + v] = reinterpret_cast<const uint4*>(src)[i];
  }
}
__global__ void kv_unpack_kernel(uint8_t* const* __restrict__ layer_ptrs, const int32_t* __restrict__ blocks,
                                 const uint8_t* __restrict__ staging, int64_t per_layer_bytes, int num_layers,
                                 int block_size, int kv_heads, int row16) {
  const int b = blockIdx.x, l = blockIdx.y;
  uint8_t* dst = layer_ptrs[l] + (int64_t)blocks[b] * per_layer_bytes;
  const uint8_t* src = staging + ((int64_t)b * num_layers + l) * per_layer_bytes;
  const int64_t n16 = per_layer_bytes / 16;
  for (int64_t i = threadIdx.x; i < n16; i += blockDim.x) {
    const int64_t pair = i / row16, v = i % row16;  // 写端 coalesced：pair = s × hk + h
    const int s = (int)(pair / kv_heads), h = (int)(pair % kv_heads);
    reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<const uint4*>(src)[((int64_t)h * block_size + s) * row16 + v];
  }
}

__global__ void add_inplace_kernel(const bf16* __restrict__ x, bf16* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) y[i] = to_bf16(to_f32(y[i]) + to_f32(x[i]));
}

__global__ void gather_rows_kernel(const bf16* __restrict__ src, bf16* __restrict__ dst,
                                   const int32_t* __restrict__ rows, int width) {
  const int64_t r = blockIdx.x;
  const bf16* s = src + (int64_t)rows[r] * width;
  bf16* d = dst + r * width;
  for (int i = threadIdx.x; i < width; i += blockDim.x) d[i] = s[i];
}

__global__ void bf16_to_f32_kernel(const bf16* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) y[i] = to_f32(x[i]);
}


// [nb][rows][cols] bf16 → [rows][nb*cols] f32：块 b 的列段落到输出列 [b*cols, (b+1)*cols)。
// 用途：TP 下按输出维切分的 GEMM（如 lm_head 列切）各 rank 局部结果经 all-gather 后是按 rank 连续的块布局，
// 这里一次 kernel 既修正布局又升 f32，省掉"先转置再转换"两趟读写。
__global__ void gather_blocks_bf16_to_f32_kernel(const bf16* __restrict__ x, float* __restrict__ y, int64_t nb,
                                                 int64_t rows, int64_t cols) {
  const int64_t n = nb * rows * cols;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += stride) {
    // 按输出线性序遍历（写合并）：i = r * (nb*cols) + b*cols + c
    const int64_t r = i / (nb * cols);
    const int64_t rem = i - r * (nb * cols);
    const int64_t b = rem / cols;
    const int64_t c = rem - b * cols;
    y[i] = to_f32(x[(b * rows + r) * cols + c]);
  }
}

inline int grid_for(int64_t n) {
  const int64_t want = (n + 255) / 256;
  return (int)(want > 4096 ? 4096 : want);
}

}  // namespace

extern "C" int add_inplace_bf16(const void* x, void* y, int64_t n, cudaStream_t stream) {
  if (n <= 0) return 0;
  add_inplace_kernel<<<grid_for(n), 256, 0, stream>>>(static_cast<const bf16*>(x), static_cast<bf16*>(y), n);
  return last_err();
}

extern "C" int gather_rows_bf16(const void* src, void* dst, const int32_t* rows, int64_t num_rows, int64_t width,
                                cudaStream_t stream) {
  if (num_rows <= 0) return 0;
  if (width <= 0) return err(cudaErrorInvalidValue);
  const int threads = width >= 256 ? 256 : (int)(((width + 31) / 32) * 32);
  gather_rows_kernel<<<(unsigned)num_rows, threads, 0, stream>>>(static_cast<const bf16*>(src),
                                                                 static_cast<bf16*>(dst), rows, (int)width);
  return last_err();
}

__global__ void scale_f32_kernel(float* __restrict__ x, int64_t n, float a) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) x[i] *= a;
}

extern "C" int scale_f32(float* x, int64_t n, float a, cudaStream_t stream) {
  if (n <= 0) return 0;
  scale_f32_kernel<<<grid_for(n), 256, 0, stream>>>(x, n, a);
  return last_err();
}

extern "C" int bf16_to_f32(const void* x, float* y, int64_t n, cudaStream_t stream) {
  if (n <= 0) return 0;
  bf16_to_f32_kernel<<<grid_for(n), 256, 0, stream>>>(static_cast<const bf16*>(x), y, n);
  return last_err();
}

extern "C" int gather_blocks_bf16_to_f32(const void* x, float* y, int64_t nb, int64_t rows, int64_t cols,
                                         cudaStream_t stream) {
  const int64_t n = nb * rows * cols;
  if (n <= 0) return 0;
  gather_blocks_bf16_to_f32_kernel<<<grid_for(n), 256, 0, stream>>>(static_cast<const bf16*>(x), y, nb, rows, cols);
  return last_err();
}

// ---------- zero_rows_where_pos0: x[t, :] = 0 where pos[t] == 0（MTP：位置 0 的 embedding 置零） ----------
namespace {
__global__ void zero_rows_pos0_kernel(bf16* __restrict__ x, const int32_t* __restrict__ pos, int64_t cols) {
  const int64_t t = blockIdx.x;
  if (pos[t] != 0) return;
  bf16* row = x + t * cols;
  for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) row[c] = to_bf16(0.f);
}
}  // namespace

extern "C" int zero_rows_where_pos0_bf16(void* x, const int32_t* pos, int64_t tokens, int64_t cols, cudaStream_t stream) {
  if (tokens <= 0) return 0;
  if (cols <= 0) return err(cudaErrorInvalidValue);
  zero_rows_pos0_kernel<<<(unsigned)tokens, 256, 0, stream>>>(static_cast<bf16*>(x), pos, cols);
  return last_err();
}

extern "C" int kv_pack(const void* const* layer_ptrs, const int32_t* blocks, void* staging, int64_t n_blocks,
                       int64_t num_layers, int64_t per_layer_bytes, int64_t block_size, int64_t kv_heads,
                       cudaStream_t stream) {
  if (n_blocks <= 0) return 0;
  if (per_layer_bytes % 16 != 0 || num_layers <= 0 || block_size <= 0 || kv_heads <= 0) return err(cudaErrorInvalidValue);
  const int64_t row = per_layer_bytes / (block_size * kv_heads);
  if (row * block_size * kv_heads != per_layer_bytes || row % 16 != 0) return err(cudaErrorInvalidValue);
  kv_pack_kernel<<<dim3((unsigned)n_blocks, (unsigned)num_layers), 256, 0, stream>>>(
      reinterpret_cast<const uint8_t* const*>(layer_ptrs), blocks, static_cast<uint8_t*>(staging), per_layer_bytes,
      (int)num_layers, (int)block_size, (int)kv_heads, (int)(row / 16));
  return last_err();
}
extern "C" int kv_unpack(void* const* layer_ptrs, const int32_t* blocks, const void* staging, int64_t n_blocks,
                         int64_t num_layers, int64_t per_layer_bytes, int64_t block_size, int64_t kv_heads,
                         cudaStream_t stream) {
  if (n_blocks <= 0) return 0;
  if (per_layer_bytes % 16 != 0 || num_layers <= 0 || block_size <= 0 || kv_heads <= 0) return err(cudaErrorInvalidValue);
  const int64_t row = per_layer_bytes / (block_size * kv_heads);
  if (row * block_size * kv_heads != per_layer_bytes || row % 16 != 0) return err(cudaErrorInvalidValue);
  kv_unpack_kernel<<<dim3((unsigned)n_blocks, (unsigned)num_layers), 256, 0, stream>>>(
      reinterpret_cast<uint8_t* const*>(layer_ptrs), blocks, static_cast<const uint8_t*>(staging), per_layer_bytes,
      (int)num_layers, (int)block_size, (int)kv_heads, (int)(row / 16));
  return last_err();
}
