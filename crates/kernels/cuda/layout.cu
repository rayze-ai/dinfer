// 布局搬运类小算子：列切片拷贝、按 head 的 strided 拷贝、广播拼接。全部 bf16 逐元素。
#include "common.cuh"
#include "dinfer_kernels.h"

using namespace dinfer;

namespace {

// dst[t, 0:width] = src[t, col0:col0+width]（src 行宽 src_cols，dst 行宽 dst_cols，写到 dst 列偏移 dcol0）
__global__ void copy_cols_kernel(const bf16* __restrict__ src, bf16* __restrict__ dst, int src_cols, int col0,
                                 int dst_cols, int dcol0, int width) {
  const int64_t t = blockIdx.x;
  const bf16* s = src + t * src_cols + col0;
  bf16* d = dst + t * dst_cols + dcol0;
  for (int i = threadIdx.x; i < width; i += blockDim.x) d[i] = s[i];
}

// 把 [T, rope] 的 k_pe 广播到 [T, H, qk] 的每个 head 的 [nope:nope+rope] 位置
__global__ void bcast_kpe_kernel(const bf16* __restrict__ kpe, bf16* __restrict__ k, int H, int qk, int nope,
                                 int rope) {
  const int64_t t = blockIdx.x;
  for (int idx = threadIdx.x; idx < H * rope; idx += blockDim.x) {
    const int h = idx / rope, r = idx - h * rope;
    k[(t * H + h) * qk + nope + r] = kpe[t * rope + r];
  }
}

// 对 [T, H, in_w] 的张量按 head 取列区间，写到 [T, H, out_w] 的列区间（in/out 行宽按 head 计）
__global__ void copy_head_cols_kernel(const bf16* __restrict__ src, bf16* __restrict__ dst, int H, int in_w,
                                      int in_col0, int out_w, int out_col0, int width) {
  const int64_t t = blockIdx.x;
  for (int idx = threadIdx.x; idx < H * width; idx += blockDim.x) {
    const int h = idx / width, i = idx - h * width;
    dst[(t * H + h) * out_w + out_col0 + i] = src[(t * H + h) * in_w + in_col0 + i];
  }
}


// dst[t, 0:width] = src[t, head, col0:col0+width]   （src: [T, H, w]）
__global__ void gather_head_kernel(const bf16* __restrict__ src, bf16* __restrict__ dst, int H, int w, int head,
                                   int col0, int width) {
  const int64_t t = blockIdx.x;
  const bf16* s = src + (t * H + head) * w + col0;
  bf16* d = dst + t * width;
  for (int i = threadIdx.x; i < width; i += blockDim.x) d[i] = s[i];
}
// dst[t, head, col0:col0+width] = src[t, 0:width]   （dst: [T, H, w]）
__global__ void scatter_head_kernel(const bf16* __restrict__ src, bf16* __restrict__ dst, int H, int w, int head,
                                    int col0, int width) {
  const int64_t t = blockIdx.x;
  const bf16* s = src + t * width;
  bf16* d = dst + (t * H + head) * w + col0;
  for (int i = threadIdx.x; i < width; i += blockDim.x) d[i] = s[i];
}

}  // namespace

extern "C" int copy_cols_bf16(const void* src, void* dst, int64_t tokens, int64_t src_cols, int64_t col0,
                              int64_t dst_cols, int64_t dcol0, int64_t width, cudaStream_t stream) {
  if (tokens <= 0 || width <= 0) return 0;
  if (col0 + width > src_cols || dcol0 + width > dst_cols) return err(cudaErrorInvalidValue);
  const int threads = width >= 256 ? 256 : (int)(((width + 31) / 32) * 32);
  copy_cols_kernel<<<(unsigned)tokens, threads, 0, stream>>>(static_cast<const bf16*>(src), static_cast<bf16*>(dst),
                                                             (int)src_cols, (int)col0, (int)dst_cols, (int)dcol0,
                                                             (int)width);
  return last_err();
}

extern "C" int bcast_kpe_bf16(const void* kpe, void* k, int64_t tokens, int64_t heads, int64_t qk, int64_t nope,
                              int64_t rope, cudaStream_t stream) {
  if (tokens <= 0) return 0;
  if (nope + rope != qk || heads <= 0) return err(cudaErrorInvalidValue);
  const int64_t work = heads * rope;
  const int threads = work >= 256 ? 256 : (int)(((work + 31) / 32) * 32);
  bcast_kpe_kernel<<<(unsigned)tokens, threads, 0, stream>>>(static_cast<const bf16*>(kpe), static_cast<bf16*>(k),
                                                             (int)heads, (int)qk, (int)nope, (int)rope);
  return last_err();
}

extern "C" int copy_head_cols_bf16(const void* src, void* dst, int64_t tokens, int64_t heads, int64_t in_w,
                                   int64_t in_col0, int64_t out_w, int64_t out_col0, int64_t width,
                                   cudaStream_t stream) {
  if (tokens <= 0 || width <= 0) return 0;
  if (in_col0 + width > in_w || out_col0 + width > out_w || heads <= 0) return err(cudaErrorInvalidValue);
  const int64_t work = heads * width;
  const int threads = work >= 256 ? 256 : (int)(((work + 31) / 32) * 32);
  copy_head_cols_kernel<<<(unsigned)tokens, threads, 0, stream>>>(static_cast<const bf16*>(src),
                                                                  static_cast<bf16*>(dst), (int)heads, (int)in_w,
                                                                  (int)in_col0, (int)out_w, (int)out_col0, (int)width);
  return last_err();
}

extern "C" int gather_head_bf16(const void* src, void* dst, int64_t tokens, int64_t heads, int64_t w, int64_t head,
                                int64_t col0, int64_t width, cudaStream_t stream) {
  if (tokens <= 0 || width <= 0) return 0;
  if (head < 0 || head >= heads || col0 + width > w) return err(cudaErrorInvalidValue);
  const int threads = width >= 256 ? 256 : (int)(((width + 31) / 32) * 32);
  gather_head_kernel<<<(unsigned)tokens, threads, 0, stream>>>(static_cast<const bf16*>(src), static_cast<bf16*>(dst),
                                                               (int)heads, (int)w, (int)head, (int)col0, (int)width);
  return last_err();
}

extern "C" int scatter_head_bf16(const void* src, void* dst, int64_t tokens, int64_t heads, int64_t w, int64_t head,
                                 int64_t col0, int64_t width, cudaStream_t stream) {
  if (tokens <= 0 || width <= 0) return 0;
  if (head < 0 || head >= heads || col0 + width > w) return err(cudaErrorInvalidValue);
  const int threads = width >= 256 ? 256 : (int)(((width + 31) / 32) * 32);
  scatter_head_kernel<<<(unsigned)tokens, threads, 0, stream>>>(static_cast<const bf16*>(src),
                                                                static_cast<bf16*>(dst), (int)heads, (int)w,
                                                                (int)head, (int)col0, (int)width);
  return last_err();
}
