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

// 向量化版本（B4 第 1 轮）：grid = (tokens, ceil(inter/8/256))，每线程 8 个元素一次 16 B 读写，一个 token 拆成
// 多个 block 并行而不是一个 block 串行 inter/256 轮——decode T=1 时旧版每 token 只有 1 个 block × 12 轮串行访存
// （Qwen3 inter=3072），B300 上 6.9 µs，带宽下限 0.5 µs。
constexpr int SM_THREADS = 256;
__global__ void __launch_bounds__(SM_THREADS) silu_and_mul_vec_kernel(const bf16* __restrict__ x,
                                                                      bf16* __restrict__ out, int inter) {
  const int64_t row = blockIdx.x;
  const int i = blockIdx.y * SM_THREADS + threadIdx.x;  // 向量下标（8 元素一组）
  if (i >= inter / 8) return;
  const uint4* gate = reinterpret_cast<const uint4*>(x + row * 2 * inter);
  const uint4* up = reinterpret_cast<const uint4*>(x + row * 2 * inter + inter);
  uint4* o = reinterpret_cast<uint4*>(out + row * inter);
  const uint4 ga = gate[i], ua = up[i];
  const __nv_bfloat162* gp = reinterpret_cast<const __nv_bfloat162*>(&ga);
  const __nv_bfloat162* upp = reinterpret_cast<const __nv_bfloat162*>(&ua);
  uint4 ob;
  __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&ob);
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const float2 g = __bfloat1622float2(gp[c]), u = __bfloat1622float2(upp[c]);
    op[c] = __floats2bfloat162_rn(silu(g.x) * u.x, silu(g.y) * u.y);
  }
  o[i] = ob;
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
  const bool aligned = (inter % 8 == 0) &&
                       ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(out)) & 15) == 0;
  if (aligned) {
    const unsigned nvec = (unsigned)(inter / 8);
    const dim3 grid((unsigned)tokens, (nvec + SM_THREADS - 1) / SM_THREADS);
    silu_and_mul_vec_kernel<<<grid, SM_THREADS, 0, stream>>>(static_cast<const bf16*>(x), static_cast<bf16*>(out),
                                                              (int)inter);
    return last_err();
  }
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
