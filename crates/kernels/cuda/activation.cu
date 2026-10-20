#include "common.cuh"
#include "kernels.h"

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

// SituAndMul（Kimi-K3 `hidden_act=situ`，HF modeling_kimi_linear.py::SituAndMul）：
//   a = beta·tanh(gate/beta)·sigmoid(gate)；linear_beta>0 时 up ← linear_beta·tanh(up/linear_beta)；out = bf16(a·up)
// 全部 f32 计算（HF 把 gate/up cast 到 f32 再算），beta / linear_beta 由 host 传（模型参数，不写死）。
__global__ void situ_and_mul_kernel(const bf16* __restrict__ x, bf16* __restrict__ out, int inter, float beta,
                                    float linear_beta) {
  const int64_t row = blockIdx.x;
  const bf16* gate = x + row * 2 * inter;
  const bf16* up = gate + inter;
  bf16* o = out + row * inter;
  for (int i = threadIdx.x; i < inter; i += blockDim.x) {
    const float g = to_f32(gate[i]);
    float u = to_f32(up[i]);
    const float a = beta * tanhf(g / beta) * (1.f / (1.f + __expf(-g)));
    if (linear_beta > 0.f) u = linear_beta * tanhf(u / linear_beta);
    o[i] = to_bf16(a * u);
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

// 输出门（Qwen3.5 attn_output_gate）：x *= sigmoid(gate)，逐元素。grid = ceil(n/8/256)，每线程 8 个元素一次 16 B。
__global__ void __launch_bounds__(SM_THREADS) mul_sigmoid_vec_kernel(bf16* __restrict__ x, const bf16* __restrict__ gate,
                                                                   int64_t nvec) {
  const int64_t i = (int64_t)blockIdx.x * SM_THREADS + threadIdx.x;
  if (i >= nvec) return;
  uint4* xp = reinterpret_cast<uint4*>(x);
  const uint4 ga = reinterpret_cast<const uint4*>(gate)[i];
  uint4 xa = xp[i];
  const __nv_bfloat162* gp = reinterpret_cast<const __nv_bfloat162*>(&ga);
  __nv_bfloat162* xh = reinterpret_cast<__nv_bfloat162*>(&xa);
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const float2 g = __bfloat1622float2(gp[c]);
    const float2 v = __bfloat1622float2(xh[c]);
    const float s0 = 1.f / (1.f + __expf(-g.x)), s1 = 1.f / (1.f + __expf(-g.y));
    xh[c] = __floats2bfloat162_rn(v.x * s0, v.y * s1);
  }
  xp[i] = xa;
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

extern "C" int situ_and_mul_bf16(const void* x, void* out, int64_t tokens, int64_t inter, float beta, float linear_beta,
                                 cudaStream_t stream) {
  if (tokens <= 0 || inter <= 0 || !(beta > 0.f)) return err(cudaErrorInvalidValue);
  situ_and_mul_kernel<<<(unsigned)tokens, 256, 0, stream>>>(static_cast<const bf16*>(x), static_cast<bf16*>(out),
                                                            (int)inter, beta, linear_beta);
  return err(cudaGetLastError());
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

// 行式版：x 紧凑 [rows, cols]，gate 是 [rows, gate_stride] 缓冲的列段（融合 qkv+gate GEMM 输出里的 gate 段，第 9 轮）。
// cols / gate_stride 都是 8 的倍数（16 B 向量）。
__global__ void __launch_bounds__(SM_THREADS) mul_sigmoid_rows_kernel(bf16* __restrict__ x, const bf16* __restrict__ gate,
                                                                    int64_t nvec, int cols8, int64_t gate_stride) {
  const int64_t i = (int64_t)blockIdx.x * SM_THREADS + threadIdx.x;
  if (i >= nvec) return;
  const int64_t r = i / cols8, c8 = i % cols8;
  uint4* xp = reinterpret_cast<uint4*>(x) + i;
  const uint4 ga = *reinterpret_cast<const uint4*>(gate + r * gate_stride + c8 * 8);
  uint4 xa = *xp;
  const __nv_bfloat162* gp = reinterpret_cast<const __nv_bfloat162*>(&ga);
  __nv_bfloat162* xh = reinterpret_cast<__nv_bfloat162*>(&xa);
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const float2 g = __bfloat1622float2(gp[c]);
    const float2 v = __bfloat1622float2(xh[c]);
    const float s0 = 1.f / (1.f + __expf(-g.x)), s1 = 1.f / (1.f + __expf(-g.y));
    xh[c] = __floats2bfloat162_rn(v.x * s0, v.y * s1);
  }
  *xp = xa;
}

extern "C" int mul_sigmoid_rows_bf16(void* x, const void* gate, int64_t rows, int64_t cols, int64_t gate_stride,
                                     cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return 0;
  if (cols % 8 != 0 || gate_stride % 8 != 0 || gate_stride < cols) return err(cudaErrorInvalidValue);
  const int64_t nvec = rows * (cols / 8);
  const unsigned grid = (unsigned)((nvec + SM_THREADS - 1) / SM_THREADS);
  mul_sigmoid_rows_kernel<<<grid, SM_THREADS, 0, stream>>>(static_cast<bf16*>(x), static_cast<const bf16*>(gate), nvec,
                                                          (int)(cols / 8), gate_stride);
  return last_err();
}

// 输出门：x *= sigmoid(gate)（原地），n 为元素数；n 必须是 8 的倍数且两个指针 16 B 对齐（调用方保证）。
extern "C" int mul_sigmoid_bf16(void* x, const void* gate, int64_t n, cudaStream_t stream) {
  if (n <= 0) return 0;
  if (n % 8 != 0) return err(cudaErrorInvalidValue);
  const int64_t nvec = n / 8;
  const unsigned grid = (unsigned)((nvec + SM_THREADS - 1) / SM_THREADS);
  mul_sigmoid_vec_kernel<<<grid, SM_THREADS, 0, stream>>>(static_cast<bf16*>(x), static_cast<const bf16*>(gate), nvec);
  return last_err();
}
