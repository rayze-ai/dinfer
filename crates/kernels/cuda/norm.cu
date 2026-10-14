#include "common.cuh"
#include "dinfer_kernels.h"

using namespace dinfer;

namespace {

template <int BLOCK, bool FUSED_ADD>
__global__ void rmsnorm_kernel(const bf16* __restrict__ x, bf16* __restrict__ res, const bf16* __restrict__ w,
                               bf16* __restrict__ out, int hidden, float eps) {
  const int64_t row = blockIdx.x;
  const bf16* xr = x + row * hidden;
  bf16* rr = FUSED_ADD ? res + row * hidden : nullptr;
  bf16* orow = out + row * hidden;

  // 第一遍：累加平方和（FUSED_ADD 时顺便写回 res）。
  float ss = 0.f;
  for (int i = threadIdx.x; i < hidden; i += BLOCK) {
    float v = to_f32(xr[i]);
    if (FUSED_ADD) {
      v += to_f32(rr[i]);
      rr[i] = to_bf16(v);
    }
    ss += v * v;
  }
  ss = block_reduce_sum<BLOCK>(ss);
  const float inv = rsqrtf(ss / hidden + eps);

  // 第二遍：归一化并乘权重。
  const bf16* src = FUSED_ADD ? rr : xr;
  for (int i = threadIdx.x; i < hidden; i += BLOCK) {
    orow[i] = to_bf16(to_f32(src[i]) * inv * to_f32(w[i]));
  }
}

template <bool FUSED_ADD>
int launch(const void* x, void* res, const void* w, void* out, int64_t tokens, int64_t hidden, float eps,
           cudaStream_t s) {
  if (tokens <= 0) return 0;
  if (hidden <= 0 || hidden > (1 << 20)) return err(cudaErrorInvalidValue);
  const dim3 grid((unsigned)tokens);
  auto xp = static_cast<const bf16*>(x);
  auto rp = static_cast<bf16*>(res);
  auto wp = static_cast<const bf16*>(w);
  auto op = static_cast<bf16*>(out);
  if (hidden <= 1024) {
    rmsnorm_kernel<256, FUSED_ADD><<<grid, 256, 0, s>>>(xp, rp, wp, op, (int)hidden, eps);
  } else {
    rmsnorm_kernel<1024, FUSED_ADD><<<grid, 1024, 0, s>>>(xp, rp, wp, op, (int)hidden, eps);
  }
  return last_err();
}

}  // namespace

extern "C" int rmsnorm_bf16(const void* x, const void* w, void* out, int64_t tokens, int64_t hidden, float eps,
                            cudaStream_t stream) {
  return launch<false>(x, nullptr, w, out, tokens, hidden, eps, stream);
}

extern "C" int fused_add_rmsnorm_bf16(const void* x, void* res, const void* w, void* out, int64_t tokens,
                                      int64_t hidden, float eps, cudaStream_t stream) {
  return launch<true>(x, res, w, out, tokens, hidden, eps, stream);
}
