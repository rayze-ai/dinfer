#include "common.cuh"
#include "dinfer_kernels.h"

using namespace dinfer;

namespace {

// rows：非空时输出第 r 行取 x 的第 rows[r] 行（gather + rmsnorm 融合，Q3 第 59 轮：lm_head 前只取每请求最后一行）。
// 向量化版本（B4 第 1 轮）：每线程一次 16 B（8 × bf16）读 x / res / w，所有访存在归约前一次发出，
// 不再按 hidden/BLOCK 轮次串行等待——decode T=1 时每 token 只有一个 block，串行访存链在 B300 这类高延迟 HBM 上
// 把 0.5 µs 的工作拖到 3.7 µs。VPT = 每线程向量数（hidden / 8 / BLOCK），寄存器驻留，不读两遍。
template <int BLOCK, int VPT, bool FUSED_ADD>
__global__ void __launch_bounds__(BLOCK) rmsnorm_vec_kernel(const bf16* __restrict__ x, bf16* __restrict__ res,
                                                           const bf16* __restrict__ w, bf16* __restrict__ out,
                                                           const int32_t* __restrict__ rows, int hidden, float eps) {
  const int64_t row = blockIdx.x;
  const int64_t src_row = rows ? rows[row] : row;
  const uint4* xr = reinterpret_cast<const uint4*>(x + src_row * hidden);
  uint4* rr = FUSED_ADD ? reinterpret_cast<uint4*>(res + row * hidden) : nullptr;
  const uint4* wv = reinterpret_cast<const uint4*>(w);
  uint4* orow = reinterpret_cast<uint4*>(out + row * hidden);
  const int nvec = hidden / 8;

  float v[VPT][8];
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < VPT; ++j) {
    const int i = threadIdx.x + j * BLOCK;
    if (i < nvec) {
      uint4 xa = xr[i];
      const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xa);
      float2 f[4];
#pragma unroll
      for (int c = 0; c < 4; ++c) f[c] = __bfloat1622float2(xp[c]);
      if (FUSED_ADD) {
        uint4 ra = rr[i];
        const __nv_bfloat162* rp = reinterpret_cast<const __nv_bfloat162*>(&ra);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const float2 g = __bfloat1622float2(rp[c]);
          f[c].x += g.x; f[c].y += g.y;
        }
        uint4 wb;
        __nv_bfloat162* wp = reinterpret_cast<__nv_bfloat162*>(&wb);
#pragma unroll
        for (int c = 0; c < 4; ++c) wp[c] = __float22bfloat162_rn(f[c]);
        rr[i] = wb;
        // 下游按 bf16 残差再读：用舍入后的值算平方和与归一化，与标量版逐位一致
#pragma unroll
        for (int c = 0; c < 4; ++c) f[c] = __bfloat1622float2(wp[c]);
      }
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        v[j][2 * c] = f[c].x; v[j][2 * c + 1] = f[c].y;
        ss += f[c].x * f[c].x + f[c].y * f[c].y;
      }
    }
  }
  ss = block_reduce_sum<BLOCK>(ss);
  const float inv = rsqrtf(ss / hidden + eps);
#pragma unroll
  for (int j = 0; j < VPT; ++j) {
    const int i = threadIdx.x + j * BLOCK;
    if (i < nvec) {
      uint4 wa = wv[i];
      const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wa);
      uint4 ob;
      __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&ob);
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float2 g = __bfloat1622float2(wp[c]);
        op[c] = __floats2bfloat162_rn(v[j][2 * c] * inv * g.x, v[j][2 * c + 1] * inv * g.y);
      }
      orow[i] = ob;
    }
  }
}

// 标量回退：hidden 不是 8 的倍数或 > 8192 时用（与旧实现相同）。
template <int BLOCK, bool FUSED_ADD>
__global__ void rmsnorm_kernel(const bf16* __restrict__ x, bf16* __restrict__ res, const bf16* __restrict__ w,
                               bf16* __restrict__ out, const int32_t* __restrict__ rows, int hidden, float eps) {
  const int64_t row = blockIdx.x;
  const int64_t src_row = rows ? rows[row] : row;
  const bf16* xr = x + src_row * hidden;
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
int launch(const void* x, void* res, const void* w, void* out, const int32_t* rows, int64_t tokens, int64_t hidden,
           float eps, cudaStream_t s) {
  if (tokens <= 0) return 0;
  if (hidden <= 0 || hidden > (1 << 20)) return err(cudaErrorInvalidValue);
  const dim3 grid((unsigned)tokens);
  auto xp = static_cast<const bf16*>(x);
  auto rp = static_cast<bf16*>(res);
  auto wp = static_cast<const bf16*>(w);
  auto op = static_cast<bf16*>(out);
  const bool aligned = (hidden % 8 == 0) && ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(w) |
                                              reinterpret_cast<uintptr_t>(out) |
                                              (FUSED_ADD ? reinterpret_cast<uintptr_t>(res) : 0)) & 15) == 0;
  if (aligned && hidden <= 1024) {
    // ≤ 1024：128 线程 × 1 向量（hidden=1024 正好满），一次访存
    rmsnorm_vec_kernel<128, 1, FUSED_ADD><<<grid, 128, 0, s>>>(xp, rp, wp, op, rows, (int)hidden, eps);
  } else if (aligned && hidden <= 4096) {
    rmsnorm_vec_kernel<256, 2, FUSED_ADD><<<grid, 256, 0, s>>>(xp, rp, wp, op, rows, (int)hidden, eps);
  } else if (aligned && hidden <= 8192) {
    rmsnorm_vec_kernel<256, 4, FUSED_ADD><<<grid, 256, 0, s>>>(xp, rp, wp, op, rows, (int)hidden, eps);
  } else if (hidden <= 1024) {
    rmsnorm_kernel<256, FUSED_ADD><<<grid, 256, 0, s>>>(xp, rp, wp, op, rows, (int)hidden, eps);
  } else {
    rmsnorm_kernel<1024, FUSED_ADD><<<grid, 1024, 0, s>>>(xp, rp, wp, op, rows, (int)hidden, eps);
  }
  return last_err();
}

}  // namespace

extern "C" int rmsnorm_bf16(const void* x, const void* w, void* out, int64_t tokens, int64_t hidden, float eps,
                            cudaStream_t stream) {
  return launch<false>(x, nullptr, w, out, nullptr, tokens, hidden, eps, stream);
}

// out[r] = rmsnorm(x[rows[r]]) · w，r < num_rows（替代 gather_rows + rmsnorm 两次 launch）
extern "C" int gather_rmsnorm_bf16(const void* x, const int32_t* rows, const void* w, void* out, int64_t num_rows,
                                   int64_t hidden, float eps, cudaStream_t stream) {
  return launch<false>(x, nullptr, w, out, rows, num_rows, hidden, eps, stream);
}

extern "C" int fused_add_rmsnorm_bf16(const void* x, void* res, const void* w, void* out, int64_t tokens,
                                      int64_t hidden, float eps, cudaStream_t stream) {
  return launch<true>(x, res, w, out, nullptr, tokens, hidden, eps, stream);
}
