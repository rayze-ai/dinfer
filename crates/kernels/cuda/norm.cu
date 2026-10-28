#include <cuda_fp8.h>

#include "common.cuh"
#include "kernels.h"

using namespace dinfer;

namespace {

// 平方和，归约结构与 vLLM BI 的 Triton `_rms_norm_kernel`（BLOCK_SIZE=1024、4 warp × 32 lane × 8 元素）逐位同序：
// 8 元素组（gs，首元素乘 + 7 次 fma）→ 每 warp 32 组 = 256 列一段做 xor 蝶形 → 每 4 段 = 1024 列一块做 4-lane 蝶形 `(s0+s2)+(s1+s3)`
// → 块间从 0 顺序累加。旧写法（线程内 a·a+b·b 两两加 + 全 block 一次蝶形）在 Kaon-V3 v_norm（h=256）上差 2 元素（§53）。
// 要求 hidden % 8 == 0 且 hidden ≤ 8192（段数 ≤ 32）——vec kernel 的使用条件本就如此。
template <int BLOCK, int VPT>
__device__ __forceinline__ float rms_ss_triton(const float (&gs)[VPT], int nvec) {
  static_assert(BLOCK % 32 == 0 && BLOCK * VPT / 32 <= 32, "rms_ss_triton: 段数 > 32");
  __shared__ float seg[32];
  __shared__ float tot_s;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int nseg = (nvec + 31) / 32;
#pragma unroll
  for (int j = 0; j < VPT; ++j) {
    float g = gs[j];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) g = __fadd_rn(g, __shfl_xor_sync(0xffffffffu, g, o));
    const int sidx = j * (BLOCK / 32) + warp;
    if (lane == 0 && sidx < nseg) seg[sidx] = g;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float tot = 0.f;
    for (int c = 0; 4 * c < nseg; ++c) {
      float a[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) a[k] = (4 * c + k < nseg) ? seg[4 * c + k] : 0.f;
      tot = __fadd_rn(tot, __fadd_rn(__fadd_rn(a[0], a[2]), __fadd_rn(a[1], a[3])));
    }
    tot_s = tot;
  }
  __syncthreads();
  return tot_s;
}

// 1/rms：逐条对应 vLLM BI Triton `_rms_norm_kernel` 在 sm_120 上的 **SASS**（Triton 自带 ptxas 编出，§55）：
//   R9 = MUFU.RCP(n)；t = FFMA(R9, ss, eps)；rms = MUFU.SQRT(t)；inv = MUFU.RCP(rms) × 1
// 即 PTX 的 div.full(ss,n) 被拆成 ss·rcp(n) 并与 +eps 融合成一次 fma（mean 从不单独舍入），div.full(1,rms) = rcp(rms)。
// 只看 PTX / 源码都会写错（§52 的"乘倒数 + IEEE"、§55 前两版都被推翻）；n、rms 均为正规数，SASS 里的缩放分支不触发。
__device__ __forceinline__ float rcp_approx(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ float sqrt_approx(float x) {
  float r;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
// compile 口径（vLLM torch.compile 下**实际运行**的 norm，来源见 `docs/loops/kaon3-parity.md` 第 19 轮）：
// 图里是 `torch.ops.vllm_ir.rms_norm`（`vllm/ir/ops/layernorm.py:10`）被 inductor 融进相邻逐元素链，
// 语义在生成代码里等价于「全程 fp32、**只在 store 时舍入一次**」——源码里 `x.to(weight.dtype)` 那次 bf16
// 舍入被 inductor 消除了（生成代码 `tmp14 = tmp13.to(tl.float32)` 是空转，A/B 也证实 bf16 版与 f32 版逐位相同）。
// 与 BI 版**唯一的差别是 inv 的算法**：
//   BI     : rcp(sqrt(fma(rcp(n), ss, eps)))   —— mean 从不单独舍入
//   compile: rsqrt(ss / n + eps)               —— mean 单独舍入（torch `.mean()`），再 `rsqrt`
// n 与 ss 都是正规数，`ss/n` 用 IEEE 除法（`__fdiv_rn`）。
__device__ __forceinline__ float rms_inv_compile(float ss, int hidden, float eps) {
  return rsqrtf(__fdiv_rn(ss, (float)hidden) + eps);
}

// `mode`：0 = compile 无中间舍入，1 = eager BI，**2 = compile 且 `x·inv` 后先舍 bf16 再乘 w**。
//
// 2 的依据（2026-10-23，`docs/loops/kaon3-compile.md` 第 3 轮）：vLLM compile 的 FX 图里
// `vllm_ir.rms_norm` 的**返回类型就是 bf16**，且 `ir/ops/layernorm.py:17` 源码是
// `x = x.to(weight.dtype) * weight`（先舍 bf16 再乘 w）。mode 0 的「fp32 连乘、末尾舍一次」
// 是早期按生成代码推断的，与 FX 图矛盾——两种口径都留着 A/B，由 `RMSNORM_MODE=2` 选。
__device__ __forceinline__ float rms_inv(float ss, int hidden, float eps, int mode) {
  return (mode == 1) ? rcp_approx(sqrt_approx(__fmaf_rn(rcp_approx((float)hidden), ss, eps)))
                     : rms_inv_compile(ss, hidden, eps);
}

/// `y = bf16(x·inv)` —— mode 2 的中间舍入点；mode 0 时原样返回 f32（不引入额外舍入）。
__device__ __forceinline__ float norm_mid(float y, int mode) {
  return mode == 2 ? __bfloat162float(__float2bfloat16_rn(y)) : y;
}

// rows：非空时输出第 r 行取 x 的第 rows[r] 行（gather + rmsnorm 融合，Q3 第 59 轮：lm_head 前只取每请求最后一行）。
// 向量化版本（B4 第 1 轮）：每线程一次 16 B（8 × bf16）读 x / res / w，所有访存在归约前一次发出，
// 不再按 hidden/BLOCK 轮次串行等待——decode T=1 时每 token 只有一个 block，串行访存链在 B300 这类高延迟 HBM 上
// 把 0.5 µs 的工作拖到 3.7 µs。VPT = 每线程向量数（hidden / 8 / BLOCK），寄存器驻留，不读两遍。
// 第 7 轮：w 的读也提前到归约前发出（原来归约后才读 w，多一次 L2 往返串在链上）；BLOCK 可取 1024（layout 2）。
// B13（loop kaon3-decode-opt）：NORM_ADD=true 时 out 行原地累加：out[i] = bf16(f32(out[i]) + f32(bf16(norm·w)))，
// 与 `add_inplace_bf16_n` 的口径同式（norm 先舍 bf16，再 f32 加，再舍）。省掉 norm → 残差加的一次 launch。
__device__ __forceinline__ uint4 bf16x8_add(uint4 a, uint4 b) {
  uint4 o;
  const __nv_bfloat162* pa = reinterpret_cast<const __nv_bfloat162*>(&a);
  const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&b);
  __nv_bfloat162* po = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const float2 x = __bfloat1622float2(pa[c]), y = __bfloat1622float2(pb[c]);
    po[c] = __floats2bfloat162_rn(x.x + y.x, x.y + y.y);
  }
  return o;
}
__device__ __forceinline__ uint4 bf16x8_scale(uint4 a, float s) {
  uint4 o;
  const __nv_bfloat162* pa = reinterpret_cast<const __nv_bfloat162*>(&a);
  __nv_bfloat162* po = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const float2 x = __bfloat1622float2(pa[c]);
    po[c] = __floats2bfloat162_rn(x.x * s, x.y * s);
  }
  return o;
}
template <int BLOCK, int VPT, bool FUSED_ADD, bool NORM_ADD>
__global__ void __launch_bounds__(BLOCK) rmsnorm_vec_kernel(const bf16* __restrict__ x, bf16* __restrict__ res,
                                                           const bf16* __restrict__ w, bf16* __restrict__ out,
                                                           const int32_t* __restrict__ rows, int hidden, float eps,
                                                           int mode, float post_scale) {
  const int64_t row = blockIdx.x;
  const int64_t src_row = rows ? rows[row] : row;
  const uint4* xr = reinterpret_cast<const uint4*>(x + src_row * hidden);
  uint4* rr = FUSED_ADD ? reinterpret_cast<uint4*>(res + row * hidden) : nullptr;
  const uint4* wv = reinterpret_cast<const uint4*>(w);
  uint4* orow = reinterpret_cast<uint4*>(out + row * hidden);
  const int nvec = hidden / 8;

  float v[VPT][8];
  uint4 wa[VPT];
  float gs[VPT];   // 每个 8 元素组的平方和（组内按元素顺序累加，Triton 线程内归约同序）
#pragma unroll
  for (int j = 0; j < VPT; ++j) {
    const int i = threadIdx.x + j * BLOCK;
    gs[j] = 0.f;
    if (i < nvec) {
      uint4 xa = xr[i];
      // w == nullptr → 无权重 RMSNorm（Gemma4 的 v_norm / router 预处理），权重当作 1.0
      wa[j] = w ? wv[i] : make_uint4(0x3F803F80u, 0x3F803F80u, 0x3F803F80u, 0x3F803F80u);
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
      }
      // PTX（§55 实抓）：`mul.f32` 的是**第 1 个**元素，再 fma 第 0 个，然后 2..7（顺序 1,0,2,…,7——编译器调度结果，
      // 逐位相关：照 0..7 做在 Kaon-V3 L3/L4 的 3 行上 inv 偏 1 ulp）。fma.rn.f32.bf16 = 积精确、融合累加。
      float acc = __fmul_rn(v[j][1], v[j][1]);
      acc = __fmaf_rn(v[j][0], v[j][0], acc);
#pragma unroll
      for (int e = 2; e < 8; ++e) acc = __fmaf_rn(v[j][e], v[j][e], acc);
      gs[j] = acc;
    }
  }
  const float inv = rms_inv(rms_ss_triton<BLOCK, VPT>(gs, nvec), hidden, eps, mode);
#pragma unroll
  for (int j = 0; j < VPT; ++j) {
    const int i = threadIdx.x + j * BLOCK;
    if (i < nvec) {
      const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wa[j]);
      uint4 ob;
      __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&ob);
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float2 g = __bfloat1622float2(wp[c]);
        op[c] = __floats2bfloat162_rn(norm_mid(v[j][2 * c] * inv, mode) * g.x,
                                      norm_mid(v[j][2 * c + 1] * inv, mode) * g.y);
      }
      orow[i] = NORM_ADD ? (post_scale != 1.f ? bf16x8_scale(bf16x8_add(orow[i], ob), post_scale) : bf16x8_add(orow[i], ob)) : ob;
    }
  }
}

// 第 7 轮 layout 1：一 warp 一行、kNormWarps 个 warp 一 block（block 数 = ceil(rows / kNormWarps)）。
// 没有 block 归约 / __syncthreads，只有 warp shuffle；每 lane 顺序持 VPL = hidden/256 个 16 B 向量
// （hidden=5120 → 20 个，x 以 bf16 打包驻留寄存器，算两次转换省寄存器）。w 在归约后才读（否则 VPL 个 uint4 再翻倍）。
// 适用 hidden ≤ 4096（VPL ≤ 16）；更大的 hidden 寄存器压力大，由 host（ArchProfile）决定是否选它。
constexpr int kNormWarps = 4;

template <int VPL, bool FUSED_ADD, bool NORM_ADD>
__global__ void __launch_bounds__(kNormWarps * 32) rmsnorm_warp_kernel(const bf16* __restrict__ x,
                                                                        bf16* __restrict__ res,
                                                                        const bf16* __restrict__ w,
                                                                        bf16* __restrict__ out,
                                                                        const int32_t* __restrict__ rows, int hidden,
                                                                        int64_t tokens, float eps, int mode, float post_scale) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int64_t row = (int64_t)blockIdx.x * kNormWarps + warp;
  if (row >= tokens) return;
  const int64_t src_row = rows ? rows[row] : row;
  const uint4* xr = reinterpret_cast<const uint4*>(x + src_row * hidden);
  uint4* rr = FUSED_ADD ? reinterpret_cast<uint4*>(res + row * hidden) : nullptr;
  const uint4* wv = reinterpret_cast<const uint4*>(w);
  uint4* orow = reinterpret_cast<uint4*>(out + row * hidden);
  const int nvec = hidden / 8;

  uint4 xa[VPL];
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < VPL; ++j) {
    const int i = lane + j * 32;
    if (i < nvec) {
      xa[j] = xr[i];
      __nv_bfloat162* xp = reinterpret_cast<__nv_bfloat162*>(&xa[j]);
      if (FUSED_ADD) {
        uint4 ra = rr[i];
        const __nv_bfloat162* rp = reinterpret_cast<const __nv_bfloat162*>(&ra);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          float2 f = __bfloat1622float2(xp[c]);
          const float2 g = __bfloat1622float2(rp[c]);
          f.x += g.x; f.y += g.y;
          xp[c] = __float22bfloat162_rn(f);
        }
        rr[i] = xa[j];  // 舍入后的残差写回；下面平方和 / 归一化都用舍入后的值（与 block 版逐位一致）
      }
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float2 f = __bfloat1622float2(xp[c]);
        ss += f.x * f.x + f.y * f.y;
      }
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float inv = rms_inv(ss, hidden, eps, mode);
#pragma unroll
  for (int j = 0; j < VPL; ++j) {
    const int i = lane + j * 32;
    if (i < nvec) {
      const uint4 wa = w ? wv[i] : make_uint4(0x3F803F80u, 0x3F803F80u, 0x3F803F80u, 0x3F803F80u);
      const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wa);
      const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xa[j]);
      uint4 ob;
      __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&ob);
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float2 f = __bfloat1622float2(xp[c]);
        const float2 g = __bfloat1622float2(wp[c]);
        op[c] = __floats2bfloat162_rn(norm_mid(f.x * inv, mode) * g.x, norm_mid(f.y * inv, mode) * g.y);
      }
      orow[i] = NORM_ADD ? (post_scale != 1.f ? bf16x8_scale(bf16x8_add(orow[i], ob), post_scale) : bf16x8_add(orow[i], ob)) : ob;
    }
  }
}

// 标量回退：hidden 不是 8 的倍数或 > 8192 时用（与旧实现相同）。
template <int BLOCK, bool FUSED_ADD, bool NORM_ADD>
__global__ void rmsnorm_kernel(const bf16* __restrict__ x, bf16* __restrict__ res, const bf16* __restrict__ w,
                               bf16* __restrict__ out, const int32_t* __restrict__ rows, int hidden, float eps,
                               int mode, float post_scale) {
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
  const float inv = rms_inv(ss, hidden, eps, mode);

  // 第二遍：归一化并乘权重。
  const bf16* src = FUSED_ADD ? rr : xr;
  for (int i = threadIdx.x; i < hidden; i += BLOCK) {
    const bf16 nv = to_bf16(norm_mid(to_f32(src[i]) * inv, mode) * (w ? to_f32(w[i]) : 1.f));
    orow[i] = NORM_ADD ? (post_scale != 1.f ? to_bf16(to_f32(to_bf16(to_f32(orow[i]) + to_f32(nv))) * post_scale) : to_bf16(to_f32(orow[i]) + to_f32(nv))) : nv;
  }
}

// ---------------------------------------------------------------------------------------------
// norm → fp8 融合（vLLM **compile 默认口径**，`docs/loops/kaon3-parity.md` 第 20 轮）
//
// vLLM 不开 `enforce_eager` 时（默认就是 compile），inductor 把 RMSNorm 与**下游 fp8 量化**融进同一个
// Triton kernel：`x·rsqrt` → 乘 w → `×(1/scale)` → `clamp(±448)` → `to(fp8)`，**中间没有 bf16 往返**。
// eager 的 BI 路径则是 norm 输出 bf16，再由独立的 quant kernel 读。
// 差别实测（[24,2816]，同一 a_scale）：**668/67584 个 fp8 元素不同，max = 8.0**（正好一个 fp8 步长）。
// 这正是端到端 |Δlp| 的主要来源：改 inv 的算法（rsqrt vs rcp·sqrt）在 bf16 输出下**逐位无差异**。
//
// 所以这里必须把两步合成一个 kernel 才能对上 compile。`w` 仍是 bf16（vLLM 也按 f32 乘）。
// ---------------------------------------------------------------------------------------------
template <int BLOCK, int VPT>
__global__ void __launch_bounds__(BLOCK) rmsnorm_quant_kernel(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                                             uint8_t* __restrict__ out, int hidden, float eps,
                                                             float inv_scale) {
  const int64_t row = blockIdx.x;
  const uint4* xr = reinterpret_cast<const uint4*>(x + row * hidden);
  const uint4* wv = reinterpret_cast<const uint4*>(w);
  uint8_t* orow = out + row * hidden;
  const int nvec = hidden / 8;

  float v[VPT][8];
  float gs[VPT];
#pragma unroll
  for (int j = 0; j < VPT; ++j) {
    const int i = threadIdx.x + j * BLOCK;
    gs[j] = 0.f;
    if (i < nvec) {
      const uint4 xa = xr[i];
      const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xa);
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float2 f = __bfloat1622float2(xp[c]);
        v[j][2 * c] = f.x; v[j][2 * c + 1] = f.y;
      }
      float acc = __fmul_rn(v[j][1], v[j][1]);
      acc = __fmaf_rn(v[j][0], v[j][0], acc);
#pragma unroll
      for (int e = 2; e < 8; ++e) acc = __fmaf_rn(v[j][e], v[j][e], acc);
      gs[j] = acc;
    }
  }
  const float inv = rms_inv_compile(rms_ss_triton<BLOCK, VPT>(gs, nvec), hidden, eps);
#pragma unroll
  for (int j = 0; j < VPT; ++j) {
    const int i = threadIdx.x + j * BLOCK;
    if (i < nvec) {
      const uint4 wb = wv[i];
      const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wb);
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float2 g = __bfloat1622float2(wp[c]);
        // fp32 一路算到 fp8：不做 bf16 往返（这正是与 eager 的差别所在）
        const float y0 = v[j][2 * c] * inv * g.x * inv_scale;
        const float y1 = v[j][2 * c + 1] * inv * g.y * inv_scale;
        orow[i * 8 + 2 * c] =
            static_cast<uint8_t>(__nv_cvt_float_to_fp8(y0, __NV_SATFINITE, __NV_E4M3));
        orow[i * 8 + 2 * c + 1] =
            static_cast<uint8_t>(__nv_cvt_float_to_fp8(y1, __NV_SATFINITE, __NV_E4M3));
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// router 预处理的融合（vLLM **compile 口径**，`docs/loops/kaon3-parity.md` 第 23 轮）
//
// FX 图里是 `rms_norm(x, None) → × root_size → × scale`（`computation_graph.py:1400-1406`，三步都标 bf16），
// 但实测 inductor 把这三次乘法**融成一条 fp32 链、只在写回时舍入一次**：
//   (y·rs)·sc 两步 bf16 → 17299/67584 不同；`y.float()·rs·sc → bf16` → **0 不同**。
// 我们原来分 `rmsnorm_noweight` + `scale_bf16_n` + `mul_bf16_n` 三步做，中间多两次 bf16 舍入。
// 这里把三件事合成一个 kernel：无权重 norm 的 inv 用 compile 口径，输出在 fp32 里连乘 rs、scale 后舍入一次。
template <int BLOCK, int VPT>
__global__ void __launch_bounds__(BLOCK) rmsnorm_scale2_kernel(const bf16* __restrict__ x,
                                                              const bf16* __restrict__ scale,
                                                              bf16* __restrict__ out, int hidden, float eps,
                                                              float rs, int mode) {
  const int64_t row = blockIdx.x;
  const uint4* xr = reinterpret_cast<const uint4*>(x + row * hidden);
  const uint4* sv = reinterpret_cast<const uint4*>(scale);
  uint4* orow = reinterpret_cast<uint4*>(out + row * hidden);
  const int nvec = hidden / 8;

  float v[VPT][8];
  float gs[VPT];
#pragma unroll
  for (int j = 0; j < VPT; ++j) {
    const int i = threadIdx.x + j * BLOCK;
    gs[j] = 0.f;
    if (i < nvec) {
      const uint4 xa = xr[i];
      const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xa);
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float2 f = __bfloat1622float2(xp[c]);
        v[j][2 * c] = f.x; v[j][2 * c + 1] = f.y;
      }
      float acc = __fmul_rn(v[j][1], v[j][1]);
      acc = __fmaf_rn(v[j][0], v[j][0], acc);
#pragma unroll
      for (int e = 2; e < 8; ++e) acc = __fmaf_rn(v[j][e], v[j][e], acc);
      gs[j] = acc;
    }
  }
  const float inv = rms_inv(rms_ss_triton<BLOCK, VPT>(gs, nvec), hidden, eps, mode);  // B30b：mode 1 走 BI 口径的 inv
#pragma unroll
  for (int j = 0; j < VPT; ++j) {
    const int i = threadIdx.x + j * BLOCK;
    if (i < nvec) {
      const uint4 sb = sv[i];
      const __nv_bfloat162* sp = reinterpret_cast<const __nv_bfloat162*>(&sb);
      uint4 ob;
      __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&ob);
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float2 g = __bfloat1622float2(sp[c]);
        // mode 0：fp32 全程 norm → ×rs → ×scale，只在写回时舍一次。
        // mode 2：照 FX 图的三次 bf16 舍入（norm 出口、×rs 后、×scale 后）。
        // mode 1（B30b）：复刻 rmsnorm_noweight → scale_bf16 → mul_row 三次 launch 的三个舍入点：bf16(x·inv) → bf16(·rs) → bf16(·g)。
        const int mm = mode == 1 ? 2 : mode;
        const float y0 = norm_mid(norm_mid(v[j][2 * c] * inv, mm) * rs, mm) * g.x;
        const float y1 = norm_mid(norm_mid(v[j][2 * c + 1] * inv, mm) * rs, mm) * g.y;
        op[c] = __floats2bfloat162_rn(y0, y1);
      }
      orow[i] = ob;
    }
  }
}

// layout（host 给，`ArchProfile::rmsnorm_layout`）：0 一 block(256 线程) 一行；1 一 warp 一行（hidden ≤ 4096，
// 否则退回 0）；2 一 block(1024 线程) 一行（hidden ≤ 8192）。hidden 不对齐 / > 8192 一律标量版。
template <bool FUSED_ADD, bool NORM_ADD = false>
int launch(const void* x, void* res, const void* w, void* out, const int32_t* rows, int64_t tokens, int64_t hidden,
           float eps, int layout, int mode, cudaStream_t s, float post_scale = 1.f) {
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
  if (aligned && layout == 1 && hidden <= 4096) {
    const dim3 wgrid((unsigned)((tokens + kNormWarps - 1) / kNormWarps));
    const int h = (int)hidden;
    if (hidden <= 256) {
      rmsnorm_warp_kernel<1, FUSED_ADD, NORM_ADD><<<wgrid, kNormWarps * 32, 0, s>>>(xp, rp, wp, op, rows, h, tokens, eps, mode, post_scale);
    } else if (hidden <= 1024) {
      rmsnorm_warp_kernel<4, FUSED_ADD, NORM_ADD><<<wgrid, kNormWarps * 32, 0, s>>>(xp, rp, wp, op, rows, h, tokens, eps, mode, post_scale);
    } else if (hidden <= 2048) {
      rmsnorm_warp_kernel<8, FUSED_ADD, NORM_ADD><<<wgrid, kNormWarps * 32, 0, s>>>(xp, rp, wp, op, rows, h, tokens, eps, mode, post_scale);
    } else {
      rmsnorm_warp_kernel<16, FUSED_ADD, NORM_ADD><<<wgrid, kNormWarps * 32, 0, s>>>(xp, rp, wp, op, rows, h, tokens, eps, mode, post_scale);
    }
    return last_err();
  }
  if (aligned && layout == 2 && hidden <= 8192) {
    if (hidden <= 1024) {
      rmsnorm_vec_kernel<128, 1, FUSED_ADD, NORM_ADD><<<grid, 128, 0, s>>>(xp, rp, wp, op, rows, (int)hidden, eps, mode, post_scale);
    } else {
      // 1024 线程 × 1 向量（hidden=8192 正好满），归约两级各一次 shuffle
      rmsnorm_vec_kernel<1024, 1, FUSED_ADD, NORM_ADD><<<grid, 1024, 0, s>>>(xp, rp, wp, op, rows, (int)hidden, eps, mode, post_scale);
    }
    return last_err();
  }
  if (aligned && hidden <= 1024) {
    // ≤ 1024：128 线程 × 1 向量（hidden=1024 正好满），一次访存
    rmsnorm_vec_kernel<128, 1, FUSED_ADD, NORM_ADD><<<grid, 128, 0, s>>>(xp, rp, wp, op, rows, (int)hidden, eps, mode, post_scale);
  } else if (aligned && hidden <= 4096) {
    rmsnorm_vec_kernel<256, 2, FUSED_ADD, NORM_ADD><<<grid, 256, 0, s>>>(xp, rp, wp, op, rows, (int)hidden, eps, mode, post_scale);
  } else if (aligned && hidden <= 8192) {
    rmsnorm_vec_kernel<256, 4, FUSED_ADD, NORM_ADD><<<grid, 256, 0, s>>>(xp, rp, wp, op, rows, (int)hidden, eps, mode, post_scale);
  } else if (hidden <= 1024) {
    rmsnorm_kernel<256, FUSED_ADD, NORM_ADD><<<grid, 256, 0, s>>>(xp, rp, wp, op, rows, (int)hidden, eps, mode, post_scale);
  } else {
    rmsnorm_kernel<1024, FUSED_ADD, NORM_ADD><<<grid, 1024, 0, s>>>(xp, rp, wp, op, rows, (int)hidden, eps, mode, post_scale);
  }
  return last_err();
}

// norm → fp8（compile 口径）。`a_scale` 是静态激活 scale：输出 = fp8(norm(x) · (1/a_scale))。
template <int BLOCK, int VPT>
static int launch_quant(const void* x, const void* w, void* out, int64_t tokens, int64_t hidden, float eps, float a_scale,
                        cudaStream_t s) {
  if (!(a_scale > 0.f)) return err(cudaErrorInvalidValue);
  const dim3 grid((unsigned)tokens);
  rmsnorm_quant_kernel<BLOCK, VPT><<<grid, BLOCK, 0, s>>>(static_cast<const bf16*>(x), static_cast<const bf16*>(w),
                                                          static_cast<uint8_t*>(out), (int)hidden, eps, 1.f / a_scale);
  return last_err();
}


// router 预处理：`out = bf16(fp32(norm(x, 无权重) · rs · scale))`。
template <int BLOCK, int VPT>
static int launch_scale2(const void* x, const void* scale, void* out, int64_t tokens, int64_t hidden, float eps, float rs,
                         int mode, cudaStream_t s) {
  rmsnorm_scale2_kernel<BLOCK, VPT><<<(unsigned)tokens, BLOCK, 0, s>>>(
      static_cast<const bf16*>(x), static_cast<const bf16*>(scale), static_cast<bf16*>(out), (int)hidden, eps, rs, mode);
  return last_err();
}



// ---- B23（loop kaon3-decode-opt D148）：attention 前段融合 ----
// 一个 token 一个 block，一个 warp 一个头行：逐头 RMSNorm（q/k 带权重、v 无权重）→ 原地 RoPE（q/k，neox，span 配对）
// → q/k/v fp8 往返（原地写回）→ k/v 写 paged cache（e4m3 或 bf16）。替代 head_rmsnorm×3 + rope + cache_write_fp8rt
// 共 5 次 launch。每一段都**逐位复刻**原 kernel：
//  - norm = `rmsnorm_vec_kernel<128,1>`（Sm120 Block256 布局下 hidden ≤ 1024 的实例）：原 kernel 一行一 block 128 线程、
//    线程 i 持向量 i，warp w 的 32 个向量先 xor 蝶形成 seg[w]，再由线程 0 按 4 段一组 `fadd(tot, fadd(fadd(a0,a2),fadd(a1,a3)))`
//    累加。这里一 warp 一行，lane 持向量 lane + 32·j（j < D/256），同样的 32 个值、同样的蝶形序 → seg[j] 逐位同，再按同式合并。
//  - rope = `rope_kernel<true>`（每步舍 bf16），fp8 往返 = `fp8_rt`，cache 布局 = `slot_off`（gqa.cu）。
//  warp 之间完全独立：无 __syncthreads；smem 每 warp 一行 bf16（≤ 1 KB）。
__device__ __forceinline__ bf16 fp8_rt_n(bf16 x) {
  const __nv_fp8_storage_t q8 = __nv_cvt_float_to_fp8(to_f32(x), __NV_SATFINITE, __NV_E4M3);
  return to_bf16(__half2float(__half(__nv_cvt_fp8_to_halfraw(q8, __NV_E4M3))));
}
template <int VP, bool STORE_FP8>
__global__ void __launch_bounds__(1024) attn_pre_fused_kernel(bf16* __restrict__ q, bf16* __restrict__ k, bf16* __restrict__ v,
                                                              const bf16* __restrict__ qw, const bf16* __restrict__ kw,
                                                              const int32_t* __restrict__ pos, const float* __restrict__ cos_sin,
                                                              bf16* __restrict__ cache, const int64_t* __restrict__ slot_mapping,
                                                              int q_heads, int kv_heads, int rope_dim, int span, int block_size,
                                                              float eps, int mode) {
  constexpr int D = 256 * VP;
  extern __shared__ __align__(16) uint8_t pre_smem[];
  const int64_t t = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nrows = q_heads + 2 * kv_heads;
  if (warp >= nrows) return;
  int kind, h;
  bf16* row;
  const bf16* w;
  if (warp < q_heads) { kind = 0; h = warp; row = q + (t * q_heads + h) * D; w = qw; }
  else if (warp < q_heads + kv_heads) { kind = 1; h = warp - q_heads; row = k + (t * kv_heads + h) * D; w = kw; }
  else { kind = 2; h = warp - q_heads - kv_heads; row = v + (t * kv_heads + h) * D; w = nullptr; }
  bf16* srow = reinterpret_cast<bf16*>(pre_smem) + (size_t)warp * D;
  // ---- 1. norm（复刻 rmsnorm_vec_kernel<128,1>）----
  const uint4* xr = reinterpret_cast<const uint4*>(row);
  const uint4* wv = reinterpret_cast<const uint4*>(w);
  float vv[VP][8];
  uint4 wa[VP];
  float gs[VP];
#pragma unroll
  for (int j = 0; j < VP; ++j) {
    const int i = lane + 32 * j;
    uint4 xa = xr[i];
    wa[j] = w ? wv[i] : make_uint4(0x3F803F80u, 0x3F803F80u, 0x3F803F80u, 0x3F803F80u);
    const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xa);
#pragma unroll
    for (int c = 0; c < 4; ++c) { const float2 f = __bfloat1622float2(xp[c]); vv[j][2 * c] = f.x; vv[j][2 * c + 1] = f.y; }
    float acc = __fmul_rn(vv[j][1], vv[j][1]);
    acc = __fmaf_rn(vv[j][0], vv[j][0], acc);
#pragma unroll
    for (int e = 2; e < 8; ++e) acc = __fmaf_rn(vv[j][e], vv[j][e], acc);
    gs[j] = acc;
  }
  float seg[VP];
#pragma unroll
  for (int j = 0; j < VP; ++j) {
    float g = gs[j];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) g = __fadd_rn(g, __shfl_xor_sync(0xffffffffu, g, o));
    seg[j] = g;
  }
  float tot = 0.f;
#pragma unroll
  for (int c = 0; 4 * c < VP; ++c) {
    float a[4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) a[kk] = (4 * c + kk < VP) ? seg[4 * c + kk] : 0.f;
    tot = __fadd_rn(tot, __fadd_rn(__fadd_rn(a[0], a[2]), __fadd_rn(a[1], a[3])));
  }
  const float inv = rms_inv(tot, D, eps, mode);
  uint4* so = reinterpret_cast<uint4*>(srow);
#pragma unroll
  for (int j = 0; j < VP; ++j) {
    const int i = lane + 32 * j;
    const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wa[j]);
    uint4 ob;
    __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&ob);
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const float2 g = __bfloat1622float2(wp[c]);
      op[c] = __floats2bfloat162_rn(norm_mid(vv[j][2 * c] * inv, mode) * g.x, norm_mid(vv[j][2 * c + 1] * inv, mode) * g.y);
    }
    so[i] = ob;
  }
  __syncwarp();
  // ---- 2. rope（q/k；复刻 rope_kernel<true>：每步舍 bf16）----
  if (kind < 2) {
    const int half = rope_dim / 2;
    const float* cs = cos_sin + (int64_t)pos[t] * rope_dim;
    for (int i = lane; i < half; i += 32) {
      const float c = cs[i], s = cs[half + i];
      const int a = i, b = i + span;
      const float x0 = to_f32(srow[a]), x1 = to_f32(srow[b]);
      const float p0 = to_f32(to_bf16(x0 * c)), p1 = to_f32(to_bf16(x1 * s));
      const float p2 = to_f32(to_bf16(x1 * c)), p3 = to_f32(to_bf16(x0 * s));
      srow[a] = to_bf16(p0 - p1);
      srow[b] = to_bf16(p2 + p3);
    }
    __syncwarp();
  }
  // ---- 3. fp8 往返写回 + k/v 写 cache（复刻 gqa_cache_write_fp8rt_kernel）----
  const int64_t slot = slot_mapping[t];
  bf16* dst = nullptr;
  uint8_t* dst8 = nullptr;
  if (kind > 0 && slot >= 0) {
    const int64_t e0 = slot * (int64_t)kv_heads * 2 * D + (int64_t)(h * 2 + (kind - 1)) * D;
    dst = cache + e0;
    dst8 = reinterpret_cast<uint8_t*>(cache) + e0;
  }
  for (int d = lane; d < D; d += 32) {
    const bf16 r = fp8_rt_n(srow[d]);
    row[d] = r;
    if (dst8) {
      if (STORE_FP8) dst8[d] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(to_f32(r), __NV_SATFINITE, __NV_E4M3));
      else dst[d] = r;
    }
  }
}
}  // namespace

// B13：out_res[r] = bf16(out_res[r] + bf16(rmsnorm(x[r])·w))——norm 后直接加到残差流（代 rmsnorm + add_inplace 两次 launch）。
extern "C" int rmsnorm_add_bf16(const void* x, const void* w, void* out_res, int64_t tokens, int64_t hidden, float eps,
                                int layout, int mode, cudaStream_t stream) {
  return launch<false, true>(x, nullptr, w, out_res, nullptr, tokens, hidden, eps, layout, mode, stream);
}
extern "C" int rmsnorm_bf16(const void* x, const void* w, void* out, int64_t tokens, int64_t hidden, float eps,
                            int layout, int mode, cudaStream_t stream) {
  return launch<false>(x, nullptr, w, out, nullptr, tokens, hidden, eps, layout, mode, stream);
}

// out[r] = rmsnorm(x[rows[r]]) · w，r < num_rows（替代 gather_rows + rmsnorm 两次 launch）
extern "C" int gather_rmsnorm_bf16(const void* x, const int32_t* rows, const void* w, void* out, int64_t num_rows,
                                   int64_t hidden, float eps, int layout, int mode, cudaStream_t stream) {
  return launch<false>(x, nullptr, w, out, rows, num_rows, hidden, eps, layout, mode, stream);
}

// B30a（loop kaon3-decode-opt D149）：层尾三合一。res[i] = bf16(res[i] + x[i])（写回 res）；out[i] = bf16(bf16(out[i] + bf16(rmsnorm(res)·w)) · s)。
// 与 add_inplace(x→res) → rmsnorm_add(res, w → out) → scale(out, s) 三次 launch 逐位同（各自的舍入点一个不少）。
extern "C" int fused_add_rmsnorm_add_scale_bf16(const void* x, void* res, const void* w, void* out_res, int64_t tokens,
                                                int64_t hidden, float eps, float post_scale, int layout, int mode,
                                                cudaStream_t stream) {
  return launch<true, true>(x, res, w, out_res, nullptr, tokens, hidden, eps, layout, mode, stream, post_scale);
}
extern "C" int fused_add_rmsnorm_bf16(const void* x, void* res, const void* w, void* out, int64_t tokens,
                                      int64_t hidden, float eps, int layout, int mode, cudaStream_t stream) {
  return launch<true>(x, res, w, out, nullptr, tokens, hidden, eps, layout, mode, stream);
}

// norm + 量化融合：`out[i] = fp8(norm(x)[i] * (1/a_scale))`，out 是 e4m3 字节。
// 与 vLLM compile（inductor 融合 norm+quant）同口径；eager BI 口径请用 rmsnorm_bf16 + quant_fp8_bf16 两步。
extern "C" int rmsnorm_quant_fp8_bf16(const void* x, const void* w, void* out, int64_t tokens, int64_t hidden, float eps,
                                      float a_scale, cudaStream_t stream) {
  if (tokens <= 0) return 0;
  if (hidden <= 0 || hidden > 8192 || (hidden % 8) != 0) return err(cudaErrorInvalidValue);
  if (hidden <= 1024) return launch_quant<128, 1>(x, w, out, tokens, hidden, eps, a_scale, stream);
  if (hidden <= 2048) return launch_quant<256, 1>(x, w, out, tokens, hidden, eps, a_scale, stream);
  if (hidden <= 4096) return launch_quant<256, 2>(x, w, out, tokens, hidden, eps, a_scale, stream);
  return launch_quant<256, 4>(x, w, out, tokens, hidden, eps, a_scale, stream);
}

// router 预处理融合（vLLM compile 口径）：`out = bf16(norm(x) · rs · scale)`，norm 无权重。
// 与 `rmsnorm_noweight` + `scale_bf16_n` + `mul_bf16_n` 三步的差别仅在 bf16 舍入次数（这里只舍一次）。
extern "C" int rmsnorm_scale2_bf16(const void* x, const void* scale, void* out, int64_t tokens, int64_t hidden,
                                   float eps, float rs, int mode, cudaStream_t stream) {
  if (tokens <= 0) return 0;
  if (hidden <= 0 || hidden > 8192 || (hidden % 8) != 0) return err(cudaErrorInvalidValue);
  if (mode != 0 && mode != 2) {
    // eager 口径（1）没有实测过这条融合，退回三步由调用方负责 → 这里直接不支持
    return err(cudaErrorInvalidValue);
  }
  if (hidden <= 1024) return launch_scale2<128, 1>(x, scale, out, tokens, hidden, eps, rs, mode, stream);
  if (hidden <= 2048) return launch_scale2<256, 1>(x, scale, out, tokens, hidden, eps, rs, mode, stream);
  if (hidden <= 4096) return launch_scale2<256, 2>(x, scale, out, tokens, hidden, eps, rs, mode, stream);
  return launch_scale2<256, 4>(x, scale, out, tokens, hidden, eps, rs, mode, stream);
}

// B23：见 attn_pre_fused_kernel。head_dim ∈ {256, 512}；rope 为 neox 且 span + rope_dim/2 ≤ head_dim；block = (q+2kv) 个 warp ≤ 32。
extern "C" int attn_pre_fused_bf16(void* q, void* k, void* v, const void* q_w, const void* k_w, const int32_t* positions,
                                   const float* cos_sin, void* cache, const int64_t* slot_mapping, int64_t tokens,
                                   int64_t q_heads, int64_t kv_heads, int64_t head_dim, int64_t rope_dim, int64_t span,
                                   int64_t block_size, float eps, int mode, int kv_fp8, cudaStream_t stream) {
  if (tokens <= 0) return 0;
  if (!(head_dim == 256 || head_dim == 512) || q_heads <= 0 || kv_heads <= 0 || q_heads + 2 * kv_heads > 32 ||
      rope_dim <= 0 || (rope_dim & 1) || rope_dim > head_dim || span < rope_dim / 2 || span + rope_dim / 2 > head_dim ||
      block_size <= 0 || !q_w || !k_w)
    return err(cudaErrorInvalidValue);
  const int nrows = (int)(q_heads + 2 * kv_heads);
  const size_t smem = (size_t)nrows * head_dim * sizeof(bf16);
  const dim3 grid((unsigned)tokens), block((unsigned)(nrows * 32));
#define PRE_LAUNCH(VP, F8) \
  attn_pre_fused_kernel<VP, F8><<<grid, block, smem, stream>>>( \
      static_cast<bf16*>(q), static_cast<bf16*>(k), static_cast<bf16*>(v), static_cast<const bf16*>(q_w), \
      static_cast<const bf16*>(k_w), positions, cos_sin, static_cast<bf16*>(cache), slot_mapping, (int)q_heads, (int)kv_heads, \
      (int)rope_dim, (int)span, (int)block_size, eps, mode)
  if (head_dim == 256) { if (kv_fp8) PRE_LAUNCH(1, true); else PRE_LAUNCH(1, false); }
  else { if (kv_fp8) PRE_LAUNCH(2, true); else PRE_LAUNCH(2, false); }
#undef PRE_LAUNCH
  return last_err();
}
