#pragma once
// RMSNorm 的数值核（从 norm.cu 抽出，供 norm.cu / moe_grouped_sm120.cu（B34：norm+grouped quant 融合）共用；
// 任何改动两处同时生效，逐位口径只有这一份）。
#include <cuda_bf16.h>
#include <cuda_runtime.h>

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

