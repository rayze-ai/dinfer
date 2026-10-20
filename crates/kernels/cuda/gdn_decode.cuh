// GDN decode 递归的共享件：gate / 激活小函数、warp 多值归约、寄存器驻留 tile 递推（decode_tile_body）。
// 被 gdn.cu（独立 recurrent_decode_kernel）与 mega_sm100.cu（单层 persistent kernel 的 GDN 阶段）共用，
// 两处行为必须一致，所以只此一份。匿名命名空间：每个 TU 自己一份实例，不导出符号。
#pragma once
#include "common.cuh"

namespace dinfer {
namespace {

__device__ __forceinline__ float sigmoid_f(float x) { return 1.f / (1.f + __expf(-x)); }

// softplus，阈值 20（fp32 下 exp(20) 还没溢出，但与 Triton 参考保持同一分支以免对拍漂移）
__device__ __forceinline__ float softplus_f(float x) {
  return x > 20.f ? x : __logf(1.f + __expf(x));
}

__device__ __forceinline__ float act_f(float x, int act) {  // 0 = silu, 1 = sigmoid
  return act == 1 ? sigmoid_f(x) : x * sigmoid_f(x);
}

// ---------------------------------------------------------------- 递归核心：decode 专用（寄存器版）
// 老 recurrent_kernel 把整个 [V, K] 状态装进 shared，再按 v 行逐线程串行扫 K 维两趟：decode（len=1）时
// 整个 kernel 只是"读一遍状态 + 两个 K 维点积 + 写一遍状态"，却被 block 归约 / 同步和串行 K 循环拖成
// 延迟绑定，B=32 时不随 batch 扩展（docs/loops/qwen35-perf.md 第 3g 行）。本版：
//   - 每 block 一个 (序列, v 头, V 块 BV)；kDecodeWarps 个 warp，每 warp 负责 ROWS = BV/kDecodeWarps 行状态；
//   - 每行 [K] f32 按 lane 切成 KPL = K/32 个连续元素（KPL=4 → 16 B 向量读写），全程在寄存器，不经 shared；
//   - K 维点积 = 每 lane KPL 次乘加 + warp shuffle 归约；q/k 的 l2 归一化每 warp 重复算（K 维一次 warp 归约，
//     远比 block 级 __syncthreads 便宜），没有任何 block 级同步；
//   - 支持任意 len（逐 token 循环，状态留在寄存器），所以 host 把 prefill 尾段误送进来也不影响正确性。
// BV 由 host 给（`ArchProfile::gdn_decode_bv`），kDecodeWarps 是结构常量（不是按卡调的数值）。
//
// 第 7 轮（2026-10-20）两个可选变体，模板开关、host 由 `decode_variant` 选（`ArchProfile::gdn_decode_variant`，
// `GDN_DECODE_VARIANT=base|tr|pipe|pipe_base` 覆盖）：
//   TR（转置归约）：每 token 只做两次"多值 warp 归约"（蝶式折半：N 个部分和 → log2 N 轮共 ~N 次 shuffle，
//     而不是 N 次 × 5 shuffle）。第一次归约 {q·q, k·k, k·q}；第二次把 ROWS 行的 {S·k, S·q} 一次归约（2·ROWS ≤ 32 个值），
//     行 i 的结果落在 lane i·S（p）和 lane 16+i·S（o₀）。数学上用 S_new = eg·S_old + v'·k 展开：
//     p = eg·(S_old·k)，o = eg·(S_old·q) + v'·(k·q)，S 只扫一遍（一次 FMA 遍）。shuffle 数 ROWS=16 时 170 → ~57。
//   PIPE（持久 block + cp.async 流水）：grid 只铺一波（occupancy × SM 数），每 block 顺序吃多个 (seq, v 头, V 块)
//     tile；算当前 tile 时下一 tile 的状态已在用 cp.async 进 smem（单缓冲：每 lane 先把自己那 16 B 取进寄存器，
//     再对同一位置发下一 tile 的 cp.async，不需要 block 同步）。目的：B300 这类 HBM 延迟/带宽比高的卡上，
//     "读 → 算 → 写"锁步的 block 波之间 HBM 空转，流水把下一块的读藏在当前块的算 + 写后面。
constexpr int kDecodeWarps = 4;

template <int KPL> struct DecodeVec;
template <> struct DecodeVec<4> { using type = float4; };
template <> struct DecodeVec<2> { using type = float2; };

__device__ __forceinline__ float warp_sum_f(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// 多值 warp 归约（蝶式折半）：输入每 lane 持 N 个部分和 v[0..N)（N 为 2 的幂，≤ 32），
// 返回后 lane L 持有第 (L / (32/N)) 个值的全和（每个值由 32/N 个连续 lane 冗余持有）。
// 共 N-1 + log2(32/N) 次 shuffle（N=32：31；N=16：19；N=4：6），远少于 N 次独立 warp_sum 的 5N。
// 全部用模板递归展开：数组下标都是编译期常量，v[] 留在寄存器（循环变量版会把 v[] 压进 local memory）。
template <int N, int HALF>
struct WarpSumMulti {
  static __device__ __forceinline__ float run(float (&v)[N], int lane) {
    constexpr int HN = N / 2;
    const bool upper = (lane & HALF) != 0;
    float w[HN];
#pragma unroll
    for (int i = 0; i < HN; ++i) {
      const float keep = upper ? v[i + HN] : v[i];
      const float send = upper ? v[i] : v[i + HN];
      w[i] = keep + __shfl_xor_sync(0xffffffffu, send, HALF);
    }
    return WarpSumMulti<HN, HALF / 2>::run(w, lane);
  }
};
template <int HALF>
struct WarpSumMulti<1, HALF> {
  static __device__ __forceinline__ float run(float (&v)[1], int) {
    float r = v[0];
#pragma unroll
    for (int o = HALF; o > 0; o >>= 1) r += __shfl_xor_sync(0xffffffffu, r, o);
    return r;
  }
};
template <int N>
__device__ __forceinline__ float warp_sum_multi(float (&v)[N], int lane) {
  static_assert(N >= 1 && N <= 32 && (N & (N - 1)) == 0, "N 须为 2 的幂且 ≤ 32");
  return WarpSumMulti<N, 16>::run(v, lane);
}

template <int KPL>
__device__ __forceinline__ void vec_load(float (&r)[KPL], const float* p) {
  using Vec = typename DecodeVec<KPL>::type;
  const Vec v = *reinterpret_cast<const Vec*>(p);
  if constexpr (KPL == 4) { r[0] = v.x; r[1] = v.y; r[2] = v.z; r[3] = v.w; }
  else { r[0] = v.x; r[1] = v.y; }
}

template <int KPL>
__device__ __forceinline__ void vec_store(float* p, const float (&r)[KPL]) {
  using Vec = typename DecodeVec<KPL>::type;
  Vec v;
  if constexpr (KPL == 4) { v.x = r[0]; v.y = r[1]; v.z = r[2]; v.w = r[3]; }
  else { v.x = r[0]; v.y = r[1]; }
  *reinterpret_cast<Vec*>(p) = v;
}

__device__ __forceinline__ void cp_async16_smem(void* smem, const void* gmem) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit_g() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all_g() { asm volatile("cp.async.wait_group 0;\n" ::); }

struct DecodeArgs {
  const bf16* mixed;
  const float* g_in;
  const float* beta_in;
  const bf16* a_raw;
  const bf16* b_raw;
  const float* a_log;
  const float* dt_bias;
  bf16* out;
  float* state;
  const int* cu_seqlens;
  const int* state_indices;
  int H, HV, K, V, group;
  float scale;
  long state_stride_tok;
  long ab_stride;  // a_raw/b_raw 的行距（独立缓冲 = HV；融合 in_proj 列段 = in_proj 行宽）
};

// 一个 warp 对一个 (seq, v 头, ROWS 行) 的全部 token 递推；状态 r 进来是旧值、出去是新值（调用方负责装 / 存）。
template <int ROWS, int KPL, bool FUSE_GATE, bool TR>
__device__ __forceinline__ void decode_tile_body(const DecodeArgs& a, float (&r)[ROWS][KPL], int seq, int hv,
                                                 int v0, int bos, int len, int lane) {
  // TR：2·ROWS 个值一次归约后，每个值由 S = 32/(2·ROWS) 个连续 lane 持有；lane < 16 持 p(行 my)，lane ≥ 16 持 o₀(行 my)
  constexpr int S = 16 / ROWS;
  static_assert(!TR || (2 * ROWS <= 32 && 16 % ROWS == 0), "TR 要求 ROWS ∈ {4, 8, 16}");
  const int h = hv / a.group;
  const long qkv_row = 2L * a.H * a.K + (long)a.HV * a.V;
  const bf16* qp = a.mixed + (long)h * a.K + lane * KPL;
  const bf16* kp = a.mixed + (long)a.H * a.K + (long)h * a.K + lane * KPL;
  const bf16* vp = a.mixed + 2L * a.H * a.K + (long)hv * a.V + v0;
  const int my = TR ? (lane & 15) / S : 0;  // TR：本 lane 对应的行号

  for (int t = 0; t < len; ++t) {
    const long row = bos + t;
    float q[KPL], k[KPL];
    float qsq = 0.f, ksq = 0.f, kq = 0.f;
#pragma unroll
    for (int j = 0; j < KPL; ++j) {
      q[j] = to_f32(qp[row * qkv_row + j]);
      k[j] = to_f32(kp[row * qkv_row + j]);
      qsq += q[j] * q[j];
      ksq += k[j] * k[j];
      kq += k[j] * q[j];
    }
    if (TR) {
      float s3[4] = {qsq, ksq, kq, 0.f};
      const float mine = warp_sum_multi<4>(s3, lane);  // lane 0-7 qsq，8-15 ksq，16-23 kq
      qsq = __shfl_sync(0xffffffffu, mine, 0);
      ksq = __shfl_sync(0xffffffffu, mine, 8);
      kq = __shfl_sync(0xffffffffu, mine, 16);
    } else {
      qsq = warp_sum_f(qsq);
      ksq = warp_sum_f(ksq);
    }
    const float qn = 1.f / sqrtf(qsq + 1e-6f) * a.scale;
    const float kn = 1.f / sqrtf(ksq + 1e-6f);
#pragma unroll
    for (int j = 0; j < KPL; ++j) {
      q[j] *= qn;
      k[j] *= kn;
    }

    float g, beta;
    if (FUSE_GATE) {
      const float A = __expf(a.a_log[hv]);
      g = -A * softplus_f(to_f32(a.a_raw[row * a.ab_stride + hv]) + a.dt_bias[hv]);
      beta = sigmoid_f(to_f32(a.b_raw[row * a.ab_stride + hv]));
    } else {
      g = a.g_in[row * a.HV + hv];
      beta = a.beta_in[row * a.HV + hv];
    }
    const float eg = __expf(g);

    if constexpr (TR) {
      kq *= qn * kn;
      // 一遍扫状态：行 i 的 S_old·k → pr[i]，S_old·q → pr[ROWS+i]
      float pr[2 * ROWS];
#pragma unroll
      for (int i = 0; i < ROWS; ++i) {
        float p = 0.f, o = 0.f;
#pragma unroll
        for (int j = 0; j < KPL; ++j) {
          p += r[i][j] * k[j];
          o += r[i][j] * q[j];
        }
        pr[i] = p;
        pr[ROWS + i] = o;
      }
      const float mine = warp_sum_multi<2 * ROWS>(pr, lane);  // lane < 16：p(my)；lane ≥ 16：o₀(my)
      // 低半 lane 算 v'，高半 lane 从 xor 16 取 v' 后算 o
      const float vin = to_f32(vp[row * qkv_row + my]);  // 行 my 的 v（lane ≥ 16 也读，便宜）
      const float vprime_lo = (vin - eg * mine) * beta;
      const float vprime = __shfl_xor_sync(0xffffffffu, vprime_lo, 16);  // 高半拿到低半的 v'
      if (lane >= 16 && (lane & (S - 1)) == 0) {
        a.out[(row * a.HV + hv) * a.V + v0 + my] = to_bf16(eg * mine + vprime * kq);
      }
      // 状态更新：S = eg·S + v'_i·k，v'_i 从 lane i·S 广播
#pragma unroll
      for (int i = 0; i < ROWS; ++i) {
        const float vpi = __shfl_sync(0xffffffffu, vprime_lo, i * S);
#pragma unroll
        for (int j = 0; j < KPL; ++j) r[i][j] = fmaf(eg, r[i][j], vpi * k[j]);
      }
    } else {
      float o_mine = 0.f;  // lane i 保存第 i 行的输出，最后 ROWS 个 lane 连续写出
#pragma unroll
      for (int i = 0; i < ROWS; ++i) {
        float p = 0.f;
#pragma unroll
        for (int j = 0; j < KPL; ++j) {
          r[i][j] *= eg;
          p += r[i][j] * k[j];
        }
        p = warp_sum_f(p);
        const float vin = to_f32(vp[row * qkv_row + i]);  // 全 warp 同址，广播
        const float vprime = (vin - p) * beta;
        float o = 0.f;
#pragma unroll
        for (int j = 0; j < KPL; ++j) {
          r[i][j] += vprime * k[j];
          o += r[i][j] * q[j];
        }
        o = warp_sum_f(o);
        if (lane == i) o_mine = o;
      }
      if (lane < ROWS) a.out[(row * a.HV + hv) * a.V + v0 + lane] = to_bf16(o_mine);
    }
  }
}

}  // namespace
}  // namespace dinfer
