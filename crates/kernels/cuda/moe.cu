// MoE 参考 kernel（sm80+）。不追求性能：每 token 一个 block 做路由；专家分组用 count + prefix + scatter。
//
//   moe_route_f32        logits[T,E] fp32 → topk_ids[T,K] i32, topk_w[T,K] f32
//                        语义 = HF DeepseekV3TopkRouter / vLLM grouped_topk（n_group=1）：
//                        scores = sigmoid(logits); choice = scores + bias; top-k by choice;
//                        weights = scores[chosen]（不含 bias）; if renorm: w /= sum(w)+1e-20; w *= scale
//   moe_bucket_bf16      按专家分桶：counts[E] → offsets[E+1]（exclusive scan），sorted_tok[T*K] / sorted_k[T*K]
//                        （两 kernel：count + fill；offsets 由 host 读 counts 做 scan 再传回，或 GPU 单 block scan）
//   moe_gather_bf16      按 sorted_tok 把 x[T,H] 行收集到 xg[T*K, H]
//   moe_scatter_add_bf16 out[tok] += w * y[i]（i 为 sorted 下标）
#include <cstdlib>
#include "common.cuh"
#include "kernels.h"

using namespace dinfer;

namespace {

constexpr int MAX_E = 1024;  // GLM 256 / K3 896
constexpr int MAX_K = 32;

__global__ void route_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                             int32_t* __restrict__ topk_ids, float* __restrict__ topk_w, int E, int K, int renorm,
                             float scale) {
  const int64_t t = blockIdx.x;
  __shared__ float s_score[MAX_E];
  __shared__ float s_choice[MAX_E];
  __shared__ int s_sel[MAX_K];
  for (int e = threadIdx.x; e < E; e += blockDim.x) {
    const float sc = 1.f / (1.f + __expf(-logits[t * E + e]));
    s_score[e] = sc;
    s_choice[e] = sc + (bias ? bias[e] : 0.f);
  }
  __syncthreads();
  // 单线程做 K 次 argmax（E ≤ 1024，K ≤ 32：参考实现足够）。相等取小下标，与 torch.topk 稳定序一致。
  if (threadIdx.x == 0) {
    float wsum = 0.f;
    for (int k = 0; k < K; ++k) {
      int best = -1;
      float bv = -INFINITY;
      for (int e = 0; e < E; ++e) {
        const float v = s_choice[e];
        if (v > bv) {
          bv = v;
          best = e;
        }
      }
      s_sel[k] = best;
      s_choice[best] = -INFINITY;
      wsum += s_score[best];
    }
    const float denom = renorm ? (wsum + 1e-20f) : 1.f;
    for (int k = 0; k < K; ++k) {
      topk_ids[t * K + k] = s_sel[k];
      topk_w[t * K + k] = s_score[s_sel[k]] / denom * scale;
    }
  }
}

// logits[t,e] = Σ_d f32(x[t,d]) · f32(w[e,d])，全程 fp32（HF 路由把 x 与 gate.weight 都 cast 到 fp32 再 linear）。
// 一个 block 算一个 (t, e)；参考版，不追求性能。
__global__ void gate_kernel(const bf16* __restrict__ x, const bf16* __restrict__ w, float* __restrict__ logits, int E,
                            int H) {
  const int64_t t = blockIdx.x / E;
  const int e = blockIdx.x % E;
  const bf16* xr = x + t * H;
  const bf16* wr = w + (int64_t)e * H;
  float acc = 0.f;
  for (int d = threadIdx.x; d < H; d += blockDim.x) acc += to_f32(xr[d]) * to_f32(wr[d]);
  acc = block_reduce_sum<128>(acc);
  if (threadIdx.x == 0) logits[t * E + e] = acc;
}

// R1 v2（loop kaon3-decode-opt D174）：router 小 M bf16 gemv。logits_bf16[t,e] = bf16(Σ_d f32(x[t,d])·f32(w[e,d]))。
// 为什么：M=8/N=128/K=2816 这个形状 cuBLAS 选 cutlass wmma 16x16（grid 8 个 CTA）16.7 us，M=1 的 gemvx 3.4 us；
// 权重只有 720 KB，带宽下限 < 1 us，是并行度 / 延迟问题。v1（warp 一行、grid 32 块）latency-bound 反而更慢（D174）。
// v2：一个 block 一个专家行，8 warp 在 K 上切片（warp w 取 chunk w, w+8, …），lane 按 16 B 跨步读 w 与 M 行 x，
// fp32 累加；warp 内 xor 树归约 → smem[8][M] → warp 0 按固定顺序求和。累加序只依赖 K 与模板 M_TILE，不依赖 M/grid → 批不变。
// M_TILE ∈ {1,2,4,8,16}，H % 8 == 0。
constexpr int kRouterWarps = 8;
template <int M_TILE>
__global__ void __launch_bounds__(kRouterWarps * 32) router_gemv_kernel(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                                                        bf16* __restrict__ out, int M, int E, int H) {
  __shared__ float part[kRouterWarps][M_TILE];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int e = blockIdx.x;
  const uint4* wr = reinterpret_cast<const uint4*>(w + (int64_t)e * H);
  const int nchunk = H >> 3;
  float acc[M_TILE];
#pragma unroll
  for (int t = 0; t < M_TILE; ++t) acc[t] = 0.f;
  for (int c = warp * 32 + lane; c < nchunk; c += kRouterWarps * 32) {
    const uint4 wv = __ldg(wr + c);
    const bf16* w8 = reinterpret_cast<const bf16*>(&wv);
    float wf[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) wf[i] = to_f32(w8[i]);
#pragma unroll
    for (int t = 0; t < M_TILE; ++t) {
      if (t < M) {
        const uint4 xv = __ldg(reinterpret_cast<const uint4*>(x + (int64_t)t * H) + c);
        const bf16* x8 = reinterpret_cast<const bf16*>(&xv);
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[t] = fmaf(to_f32(x8[i]), wf[i], acc[t]);
      }
    }
  }
#pragma unroll
  for (int t = 0; t < M_TILE; ++t) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], o);
    if (lane == 0) part[warp][t] = acc[t];
  }
  __syncthreads();
  if (warp == 0 && lane < M_TILE && lane < M) {
    float v = 0.f;
#pragma unroll
    for (int k = 0; k < kRouterWarps; ++k) v += part[k][lane];
    out[(int64_t)lane * E + e] = to_bf16(v);
  }
}

__global__ void count_kernel(const int32_t* __restrict__ topk_ids, int32_t* __restrict__ counts, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) atomicAdd(&counts[topk_ids[i]], 1);
}

// 单 block exclusive scan（E ≤ 1024）：offsets[0..E] ；并把 cursor[e] = offsets[e] 作为填充游标。
// 同时产出 tile_off[0..E]：按 BM=32 行一块的 tile 前缀和（tile_off[e] = Σ_{e'<e} ceil(count[e']/32)），
// 供 grouped GEMM 用一维 tile 表代替 (专家 × 行块) 的满格 grid（loop kaon3-decode-opt B8：空 block 5.8M → 0）。
__global__ void scan_kernel(const int32_t* __restrict__ counts, int32_t* __restrict__ offsets,
                            int32_t* __restrict__ cursor, int32_t* __restrict__ tile_off, int E) {
  __shared__ int s[MAX_E];
  for (int e = threadIdx.x; e < E; e += blockDim.x) s[e] = counts[e];
  __syncthreads();
  if (threadIdx.x == 0) {
    int acc = 0, tacc = 0, tacc16 = 0;
    int32_t* tile_off16 = tile_off + (E + 1);  // B16b：第二张表 BM=16（decode 用），紧跟 BM=32 表之后
    for (int e = 0; e < E; ++e) {
      offsets[e] = acc;
      cursor[e] = acc;
      tile_off[e] = tacc;
      tile_off16[e] = tacc16;
      tacc += (s[e] + 31) / 32;
      tacc16 += (s[e] + 15) / 16;
      acc += s[e];
    }
    offsets[E] = acc;
    tile_off[E] = tacc;
    tile_off16[E] = tacc16;
  }
}

// 把 (t, k) 对填到其专家的区间；区间内顺序不定（无需稳定：归约是加法）。
__global__ void fill_kernel(const int32_t* __restrict__ topk_ids, int32_t* __restrict__ cursor,
                            int32_t* __restrict__ sorted_tok, int32_t* __restrict__ sorted_k,
                            int32_t* __restrict__ inv, int K, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    const int e = topk_ids[i];
    const int pos = atomicAdd(&cursor[e], 1);
    sorted_tok[pos] = (int32_t)(i / K);
    sorted_k[pos] = (int32_t)(i % K);
    inv[i] = pos;  // B11：逆映射 (tok,k) → sorted 位置，combine 用
  }
}

__global__ void gather_kernel(const bf16* __restrict__ x, bf16* __restrict__ xg, const int32_t* __restrict__ sorted_tok,
                              int H) {
  const int64_t i = blockIdx.x;
  const bf16* src = x + (int64_t)sorted_tok[i] * H;
  bf16* dst = xg + i * H;
  for (int d = threadIdx.x; d < H; d += blockDim.x) dst[d] = src[d];
}

// out[sorted_tok[i], :] += topk_w[sorted_tok[i]*K + sorted_k[i]] * y[i, :]
// 同一 token 的 K 个专家可能并发写同一行 → 用 fp32 atomicAdd 到临时 fp32 累加器更稳；
// 参考版直接 bf16 读-加-写会有竞争。这里 out 用 fp32 累加器，最后 cast。
// mode：0 = 直接 `acc += f32(w) * f32(y)`（f32 权重，我们原口径）；
//   1 = vLLM `cutlass_moe` 合并口径：`w = bf16(f32(topk_w·per_expert_scale))`，每项 `bf16(w · f32(y))` 再累加。
// per_expert 为空指针时视为全 1（无 per-expert scale 的模型）。kernel 体内无 #if，口径由运行时参数选。
__global__ void scatter_add_kernel(const bf16* __restrict__ y, float* __restrict__ acc,
                                   const int32_t* __restrict__ sorted_tok, const int32_t* __restrict__ sorted_k,
                                   const float* __restrict__ topk_w, const void* __restrict__ per_expert,
                                   float w_scale, int K, int H, int mode) {
  const int64_t i = blockIdx.x;
  const int tok = sorted_tok[i];
  const int e = sorted_k[i];
  float w = topk_w[(int64_t)tok * K + e];
  // vLLM：`per_expert_scale[topk_ids].to(topk_weights.dtype)`——per_expert 是 bf16 参数，参与 f32 运算前转 f32。
  if (per_expert) w *= to_f32(static_cast<const bf16*>(per_expert)[e]) * w_scale;
  else w *= w_scale;
  const bf16* src = y + i * H;
  float* dst = acc + (int64_t)tok * H;
  for (int d = threadIdx.x; d < H; d += blockDim.x) {
    const float v = to_f32(src[d]);
    if (mode == 1) {
      const float wb = to_f32(to_bf16(w));
      atomicAdd(&dst[d], to_f32(to_bf16(wb * v)));
    } else {
      atomicAdd(&dst[d], w * v);
    }
  }
}

// out = bf16(acc) + out：先把 fp32 累加器 cast 成 bf16，再与 out（shared expert 输出）bf16 相加 → 与 HF
// `moe(...).type(bf16) + shared_experts(x)` 的舍入顺序一致。
__global__ void f32_add_to_bf16_kernel(const float* __restrict__ acc, bf16* __restrict__ out, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) out[i] = to_bf16(to_f32(to_bf16(acc[i])) + to_f32(out[i]));
}

inline int grid_for(int64_t n) {
  const int64_t want = (n + 255) / 256;
  return (int)(want > 4096 ? 4096 : want);
}

}  // namespace

extern "C" int moe_gate_f32(const void* x, const void* w, float* logits, int64_t tokens, int64_t num_experts,
                            int64_t hidden, cudaStream_t stream) {
  if (tokens <= 0) return 0;
  if (tokens * num_experts > 0x7fffffff) return err(cudaErrorInvalidValue);
  gate_kernel<<<(unsigned)(tokens * num_experts), 128, 0, stream>>>(static_cast<const bf16*>(x),
                                                                    static_cast<const bf16*>(w), logits,
                                                                    (int)num_experts, (int)hidden);
  return last_err();
}

extern "C" int router_gemv_bf16(const void* x, const void* w, void* out, int64_t tokens, int64_t num_experts,
                                int64_t hidden, cudaStream_t stream) {
  if (tokens <= 0) return 0;
  if (tokens > 16 || hidden % 8 != 0 || num_experts <= 0) return err(cudaErrorInvalidValue);
  const int warps = kRouterWarps;
  const dim3 grid((unsigned)num_experts);
  const bf16* px = static_cast<const bf16*>(x);
  const bf16* pw = static_cast<const bf16*>(w);
  bf16* po = static_cast<bf16*>(out);
  const int M = (int)tokens, E = (int)num_experts, H = (int)hidden;
  if (M <= 1) router_gemv_kernel<1><<<grid, warps * 32, 0, stream>>>(px, pw, po, M, E, H);
  else if (M <= 2) router_gemv_kernel<2><<<grid, warps * 32, 0, stream>>>(px, pw, po, M, E, H);
  else if (M <= 4) router_gemv_kernel<4><<<grid, warps * 32, 0, stream>>>(px, pw, po, M, E, H);
  else if (M <= 8) router_gemv_kernel<8><<<grid, warps * 32, 0, stream>>>(px, pw, po, M, E, H);
  else router_gemv_kernel<16><<<grid, warps * 32, 0, stream>>>(px, pw, po, M, E, H);
  return last_err();
}

extern "C" int moe_route_f32(const float* logits, const float* bias, int32_t* topk_ids, float* topk_w, int64_t tokens,
                             int64_t num_experts, int64_t topk, int renormalize, float scale, cudaStream_t stream) {
  if (tokens <= 0) return 0;
  if (num_experts <= 0 || num_experts > MAX_E || topk <= 0 || topk > MAX_K || topk > num_experts)
    return err(cudaErrorInvalidValue);
  route_kernel<<<(unsigned)tokens, 256, 0, stream>>>(logits, bias, topk_ids, topk_w, (int)num_experts, (int)topk,
                                                     renormalize, scale);
  return last_err();
}

// counts/offsets/cursor: 调用方提供的 [E] / [E+1] / [E] 缓冲（counts 需先清零）。
// ---------------------------------------------------------------------------
// Gemma4 路由：**全专家 softmax → 按原始 logits 取 top-k → top-k 上重新归一化 → 乘 per_expert_scale**。
// 与上面 route_kernel 的 sigmoid 语义完全不同（那是 DeepSeekV3 / GLM / K3 的口径），不能混用。
// 依据 vLLM（镜像 0.26.1rc1）在 CUDA 上实际走的 `_gemma4_routing_kernel`（Triton 融合版，非 torch 版），
// 逐条对应其 PTX / SASS（§56，`num_warps=1`、BLOCK_E = next_pow2(E)，每 lane 持排序后的 EPT = BLOCK_E/32 个位置）：
//   raw_k = ex2.approx.ftz((logit_k − max) · 0x3FB8AA3B)          // 按 logits 降序（并列取小下标）
//   lane_t = ((raw_{EPT·t} + raw_{EPT·t+1}) + …)（仅位置 < K）→ 32 lane xor 蝶形 → renorm
//   inv = rcp.approx(renorm)；w_k = (raw_k · inv) · per_expert_scale[id_k]
// 旧写法（全专家 softmax 再在 top-k 上重归一化，两次除法）数学等价但舍入不同：Kaon-V3 L7 token 10 的 moe_y 差 344 元素。
__device__ __forceinline__ float route_ex2_approx(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
// exact=1：精确 exp2f；0：SFU ex2.approx。x 已是 (logit-m)·log2e（2 为底）。
__device__ __forceinline__ float route_ex2(float x, int exact) {
  return exact ? exp2f(x) : route_ex2_approx(x);
}
// vLLM torch softmax 用 `expf(logit-m)`（e 为底，不走 log2e 换算）。f32 下 exp2f(x·log2e)≠expf(x)。
// 传入原始 (logit-m)，exact=1 走 expf 对齐 vLLM。
__device__ __forceinline__ float route_exp_e(float dx, int exact) {
  return exact ? expf(dx) : route_ex2_approx(__fmul_rn(dx, 1.4426950408889634f));
}
__device__ __forceinline__ float route_rcp(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
/// `exact != 0` 时用 IEEE 精确倒数（Triton 的 `1.0 / renorm_raw` 编译成 div.rn）。
__device__ __forceinline__ float route_inv(float x, int exact) {
  return exact ? __frcp_rn(x) : route_rcp(x);
}
// 一行 token 的 Gemma4 路由权重（口径见上注释）；单线程执行。route_softmax_kernel 与融合版 moe_route_bucket_small 共用，
// 保两条路径逐位同。lg = 该 token 的 logits 行（E 个），ids = 该 token 的 top-k id（K 个），写 w[K]。
__device__ __forceinline__ void route_softmax_row(const float* __restrict__ lg, const int32_t* __restrict__ ids,
                                                  const void* __restrict__ per_expert_scale, float* __restrict__ w,
                                                  int E, int K, int exact_rcp) {
  float m = -INFINITY;
  for (int e = 0; e < E; ++e) m = fmaxf(m, lg[e]);
  // vLLM gemma4：全体 E 个专家 softmax 概率（prob_e = exp(l_e-m)/Z，Z=全体和），再 gather top-k、renorm。
  // f64 下与"只对 topk 的 exp 求和"等价，但 f32 下 vLLM 先按 Z 除（每个 prob 舍 f32）再求 topk 和，
  // 归约舍入不同 → 必须复刻全体 Z（loop kaon3-moe 第 21 轮：差 8.2e-3）。
  float Z = 0.f;
  for (int e = 0; e < E; ++e) Z = __fadd_rn(Z, route_exp_e(__fsub_rn(lg[e], m), exact_rcp));
  const float invZ = route_inv(Z, exact_rcp);
  float raw[MAX_K];
  for (int k = 0; k < K; ++k)
    raw[k] = __fmul_rn(route_exp_e(__fsub_rn(lg[ids[k]], m), exact_rcp), invZ);
  int block_e = 1;
  while (block_e < E) block_e <<= 1;
  const int ept = block_e >= 32 ? block_e / 32 : 1;
  float lane[32];
  for (int l = 0; l < 32; ++l) {
    float acc = 0.f;
    bool first = true;
    for (int j = 0; j < ept; ++j) {
      const int pos = l * ept + j;
      if (pos >= K) continue;
      acc = first ? raw[pos] : __fadd_rn(acc, raw[pos]);
      first = false;
    }
    lane[l] = acc;
  }
  for (int o = 16; o > 0; o >>= 1) {
    float nxt[32];
    for (int l = 0; l < 32; ++l) nxt[l] = __fadd_rn(lane[l], lane[l ^ o]);
    for (int l = 0; l < 32; ++l) lane[l] = nxt[l];
  }
  const float renorm = lane[0] > 0.f ? lane[0] : 1.f;
  const float inv = route_inv(renorm, exact_rcp);
  for (int k = 0; k < K; ++k) {
    const int e = ids[k];
    const float sc = per_expert_scale ? to_f32(static_cast<const bf16*>(per_expert_scale)[e]) : 1.f;
    w[k] = __fmul_rn(__fmul_rn(raw[k], inv), sc);
  }
}
// B32（loop kaon3-decode-opt D148）：route_softmax_row 的 warp 版。全 32 lane 协作；**数值与单 lane 版逐位同**：
//  - m：max 的归约序无关（max 可交换结合）；
//  - Z：单 lane 版是 e=0..E-1 顺序 `__fadd_rn`。warp 版 lane l 先算自己段 [l·ept, (l+1)·ept) 的 exp，再由 lane 0 按 e 升序
//    从 shfl 取值顺序累加（exp 并行算、求和仍串行 → 位同，但 exp（E 次 __expf/MUFU）并行化掉了 128→4 次/lane）；
//  - raw/renorm/w 与原版同式（K ≤ 32，一 lane 一个 k）。
__device__ __forceinline__ float route_lg(const void* __restrict__ lg, int e, int lg_bf16) {
  return lg_bf16 ? to_f32(static_cast<const bf16*>(lg)[e]) : static_cast<const float*>(lg)[e];
}
__device__ __forceinline__ void route_softmax_row_warp(const void* __restrict__ lg, const int32_t* __restrict__ ids,
                                                       const void* __restrict__ per_expert_scale, float* __restrict__ w,
                                                       int E, int K, int exact_rcp, int lane, int lg_bf16) {
  float m = -INFINITY;
  for (int e = lane; e < E; e += 32) m = fmaxf(m, route_lg(lg, e, lg_bf16));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  int block_e = 1;
  while (block_e < E) block_e <<= 1;
  const int ept = block_e >= 32 ? block_e / 32 : 1;   // 每 lane 连续 ept 个专家（E=128 → 4）
  float ex[32];  // ept ≤ 32（E ≤ 1024）；下标全部编译期展开（避免落 local memory）
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    const int e = lane * ept + j;
    ex[j] = (j < ept && e < E) ? route_exp_e(__fsub_rn(route_lg(lg, e, lg_bf16), m), exact_rcp) : 0.f;
  }
  // Z：e 升序串行累加（与单 lane 版同序）。全 warp 同步做（每 lane 得到同一个 Z），依次从 lane l 取 ex[j]。
  float Z = 0.f;
  for (int l = 0; l < 32; ++l) {
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      if (j < ept) {
        const float v = __shfl_sync(0xffffffffu, ex[j], l);
        if (l * ept + j < E) Z = __fadd_rn(Z, v);
      }
    }
  }
  const float invZ = route_inv(Z, exact_rcp);
  // raw[k]（一 lane 一个 k）：exp 重算一次（与原版同一表达式 → 同值）
  float rawk = 0.f;
  if (lane < K) rawk = __fmul_rn(route_exp_e(__fsub_rn(route_lg(lg, ids[lane], lg_bf16), m), exact_rcp), invZ);
  // renorm：原版把 raw[0..K) 按 pos = l·ept' + j 分到 32 个 lane 槽后蝶形求和（ept' 按 block_e 算）。K ≤ ept' 时每槽最多 1 个
  // → 槽 l 的值 = raw[l]（l<K）或 0，蝶形序 xor 16..1 与原版同。K > ept' 时原版槽内顺序累加——Kaon K=8、E=128 → ept'=4，
  // 槽 l 含 raw[4l..4l+4)：这里按同序先在 lane 内做不出来（raw 在不同 lane）→ 用 shfl 把 raw[4l..4l+4) 拉到 lane l 顺序加。
  float slot = 0.f;
  {
    bool first = true;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      if (j < ept) {
        const int pos = lane * ept + j;
        const float v = __shfl_sync(0xffffffffu, rawk, pos < 32 ? pos : 0);
        if (pos < K) { slot = first ? v : __fadd_rn(slot, v); first = false; }
      }
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) slot = __fadd_rn(slot, __shfl_xor_sync(0xffffffffu, slot, o));
  const float renorm = slot > 0.f ? slot : 1.f;
  const float inv = route_inv(renorm, exact_rcp);
  if (lane < K) {
    const int e = ids[lane];
    const float sc = per_expert_scale ? to_f32(static_cast<const bf16*>(per_expert_scale)[e]) : 1.f;
    w[lane] = __fmul_rn(__fmul_rn(rawk, inv), sc);
  }
}
__global__ void route_softmax_kernel(const float* __restrict__ logits, const int32_t* __restrict__ topk_ids,
                                     const void* __restrict__ per_expert_scale, float* __restrict__ topk_w, int E,
                                     int K, int exact_rcp) {
  if (threadIdx.x != 0) return;
  const int64_t t = blockIdx.x;
  route_softmax_row(logits + t * E, topk_ids + t * K, per_expert_scale, topk_w + t * K, E, K, exact_rcp);
}
// 按原始 logits 取 top-k（Gemma4 口径：不看 softmax 概率的大小关系，两者单调等价，但按 logits 可避免
// exp 下溢导致全 0 时排序不稳定）。
__global__ void route_topk_ids_kernel(const float* __restrict__ logits, int32_t* __restrict__ topk_ids, int E,
                                      int K) {
  const int64_t t = blockIdx.x;
  __shared__ float s_val[MAX_E];
  for (int e = threadIdx.x; e < E; e += blockDim.x) s_val[e] = logits[t * E + e];
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int k = 0; k < K; ++k) {
      int best = -1;
      float bv = -INFINITY;
      for (int e = 0; e < E; ++e) {
        const float v = s_val[e];
        if (v > bv) {
          bv = v;
          best = e;
        }
      }
      topk_ids[t * K + k] = best;
      s_val[best] = -INFINITY;
    }
  }
}

extern "C" int moe_route_softmax_f32(const float* logits, int32_t* topk_ids, const void* per_expert_scale,
                                     float* topk_w, int64_t tokens, int64_t num_experts, int64_t topk,
                                     cudaStream_t stream) {
  if (num_experts <= 0 || num_experts > MAX_E || topk <= 0 || topk > MAX_K || topk > num_experts) return -1;
  if (tokens <= 0) return 0;
  route_topk_ids_kernel<<<(unsigned)tokens, 256, 0, stream>>>(logits, topk_ids, (int)num_experts, (int)topk);
  // `ROUTER_RCP_EXACT=1`：router 归约用精确倒数（对照 Triton `1.0 / renorm`），A/B 用。
  static const int exact_rcp = [] {
    const char* e = getenv("ROUTER_RCP_EXACT");
    return (e && e[0] == '1') ? 1 : 0;
  }();
  route_softmax_kernel<<<(unsigned)tokens, 256, 0, stream>>>(logits, topk_ids, per_expert_scale, topk_w,
                                                             (int)num_experts, (int)topk, exact_rcp);
  return 0;
}

// ---------------------------------------------------------------------------
// B10（loop kaon3-decode-opt）：decode 小 batch 的路由前处理一次 launch 做完：
//   top-k（照 route_topk_ids_kernel 的选择顺序）→ 权重（route_softmax_row，逐位同）→ counts/offsets/cursor/tile_off
//   → sorted_tok/sorted_k → row2e。替代 route_topk_ids + route_softmax + memset + count + scan + fill + row2expert 7 次。
// 单 block；tokens ≤ kRouteSmallMaxTok；每线程一个 token（≤ blockDim）。sorted 区间内顺序由 smem 原子决定（与旧版一样"不定"）。
constexpr int kRouteSmallMaxTok = 64;
__global__ void __launch_bounds__(256) route_bucket_small_kernel(
    const void* __restrict__ logits, const void* __restrict__ per_expert_scale, int32_t* __restrict__ topk_ids,
    float* __restrict__ topk_w, int32_t* __restrict__ counts, int32_t* __restrict__ offsets,
    int32_t* __restrict__ cursor, int32_t* __restrict__ sorted_tok, int32_t* __restrict__ sorted_k,
    int32_t* __restrict__ tile_off, int32_t* __restrict__ row2e, int32_t* __restrict__ inv, int tokens, int E, int K,
    int exact_rcp, int lg_bf16, int warp_softmax) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  float* s_val = reinterpret_cast<float*>(smem_raw);           // [tokens][E] 工作副本（top-k 时置 -inf）
  float* s_raw = s_val + (size_t)tokens * E;                    // [tokens][E] 原始 logits（softmax 用；D166）
  int32_t* s_cnt = reinterpret_cast<int32_t*>(s_raw + (size_t)tokens * E);  // [E]
  int32_t* s_cur = s_cnt + E;                                   // [E]
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nwarp = blockDim.x >> 5;
  for (int i = tid; i < tokens * E; i += blockDim.x) { const float v = route_lg(logits, i, lg_bf16); s_val[i] = v; s_raw[i] = v; }  // B33c：bf16 logits 直接读
  for (int e = tid; e < E; e += blockDim.x) s_cnt[e] = 0;
  __syncthreads();
  // B19（D73）：每 token 一 warp。top-k 第 k 轮：lane 各扫 E/32 个专家取局部 (max, 最小下标)，warp xor 归约
  // （值大者胜；相等取小下标）—— 与串行 `if (v > bv)` 从 0 扫到 E-1 的结果完全相同（串行也是并列取先出现的小下标）。
  for (int t = warp; t < tokens; t += nwarp) {
    float* row = s_val + (size_t)t * E;
    int32_t* ids = topk_ids + (size_t)t * K;
    for (int k = 0; k < K; ++k) {
      float bv = -INFINITY;
      int bi = 0x7fffffff;
      for (int e = lane; e < E; e += 32) {
        const float v = row[e];
        if (v > bv) { bv = v; bi = e; }
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
      if (lane == 0) {
        ids[k] = bi;
        atomicAdd(&s_cnt[bi], 1);
      }
      if (bi >= lane && ((bi - lane) & 31) == 0) row[bi] = -INFINITY;  // 该专家属于本 lane 的扫描区，置 -inf
      __syncwarp();
    }
    __syncwarp();
    // B32（D161）：warp 版 softmax 实测 c1 +3.5% / c8 +2.7%（Z 串行 shfl 链 + ex[32] 寄存器压力），默认关；`ROUTE_SOFTMAX_WARP=1` 开。
    // D166：softmax 读 smem 里已 cast 的整行副本 s_raw（与 s_val 同源、未被 top-k 置 -inf），bf16/f32 入口一致，无 local 数组。
    const float* lgrow = s_raw + (size_t)t * E;
    if (warp_softmax) {
      route_softmax_row_warp(lgrow, ids, per_expert_scale, topk_w + (size_t)t * K, E, K, exact_rcp, lane, 0);
    } else if (lane == 0) {
      route_softmax_row(lgrow, ids, per_expert_scale, topk_w + (size_t)t * K, E, K, exact_rcp);
    }
  }
  __syncthreads();
  if (tid == 0) {
    int acc = 0, tacc = 0, tacc16 = 0;
    int32_t* tile_off16 = tile_off + (E + 1);
    for (int e = 0; e < E; ++e) {
      counts[e] = s_cnt[e];
      offsets[e] = acc;
      s_cur[e] = acc;
      tile_off[e] = tacc;
      tile_off16[e] = tacc16;
      tacc += (s_cnt[e] + 31) / 32;
      tacc16 += (s_cnt[e] + 15) / 16;
      acc += s_cnt[e];
    }
    offsets[E] = acc;
    tile_off[E] = tacc;
    tile_off16[E] = tacc16;
  }
  __syncthreads();
  for (int t = warp; t < tokens; t += nwarp) {
    if (lane < K) {  // 一 lane 一个 (t,k)：原子游标分配位置（区间内顺序不定，与旧版同）
      const int e = topk_ids[(size_t)t * K + lane];
      const int pos = atomicAdd(&s_cur[e], 1);
      sorted_tok[pos] = t;
      sorted_k[pos] = lane;
      inv[(size_t)t * K + lane] = pos;
    }
  }
  __syncthreads();
  // cursor 终值 = offsets[e+1]（旧版 fill 完成后的状态）；row2e[r] = 所在区间
  for (int e = tid; e < E; e += blockDim.x) cursor[e] = s_cur[e];
  const int R = tokens * K;
  for (int r = tid; r < R; r += blockDim.x) {
    int lo = 0, hi = E;
    while (hi - lo > 1) {
      const int mid = (lo + hi) >> 1;
      if (offsets[mid] <= r) lo = mid; else hi = mid;
    }
    row2e[r] = lo;
  }
}

extern "C" int moe_route_bucket_small(const void* logits, int logits_bf16, const void* per_expert_scale, int32_t* topk_ids,
                                      float* topk_w, int32_t* counts, int32_t* offsets, int32_t* cursor,
                                      int32_t* sorted_tok, int32_t* sorted_k, int32_t* tile_off, int32_t* row2e,
                                      int32_t* inv, int64_t tokens, int64_t num_experts, int64_t topk,
                                      cudaStream_t stream) {
  static const int warp_sm = []{ const char* v = getenv("ROUTE_SOFTMAX_WARP"); return (v && v[0] == '1') ? 1 : 0; }();
  if (tokens <= 0) return 0;
  if (tokens > kRouteSmallMaxTok || num_experts <= 0 || num_experts > MAX_E || topk <= 0 || topk > MAX_K ||
      topk > num_experts)
    return err(cudaErrorInvalidValue);
  static const int exact_rcp = [] {
    const char* e = getenv("ROUTER_RCP_EXACT");
    return (e && e[0] == '1') ? 1 : 0;
  }();
  const size_t smem = 2 * (size_t)tokens * num_experts * sizeof(float) + 2 * (size_t)num_experts * sizeof(int32_t);  // D166：s_val + s_raw
  if (smem > 48 * 1024) {
    static size_t attr = 0;
    if (smem > attr) {
      if (cudaFuncSetAttribute(route_bucket_small_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) !=
          cudaSuccess)
        return last_err();
      attr = smem;
    }
  }
  route_bucket_small_kernel<<<1, 256, smem, stream>>>(logits, per_expert_scale, topk_ids, topk_w, counts, offsets,
                                                      cursor, sorted_tok, sorted_k, tile_off, row2e, inv, (int)tokens,
                                                      (int)num_experts, (int)topk, exact_rcp, logits_bf16, warp_sm);
  return last_err();
}

extern "C" int moe_bucket(const int32_t* topk_ids, int32_t* counts, int32_t* offsets, int32_t* cursor,
                          int32_t* sorted_tok, int32_t* sorted_k, int32_t* tile_off, int32_t* inv, int64_t tokens,
                          int64_t num_experts, int64_t topk, cudaStream_t stream) {
  if (tokens <= 0) return 0;
  if (num_experts <= 0 || num_experts > MAX_E) return err(cudaErrorInvalidValue);
  const int64_t n = tokens * topk;
  if (cudaMemsetAsync(counts, 0, num_experts * sizeof(int32_t), stream) != cudaSuccess) return last_err();
  count_kernel<<<grid_for(n), 256, 0, stream>>>(topk_ids, counts, n);
  scan_kernel<<<1, 1024, 0, stream>>>(counts, offsets, cursor, tile_off, (int)num_experts);
  fill_kernel<<<grid_for(n), 256, 0, stream>>>(topk_ids, cursor, sorted_tok, sorted_k, inv, (int)topk, n);
  return last_err();
}

extern "C" int moe_gather_bf16(const void* x, void* xg, const int32_t* sorted_tok, int64_t n, int64_t hidden,
                               cudaStream_t stream) {
  if (n <= 0) return 0;
  const int threads = hidden >= 256 ? 256 : (int)(((hidden + 31) / 32) * 32);
  gather_kernel<<<(unsigned)n, threads, 0, stream>>>(static_cast<const bf16*>(x), static_cast<bf16*>(xg), sorted_tok,
                                                     (int)hidden);
  return last_err();
}

// acc: [tokens, hidden] fp32，需先清零；y: [n, hidden] bf16（n = tokens*topk 的排序后行）。
extern "C" int moe_scatter_add_f32(const void* y, float* acc, const int32_t* sorted_tok, const int32_t* sorted_k,
                                   const float* topk_w, const void* per_expert, float w_scale,
                                   int64_t n, int64_t topk, int64_t hidden, int mode, cudaStream_t stream) {
  if (n <= 0) return 0;
  const int threads = hidden >= 256 ? 256 : (int)(((hidden + 31) / 32) * 32);
  scatter_add_kernel<<<(unsigned)n, threads, 0, stream>>>(static_cast<const bf16*>(y), acc, sorted_tok, sorted_k,
                                                          topk_w, per_expert, w_scale, (int)topk, (int)hidden, mode);
  return last_err();
}

// out[:n] (bf16) += acc[:n] (fp32)
// B11：MoE 输出合并一次 launch：out[tok,:] = bf16(Σ_k w·y[inv[tok,k],:])，替代 memset(acc)+scatter_add(atomic)+memset(y)+f32_add_to_bf16。
// mode 同 scatter_add_kernel（1 = vLLM bf16 权重口径：Σ f32(bf16(bf16(w)·v))；0 = f32）。末尾与 f32_add_to_bf16 同：
// bf16(f32(bf16(acc)) + 0)。累加序固定 k 升序（旧版 atomic 顺序不定）。grid = tokens。
__global__ void combine_kernel(const bf16* __restrict__ y, const float* __restrict__ topk_w,
                               const int32_t* __restrict__ inv, const void* __restrict__ per_expert,
                               const int32_t* __restrict__ topk_ids, float w_scale, bf16* __restrict__ out, int K,
                               int H, int mode) {
  const int64_t tok = blockIdx.x;
  float w[MAX_K];
  int32_t pos[MAX_K];
  for (int k = 0; k < K; ++k) {
    float ww = topk_w[tok * K + k];
    if (per_expert) ww *= to_f32(static_cast<const bf16*>(per_expert)[topk_ids[tok * K + k]]) * w_scale;
    else ww *= w_scale;
    w[k] = mode == 1 ? to_f32(to_bf16(ww)) : ww;
    pos[k] = inv[tok * K + k];
  }
  for (int d = threadIdx.x; d < H; d += blockDim.x) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
      const float v = to_f32(y[(int64_t)pos[k] * H + d]);
      acc += mode == 1 ? to_f32(to_bf16(w[k] * v)) : w[k] * v;
    }
    out[tok * H + d] = to_bf16(to_f32(to_bf16(acc)));
  }
}
extern "C" int moe_combine_bf16(const void* y, const float* topk_w, const int32_t* inv, const void* per_expert,
                                const int32_t* topk_ids, float w_scale, void* out, int64_t tokens, int64_t topk,
                                int64_t hidden, int mode, cudaStream_t stream) {
  if (tokens <= 0) return 0;
  if (topk <= 0 || topk > MAX_K) return err(cudaErrorInvalidValue);
  const int threads = hidden >= 256 ? 256 : (int)(((hidden + 31) / 32) * 32);
  combine_kernel<<<(unsigned)tokens, threads, 0, stream>>>(static_cast<const bf16*>(y), topk_w, inv, per_expert,
                                                           topk_ids, w_scale, static_cast<bf16*>(out), (int)topk,
                                                           (int)hidden, mode);
  return last_err();
}

extern "C" int f32_add_to_bf16(const float* acc, void* out, int64_t n, cudaStream_t stream) {
  if (n <= 0) return 0;
  f32_add_to_bf16_kernel<<<grid_for(n), 256, 0, stream>>>(acc, static_cast<bf16*>(out), n);
  return last_err();
}
