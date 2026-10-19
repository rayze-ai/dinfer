// decode attention v3（B4 第 4 轮，docs/loops/b4-sm100.md "decode v3 设计"）。参考 FlashInfer `decode.cuh` 的结构重写，
// 读我们自己的 paged 布局 `[slot][Hkv][2][D]`，不引库。
//
// 针对哪代卡：sm80+ 通用（cp.async 16 B）；B300 上 v2 只到 KV 带宽 1/4，是这版的直接动机。
// 与 v2（gqa.cu::gqa_decode_v2_kernel）差在哪、为什么快：
//   1. 16 lane / key、每 lane 8 维一次 16 B（v2：8 lane / key，每 lane 两次 16 B）；GQA 组内 GROUP 个 q 头共享同一份 K/V 读，
//      点积对每个头各做一次 16 lane shuffle（v2 对两个头串行、且每轮还做跨 key 槽的 max/sum 归约）。
//   2. K/V 经 cp.async 进 smem，PIPE 级流水：每级 = 一个 tile（V3_KEYS 个 key）的 K+V，`wait_group<PIPE-1>` 只等最老一级，
//      等待粒度从"等自己的读"变成"等整组"，在飞字节数 = PIPE × tile（v2 是每 lane 自己预取一轮）。
//   3. 4 个 key 槽各自维护 (m, l, acc)，循环结束后用 smem 合并一次（v2 每轮都合）。
//   4. split 规划：grid.z 仍按 batch 固定（CUDA graph），kernel 内 eff = ceil(n / keys_per_split)，多余 block 退出；
//      partial 布局与 v2 相同（[b][h][splits][D+2]），复用 gqa.cu 的 merge。work-item 展平留给 ArchProfile 轮。
// 对拍：ops_parity.sh（GQA_DECODE_V3=1 时 gqa_decode / gqa_mixed 走本实现），两卡都跑。
// 运行时可调参数只从 host 传入（keys_per_split）；模板参数 GROUP 由 host 按 heads/kv_heads 选。
#include <cstdlib>
#include "common.cuh"
#include "kernels.h"

using namespace dinfer;

namespace {

constexpr int V3_D = 128;                   // head_dim（host 校验 == 128）
constexpr int V3_VEC = 8;                   // 每 lane 维数（16 B bf16）
constexpr int V3_LPK = V3_D / V3_VEC;       // 16 lane / key
constexpr int V3_THREADS = 128;             // 4 warp
constexpr int V3_KEYS = V3_THREADS / V3_LPK;// 每 tile 8 key（每 warp 2 key）
constexpr int V3_PIPE = 2;                  // cp.async 流水级数

__device__ __forceinline__ void cp16(void* smem, const void* gmem) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ float sum16(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 8);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v;
}
__device__ __forceinline__ int64_t slot_off3(int64_t slot, int kv_heads) { return slot * kv_heads * 2 * V3_D; }

template <int GROUP>
__global__ void __launch_bounds__(V3_THREADS) gqa_decode_v3_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ cache, const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens, const int32_t* __restrict__ cu_seqlens_q, float* __restrict__ scratch,
    bf16* __restrict__ out_direct, int H, int Hkv, int block_size, int max_blocks, float scale, int q_stride, int splits,
    int keys_per_split) {
  const int b = blockIdx.x, hk = blockIdx.y, split = blockIdx.z;
  int row = b;
  if (cu_seqlens_q) {
    if (cu_seqlens_q[b + 1] - cu_seqlens_q[b] != 1) return;
    row = cu_seqlens_q[b];
  }
  const int n = seq_lens[b];
  const int eff = min(splits, max(1, (n + keys_per_split - 1) / keys_per_split));
  if (split >= eff) return;
  const int seg = (n + eff - 1) / eff;
  const int k_begin = split * seg, k_end = min(n, k_begin + seg);

  const int tid = threadIdx.x;
  const int ks = tid / V3_LPK;          // 0..7：本线程的 key 槽
  const int dl = (tid % V3_LPK) * V3_VEC;  // 本线程起始维度
  const int32_t* bt = block_table + (int64_t)b * max_blocks;

  __shared__ __align__(16) bf16 s_k[V3_PIPE][V3_KEYS][V3_D];
  __shared__ __align__(16) bf16 s_v[V3_PIPE][V3_KEYS][V3_D];
  __shared__ float s_m[GROUP][V3_KEYS], s_l[GROUP][V3_KEYS];
  __shared__ float s_acc[GROUP][V3_KEYS][V3_D];

  // q：本 lane 的 8 维 × GROUP 个头，预乘 scale（log2 域：用 exp2f 省一次乘）
  float qr[GROUP][V3_VEC];
  const float scale_log2 = scale * 1.4426950408889634f;
#pragma unroll
  for (int g = 0; g < GROUP; ++g) {
    const uint4 r = *reinterpret_cast<const uint4*>(q + (int64_t)row * q_stride + (int64_t)(hk * GROUP + g) * V3_D + dl);
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&r);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __bfloat1622float2(p[i]);
      qr[g][2 * i] = f.x * scale_log2; qr[g][2 * i + 1] = f.y * scale_log2;
    }
  }

  float m[GROUP], l[GROUP], acc[GROUP][V3_VEC];
#pragma unroll
  for (int g = 0; g < GROUP; ++g) {
    m[g] = -INFINITY; l[g] = 0.f;
#pragma unroll
    for (int c = 0; c < V3_VEC; ++c) acc[g][c] = 0.f;
  }

  const int ntiles = (k_end - k_begin + V3_KEYS - 1) / V3_KEYS;
  // 发出第 t 个 tile 的 K/V 读到 stage s（本线程负责 key = k_begin + t*8 + ks 的 dl..dl+8 维）
  auto issue = [&](int t, int s) {
    const int key = k_begin + t * V3_KEYS + ks;
    if (key < k_end) {
      const int64_t blk = bt[key / block_size];
      const bf16* slotp = cache + slot_off3(blk * block_size + (key % block_size), Hkv);
      cp16(&s_k[s][ks][dl], slotp + (hk * 2) * V3_D + dl);
      cp16(&s_v[s][ks][dl], slotp + (hk * 2 + 1) * V3_D + dl);
    }
    cp_commit();  // 空组也 commit，保证 wait_group 计数一致
  };
#pragma unroll
  for (int s = 0; s < V3_PIPE; ++s) issue(s, s);

  for (int t = 0; t < ntiles; ++t) {
    const int s = t % V3_PIPE;
    cp_wait<V3_PIPE - 1>();  // 第 t 组已落地（之后发出的 PIPE-1 组可仍在飞）
    __syncthreads();
    const int key = k_begin + t * V3_KEYS + ks;
    const bool valid = key < k_end;
    float kf[V3_VEC], vf[V3_VEC];
    {
      const uint4 rk = *reinterpret_cast<const uint4*>(&s_k[s][ks][dl]);
      const uint4 rv = *reinterpret_cast<const uint4*>(&s_v[s][ks][dl]);
      const __nv_bfloat162* pk = reinterpret_cast<const __nv_bfloat162*>(&rk);
      const __nv_bfloat162* pv = reinterpret_cast<const __nv_bfloat162*>(&rv);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 a = __bfloat1622float2(pk[i]), c = __bfloat1622float2(pv[i]);
        kf[2 * i] = a.x; kf[2 * i + 1] = a.y; vf[2 * i] = c.x; vf[2 * i + 1] = c.y;
      }
    }
#pragma unroll
    for (int g = 0; g < GROUP; ++g) {
      float part = 0.f;
#pragma unroll
      for (int c = 0; c < V3_VEC; ++c) part += qr[g][c] * kf[c];
      const float sc = sum16(part);  // 16 lane 内归约，全 warp 参与（两个 key 槽互不干扰：xor ≤ 8）
      const float s_val = valid ? sc : -INFINITY;
      const float m_new = fmaxf(m[g], s_val);
      // m 仍为 -inf（本 key 槽至今无有效 key）时 l/acc 都是 0，alpha 取 0 避免 exp2f(-inf - -inf) = NaN
      const float alpha = (m[g] == -INFINITY) ? 0.f : exp2f(m[g] - m_new);
      const float p = valid ? exp2f(s_val - m_new) : 0.f;
      l[g] = l[g] * alpha + p;
#pragma unroll
      for (int c = 0; c < V3_VEC; ++c) acc[g][c] = acc[g][c] * alpha + p * vf[c];
      m[g] = m_new;
    }
    __syncthreads();  // 本 stage 读完才能被下一次 issue 覆盖
    if (t + V3_PIPE < ntiles) issue(t + V3_PIPE, s); else cp_commit();
  }
  cp_wait<0>();

  // 跨 8 个 key 槽合并（循环外只做一次）
#pragma unroll
  for (int g = 0; g < GROUP; ++g) {
    if ((tid % V3_LPK) == 0) { s_m[g][ks] = m[g]; s_l[g][ks] = l[g]; }
#pragma unroll
    for (int c = 0; c < V3_VEC; ++c) s_acc[g][ks][dl + c] = acc[g][c];
  }
  __syncthreads();
  if (tid < V3_D) {
    const int d = tid;
#pragma unroll
    for (int g = 0; g < GROUP; ++g) {
      float M = -INFINITY;
#pragma unroll
      for (int k = 0; k < V3_KEYS; ++k) M = fmaxf(M, s_m[g][k]);
      float L = 0.f, O = 0.f;
#pragma unroll
      for (int k = 0; k < V3_KEYS; ++k) {
        const float a = (s_m[g][k] == -INFINITY) ? 0.f : exp2f(s_m[g][k] - M);
        L += s_l[g][k] * a;
        O += s_acc[g][k][d] * a;
      }
      const int h = hk * GROUP + g;
      if (out_direct) {
        // 本 block 一个有效 key 都没有（seq_len 0 的占位行）：写 0 而不是 0/0
        out_direct[((int64_t)row * H + h) * V3_D + d] = to_bf16(L > 0.f ? O / L : 0.f);
      } else {
        float* part = scratch + (((int64_t)b * H + h) * splits + split) * (V3_D + 2);
        // m 存回自然对数域，与 v2 partial / merge 口径一致（merge 用 __expf 且跳过 -inf）；-inf 原样保留做"空段"标记
        if (d == 0) { part[0] = (M == -INFINITY) ? -INFINITY : M * 0.6931471805599453f; part[1] = L; }
        part[2 + d] = O;
      }
    }
  }
}

}  // namespace

extern "C" int gqa_decode_v3_bf16(const void* q, const void* cache, const int32_t* block_table, const int32_t* seq_lens,
                                  const int32_t* cu_seqlens_q, float* scratch, void* out_direct, int64_t num_reqs,
                                  int64_t heads, int64_t kv_heads, int64_t block_size, int64_t max_blocks, float scale,
                                  int64_t q_stride, int64_t splits, int64_t keys_per_split, cudaStream_t stream) {
  const int group = (int)(heads / kv_heads);
  dim3 grid((unsigned)num_reqs, (unsigned)kv_heads, (unsigned)splits);
#define V3_LAUNCH(G)                                                                                                   \
  gqa_decode_v3_kernel<G><<<grid, V3_THREADS, 0, stream>>>(                                                            \
      static_cast<const bf16*>(q), static_cast<const bf16*>(cache), block_table, seq_lens, cu_seqlens_q, scratch,     \
      static_cast<bf16*>(out_direct), (int)heads, (int)kv_heads, (int)block_size, (int)max_blocks, scale,             \
      (int)q_stride, (int)splits, (int)keys_per_split)
  switch (group) {
    case 1: V3_LAUNCH(1); break;
    case 2: V3_LAUNCH(2); break;
    case 4: V3_LAUNCH(4); break;
    default: return (int)cudaErrorInvalidValue;
  }
#undef V3_LAUNCH
  return (int)cudaGetLastError();
}
