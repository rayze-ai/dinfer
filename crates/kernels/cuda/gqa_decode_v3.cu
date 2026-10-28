// decode attention v3（B4 第 4 轮，docs/loops/b4-sm100.md "decode v3 设计"）。参考 FlashInfer `decode.cuh` 的结构重写，
// 读我们自己的 paged 布局 `[slot][Hkv][2][D]`，不引库。
//
// 针对哪代卡：sm80+ 通用（cp.async 16 B）；B300 上 v2 只到 KV 带宽 1/4，是这版的直接动机。
// 与 v2（gqa.cu::gqa_decode_v2_kernel）差在哪、为什么快：
//   1. D/8 lane / key（D=128 → 16、D=256 → 32）、每 lane 8 维一次 16 B（v2：8 lane / key，每 lane 两次 16 B）；
//      GQA 组内 GROUP 个 q 头共享同一份 K/V 读，点积对每个头各做一次 LPK lane shuffle
//      （v2 对两个头串行、且每轮还做跨 key 槽的 max/sum 归约）。
//   2. K/V 经 cp.async 进 smem，PIPE 级流水：每级 = 一个 tile（128/LPK 个 key：D=128 → 8、D=256 → 4）的 K+V，`wait_group<PIPE-1>` 只等最老一级，
//      等待粒度从"等自己的读"变成"等整组"，在飞字节数 = PIPE × tile（v2 是每 lane 自己预取一轮）。
//   3. 每线程的 key 槽各自维护 (m, l, acc)，循环结束后用 smem 合并一次（v2 每轮都合）。
//   4. split 规划：grid.z 仍按 batch 固定（CUDA graph），kernel 内 eff = ceil(n / keys_per_split)，多余 block 退出；
//      partial 布局与 v2 相同（[b][h][splits][D+2]），复用 gqa.cu 的 merge。work-item 展平留给 ArchProfile 轮。
// 对拍：ops_parity.sh（GQA_DECODE_V3=1 时 gqa_decode / gqa_mixed 走本实现），两卡都跑。
// 运行时可调参数只从 host 传入（keys_per_split）；模板参数 GROUP / D 由 host 按 heads/kv_heads / head_dim 选：
// D=128 支持 GROUP 1/2/4，D=256 支持 GROUP 1/2/4/6（Qwen3.5：24 q 头 / 4 kv 头）。
#include <cstdlib>
#include "common.cuh"
#include "kernels.h"

using namespace dinfer;

#include "gqa_decode_v3_core.cuh"

using namespace dinfer::gqa_v3;  // V3_THREADS / V3_LPK / gqa_decode_v3_core 等（B1 抽到 gqa_decode_v3_core.cuh）

namespace {

// 原入口：grid=(batch,Hkv,split)，128 线程，薄壳调 core（active 恒真 → 行为不变）。
template <int GROUP, int D>
__global__ void __launch_bounds__(V3_THREADS) gqa_decode_v3_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ cache, const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens, const int32_t* __restrict__ cu_seqlens_q, float* __restrict__ scratch,
    bf16* __restrict__ out_direct, int H, int Hkv, int block_size, int max_blocks, float scale, int q_stride, int splits,
    int keys_per_split, int window) {
  gqa_decode_v3_core<GROUP, D>(q, cache, block_table, seq_lens, cu_seqlens_q, scratch, out_direct, H, Hkv, block_size,
                               max_blocks, scale, q_stride, splits, keys_per_split, window, blockIdx.x, blockIdx.y,
                               blockIdx.z, threadIdx.x);
}

// ---------------------------------------------------------------------------------------------------------------
// D=512 变体（Gemma4 / Kaon-V3 全注意力层：16 q 头 / 2 kv 头 → GROUP=8，loop kaon3-decode-opt B3）。
// 与 v3 同算法（split-K flash-decode、cp.async 双缓冲、同 scratch 布局 → 共用 gqa_decode_merge_v2_kernel），差别：
//   - 256 线程：一个 key 占 64 lane（2 warp），tile = 4 key；点积先 warp 内 shfl 归约，再两 warp 经 smem 合并；
//   - 跨 key 槽合并逐 GROUP 做（s_acc 只放一个 g 的 [KEYS][D]，控 smem）。
// 之前 D=512 decode 走 gqa.cu 通用慢路径（每 (req, q头) 一个 block、三遍扫 KV、标量点积、无 GQA 共享）：5 层 × 369 us。
// ---------------------------------------------------------------------------------------------------------------
constexpr int D5_D = 512;
constexpr int D5_THREADS = 256;
constexpr int D5_LPK = D5_D / V3_VEC;          // 64 lane / key
constexpr int D5_KEYS = D5_THREADS / D5_LPK;   // 4 key / tile
template <int GROUP>
__global__ void __launch_bounds__(D5_THREADS) gqa_decode_d512_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ cache, const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens, const int32_t* __restrict__ cu_seqlens_q, float* __restrict__ scratch,
    bf16* __restrict__ out_direct, int H, int Hkv, int block_size, int max_blocks, float scale, int q_stride, int splits,
    int keys_per_split, int window) {
  constexpr int D = D5_D, KEYS = D5_KEYS, LPK = D5_LPK;
  const int b = blockIdx.x, hk = blockIdx.y, split = blockIdx.z, tid = threadIdx.x;
  int row = b;
  if (cu_seqlens_q) {
    if (cu_seqlens_q[b + 1] - cu_seqlens_q[b] != 1) return;
    row = cu_seqlens_q[b];
  }
  const int n = seq_lens[b];
  const int k0 = window > 0 ? max(0, n - window) : 0;
  const int nk = n - k0;
  const int eff = min(splits, max(1, (nk + keys_per_split - 1) / keys_per_split));
  if (split >= eff) return;  // 块一致：全体一起退
  const int seg = (nk + eff - 1) / eff;
  const int k_begin = k0 + split * seg, k_end = min(n, k_begin + seg);
  const int ks = tid / LPK;               // key 槽 0..3
  const int dl = (tid % LPK) * V3_VEC;    // 本线程起始维度
  const int warp = tid >> 5, lane = tid & 31, half = warp & 1;  // 一个 key = warp 对 (2ks, 2ks+1)
  const int32_t* bt = block_table + (int64_t)b * max_blocks;
  __shared__ __align__(16) bf16 s_k[V3_PIPE][KEYS][D];
  __shared__ __align__(16) bf16 s_v[V3_PIPE][KEYS][D];
  __shared__ float s_part[GROUP][KEYS][2];
  __shared__ float s_m[GROUP][KEYS], s_l[GROUP][KEYS];
  __shared__ float s_acc[KEYS][D];
  float qr[GROUP][V3_VEC];
  const float scale_log2 = scale * 1.4426950408889634f;
#pragma unroll
  for (int g = 0; g < GROUP; ++g) {
    const uint4 r = *reinterpret_cast<const uint4*>(q + (int64_t)row * q_stride + (int64_t)(hk * GROUP + g) * D + dl);
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
  const int ntiles = (k_end - k_begin + KEYS - 1) / KEYS;
  auto issue = [&](int t, int s) {
    const int key = k_begin + t * KEYS + ks;
    const bf16* slotp = cache;
    int bytes = 0;
    if (key < k_end) {
      const int64_t blk = bt[key / block_size];
      slotp = cache + slot_off3<D>(blk * block_size + (key % block_size), Hkv);
      bytes = 16;
    }
    cp16z(&s_k[s][ks][dl], slotp + (hk * 2) * D + dl, bytes);
    cp16z(&s_v[s][ks][dl], slotp + (hk * 2 + 1) * D + dl, bytes);
    cp_commit();
  };
#pragma unroll
  for (int s = 0; s < V3_PIPE; ++s) issue(s, s);
  for (int t = 0; t < ntiles; ++t) {
    const int s = t % V3_PIPE;
    cp_wait<V3_PIPE - 1>();
    __syncthreads();
    const int key = k_begin + t * KEYS + ks;
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
    // 点积：warp 内 32 lane 归约 → 两 warp 的部分和经 smem 相加
#pragma unroll
    for (int g = 0; g < GROUP; ++g) {
      float part = 0.f;
#pragma unroll
      for (int c = 0; c < V3_VEC; ++c) part += qr[g][c] * kf[c];
      part = sum_lpk<32>(part);
      if (lane == 0) s_part[g][ks][half] = part;
    }
    __syncthreads();
#pragma unroll
    for (int g = 0; g < GROUP; ++g) {
      const float sc = s_part[g][ks][0] + s_part[g][ks][1];
      const float s_val = valid ? sc : -INFINITY;
      const float m_new = fmaxf(m[g], s_val);
      const float alpha = (m[g] == -INFINITY) ? 0.f : exp2f(m[g] - m_new);
      const float p = valid ? exp2f(s_val - m_new) : 0.f;
      l[g] = l[g] * alpha + p;
#pragma unroll
      for (int c = 0; c < V3_VEC; ++c) acc[g][c] = acc[g][c] * alpha + p * vf[c];
      m[g] = m_new;
    }
    __syncthreads();  // 本 stage / s_part 读完才能被覆盖
    if (t + V3_PIPE < ntiles) issue(t + V3_PIPE, s); else cp_commit();
  }
  cp_wait<0>();
  // 跨 4 个 key 槽合并：逐 g（s_acc 只放一个 g）
#pragma unroll 1
  for (int g = 0; g < GROUP; ++g) {
    if ((tid % LPK) == 0) { s_m[g][ks] = m[g]; s_l[g][ks] = l[g]; }
#pragma unroll
    for (int c = 0; c < V3_VEC; ++c) s_acc[ks][dl + c] = acc[g][c];
    __syncthreads();
    float M = -INFINITY;
#pragma unroll
    for (int k = 0; k < KEYS; ++k) M = fmaxf(M, s_m[g][k]);
    const int h = hk * GROUP + g;
    for (int d = tid; d < D; d += D5_THREADS) {
      float L = 0.f, O = 0.f;
#pragma unroll
      for (int k = 0; k < KEYS; ++k) {
        const float a = (s_m[g][k] == -INFINITY) ? 0.f : exp2f(s_m[g][k] - M);
        L += s_l[g][k] * a;
        O += s_acc[k][d] * a;
      }
      if (out_direct) {
        out_direct[((int64_t)row * H + h) * D + d] = to_bf16(L > 0.f ? O / L : 0.f);
      } else {
        float* part = scratch + (((int64_t)b * H + h) * splits + split) * (D + 2);
        if (d == 0) { part[0] = (M == -INFINITY) ? -INFINITY : M * 0.6931471805599453f; part[1] = L; }
        part[2 + d] = O;
      }
    }
    __syncthreads();  // s_acc 被下一个 g 覆盖前读完
  }
}

}  // namespace

extern "C" int gqa_decode_v3_bf16(const void* q, const void* cache, const int32_t* block_table, const int32_t* seq_lens,
                                  const int32_t* cu_seqlens_q, float* scratch, void* out_direct, int64_t num_reqs,
                                  int64_t heads, int64_t kv_heads, int64_t head_dim, int64_t block_size,
                                  int64_t max_blocks, float scale, int64_t q_stride, int64_t splits,
                                  int64_t keys_per_split, int window, cudaStream_t stream) {
  const int group = (int)(heads / kv_heads);
  if (head_dim != 128 && head_dim != 256 && head_dim != 512) return (int)cudaErrorInvalidValue;
  dim3 grid((unsigned)num_reqs, (unsigned)kv_heads, (unsigned)splits);
  if (head_dim == 512) {
#define D5_LAUNCH(G)                                                                                                   \
  gqa_decode_d512_kernel<G><<<grid, D5_THREADS, 0, stream>>>(                                                          \
      static_cast<const bf16*>(q), static_cast<const bf16*>(cache), block_table, seq_lens, cu_seqlens_q, scratch,     \
      static_cast<bf16*>(out_direct), (int)heads, (int)kv_heads, (int)block_size, (int)max_blocks, scale,             \
      (int)q_stride, (int)splits, (int)keys_per_split, window)
    switch (group) {
      case 1: D5_LAUNCH(1); break;
      case 2: D5_LAUNCH(2); break;
      case 4: D5_LAUNCH(4); break;
      case 8: D5_LAUNCH(8); break;
      default: return (int)cudaErrorInvalidValue;
    }
#undef D5_LAUNCH
    return (int)cudaGetLastError();
  }
#define V3_LAUNCH(G, DD)                                                                                               \
  gqa_decode_v3_kernel<G, DD><<<grid, V3_THREADS, 0, stream>>>(                                                        \
      static_cast<const bf16*>(q), static_cast<const bf16*>(cache), block_table, seq_lens, cu_seqlens_q, scratch,     \
      static_cast<bf16*>(out_direct), (int)heads, (int)kv_heads, (int)block_size, (int)max_blocks, scale,             \
      (int)q_stride, (int)splits, (int)keys_per_split, window)
  if (head_dim == 256) {
    switch (group) {
      case 1: V3_LAUNCH(1, 256); break;
      case 2: V3_LAUNCH(2, 256); break;
      case 4: V3_LAUNCH(4, 256); break;
      case 6: V3_LAUNCH(6, 256); break;
      default: return (int)cudaErrorInvalidValue;
    }
  } else {
    switch (group) {
      case 1: V3_LAUNCH(1, 128); break;
      case 2: V3_LAUNCH(2, 128); break;
      case 4: V3_LAUNCH(4, 128); break;
      default: return (int)cudaErrorInvalidValue;
    }
  }
#undef V3_LAUNCH
  return (int)cudaGetLastError();
}
