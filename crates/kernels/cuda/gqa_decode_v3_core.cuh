// v3 decode attention 的常量 / 原语 / flash-decode **core device 函数**（从 gqa_decode_v3.cu 抽出，B1 kaon3-mega R36）。
// 既被 gqa_decode_v3.cu 的原入口 kernel 调，也被 persistent megakernel（mega_attn_sm120.cu）的 Dispatch 调。
// 放在有名字的 namespace dinfer::gqa_v3，避免匿名 namespace 导致的跨 TU 不可见。
#pragma once
#include "common.cuh"

namespace dinfer {
namespace gqa_v3 {

constexpr int V3_VEC = 8;                    // 每 lane 维数（16 B bf16）
constexpr int V3_THREADS = 128;              // 4 warp（本算法固定 128 线程：KEYS / 归约序都绑它）
constexpr int V3_PIPE = 2;                   // cp.async 流水级数
// D 模板化：LPK = D/8 lane / key（D=128 → 16 lane/key、每 warp 2 key、每 tile 8 key；D=256 → 32 lane/key、每 warp 1 key、每 tile 4 key）。
// tile 内 key 数 = 128 / LPK，故两种 D 的 K/V smem 都是 2 级 × 4 KB。
template <int D>
constexpr int V3_LPK = D / V3_VEC;
template <int D>
constexpr int V3_KEYS = V3_THREADS / V3_LPK<D>;

__device__ __forceinline__ void cp16(void* smem, const void* gmem) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a), "l"(gmem));
}
// src_bytes = 0 时零填充（越界 key）。必须用这个而不是"跳过不发"：smem 里的残留若是 inf/NaN，
// 后面 p=0 的 0·vf 仍会得 NaN 并污染累加器（2026-10-19 定位到的 D=128 混批间歇 NaN）。
__device__ __forceinline__ void cp16z(void* smem, const void* gmem, int src_bytes) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(a), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// 一个 key 的点积跨 LPK 个 lane 归约（LPK=16 只用 warp 低半、LPK=32 用整个 warp）。
template <int LPK>
__device__ __forceinline__ float sum_lpk(float v) {
  if (LPK == 32) v += __shfl_xor_sync(0xffffffffu, v, 16);
  v += __shfl_xor_sync(0xffffffffu, v, 8);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v;
}
template <int D>
__device__ __forceinline__ int64_t slot_off3(int64_t slot, int kv_heads) { return slot * kv_heads * 2 * D; }

// --- B1（kaon3-mega R36）：v3 的 Q·K·V flash-decode 主体 device 函数，供 persistent megakernel 调用。
// (b, hk, split, tid) 由调用方传入（原 kernel 从 blockIdx/threadIdx 取）；模板 GROUP/D 不变。
// 关键：这是 128 线程（V3_THREADS）算法——KEYS=128/LPK、跨 key 槽归约序都绑 128 线程。若外壳用 256 线程
// （mega::mega_kernel），只有 tid<V3_THREADS 的线程干活、其余陪跑 `__syncthreads()`（不碰 smem、ks 会越界）。
// 用 `active = tid < V3_THREADS` 门控所有 smem/global 访问；barrier 全体无条件进 → 128 线程 launch 时 active 恒真，逐位不变。
// 静态 __shared__ 与 mega 外壳的 extern 动态 smem 共存（本函数只用静态 smem，不碰传入的 dyn smem）。
// 方向三：一个 256 线程 CTA 两个 128 线程半区各跑一个独立 shard。HALVES=1（默认）= 现状（128 线程算法，
// 256 外壳只前 128 活 / 128 外壳全活），签名/行为/逐位完全不变；HALVES=2 = 整层 mega S_ATTN 用，
// half=tid>>7 取 shard_base+half 自己解码 (b,hk,split)，smem 按 half 分区，barrier 用命名 bar.sync(half+1,128) 两半区独立。
template <int GROUP, int D, int HALVES = 1>
__device__ __forceinline__ void gqa_decode_v3_core(
    const bf16* __restrict__ q, const bf16* __restrict__ cache, const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens, const int32_t* __restrict__ cu_seqlens_q, float* __restrict__ scratch,
    bf16* __restrict__ out_direct, int H, int Hkv, int block_size, int max_blocks, float scale, int q_stride, int splits,
    int keys_per_split, int b, int hk, int split, int tid, int shard_base = -1, int n_shards = -1) {
  // half/lt：HALVES=2 下 tid 0..255 分两半区（各 128 线程），每半区独立算一个 shard。
  const int half = (HALVES == 2) ? (tid >> 7) : 0;
  const int lt = (HALVES == 2) ? (tid & 127) : tid;  // 本半区内 0..127（= 原 128 线程算法的线程号）
  if (HALVES == 2) {
    const int shard = shard_base + half;
    if (n_shards >= 0 && shard >= n_shards) return;  // 奇数末对的越界半区：整半区一起退（独立 barrier 不死锁）
    split = shard % splits;
    const int bh = shard / splits;
    hk = bh % Hkv;
    b = bh / Hkv;
  }
  int row = b;
  if (cu_seqlens_q) {
    if (cu_seqlens_q[b + 1] - cu_seqlens_q[b] != 1) return;
    row = cu_seqlens_q[b];
  }
  const int n = seq_lens[b];
  const int eff = min(splits, max(1, (n + keys_per_split - 1) / keys_per_split));
  if (split >= eff) return;  // 块一致（b/split 全 CTA 同值）：全体一起退，不破 barrier
  const int seg = (n + eff - 1) / eff;
  const int k_begin = split * seg, k_end = min(n, k_begin + seg);

  constexpr int LPK = V3_LPK<D>, KEYS = V3_KEYS<D>;
  // HALVES=1：active=tid<128（128 外壳全活 / 256 外壳前 128 活，同现状）。HALVES=2：lt 恒 0..127 → 全活。
  const bool active = lt < V3_THREADS;
  const int ks = lt / LPK;                 // 0..KEYS-1：本线程的 key 槽（仅 active 时有效）
  const int dl = (lt % LPK) * V3_VEC;      // 本线程起始维度
  // barrier：HALVES=1 用全块 __syncthreads()（逐位不变）；HALVES=2 用命名 barrier 让两半区各 128 线程独立同步
  // （两 shard ntiles 不同也不互相死锁）。bar.sync id=half+1（id 0 留给 __syncthreads）。
  auto bar = [&]() {
    if (HALVES == 2) {
      asm volatile("bar.sync %0, 128;\n" ::"r"(half + 1));
    } else {
      __syncthreads();
    }
  };
  const int32_t* bt = block_table + (int64_t)b * max_blocks;

  // smem 按 half 分区：HALVES=1 只用 [0]（与现状同大小、同布局 → 逐位不变）；HALVES=2 两半区各一份不互踩。
  __shared__ __align__(16) bf16 s_k_all[HALVES][V3_PIPE][KEYS][D];
  __shared__ __align__(16) bf16 s_v_all[HALVES][V3_PIPE][KEYS][D];
  __shared__ float s_m_all[HALVES][GROUP][KEYS], s_l_all[HALVES][GROUP][KEYS];
  __shared__ float s_acc_all[HALVES][GROUP][KEYS][D];
  auto s_k = s_k_all[half];
  auto s_v = s_v_all[half];
  auto s_m = s_m_all[half];
  auto s_l = s_l_all[half];
  auto s_acc = s_acc_all[half];

  // q：本 lane 的 8 维 × GROUP 个头，预乘 scale（log2 域：用 exp2f 省一次乘）
  float qr[GROUP][V3_VEC];
  const float scale_log2 = scale * 1.4426950408889634f;
  if (active) {
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
  }

  float m[GROUP], l[GROUP], acc[GROUP][V3_VEC];
#pragma unroll
  for (int g = 0; g < GROUP; ++g) {
    m[g] = -INFINITY; l[g] = 0.f;
#pragma unroll
    for (int c = 0; c < V3_VEC; ++c) acc[g][c] = 0.f;
  }

  const int ntiles = (k_end - k_begin + KEYS - 1) / KEYS;
  // 发出第 t 个 tile 的 K/V 读到 stage s（本线程负责 key = k_begin + t*8 + ks 的 dl..dl+8 维）。
  // 只 active 线程发：非 active 的 ks 会越界 s_k/s_v；其 cp.async 组计数也不该参与。
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
    cp_commit();  // 空组也 commit，保证 wait_group 计数一致
  };
  if (active) {
#pragma unroll
    for (int s = 0; s < V3_PIPE; ++s) issue(s, s);
  }

  for (int t = 0; t < ntiles; ++t) {
    const int s = t % V3_PIPE;
    if (active) cp_wait<V3_PIPE - 1>();  // 第 t 组已落地（之后发出的 PIPE-1 组可仍在飞）
    bar();
    if (active) {
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
#pragma unroll
      for (int g = 0; g < GROUP; ++g) {
        float part = 0.f;
#pragma unroll
        for (int c = 0; c < V3_VEC; ++c) part += qr[g][c] * kf[c];
        const float sc = sum_lpk<LPK>(part);  // LPK lane 内归约（LPK=16 时两 key 槽互不干扰：xor ≤ 8）
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
    }
    bar();  // 本 stage 读完才能被下一次 issue 覆盖
    if (active) { if (t + V3_PIPE < ntiles) issue(t + V3_PIPE, s); else cp_commit(); }
  }
  if (active) cp_wait<0>();

  // 跨 KEYS 个 key 槽合并（循环外只做一次）
  if (active) {
#pragma unroll
    for (int g = 0; g < GROUP; ++g) {
      if ((lt % LPK) == 0) { s_m[g][ks] = m[g]; s_l[g][ks] = l[g]; }
#pragma unroll
      for (int c = 0; c < V3_VEC; ++c) s_acc[g][ks][dl + c] = acc[g][c];
    }
  }
  bar();
  // 合并写：stride=V3_THREADS，索引用 lt（半区内 0..127）。256 外壳 tid>=128 会与 active 撞同一个 d → 必须门控（无 barrier，安全）。
  if (active)
  for (int d = lt; d < D; d += V3_THREADS) {
#pragma unroll
    for (int g = 0; g < GROUP; ++g) {
      float M = -INFINITY;
#pragma unroll
      for (int k = 0; k < KEYS; ++k) M = fmaxf(M, s_m[g][k]);
      float L = 0.f, O = 0.f;
#pragma unroll
      for (int k = 0; k < KEYS; ++k) {
        const float a = (s_m[g][k] == -INFINITY) ? 0.f : exp2f(s_m[g][k] - M);
        L += s_l[g][k] * a;
        O += s_acc[g][k][d] * a;
      }
      const int h = hk * GROUP + g;
      if (out_direct) {
        // 本 block 一个有效 key 都没有（seq_len 0 的占位行）：写 0 而不是 0/0
        out_direct[((int64_t)row * H + h) * D + d] = to_bf16(L > 0.f ? O / L : 0.f);
      } else {
        float* part = scratch + (((int64_t)b * H + h) * splits + split) * (D + 2);
        // m 存回自然对数域，与 v2 partial / merge 口径一致（merge 用 __expf 且跳过 -inf）；-inf 原样保留做"空段"标记
        if (d == 0) { part[0] = (M == -INFINITY) ? -INFINITY : M * 0.6931471805599453f; part[1] = L; }
        part[2 + d] = O;
      }
    }
  }
}

}  // namespace gqa_v3
}  // namespace dinfer
