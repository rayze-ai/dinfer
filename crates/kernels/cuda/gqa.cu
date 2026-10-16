// GQA（分组多头）paged attention 参考 kernel（sm80+，bf16 in / fp32 累加）。Qwen3 等标准 dense 模型用。
// 结构与 mla.cu 同（每 (req, q_head) 一个 block，逐 query 行三遍扫 key：max / sum / 加权累加），差别：
//   - K/V 分开存：paged cache 布局 [num_blocks, block_size, kv_heads, 2, D]（每 slot 内按 kv_head 排，K 在 V 前）；
//   - q_head h 读 kv_head h / group（group = H / Hkv）；
//   - 无 top-k 稀疏。
//
// 三个入口：
//   gqa_cache_write_bf16   k, v: [T, Hkv, D] → cache（slot_mapping）
//   gqa_prefill_bf16       无历史的 causal prefill，直接读 k/v [T, Hkv, D]（不走 cache；cache 由 cache_write 另写）
//                          D=128 走 flash 风格 tile kernel（Q3-a，2026-10-16），其他 D 走旧参考实现
//   gqa_paged_bf16         从 paged cache 读：decode（每请求 1 行）与带历史 chunked prefill（任意 query 长）统一路径
//                          D=128 同样走 flash 风格 tile kernel（读 paged cache），其他 D 走旧参考实现
//   gqa_decode_bf16        纯 decode（每请求 1 行）split-K 版（Q3-b）：reqs × Hkv × 8 split 个 block + merge kernel
#include "common.cuh"
#include "dinfer_kernels.h"
#include <mma.h>

using namespace dinfer;

namespace {

constexpr int GQ_BLOCK = 256;
constexpr int MAX_D = 256;

__device__ __forceinline__ int64_t slot_off(int64_t slot, int kv_heads, int D) { return slot * kv_heads * 2 * D; }

__global__ void gqa_cache_write_kernel(const bf16* __restrict__ k, const bf16* __restrict__ v, bf16* __restrict__ cache,
                                       const int64_t* __restrict__ slot_mapping, int kv_heads, int D, int block_size) {
  const int64_t t = blockIdx.x;
  const int64_t slot = slot_mapping[t];
  if (slot < 0) return;
  const int64_t blk = slot / block_size, off = slot % block_size;
  bf16* dst = cache + slot_off(blk * block_size + off, kv_heads, D);
  const bf16* ks = k + t * kv_heads * D;
  const bf16* vs = v + t * kv_heads * D;
  for (int i = threadIdx.x; i < kv_heads * D; i += blockDim.x) {
    const int h = i / D, d = i % D;
    dst[(h * 2 + 0) * D + d] = ks[i];
    dst[(h * 2 + 1) * D + d] = vs[i];
  }
}

// 一个 (req, q_head) block；q/k/v: [T, H|Hkv, D]；out: [T, H, D]。
// prefill（无历史，causal）：flash 风格。grid = (ceil(max_seqlen / PF_WARPS), H, num_reqs)，block = PF_WARPS 个 warp，
// 每 warp 负责一个 query 行；K/V 按 PF_TILE 个 key 一片进 shared memory（K 转置存成 [D][TILE] 让 lane 读连续地址），
// 每 lane 算一个 key 的分数，warp 内 online softmax（m / l 随 tile 更新），PV 累加时 lane 持有 D/32 个维度、
// p_j 由 __shfl 广播。只扫到对角 tile（causal），fp32 累加、bf16 I/O。
// 复杂度 O(T²·D/2) 但并行度 = T/PF_WARPS × H × reqs 个 block（旧版 H × reqs 个 block 串行遍历 T 行）。
constexpr int PF_D = 128;               // Qwen3 head_dim；其他 D 走通用慢路径（见 host 分派）
constexpr int PF_DPL = PF_D / 32;       // 每 lane 持有的维度数

// prefill v2（Q3-e 续 3）：一个 block 处理一个 (req, kv_head) 的 PV2_BQ=32 行 query × 该 kv head 的 group 个 q head；
// 8 warp，每 warp 4 行 query（group=2 时即 2 行 × 2 head）；K/V tile 32 key 用 uint4 向量装进 smem（bf16 存，不转置）；
// 每 lane 对 tile 里 1 个 key（lane = key），对自己 4 行各算一个分数（128 维 FMA，q 在寄存器 16 维/lane × 8 lane 一行 → 改为
// 每行 q 在 smem、lane 读 K 列）。相对 v1：K/V 每 tile 服务 32×group=64 行（v1 4 行）→ 全局读量 /16；装载向量化。
constexpr int PV2_WARPS = 8;
constexpr int PV2_ROWS_PER_WARP = 4;      // 每 warp 的 (query 行, q head) 组合数
constexpr int PV2_TILE = 32;

__global__ void __launch_bounds__(PV2_WARPS * 32) gqa_prefill_flash_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v, bf16* __restrict__ out,
    const int32_t* __restrict__ cu_seqlens, int H, int Hkv, float scale, int q_stride, int kv_stride) {
  const int req = blockIdx.z, hk = blockIdx.y;
  const int group = H / Hkv;  // 1 或 2
  const int start = cu_seqlens[req], end = cu_seqlens[req + 1];
  const int T = end - start;
  const int rows_per_block = (PV2_WARPS * PV2_ROWS_PER_WARP) / group;  // group=2 → 16 rows… 注意：下面按组合数分配
  // 组合索引 c ∈ [0, PV2_WARPS*PV2_ROWS_PER_WARP)：row = c / group, g = c % group
  const int q_base = blockIdx.x * rows_per_block;
  if (q_base >= T) return;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;

  __shared__ __align__(16) bf16 sK[PV2_TILE][PF_D];
  __shared__ __align__(16) bf16 sV[PV2_TILE][PF_D];
  __shared__ float sQ[PV2_WARPS * PV2_ROWS_PER_WARP][PF_D];

  // 本 warp 的 4 个组合
  int rows[PV2_ROWS_PER_WARP], heads[PV2_ROWS_PER_WARP];
  bool act[PV2_ROWS_PER_WARP];
  int my_last = -1;
#pragma unroll
  for (int r = 0; r < PV2_ROWS_PER_WARP; ++r) {
    const int c = warp * PV2_ROWS_PER_WARP + r;
    rows[r] = q_base + c / group;
    heads[r] = hk * group + c % group;
    act[r] = rows[r] < T;
    if (act[r]) my_last = max(my_last, rows[r]);
  }
  // q → smem（每组合一行，已乘 scale）
#pragma unroll
  for (int r = 0; r < PV2_ROWS_PER_WARP; ++r) {
    const int c = warp * PV2_ROWS_PER_WARP + r;
    if (act[r]) {
      const bf16* qp = q + (int64_t)(start + rows[r]) * q_stride + (int64_t)heads[r] * PF_D;
      for (int d = lane; d < PF_D; d += 32) sQ[c][d] = to_f32(qp[d]) * scale;
    }
  }
  float m[PV2_ROWS_PER_WARP], l[PV2_ROWS_PER_WARP], acc[PV2_ROWS_PER_WARP][PF_DPL];
#pragma unroll
  for (int r = 0; r < PV2_ROWS_PER_WARP; ++r) {
    m[r] = -INFINITY; l[r] = 0.f;
#pragma unroll
    for (int c = 0; c < PF_DPL; ++c) acc[r][c] = 0.f;
  }
  const int block_last = min(q_base + rows_per_block - 1, T - 1);
  const int n_tiles = block_last / PV2_TILE + 1;
  for (int tile = 0; tile < n_tiles; ++tile) {
    const int k0 = tile * PV2_TILE;
    const int kn = min(PV2_TILE, T - k0);
    __syncthreads();
    // 装载：32 key × 128 bf16 = 32 × 16 个 uint4；256 线程每线程 2 个 uint4（K、V 各一）
    for (int e = threadIdx.x; e < PV2_TILE * (PF_D / 8); e += PV2_WARPS * 32) {
      const int j = e / (PF_D / 8), d8 = (e % (PF_D / 8)) * 8;
      uint4 kk = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
      if (j < kn) {
        const int64_t off = (int64_t)(start + k0 + j) * kv_stride + (int64_t)hk * PF_D + d8;
        kk = *reinterpret_cast<const uint4*>(k + off);
        vv = *reinterpret_cast<const uint4*>(v + off);
      }
      *reinterpret_cast<uint4*>(&sK[j][d8]) = kk;
      *reinterpret_cast<uint4*>(&sV[j][d8]) = vv;
    }
    __syncthreads();
    if (my_last < k0) continue;  // 本 warp 全部行都看不到这片
    // lane = key j；对 4 个组合各算一个分数
    float sc[PV2_ROWS_PER_WARP];
    {
      // 把本 lane 的 K 列取成 float（128 维）分批读，减少 smem 流量：按 8 维一组
      const bf16* kr = sK[lane];
#pragma unroll
      for (int r = 0; r < PV2_ROWS_PER_WARP; ++r) sc[r] = 0.f;
      for (int d8 = 0; d8 < PF_D; d8 += 8) {
        uint4 kk = *reinterpret_cast<const uint4*>(kr + d8);
        const __nv_bfloat162* kp = reinterpret_cast<const __nv_bfloat162*>(&kk);
        float kf[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) { float2 t2 = __bfloat1622float2(kp[i]); kf[2 * i] = t2.x; kf[2 * i + 1] = t2.y; }
#pragma unroll
        for (int r = 0; r < PV2_ROWS_PER_WARP; ++r) {
          const float* qr = sQ[warp * PV2_ROWS_PER_WARP + r] + d8;
#pragma unroll
          for (int i = 0; i < 8; ++i) sc[r] += qr[i] * kf[i];
        }
      }
    }
#pragma unroll
    for (int r = 0; r < PV2_ROWS_PER_WARP; ++r) {
      const int valid = act[r] ? min(kn, rows[r] - k0 + 1) : 0;  // causal：key ≤ row
      const float s_r = (lane < valid) ? sc[r] : -INFINITY;
      float tmax = s_r;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, o));
      if (tmax == -INFINITY) continue;  // 该行此片无可见 key（warp 一致：valid 对所有 lane 相同）
      const float m_new = fmaxf(m[r], tmax);
      const float alpha = __expf(m[r] - m_new);
      const float p = (lane < valid) ? __expf(s_r - m_new) : 0.f;
      float psum = p;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) psum += __shfl_xor_sync(0xffffffffu, psum, o);
      l[r] = l[r] * alpha + psum;
#pragma unroll
      for (int c = 0; c < PF_DPL; ++c) acc[r][c] *= alpha;
      m[r] = m_new;
      // PV：lane 持维度 lane+32c；遍历 valid 个 key，p 广播
      for (int j = 0; j < valid; ++j) {
        const float pj = __shfl_sync(0xffffffffu, p, j);
        const bf16* vr = sV[j];
#pragma unroll
        for (int c = 0; c < PF_DPL; ++c) acc[r][c] += pj * to_f32(vr[lane + 32 * c]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < PV2_ROWS_PER_WARP; ++r) {
    if (!act[r]) continue;
    const float inv = 1.f / l[r];
    bf16* op = out + ((int64_t)(start + rows[r]) * H + heads[r]) * PF_D;
#pragma unroll
    for (int c = 0; c < PF_DPL; ++c) op[lane + 32 * c] = to_bf16(acc[r][c] * inv);
  }
}

// prefill v3（Q3-e 续 4）：tensor core。block = (req, kv head, 32 行 query)；S = Q·Kᵀ 与 O += P·V 用 WMMA bf16 16×16×16。
// smem（动态，56 KB）：Q[2][32][128] bf16（预乘 scale）| K[32][128] bf16 | V[32][128] bf16 | S[2][32][32] f32 | P[2][32][32] bf16 |
// O[2][32][128] f32。流程每 tile：装 K/V → 8 warp 算 S 的 2 head × 4 片 → 每 warp 8 行(head,row) 做 online softmax（m/l 寄存器）、
// 写 P（bf16）并把 O 对应行乘 alpha → 8 warp 算 PV 的 2 head × 16 片各 2 片累加进 O。结束按 l 归一化写出。
using namespace nvcuda;
constexpr int V3_WARPS = 8;
constexpr int V3_BQ = 32;
constexpr int V3_TILE = 32;
constexpr int V3_G = 2;  // 最多 2 个 q head 共享一个 kv head
constexpr size_t V3_SMEM =
    (size_t)V3_G * V3_BQ * PF_D * 2 + 2 * (size_t)V3_TILE * PF_D * 2 + (size_t)V3_G * V3_BQ * V3_TILE * 4 +
    (size_t)V3_G * V3_BQ * V3_TILE * 2 + (size_t)V3_G * V3_BQ * PF_D * 4;  // 16K+16K+8K+4K+32K = 76 KB

__global__ void __launch_bounds__(V3_WARPS * 32) gqa_prefill_tc_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v, bf16* __restrict__ out,
    const int32_t* __restrict__ cu_seqlens, int H, int Hkv, float scale, int q_stride, int kv_stride) {
  extern __shared__ __align__(128) unsigned char v3_smem[];
  bf16* sQ = reinterpret_cast<bf16*>(v3_smem);                       // [G][BQ][D]
  bf16* sK = sQ + V3_G * V3_BQ * PF_D;                               // [TILE][D]
  bf16* sV = sK + V3_TILE * PF_D;                                    // [TILE][D]
  float* sS = reinterpret_cast<float*>(sV + V3_TILE * PF_D);         // [G][BQ][TILE]
  bf16* sP = reinterpret_cast<bf16*>(sS + V3_G * V3_BQ * V3_TILE);   // [G][BQ][TILE]
  float* sO = reinterpret_cast<float*>(sP + V3_G * V3_BQ * V3_TILE); // [G][BQ][D]

  const int req = blockIdx.z, hk = blockIdx.y;
  const int group = H / Hkv;
  const int start = cu_seqlens[req], end = cu_seqlens[req + 1];
  const int T = end - start;
  const int q_base = blockIdx.x * V3_BQ;
  if (q_base >= T) return;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nrows = min(V3_BQ, T - q_base);

  // Q → smem（bf16，预乘 scale），缺行补 0；O 清零
  for (int e = threadIdx.x; e < V3_G * V3_BQ * (PF_D / 8); e += V3_WARPS * 32) {
    const int g = e / (V3_BQ * (PF_D / 8)), rem = e % (V3_BQ * (PF_D / 8));
    const int r = rem / (PF_D / 8), d8 = (rem % (PF_D / 8)) * 8;
    uint4 val = make_uint4(0, 0, 0, 0);
    if (g < group && r < nrows) {
      const bf16* qp = q + (int64_t)(start + q_base + r) * q_stride + (int64_t)(hk * group + g) * PF_D + d8;
      uint4 raw = *reinterpret_cast<const uint4*>(qp);
      __nv_bfloat162* pr = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float2 f = __bfloat1622float2(pr[i]);
        pr[i] = __floats2bfloat162_rn(f.x * scale, f.y * scale);
      }
      val = raw;
    }
    *reinterpret_cast<uint4*>(sQ + ((size_t)g * V3_BQ + r) * PF_D + d8) = val;
  }
  for (int e = threadIdx.x; e < V3_G * V3_BQ * PF_D; e += V3_WARPS * 32) sO[e] = 0.f;

  // 每 warp 负责 8 个 (g, row) 的 softmax 状态：idx = warp*8 + i → g = idx / 32, row = idx % 32
  float m_r[8], l_r[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) { m_r[i] = -INFINITY; l_r[i] = 0.f; }

  const int block_last = q_base + nrows - 1;
  const int n_tiles = block_last / V3_TILE + 1;
  for (int tile = 0; tile < n_tiles; ++tile) {
    const int k0 = tile * V3_TILE;
    const int kn = min(V3_TILE, T - k0);
    __syncthreads();
    // 装 K/V tile（uint4），缺 key 补 0
    for (int e = threadIdx.x; e < V3_TILE * (PF_D / 8); e += V3_WARPS * 32) {
      const int j = e / (PF_D / 8), d8 = (e % (PF_D / 8)) * 8;
      uint4 kk = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
      if (j < kn) {
        const int64_t off = (int64_t)(start + k0 + j) * kv_stride + (int64_t)hk * PF_D + d8;
        kk = *reinterpret_cast<const uint4*>(k + off);
        vv = *reinterpret_cast<const uint4*>(v + off);
      }
      *reinterpret_cast<uint4*>(sK + j * PF_D + d8) = kk;
      *reinterpret_cast<uint4*>(sV + j * PF_D + d8) = vv;
    }
    __syncthreads();
    // S = Q·Kᵀ：G×(2×2) 个 16×16 片 = 最多 8 片，一 warp 一片（group=1 时后 4 个 warp 空转）
    {
      const int g = warp / 4, tr = (warp % 4) / 2, tc = warp % 2;
      if (g < group) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
        wmma::fill_fragment(c, 0.f);
#pragma unroll
        for (int kk = 0; kk < PF_D; kk += 16) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> b;
          wmma::load_matrix_sync(a, sQ + ((size_t)g * V3_BQ + tr * 16) * PF_D + kk, PF_D);
          wmma::load_matrix_sync(b, sK + (size_t)(tc * 16) * PF_D + kk, PF_D);  // Kᵀ 的 col_major == K 的 row_major
          wmma::mma_sync(c, a, b, c);
        }
        wmma::store_matrix_sync(sS + ((size_t)g * V3_BQ + tr * 16) * V3_TILE + tc * 16, c, V3_TILE, wmma::mem_row_major);
      }
    }
    __syncthreads();
    // online softmax：每 warp 8 个 (g,row)；lane = key 列 j
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int idx = warp * 8 + i, g = idx / V3_BQ, row = idx % V3_BQ;
      bf16* prow = sP + ((size_t)g * V3_BQ + row) * V3_TILE;
      const bool live = g < group && row < nrows;
      const int valid = live ? min(kn, q_base + row - k0 + 1) : 0;  // causal
      float sc = -INFINITY;
      if (lane < valid) sc = sS[((size_t)g * V3_BQ + row) * V3_TILE + lane];
      float tmax = sc;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, o));
      float p = 0.f, alpha = 1.f;
      if (tmax != -INFINITY) {
        const float m_new = fmaxf(m_r[i], tmax);
        alpha = __expf(m_r[i] - m_new);
        p = (lane < valid) ? __expf(sc - m_new) : 0.f;
        float psum = p;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) psum += __shfl_xor_sync(0xffffffffu, psum, o);
        l_r[i] = l_r[i] * alpha + psum;
        m_r[i] = m_new;
      }
      prow[lane] = to_bf16(p);
      if (alpha != 1.f) {
        float* orow = sO + ((size_t)g * V3_BQ + row) * PF_D;
#pragma unroll
        for (int c = 0; c < PF_DPL; ++c) orow[lane + 32 * c] *= alpha;
      }
    }
    __syncthreads();
    // O += P·V：G×(2×8) 片 = 最多 32 片，8 warp 各 4 片（group=1 时各 2 片）
    for (int f = warp; f < group * 16; f += V3_WARPS) {
      const int g = f / 16, tr = (f % 16) / 8, tcol = f % 8;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
      float* optr = sO + ((size_t)g * V3_BQ + tr * 16) * PF_D + tcol * 16;
      wmma::load_matrix_sync(c, optr, PF_D, wmma::mem_row_major);
#pragma unroll
      for (int kk = 0; kk < V3_TILE; kk += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b;
        wmma::load_matrix_sync(a, sP + ((size_t)g * V3_BQ + tr * 16) * V3_TILE + kk, V3_TILE);
        wmma::load_matrix_sync(b, sV + (size_t)kk * PF_D + tcol * 16, PF_D);
        wmma::mma_sync(c, a, b, c);
      }
      wmma::store_matrix_sync(optr, c, PF_D, wmma::mem_row_major);
    }
  }
  __syncthreads();
  // 归一化写出：每 warp 自己的 8 个 (g,row)
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int idx = warp * 8 + i, g = idx / V3_BQ, row = idx % V3_BQ;
    if (g >= group || row >= nrows) continue;
    const float inv = l_r[i] > 0.f ? 1.f / l_r[i] : 0.f;
    const float* orow = sO + ((size_t)g * V3_BQ + row) * PF_D;
    bf16* op = out + ((int64_t)(start + q_base + row) * H + hk * group + g) * PF_D;
#pragma unroll
    for (int c = 0; c < PF_DPL; ++c) op[lane + 32 * c] = to_bf16(orow[lane + 32 * c] * inv);
  }
}

// paged v3（tensor core，与 prefill v3 同构）：K/V tile 从 paged cache 经 block_table 装载；可见 key = [0, hist + row]。
__global__ void __launch_bounds__(V3_WARPS * 32) gqa_paged_tc_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ cache, const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens, const int32_t* __restrict__ cu_seqlens_q, bf16* __restrict__ out, int H,
    int Hkv, int block_size, int max_blocks, float scale, int q_stride) {
  extern __shared__ __align__(128) unsigned char v3_smem[];
  bf16* sQ = reinterpret_cast<bf16*>(v3_smem);
  bf16* sK = sQ + V3_G * V3_BQ * PF_D;
  bf16* sV = sK + V3_TILE * PF_D;
  float* sS = reinterpret_cast<float*>(sV + V3_TILE * PF_D);
  bf16* sP = reinterpret_cast<bf16*>(sS + V3_G * V3_BQ * V3_TILE);
  float* sO = reinterpret_cast<float*>(sP + V3_G * V3_BQ * V3_TILE);

  const int b = blockIdx.z, hk = blockIdx.y;
  const int group = H / Hkv;
  const int q0 = cu_seqlens_q[b], q1 = cu_seqlens_q[b + 1];
  const int q_len = q1 - q0;
  const int q_base = blockIdx.x * V3_BQ;
  if (q_base >= q_len) return;
  const int seq_len = seq_lens[b];
  const int hist = seq_len - q_len;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nrows = min(V3_BQ, q_len - q_base);
  const int32_t* bt = block_table + (int64_t)b * max_blocks;

  for (int e = threadIdx.x; e < V3_G * V3_BQ * (PF_D / 8); e += V3_WARPS * 32) {
    const int g = e / (V3_BQ * (PF_D / 8)), rem = e % (V3_BQ * (PF_D / 8));
    const int r = rem / (PF_D / 8), d8 = (rem % (PF_D / 8)) * 8;
    uint4 val = make_uint4(0, 0, 0, 0);
    if (g < group && r < nrows) {
      const bf16* qp = q + (int64_t)(q0 + q_base + r) * q_stride + (int64_t)(hk * group + g) * PF_D + d8;
      uint4 raw = *reinterpret_cast<const uint4*>(qp);
      __nv_bfloat162* pr = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float2 f = __bfloat1622float2(pr[i]);
        pr[i] = __floats2bfloat162_rn(f.x * scale, f.y * scale);
      }
      val = raw;
    }
    *reinterpret_cast<uint4*>(sQ + ((size_t)g * V3_BQ + r) * PF_D + d8) = val;
  }
  for (int e = threadIdx.x; e < V3_G * V3_BQ * PF_D; e += V3_WARPS * 32) sO[e] = 0.f;
  float m_r[8], l_r[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) { m_r[i] = -INFINITY; l_r[i] = 0.f; }

  const int block_last = hist + q_base + nrows - 1;
  const int n_tiles = block_last / V3_TILE + 1;
  for (int tile = 0; tile < n_tiles; ++tile) {
    const int k0 = tile * V3_TILE;
    const int kn = min(V3_TILE, seq_len - k0);
    __syncthreads();
    for (int e = threadIdx.x; e < V3_TILE * (PF_D / 8); e += V3_WARPS * 32) {
      const int j = e / (PF_D / 8), d8 = (e % (PF_D / 8)) * 8;
      uint4 kk = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
      if (j < kn) {
        const int pos = k0 + j;
        const int64_t blk = bt[pos / block_size];
        const bf16* base = cache + slot_off(blk * block_size + (pos % block_size), Hkv, PF_D) + (hk * 2) * PF_D + d8;
        kk = *reinterpret_cast<const uint4*>(base);
        vv = *reinterpret_cast<const uint4*>(base + PF_D);
      }
      *reinterpret_cast<uint4*>(sK + j * PF_D + d8) = kk;
      *reinterpret_cast<uint4*>(sV + j * PF_D + d8) = vv;
    }
    __syncthreads();
    {
      const int g = warp / 4, tr = (warp % 4) / 2, tc = warp % 2;
      if (g < group) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
        wmma::fill_fragment(c, 0.f);
#pragma unroll
        for (int kk = 0; kk < PF_D; kk += 16) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> bfr;
          wmma::load_matrix_sync(a, sQ + ((size_t)g * V3_BQ + tr * 16) * PF_D + kk, PF_D);
          wmma::load_matrix_sync(bfr, sK + (size_t)(tc * 16) * PF_D + kk, PF_D);
          wmma::mma_sync(c, a, bfr, c);
        }
        wmma::store_matrix_sync(sS + ((size_t)g * V3_BQ + tr * 16) * V3_TILE + tc * 16, c, V3_TILE, wmma::mem_row_major);
      }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int idx = warp * 8 + i, g = idx / V3_BQ, row = idx % V3_BQ;
      bf16* prow = sP + ((size_t)g * V3_BQ + row) * V3_TILE;
      const bool live = g < group && row < nrows;
      const int valid = live ? min(kn, hist + q_base + row - k0 + 1) : 0;
      float sc = -INFINITY;
      if (lane < valid) sc = sS[((size_t)g * V3_BQ + row) * V3_TILE + lane];
      float tmax = sc;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, o));
      float p = 0.f, alpha = 1.f;
      if (tmax != -INFINITY) {
        const float m_new = fmaxf(m_r[i], tmax);
        alpha = __expf(m_r[i] - m_new);
        p = (lane < valid) ? __expf(sc - m_new) : 0.f;
        float psum = p;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) psum += __shfl_xor_sync(0xffffffffu, psum, o);
        l_r[i] = l_r[i] * alpha + psum;
        m_r[i] = m_new;
      }
      prow[lane] = to_bf16(p);
      if (alpha != 1.f) {
        float* orow = sO + ((size_t)g * V3_BQ + row) * PF_D;
#pragma unroll
        for (int c = 0; c < PF_DPL; ++c) orow[lane + 32 * c] *= alpha;
      }
    }
    __syncthreads();
    for (int f = warp; f < group * 16; f += V3_WARPS) {
      const int g = f / 16, tr = (f % 16) / 8, tcol = f % 8;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
      float* optr = sO + ((size_t)g * V3_BQ + tr * 16) * PF_D + tcol * 16;
      wmma::load_matrix_sync(c, optr, PF_D, wmma::mem_row_major);
#pragma unroll
      for (int kk = 0; kk < V3_TILE; kk += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> bfr;
        wmma::load_matrix_sync(a, sP + ((size_t)g * V3_BQ + tr * 16) * V3_TILE + kk, V3_TILE);
        wmma::load_matrix_sync(bfr, sV + (size_t)kk * PF_D + tcol * 16, PF_D);
        wmma::mma_sync(c, a, bfr, c);
      }
      wmma::store_matrix_sync(optr, c, PF_D, wmma::mem_row_major);
    }
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int idx = warp * 8 + i, g = idx / V3_BQ, row = idx % V3_BQ;
    if (g >= group || row >= nrows) continue;
    const float inv = l_r[i] > 0.f ? 1.f / l_r[i] : 0.f;
    const float* orow = sO + ((size_t)g * V3_BQ + row) * PF_D;
    bf16* op = out + ((int64_t)(q0 + q_base + row) * H + hk * group + g) * PF_D;
#pragma unroll
    for (int c = 0; c < PF_DPL; ++c) op[lane + 32 * c] = to_bf16(orow[lane + 32 * c] * inv);
  }
}

// 通用慢路径（任意 D ≤ MAX_D）：旧参考实现，每 (req, q_head) 一个 block 逐 query 行三遍扫 key。对拍 fixture（D=32）走这里。
__global__ void gqa_prefill_ref_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
                                       bf16* __restrict__ out, const int32_t* __restrict__ cu_seqlens, int H, int Hkv, int D,
                                       float scale, int q_stride, int kv_stride) {

  const int req = blockIdx.x, h = blockIdx.y;
  const int hk = h / (H / Hkv);
  const int start = cu_seqlens[req], end = cu_seqlens[req + 1];
  const int T = end - start;
  if (T <= 0) return;
  __shared__ float s_q[MAX_D];
  __shared__ float s_p[GQ_BLOCK];
  __shared__ float s_stat[2];
  for (int i = 0; i < T; ++i) {
    const bf16* qi = q + (int64_t)(start + i) * q_stride + (int64_t)h * D;
    for (int d = threadIdx.x; d < D; d += GQ_BLOCK) s_q[d] = to_f32(qi[d]);
    __syncthreads();
    const int n = i + 1;
    auto kptr = [&](int j) { return k + (int64_t)(start + j) * kv_stride + (int64_t)hk * D; };
    auto vptr = [&](int j) { return v + (int64_t)(start + j) * kv_stride + (int64_t)hk * D; };
    float lmax = -INFINITY;
    for (int j = threadIdx.x; j < n; j += GQ_BLOCK) {
      const bf16* kj = kptr(j);
      float s = 0.f;
      for (int d = 0; d < D; ++d) s += s_q[d] * to_f32(kj[d]);
      lmax = fmaxf(lmax, s * scale);
    }
    const float m = block_reduce_max<GQ_BLOCK>(lmax);
    float lsum = 0.f;
    for (int j = threadIdx.x; j < n; j += GQ_BLOCK) {
      const bf16* kj = kptr(j);
      float s = 0.f;
      for (int d = 0; d < D; ++d) s += s_q[d] * to_f32(kj[d]);
      lsum += __expf(s * scale - m);
    }
    const float denom = block_reduce_sum<GQ_BLOCK>(lsum);
    if (threadIdx.x == 0) {
      s_stat[0] = m;
      s_stat[1] = denom;
    }
    __syncthreads();
    float acc[MAX_D / GQ_BLOCK + 1];
#pragma unroll
    for (int c = 0; c < MAX_D / GQ_BLOCK + 1; ++c) acc[c] = 0.f;
    for (int base = 0; base < n; base += GQ_BLOCK) {
      const int j = base + threadIdx.x;
      float p = 0.f;
      if (j < n) {
        const bf16* kj = kptr(j);
        float s = 0.f;
        for (int d = 0; d < D; ++d) s += s_q[d] * to_f32(kj[d]);
        p = __expf(s * scale - s_stat[0]) / s_stat[1];
      }
      s_p[threadIdx.x] = p;
      __syncthreads();
      const int cnt = min(GQ_BLOCK, n - base);
      for (int d = threadIdx.x, c = 0; d < D; d += GQ_BLOCK, ++c) {
        float a = 0.f;
        for (int jj = 0; jj < cnt; ++jj) a += s_p[jj] * to_f32(vptr(base + jj)[d]);
        acc[c] += a;
      }
      __syncthreads();
    }
    bf16* oi = out + ((int64_t)(start + i) * H + h) * D;
    for (int d = threadIdx.x, c = 0; d < D; d += GQ_BLOCK, ++c) oi[d] = to_bf16(acc[c]);
    __syncthreads();
  }
}

// 一个 (req, q_head) block；q: [Tq, H, D]（Tq = Σ query 长）；cache 见上；cu_seqlens_q: [B+1]；seq_lens: [B] = 本步后总长。
// query 行 i 的绝对位置 pos = seq_len - q_len + i，可见 key = [0, pos]。decode 即 q_len = 1。
__global__ void gqa_paged_kernel(const bf16* __restrict__ q, const bf16* __restrict__ cache,
                                 const int32_t* __restrict__ block_table, const int32_t* __restrict__ seq_lens,
                                 const int32_t* __restrict__ cu_seqlens_q, bf16* __restrict__ out, int H, int Hkv, int D,
                                 int block_size, int max_blocks, float scale, int q_stride) {
  const int b = blockIdx.x, h = blockIdx.y;
  const int hk = h / (H / Hkv);
  const int q0 = cu_seqlens_q[b], q1 = cu_seqlens_q[b + 1];
  const int q_len = q1 - q0;
  const int seq_len = seq_lens[b];
  if (q_len <= 0) return;
  const int hist = seq_len - q_len;
  __shared__ float s_q[MAX_D];
  __shared__ float s_p[GQ_BLOCK];
  __shared__ float s_stat[2];
  const int32_t* bt = block_table + (int64_t)b * max_blocks;
  auto kptr = [&](int j) {
    const int64_t blk = bt[j / block_size];
    return cache + slot_off(blk * block_size + (j % block_size), Hkv, D) + (hk * 2 + 0) * D;
  };
  auto vptr = [&](int j) {
    const int64_t blk = bt[j / block_size];
    return cache + slot_off(blk * block_size + (j % block_size), Hkv, D) + (hk * 2 + 1) * D;
  };
  for (int i = 0; i < q_len; ++i) {
    const int row = q0 + i;
    const int n = hist + i + 1;
    const bf16* qi = q + (int64_t)row * q_stride + (int64_t)h * D;
    for (int d = threadIdx.x; d < D; d += GQ_BLOCK) s_q[d] = to_f32(qi[d]);
    __syncthreads();
    float lmax = -INFINITY;
    for (int j = threadIdx.x; j < n; j += GQ_BLOCK) {
      const bf16* kj = kptr(j);
      float s = 0.f;
      for (int d = 0; d < D; ++d) s += s_q[d] * to_f32(kj[d]);
      lmax = fmaxf(lmax, s * scale);
    }
    const float m = block_reduce_max<GQ_BLOCK>(lmax);
    float lsum = 0.f;
    for (int j = threadIdx.x; j < n; j += GQ_BLOCK) {
      const bf16* kj = kptr(j);
      float s = 0.f;
      for (int d = 0; d < D; ++d) s += s_q[d] * to_f32(kj[d]);
      lsum += __expf(s * scale - m);
    }
    const float denom = block_reduce_sum<GQ_BLOCK>(lsum);
    if (threadIdx.x == 0) {
      s_stat[0] = m;
      s_stat[1] = denom;
    }
    __syncthreads();
    float acc[MAX_D / GQ_BLOCK + 1];
#pragma unroll
    for (int c = 0; c < MAX_D / GQ_BLOCK + 1; ++c) acc[c] = 0.f;
    for (int base = 0; base < n; base += GQ_BLOCK) {
      const int j = base + threadIdx.x;
      float p = 0.f;
      if (j < n) {
        const bf16* kj = kptr(j);
        float s = 0.f;
        for (int d = 0; d < D; ++d) s += s_q[d] * to_f32(kj[d]);
        p = __expf(s * scale - s_stat[0]) / s_stat[1];
      }
      s_p[threadIdx.x] = p;
      __syncthreads();
      const int cnt = min(GQ_BLOCK, n - base);
      for (int d = threadIdx.x, c = 0; d < D; d += GQ_BLOCK, ++c) {
        float a = 0.f;
        for (int jj = 0; jj < cnt; ++jj) a += s_p[jj] * to_f32(vptr(base + jj)[d]);
        acc[c] += a;
      }
      __syncthreads();
    }
    bf16* oi = out + ((int64_t)row * H + h) * D;
    for (int d = threadIdx.x, c = 0; d < D; d += GQ_BLOCK, ++c) oi[d] = to_bf16(acc[c]);
    __syncthreads();
  }
}


// paged（decode / 带历史 chunked prefill）的 flash 风格版本（D=128）：grid = (ceil(max_q_len / PF_WARPS), H, num_reqs)，
// 每 warp 一个 query 行；K/V 从 paged cache 按 PF_TILE 个 key 一片装进 shared memory（经 block_table 定位 slot）。
// query 行 i 的绝对位置 pos = hist + i，可见 key = [0, pos]。decode 时 q_len = 1 → 每 (req, head) 一个 block、一个活跃 warp，
// 比旧版（256 线程三遍扫）少两遍扫 + 省掉重复的 QK 点积；真正的 decode 并行度提升（split-K）留 Q3-b。
// paged v2（与 prefill v2 同构）：block = (req, kv_head, 32 行 query 块)，8 warp × 4 (row, head) 组合；K/V tile 从 paged cache
// 按 slot 用 uint4 装进 smem；可见 key = [0, hist + row]。decode（q_len=1）不走这里（走 split-K），这里服务带历史的 chunked prefill 混批。
__global__ void __launch_bounds__(PV2_WARPS * 32) gqa_paged_flash_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ cache, const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens, const int32_t* __restrict__ cu_seqlens_q, bf16* __restrict__ out, int H,
    int Hkv, int block_size, int max_blocks, float scale, int q_stride) {
  const int b = blockIdx.z, hk = blockIdx.y;
  const int group = H / Hkv;
  const int q0 = cu_seqlens_q[b], q1 = cu_seqlens_q[b + 1];
  const int q_len = q1 - q0;
  const int rows_per_block = (PV2_WARPS * PV2_ROWS_PER_WARP) / group;
  const int q_base = blockIdx.x * rows_per_block;
  if (q_base >= q_len) return;
  const int seq_len = seq_lens[b];
  const int hist = seq_len - q_len;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int32_t* bt = block_table + (int64_t)b * max_blocks;

  __shared__ __align__(16) bf16 sK[PV2_TILE][PF_D];
  __shared__ __align__(16) bf16 sV[PV2_TILE][PF_D];
  __shared__ float sQ[PV2_WARPS * PV2_ROWS_PER_WARP][PF_D];

  int rows[PV2_ROWS_PER_WARP], heads[PV2_ROWS_PER_WARP];
  bool act[PV2_ROWS_PER_WARP];
  int my_last = -1;
#pragma unroll
  for (int r = 0; r < PV2_ROWS_PER_WARP; ++r) {
    const int c = warp * PV2_ROWS_PER_WARP + r;
    rows[r] = q_base + c / group;
    heads[r] = hk * group + c % group;
    act[r] = rows[r] < q_len;
    if (act[r]) my_last = max(my_last, hist + rows[r]);
  }
#pragma unroll
  for (int r = 0; r < PV2_ROWS_PER_WARP; ++r) {
    const int c = warp * PV2_ROWS_PER_WARP + r;
    if (act[r]) {
      const bf16* qp = q + (int64_t)(q0 + rows[r]) * q_stride + (int64_t)heads[r] * PF_D;
      for (int d = lane; d < PF_D; d += 32) sQ[c][d] = to_f32(qp[d]) * scale;
    }
  }
  float m[PV2_ROWS_PER_WARP], l[PV2_ROWS_PER_WARP], acc[PV2_ROWS_PER_WARP][PF_DPL];
#pragma unroll
  for (int r = 0; r < PV2_ROWS_PER_WARP; ++r) {
    m[r] = -INFINITY; l[r] = 0.f;
#pragma unroll
    for (int c = 0; c < PF_DPL; ++c) acc[r][c] = 0.f;
  }
  const int block_last = hist + min(q_base + rows_per_block - 1, q_len - 1);
  const int n_tiles = block_last / PV2_TILE + 1;
  for (int tile = 0; tile < n_tiles; ++tile) {
    const int k0 = tile * PV2_TILE;
    const int kn = min(PV2_TILE, seq_len - k0);
    __syncthreads();
    for (int e = threadIdx.x; e < PV2_TILE * (PF_D / 8); e += PV2_WARPS * 32) {
      const int j = e / (PF_D / 8), d8 = (e % (PF_D / 8)) * 8;
      uint4 kk = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
      if (j < kn) {
        const int pos = k0 + j;
        const int64_t blk = bt[pos / block_size];
        const bf16* base = cache + slot_off(blk * block_size + (pos % block_size), Hkv, PF_D) + (hk * 2) * PF_D + d8;
        kk = *reinterpret_cast<const uint4*>(base);
        vv = *reinterpret_cast<const uint4*>(base + PF_D);
      }
      *reinterpret_cast<uint4*>(&sK[j][d8]) = kk;
      *reinterpret_cast<uint4*>(&sV[j][d8]) = vv;
    }
    __syncthreads();
    if (my_last < k0) continue;
    float sc[PV2_ROWS_PER_WARP];
    {
      const bf16* kr = sK[lane];
#pragma unroll
      for (int r = 0; r < PV2_ROWS_PER_WARP; ++r) sc[r] = 0.f;
      for (int d8 = 0; d8 < PF_D; d8 += 8) {
        uint4 kk = *reinterpret_cast<const uint4*>(kr + d8);
        const __nv_bfloat162* kp = reinterpret_cast<const __nv_bfloat162*>(&kk);
        float kf[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) { float2 t2 = __bfloat1622float2(kp[i]); kf[2 * i] = t2.x; kf[2 * i + 1] = t2.y; }
#pragma unroll
        for (int r = 0; r < PV2_ROWS_PER_WARP; ++r) {
          const float* qr = sQ[warp * PV2_ROWS_PER_WARP + r] + d8;
#pragma unroll
          for (int i = 0; i < 8; ++i) sc[r] += qr[i] * kf[i];
        }
      }
    }
#pragma unroll
    for (int r = 0; r < PV2_ROWS_PER_WARP; ++r) {
      const int valid = act[r] ? min(kn, hist + rows[r] - k0 + 1) : 0;
      const float s_r = (lane < valid) ? sc[r] : -INFINITY;
      float tmax = s_r;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, o));
      if (tmax == -INFINITY) continue;
      const float m_new = fmaxf(m[r], tmax);
      const float alpha = __expf(m[r] - m_new);
      const float p = (lane < valid) ? __expf(s_r - m_new) : 0.f;
      float psum = p;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) psum += __shfl_xor_sync(0xffffffffu, psum, o);
      l[r] = l[r] * alpha + psum;
#pragma unroll
      for (int c = 0; c < PF_DPL; ++c) acc[r][c] *= alpha;
      m[r] = m_new;
      for (int j = 0; j < valid; ++j) {
        const float pj = __shfl_sync(0xffffffffu, p, j);
        const bf16* vr = sV[j];
#pragma unroll
        for (int c = 0; c < PF_DPL; ++c) acc[r][c] += pj * to_f32(vr[lane + 32 * c]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < PV2_ROWS_PER_WARP; ++r) {
    if (!act[r]) continue;
    const float inv = 1.f / l[r];
    bf16* op = out + ((int64_t)(q0 + rows[r]) * H + heads[r]) * PF_D;
#pragma unroll
    for (int c = 0; c < PF_DPL; ++c) op[lane + 32 * c] = to_bf16(acc[r][c] * inv);
  }
}

// ---- decode（每请求 1 行 query）split-K（D=128）----
// grid = (num_reqs, Hkv, DC_SPLITS)：一个 block 负责一个 (req, kv_head) 的 1/DC_SPLITS 段 key，对该 kv_head 服务的全部
// q_head 各算一份 partial（m, l, acc[D]）；block 内 4 个 warp 再把段切 4 份，每 lane 一个 key 直接从 cache 流式读
//（decode 无 K/V 复用，不走 smem tile），warp 内 online softmax，warp 间在 smem 合并；写 scratch。
// 第二个 kernel 按 (req, q_head) 把 DC_SPLITS 个 partial 做 log-sum-exp 合并并归一化。
// 并行度：reqs × Hkv × 8 个 block × 4 warp（旧版 reqs × H 个 block × 1 活跃 warp）；每 key 的 K 只读一次服务 group 个 q_head。
constexpr int DC_SPLITS = 8;
constexpr int DC_WARPS = 4;

__global__ void __launch_bounds__(DC_WARPS * 32) gqa_decode_split_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ cache, const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens, float* __restrict__ scratch, int H, int Hkv, int block_size, int max_blocks,
    float scale, int q_stride) {
  const int b = blockIdx.x, hk = blockIdx.y, split = blockIdx.z;
  const int group = H / Hkv;
  const int n = seq_lens[b];
  const int seg = (n + DC_SPLITS - 1) / DC_SPLITS;
  const int k_begin = split * seg, k_end = min(n, k_begin + seg);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int32_t* bt = block_table + (int64_t)b * max_blocks;
  __shared__ float s_m[DC_WARPS], s_l[DC_WARPS];
  __shared__ float s_acc[DC_WARPS][PF_D];
  __shared__ float s_q[PF_D];

  for (int g = 0; g < group; ++g) {
    const int h = hk * group + g;
    float* part = scratch + (((int64_t)b * H + h) * DC_SPLITS + split) * (PF_D + 2);
    __syncthreads();
    for (int d = threadIdx.x; d < PF_D; d += DC_WARPS * 32)
      s_q[d] = to_f32(q[(int64_t)b * q_stride + (int64_t)h * PF_D + d]) * scale;
    __syncthreads();
    float m = -INFINITY, l = 0.f;
    float acc[PF_DPL];
#pragma unroll
    for (int c = 0; c < PF_DPL; ++c) acc[c] = 0.f;
    if (k_begin < k_end) {
      // warp 负责的 key 子段
      const int wlen = (k_end - k_begin + DC_WARPS - 1) / DC_WARPS;
      const int w0 = k_begin + warp * wlen, w1 = min(k_end, w0 + wlen);
      for (int base = w0; base < w1; base += 32) {
        const int j = base + lane;
        float sc = -INFINITY;
        const bf16* kp = nullptr;
        if (j < w1) {
          const int64_t blk = bt[j / block_size];
          kp = cache + slot_off(blk * block_size + (j % block_size), Hkv, PF_D) + (hk * 2) * PF_D;
          float a = 0.f;
#pragma unroll 8
          for (int d = 0; d < PF_D; ++d) a += s_q[d] * to_f32(kp[d]);
          sc = a;
        }
        float tmax = sc;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, o));
        const float m_new = fmaxf(m, tmax);
        const float alpha = __expf(m - m_new);
        const float pj_mine = (j < w1) ? __expf(sc - m_new) : 0.f;
        float psum = pj_mine;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) psum += __shfl_xor_sync(0xffffffffu, psum, o);
        l = l * alpha + psum;
#pragma unroll
        for (int c = 0; c < PF_DPL; ++c) acc[c] *= alpha;
        m = m_new;
        const int cnt = min(32, w1 - base);
        for (int jj = 0; jj < cnt; ++jj) {
          const float pj = __shfl_sync(0xffffffffu, pj_mine, jj);
          // V 行：由持有该 key 的 lane 的指针推出；每 lane 读自己负责的维度
          const int key = base + jj;
          const int64_t blk = bt[key / block_size];
          const bf16* vp = cache + slot_off(blk * block_size + (key % block_size), Hkv, PF_D) + (hk * 2 + 1) * PF_D;
#pragma unroll
          for (int c = 0; c < PF_DPL; ++c) acc[c] += pj * to_f32(vp[lane + 32 * c]);
        }
      }
    }
    // warp 间合并
    if (lane == 0) {
      s_m[warp] = m;
      s_l[warp] = l;
    }
#pragma unroll
    for (int c = 0; c < PF_DPL; ++c) s_acc[warp][lane + 32 * c] = acc[c];
    __syncthreads();
    if (warp == 0) {
      float M = -INFINITY;
#pragma unroll
      for (int w = 0; w < DC_WARPS; ++w) M = fmaxf(M, s_m[w]);
      float L = 0.f;
      float o[PF_DPL];
#pragma unroll
      for (int c = 0; c < PF_DPL; ++c) o[c] = 0.f;
#pragma unroll
      for (int w = 0; w < DC_WARPS; ++w) {
        const float a = (s_m[w] == -INFINITY) ? 0.f : __expf(s_m[w] - M);
        L += s_l[w] * a;
#pragma unroll
        for (int c = 0; c < PF_DPL; ++c) o[c] += s_acc[w][lane + 32 * c] * a;
      }
      if (lane == 0) {
        part[0] = M;
        part[1] = L;
      }
#pragma unroll
      for (int c = 0; c < PF_DPL; ++c) part[2 + lane + 32 * c] = o[c];
    }
  }
}

// ---- decode v2（Q3-e 续）：向量化 + 一 warp 四 key ----
// grid = (num_reqs, Hkv, splits)，block = 4 warp。每 warp 一次处理 4 个 key：8 lane 一 key，每 lane 负责 16 维（两次 uint4 =
// 16 个 bf16 读 K），8-lane shuffle 归约得分数；group 个 q head（Qwen3 = 2）同时算（K 读一次服务两头）。
// online softmax 在 4-key 组内先做 lane 级，再归并到 warp 级。V 累加：每 lane 持 16 维（同 K 的分工），用 key 的 p 乘加。
// block_table 每 16 key（block_size 对齐段）查一次。splits 由 host 自适应。scratch 布局同 v1：[B][H][splits][D+2]。
constexpr int D2_WARPS = 4;
constexpr int D2_KPW = 4;                 // keys per warp-iteration
constexpr int D2_LPK = 8;                 // lanes per key
constexpr int D2_DPL = PF_D / D2_LPK;     // 16 dims per lane
constexpr int D2_MAXG = 2;                // group size (H / Hkv) supported in registers

__device__ __forceinline__ float sum8(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v;
}

__global__ void __launch_bounds__(D2_WARPS * 32) gqa_decode_v2_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ cache, const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens, float* __restrict__ scratch, bf16* __restrict__ out_direct, int H, int Hkv,
    int block_size, int max_blocks, float scale, int q_stride, int splits) {
  const int b = blockIdx.x, hk = blockIdx.y, split = blockIdx.z;
  const int group = H / Hkv;  // ≤ D2_MAXG
  const int n = seq_lens[b];
  const int seg = (n + splits - 1) / splits;
  const int k_begin = split * seg, k_end = min(n, k_begin + seg);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int kslot = lane / D2_LPK;          // 0..3：本 lane 处理的 key 槽
  const int dl = (lane % D2_LPK) * D2_DPL;  // 本 lane 的起始维度
  const int32_t* bt = block_table + (int64_t)b * max_blocks;

  __shared__ float s_q[D2_MAXG][PF_D];
  __shared__ float s_m[D2_MAXG][D2_WARPS], s_l[D2_MAXG][D2_WARPS];
  __shared__ float s_acc[D2_MAXG][D2_WARPS][PF_D];

  for (int g = 0; g < group; ++g)
    for (int d = threadIdx.x; d < PF_D; d += D2_WARPS * 32)
      s_q[g][d] = to_f32(q[(int64_t)b * q_stride + (int64_t)(hk * group + g) * PF_D + d]) * scale;
  __syncthreads();
  // 每 lane 把自己 16 维的 q 放寄存器（两个 head）
  float qreg[D2_MAXG][D2_DPL];
#pragma unroll
  for (int g = 0; g < D2_MAXG; ++g)
#pragma unroll
    for (int c = 0; c < D2_DPL; ++c) qreg[g][c] = (g < group) ? s_q[g][dl + c] : 0.f;

  float m[D2_MAXG], l[D2_MAXG], acc[D2_MAXG][D2_DPL];
#pragma unroll
  for (int g = 0; g < D2_MAXG; ++g) {
    m[g] = -INFINITY; l[g] = 0.f;
#pragma unroll
    for (int c = 0; c < D2_DPL; ++c) acc[g][c] = 0.f;
  }

  if (k_begin < k_end) {
    const int wlen = (k_end - k_begin + D2_WARPS - 1) / D2_WARPS;
    const int w0 = k_begin + warp * wlen, w1 = min(k_end, w0 + wlen);
    for (int base = w0; base < w1; base += D2_KPW) {
      const int key = base + kslot;
      const bool valid = key < w1;
      // K 行指针（valid 时）
      const bf16* kp = nullptr; const bf16* vp = nullptr;
      if (valid) {
        const int64_t blk = bt[key / block_size];
        const bf16* slotp = cache + slot_off(blk * block_size + (key % block_size), Hkv, PF_D);
        kp = slotp + (hk * 2) * PF_D + dl;
        vp = slotp + (hk * 2 + 1) * PF_D + dl;
      }
      // 读 16 个 bf16 的 K（两次 uint4）
      float kf[D2_DPL];
      if (valid) {
        const uint4* kp4 = reinterpret_cast<const uint4*>(kp);
        uint4 r0 = kp4[0], r1 = kp4[1];
        const __nv_bfloat162* p0 = reinterpret_cast<const __nv_bfloat162*>(&r0);
        const __nv_bfloat162* p1 = reinterpret_cast<const __nv_bfloat162*>(&r1);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float2 a = __bfloat1622float2(p0[i]);
          float2 c2 = __bfloat1622float2(p1[i]);
          kf[2 * i] = a.x; kf[2 * i + 1] = a.y;
          kf[8 + 2 * i] = c2.x; kf[8 + 2 * i + 1] = c2.y;
        }
      } else {
#pragma unroll
        for (int c = 0; c < D2_DPL; ++c) kf[c] = 0.f;
      }
      // V 同样向量读
      float vf[D2_DPL];
      if (valid) {
        const uint4* vp4 = reinterpret_cast<const uint4*>(vp);
        uint4 r0 = vp4[0], r1 = vp4[1];
        const __nv_bfloat162* p0 = reinterpret_cast<const __nv_bfloat162*>(&r0);
        const __nv_bfloat162* p1 = reinterpret_cast<const __nv_bfloat162*>(&r1);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float2 a = __bfloat1622float2(p0[i]);
          float2 c2 = __bfloat1622float2(p1[i]);
          vf[2 * i] = a.x; vf[2 * i + 1] = a.y;
          vf[8 + 2 * i] = c2.x; vf[8 + 2 * i + 1] = c2.y;
        }
      } else {
#pragma unroll
        for (int c = 0; c < D2_DPL; ++c) vf[c] = 0.f;
      }
#pragma unroll
      for (int g = 0; g < D2_MAXG; ++g) {
        if (g >= group) break;
        float part = 0.f;
#pragma unroll
        for (int c = 0; c < D2_DPL; ++c) part += qreg[g][c] * kf[c];
        // 全 warp 都要参与 shuffle（full mask），不能按 valid 分叉；无效 key 的 part 已为 0，归约后再置 -inf
        const float red = sum8(part);
        const float sc = valid ? red : -INFINITY;
        // 4 个 key 的 max：跨 kslot 归约（xor 8, 16）
        float kmax = sc;
        kmax = fmaxf(kmax, __shfl_xor_sync(0xffffffffu, kmax, 8));
        kmax = fmaxf(kmax, __shfl_xor_sync(0xffffffffu, kmax, 16));
        const float m_new = fmaxf(m[g], kmax);
        const float alpha = __expf(m[g] - m_new);
        const float p = valid ? __expf(sc - m_new) : 0.f;
        float psum = p;
        psum += __shfl_xor_sync(0xffffffffu, psum, 8);
        psum += __shfl_xor_sync(0xffffffffu, psum, 16);
        l[g] = l[g] * alpha + psum;
        // acc：本 lane 只累加自己 key 的贡献；4 个 key 的同一维度分散在 4 个 kslot 的 lane 上，最后归并
#pragma unroll
        for (int c = 0; c < D2_DPL; ++c) acc[g][c] = acc[g][c] * alpha + p * vf[c];
        m[g] = m_new;
      }
    }
  }
  // 1) 4 个 kslot 的 acc 归并到 kslot 0（同维度 lane：xor 8, 16）；m/l 在 4 个 kslot 上已一致
#pragma unroll
  for (int g = 0; g < D2_MAXG; ++g)
#pragma unroll
    for (int c = 0; c < D2_DPL; ++c) {
      float v = acc[g][c];
      v += __shfl_xor_sync(0xffffffffu, v, 8);
      v += __shfl_xor_sync(0xffffffffu, v, 16);
      acc[g][c] = v;
    }
  // 2) warp 间合并（smem）
  for (int g = 0; g < group; ++g) {
    if (lane < D2_LPK) {
#pragma unroll
      for (int c = 0; c < D2_DPL; ++c) s_acc[g][warp][dl + c] = acc[g][c];
    }
    if (lane == 0) { s_m[g][warp] = m[g]; s_l[g][warp] = l[g]; }
  }
  __syncthreads();
  if (warp == 0) {
    for (int g = 0; g < group; ++g) {
      const int h = hk * group + g;
      float* part = scratch + (((int64_t)b * H + h) * splits + split) * (PF_D + 2);
      float M = -INFINITY;
#pragma unroll
      for (int w = 0; w < D2_WARPS; ++w) M = fmaxf(M, s_m[g][w]);
      float L = 0.f;
      float o[PF_DPL];
#pragma unroll
      for (int c = 0; c < PF_DPL; ++c) o[c] = 0.f;
#pragma unroll
      for (int w = 0; w < D2_WARPS; ++w) {
        const float a = (s_m[g][w] == -INFINITY) ? 0.f : __expf(s_m[g][w] - M);
        L += s_l[g][w] * a;
#pragma unroll
        for (int c = 0; c < PF_DPL; ++c) o[c] += s_acc[g][w][lane + 32 * c] * a;
      }
      if (out_direct) {
        // splits == 1：直接归一化写输出，省掉 merge kernel
        const float inv = 1.f / L;
        bf16* op = out_direct + ((int64_t)b * H + h) * PF_D;
#pragma unroll
        for (int c = 0; c < PF_DPL; ++c) op[lane + 32 * c] = to_bf16(o[c] * inv);
      } else {
        if (lane == 0) { part[0] = M; part[1] = L; }
#pragma unroll
        for (int c = 0; c < PF_DPL; ++c) part[2 + lane + 32 * c] = o[c];
      }
    }
  }
}

__global__ void gqa_decode_merge_v2_kernel(const float* __restrict__ scratch, bf16* __restrict__ out, int H, int splits) {
  const int b = blockIdx.x, h = blockIdx.y;
  const float* base = scratch + ((int64_t)b * H + h) * splits * (PF_D + 2);
  float M = -INFINITY;
  for (int s = 0; s < splits; ++s) M = fmaxf(M, base[s * (PF_D + 2)]);
  float L = 0.f, o = 0.f;
  const int d = threadIdx.x;
  for (int s = 0; s < splits; ++s) {
    const float* p = base + s * (PF_D + 2);
    const float a = (p[0] == -INFINITY) ? 0.f : __expf(p[0] - M);
    L += p[1] * a;
    o += p[2 + d] * a;
  }
  out[((int64_t)b * H + h) * PF_D + d] = to_bf16(o / L);
}

__global__ void gqa_decode_merge_kernel(const float* __restrict__ scratch, bf16* __restrict__ out, int H) {
  const int b = blockIdx.x, h = blockIdx.y;
  const float* base = scratch + ((int64_t)b * H + h) * DC_SPLITS * (PF_D + 2);
  float M = -INFINITY;
#pragma unroll
  for (int s = 0; s < DC_SPLITS; ++s) M = fmaxf(M, base[s * (PF_D + 2)]);
  float L = 0.f;
  float o = 0.f;
  const int d = threadIdx.x;  // blockDim = PF_D
#pragma unroll
  for (int s = 0; s < DC_SPLITS; ++s) {
    const float* p = base + s * (PF_D + 2);
    const float a = (p[0] == -INFINITY) ? 0.f : __expf(p[0] - M);
    L += p[1] * a;
    o += p[2 + d] * a;
  }
  out[((int64_t)b * H + h) * PF_D + d] = to_bf16(o / L);
}

}  // namespace

extern "C" int gqa_cache_write_bf16(const void* k, const void* v, void* cache, const int64_t* slot_mapping,
                                    int64_t tokens, int64_t kv_heads, int64_t head_dim, int64_t block_size,
                                    cudaStream_t stream) {
  if (tokens <= 0) return 0;
  if (kv_heads <= 0 || head_dim <= 0 || block_size <= 0) return err(cudaErrorInvalidValue);
  gqa_cache_write_kernel<<<(unsigned)tokens, 256, 0, stream>>>(static_cast<const bf16*>(k), static_cast<const bf16*>(v),
                                                               static_cast<bf16*>(cache), slot_mapping, (int)kv_heads,
                                                               (int)head_dim, (int)block_size);
  return last_err();
}

extern "C" int gqa_prefill_bf16(const void* q, const void* k, const void* v, void* out, const int32_t* cu_seqlens,
                                int64_t num_reqs, int64_t max_seqlen, int64_t heads, int64_t kv_heads, int64_t head_dim,
                                float scale, int64_t q_stride, int64_t kv_stride, cudaStream_t stream) {
  if (q_stride <= 0) q_stride = heads * head_dim;
  if (kv_stride <= 0) kv_stride = kv_heads * head_dim;
  if (num_reqs <= 0) return 0;
  if (heads <= 0 || kv_heads <= 0 || head_dim <= 0 || head_dim > MAX_D || heads % kv_heads != 0 || max_seqlen <= 0)
    return err(cudaErrorInvalidValue);
  if (head_dim == PF_D && heads / kv_heads <= 2) {
    static bool v3_attr_set = false;
    if (!v3_attr_set) {
      if (cudaFuncSetAttribute(gqa_prefill_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)V3_SMEM) !=
          cudaSuccess)
        return last_err();
      v3_attr_set = true;
    }
    dim3 grid((unsigned)((max_seqlen + V3_BQ - 1) / V3_BQ), (unsigned)kv_heads, (unsigned)num_reqs);
    gqa_prefill_tc_kernel<<<grid, V3_WARPS * 32, V3_SMEM, stream>>>(
        static_cast<const bf16*>(q), static_cast<const bf16*>(k), static_cast<const bf16*>(v), static_cast<bf16*>(out),
        cu_seqlens, (int)heads, (int)kv_heads, scale, (int)q_stride, (int)kv_stride);
    return last_err();
  }
  dim3 grid((unsigned)num_reqs, (unsigned)heads);
  gqa_prefill_ref_kernel<<<grid, GQ_BLOCK, 0, stream>>>(static_cast<const bf16*>(q), static_cast<const bf16*>(k),
                                                        static_cast<const bf16*>(v), static_cast<bf16*>(out), cu_seqlens,
                                                        (int)heads, (int)kv_heads, (int)head_dim, scale, (int)q_stride,
                                                        (int)kv_stride);
  return last_err();
}

extern "C" int gqa_paged_bf16(const void* q, const void* cache, const int32_t* block_table, const int32_t* seq_lens,
                              const int32_t* cu_seqlens_q, void* out, int64_t num_reqs, int64_t max_q_len, int64_t heads,
                              int64_t kv_heads, int64_t head_dim, int64_t block_size, int64_t max_blocks, float scale,
                              int64_t q_stride, cudaStream_t stream) {
  if (q_stride <= 0) q_stride = heads * head_dim;
  if (num_reqs <= 0) return 0;
  if (head_dim <= 0 || head_dim > MAX_D || heads <= 0 || kv_heads <= 0 || heads % kv_heads != 0 || block_size <= 0 ||
      max_q_len <= 0)
    return err(cudaErrorInvalidValue);
  if (head_dim == PF_D && heads / kv_heads <= 2) {
    static bool v3p_attr_set = false;
    if (!v3p_attr_set) {
      if (cudaFuncSetAttribute(gqa_paged_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)V3_SMEM) !=
          cudaSuccess)
        return last_err();
      v3p_attr_set = true;
    }
    dim3 grid((unsigned)((max_q_len + V3_BQ - 1) / V3_BQ), (unsigned)kv_heads, (unsigned)num_reqs);
    gqa_paged_tc_kernel<<<grid, V3_WARPS * 32, V3_SMEM, stream>>>(
        static_cast<const bf16*>(q), static_cast<const bf16*>(cache), block_table, seq_lens, cu_seqlens_q,
        static_cast<bf16*>(out), (int)heads, (int)kv_heads, (int)block_size, (int)max_blocks, scale, (int)q_stride);
    return last_err();
  }
  if (head_dim == PF_D && heads / kv_heads <= 2 && false) {
    const int64_t group = heads / kv_heads;
    const int64_t rows_per_block = (PV2_WARPS * PV2_ROWS_PER_WARP) / group;
    dim3 grid((unsigned)((max_q_len + rows_per_block - 1) / rows_per_block), (unsigned)kv_heads, (unsigned)num_reqs);
    gqa_paged_flash_kernel<<<grid, PV2_WARPS * 32, 0, stream>>>(
        static_cast<const bf16*>(q), static_cast<const bf16*>(cache), block_table, seq_lens, cu_seqlens_q,
        static_cast<bf16*>(out), (int)heads, (int)kv_heads, (int)block_size, (int)max_blocks, scale, (int)q_stride);
  } else {
    dim3 grid((unsigned)num_reqs, (unsigned)heads);
    gqa_paged_kernel<<<grid, GQ_BLOCK, 0, stream>>>(static_cast<const bf16*>(q), static_cast<const bf16*>(cache),
                                                    block_table, seq_lens, cu_seqlens_q, static_cast<bf16*>(out),
                                                    (int)heads, (int)kv_heads, (int)head_dim, (int)block_size,
                                                    (int)max_blocks, scale, (int)q_stride);
  }
  return last_err();
}

// decode 专用（每请求恰 1 行 query，q: [B, H, D]，out 同形）。scratch: ≥ B × H × 8 × (D + 2) 个 f32。D 必须为 128。
extern "C" int gqa_decode_bf16(const void* q, const void* cache, const int32_t* block_table, const int32_t* seq_lens,
                               void* out, float* scratch, int64_t num_reqs, int64_t heads, int64_t kv_heads,
                               int64_t head_dim, int64_t block_size, int64_t max_blocks, float scale, int64_t q_stride,
                               cudaStream_t stream) {
  if (q_stride <= 0) q_stride = heads * head_dim;
  if (num_reqs <= 0) return 0;
  if (head_dim != PF_D || heads <= 0 || kv_heads <= 0 || heads % kv_heads != 0 || block_size <= 0)
    return err(cudaErrorInvalidValue);
  const int group = (int)(heads / kv_heads);
  if (group <= D2_MAXG && block_size % 16 == 0) {
    // splits 自适应：目标 ≥ 256 个 block；上限 8（scratch 按 8 分配）
    int splits = (int)((256 + num_reqs * kv_heads - 1) / (num_reqs * kv_heads));
    splits = splits < 1 ? 1 : (splits > DC_SPLITS ? DC_SPLITS : splits);
    dim3 grid1((unsigned)num_reqs, (unsigned)kv_heads, (unsigned)splits);
    bf16* direct = splits == 1 ? static_cast<bf16*>(out) : nullptr;
    gqa_decode_v2_kernel<<<grid1, D2_WARPS * 32, 0, stream>>>(
        static_cast<const bf16*>(q), static_cast<const bf16*>(cache), block_table, seq_lens, scratch, direct, (int)heads,
        (int)kv_heads, (int)block_size, (int)max_blocks, scale, (int)q_stride, splits);
    if (splits > 1) {
      dim3 grid2((unsigned)num_reqs, (unsigned)heads);
      gqa_decode_merge_v2_kernel<<<grid2, PF_D, 0, stream>>>(scratch, static_cast<bf16*>(out), (int)heads, splits);
    }
    return last_err();
  }
  dim3 grid1((unsigned)num_reqs, (unsigned)kv_heads, (unsigned)DC_SPLITS);
  gqa_decode_split_kernel<<<grid1, DC_WARPS * 32, 0, stream>>>(
      static_cast<const bf16*>(q), static_cast<const bf16*>(cache), block_table, seq_lens, scratch, (int)heads,
      (int)kv_heads, (int)block_size, (int)max_blocks, scale, (int)q_stride);
  dim3 grid2((unsigned)num_reqs, (unsigned)heads);
  gqa_decode_merge_kernel<<<grid2, PF_D, 0, stream>>>(scratch, static_cast<bf16*>(out), (int)heads);
  return last_err();
}
