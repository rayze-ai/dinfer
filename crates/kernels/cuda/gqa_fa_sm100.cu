// 全注意力 prefill 的 sm100（tcgen05 / UMMA）路径 —— r12a 第二步，docs/loops/qwen35-tcgen05.md。
//
// 为什么单开文件：CLAUDE.md 的多卡型规则要求按 arch 分文件、kernel 体内禁止 `#if __CUDA_ARCH__`；
// 旧路径（`gqa.cu` 的 v5 / v6）一行不改，只在分发处加分支。
//
// 结构（一个 CTA 一个 Q 块：BQ=128 行，128 线程，每线程独占一行）：
//   * Q 常驻 smem，按 64 列切成 4 块（128 B swizzle、K-major）。K 轴一个 swizzle atom = 64 个 bf16，
//     所以一个 MMA 组的 K 跨度必须恰好是 64 列（整行的 64 列连续块拼成 256 维的一次点积）。
//   * 每个 key tile（NKI=64 个 key）：
//       QK → S 在 TMEM（64 列），S 取回寄存器 → softmax → P 写 smem（K-major [128,64]）
//       O 按 alpha 重缩放：tcgen05.ld → 乘 → tcgen05.st
//       PV → O 累加在 TMEM（256 列，一条 N=256 的 MMA 组，V 是 MN-major、LBO 指向下一个 64 维块）
//   * **两遍 softmax**：先取回整个 tile 的 S 算行 max 与行和（每线程一行，行内归约不需要跨 lane），
//     P 只写一次 smem，O 的 rescale 每个 tile 只做一次。
//   * O 全程留在 TMEM，最后一次取回、除以行和、写 out。
//
// TMEM 预算：S 64 + O 256 = 320 列 → alloc 512（2 的幂；每 SM 只有一份 → 天然 1 CTA/SM；
// umma_peak 实测 1 CTA/SM 就能喂满 tensor core，592 block 反而掉到 1809 TF/s）。
#include "common.cuh"
#include "umma.cuh"

namespace {

using dinfer::bf16;

constexpr int kD = 256;       // 头维
constexpr int kBQ = 128;      // 每个 Q 块的行数 = UMMA 的 M = TMEM 的 lane 数
constexpr int kNKI = 64;      // 每个 key tile 的 key 数 = QK 的 N = S 占的 TMEM 列数
constexpr int kDChunks = kD / 64;      // Q/K/V 都按 64 列分块（一个 swizzle atom）
constexpr int kThreads = kBQ;          // 每线程一行
constexpr int kMmaPerK = kNKI / 16;    // 一个 64-key atom 里的 4 条 K=16 MMA
constexpr int kSlots = 2;              // K/V/P 的缓冲数（双缓冲）

constexpr int kSCol = 0;     // S 的 TMEM 列偏移
constexpr int kOCol = kNKI;  // O 的 TMEM 列偏移

// smem：sQ | sK[kSlots] | sV[kSlots] | sP[kSlots] | tail
// 所有偏移一律用**字节**；指针换算走 bf16_of()，避免"字节当元素"的步长错误（踩过两次）。
constexpr int kOneElems = kNKI * kNKI;                        // 一块 64x64 bf16 = 4096 个元素 = 8192 B
struct Layout {
  static constexpr int q_bytes = kBQ * kNKI * 2 * kDChunks;   // 64 KB
  static constexpr int kv_one = kOneElems * 2 * kDChunks;     // 32 KB（K 或 V 的全部 4 块）
  static constexpr int p_one = kBQ * kNKI * 2;                // 16 KB
  static constexpr int q_off = 0;
  static constexpr int k_off = q_bytes;
  static constexpr int v_off = k_off + kv_one * kSlots;
  static constexpr int p_off = v_off + kv_one * kSlots;
  static constexpr int tail = p_off + p_one * kSlots;
  static constexpr int bytes = tail + 128;                    // tmem 槽 + mbarrier
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// 128 B swizzle 里 (row, piece) 的 16 B 槽位（piece = d/8）。
__device__ __forceinline__ int swz_piece(int row, int piece) { return row * 8 + (piece ^ (row & 7)); }

// 把 gmem 的一行 16 B（8 个 bf16）写进 swizzle tile。bytes=0 → 零填充。
__device__ __forceinline__ void stage16(bf16* tile, int row, int piece, const bf16* src, int bytes) {
  const uint32_t a = smem_u32(tile + (size_t)swz_piece(row, piece) * 8);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(a), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }

// 分段计时（R12A_PROF=1 时由 host 传非空指针）：[0] 等 QK [1] softmax [2] rescale [3] PV+装载 [4] tile 总数
template <bool PAGED>
__global__ void __launch_bounds__(kThreads) gqa_fa_umma_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v, const bf16* __restrict__ cache,
    const int32_t* __restrict__ block_table, const int32_t* __restrict__ seq_lens, const int32_t* __restrict__ cu_seqlens,
    bf16* __restrict__ out, int H, int Hkv, float scale_log2, int q_stride, int kv_stride, int block_size, int max_blocks,
    int skip_single, unsigned long long* prof) {
  extern __shared__ __align__(128) unsigned char smem[];
  bf16* sQ = reinterpret_cast<bf16*>(smem + Layout::q_off);
  bf16* sK = reinterpret_cast<bf16*>(smem + Layout::k_off);
  bf16* sV = reinterpret_cast<bf16*>(smem + Layout::v_off);
  bf16* sP = reinterpret_cast<bf16*>(smem + Layout::p_off);
  // 槽步长按字节 → 元素
  constexpr int kKvSlotElems = Layout::kv_one / 2, kPSlotElems = Layout::p_one / 2;
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(smem + Layout::tail);
  uint64_t* bar = reinterpret_cast<uint64_t*>(smem + Layout::tail + 16);  // 槽 0 = QK，槽 1 = PV

  const int tid = threadIdx.x, warp = tid >> 5;
  const int b = blockIdx.z, h = blockIdx.y;
  const int group = H / Hkv, hk = h / group;
  const int q0 = cu_seqlens[b], q_len = cu_seqlens[b + 1] - q0;
  if (PAGED && skip_single && q_len == 1) return;
  // 反序：gridDim.x-1-blockIdx.x → 行号最大（key 最多）的 block 先发，尾波不被最重的拖。
  const int q_base = (gridDim.x - 1 - blockIdx.x) * kBQ;
  if (q_base >= q_len) return;
  const int seq_len = PAGED ? seq_lens[b] : q_len;
  const int hist = seq_len - q_len;
  const int32_t* bt = PAGED ? block_table + (int64_t)b * max_blocks : nullptr;

  if (tid == 0) {
    for (int i = 0; i < 8; ++i) umma::mbar_init(bar + i, 1);
    umma::mbar_init_fence();
  }
  __syncthreads();
  if (warp == 0) {
    umma::tmem_alloc(tmem_slot, 512);
    umma::tmem_relinquish();
  }
  __syncthreads();
  const uint32_t tmem = *tmem_slot;
  const uint32_t tS = tmem + kSCol, tO = tmem + kOCol;
  const uint32_t idesc_s = umma::make_instr_desc_bf16(kBQ, kNKI, false, false);
  const uint32_t idesc_o = umma::make_instr_desc_bf16(kBQ, kD, false, /*b_mn=*/true);
  const uint32_t lane_base = (uint32_t)warp * 32u << 16;  // 本 warp 的 TMEM lane 基址

  const int row = tid;  // 每线程独占一行（128 线程 = BQ 行）

  // ---------------- Q → smem（4 个 64 列块；越界行零填充）----------------
  {
    const bool live = row < q_len;
    const bf16* pq = q + (int64_t)(q0 + q_base + (live ? row : 0)) * q_stride + (int64_t)h * kD;
#pragma unroll
    for (int c = 0; c < kDChunks; ++c) {
      const bf16* pc = pq + c * 64;
      bf16* tile = sQ + (size_t)c * kBQ * kNKI;
#pragma unroll
      for (int p = 0; p < 8; ++p) stage16(tile, row, p, pc + p * 8, live ? 16 : 0);
    }
    cp_commit();
  }

  // K/V tile 装载：每线程负责 1 个 key（tid&63），前后两半点线程分担 4 个维度块。
  auto load_kv = [&](int tile, int slot) {
    const int k0 = tile * kNKI;
    const int key0 = tid & 63;
    const int half = tid >> 6;
    const int pos = k0 + key0;
    const bool live = pos < seq_len;
    const bf16 *pk, *pv;
    if (PAGED) {
      const int64_t blk = bt[pos / block_size];
      const bf16* base = cache + (((int64_t)(blk * block_size + pos % block_size)) * Hkv + hk) * 2 * kD;
      pk = base;
      pv = base + kD;
    } else {
      const int64_t t = q0 + (live ? pos : 0);
      pk = k + t * kv_stride + (int64_t)hk * kD;
      pv = v + t * kv_stride + (int64_t)hk * kD;
    }
    bf16* kt = sK + (size_t)slot * kKvSlotElems;
    bf16* vt = sV + (size_t)slot * kKvSlotElems;
#pragma unroll
    for (int c = 0; c < kDChunks; ++c) {
      if (((c + half) & 1) != 0) continue;  // 两半交替负责块
      const int bytes = live ? 16 : 0;
      bf16* kc = kt + (size_t)c * kOneElems;
      bf16* vc = vt + (size_t)c * kOneElems;
#pragma unroll
      for (int p = 0; p < 8; ++p) {
        stage16(kc, key0, p, pk + c * 64 + p * 8, bytes);
        stage16(vc, key0, p, pv + c * 64 + p * 8, bytes);
      }
    }
  };

  // QK(tile)：A = Q（4 个 64 列块在 K 轴上拼接），B = K。S 覆盖写（K 链首条 !accumulate）。
  auto qk = [&](int slot) {
    if (warp == 0 && umma::elect_one()) {
      const bf16* kt = sK + (size_t)slot * kKvSlotElems;
#pragma unroll
      for (int c = 0; c < kDChunks; ++c) {
        const uint64_t da = umma::k_major_128b_desc(smem_u32(sQ + (size_t)c * kBQ * kNKI));
        const uint64_t db = umma::k_major_128b_desc(smem_u32(kt + (size_t)c * kOneElems));
#pragma unroll
        for (int kk = 0; kk < kMmaPerK; ++kk) {
          const uint32_t off = (uint32_t)(kk * 16 * 2) >> 4;
          umma::mma_f16(tS, da + off, db + off, idesc_s, c > 0 || kk > 0);
        }
      }
      umma::mma_commit(bar);
    }
  };

  // PV(tile)：A = P（K-major [128,64]），B = V（MN-major，n = 头维、k = key）。
  // V 的 4 个 64 维块在 MN 轴上拼接 → 一条 N=256 的 MMA 组，LBO 指向下一个 MN 块。
  auto pv = [&](int slot, bool first) {
    if (warp == 0 && umma::elect_one()) {
      const bf16* vt = sV + (size_t)slot * kKvSlotElems;
      const bf16* pt = sP + (size_t)slot * kPSlotElems;
      const uint64_t da = umma::k_major_128b_desc(smem_u32(pt));
      // atom = 8 key × 128 B；LBO = MN 方向下一块（本 tile 4 块，各 64 key × 128 B）；
      // SBO = K 方向下一个 atom（8 key）= 1024 B。
      const uint64_t db = umma::mn_major_128b_desc(smem_u32(vt), /*lbo=*/kNKI * 128, /*sbo=*/8 * 128);
#pragma unroll
      for (int kk = 0; kk < kMmaPerK; ++kk) {
        const uint32_t off_a = (uint32_t)(kk * 16 * 2) >> 4;    // P：16 个 key
        const uint32_t off_b = (uint32_t)(kk * 16 * 128) >> 4;  // V：16 个 key × 128 B
        umma::mma_f16(tO, da + off_a, db + off_b, idesc_o, !(first && kk == 0));
      }
      umma::mma_commit(bar + 1);
    }
  };

  float m_row = -INFINITY, l_row = 0.f;
  float s[kNKI];
  int phase_a = 0, phase_b = 0;

  const int last_key = hist + q_base + kBQ - 1;
  const int n_tiles = min(last_key, seq_len - 1) / kNKI + 1;

  // 预装 tile 0 与 tile 1（两个槽）。cp.async.wait_group 1 = 除最近 1 组外全完成 →
  // 保证 tile 0 到位、tile 1 可以还在路上。
  load_kv(0, 0);
  cp_commit();
  if (1 < n_tiles) load_kv(1, 1);
  cp_commit();
  asm volatile("cp.async.wait_group 1;\n" ::);
  __syncthreads();

  qk(0);
  for (int tile = 0; tile < n_tiles; ++tile) {
    const int slot = tile & (kSlots - 1);
    const int k0 = tile * kNKI;

    // ---- 等 QK(tile) ----
    unsigned long long t0 = (prof && tid == 0) ? clock64() : 0;
    umma::mbar_wait(bar, phase_a);
    phase_a ^= 1;
    umma::fence_after_sync();
    __syncthreads();
    unsigned long long t1 = (prof && tid == 0) ? clock64() : 0;

    // ---- softmax：本线程独占第 tid 行 ----
    {
      uint32_t v0[32], v1[32];
      umma::tmem_ld32_x32(tS + lane_base + 0u, v0);
      umma::tmem_ld32_x32(tS + lane_base + 32u, v1);
      umma::tmem_ld_wait();
#pragma unroll
      for (int i = 0; i < 32; ++i) s[i] = __uint_as_float(v0[i]) * scale_log2;
#pragma unroll
      for (int i = 0; i < 32; ++i) s[32 + i] = __uint_as_float(v1[i]) * scale_log2;
    }
    // mask：本 block 第 0 行最后一个可见 key = hist+q_base；只有末尾 tile 需要逐元素比较。
    const bool need_mask = (k0 + kNKI - 1 > hist + q_base);
    const int vis = hist + q_base + row;
    float tmax = -INFINITY;
    if (need_mask) {
      const int key_ok = vis - k0;
#pragma unroll
      for (int i = 0; i < kNKI; ++i) {
        if (i > key_ok) s[i] = -INFINITY;
        tmax = fmaxf(tmax, s[i]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < kNKI; ++i) tmax = fmaxf(tmax, s[i]);
    }
    const float m_new = fmaxf(m_row, tmax);
    const float alpha = (m_row == -INFINITY) ? 1.f : umma::ex2(m_row - m_new);
    float psum = 0.f;
    if (m_new == -INFINITY) {  // 整行不可见（padding 行）
#pragma unroll
      for (int i = 0; i < kNKI; ++i) s[i] = 0.f;
    } else {
#pragma unroll
      for (int i = 0; i < kNKI; ++i) {
        s[i] = (s[i] == -INFINITY) ? 0.f : umma::ex2(s[i] - m_new);
        psum += s[i];
      }
    }
    l_row = l_row * alpha + psum;
    m_row = m_new;

    // ---- P → smem（每线程写自己那一行的 8 个 16 B 片）----
    {
      bf16* pt = sP + (size_t)slot * kPSlotElems;
#pragma unroll
      for (int p = 0; p < 8; ++p) {
        __nv_bfloat162 two[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int i = p * 8 + j * 2;
          two[j] = __floats2bfloat162_rn(s[i], s[i + 1]);
        }
        const int slotp = swz_piece(row, p);
        *reinterpret_cast<uint4*>(pt + (size_t)slotp * 8) = *reinterpret_cast<const uint4*>(two);
      }
      umma::fence_proxy_async_shared();  // 普通 st.shared → UMMA 读，必须过 async proxy 栅栏
    }

    unsigned long long t2 = (prof && tid == 0) ? clock64() : 0;
    // ---- rescale O ----
    // PV(tile-1) 已在上轮"装 tile+1"那步前等过（见循环末尾），此处可直接读写 O。
    // `__syncthreads_or` 是全 block barrier，不能放进 `if (tile > 0)`：tile 0 会少一次 barrier，
    // 与后面的 `__syncthreads()` 错配 → 死锁（踩过）。
    const bool need_rescale = (tile > 0) && __syncthreads_or(alpha != 1.f);
    if (need_rescale) {
      // alpha 逐行不同：按 warp 汇总，只有需要的 warp 才读改写 TMEM
      if (__any_sync(0xffffffffu, alpha != 1.f)) {
#pragma unroll
        for (int c = 0; c < kD / 32; ++c) {
          uint32_t ov[32];
          const uint32_t addr = tO + lane_base + (uint32_t)(c * 32);
          umma::tmem_ld32_x32(addr, ov);
          umma::tmem_ld_wait();
#pragma unroll
          for (int i = 0; i < 32; ++i) ov[i] = __float_as_uint(__uint_as_float(ov[i]) * alpha);
          umma::tmem_st32_x32(addr, ov);
          umma::tmem_st_wait();
        }
      }
    }
    // barrier 必须无条件：rescale 与否由 need_rescale 决定，但同步点每个 tile 都在。
    umma::fence_before_sync();
    __syncthreads();

    unsigned long long t3 = (prof && tid == 0) ? clock64() : 0;
    // ---- 流水：发 PV(tile) → 发 QK(tile+1)（K/V(tile+1) 已装好）→ 等 PV(tile) → 装 tile+2 ----
    // 只有 2 个 K/V 槽：tile 用 slot、tile+1 用 slot^1。所以 tile+2 只能进 slot，
    // 而 slot 正被 PV(tile) 读 → 必须等 PV(tile) 完成。等待放在发完 QK(tile+1) 之后，
    // 这样 PV(tile) 与 QK(tile+1) 的 MMA 在执行期重叠，等的只是两者的尾巴。
    pv(slot, tile == 0);
    if (tile + 1 < n_tiles) {
      asm volatile("cp.async.wait_group 0;\n" ::);  // 等 K/V(tile+1)（上一轮已发）
      __syncthreads();
      qk(slot ^ 1);
    }
    if (tile + 2 < n_tiles) {
      umma::mbar_wait(bar + 1, phase_b);
      phase_b ^= 1;
      umma::fence_after_sync();
      load_kv(tile + 2, slot);
      cp_commit();
    } else if (tile + 1 < n_tiles) {
      // 收尾：仍要等 PV(tile)，否则下一轮 rescale 会与它竞争。
      umma::mbar_wait(bar + 1, phase_b);
      phase_b ^= 1;
      umma::fence_after_sync();
    }
    if (prof && tid == 0) {
      unsigned long long t4 = clock64();
      atomicAdd(&prof[0], t1 - t0);
      atomicAdd(&prof[1], t2 - t1);
      atomicAdd(&prof[2], t3 - t2);
      atomicAdd(&prof[3], t4 - t3);
      atomicAdd(&prof[4], 1);
    }
  }
  // 收尾：每个 tile 的 PV 都在本轮末尾等过，这里只需保证所有线程到达。
  __syncthreads();

  // ---------------- O 取回 + 归一化 + 写回 ----------------
  if (row < q_len) {
    const float inv = l_row > 0.f ? 1.f / l_row : 0.f;
    bf16* op = out + (int64_t)(q0 + q_base + row) * H * kD + (int64_t)h * kD;
#pragma unroll
    for (int c = 0; c < kD / 32; ++c) {
      uint32_t ov[32];
      umma::tmem_ld32_x32(tO + lane_base + (uint32_t)(c * 32), ov);
      umma::tmem_ld_wait();
#pragma unroll
      for (int i = 0; i < 32; ++i) op[c * 32 + i] = __float2bfloat16(__uint_as_float(ov[i]) * inv);
    }
  }
  __syncthreads();
  if (warp == 0) umma::tmem_dealloc(tmem, 512);
}

}  // namespace

// cfg >= 5000：tcgen05 路径。返回 cudaErrorInvalidValue 表示"该形状不走这条路"，调用方回退 v5/v6。
extern "C" int gqa_fa_umma_launch(int paged, int max_q, int heads, int num_reqs, const void* q, const void* k,
                                  const void* v, const void* cache, const int32_t* bt, const int32_t* seq_lens,
                                  const int32_t* cu, void* out, int H, int Hkv, float scale_log2, int q_stride,
                                  int kv_stride, int block_size, int max_blocks, int skip_single, void* prof,
                                  cudaStream_t stream) {
  const size_t smem = Layout::bytes;
  dim3 grid((unsigned)((max_q + kBQ - 1) / kBQ), (unsigned)heads, (unsigned)num_reqs);
  auto launch = [&](auto kern) {
    static int attr = 0;
    if (attr == 0) {
      attr = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess ? 1 : -1;
      if (attr < 0) (void)cudaGetLastError();
    }
    if (attr < 0) return (int)cudaErrorInvalidConfiguration;
    kern<<<grid, kThreads, smem, stream>>>(static_cast<const bf16*>(q), static_cast<const bf16*>(k),
                                           static_cast<const bf16*>(v), static_cast<const bf16*>(cache), bt, seq_lens, cu,
                                           static_cast<bf16*>(out), H, Hkv, scale_log2, q_stride, kv_stride, block_size,
                                           max_blocks, skip_single, static_cast<unsigned long long*>(prof));
    return (int)cudaGetLastError();
  };
  if (paged) return launch(gqa_fa_umma_kernel<true>);
  return launch(gqa_fa_umma_kernel<false>);
}
