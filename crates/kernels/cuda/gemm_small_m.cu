// 小 M（decode，M ≤ 32）bf16 GEMM：C[M,N] = A[M,K] · W[N,K]^T (+ bias[N])，fp32 累加。
//
// 适用：所有 arch 共用一份（纯访存型：权重只读一遍，M 行复用同一段权重读）。
// 为什么快：cuBLASLt 对 M=1 常选 128×192 / 128×256 大 tile（算力型配置，B300 上 43% 带宽）；
// 这里每个 warp 负责 W 的 1 行，每 lane 一次 16 B 向量读，K 维可按 split-K 切成多个 block
// 让小 N（out_proj N=5120、in_proj_a N=48）也能铺满 SM；M 行在寄存器里各一个累加器。
// A 的一段 K（kTileK）先放共享内存，lane 以 16 B 广播读，避免每个 W 向量配 M 次 L1 读。
//
// 结构常量（不是调优数值）：kWarps（每 block warp 数）、kTileK（共享内存 A 分块）。
// 调优数值 `splits`（split-K 份数）由 host 给（ArchProfile 默认 + 启动自调优），kernel 不猜。
// 限制：K % 8 == 0（16 B 对齐向量读）；splits > 1 时需要 partial[splits, M, N] f32 工作区，
// 由 `gemm_small_m_reduce` 归约 + 加 bias + 转 bf16。
// M_TILE ∈ {1,2,4,8,16,32}（第 9 轮加 16/32：M≥16 时 cuBLASLt 对小 N 选 splitK 3–5 + 独立 reduce kernel，
// 本 kernel 一次 launch 权重只读一遍、M 行寄存器累加；大 N 时 fp32 FMA 算力型会输给 Lt，由自调优决定）。
// 共享内存 A 分块随 M_TILE 缩（M_TILE·TK·2 B ≤ 32 KB，静态 shared 上限 48 KB）。
// 对拍：`ops_parity.sh` 的 gemm_small_m case（真形状 + M∈{1,3,4,8,16,32}，两卡都跑）。
#include "common.cuh"

namespace dinfer {
namespace {

constexpr int kWarps = 4;
constexpr int kBlock = kWarps * 32;
constexpr int kTileK = 1024;  // 每次放进共享内存的 A 列数（M_TILE ≤ 16）；bf16 → 每行 2 KB，M=16 共 32 KB
// M_TILE 对应的 A 分块列数：32 行时减半，保持 32 KB。
template <int M_TILE>
constexpr int tile_k() { return M_TILE <= 16 ? kTileK : kTileK / 2; }

__device__ __forceinline__ void fma8(float& acc, uint4 wv, uint4 av) {
  const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
  const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&av);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 wf = __bfloat1622float2(w2[i]);
    const float2 af = __bfloat1622float2(a2[i]);
    acc = fmaf(wf.x, af.x, acc);
    acc = fmaf(wf.y, af.y, acc);
  }
}

// grid = (ceil(N / kWarps), splits)，block = kBlock。blockIdx.y 负责 K 的第 y 段。
template <int M_TILE>
__global__ void __launch_bounds__(kBlock) small_m_kernel(const bf16* __restrict__ a, const bf16* __restrict__ w,
                                                         const bf16* __restrict__ bias, bf16* __restrict__ out,
                                                         float* __restrict__ partial, int m, int n, int k,
                                                         int k_per_split) {
  constexpr int TK = tile_k<M_TILE>();
  __shared__ __align__(16) bf16 a_tile[M_TILE][TK];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row_n = blockIdx.x * kWarps + warp;  // 本 warp 负责的 W 行（可能越界，仍参与 __syncthreads）
  const int k_begin = blockIdx.y * k_per_split;
  const int k_end = min(k, k_begin + k_per_split);

  float acc[M_TILE];
#pragma unroll
  for (int i = 0; i < M_TILE; ++i) acc[i] = 0.f;

  const bf16* wrow = w + (long)row_n * k;
  for (int k0 = k_begin; k0 < k_end; k0 += TK) {
    const int kt = min(TK, k_end - k0);  // 本段列数（8 的倍数）
    // A[0..m, k0..k0+kt) → 共享内存；行 ≥ m 的不读（不会被用到）
    for (int idx = threadIdx.x; idx < M_TILE * (TK / 8); idx += kBlock) {
      const int r = idx / (TK / 8), c8 = idx % (TK / 8);
      if (r < m && c8 * 8 < kt) {
        *reinterpret_cast<uint4*>(&a_tile[r][c8 * 8]) =
            *reinterpret_cast<const uint4*>(a + (long)r * k + k0 + c8 * 8);
      }
    }
    __syncthreads();
    if (row_n < n) {
      // lane 读第 lane、lane+32、… 个 16 B 向量
      for (int c8 = lane; c8 * 8 < kt; c8 += 32) {
        const uint4 wv = *reinterpret_cast<const uint4*>(wrow + k0 + c8 * 8);
#pragma unroll
        for (int r = 0; r < M_TILE; ++r) {
          const uint4 av = *reinterpret_cast<const uint4*>(&a_tile[r][c8 * 8]);
          fma8(acc[r], wv, av);
        }
      }
    }
    __syncthreads();  // 下一段会覆盖 a_tile
  }
  if (row_n >= n) return;

#pragma unroll
  for (int r = 0; r < M_TILE; ++r) {
    float v = acc[r];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    acc[r] = v;
  }
  if (lane != 0) return;
  if (partial) {
    float* p = partial + ((long)blockIdx.y * m) * n;
    for (int r = 0; r < M_TILE; ++r)
      if (r < m) p[(long)r * n + row_n] = acc[r];
  } else {
    const float b = bias ? to_f32(bias[row_n]) : 0.f;
    for (int r = 0; r < M_TILE; ++r)
      if (r < m) out[(long)r * n + row_n] = to_bf16(acc[r] + b);
  }
}

// out[M,N] = Σ_s partial[s,M,N] (+ bias)
__global__ void reduce_kernel(const float* __restrict__ partial, const bf16* __restrict__ bias,
                              bf16* __restrict__ out, int m, int n, int splits) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long mn = (long)m * n;
  if (i >= mn) return;
  float v = bias ? to_f32(bias[i % n]) : 0.f;
  for (int s = 0; s < splits; ++s) v += partial[(long)s * mn + i];
  out[i] = to_bf16(v);
}

template <int M_TILE>
void launch(const bf16* a, const bf16* w, const bf16* bias, bf16* out, float* partial, int m, int n, int k,
            int splits, int k_per_split, cudaStream_t stream) {
  dim3 grid((unsigned)((n + kWarps - 1) / kWarps), (unsigned)splits);
  small_m_kernel<M_TILE><<<grid, kBlock, 0, stream>>>(a, w, bias, out, partial, m, n, k, k_per_split);
}


// ============================================================================================
// tensor-core 小 M kernel（r11b，docs/loops/qwen35-decode-r11b.md）：C[M,N] = A[M,K]·W[N,K]^T，mma.m16n8k16 bf16。
// 为什么：FMA 版每 16 B 权重要 M×8 条 FMA + 转换，M=8/32 时算力型（1.7 / 0.46 TB/s）；cuBLASLt 对这些形状停在 ~6 TB/s。
// 结构（第 3 轮定稿；第 1/2 轮把 W 当 mma "A" 操作数、激活走 smem 当 "B"，smem 读量 = W 读量，M=32 时卡在 smem 带宽 3–4 TB/s）：
//   - 激活 A[M,K] 当 mma 的 "A"（m16，M ≤ 16 一组、32 两组），按 TK 分块进 smem，每 32 列 chunk 每 lane 读 2 × 16 B 到寄存器；
//   - W 当 mma 的 "B"（n8 = W 的 8 行），**直接 16 B 向量读 global，不经 smem**；每 warp 负责 R 条 8 行 stripe，
//     同一份 A 寄存器片段复用 R 次 → smem 读量 = W 的 1/R；
//   - K 维按 lane 置换（lane t 持列 [t·8, t·8+8)，前 4 列给第 1 条 mma、后 4 列给第 2 条），A / B 一致即可，K 只是归约维。
// split-K：blockIdx.y 负责 K 的一段；splits > 1 时 partial[splits,M,N] f32 + 每 stripe（8 行 N）一个计数器，
// **最后到达的 warp 归约**（__threadfence + atomicAdd，自复位，CUDA graph 安全），不再有独立 reduce kernel。
// 调优数值 splits / warps / stripes 由 host 给（ArchProfile 候选 + 启动自调优），kernel 不猜。限制：K % 32 == 0。
constexpr int kMmaBatch = 4;  // 每轮一起发出的 32 列 chunk 数（K 段按 32·kMmaBatch 对齐，host 保证）
constexpr int kMmaPad = 8;  // 行 pad 16 B：保持每行 16 B 对齐（8 lane 相位读同一行 128 B 连续，天然无 bank 冲突）
template <int M_TILE>
constexpr int mma_tk() { return M_TILE <= 16 ? 1024 : 512; }  // 分块列数：smem 行数 = m16 组 × 16（M=8 也 16 行），≈ 33 KB

__device__ __forceinline__ void mma16816_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// grid = (ceil(N / (8·R·WARPS)), splits)，block = 32·WARPS。
template <int M_TILE, int WARPS, int R>
__global__ void __launch_bounds__(32 * WARPS) small_m_mma_kernel(const bf16* __restrict__ a, const bf16* __restrict__ w,
                                                                 const bf16* __restrict__ bias, bf16* __restrict__ out,
                                                                 float* __restrict__ partial, int* __restrict__ counters,
                                                                 int m, int n, int k, int k_per_split, int splits) {
  static_assert(M_TILE == 8 || M_TILE == 16 || M_TILE == 32, "M_TILE");
  constexpr int MG = (M_TILE + 15) / 16;  // m16 组数（M=8 一组只用前 8 行）
  constexpr int TK = mma_tk<M_TILE>();
  constexpr int ROWS = MG * 16;
  __shared__ __align__(16) bf16 a_tile[ROWS][TK + kMmaPad];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int n0 = (blockIdx.x * WARPS + warp) * (8 * R);  // 本 warp 的 W 行起点（可能越界：仍参与 __syncthreads）
  const int k_begin = blockIdx.y * k_per_split;
  const int k_end = min(k, k_begin + k_per_split);

  float acc[R][MG][4];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int i = 0; i < MG; ++i) acc[r][i][0] = acc[r][i][1] = acc[r][i][2] = acc[r][i][3] = 0.f;

  const bool active = n0 < n;
  // stripe r 的 lane 读 W 行 (n0 + 8r + g)，列偏移 t·8；越界行 clamp（读到的值不写出）
  const bf16* wp[R];
#pragma unroll
  for (int r = 0; r < R; ++r) wp[r] = w + (long)min(n0 + 8 * r + g, n - 1) * k + t * 8;

  for (int k0 = k_begin; k0 < k_end; k0 += TK) {
    const int kt = min(TK, k_end - k0);  // 本段列数（32 的倍数）
    // 行 ≥ m 的 smem 不读（寄存器会读到残留值，但对应输出列不写出；算进 mma 不影响别的列）
    for (int idx = threadIdx.x; idx < ROWS * (TK / 8); idx += 32 * WARPS) {
      const int rr = idx / (TK / 8), c8 = idx % (TK / 8);
      if (rr < m && c8 * 8 < kt)
        *reinterpret_cast<uint4*>(&a_tile[rr][c8 * 8]) = *reinterpret_cast<const uint4*>(a + (long)rr * k + k0 + c8 * 8);
    }
    __syncthreads();
    if (active) {
      // 每轮 kMmaBatch 个 32 列 chunk：先把 R × kMmaBatch 条 16 B W 装载全部发出（在途 R·kMmaBatch·512 B/warp），再做 mma。
      // 第 3 轮每轮只 1 chunk（R KB 在途）停在 4–5 TB/s：HBM 延迟 × 8 TB/s 需要每 SM ≥ ~45 KB 在途。
      for (int c = 0; c < kt; c += 32 * kMmaBatch) {
        uint4 wv[R][kMmaBatch];
#pragma unroll
        for (int j = 0; j < kMmaBatch; ++j)
#pragma unroll
          for (int r = 0; r < R; ++r) wv[r][j] = __ldg(reinterpret_cast<const uint4*>(wp[r] + k0 + c + j * 32));
#pragma unroll
        for (int j = 0; j < kMmaBatch; ++j) {
          const int cc = c + j * 32;
          // A 片段：行 g / g+8（每组），列 [cc + t·8, +8)
          uint4 av0[MG], av1[MG];
#pragma unroll
          for (int i = 0; i < MG; ++i) {
            av0[i] = *reinterpret_cast<const uint4*>(&a_tile[i * 16 + g][cc + t * 8]);
            av1[i] = *reinterpret_cast<const uint4*>(&a_tile[i * 16 + g + 8][cc + t * 8]);
          }
#pragma unroll
          for (int r = 0; r < R; ++r)
#pragma unroll
            for (int i = 0; i < MG; ++i) {
              mma16816_bf16(acc[r][i], av0[i].x, av1[i].x, av0[i].y, av1[i].y, wv[r][j].x, wv[r][j].y);
              mma16816_bf16(acc[r][i], av0[i].z, av1[i].z, av0[i].w, av1[i].w, wv[r][j].z, wv[r][j].w);
            }
        }
      }
    }
    __syncthreads();
  }
  if (!active) return;
  // D 片段：c0/c1 → (M 行 g, N 列 2t/2t+1)，c2/c3 → (M 行 g+8, 同列)；N 列 = n0 + 8r + 2t (+1)
  if (splits <= 1) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int nc = n0 + 8 * r + 2 * t;
      if (nc >= n) continue;
      const bool two = nc + 1 < n;
      const float b0 = bias ? to_f32(bias[nc]) : 0.f;
      const float b1 = bias && two ? to_f32(bias[nc + 1]) : 0.f;
#pragma unroll
      for (int i = 0; i < MG; ++i) {
        const int m0 = i * 16 + g, m1 = m0 + 8;
        if (m0 < m) {
          out[(long)m0 * n + nc] = to_bf16(acc[r][i][0] + b0);
          if (two) out[(long)m0 * n + nc + 1] = to_bf16(acc[r][i][1] + b1);
        }
        if (m1 < m) {
          out[(long)m1 * n + nc] = to_bf16(acc[r][i][2] + b0);
          if (two) out[(long)m1 * n + nc + 1] = to_bf16(acc[r][i][3] + b1);
        }
      }
    }
    return;
  }
  float* p = partial + (long)blockIdx.y * m * n;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int nc = n0 + 8 * r + 2 * t;
    if (nc >= n) continue;
    const bool two = nc + 1 < n;
#pragma unroll
    for (int i = 0; i < MG; ++i) {
      const int m0 = i * 16 + g, m1 = m0 + 8;
      if (m0 < m) {
        p[(long)m0 * n + nc] = acc[r][i][0];
        if (two) p[(long)m0 * n + nc + 1] = acc[r][i][1];
      }
      if (m1 < m) {
        p[(long)m1 * n + nc] = acc[r][i][2];
        if (two) p[(long)m1 * n + nc + 1] = acc[r][i][3];
      }
    }
  }
  __threadfence();
  // 每 stripe 一个计数器（index = N 行 / 8）；最后到达的 warp 归约该 stripe 的 8 行 × m 列
  int* ctr = counters + n0 / 8;
  int last = 0;
  if (lane < R) {
    const int ns = n0 + 8 * lane;
    if (ns < n) last = (atomicAdd(ctr + lane, 1) == splits - 1);
  }
  const unsigned mask = __ballot_sync(0xffffffffu, last);
  if (mask == 0) return;
  __threadfence();
  for (int r = 0; r < R; ++r) {
    if (!((mask >> r) & 1u)) continue;
    const int nr = n0 + 8 * r + (lane & 7);
    if (nr < n) {
      const float b = bias ? to_f32(bias[nr]) : 0.f;
      for (int mc = lane >> 3; mc < m; mc += 4) {
        float v = b;
        for (int sp = 0; sp < splits; ++sp) v += __ldcg(partial + ((long)sp * m + mc) * n + nr);
        out[(long)mc * n + nr] = to_bf16(v);
      }
    }
    if (lane == 0) ctr[r] = 0;  // 自复位：下一次 launch / graph 重放前计数器回零
  }
}

template <int M_TILE, int WARPS, int R>
void launch_mma(const bf16* a, const bf16* w, const bf16* bias, bf16* out, float* partial, int* counters, int m,
                int n, int k, int splits, int k_per_split, cudaStream_t stream) {
  dim3 grid((unsigned)((n + 8 * R * WARPS - 1) / (8 * R * WARPS)), (unsigned)splits);
  small_m_mma_kernel<M_TILE, WARPS, R><<<grid, 32 * WARPS, 0, stream>>>(a, w, bias, out, partial, counters, m, n, k,
                                                                        k_per_split, splits);
}
template <int M_TILE, int WARPS>
void launch_mma_r(int stripes, const bf16* a, const bf16* w, const bf16* bias, bf16* out, float* partial, int* counters,
                  int m, int n, int k, int splits, int k_per_split, cudaStream_t stream) {
  switch (stripes) {
    case 1: launch_mma<M_TILE, WARPS, 1>(a, w, bias, out, partial, counters, m, n, k, splits, k_per_split, stream); break;
    case 2: launch_mma<M_TILE, WARPS, 2>(a, w, bias, out, partial, counters, m, n, k, splits, k_per_split, stream); break;
    default: launch_mma<M_TILE, WARPS, 4>(a, w, bias, out, partial, counters, m, n, k, splits, k_per_split, stream); break;
  }
}
template <int M_TILE>
void launch_mma_w(int warps, int stripes, const bf16* a, const bf16* w, const bf16* bias, bf16* out, float* partial,
                  int* counters, int m, int n, int k, int splits, int k_per_split, cudaStream_t stream) {
  if (warps == 8)
    launch_mma_r<M_TILE, 8>(stripes, a, w, bias, out, partial, counters, m, n, k, splits, k_per_split, stream);
  else
    launch_mma_r<M_TILE, 4>(stripes, a, w, bias, out, partial, counters, m, n, k, splits, k_per_split, stream);
}
}  // namespace

extern "C" {

// splits：K 维切几份（1 = 不切，直接写 out）。>1 时 partial 至少 splits·M·N 个 f32。
// 返回 cudaError_t；m > 32 / k % 8 != 0 返回 cudaErrorInvalidValue（host 侧不该走到这里）。
int gemm_small_m_bf16(const void* a, const void* w, const void* bias, void* out, float* partial, int64_t m,
                      int64_t n, int64_t k, int64_t splits, cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0) return 0;
  if (m > 32 || (k % 8) != 0 || splits <= 0) return (int)cudaErrorInvalidValue;
  if (splits > 1 && !partial) return (int)cudaErrorInvalidValue;
  // 每份按 8 对齐；份数不超过实际需要
  int64_t kps = ((k + splits - 1) / splits + 7) / 8 * 8;
  int64_t s = (k + kps - 1) / kps;
  float* part = s > 1 ? partial : nullptr;
  const bf16* bias_for_kernel = s > 1 ? nullptr : (const bf16*)bias;
  const bf16* A = (const bf16*)a;
  const bf16* W = (const bf16*)w;
  bf16* O = (bf16*)out;
  if (m == 1) launch<1>(A, W, bias_for_kernel, O, part, (int)m, (int)n, (int)k, (int)s, (int)kps, stream);
  else if (m == 2) launch<2>(A, W, bias_for_kernel, O, part, (int)m, (int)n, (int)k, (int)s, (int)kps, stream);
  else if (m <= 4) launch<4>(A, W, bias_for_kernel, O, part, (int)m, (int)n, (int)k, (int)s, (int)kps, stream);
  else if (m <= 8) launch<8>(A, W, bias_for_kernel, O, part, (int)m, (int)n, (int)k, (int)s, (int)kps, stream);
  else if (m <= 16) launch<16>(A, W, bias_for_kernel, O, part, (int)m, (int)n, (int)k, (int)s, (int)kps, stream);
  else launch<32>(A, W, bias_for_kernel, O, part, (int)m, (int)n, (int)k, (int)s, (int)kps, stream);
  if (s > 1) {
    const long mn = (long)m * n;
    const int blocks = (int)((mn + 255) / 256);
    reduce_kernel<<<blocks, 256, 0, stream>>>(partial, (const bf16*)bias, O, (int)m, (int)n, (int)s);
  }
  return last_err();
}

// tensor-core 版入口：K % 128 == 0，M ≤ 32，warps ∈ {4, 8}，stripes ∈ {1,2,4}；splits > 1 时 partial[splits,M,N] f32 与
// counters[ceil(N/8)] int（调用方保证初值 0；kernel 自复位）。见 small_m_mma_kernel。
int gemm_small_m_mma_bf16(const void* a, const void* w, const void* bias, void* out, float* partial, int* counters,
                          int64_t m, int64_t n, int64_t k, int64_t splits, int64_t warps, int64_t stripes,
                          cudaStream_t stream) {
  constexpr int64_t kAlign = 32 * kMmaBatch;  // 128
  if (m <= 0 || m > 32 || n <= 0 || k <= 0 || k % kAlign != 0 || splits <= 0) return 1;
  if (warps != 4 && warps != 8) return 1;
  if (stripes != 1 && stripes != 2 && stripes != 4) return 1;
  if (splits > 1 && (!partial || !counters)) return 1;
  // 每段 128 的倍数，份数 ≤ k/128
  int64_t sp = splits;
  int64_t per = ((k / kAlign + sp - 1) / sp) * kAlign;
  sp = (k + per - 1) / per;
  if (sp <= 1) { sp = 1; per = k; }
  const bf16* A = static_cast<const bf16*>(a);
  const bf16* W = static_cast<const bf16*>(w);
  const bf16* B = static_cast<const bf16*>(bias);
  bf16* O = static_cast<bf16*>(out);
  const int mi = (int)m, ni = (int)n, ki = (int)k, si = (int)sp, pi = (int)per, wi = (int)warps, ri = (int)stripes;
  if (m <= 8)
    launch_mma_w<8>(wi, ri, A, W, B, O, partial, counters, mi, ni, ki, si, pi, stream);
  else if (m <= 16)
    launch_mma_w<16>(wi, ri, A, W, B, O, partial, counters, mi, ni, ki, si, pi, stream);
  else
    launch_mma_w<32>(wi, ri, A, W, B, O, partial, counters, mi, ni, ki, si, pi, stream);
  return (int)cudaGetLastError();
}

}  // namespace dinfer
}  // extern "C"
