// Qwen3.5 GDN 层 decode 的**单 persistent kernel**（质变实验第一阶段，docs/loops/mega-sm100.md、design.md §14.3）。
//
// 把现路径的 11 个 kernel（rmsnorm → in_proj GEMV → conv1d_update → GDN 递归(fused gate) → gated_rmsnorm → out_proj
// → add+rmsnorm → gate_up GEMV → silu·mul → down GEMV → add）编成一张静态任务表，一次 launch、一个常驻 grid 跑完：
//   - grid = occupancy × SM（host 查），block = kWarps 个 warp；任务 id 全局静态编号 = (阶段, 分片)，CTA 按
//     task = blockIdx.x + i·gridDim.x 顺序领取（阶段内等代价，静态轮转即可；id 单调保证无环等待）；
//   - 依赖 = 全局 ready[stage] 计数器：生产者 `__syncthreads → threadfence → red.release.gpu +1`，
//     消费者 thread 0 `ld.acquire.gpu` 自旋到 ready[dep] == count[dep] 再 `__syncthreads`（dep = 前一个非空阶段）；
//   - GEMV 阶段：权重按 N 切 tile 给任务（tile = kWarps·G·8 行），K 维再按 split-K 切（任务 = tile × split），
//     权重 tile 用 cp.async.cg 16 B 多级流水进 smem（**依赖等待前就发**，把"等上一阶段"藏在权重流里）；
//     激活 A（M 行 × K）同样 cp.async 进 smem；算子是 mma.sync.m16n8k16 bf16（A/B 都按"每 lane 16 B 连续 K"
//     的置换布局装载，A、B 同置换所以乘积不变）；split-K 用 f32 partial + 每 tile 信号量，最后到的任务归约写 bf16；
//     尾声按 mode 融合：out_proj / down 直接 res += bf16(sum)（省 add 阶段），gate_up 的 tile 同时覆盖 gate 行与
//     对应 up 行（虚拟行号 → 物理行），尾声直接写 act = silu(g)·u（省 silu 阶段）→ 11 阶段只剩 8 个非空；
//   - GDN 递归阶段：复用 gdn_decode.cuh 的寄存器方案；任务 = (seq, 连续 gdn_hv 个 v 头)，每 warp ROWS 行状态，
//     下一 tile 的状态用 cp.async 预取进 smem（每 lane 只取自己要读的 16 B，warp 内无需同步）—— 常驻 grid 占用
//     只有 1–2 CTA/SM，必须用软件流水补占用（第 1 轮 B=32 1536 任务 × 4.9 µs 串行 = 80 µs）；
//   - 阶段间激活走全局（L2 常驻），本 kernel 内"先读后被别的 SM 改写再读"的缓冲（res、partial、o/z）一律 __ldcg。
//
// 所有分片大小 / 流水深度 / CTA 数 / split 数都由 host（`ArchProfile::mega_params` + 启动自调优）给，kernel 内只有
// 结构常量（warp 数、mma 形状）。按多卡型分文件规则本文件只给 Blackwell 编（build.rs 按 CUDA_ARCHS 选），
// 体内不写 `#if __CUDA_ARCH__`；第一版只用 sm80+ 指令（cp.async.cg / mma.sync / ld.acquire），TMA 留第二版。
//
// 规格 / 记录：docs/loops/mega-sm100.md。对拍与计时：crates/kernels/tests/mega_parity.rs。
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "common.cuh"
#include "gdn_decode.cuh"
#include "kernels.h"

namespace dinfer {
namespace {

constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kChunkK = 32;  // 一个 K 块 = 两条 m16n8k16 = 每 lane 16 B（8 个 bf16）
constexpr int kStageCount = 11;
constexpr int kMaxPipe = 6;
constexpr int kGdnKpl = 4;  // K = 128

enum Stage : int {
  S_NORM0 = 0,
  S_INPROJ,
  S_CONV,
  S_GDN,
  S_GNORM,
  S_OUTPROJ,
  S_ADDNORM,
  S_GATEUP,
  S_SILU,  // 已融进 gate_up 尾声（count = 0）
  S_DOWN,
  S_ADD,   // 已融进 down 尾声（count = 0）
};

// 一个 GEMV 阶段：acc[M, N] = A[M, K] · W[N, K]^T（f32 累加）。mode：0 = out ← bf16(acc)；
// 1 = out(res) ← bf16(res + bf16(acc))（与基线 add_inplace 的两次舍入一致）；
// 2 = silu·mul：tile 的前半行是 gate 行 [n0/2, n0/2 + TN/2)、后半是对应 up 行（物理行 + inter），out[M, inter] ← silu(g)·u。
struct GemvDesc {
  const bf16* w;
  const bf16* a;
  bf16* out;
  float* partial;  // [splits][M][N] f32（splits > 1）
  int* sem;        // [tiles]
  int n, k, tiles, splits, spl, total, mode, inter;
};

struct Params {
  MegaLayerArgs a;
  GemvDesc gemv[4];  // in_proj / out_proj / gate_up / down
  int count[kStageCount];
  int start[kStageCount + 1];
  int dep[kStageCount];  // 依赖的阶段（-1 无）
  int conv_dim, in_row, z_off, a_off, b_off, group;
  int pipe;  // 流水深度（运行时，2 ≤ pipe ≤ kMaxPipe）
  int gdn_hv;  // 每个 GDN 任务连续处理的 v 头数
  int* ready;  // [kStageCount]
  int* done;
};

// ---------------------------------------------------------------- 同步原语
__device__ __forceinline__ int ld_acquire_gpu(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void red_release_gpu_add(int* p, int v) {
  asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
  return t;
}
__shared__ unsigned long long s_wait_end;  // prof：本任务依赖等待结束时刻（thread 0 写 / 读）

// 等依赖阶段全部完成。只 thread 0 自旋，其余在 barrier 等。
__device__ __forceinline__ void wait_stage(const Params& p, int stage) {
  const int dep = p.dep[stage];
  if (dep < 0) {
    if (p.a.prof && threadIdx.x == 0) s_wait_end = gtimer();
    return;
  }
  if (threadIdx.x == 0) {
    const int need = p.count[dep];
    const int* flag = p.ready + dep;
    while (ld_acquire_gpu(flag) < need) __nanosleep(40);
    if (p.a.prof) s_wait_end = gtimer();
  }
  __syncthreads();
}

__device__ __forceinline__ void signal_stage(const Params& p, int stage) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    red_release_gpu_add(p.ready + stage, 1);
  }
}

// ---------------------------------------------------------------- cp.async（带 zfill）
__device__ __forceinline__ void cp_async16_zfill(void* smem, const void* gmem, int src_bytes) {
  const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(src_bytes));
}
// 第 i 轮顶部已提交 pipe-1+i 个 group，需要 group i 完成 → 允许挂起 pipe-2 个。
// wait_group 的立即数必须编译期常量；pipe 是 block 一致的运行时值（≥ 2）。
__device__ __forceinline__ void cp_async_wait_pipe(int pipe) {
  switch (pipe) {
    case 2: asm volatile("cp.async.wait_group 0;\n" ::); break;
    case 3: asm volatile("cp.async.wait_group 1;\n" ::); break;
    case 4: asm volatile("cp.async.wait_group 2;\n" ::); break;
    case 5: asm volatile("cp.async.wait_group 3;\n" ::); break;
    default: asm volatile("cp.async.wait_group 4;\n" ::); break;
  }
}

__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t a0, const uint32_t a1, const uint32_t a2,
                                         const uint32_t a3, const uint32_t b0, const uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// smem tile 行内 16 B 块的 swizzle：mma 片段读是 8 行 × 4 块一个 wavefront；行距 4U 块（U=2 → 32 B，4 → 64 B，8 → 128 B）。
// 奇数行把块号的 bit2 翻转（U ≥ 4 时 8 行落 8 个 bank 组；U=2 行距 32 B 本身 4 行一组错开，再翻 bit2 无害）。
__device__ __forceinline__ int swz(int row, int chunk) { return chunk ^ ((row & 1) << 2); }

// mode 2 的虚拟行号 → 物理权重行：tile 内前半是 gate 行，后半是同列的 up 行（+inter）。
__device__ __forceinline__ int phys_row(const GemvDesc& g, int tile, int TN, int r) {
  if (g.mode != 2) return tile * TN + r;
  const int half = TN / 2;
  const int col = tile * half + (r < half ? r : r - half);
  return r < half ? col : col + g.inter;
}

// ---------------------------------------------------------------- GEMV 任务
// MB：M 方向 16 行块数（M ≤ 16·MB）；G：每 warp 的 8 行 N 组数（tile = kWarps·G·8 行）；U：每流水 stage 的 K 块数。
template <int MB, int G, int U>
__device__ __forceinline__ void gemv_task(const Params& p, const GemvDesc& g, int shard, int stage, uint8_t* smem) {
  constexpr int TN = kWarps * G * 8;
  constexpr int CPR = 4 * U;  // 每行每 stage 的 16 B 块数
  constexpr int KS = kChunkK * U;  // 每 stage 的 K 列数
  constexpr int W_BYTES = TN * CPR * 16;
  constexpr int A_BYTES = MB * 16 * CPR * 16;
  constexpr int W_PER_T = TN * CPR / kThreads;
  constexpr int A_PER_T = (MB * 16 * CPR + kThreads - 1) / kThreads;
  static_assert(TN * CPR % kThreads == 0, "W tile 块数须被线程数整除");

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int m = (int)p.a.batch;
  const int tile = shard / g.splits, split = shard - tile * g.splits;
  const int s_begin = min(split * g.spl, g.total);
  const int s_end = min(s_begin + g.spl, g.total);
  const int nst = s_end - s_begin;
  const int kbase = s_begin * KS;
  const int pipe = p.pipe;
  uint8_t* wbuf = smem;
  uint8_t* abuf = smem + pipe * W_BYTES;
  const int n_lim = g.mode == 2 ? 2 * g.inter : g.n;  // 物理行上界

  auto issue_w = [&](int st, int buf) {
    const int koff = kbase + st * KS;
    uint8_t* dst = wbuf + buf * W_BYTES;
#pragma unroll
    for (int i = 0; i < W_PER_T; ++i) {
      const int idx = tid + i * kThreads;
      const int row = idx / CPR, chunk = idx - row * CPR;
      const int nrow = phys_row(g, tile, TN, row);
      const bool ok = nrow < n_lim && (g.mode != 2 || (nrow % g.inter) < g.inter);
      const bf16* src = g.w + (long)(ok ? nrow : 0) * g.k + koff + chunk * 8;
      cp_async16_zfill(dst + (row * CPR + swz(row, chunk)) * 16, src, ok ? 16 : 0);
    }
  };
  auto issue_a = [&](int st, int buf) {
    const int koff = kbase + st * KS;
    uint8_t* dst = abuf + buf * A_BYTES;
#pragma unroll
    for (int i = 0; i < A_PER_T; ++i) {
      const int idx = tid + i * kThreads;
      if (idx < MB * 16 * CPR) {
        const int row = idx / CPR, chunk = idx - row * CPR;
        const bool ok = row < m;
        const bf16* src = g.a + (long)(ok ? row : 0) * g.k + koff + chunk * 8;
        cp_async16_zfill(dst + (row * CPR + swz(row, chunk)) * 16, src, ok ? 16 : 0);
      }
    }
  };

  float acc[G][MB][4];
#pragma unroll
  for (int i = 0; i < G; ++i)
#pragma unroll
    for (int j = 0; j < MB; ++j)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;

  // 序幕：权重流水先发（不依赖上一阶段），再等依赖，再发 A。group s = {W(0..pipe-2), A(0)} / {A(s)}。
  const int pro = min(pipe - 1, nst);
  for (int s = 0; s < pro; ++s) issue_w(s, s);
  wait_stage(p, stage);
  for (int s = 0; s < pipe - 1; ++s) {
    if (s < nst) issue_a(s, s);
    cp_async_commit_g();
  }

  const int t4 = lane & 3, r8 = lane >> 2;
  for (int i = 0; i < nst; ++i) {
    cp_async_wait_pipe(pipe);
    __syncthreads();
    {
      const int nx = i + pipe - 1;
      if (nx < nst) {
        const int buf = nx % pipe;
        issue_w(nx, buf);
        issue_a(nx, buf);
      }
      cp_async_commit_g();
    }
    const int buf = i % pipe;
    const uint8_t* wb = wbuf + buf * W_BYTES;
    const uint8_t* ab = abuf + buf * A_BYTES;
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int chunk = u * 4 + t4;
      uint4 af[MB][2];
#pragma unroll
      for (int mb = 0; mb < MB; ++mb) {
        const int rlo = mb * 16 + r8, rhi = rlo + 8;
        af[mb][0] = *reinterpret_cast<const uint4*>(ab + (rlo * CPR + swz(rlo, chunk)) * 16);
        af[mb][1] = *reinterpret_cast<const uint4*>(ab + (rhi * CPR + swz(rhi, chunk)) * 16);
      }
#pragma unroll
      for (int gi = 0; gi < G; ++gi) {
        const int wrow = (warp * G + gi) * 8 + r8;
        const uint4 bf = *reinterpret_cast<const uint4*>(wb + (wrow * CPR + swz(wrow, chunk)) * 16);
#pragma unroll
        for (int mb = 0; mb < MB; ++mb) {
          mma16816(acc[gi][mb], af[mb][0].x, af[mb][1].x, af[mb][0].y, af[mb][1].y, bf.x, bf.y);
          mma16816(acc[gi][mb], af[mb][0].z, af[mb][1].z, af[mb][0].w, af[mb][1].w, bf.z, bf.w);
        }
      }
    }
  }
  cp_async_wait_all_g();
  __syncthreads();  // 尾声可能复用 smem（mode 2 staging）

  // 尾声：acc[gi][mb] 的 c0,c1 = (行 mb·16 + r8, 列 2·t4 + {0,1})，c2,c3 = 行 +8（列 = tile 内虚拟行号）。
  auto store_pair = [&](int r, int vcol, float v0, float v1) {
    // vcol：tile 内虚拟列（偶数）；对 mode 0/1 物理列 = n0 + vcol
    if (g.mode == 2) return;  // mode 2 走 smem staging
    const int ncol = tile * TN + vcol;
    if (ncol >= g.n) return;
    bf16* dst = g.out + (long)r * g.n + ncol;
    if (g.mode == 1) {
      const __nv_bfloat162 rb = __ldcg(reinterpret_cast<const __nv_bfloat162*>(dst));
      const float2 rf = __bfloat1622float2(rb);
      const float2 sf = __bfloat1622float2(__floats2bfloat162_rn(v0, v1));
      *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(rf.x + sf.x, rf.y + sf.y);
    } else {
      *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
    }
  };

  if (g.splits == 1) {
    if (g.mode == 2) {
      // staging：f32 [m][TN]（m ≤ 32，TN ≤ 256 → ≤ 32 KB，远小于流水 smem）
      float* stg = reinterpret_cast<float*>(smem);
#pragma unroll
      for (int gi = 0; gi < G; ++gi) {
        const int vcol = (warp * G + gi) * 8 + 2 * t4;
#pragma unroll
        for (int mb = 0; mb < MB; ++mb) {
          const int rlo = mb * 16 + r8, rhi = rlo + 8;
          if (rlo < m) *reinterpret_cast<float2*>(stg + rlo * TN + vcol) = make_float2(acc[gi][mb][0], acc[gi][mb][1]);
          if (rhi < m) *reinterpret_cast<float2*>(stg + rhi * TN + vcol) = make_float2(acc[gi][mb][2], acc[gi][mb][3]);
        }
      }
      __syncthreads();
      const int half = TN / 2, c0 = tile * half;
      for (int idx = tid; idx < m * half; idx += kThreads) {
        const int r = idx / half, c = idx - r * half;
        if (c0 + c < g.inter) {
          const float gv = stg[r * TN + c], uv = stg[r * TN + half + c];
          g.out[(long)r * g.inter + c0 + c] = to_bf16(silu(gv) * uv);
        }
      }
      return;
    }
#pragma unroll
    for (int gi = 0; gi < G; ++gi) {
      const int vcol = (warp * G + gi) * 8 + 2 * t4;
#pragma unroll
      for (int mb = 0; mb < MB; ++mb) {
        const int rlo = mb * 16 + r8, rhi = rlo + 8;
        if (rlo < m) store_pair(rlo, vcol, acc[gi][mb][0], acc[gi][mb][1]);
        if (rhi < m) store_pair(rhi, vcol, acc[gi][mb][2], acc[gi][mb][3]);
      }
    }
    return;
  }
  // split-K：partial 按物理列 [split][m][n]（mode 2 的 n = 2·inter，gate / up 自然落在 c 与 inter + c）
  float* part = g.partial + (long)split * m * g.n;
#pragma unroll
  for (int gi = 0; gi < G; ++gi) {
    const int vrow = (warp * G + gi) * 8 + 2 * t4;
    const int ncol = phys_row(g, tile, TN, vrow);
    if (ncol >= n_lim) continue;
#pragma unroll
    for (int mb = 0; mb < MB; ++mb) {
      const int rlo = mb * 16 + r8, rhi = rlo + 8;
      if (rlo < m) *reinterpret_cast<float2*>(part + (long)rlo * g.n + ncol) = make_float2(acc[gi][mb][0], acc[gi][mb][1]);
      if (rhi < m) *reinterpret_cast<float2*>(part + (long)rhi * g.n + ncol) = make_float2(acc[gi][mb][2], acc[gi][mb][3]);
    }
  }
  __shared__ int s_last;
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    const int old = atomicAdd(g.sem + tile, 1);
    s_last = (old == g.splits - 1) ? 1 : 0;
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  if (g.mode == 2) {
    const int half = TN / 2, c0 = tile * half;
    const int ncols = min(half, g.inter - c0);
    for (int idx = tid; idx < m * ncols; idx += kThreads) {
      const int r = idx / ncols, c = idx - r * ncols;
      float gv = 0.f, uv = 0.f;
      for (int sp = 0; sp < g.splits; ++sp) {
        const float* pr = g.partial + ((long)sp * m + r) * g.n;
        gv += __ldcg(pr + c0 + c);
        uv += __ldcg(pr + g.inter + c0 + c);
      }
      g.out[(long)r * g.inter + c0 + c] = to_bf16(silu(gv) * uv);
    }
  } else {
    const int n0 = tile * TN;
    const int ncols = min(TN, g.n - n0);
    for (int idx = tid * 2; idx < m * ncols; idx += kThreads * 2) {
      const int r = idx / ncols, c = idx - r * ncols;  // ncols 偶数，idx 偶 → c 偶且 c+1 < ncols
      const long off = (long)r * g.n + n0 + c;
      float2 s = make_float2(0.f, 0.f);
      for (int sp = 0; sp < g.splits; ++sp) {
        const float2 v = __ldcg(reinterpret_cast<const float2*>(g.partial + (long)sp * m * g.n + off));
        s.x += v.x;
        s.y += v.y;
      }
      store_pair(r, c, s.x, s.y);
    }
  }
  if (tid == 0) g.sem[tile] = 0;  // 自复位（下次 launch 前不需要 memset）
}

// ---------------------------------------------------------------- rmsnorm（一行 / 任务）
// 口径与 norm.cu rmsnorm_vec_kernel 一致：out = bf16(x · rstd · w)，f32 平方和。16 B 向量；hidden ≤ 8·kThreads·kNormMaxVec。
constexpr int kNormMaxVec = 4;
__device__ __forceinline__ void rmsnorm_row(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                            bf16* __restrict__ out, int hidden, float eps) {
  const int tid = threadIdx.x;
  const int nvec = hidden / 8;
  const uint4* xp = reinterpret_cast<const uint4*>(x);
  const uint4* wp = reinterpret_cast<const uint4*>(w);
  uint4* op = reinterpret_cast<uint4*>(out);
  uint4 xa[kNormMaxVec], wa[kNormMaxVec];
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < kNormMaxVec; ++j) {
    const int i = tid + j * kThreads;
    if (i < nvec) {
      xa[j] = __ldcg(xp + i);
      wa[j] = wp[i];
    }
  }
#pragma unroll
  for (int j = 0; j < kNormMaxVec; ++j) {
    const int i = tid + j * kThreads;
    if (i < nvec) {
      const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xa[j]);
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float2 f = __bfloat1622float2(x2[c]);
        ss += f.x * f.x + f.y * f.y;
      }
    }
  }
  ss = block_reduce_sum<kThreads>(ss);
  const float inv = rsqrtf(ss / hidden + eps);
#pragma unroll
  for (int j = 0; j < kNormMaxVec; ++j) {
    const int i = tid + j * kThreads;
    if (i < nvec) {
      const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xa[j]);
      const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wa[j]);
      uint4 ob;
      __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&ob);
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float2 f = __bfloat1622float2(x2[c]);
        const float2 g = __bfloat1622float2(w2[c]);
        o2[c] = __floats2bfloat162_rn(f.x * inv * g.x, f.y * inv * g.y);
      }
      op[i] = ob;
    }
  }
}

// ---------------------------------------------------------------- conv1d update（任务 = (seq, 2·kThreads 维一段)）
// 每线程 2 个相邻维：先把 state / x / w 全部读进寄存器（无依赖，一起在飞），再算、再写。
constexpr int kConvDimsPerThread = 2;
constexpr int kConvChunk = kThreads * kConvDimsPerThread;
constexpr int kConvMaxWidth = 8;
__device__ __forceinline__ void conv_task(const Params& p, int shard) {
  const int chunks = (p.conv_dim + kConvChunk - 1) / kConvChunk;
  const int b = shard / chunks, c = shard - b * chunks;
  const int slot = p.a.state_indices ? p.a.state_indices[b] : b;
  if (slot < 0) return;
  const int width = (int)p.a.width, sl = width - 1;
  const bf16* x = (const bf16*)p.a.qkvzab + (long)b * p.in_row;
  const bf16* w = (const bf16*)p.a.conv_w;
  bf16* st_base = (bf16*)p.a.conv_state + (long)slot * p.conv_dim * sl;
  bf16* out = (bf16*)p.a.conv_out + (long)b * p.conv_dim;
  const int d0 = c * kConvChunk + threadIdx.x * kConvDimsPerThread;
  if (d0 >= p.conv_dim) return;
  float xs[kConvDimsPerThread], sv[kConvDimsPerThread][kConvMaxWidth], wv[kConvDimsPerThread][kConvMaxWidth];
#pragma unroll
  for (int e = 0; e < kConvDimsPerThread; ++e) {
    const int d = d0 + e;
    xs[e] = to_f32(x[d]);
#pragma unroll
    for (int j = 0; j < kConvMaxWidth; ++j) {
      sv[e][j] = j < sl ? to_f32(st_base[(long)d * sl + j]) : 0.f;
      wv[e][j] = j < width ? to_f32(w[d * width + j]) : 0.f;
    }
  }
#pragma unroll
  for (int e = 0; e < kConvDimsPerThread; ++e) {
    const int d = d0 + e;
    float acc = wv[e][sl] * xs[e];
#pragma unroll
    for (int j = 0; j < kConvMaxWidth - 1; ++j)
      if (j < sl) acc += wv[e][j] * sv[e][j];
    out[d] = to_bf16(act_f(acc, 0));
    bf16* st = st_base + (long)d * sl;
#pragma unroll
    for (int j = 0; j < kConvMaxWidth - 1; ++j)
      if (j + 1 < sl) st[j] = to_bf16(sv[e][j + 1]);
    st[sl - 1] = to_bf16(xs[e]);
  }
}

// ---------------------------------------------------------------- GDN 递归（任务 = (seq, 连续 gdn_hv 个 v 头)）
// 每 warp ROWS 行状态（ROWS·kWarps = V），tile = 一个 (seq, hv)；任务内按 hv 循环，下一 tile 的状态用 cp.async
// 预取进 smem（每 lane 只取自己要读的 16 B 块，warp 内无需 __syncthreads）。smem 复用 GEMV 流水区（任务边界有 barrier）。
template <int ROWS>
__device__ __forceinline__ void gdn_task(const Params& p, int shard, uint8_t* smem) {
  static_assert(ROWS * kWarps == 128, "V = 128 一个任务整列");
  constexpr int KPL = kGdnKpl;
  constexpr int TILE_F = 128 * 128;  // 一个 (seq, hv) 的状态 f32 数
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int HV = (int)p.a.heads_v;
  const int groups = (HV + p.gdn_hv - 1) / p.gdn_hv;
  const int seq = shard / groups, grp = shard - seq * groups;
  const int hv0 = grp * p.gdn_hv, hv1 = min(hv0 + p.gdn_hv, HV);
  const int slot = p.a.state_indices ? p.a.state_indices[seq] : seq;
  if (slot < 0) return;
  DecodeArgs a;
  a.mixed = (const bf16*)p.a.conv_out;
  a.g_in = nullptr;
  a.beta_in = nullptr;
  a.a_raw = (const bf16*)p.a.qkvzab + p.a_off;
  a.b_raw = (const bf16*)p.a.qkvzab + p.b_off;
  a.a_log = p.a.a_log;
  a.dt_bias = p.a.dt_bias;
  a.out = (bf16*)p.a.o;
  a.state = (float*)p.a.rec_state;
  a.cu_seqlens = nullptr;
  a.state_indices = nullptr;
  a.H = (int)p.a.heads_k;
  a.HV = HV;
  a.K = (int)p.a.k_dim;
  a.V = (int)p.a.v_dim;
  a.group = p.group;
  a.scale = p.a.scale;
  a.state_stride_tok = (long)p.a.state_stride_tok;
  a.ab_stride = (long)p.in_row;
  const int v0 = warp * ROWS;
  float* sbuf = reinterpret_cast<float*>(smem);  // [2][TILE_F]，本 warp 只碰自己 ROWS 行
  auto tile_ptr = [&](int hv) {
    return a.state + (long)slot * a.state_stride_tok + (long)hv * a.V * a.K + (long)v0 * a.K + lane * KPL;
  };
  auto prefetch = [&](int hv, int buf) {
    const float* src = tile_ptr(hv);
    float* dst = sbuf + buf * TILE_F + (long)v0 * a.K + lane * KPL;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) cp_async16_smem(dst + (long)i * a.K, src + (long)i * a.K);
    cp_async_commit_g();
  };
  // gdn_bufs（host 给，按 smem 预算）：2 = 双缓冲，算当前 tile 时下一 tile 已在飞；1 = 单缓冲（sm8x 99 KB 放不下
  // 两份 64 KB），下一 tile 的读在本 tile 写回后才发，只能藏在"写回 + 下一轮装载"后面。
  const bool dbl = p.a.gdn_bufs >= 2;
  prefetch(hv0, 0);
  for (int hv = hv0; hv < hv1; ++hv) {
    const int buf = dbl ? (hv - hv0) & 1 : 0;
    if (dbl && hv + 1 < hv1) {
      prefetch(hv + 1, buf ^ 1);
      asm volatile("cp.async.wait_group 1;\n" ::);
    } else {
      cp_async_wait_all_g();
    }
    // 每 lane 只读自己 cp.async 的 16 B 块：wait_group 后即可见，warp 内无需同步
    const float* sp = sbuf + buf * TILE_F + (long)v0 * a.K + lane * KPL;
    float r[ROWS][KPL];
#pragma unroll
    for (int i = 0; i < ROWS; ++i) vec_load<KPL>(r[i], sp + (long)i * a.K);
    if (!dbl && hv + 1 < hv1) prefetch(hv + 1, 0);  // 已装进寄存器，smem 可覆盖
    decode_tile_body<ROWS, KPL, true, true>(a, r, seq, hv, v0, /*bos=*/seq, /*len=*/1, lane);
    float* Sg = tile_ptr(hv);
#pragma unroll
    for (int i = 0; i < ROWS; ++i) vec_store<KPL>(Sg + (long)i * a.K, r[i]);
  }
}

// ---------------------------------------------------------------- gated rmsnorm（任务 = kWarps 行，一 warp 一行 (b, hv)）
__device__ __forceinline__ void gnorm_task(const Params& p, int shard) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const long rows = p.a.batch * p.a.heads_v;
  const long row = (long)shard * kWarps + warp;
  if (row >= rows) return;
  const int D = (int)p.a.v_dim;
  const int hv = (int)(row % p.a.heads_v), b = (int)(row / p.a.heads_v);
  const bf16* xr = (const bf16*)p.a.o + row * D;
  const bf16* zr = (const bf16*)p.a.qkvzab + (long)b * p.in_row + p.z_off + (long)hv * D;
  const bf16* w = (const bf16*)p.a.gnorm_w;
  bf16* out = (bf16*)p.a.n + row * D;
  constexpr int kMaxDpl = 8;
  float xv[kMaxDpl], zv[kMaxDpl];
  float ss = 0.f;
#pragma unroll
  for (int c = 0; c < kMaxDpl; ++c) {
    const int d = lane + 32 * c;
    xv[c] = zv[c] = 0.f;
    if (d < D) {
      xv[c] = to_f32(xr[d]);
      zv[c] = to_f32(zr[d]);
      ss += xv[c] * xv[c];
    }
  }
  ss = warp_sum_f(ss);
  const float rstd = rsqrtf(ss / D + p.a.eps);
#pragma unroll
  for (int c = 0; c < kMaxDpl; ++c) {
    const int d = lane + 32 * c;
    if (d < D) out[d] = to_bf16(xv[c] * rstd * to_f32(w[d]) * act_f(zv[c], 0));
  }
}

// ---------------------------------------------------------------- persistent kernel
// OCC：每 SM 常驻 CTA 数（launch_bounds 的寄存器上限 = 64K / (kThreads·OCC)）；GDN 每 warp 行数随之 16 / 8。
template <int MB, int G, int U, int OCC>
__global__ void __launch_bounds__(kThreads, OCC) mega_layer_kernel(const Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int ROWS = 128 / kWarps;  // 16：V=128 一个任务整列
  (void)OCC;
  const int total = p.start[kStageCount];
  for (int task = blockIdx.x; task < total; task += gridDim.x) {
    int stage = 0;
    while (task >= p.start[stage + 1]) ++stage;
    const int shard = task - p.start[stage];
    unsigned long long t0 = 0;
    if (p.a.prof) t0 = gtimer();
    switch (stage) {
      case S_INPROJ: gemv_task<MB, G, U>(p, p.gemv[0], shard, stage, smem); break;
      case S_OUTPROJ: gemv_task<MB, G, U>(p, p.gemv[1], shard, stage, smem); break;
      case S_GATEUP: gemv_task<MB, G, U>(p, p.gemv[2], shard, stage, smem); break;
      case S_DOWN: gemv_task<MB, G, U>(p, p.gemv[3], shard, stage, smem); break;
      case S_NORM0:
        wait_stage(p, stage);
        rmsnorm_row((const bf16*)p.a.res + (long)shard * p.a.hidden, (const bf16*)p.a.norm_w,
                    (bf16*)p.a.xn + (long)shard * p.a.hidden, (int)p.a.hidden, p.a.eps);
        break;
      case S_CONV: wait_stage(p, stage); conv_task(p, shard); break;
      case S_GDN: wait_stage(p, stage); gdn_task<ROWS>(p, shard, smem); break;
      case S_GNORM: wait_stage(p, stage); gnorm_task(p, shard); break;
      case S_ADDNORM:
        wait_stage(p, stage);
        rmsnorm_row((const bf16*)p.a.res + (long)shard * p.a.hidden, (const bf16*)p.a.post_norm_w,
                    (bf16*)p.a.xn2 + (long)shard * p.a.hidden, (int)p.a.hidden, p.a.eps);
        break;
      default: break;
    }
    signal_stage(p, stage);
    if (p.a.prof && threadIdx.x == 0) {  // prof[stage][4] = {~min_start, max_end, Σwait, Σbusy}（ns，globaltimer）
      const unsigned long long t1 = gtimer();
      unsigned long long* q = p.a.prof + stage * 4;
      atomicMax(q, ~t0);
      atomicMax(q + 1, t1);
      atomicAdd(q + 2, s_wait_end - t0);
      atomicAdd(q + 3, t1 - s_wait_end);
    }
  }
  // 全部 CTA 收工后由最后一个把 ready / done 清零：下次 launch（含 graph 重放）不需要 host memset。
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const int old = atomicAdd(p.done, 1);
    if (old == (int)gridDim.x - 1) {
      for (int s = 0; s < kStageCount; ++s) p.ready[s] = 0;
      *p.done = 0;
      __threadfence();
    }
  }
}

// ---------------------------------------------------------------- host
constexpr int kCountersFixed = kStageCount + 1;

int ceil_div(long a, long b) { return (int)((a + b - 1) / b); }
bool mega_debug() {
  static int on = -1;
  if (on < 0) on = getenv("MEGA_DEBUG") ? 1 : 0;
  return on == 1;
}
#define MEGA_FAIL(fmt, ...)                                                        \
  do {                                                                            \
    if (mega_debug()) fprintf(stderr, "mega_sm100 invalid: " fmt "\n", ##__VA_ARGS__); \
    return (int)cudaErrorInvalidValue;                                            \
  } while (0)

template <int MB, int G, int U, int OCC>
int launch(Params& p, const MegaLayerArgs& a, cudaStream_t stream) {
  constexpr int TN = kWarps * G * 8;
  constexpr int CPR = 4 * U;
  const int pipe = p.pipe;
  size_t smem = (size_t)pipe * (TN * CPR * 16 + MB * 16 * CPR * 16);
  const size_t gdn_smem = (size_t)(a.gdn_bufs >= 2 ? 2 : 1) * 128 * 128 * 4;  // GDN 状态 tile（单 / 双缓冲）
  if (smem < gdn_smem) smem = gdn_smem;
  int dev = 0, sms = 1, max_smem = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  if ((int)smem > max_smem) MEGA_FAIL("smem %d > max %d (pipe %d G %d U %d MB %d)", (int)smem, max_smem, pipe, G, U, MB);
  auto kern = mega_layer_kernel<MB, G, U, OCC>;
  static int smem_set = 0;
  if (smem_set < (int)smem) {
    if (const int rc = err(cudaFuncSetAttribute((const void*)kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)))
      return rc;
    smem_set = (int)smem;
  }
  int occ = 0;
  if (const int rc = err(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, kThreads, smem))) return rc;
  if (occ <= 0) MEGA_FAIL("occupancy 0 (smem %d)", (int)smem);
  if (occ > OCC) occ = OCC;
  int grid = sms * occ;
  if (grid > p.start[kStageCount]) grid = p.start[kStageCount];
  if (a.info) {
    a.info[0] = grid;
    a.info[1] = occ;
    a.info[2] = (int)smem;
  }
  // GEMV tiles 用的 TN 由模板给；host（Rust）算 tiles 时必须用同一个 TN（groups·kWarps·8），这里复核。
  for (int i = 0; i < 4; ++i) {
    GemvDesc& g = p.gemv[i];
    const int expect = g.mode == 2 ? ceil_div(g.inter, TN / 2) : ceil_div(g.n, TN);
    if (g.tiles != expect) MEGA_FAIL("gemv %d tiles %d != %d", i, g.tiles, expect);
    if (g.k % (kChunkK * U) != 0) MEGA_FAIL("gemv %d k %d %% %d", i, g.k, kChunkK * U);
    g.total = g.k / (kChunkK * U);
    if (g.splits < 1 || g.splits > g.total) MEGA_FAIL("gemv %d splits %d total %d", i, g.splits, g.total);
    g.spl = ceil_div(g.total, g.splits);
  }
  kern<<<grid, kThreads, smem, stream>>>(p);
  return last_err();
}

template <int G, int U, int OCC>
int launch_mb(Params& p, const MegaLayerArgs& a, cudaStream_t stream) {
  if (a.batch <= 16) return launch<1, G, U, OCC>(p, a, stream);
  return launch<2, G, U, OCC>(p, a, stream);
}

}  // namespace

extern "C" {

int mega_gdn_layer_decode_sm100(const MegaLayerArgs* ap, cudaStream_t stream) {
  if (!ap) return (int)cudaErrorInvalidValue;
  const MegaLayerArgs& a = *ap;
  if (a.batch <= 0) return 0;
  if (a.batch > 32 || a.k_dim != 128 || a.v_dim != 128 || a.hidden % 8 != 0 ||
      a.hidden > 8 * kThreads * kNormMaxVec || a.width < 2 || a.width > kConvMaxWidth || a.heads_v % a.heads_k != 0 ||
      a.inter % 2 != 0 || a.gdn_hv < 1 || a.gdn_bufs < 1)
    MEGA_FAIL("shape: batch %ld k %ld v %ld hidden %ld width %ld hv %ld hk %ld inter %ld gdn_hv %ld gdn_bufs %ld", (long)a.batch,
              (long)a.k_dim, (long)a.v_dim, (long)a.hidden, (long)a.width, (long)a.heads_v, (long)a.heads_k, (long)a.inter,
              (long)a.gdn_hv, (long)a.gdn_bufs);
  if (a.stages < 2 || a.stages > kMaxPipe) MEGA_FAIL("stages %ld", (long)a.stages);

  Params p{};
  p.a = a;
  p.pipe = (int)a.stages;
  p.gdn_hv = (int)a.gdn_hv;
  p.conv_dim = (int)(2 * a.heads_k * a.k_dim + a.heads_v * a.v_dim);
  p.z_off = p.conv_dim;
  p.a_off = p.conv_dim + (int)(a.heads_v * a.v_dim);
  p.b_off = p.a_off + (int)a.heads_v;
  p.in_row = p.b_off + (int)a.heads_v;
  p.group = (int)(a.heads_v / a.heads_k);
  const int TN = kWarps * (int)a.groups * 8;
  const int B = (int)a.batch, hidden = (int)a.hidden, inter = (int)a.inter;
  const int sems_in = ceil_div(p.in_row, TN), sems_out = ceil_div(hidden, TN), sems_gu = ceil_div(inter, TN / 2),
            sems_down = ceil_div(hidden, TN);
  p.ready = a.counters;
  p.done = a.counters + kStageCount;
  int* sem = a.counters + kCountersFixed;
  p.gemv[0] = GemvDesc{(const bf16*)a.in_w, (const bf16*)a.xn, (bf16*)a.qkvzab, a.partial, sem, p.in_row, hidden, sems_in,
                       (int)a.splits_in, 0, 0, 0, 0};
  sem += sems_in;
  p.gemv[1] = GemvDesc{(const bf16*)a.out_w, (const bf16*)a.n, (bf16*)a.res, a.partial, sem, hidden,
                       (int)(a.heads_v * a.v_dim), sems_out, (int)a.splits_out, 0, 0, 1, 0};
  sem += sems_out;
  p.gemv[2] = GemvDesc{(const bf16*)a.gate_up_w, (const bf16*)a.xn2, (bf16*)a.act, a.partial, sem, 2 * inter, hidden,
                       sems_gu, (int)a.splits_gu, 0, 0, 2, inter};
  sem += sems_gu;
  p.gemv[3] = GemvDesc{(const bf16*)a.down_w, (const bf16*)a.act, (bf16*)a.res, a.partial, sem, hidden, inter, sems_down,
                       (int)a.splits_down, 0, 0, 1, 0};

  p.count[S_NORM0] = B;
  p.count[S_INPROJ] = sems_in * (int)a.splits_in;
  p.count[S_CONV] = B * ceil_div(p.conv_dim, kConvChunk);
  p.count[S_GDN] = B * ceil_div(a.heads_v, p.gdn_hv);
  p.count[S_GNORM] = ceil_div((long)B * a.heads_v, kWarps);
  p.count[S_OUTPROJ] = sems_out * (int)a.splits_out;
  p.count[S_ADDNORM] = B;
  p.count[S_GATEUP] = sems_gu * (int)a.splits_gu;
  p.count[S_SILU] = 0;
  p.count[S_DOWN] = sems_down * (int)a.splits_down;
  p.count[S_ADD] = 0;
  p.start[0] = 0;
  int last = -1;
  for (int s = 0; s < kStageCount; ++s) {
    p.start[s + 1] = p.start[s] + p.count[s];
    p.dep[s] = last;
    if (p.count[s] > 0) last = s;
  }

  const int occ = a.ctas_per_sm >= 2 ? 2 : 1;
  switch (occ * 256 + (int)a.groups * 16 + (int)a.unroll) {
    case 1 * 256 + 2 * 16 + 2: return launch_mb<2, 2, 1>(p, a, stream);
    case 1 * 256 + 2 * 16 + 4: return launch_mb<2, 4, 1>(p, a, stream);
    case 1 * 256 + 2 * 16 + 8: return launch_mb<2, 8, 1>(p, a, stream);
    case 1 * 256 + 4 * 16 + 2: return launch_mb<4, 2, 1>(p, a, stream);
    case 1 * 256 + 4 * 16 + 4: return launch_mb<4, 4, 1>(p, a, stream);
    case 1 * 256 + 4 * 16 + 8: return launch_mb<4, 8, 1>(p, a, stream);
    case 2 * 256 + 2 * 16 + 2: return launch_mb<2, 2, 2>(p, a, stream);
    case 2 * 256 + 2 * 16 + 4: return launch_mb<2, 4, 2>(p, a, stream);
    default: MEGA_FAIL("no instance for occ %d groups %ld unroll %ld", occ, (long)a.groups, (long)a.unroll);
  }
}

}  // extern "C"
}  // namespace dinfer
