// Qwen3.5 GDN 层 decode 的**单 persistent kernel**（质变实验第一阶段，docs/loops/mega-sm100.md、design.md §14.3）。
//
// 把现路径的 11 个 kernel（rmsnorm → in_proj GEMV → conv1d_update → GDN 递归(fused gate) → gated_rmsnorm → out_proj
// → add+rmsnorm → gate_up GEMV → silu·mul → down GEMV → add）编成一张静态任务表，一次 launch、一个常驻 grid 跑完：
//   - grid = occupancy × SM（host 查），block = kWarps 个 warp；任务 id 全局静态编号 = (阶段, 分片)，CTA 按
//     task = blockIdx.x + i·gridDim.x 顺序领取（阶段内等代价，静态轮转即可；id 单调保证无环等待）；
//   - 依赖 = 全局 ready[stage] 计数器：生产者 `__syncthreads → threadfence → red.release.gpu +1`，
//     消费者 thread 0 `ld.acquire.gpu` 自旋到 ready[stage-1] == count[stage-1] 再 `__syncthreads`；
//   - GEMV 阶段：权重按 N 切 tile 给任务（tile = kWarps·G·8 行），K 维再按 split-K 切（任务 = tile × split），
//     权重 tile 用 cp.async.cg 16 B 多级流水进 smem（**依赖等待前就发**，把"等上一阶段"藏在权重流里）；
//     激活 A（M 行 × K）同样 cp.async 进 smem；算子是 mma.sync.m16n8k16 bf16（A/B 都按"每 lane 16 B 连续 K"
//     的置换布局装载，A、B 同置换所以乘积不变）；split-K 用 f32 partial + 每 tile 信号量，最后到的任务归约写 bf16；
//   - GDN 递归阶段：复用 gdn_decode.cuh 的寄存器方案（任务 = (seq, hv, V 块)，每 warp ROWS 行状态）；
//   - 阶段间激活走全局（L2 常驻），本 kernel 内"先读后被别的 SM 改写再读"的缓冲（res、partial）一律 __ldcg 绕 L1。
//
// 所有分片大小 / 流水深度 / CTA 数 / split 数都由 host（`ArchProfile::mega_params` + 启动自调优）给，kernel 内只有
// 结构常量（warp 数、mma 形状）。按多卡型分文件规则本文件只给 Blackwell 编（build.rs 按 CUDA_ARCHS 选），
// 体内不写 `#if __CUDA_ARCH__`；第一版只用 sm80+ 指令（cp.async.cg / mma.sync / ld.acquire），TMA 留第二版。
//
// 规格 / 记录：docs/loops/mega-sm100.md。对拍与计时：crates/kernels/tests/mega_parity.rs。
#include <cstdint>

#include "common.cuh"
#include "gdn_decode.cuh"
#include "kernels.h"

namespace dinfer {
namespace {

constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kChunkK = 32;  // 一个 K 块 = 两条 m16n8k16 = 每 lane 16 B（8 个 bf16）
constexpr int kStageCount = 11;
constexpr int kGdnRows = 16;  // 每 warp 的状态行数：16 行 × 16 B/lane 一次发 16 个向量读（8 KB/warp 在飞），任务 = (seq, hv) 整个 V=128
// 第 1 轮用 4 行（BV=32，与独立 kernel 同）：B=32 6144 个小任务在 148 个常驻 CTA 上串行 41 波、每波一次状态读延迟，409 µs。
constexpr int kGdnBv = kWarps * kGdnRows;
constexpr int kMaxPipe = 4;

enum Stage : int {
  S_NORM0 = 0,
  S_INPROJ,
  S_CONV,
  S_GDN,
  S_GNORM,
  S_OUTPROJ,
  S_ADDNORM,
  S_GATEUP,
  S_SILU,
  S_DOWN,
  S_ADD,
};

// 一个 GEMV 阶段：out[M,N] = A[M,K] · W[N,K]^T，bf16 进 bf16 出（f32 累加）。
struct GemvDesc {
  const bf16* w;
  const bf16* a;
  bf16* out;
  float* partial;  // [splits][M][N] f32（splits > 1）
  int* sem;        // [tiles]
  int n, k, tiles, splits, spl, total;  // spl = 每 split 的流水 stage 数，total = K / (kChunkK·U)
};

struct Params {
  MegaLayerArgs a;
  GemvDesc gemv[4];  // in_proj / out_proj / gate_up / down
  int count[kStageCount];
  int start[kStageCount + 1];
  int conv_dim, in_row, z_off, a_off, b_off, group;
  int pipe;  // 流水深度（运行时，≤ kMaxPipe）
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

// 等上一阶段全部完成（stage 0 无依赖）。只 thread 0 自旋，其余在 barrier 等。
__device__ __forceinline__ void wait_stage(const Params& p, int stage) {
  if (stage == 0) {
    if (p.a.prof && threadIdx.x == 0) s_wait_end = gtimer();
    return;
  }
  if (threadIdx.x == 0) {
    const int need = p.count[stage - 1];
    const int* flag = p.ready + (stage - 1);
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
    default: asm volatile("cp.async.wait_group 2;\n" ::); break;
  }
}

__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t a0, const uint32_t a1, const uint32_t a2,
                                         const uint32_t a3, const uint32_t b0, const uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// smem tile 行内 16 B 块的 swizzle：行距 4U 块（U ∈ {4,8} → 64 / 128 B），mma 片段读是 8 行 × 4 块一个 wavefront，
// 奇数行把块号的 bit2 翻转，8 行落到 8 个不同 bank 组。cp.async 写也走同一映射，写同样无冲突。
__device__ __forceinline__ int swz(int row, int chunk) { return chunk ^ ((row & 1) << 2); }

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
  const int n0 = tile * TN;
  const int s_begin = min(split * g.spl, g.total);
  const int s_end = min(s_begin + g.spl, g.total);
  const int nst = s_end - s_begin;
  const int kbase = s_begin * KS;
  const int pipe = p.pipe;
  uint8_t* wbuf = smem;
  uint8_t* abuf = smem + pipe * W_BYTES;

  auto issue_w = [&](int st, int buf) {
    const int koff = kbase + st * KS;
    uint8_t* dst = wbuf + buf * W_BYTES;
#pragma unroll
    for (int i = 0; i < W_PER_T; ++i) {
      const int idx = tid + i * kThreads;
      const int row = idx / CPR, chunk = idx - row * CPR;
      const int nrow = n0 + row;
      const bool ok = nrow < g.n;
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

  // 尾声：acc[gi][mb] 的 c0,c1 = (行 mb·16 + r8, 列 2·t4 + {0,1})，c2,c3 = 行 +8。
  if (g.splits == 1) {
#pragma unroll
    for (int gi = 0; gi < G; ++gi) {
      const int ncol = n0 + (warp * G + gi) * 8 + 2 * t4;
      if (ncol >= g.n) continue;
#pragma unroll
      for (int mb = 0; mb < MB; ++mb) {
        const int rlo = mb * 16 + r8, rhi = rlo + 8;
        if (rlo < m)
          *reinterpret_cast<__nv_bfloat162*>(g.out + (long)rlo * g.n + ncol) =
              __floats2bfloat162_rn(acc[gi][mb][0], acc[gi][mb][1]);
        if (rhi < m)
          *reinterpret_cast<__nv_bfloat162*>(g.out + (long)rhi * g.n + ncol) =
              __floats2bfloat162_rn(acc[gi][mb][2], acc[gi][mb][3]);
      }
    }
    return;
  }
  float* part = g.partial + (long)split * m * g.n;
#pragma unroll
  for (int gi = 0; gi < G; ++gi) {
    const int ncol = n0 + (warp * G + gi) * 8 + 2 * t4;
    if (ncol >= g.n) continue;
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
    *reinterpret_cast<__nv_bfloat162*>(g.out + off) = __floats2bfloat162_rn(s.x, s.y);
  }
  if (tid == 0) g.sem[tile] = 0;  // 自复位（下次 launch 前不需要 memset）
}

// ---------------------------------------------------------------- rmsnorm（一行 / 任务；fused：res += x 后归一化 res）
// 口径与 norm.cu rmsnorm_vec_kernel 一致：fused 时先 bf16 落残差，再用落点后的值算平方和与归一化；
// out = bf16(x · rstd · w)。hidden ≤ 16·2·kThreads。
constexpr int kNormMaxPairs = 16;
__device__ __forceinline__ void rmsnorm_row(const bf16* __restrict__ x, bf16* res, const bf16* __restrict__ w,
                                            bf16* __restrict__ out, int hidden, float eps) {
  const int tid = threadIdx.x;
  const int pairs = hidden / 2;
  const unsigned* xp = reinterpret_cast<const unsigned*>(x);
  unsigned* rp = reinterpret_cast<unsigned*>(res);
  float2 v[kNormMaxPairs];
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < kNormMaxPairs; ++j) {
    const int i = tid + j * kThreads;
    v[j] = make_float2(0.f, 0.f);
    if (i < pairs) {
      unsigned xa = __ldcg(xp + i);
      float2 f = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&xa));
      if (res) {
        unsigned ra = __ldcg(rp + i);
        const float2 r = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&ra));
        f.x += r.x;
        f.y += r.y;
        const __nv_bfloat162 rb = __floats2bfloat162_rn(f.x, f.y);
        rp[i] = *reinterpret_cast<const unsigned*>(&rb);
        f = __bfloat1622float2(rb);
      }
      v[j] = f;
      ss += f.x * f.x + f.y * f.y;
    }
  }
  ss = block_reduce_sum<kThreads>(ss);
  const float inv = rsqrtf(ss / hidden + eps);
  const unsigned* wp = reinterpret_cast<const unsigned*>(w);
  unsigned* op = reinterpret_cast<unsigned*>(out);
#pragma unroll
  for (int j = 0; j < kNormMaxPairs; ++j) {
    const int i = tid + j * kThreads;
    if (i < pairs) {
      unsigned wa = wp[i];
      const float2 g = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&wa));
      const __nv_bfloat162 o = __floats2bfloat162_rn(v[j].x * inv * g.x, v[j].y * inv * g.y);
      op[i] = *reinterpret_cast<const unsigned*>(&o);
    }
  }
}

// ---------------------------------------------------------------- conv1d update（任务 = (seq, 8·kThreads 维一段)）
constexpr int kConvDimsPerThread = 8;
constexpr int kConvChunk = kThreads * kConvDimsPerThread;
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
#pragma unroll
  for (int e = 0; e < kConvDimsPerThread; ++e) {
    const int d = d0 + e;
    if (d >= p.conv_dim) break;
    bf16* st = st_base + (long)d * sl;
    float acc = 0.f;
    for (int j = 0; j < sl; ++j) acc += to_f32(w[d * width + j]) * to_f32(st[j]);
    const float xt = to_f32(x[d]);
    acc += to_f32(w[d * width + sl]) * xt;
    out[d] = to_bf16(act_f(acc, 0));
    for (int j = 0; j + 1 < sl; ++j) st[j] = st[j + 1];
    st[sl - 1] = to_bf16(xt);
  }
}

// ---------------------------------------------------------------- GDN 递归（任务 = (seq, hv, V 块 kGdnBv)）
__device__ __forceinline__ void gdn_task(const Params& p, int shard) {
  constexpr int KPL = 4;  // K = 128
  const int vblocks = (int)p.a.v_dim / kGdnBv;
  const int hv = (shard / vblocks) % (int)p.a.heads_v;
  const int seq = shard / (vblocks * (int)p.a.heads_v);
  const int vb = shard % vblocks;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
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
  a.HV = (int)p.a.heads_v;
  a.K = (int)p.a.k_dim;
  a.V = (int)p.a.v_dim;
  a.group = p.group;
  a.scale = p.a.scale;
  a.state_stride_tok = (long)p.a.state_stride_tok;
  a.ab_stride = (long)p.in_row;
  const int v0 = vb * kGdnBv + warp * kGdnRows;
  float* Sg = a.state + (long)slot * a.state_stride_tok + (long)hv * a.V * a.K + (long)v0 * a.K + lane * KPL;
  float r[kGdnRows][KPL];
#pragma unroll
  for (int i = 0; i < kGdnRows; ++i) vec_load<KPL>(r[i], Sg + (long)i * a.K);
  decode_tile_body<kGdnRows, KPL, true, true>(a, r, seq, hv, v0, /*bos=*/seq, /*len=*/1, lane);
#pragma unroll
  for (int i = 0; i < kGdnRows; ++i) vec_store<KPL>(Sg + (long)i * a.K, r[i]);
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

// ---------------------------------------------------------------- silu·mul（任务 = (b, 2·kThreads 元素一段)）
constexpr int kEwPairsPerThread = 4;
constexpr int kEwChunk = kThreads * kEwPairsPerThread * 2;
__device__ __forceinline__ void silu_task(const Params& p, int shard) {
  const int inter = (int)p.a.inter;
  const int chunks = (inter + kEwChunk - 1) / kEwChunk;
  const int b = shard / chunks, c = shard - b * chunks;
  const bf16* gu = (const bf16*)p.a.gu + (long)b * 2 * inter;
  bf16* act = (bf16*)p.a.act + (long)b * inter;
#pragma unroll
  for (int j = 0; j < kEwPairsPerThread; ++j) {
    const int i = c * kEwChunk + (threadIdx.x + j * kThreads) * 2;
    if (i + 1 < inter) {
      unsigned ga = __ldcg(reinterpret_cast<const unsigned*>(gu + i));
      unsigned ua = __ldcg(reinterpret_cast<const unsigned*>(gu + inter + i));
      const float2 g = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&ga));
      const float2 u = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&ua));
      *reinterpret_cast<__nv_bfloat162*>(act + i) = __floats2bfloat162_rn(silu(g.x) * u.x, silu(g.y) * u.y);
    }
  }
}

// ---------------------------------------------------------------- 残差加（任务 = (b, 一段)）：res += d
__device__ __forceinline__ void add_task(const Params& p, int shard) {
  const int hidden = (int)p.a.hidden;
  const int chunks = (hidden + kEwChunk - 1) / kEwChunk;
  const int b = shard / chunks, c = shard - b * chunks;
  const bf16* d = (const bf16*)p.a.d + (long)b * hidden;
  bf16* res = (bf16*)p.a.res + (long)b * hidden;
#pragma unroll
  for (int j = 0; j < kEwPairsPerThread; ++j) {
    const int i = c * kEwChunk + (threadIdx.x + j * kThreads) * 2;
    if (i + 1 < hidden) {
      unsigned da = __ldcg(reinterpret_cast<const unsigned*>(d + i));
      unsigned ra = __ldcg(reinterpret_cast<const unsigned*>(res + i));
      const float2 x = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&da));
      const float2 r = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&ra));
      *reinterpret_cast<__nv_bfloat162*>(res + i) = __floats2bfloat162_rn(r.x + x.x, r.y + x.y);
    }
  }
}

// ---------------------------------------------------------------- persistent kernel
template <int MB, int G, int U>
__global__ void __launch_bounds__(kThreads, 1) mega_layer_kernel(const Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
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
        rmsnorm_row((const bf16*)p.a.res + (long)shard * p.a.hidden, nullptr, (const bf16*)p.a.norm_w,
                    (bf16*)p.a.xn + (long)shard * p.a.hidden, (int)p.a.hidden, p.a.eps);
        break;
      case S_CONV: wait_stage(p, stage); conv_task(p, shard); break;
      case S_GDN: wait_stage(p, stage); gdn_task(p, shard); break;
      case S_GNORM: wait_stage(p, stage); gnorm_task(p, shard); break;
      case S_ADDNORM:
        wait_stage(p, stage);
        rmsnorm_row((const bf16*)p.a.h2 + (long)shard * p.a.hidden, (bf16*)p.a.res + (long)shard * p.a.hidden,
                    (const bf16*)p.a.post_norm_w, (bf16*)p.a.xn2 + (long)shard * p.a.hidden, (int)p.a.hidden, p.a.eps);
        break;
      case S_SILU: wait_stage(p, stage); silu_task(p, shard); break;
      case S_ADD: wait_stage(p, stage); add_task(p, shard); break;
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

template <int MB, int G, int U>
int launch(Params& p, const MegaLayerArgs& a, cudaStream_t stream) {
  constexpr int TN = kWarps * G * 8;
  constexpr int CPR = 4 * U;
  const int pipe = p.pipe;
  const size_t smem = (size_t)pipe * (TN * CPR * 16 + MB * 16 * CPR * 16);
  int dev = 0, sms = 1, max_smem = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  if ((int)smem > max_smem) return (int)cudaErrorInvalidValue;
  auto kern = mega_layer_kernel<MB, G, U>;
  static int smem_set = 0;
  if (smem_set < (int)smem) {
    if (const int rc = err(cudaFuncSetAttribute((const void*)kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)))
      return rc;
    smem_set = (int)smem;
  }
  int occ = 0;
  if (const int rc = err(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, kThreads, smem))) return rc;
  if (occ <= 0) return (int)cudaErrorInvalidValue;
  if (a.ctas_per_sm > 0 && occ > a.ctas_per_sm) occ = a.ctas_per_sm;
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
    if (g.tiles != ceil_div(g.n, TN)) return (int)cudaErrorInvalidValue;
    if (g.k % (kChunkK * U) != 0) return (int)cudaErrorInvalidValue;
    g.total = g.k / (kChunkK * U);
    if (g.splits < 1 || g.splits > g.total) return (int)cudaErrorInvalidValue;
    g.spl = ceil_div(g.total, g.splits);
  }
  kern<<<grid, kThreads, smem, stream>>>(p);
  return last_err();
}

template <int G, int U>
int launch_mb(Params& p, const MegaLayerArgs& a, cudaStream_t stream) {
  if (a.batch <= 16) return launch<1, G, U>(p, a, stream);
  return launch<2, G, U>(p, a, stream);
}

}  // namespace

extern "C" {

int mega_gdn_layer_decode_sm100(const MegaLayerArgs* ap, cudaStream_t stream) {
  if (!ap) return (int)cudaErrorInvalidValue;
  const MegaLayerArgs& a = *ap;
  if (a.batch <= 0) return 0;
  if (a.batch > 32 || a.k_dim != 128 || a.v_dim % kGdnBv != 0 || a.hidden % (2 * kThreads) != 0 ||
      a.hidden > 2 * kThreads * kNormMaxPairs || a.v_dim > 8 * 32 || a.width < 2 || a.heads_v % a.heads_k != 0 ||
      a.inter % 2 != 0)
    return (int)cudaErrorInvalidValue;
  if (a.stages < 2 || a.stages > kMaxPipe) return (int)cudaErrorInvalidValue;

  Params p{};
  p.a = a;
  p.pipe = (int)a.stages;
  p.conv_dim = (int)(2 * a.heads_k * a.k_dim + a.heads_v * a.v_dim);
  p.z_off = p.conv_dim;
  p.a_off = p.conv_dim + (int)(a.heads_v * a.v_dim);
  p.b_off = p.a_off + (int)a.heads_v;
  p.in_row = p.b_off + (int)a.heads_v;
  p.group = (int)(a.heads_v / a.heads_k);
  const int TN = kWarps * (int)a.groups * 8;
  const int B = (int)a.batch, hidden = (int)a.hidden, inter = (int)a.inter;
  const int sems_in = ceil_div(p.in_row, TN), sems_out = ceil_div(hidden, TN), sems_gu = ceil_div(2 * inter, TN),
            sems_down = ceil_div(hidden, TN);
  p.ready = a.counters;
  p.done = a.counters + kStageCount;
  int* sem = a.counters + kCountersFixed;
  p.gemv[0] = GemvDesc{(const bf16*)a.in_w, (const bf16*)a.xn, (bf16*)a.qkvzab, a.partial, sem, p.in_row, hidden, sems_in,
                       (int)a.splits_in, 0, 0};
  sem += sems_in;
  p.gemv[1] = GemvDesc{(const bf16*)a.out_w, (const bf16*)a.n, (bf16*)a.h2, a.partial, sem, hidden,
                       (int)(a.heads_v * a.v_dim), sems_out, (int)a.splits_out, 0, 0};
  sem += sems_out;
  p.gemv[2] = GemvDesc{(const bf16*)a.gate_up_w, (const bf16*)a.xn2, (bf16*)a.gu, a.partial, sem, 2 * inter, hidden,
                       sems_gu, (int)a.splits_gu, 0, 0};
  sem += sems_gu;
  p.gemv[3] = GemvDesc{(const bf16*)a.down_w, (const bf16*)a.act, (bf16*)a.d, a.partial, sem, hidden, inter, sems_down,
                       (int)a.splits_down, 0, 0};

  p.count[S_NORM0] = B;
  p.count[S_INPROJ] = sems_in * (int)a.splits_in;
  p.count[S_CONV] = B * ceil_div(p.conv_dim, kConvChunk);
  p.count[S_GDN] = B * (int)a.heads_v * (int)(a.v_dim / kGdnBv);
  p.count[S_GNORM] = ceil_div((long)B * a.heads_v, kWarps);
  p.count[S_OUTPROJ] = sems_out * (int)a.splits_out;
  p.count[S_ADDNORM] = B;
  p.count[S_GATEUP] = sems_gu * (int)a.splits_gu;
  p.count[S_SILU] = B * ceil_div(inter, kEwChunk);
  p.count[S_DOWN] = sems_down * (int)a.splits_down;
  p.count[S_ADD] = B * ceil_div(hidden, kEwChunk);
  p.start[0] = 0;
  for (int s = 0; s < kStageCount; ++s) p.start[s + 1] = p.start[s] + p.count[s];

  switch ((int)a.groups * 16 + (int)a.unroll) {
    case 2 * 16 + 4: return launch_mb<2, 4>(p, a, stream);
    case 2 * 16 + 8: return launch_mb<2, 8>(p, a, stream);
    case 4 * 16 + 4: return launch_mb<4, 4>(p, a, stream);
    case 4 * 16 + 8: return launch_mb<4, 8>(p, a, stream);
    default: return (int)cudaErrorInvalidValue;
  }
}

}  // extern "C"
}  // namespace dinfer
