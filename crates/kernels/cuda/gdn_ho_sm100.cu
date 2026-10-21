// gdn_ho_sm100：GDN chunk prefill 的 ho 段 tcgen05（UMMA）版 —— r11d 里程碑 2。
//
// 背景（docs/loops/qwen35-gdn-r11d.md「作废原因」）：把 wy 熔进 ho 是 2.5× FLOP，作废。
// 真正的缺口是利用率：ho 只有 mma.sync 峰值 575 TF/s 的 15%，wy 只有 12%。
// FlashInfer 293 TF/s 已超 48-SM 的 mma.sync 上限 187 → 它必然走 tcgen05 且 BV = 全 V。
//
// 本 kernel：**BV = V = 128**（h 的整个 V 维在一个 block，不切片、无重复计算），grid = (HV, seq)。
// 四个 GEMM 全走 tcgen05，**h 留在寄存器**（f32 主本），TMEM 只做 GEMM 累加器：
//   - h 若常驻 TMEM，每 chunk 的 `h = glast·h + ...` 要 ld→scale→st 一次 128 列，128 个 chunk 就吃掉整个预算
//   - 留寄存器后缩放免费；每 chunk 开头转 bf16 写 smem 当 B 操作数（本来就要写）
//   - BV=128 时 h 要 128 个 f32/线程，仍 < 255（mma.sync 版卡在 BV=64 是因为要 mma fragment 布局）
//
// 四个 GEMM（BT=64 token，K=V=128）：
//   (1) wh = w·hᵀ           A=w[BT,K]   B=hᵀ[K,V]       → 临时累加器
//   (2) o  = (q⊙dec)·hᵀ     A=q[BT,K]   B=hᵀ[K,V]       → t_o   （dec 折进 A，省一次 TMEM 往返）
//   (3) o += Aq·v'          A=Aq[BT,BT] B=v'[BT,V]      → t_o 累加
//   (4) h  = glast·h + v'ᵀ·kg  A=v'ᵀ[V,BT] B=kg[BT,K]   → 临时累加器，取回后在寄存器里组合
//
// (4) 的转置：v' 在 smem 里是 [BT, V]（V 连续）。当 A 用时 V 是 M、BT 是 K → MN-major 描述符
// （a_mn=true）当转置视图用，与 r12a 对 V 的做法相同。
//
// 只在 CUDA_ARCHS 含 100/103 时编入（build.rs 的 sm100_only 白名单）。
#include <cuda_pipeline.h>
#include "common.cuh"
#include <cstdio>
#include "umma.cuh"

namespace {

using dinfer::bf16;
using umma::smem_u32;

constexpr int kBT = 64;
constexpr int kK = 128;
constexpr int kV = 128;
constexpr int kM = 128;      // tcgen05 的 A 操作数 M 必须是 128（M=64 静默算错，见 knowledge/kernel-rules）
constexpr int kThreads = 128;
constexpr int kDepth = 4;

struct Smem {
  static constexpr int kblk = kK / 64;                  // 2 个 64 列的 K 块
  // 凡是要当 A 操作数的矩阵，M 方向都补到 kM=128（多出的行填 0）。
  static constexpr int off_h = 0;                       // h 的 bf16 影子 [V=128][64]×kblk（当 B）
  static constexpr int h_elems = kV * 64 * kblk;
  static constexpr int off_q = off_h + h_elems;         // q [kM][64]×kblk（当 A）
  static constexpr int am_elems = kM * 64 * kblk;
  static constexpr int off_w = off_q + am_elems;        // w [kM][64]×kblk（当 A）
  static constexpr int kg_elems = kBT * 64 * kblk;
  static constexpr int off_kg = off_w + am_elems;       // kg [BT][64]×kblk（当 B，MN=K，不用补）
  static constexpr int off_aq = off_kg + kg_elems;      // Aq [kM][kM]（当 A）
  static constexpr int aq_elems = kM * 64;
  static constexpr int off_vp = off_aq + aq_elems;      // v' 当 B：[V][token]，K 维 = token 连续
  static constexpr int vp_elems = kV * 64;
  static constexpr int off_vt = off_vp + vp_elems;      // v'ᵀ [V][kM]（当 A，K 维 = token）
  static constexpr int vt_elems = 0;  // 与 off_vp 共用（内容相同）
  static constexpr int off_kgt = off_vt + vt_elems;     // kgᵀ [K=128][token=64]（(4) 的 B，K 维 = token 连续）
  static constexpr int kgt_elems = kK * 64;
  static constexpr int bf16_elems = off_kgt + kgt_elems;
  static constexpr int bf16_bytes = bf16_elems * 2;
  static constexpr int off_tmem = (bf16_bytes + 127) & ~127;
  static constexpr int off_bar = off_tmem + 16;
  // h 的 f32 主本不放 smem（M=128 补零后 bf16 区已占 176 KB，227 KB 上限装不下 64 KB 的 f32）。
  // 改放全局 state：每 chunk 读一次、写一次（16 KB + 16 KB），在 6 TB/s 下可忽略。
  static constexpr int off_dec = off_bar + kDepth * 8;
  static constexpr int bytes = off_dec + kM * 4;
};

constexpr int kTmemH = 128;     // h 的 f32 主本（[V][K]，N=K=128 列）
constexpr int kTmemAcc = 128;   // 工作累加器（N=V=128 列）

__device__ __forceinline__ void stage_tile(bf16* dst, const bf16* src, int rows, int src_stride, int tid) {
  for (int i = tid; i < rows * 8; i += kThreads) {
    const int row = i >> 3, blk = i & 7;
    const int phys = row * 8 + (blk ^ (row & 7));
    *reinterpret_cast<uint4*>(dst + (size_t)phys * 8) =
        *reinterpret_cast<const uint4*>(src + (size_t)row * src_stride + blk * 8);
  }
}

// q 装载时先乘 dec（按 token 行），使 (2) 直接产出 (q·hᵀ)·dec，与 (3) 能在同一累加器上连续累加。
// cp.async 版：与 stage_tile 同样的 128 B swizzle 目标，但用异步拷贝（实测比 ld/st 快 6×）。
__device__ __forceinline__ void stage_tile_async(bf16* dst, const bf16* src, int rows, int src_stride, int tid) {
  for (int i = tid; i < rows * 8; i += kThreads) {
    const int row = i >> 3, blk = i & 7;
    const int phys = row * 8 + (blk ^ (row & 7));
    __pipeline_memcpy_async(dst + (size_t)phys * 8, src + (size_t)row * src_stride + blk * 8, 16);
  }
}

// 补零行只需在 chunk 循环**外**写一次（它们永远是 0，每 chunk 重写纯属浪费装载带宽）。
__device__ __forceinline__ void zero_pad_rows(bf16* dst, int rows, int rows_pad, int tid) {
  for (int i = tid; i < rows_pad * 8; i += kThreads) {
    const int row = i >> 3, blk = i & 7;
    if (row >= rows) {
      const int phys = row * 8 + (blk ^ (row & 7));
      *reinterpret_cast<uint4*>(dst + (size_t)phys * 8) = make_uint4(0, 0, 0, 0);
    }
  }
}

__device__ __forceinline__ void stage_q_dec(bf16* dst, const bf16* src, int rows, int src_stride, const float* dec,
                                            int tid) {
  for (int i = tid; i < rows * 8; i += kThreads) {
    const int row = i >> 3, blk = i & 7;
    const int phys = row * 8 + (blk ^ (row & 7));
    const float d = dec[row];
    const uint4 raw = *reinterpret_cast<const uint4*>(src + (size_t)row * src_stride + blk * 8);
    const __nv_bfloat162* r2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
    uint4 out;
    uint32_t* o32 = reinterpret_cast<uint32_t*>(&out);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const __nv_bfloat162 v2 = __floats2bfloat162_rn(dinfer::to_f32(r2[e].x) * d, dinfer::to_f32(r2[e].y) * d);
      o32[e] = *reinterpret_cast<const uint32_t*>(&v2);
    }
    *reinterpret_cast<uint4*>(dst + (size_t)phys * 8) = out;
  }
}

// 把 [rows,64] 的 bf16 行主序 tile 写成 128 B swizzle，并把 [rows, kM) 行补零。
__device__ __forceinline__ void stage_tile_pad(bf16* dst, const bf16* src, int rows, int src_stride, int tid) {
  const int nblk = kM * 8;
  for (int i = tid; i < nblk; i += kThreads) {
    const int row = i >> 3, blk = i & 7;
    const int phys = row * 8 + (blk ^ (row & 7));
    uint4 v = make_uint4(0, 0, 0, 0);
    if (row < rows) v = *reinterpret_cast<const uint4*>(src + (size_t)row * src_stride + blk * 8);
    *reinterpret_cast<uint4*>(dst + (size_t)phys * 8) = v;
  }
}

// 补零到 rows_pad 行（kM 或 kBT 都可用）。
__device__ __forceinline__ void stage_tile_pad_rows(bf16* dst, const bf16* src, int rows, int src_stride,
                                                    int rows_pad, int tid) {
  for (int i = tid; i < rows_pad * 8; i += kThreads) {
    const int row = i >> 3, blk = i & 7;
    const int phys = row * 8 + (blk ^ (row & 7));
    uint4 v = make_uint4(0, 0, 0, 0);
    if (row < rows) v = *reinterpret_cast<const uint4*>(src + (size_t)row * src_stride + blk * 8);
    *reinterpret_cast<uint4*>(dst + (size_t)phys * 8) = v;
  }
}

// 取回本线程所在 lane 的整行（warp w 覆盖 lane 32w..32w+31）。
// **`tcgen05.ld` 是异步的**：全部发完再等一次即可。原来每 4 列就 `tcgen05.wait::ld` 一次，
// 每 chunk 3 次取回 × 32 次等 = 96 次等待，是 61 µs/chunk 的主因（B300 实测）。
__device__ __forceinline__ void tmem_get_row(float* dst, uint32_t tmem_c, int ncols, int tid) {
  const int warp = tid >> 5;
  // x32：一次取 32 列，128 列 = 4 条指令，最后统一等一次。
  for (int c = 0; c < ncols; c += 32) {
    uint32_t v[32];
    umma::tmem_ld32_x32(umma::tmem_addr(tmem_c, warp, c), v);
#pragma unroll
    for (int e = 0; e < 32; ++e) dst[c + e] = __uint_as_float(v[e]);
  }
  umma::tmem_ld_wait();
}

// K-major 面板乘法：A=[rows_a, kK]、B=[rows_b, kK]，逐 64 列块累加。
__device__ __forceinline__ void mma_panel(uint32_t tmem_c, const bf16* a, int rows_a, const bf16* b, int rows_b,
                                          uint32_t idesc, bool accumulate_first, bool b_mn_major = false,
                                          uint32_t b_lbo = 0) {
  const uint64_t da0 = umma::k_major_128b_desc(smem_u32(a));
  const uint64_t db0 = b_mn_major ? umma::mn_major_128b_desc(smem_u32(b), b_lbo, 8 * 128)
                                  : umma::k_major_128b_desc(smem_u32(b));
  const uint32_t ablokk = (uint32_t)(rows_a * 64 * 2) >> 4;
  const uint32_t bblokk = (uint32_t)(rows_b * 64 * 2) >> 4;
#pragma unroll
  for (int kb = 0; kb < Smem::kblk; ++kb) {
    const uint64_t da = da0 + (uint32_t)kb * ablokk;
    const uint64_t db = db0 + (uint32_t)kb * bblokk;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const uint32_t ko = (uint32_t)(k * 16 * 2) >> 4;
      umma::mma_f16(tmem_c, da + ko, db + ko, idesc, accumulate_first || kb > 0 || k > 0);
    }
  }
}


__device__ __forceinline__ void chk(const void* p, const void* base, int smem_bytes, const char* tag, int a, int b) {
  const long off = (long)((const char*)p - (const char*)base);
  (void)tag; (void)a; (void)b; (void)off; (void)smem_bytes;
}
int set_smem(const void* fn, int bytes) {
  return dinfer::err(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

__global__ void __launch_bounds__(kThreads) gdn_ho_umma_kernel(
    const bf16* __restrict__ qn, const float* __restrict__ gc, const bf16* __restrict__ aq,
    const bf16* __restrict__ u, const bf16* __restrict__ w, const bf16* __restrict__ kg, bf16* __restrict__ out,
    float* __restrict__ state, const int* __restrict__ cu_seqlens, const int* __restrict__ cu_nt,
    const int* __restrict__ state_indices, int64_t state_stride_tok, int H, int HV, int V, int group) {
  extern __shared__ __align__(128) unsigned char smraw[];
  bf16* sm16 = reinterpret_cast<bf16*>(smraw);
  bf16* sh = sm16 + Smem::off_h;
  bf16* sq = sm16 + Smem::off_q;
  bf16* sw = sm16 + Smem::off_w;
  bf16* skgn = sm16 + Smem::off_kg;  // kg 自然布局中转区（[BT][64]×kblk，16 KB）
  bf16* sAq = sm16 + Smem::off_aq;
  bf16* svp = sm16 + Smem::off_vp;
  bf16* svt = svp;  // 与 svp 同一块（内容相同）
  bf16* skgt = sm16 + Smem::off_kgt;
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(smraw + Smem::off_tmem);
  uint64_t* bar = reinterpret_cast<uint64_t*>(smraw + Smem::off_bar);
  float* sdec = reinterpret_cast<float*>(smraw + Smem::off_dec);

  const int hv = blockIdx.x, seq = blockIdx.y;
  const int h = hv / group;
  const int tid = threadIdx.x, warp = tid >> 5;

  const int slot = state_indices ? state_indices[seq] : seq;
  const int bos = cu_seqlens[seq], eos = cu_seqlens[seq + 1];
  const int len_seq = eos - bos;
  if (len_seq <= 0 || slot < 0) return;
  const int nt = cu_nt[seq];

  if (tid == 0) {
    for (int i = 0; i < kDepth; ++i) umma::mbar_init(bar + i, 1);
    umma::mbar_init_fence();
  }
  __syncthreads();
  if (warp == 0) {
    umma::tmem_alloc(tmem_slot, kTmemH + kTmemAcc);
    umma::tmem_relinquish();
  }
  __syncthreads();
  const uint32_t t_h = *tmem_slot;          // h 的 f32 主本（列 0..127）
  const uint32_t t_acc = t_h + kTmemH;      // 工作累加器（列 128..255）

  // tcgen05 的 A 操作数 M 必须是 128（M=64 静默算错，见 knowledge/kernel-rules）。
  // 所以所有以 token 为 M 的矩阵都在 M 方向补零到 kM=128，结果只取前 len 行。
  const uint32_t idesc_bv = umma::make_instr_desc_bf16(kM, kV, false, false);
  // (3) Aq·v'：A=Aq[BT,BT] K-major，B=v'[BT,V] K-major（K 维 = token）
  // (4) v'ᵀ·kg：A=v'ᵀ → v' 的 MN-major 视图（M=V、K=BT）；B=kg[BT,K] K-major
  const uint32_t idesc_vk = umma::make_instr_desc_bf16(kV, kK, false, false);  // (4) 的 M 是 V=128，天然合规

  const long vstride = (long)HV * V;   // 按 hv 分段的行跨度
  const long qstride = (long)H * kK;   // qn 按 h 分段的行跨度

  // h 的 f32 主本放**全局 state**：M=128 补零后 smem bf16 区已 176 KB，装不下 64 KB 的 f32。
  // 每 chunk 读一次写一次（各 16 KB），在 6 TB/s 下 ~0.05 µs。
  // 128 线程各负责 V=128 的一行（myrow），只做 bf16 影子。
  const int myrow = warp * 32 + (tid & 31);
  float* hslot = state + (long)slot * state_stride_tok + (long)hv * V * kK;
  // segment 计时用
  float hbuf[kK];
  // 初值：全局 state -> 寄存器 -> TMEM + smem 影子
  auto push_h = [&](const float* src) {
    // 本线程负责 lane = myrow（V=128 一行），把 h 的整行写进 TMEM 的 lane myrow
    uint32_t packed[32];
    for (int c = 0; c < kK; c += 32) {
#pragma unroll
      for (int e = 0; e < 32; ++e) packed[e] = __float_as_uint(src[c + e]);
      umma::tmem_st32_x32(umma::tmem_addr(t_h, warp, c), packed);
    }
    umma::tmem_st_wait();
  };
  auto pull_h = [&](float* dst) {
    for (int c = 0; c < kK; c += 32) {
      uint32_t v[32];
      umma::tmem_ld32_x32(umma::tmem_addr(t_h, warp, c), v);
#pragma unroll
      for (int e = 0; e < 32; ++e) dst[c + e] = __uint_as_float(v[e]);
    }
    umma::tmem_ld_wait();
  };
  auto stage_h_from = [&](const float* src) {
    for (int kk = 0; kk < kK; ++kk) {
      const int kb = kk >> 6, kc = kk & 63;
      sh[kb * (kV * 64) + (size_t)(myrow * 8 + ((kc >> 3) ^ (myrow & 7))) * 8 + (kc & 7)] =
          dinfer::to_bf16(src[kk]);
    }
  };
  for (int kk = 0; kk < kK; ++kk) hbuf[kk] = hslot[(size_t)myrow * kK + kk];
  push_h(hbuf);
  stage_h_from(hbuf);
  __syncthreads();

  // 只留 hrow（f32 主本，128 个）+ tmp（四个 GEMM 的取回缓冲复用同一个，128 个）——
  // 四个独立数组会让寄存器爆到 255 + spill（实测 STACK 136）。
  float tmp[kK];
  // 补零行只在循环外写一次：q / w / Aq 的 M 方向 [len, kM) 永远是 0。
  for (int kb = 0; kb < Smem::kblk; ++kb) {
    zero_pad_rows(sq + kb * (kM * 64), kBT, kM, tid);
    zero_pad_rows(sw + kb * (kM * 64), kBT, kM, tid);
  }
  zero_pad_rows(sAq, kBT, kM, tid);
  // kgT 只有 token 行 < len 才有写；末块的 [len, kBT) 列要先清一次（B 的 K 维末尾补零）。
  for (int i = tid; i < kK * kBT; i += kThreads) skgt[i] = dinfer::to_bf16(0.f);
  __syncthreads();
  // mbarrier 的 parity phase：每完成一次翻转。原代码三处都固定传 0，
  // chunk 1 起等的是尚未发生的到达 -> 读到上一轮/半成品数据（表现为每次运行结果都不同）。
  uint32_t ph0 = 0, ph1 = 0, ph3 = 0;
  long long t_load=0, t_g1=0, t_vp=0, t_g23=0, t_o=0, t_g4=0, t_hu=0;
  long long t_issue=0, t_wait=0, t_sync=0, t_ktr=0, t_part[4]={0,0,0,0};
  for (int ci = 0; ci < nt; ++ci) {
    const int len = min(kBT, len_seq - ci * kBT);
    const long row0 = bos + (long)ci * kBT;
    const long ubase = (row0 * vstride) + (long)hv * V;

    const long long _t0 = (tid == 0) ? clock64() : 0;
    long long _tl1 = _t0, _tl2 = _t0, _tl3 = _t0;
    long long _tk[4] = {0, 0, 0, 0};
    // 装载 q（按 h 段）、w、kgᵀ 全部走 cp.async 同一组；Aq 也进这一组，一起等一次。
    const bf16* aqc = reinterpret_cast<const bf16*>(aq) + ((long)ci * HV + hv) * kBT * kBT;
    for (int kb = 0; kb < Smem::kblk; ++kb) {
      const int k0 = kb * 64;
      const long long _q0 = (tid == 0) ? clock64() : 0;
      stage_tile_async(sq + kb * (kM * 64), qn + row0 * qstride + (long)h * kK + k0, len, (int)qstride, tid);
      if (tid == 0) _tk[0] += clock64() - _q0;
      const long long _w0 = (tid == 0) ? clock64() : 0;
      stage_tile_async(sw + kb * (kM * 64), w + ubase + k0, len, (int)vstride, tid);
      if (tid == 0) _tk[1] += clock64() - _w0;
      // kg 只以 kgᵀ（(4) 的 B）形式被用到，[BT][K] 原样那份没人读，不再装载。
      // 源 kg[token][K] 的 8 个连续 bf16 沿 K，而 kgT 里连续的 8 个沿 token —— 是转置关系，
      // 只能"全域合并读 + 8 次 smem 散射写"（smem 写便宜，全局读才是贵的那头）。
      const long long _k0t = (tid == 0) ? clock64() : 0;
      // kg 只以 kgᵀ（(4) 的 B）形式被用到：[BT][K] 原样那份没人读。
      // 但**不要**在装载路径里做转置写 —— 8 次散射 smem 写实测 558 us/chunk，
      // 是 q/w 的 15 倍。这里跟 w 一样 cp.async 装成自然布局到 off_kg 中转区，
      // 等 wait 之后再做块转置（8×8，每线程一个整块，8 读 + 8 写）。
      stage_tile_async(skgn + kb * (kBT * 64), kg + ubase + k0, len, (int)vstride, tid);
      if (tid == 0) _tk[2] += clock64() - _k0t;
    }
    const long long _a0 = (tid == 0) ? clock64() : 0;
    stage_tile_async(sAq, aqc, len, kBT, tid);
    if (tid == 0) { _tk[3] += clock64() - _a0; _tl1 = clock64(); }
    __pipeline_commit();
    if (tid < kBT) sdec[tid] = (tid < len) ? exp2f(gc[(row0 + tid) * HV + hv]) : 0.f;
    __pipeline_wait_prior(0);
    if (tid == 0) _tl2 = clock64();
    __syncthreads();
    if (tid == 0) _tl3 = clock64();
    // kgᵀ：源 kg[token][K] 的 8 个连续 bf16 沿 K，而 kgT 里连续的 8 个沿 token。
    // 直接散射写要 8 次 2 B 写（558 us/chunk）。改成**寄存器内 8×8 块转置**：
    // 每个线程持有一个完整 8 token × 8 K 的块（8 次 uint4 读 + 8 次 uint4 写），
    // kgT 的排布保证同一块转置后在 smem 里也是 8 段连续的 16 B（见 off_kgt 布局）。
    {
      const long long _kt0 = (tid == 0) ? clock64() : 0;
      for (int kb2 = 0; kb2 < Smem::kblk; ++kb2) {
        for (int i = tid; i < kBT; i += kThreads) {
          const int tg = i >> 3, cg = i & 7;
          if (tg * 8 >= len) continue;
          uint4 r[8];
#pragma unroll
          for (int q2 = 0; q2 < 8; ++q2) {
            const int row = tg * 8 + q2;
            const int phys = row * 8 + (cg ^ (row & 7));
            r[q2] = *reinterpret_cast<const uint4*>(skgn + kb2 * (kBT * 64) + (size_t)phys * 8);
          }
          // 转置：w2[q2] 的第 k 个 bf16 = 原 (row tg*8+q2) 的第 (cg*8+k) 个
#pragma unroll
          for (int k8 = 0; k8 < 8; ++k8) {
            uint4 o2;
            uint32_t* o32 = reinterpret_cast<uint32_t*>(&o2);
#pragma unroll
            for (int q4 = 0; q4 < 4; ++q4) {
              const uint32_t lo = reinterpret_cast<const uint32_t*>(&r[q4 * 2 + 0])[k8 >> 1];
              const uint32_t hi = reinterpret_cast<const uint32_t*>(&r[q4 * 2 + 1])[k8 >> 1];
              const int sh = (k8 & 1) * 16;
              o32[q4] = ((lo >> sh) & 0xffffu) | (((hi >> sh) & 0xffffu) << 16);
            }
            const int kk = kb2 * 64 + cg * 8 + k8;
            const int dphys = kk * 8 + (tg ^ (kk & 7));
            // 只写 kgT 的 token 列块：目标行 kk、列 tg*8..tg*8+7
            *reinterpret_cast<uint4*>(skgt + (size_t)dphys * 8) = o2;
          }
        }
      }
      if (tid == 0) t_ktr += clock64() - _kt0;
    }
    // dec 折进 q，但**在 smem 里原地乘**：cp.async 装载（快）+ smem 乘法（快），
    // 避免原来"全局读 + 逐元素浮点乘 + 逐元素写"的同步路径（实测 4.63 µs/chunk）。
    for (int kb2 = 0; kb2 < Smem::kblk; ++kb2) {
      for (int i = tid; i < len * 8; i += kThreads) {
        const int row = i >> 3, blk = i & 7;
        const int phys = row * 8 + (blk ^ (row & 7));
        bf16* p8 = sq + kb2 * (kM * 64) + (size_t)phys * 8;
        uint4 raw = *reinterpret_cast<const uint4*>(p8);
        const __nv_bfloat162* r2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
        uint4 out;
        uint32_t* o32 = reinterpret_cast<uint32_t*>(&out);
        const float d = sdec[row];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const __nv_bfloat162 v2 = __floats2bfloat162_rn(dinfer::to_f32(r2[e].x) * d, dinfer::to_f32(r2[e].y) * d);
          o32[e] = *reinterpret_cast<const uint32_t*>(&v2);
        }
        *reinterpret_cast<uint4*>(p8) = out;
      }
    }
    __syncthreads();
    umma::fence_proxy_async_shared();
    const long long _t1 = (tid == 0) ? clock64() : 0;

    // ---- (1) wh = w·hᵀ → 取回 → v' = u − wh
    if (warp == 0 && umma::elect_one()) {
      mma_panel(t_acc, sw, kM, sh, kV, idesc_bv, false);
      umma::mma_commit(bar);
    }
    umma::mbar_wait(bar, ph0);
    ph0 ^= 1;
    umma::fence_after_sync();
    __syncthreads();
    const long long _t2 = (tid == 0) ? clock64() : 0;
    tmem_get_row(tmp, t_acc, kV, tid);
    // 两份布局都写（各 16 KB，代价 ~0.1 µs），彻底绕开 MN-major 的不确定性：
    //   svp [BT][V]：(3) 里当 B，K 维 = token → K-major
    //   svt [V][BT]：(4) 里当 A，K 维 = token → K-major（就是 v'ᵀ 的显式转置）
    // v' 的 token 行 [0, len) 有效，[len, kBT) 必须置 0（否则 (3)(4) 读到上一 chunk 的残留）。
    // **token 维只有 kBT=64 列**：myrow 覆盖 0..127（V 的行），>63 的线程不能写（会越出 64 列）。
    // wh 的整行只在 lane < kBT 的线程手里（tmp[128]），所以这一段的并行度受限于 64 个线程。
    // 想再快需要把 wh 先落 smem 再全线程分摊 —— 但 store 量翻倍，实测不划算，保持现状。
    if (myrow < kBT) {
      const bf16* urow = (myrow < len) ? (u + (row0 + myrow) * vstride + (long)hv * V) : nullptr;
      // 一次处理 8 个 v：uint4 读 u（全局，8 个 bf16 = 16 B 合并访存）+ 8 个 swizzle 写。
      // 原来 128 次标量读（每次 2 B、不合并）是 vp 段 11.8 µs/chunk 的根因。
      for (int c8 = 0; c8 < kV; c8 += 8) {
        uint4 raw = urow ? *reinterpret_cast<const uint4*>(urow + c8) : make_uint4(0, 0, 0, 0);
        const __nv_bfloat162* r2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int kc = c8 + 2 * e;
          const float v0 = dinfer::to_f32(r2[e].x) - tmp[kc];
          const float v1 = dinfer::to_f32(r2[e].y) - tmp[kc + 1];
          // 同一 token 的连续元素在 swizzle 后落在同一 16 B 块行内不连续 -> 逐个写
          svp[(size_t)(kc * 8 + (((myrow >> 3) ^ (kc & 7)))) * 8 + (myrow & 7)] = dinfer::to_bf16(v0);
          svp[(size_t)((kc + 1) * 8 + (((myrow >> 3) ^ ((kc + 1) & 7)))) * 8 + (myrow & 7)] =
              dinfer::to_bf16(v1);
        }
      }
    }
    __syncthreads();
    umma::fence_proxy_async_shared();
    const long long _t3 = (tid == 0) ? clock64() : 0;

    // ---- (2)+(3) o = (q⊙dec)·hᵀ + Aq·v'：两条 MMA 连续累加到同一累加器，中间不取回。
    if (warp == 0 && umma::elect_one()) {
      mma_panel(t_acc, sq, kM, sh, kV, idesc_bv, /*accumulate_first=*/false);
      // A=Aq[BT,BT]、B=v'[BT,V]：K 维 = BT = 64，一个 K 块。
      const uint64_t daq = umma::k_major_128b_desc(smem_u32(sAq));  // A = Aq[128, 64]
      // B = v'[BT][V]，K 维 = token → K-major。A=Aq[BT,BT]、B=[BT,V]，K 维都是 token=64。
      const uint64_t dvp = umma::k_major_128b_desc(smem_u32(svp));  // B = v'[V][token]
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const uint32_t ko = (uint32_t)(k * 16 * 2) >> 4;
        umma::mma_f16(t_acc, daq + ko, dvp + ko, idesc_bv, true);
      }
      umma::mma_commit(bar + 1);
    }
    umma::mbar_wait(bar + 1, ph1);
    ph1 ^= 1;
    umma::fence_after_sync();
    __syncthreads();
    const long long _t4 = (tid == 0) ? clock64() : 0;
    tmem_get_row(tmp, t_acc, kV, tid);
    if (myrow < len) {
      bf16* orow = out + (row0 + myrow) * vstride + (long)hv * V;
      // 向量写：8 个 bf16 = 16 B 一次。标量写 128 次是 30% 耗时的主因（每个都吃一次 store 延迟）。
      for (int kk = 0; kk < kV; kk += 8) {
        uint4 packed;
        uint32_t* w32 = reinterpret_cast<uint32_t*>(&packed);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const __nv_bfloat162 v2 = __floats2bfloat162_rn(tmp[kk + 2 * e], tmp[kk + 2 * e + 1]);
          w32[e] = *reinterpret_cast<const uint32_t*>(&v2);
        }
        *reinterpret_cast<uint4*>(orow + kk) = packed;
      }
    }
    __syncthreads();
    const long long _t5 = (tid == 0) ? clock64() : 0;

    // ---- (4) h = glast·h + v'ᵀ·kg
    // A = v'ᵀ（v' 的 MN-major 视图：M=V、K=BT），B = kg[BT, K] K-major。
    if (warp == 0 && umma::elect_one()) {
      // v' 的 MN-major：atom = 8 token × 128 B；LBO = MN 方向下一块（V 方向 64 个 v 一块）；
      // SBO = K 方向下一个 atom（8 token）= 1024 B。
      // A = v'ᵀ 的显式转置版 svt[V][BT]：K 维 = token 连续 → K-major，与 (1)(2) 同形。
      const uint64_t da = umma::k_major_128b_desc(smem_u32(svt));
      // B = kgᵀ[K][token]，K 维 = token 连续 → K-major，与 A 的 K 维对齐。
      const uint64_t db0 = umma::k_major_128b_desc(smem_u32(skgt));
#pragma unroll
      // B = kgT[K=128][token=64] 是一个连续块（N=K=128 行、K=token=64 列），**不分块**。
      // （原来按 64 列分两块是给 [BT][64]×kblk 布局用的，kgT 改成单块后那会写到数组外。）
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const uint32_t ko = (uint32_t)(k * 16 * 2) >> 4;  // A：16 个 token
        umma::mma_f16(t_acc, da + ko, db0 + ko, idesc_vk, k > 0);
      }
      umma::mma_commit(bar + 3);
    }
    umma::mbar_wait(bar + 3, ph3);
    ph3 ^= 1;
    umma::fence_after_sync();
    __syncthreads();
    const long long _t6 = (tid == 0) ? clock64() : 0;
    tmem_get_row(tmp, t_acc, kK, tid);
    const float glast = sdec[len - 1];
    // h 主本在 TMEM：[V][K] f32。缩放 + 累加全在寄存器里做，再一次性写回 TMEM 和 bf16 影子。
    pull_h(hbuf);
    for (int kk = 0; kk < kK; ++kk) hbuf[kk] = hbuf[kk] * glast + tmp[kk];
    push_h(hbuf);
    __syncthreads();
    stage_h_from(hbuf);
    __syncthreads();
    umma::fence_proxy_async_shared();
    if (tid == 0) {
      const long long _t8 = clock64();
      t_issue += _tl1-_t0; t_wait += _tl2-_tl1; t_sync += _tl3-_tl2;
      for (int z = 0; z < 4; ++z) t_part[z] += _tk[z];
      t_load += _t1-_t0; t_g1 += _t2-_t1; t_vp += _t3-_t2; t_g23 += _t4-_t3;
      t_o += _t5-_t4; t_g4 += _t6-_t5; t_hu += _t8-_t6;
    }
  }
  if (tid == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    const double tot = (double)(t_load+t_g1+t_vp+t_g23+t_o+t_g4+t_hu);
    printf("[prof] load %.0f%% g1 %.0f%% vp %.0f%% g23 %.0f%% o %.0f%% g4 %.0f%% hupd %.0f%% (%.2f us)\n",
      t_load*100/tot, t_g1*100/tot, t_vp*100/tot, t_g23*100/tot, t_o*100/tot, t_g4*100/tot, t_hu*100/tot, tot/1900.0);
    printf("[ld] issue %.2f wait %.2f sync %.2f | q %.2f w %.2f kgload %.2f aq %.2f kgtr %.2f us/chunk\n",
      t_issue/1900.0, t_wait/1900.0, t_sync/1900.0,
      t_part[0]/1900.0, t_part[1]/1900.0, t_part[2]/1900.0, t_part[3]/1900.0, t_ktr/1900.0);
  }

  // 末态回写
  // 末态：TMEM -> 全局 state（整条序列只写一次）
  pull_h(hbuf);
  for (int kk = 0; kk < kK; ++kk) hslot[(size_t)myrow * kK + kk] = hbuf[kk];
  umma::fence_before_sync();
  __syncthreads();
  if (warp == 0) umma::tmem_dealloc(t_h, kTmemH + kTmemAcc);
}

}  // namespace

extern "C" int gdn_ho_umma(const void* qn, const float* gc, const void* aq, const void* u, const void* w, const void* kg,
                void* out, void* state, const int* cu_seqlens, const int* cu_nt, const int* state_indices,
                int64_t state_stride_tok, int64_t num_seqs, int64_t H, int64_t HV, int64_t K, int64_t V,
                cudaStream_t stream) {
  if (num_seqs <= 0 || H <= 0 || HV <= 0 || HV % H != 0 || K != kK || V != kV)
    return dinfer::err(cudaErrorInvalidValue);
  const int group = (int)(HV / H);
  dim3 grid((unsigned)HV, (unsigned)num_seqs);
  if (const int rc = set_smem(reinterpret_cast<const void*>(&gdn_ho_umma_kernel), Smem::bytes)) return rc;
  gdn_ho_umma_kernel<<<grid, kThreads, Smem::bytes, stream>>>(
      static_cast<const bf16*>(qn), static_cast<const float*>(gc), static_cast<const bf16*>(aq),
      static_cast<const bf16*>(u), static_cast<const bf16*>(w), static_cast<const bf16*>(kg),
      static_cast<bf16*>(out), static_cast<float*>(state), cu_seqlens, cu_nt, state_indices, state_stride_tok,
      (int)H, (int)HV, (int)V, group);
  return dinfer::last_err();
}
