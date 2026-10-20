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
#include "common.cuh"
#include "umma.cuh"

namespace {

using dinfer::bf16;
using umma::smem_u32;

constexpr int kBT = 64;
constexpr int kK = 128;
constexpr int kV = 128;
constexpr int kThreads = 128;
constexpr int kDepth = 4;

struct Smem {
  static constexpr int kblk = kK / 64;                  // 2 个 64 列的 K 块
  static constexpr int off_h = 0;                       // h 的 bf16 影子 [V][64]×kblk
  static constexpr int h_elems = kV * 64 * kblk;
  static constexpr int off_q = off_h + h_elems;         // q [BT][64]×kblk
  static constexpr int row_elems = kBT * 64 * kblk;
  static constexpr int off_w = off_q + row_elems;
  static constexpr int off_kg = off_w + row_elems;
  static constexpr int off_aq = off_kg + row_elems;     // Aq [BT][BT]
  static constexpr int aq_elems = kBT * 64;
  static constexpr int off_vp = off_aq + aq_elems;      // v' [BT][V]，V 连续
  static constexpr int vp_elems = kBT * 64 * kblk;
  static constexpr int bf16_elems = off_vp + vp_elems;
  static constexpr int bf16_bytes = bf16_elems * 2;
  static constexpr int off_tmem = (bf16_bytes + 127) & ~127;
  static constexpr int off_bar = off_tmem + 16;
  static constexpr int off_hf32 = off_bar + kDepth * 8;        // h 的 f32 主本 [V][K] = 64 KB
  static constexpr int hf32_bytes = kV * kK * 4;
  static constexpr int off_dec = off_hf32 + hf32_bytes;
  static constexpr int bytes = off_dec + kBT * 4;
};

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

__device__ __forceinline__ void tmem_get_row(float* dst, uint32_t tmem_c, int ncols, int tid) {
  const int warp = tid >> 5;
  for (int c = 0; c < ncols; c += 4) {
    uint32_t v[4];
    umma::tmem_ld32_x4(umma::tmem_addr(tmem_c, warp, c), v);
    umma::tmem_ld_wait();
#pragma unroll
    for (int e = 0; e < 4; ++e) dst[c + e] = __uint_as_float(v[e]);
  }
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
  bf16* skg = sm16 + Smem::off_kg;
  bf16* sAq = sm16 + Smem::off_aq;
  bf16* svp = sm16 + Smem::off_vp;
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(smraw + Smem::off_tmem);
  uint64_t* bar = reinterpret_cast<uint64_t*>(smraw + Smem::off_bar);
  float* shf = reinterpret_cast<float*>(smraw + Smem::off_hf32);
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
    umma::tmem_alloc(tmem_slot, kTmemAcc);
    umma::tmem_relinquish();
  }
  __syncthreads();
  const uint32_t t_acc = *tmem_slot;

  // (1)(2)(3) 都是 M=BT、N=V；A 操作数 K-major。
  const uint32_t idesc_bv = umma::make_instr_desc_bf16(kBT, kV, false, false);
  // (3) Aq·v'：A=Aq[BT,BT] K-major，B=v'[BT,V] K-major（K 维 = token）
  // (4) v'ᵀ·kg：A=v'ᵀ → v' 的 MN-major 视图（M=V、K=BT）；B=kg[BT,K] K-major
  const uint32_t idesc_vk = umma::make_instr_desc_bf16(kV, kK, /*a_mn=*/true, false);

  const long vstride = (long)HV * V;   // 按 hv 分段的行跨度
  const long qstride = (long)H * kK;   // qn 按 h 分段的行跨度

  // h 的 f32 主本放 **smem**（shf，64 KB）：每个线程一行 128 个 f32 会占满寄存器预算
  // （实测 REG 255 + STACK 136 spill）。128 线程各负责 V=128 的一行。
  const int myrow = warp * 32 + (tid & 31);
  const float* hslot = state + (long)slot * state_stride_tok + (long)hv * V * kK;
  for (int kk = 0; kk < kK; ++kk) shf[(size_t)myrow * kK + kk] = hslot[(size_t)myrow * kK + kk];
  auto stage_h = [&]() {
    for (int kk = 0; kk < kK; ++kk) {
      const int kb = kk >> 6, kc = kk & 63;
      sh[kb * (kV * 64) + (size_t)(myrow * 8 + ((kc >> 3) ^ (myrow & 7))) * 8 + (kc & 7)] =
          dinfer::to_bf16(shf[(size_t)myrow * kK + kk]);
    }
  };
  stage_h();
  __syncthreads();

  // 只留 hrow（f32 主本，128 个）+ tmp（四个 GEMM 的取回缓冲复用同一个，128 个）——
  // 四个独立数组会让寄存器爆到 255 + spill（实测 STACK 136）。
  float tmp[kK];
  for (int ci = 0; ci < nt; ++ci) {
    const int len = min(kBT, len_seq - ci * kBT);
    const long row0 = bos + (long)ci * kBT;
    const long ubase = (row0 * vstride) + (long)hv * V;

    // 装载 q（按 h 段）、w / kg / Aq（按 hv 段）
    for (int kb = 0; kb < Smem::kblk; ++kb) {
      const int k0 = kb * 64;
      stage_q_dec(sq + kb * (kBT * 64), qn + row0 * qstride + (long)h * kK + k0, len, (int)qstride, sdec, tid);
      stage_tile(sw + kb * (kBT * 64), w + ubase + k0, len, (int)vstride, tid);
      stage_tile(skg + kb * (kBT * 64), kg + ubase + k0, len, (int)vstride, tid);
    }
    if (tid < kBT) sdec[tid] = (tid < len) ? exp2f(gc[(row0 + tid) * HV + hv]) : 0.f;
    __syncthreads();
    // Aq 装载时把 dec（按 token = 行）折进 A 侧：o = (q·hᵀ)·dec + Aq·v' 里 dec 只作用于第一项，
    // 所以折进 q 而不是 Aq。见下方 (2) 的装载。
    // Aq 的形状是 [num_chunks, HV, BT, BT]（每 chunk 一个 BT×BT），不是 [T, HV, V]。
    {
      // Aq 是 [num_chunks, HV, BT, BT]：一个 chunk 一个 BT×BT 矩阵，行跨度 = BT。
      const bf16* aqc = reinterpret_cast<const bf16*>(aq) + ((long)ci * HV + hv) * kBT * kBT;
      for (int i = tid; i < kBT * (kBT / 8); i += kThreads) {
        const int row = i / (kBT / 8), blk = i % (kBT / 8);
        const int phys = row * 8 + (blk ^ (row & 7));
        uint4 v = make_uint4(0, 0, 0, 0);
        if (row < len) v = *reinterpret_cast<const uint4*>(aqc + row * kBT + blk * 8);
        *reinterpret_cast<uint4*>(sAq + (size_t)phys * 8) = v;
      }
    }
    __syncthreads();
    umma::fence_proxy_async_shared();

    // ---- (1) wh = w·hᵀ → 取回 → v' = u − wh
    if (warp == 0 && umma::elect_one()) {
      mma_panel(t_acc, sw, kBT, sh, kV, idesc_bv, false);
      umma::mma_commit(bar);
    }
    umma::mbar_wait(bar, 0);
    umma::fence_after_sync();
    __syncthreads();
    tmem_get_row(tmp, t_acc, kV, tid);
    // v' 存成 **[v 块][token][64 v]**（与 r12a 的 V 布局同）：块内 token 是行、64 个 v 连续。
    //   (3) Aq·v'：B 的 MN = v（N 方向 64 一个块，LBO 跳到下一块）、K = token → MN-major 描述符
    //   (4) v'ᵀ·kg：A 的 M = v、K = token，同一块内存按 MN-major 也是 [M=V, K=token] 的转置视图
    if (myrow < len) {
      const bf16* urow = u + (row0 + myrow) * vstride + (long)hv * V;
      for (int vb = 0; vb < Smem::kblk; ++vb) {
        for (int kc = 0; kc < 64; ++kc) {
          const int vv = vb * 64 + kc;
          const float vp = dinfer::to_f32(urow[vv]) - tmp[vv];
          // 行 = token（myrow）、列 = kc（该块内的 v），行跨度 64 个 v。
          svp[vb * (kBT * 64) + (size_t)(myrow * 8 + ((kc >> 3) ^ (myrow & 7))) * 8 + (kc & 7)] =
              dinfer::to_bf16(vp);
        }
      }
    }
    __syncthreads();
    umma::fence_proxy_async_shared();

    // ---- (2)+(3) o = (q⊙dec)·hᵀ + Aq·v'：两条 MMA 连续累加到同一累加器，中间不取回。
    if (warp == 0 && umma::elect_one()) {
      mma_panel(t_acc, sq, kBT, sh, kV, idesc_bv, /*accumulate_first=*/false);
      // A=Aq[BT,BT]、B=v'[BT,V]：K 维 = BT = 64，一个 K 块。
      const uint64_t daq = umma::k_major_128b_desc(smem_u32(sAq));
      // B = v'[v 块][token][64 v]，MN = v、K = token → MN-major。
      // atom = 8 token × 128 B；LBO = MN 方向下一个 v 块 = kBT*64*2 B；SBO = K 方向下一个 atom = 1024 B。
      const uint64_t dvp = umma::mn_major_128b_desc(smem_u32(svp), /*lbo=*/kBT * 64 * 2, /*sbo=*/8 * 128);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const uint32_t ko = (uint32_t)(k * 16 * 128) >> 4;  // 16 个 token × 128 B
        umma::mma_f16(t_acc, daq + ((uint32_t)(k * 16 * 2) >> 4), dvp + ko, idesc_bv, true);
      }
      umma::mma_commit(bar + 1);
    }
    umma::mbar_wait(bar + 1, 0);
    umma::fence_after_sync();
    __syncthreads();
    tmem_get_row(tmp, t_acc, kV, tid);
    if (myrow < len) {
      bf16* orow = out + (row0 + myrow) * vstride + (long)hv * V;
      for (int kk = 0; kk < kV; ++kk) orow[kk] = dinfer::to_bf16(tmp[kk]);
    }
    __syncthreads();

    // ---- (4) h = glast·h + v'ᵀ·kg
    // A = v'ᵀ（v' 的 MN-major 视图：M=V、K=BT），B = kg[BT, K] K-major。
    if (warp == 0 && umma::elect_one()) {
      // v' 的 MN-major：atom = 8 token × 128 B；LBO = MN 方向下一块（V 方向 64 个 v 一块）；
      // SBO = K 方向下一个 atom（8 token）= 1024 B。
      // A = v'ᵀ：v' 是 [v 块][token][64 v]，MN = v、K = token。
      // 对 A 而言 MN-major 就是「同一块内存按 [K=token 行, MN=v 列] 的转置视图」，正是 v'ᵀ。
      const uint64_t da = umma::mn_major_128b_desc(smem_u32(svp), /*lbo=*/kBT * 64 * 2, /*sbo=*/8 * 128);
      const uint64_t db = umma::k_major_128b_desc(smem_u32(skg));
      // A 的 K 维 = token = 64，B 的 K 维 = token = 64；MN 方向 V=128、N=K=128。
#pragma unroll
      for (int kb = 0; kb < Smem::kblk; ++kb) {
        const uint64_t dbk = db + (uint32_t)(kb * (kBT * 64 * 2) >> 4);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const uint32_t ko = (uint32_t)(k * 16 * 2) >> 4;  // A：16 个 token
          umma::mma_f16(t_acc, da + ko, dbk + ko, idesc_vk, kb > 0 || k > 0);
        }
      }
      umma::mma_commit(bar + 3);
    }
    umma::mbar_wait(bar + 3, 0);
    umma::fence_after_sync();
    __syncthreads();
    tmem_get_row(tmp, t_acc, kK, tid);
    const float glast = sdec[len - 1];
    for (int kk = 0; kk < kK; ++kk) shf[(size_t)myrow * kK + kk] = shf[(size_t)myrow * kK + kk] * glast + tmp[kk];
    __syncthreads();
    stage_h();
    __syncthreads();
    umma::fence_proxy_async_shared();
  }

  // 末态回写
  {
    float* hout = state + (long)slot * state_stride_tok + (long)hv * V * kK;
    for (int kk = 0; kk < kK; ++kk) hout[(size_t)myrow * kK + kk] = shf[(size_t)myrow * kK + kk];
  }
  umma::fence_before_sync();
  __syncthreads();
  if (warp == 0) umma::tmem_dealloc(t_acc, kTmemAcc);
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
