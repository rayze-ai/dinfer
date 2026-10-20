// gdn_ho_sm100：GDN chunk prefill 的 ho 段 tcgen05（UMMA）版 —— r11d 里程碑 1。
//
// 背景（docs/loops/qwen35-gdn-r11d.md「作废原因」）：把 wy 熔进 ho 是 2.5× FLOP，作废。
// 真正的缺口是利用率：ho 只有 mma.sync 峰值 575 TF/s 的 15%，wy 只有 12%。
// FlashInfer 293 TF/s 已超 48-SM 的 mma.sync 上限 187 TF/s → 它必然走 tcgen05 且 BV = 全 V。
//
// 本文件当前只实现四个 GEMM 中的 **(1) wh = w·hᵀ**，目的是把 tcgen05 在这套数据布局下的
// 全部未知点一次性验掉：TMEM 分配、128 B swizzle 装载、K-major 描述符、tcgen05.ld 取回。
// (2)(3)(4) 是同样的模式重复，等这一段编过、数值对过再逐个加。
//
// 布局约定（BT=64 token，K=V=128）：
//   sh  [V=128][64] × 2 块   h 的 bf16 影子；k 维连续 → k-major，B 操作用
//   sw  [BT][64]   × 2 块    w；k 维连续 → k-major，A 操作用
//   结果 wh [BT, V] 在 TMEM 的 t_acc 上
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
constexpr int kDepth = 2;

struct Smem {
  static constexpr int kblk = kK / 64;            // 2 个 64 列的 k 块
  static constexpr int off_h = 0;                       // [V][64] × kblk  bf16
  static constexpr int h_elems = kV * 64 * kblk;        // 16 K bf16 = 32 KB
  static constexpr int off_w = off_h + h_elems;         // [BT][64] × kblk bf16
  static constexpr int w_elems = kBT * 64 * kblk;       // 8 K bf16 = 16 KB
  static constexpr int bf16_elems = off_w + w_elems;    // 24 K bf16 = 48 KB
  static constexpr int bf16_bytes = bf16_elems * 2;
  static constexpr int off_tmem = (bf16_bytes + 127) & ~127;
  static constexpr int off_bar = off_tmem + 16;
  static constexpr int bytes = off_bar + kDepth * 8;
};

constexpr int kTmemAcc = 128;   // wh 的累加器：N=V=128 列

// 把 [rows,64] 的 bf16 行主序 tile 写成 128 B swizzle（16 B 块号与 row%8 异或）。
__device__ __forceinline__ void stage_tile(bf16* dst, const bf16* src, int rows, int src_row_stride, int tid) {
  for (int i = tid; i < rows * 8; i += kThreads) {
    const int row = i >> 3, blk = i & 7;
    const int phys = row * 8 + (blk ^ (row & 7));
    *reinterpret_cast<uint4*>(dst + (size_t)phys * 8) =
        *reinterpret_cast<const uint4*>(src + (size_t)row * src_row_stride + blk * 8);
  }
}

// TMEM [rows, ncols] f32 → dst 行主序。128 线程 = 4 warp 恰好覆盖 128 lane（行 = warp*32 + lane）。
// 取回本线程所在 lane 的整行（warp w 覆盖 lane 32w..32w+31，caller 自己决定行号）。
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

__global__ void __launch_bounds__(kThreads) gdn_ho_probe_kernel(
    const bf16* __restrict__ w, const float* __restrict__ h_in, float* __restrict__ out, int len) {
  extern __shared__ __align__(128) unsigned char smraw[];
  bf16* sm16 = reinterpret_cast<bf16*>(smraw);
  bf16* sh = sm16 + Smem::off_h;
  bf16* sw = sm16 + Smem::off_w;
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(smraw + Smem::off_tmem);
  uint64_t* bar = reinterpret_cast<uint64_t*>(smraw + Smem::off_bar);
  const int tid = threadIdx.x, warp = tid >> 5;

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

  // h（f32 [V,K] 行主序）→ bf16 影子，按 k 分两块、每块 swizzle。
  // 源的行跨度是 K=128，块的列偏移 = kb*64。
  for (int i = tid; i < kV * kK; i += kThreads) {
    const int vr = i >> 7, kk = i & 127;
    const int kb = kk >> 6, kc = kk & 63;
    sh[kb * (kV * 64) + (size_t)(vr * 8 + ((kc >> 3) ^ (vr & 7))) * 8 + (kc & 7)] = dinfer::to_bf16(h_in[i]);
  }
  // w（bf16 [BT,K] 行主序，行跨度 kK）→ 同样按 k 分块。
  for (int kb = 0; kb < Smem::kblk; ++kb)
    stage_tile(sw + kb * (kBT * 64), w + kb * 64, kBT, kK, tid);
  __syncthreads();

  // wh = w · hᵀ：A=w[BT,K]，B=h 的 [V,K] k-major。K 维按 64 一块推进，共 kblk 块、每块 4 条 K=16。
  if (warp == 0 && umma::elect_one()) {
    const uint32_t idesc = umma::make_instr_desc_bf16(kBT, kV, /*a_mn=*/false, /*b_mn=*/false);
    const uint64_t da0 = umma::k_major_128b_desc(smem_u32(sw));
    const uint64_t db0 = umma::k_major_128b_desc(smem_u32(sh));
    // 第 kb 个 64 列 K 块在 smem 里的字节跨度：A 是 [BT][64]，B 是 [V][64]。
    constexpr uint32_t kAblk = (uint32_t)(kBT * 64 * 2) >> 4;
    constexpr uint32_t kBblk = (uint32_t)(kV * 64 * 2) >> 4;
#pragma unroll
    for (int kb = 0; kb < Smem::kblk; ++kb) {
      const uint64_t da = da0 + (uint32_t)kb * kAblk, db = db0 + (uint32_t)kb * kBblk;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const uint32_t ko = (uint32_t)(k * 16 * 2) >> 4;  // K=16 步进 = 32 B
        umma::mma_f16(t_acc, da + ko, db + ko, idesc, kb > 0 || k > 0);
      }
    }
    umma::mma_commit(bar);
  }
  umma::mbar_wait(bar, 0);
  umma::fence_after_sync();
  __syncthreads();

  {
    // 128 线程 = 4 warp 覆盖 128 lane；只有前 BT=64 行有输出，warp 0/1 有效。
    float row[kV];
    tmem_get_row(row, t_acc, kV, tid);
    const int r = (tid >> 5) * 32 + (tid & 31);
    if (r < kBT) {
      for (int vv = 0; vv < kV; ++vv) out[(size_t)r * kV + vv] = row[vv];
    }
  }

  umma::fence_before_sync();
  __syncthreads();
  if (warp == 0) umma::tmem_dealloc(t_acc, kTmemAcc);
}

int set_smem(const void* fn, int bytes) {
  return dinfer::err(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

}  // namespace

// host 入口（里程碑 1 自测用）：单个 block 算一次 wh = w·hᵀ。
// w: bf16 [BT,K]；h_in: f32 [V,K]；out: f32 [BT,V]。
int gdn_ho_probe(const void* w, const void* h_in, void* out, int64_t bt, int64_t k, cudaStream_t stream) {
  if (bt != kBT || k != kK) return dinfer::err(cudaErrorInvalidValue);
  if (const int rc = set_smem(reinterpret_cast<const void*>(&gdn_ho_probe_kernel), Smem::bytes)) return rc;
  gdn_ho_probe_kernel<<<1, kThreads, Smem::bytes, stream>>>(static_cast<const bf16*>(w),
                                                            static_cast<const float*>(h_in), static_cast<float*>(out),
                                                            (int)bt);
  return dinfer::last_err();
}
