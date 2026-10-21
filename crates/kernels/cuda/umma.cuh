// tcgen05（UMMA）最小工具层（r12a，docs/loops/qwen35-tcgen05.md）。
//
// 只放**与模型、形状无关**的原语：TMEM 分配、smem 描述符、指令描述符、单条 MMA、累加器取回、
// mbarrier 完成等待。tile 形状 / stage 数 / warp 数由调用方（kernel 的模板参数或运行时参数）给。
//
// 只在 sm_100a / sm_103a 的编译单元里 include（`*_sm100.cu`）；build.rs 负责这些文件只在
// CUDA_ARCHS 含 100/103 时编入。文件名带 `_sm100` 后缀即遵守 CLAUDE.md 的多卡型分文件规则。
//
// 位域与 asm 形式的来源（逐条核对过，不是猜的）：
//   - SmemDescriptor / InstrDescriptor 位域：CUTLASS `cute/arch/mma_sm100_desc.hpp`
//   - 描述符构造（K-major + SWIZZLE_128B 的 SBO/LBO）：DeepGEMM `common/sm100_utils.cuh` 的
//     `make_smem_desc` / `make_umma_desc<Major::K>`
//   - tcgen05.alloc / dealloc / relinquish：CUTLASS `cute/arch/tmem_allocator_sm100.hpp`
//   - tcgen05.mma.cta_group::1.kind::f16（不带 mask 的宽松形式）：DeepGEMM `common/sm100_utils.cuh`
//   - tcgen05.ld.sync.aligned.32x32b.xN.b32：CUTLASS `cute/arch/copy_sm100.hpp`
//   - tcgen05.commit / wait::ld：CUTLASS `cutlass/arch/barrier.h`
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace umma {

// TMEM 容量：每 SM 512 列 × 128 lane × 4 B = 256 KB。
constexpr int kTmemColumns = 512;
// 累加器 tile 的列数必须是 2 的幂且 32..512（tcgen05.alloc 的约束）。
__device__ __forceinline__ int tmem_cols_for(int n) {
  int c = 32;
  while (c < n && c < kTmemColumns) c <<= 1;
  return c;
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---------------------------------------------------------------- TMEM 分配
// 必须由**单个满 warp** 发起；结果写到 smem 的一个 u32。
__device__ __forceinline__ void tmem_alloc(uint32_t* dst_smem, int ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(
                   smem_u32(dst_smem)),
               "r"(ncols));
}
__device__ __forceinline__ void tmem_dealloc(uint32_t tmem, int ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(tmem), "r"(ncols));
}
__device__ __forceinline__ void tmem_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
}
// 取回累加器后必须等 ld 落地才能读这些寄存器。
__device__ __forceinline__ void tmem_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;\n"); }

// ------------------------------------------------------------- smem 描述符
// 64 bit；start/LBO/SBO 都是"字节地址 >> 4"。layout: 2 = SWIZZLE_128B。
__device__ __forceinline__ uint64_t make_smem_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo, uint32_t layout) {
  uint64_t d = 0;
  d |= static_cast<uint64_t>((saddr >> 4) & 0x3fffu);             // bits [0,14)
  d |= static_cast<uint64_t>((lbo >> 4) & 0x3fffu) << 16;         // bits [16,30)
  d |= static_cast<uint64_t>((sbo >> 4) & 0x3fffu) << 32;         // bits [32,46)
  d |= static_cast<uint64_t>(1u) << 46;                           // version = 1（Blackwell）
  d |= static_cast<uint64_t>((layout & 7u)) << 61;                // bits [61,64) layout_type
  return d;
}

// K-major（K 连续）+ 128 B swizzle 的一块 [MN, BK] bf16 tile。
// 约束（DeepGEMM make_umma_desc<Major::K>）：swizzle 必须等于 BK·sizeof(dtype) = 128 B，
// 即 BK = 64 个 bf16；K 轴上只有一个 swizzle atom，所以 LBO = 0，
// SBO = 8 行 × 128 B（一个 atom = 8 行 × 128 B）；K 轴推进 = 每个 bf16 2 B，>>4 后由调用方算。
constexpr uint32_t kSwz128B = 2;
constexpr int kBKTile = 64;              // 128 B / sizeof(bf16)
constexpr uint32_t kSboKMajor128B = 8 * 128;  // 1024 B

__device__ __forceinline__ uint64_t k_major_128b_desc(uint32_t saddr) {
  // CUTLASS `make_umma_desc<Major::K>` 的 canonical（mma_sm100_desc.hpp:833）：
  //   B128 : Swizzle<3,4,3> o smem_ptr o ((8,n),2):((8,SBO),1)
  // 注意：那个 `1` 是 **uint128 单位的 stride**，不是 LBO 字节字段的值。
  // B128 + K-major 时 K 方向的两半由 swizzle 自身定位，LBO 保持 0。
  // （2026-10-21 实测：把 LBO 填成 16 会让 w·hᵀ 从 4.4047 掉到 3.09 —— 里程碑 1 的 lbo=0 是对的。）
  return make_smem_desc(saddr, /*lbo=*/0, /*sbo=*/kSboKMajor128B, kSwz128B);
}

// 在一个 [MN, 64] bf16 tile 的 swizzle 布局里，(row, k) 的 smem 元素偏移。
// 128 B swizzle：16 B 一块，块号 c 与 (row % 8) 异或（CUTLASS Swizzle<3,3,3> 的等价写法，
// 与 DeepGEMM 尾处理 `col ^= row % (swizzle/16)` 同）。
__device__ __forceinline__ int swz_off(int row, int k) {
  const int c = k >> 3;                  // 16 B 块号（每块 8 个 bf16）
  const int phys = row * 8 + (c ^ (row & 7));
  return phys * 8 + (k & 7);
}

// ------------------------------------------------------------ 指令描述符
// bf16 A/B，f32 累加；a_mn_major / b_mn_major 为 0 表示 K-major。
__device__ __forceinline__ constexpr uint32_t make_instr_desc_bf16(int m, int n, bool a_mn, bool b_mn) {
  uint32_t d = 0;
  d |= 1u << 4;                 // c_format = F32   bits [4,6)
  d |= 1u << 7;                 // a_format = BF16  bits [7,10)
  d |= 1u << 10;                // b_format = BF16  bits [10,13)
  d |= (a_mn ? 1u : 0u) << 15;  // a_major
  d |= (b_mn ? 1u : 0u) << 16;  // b_major
  d |= static_cast<uint32_t>(n >> 3) << 17;  // n_dim bits [17,23)
  d |= static_cast<uint32_t>(m >> 4) << 24;  // m_dim bits [24,29)
  return d;
}

// ---------------------------------------------------------------- UMMA 发射
// accumulate=false → 覆盖累加器（K 链的第一条）；true → D = A·B + D。
__device__ __forceinline__ void mma_f16(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
                                        bool accumulate) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
      "}\n" ::"r"(tmem_c),
      "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(static_cast<uint32_t>(accumulate)));
}

// ----------------------------------------------------------------- mbarrier
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count));
}
// init 之后、被 tcgen05.commit 到达之前需要的可见性栅栏。
__device__ __forceinline__ void mbar_init_fence() {
  asm volatile("fence.mbarrier_init.release.cluster;\n");
}
// phase 是 0/1 交替的奇偶；返回时该次到达已完成。
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  uint32_t a = smem_u32(bar);
  asm volatile(
      "{\n\t"
      ".reg .pred P;\n\t"
      "WAIT%=:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
      "@!P bra WAIT%=;\n\t"
      "}\n" ::"r"(a),
      "r"(phase));
}
// 等该 CTA 之前发射的所有 tcgen05.mma 完成，到达 bar 一次。
__device__ __forceinline__ void mma_commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" ::"r"(
      smem_u32(bar)));
}
// 发 MMA 的线程与做 smem 写 / 读的线程之间需要这对栅栏（CUTLASS / DeepGEMM 同）。
__device__ __forceinline__ void fence_before_sync() { asm volatile("tcgen05.fence::before_thread_sync;\n"); }
// generic proxy 写（普通 st.shared）之后、async proxy（UMMA 读 smem）之前必须有的栅栏。
__device__ __forceinline__ void fence_proxy_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;\n");
}
__device__ __forceinline__ void fence_after_sync() { asm volatile("tcgen05.fence::after_thread_sync;\n"); }

// ------------------------------------------------------------ 累加器取回
// 32 data-path lane × 32 bit，取 4 列。warp w 必须访问 lane 32w..32w+31
// （CUTLASS copy_traits_sm100 的断言：((addr>>16)/32)%4 == (warp_id)%4）。
__device__ __forceinline__ void tmem_ld32_x4(uint32_t taddr, uint32_t (&v)[4]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
               : "r"(taddr));
}

// 32 列一次（softmax / epilogue 的主通道）。x32 是 x4 的 8 倍宽，一条指令取回一整行片段。
__device__ __forceinline__ void tmem_ld32_x32(uint32_t taddr, uint32_t (&v)[32]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
      : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]),
        "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]),
        "=r"(v[16]), "=r"(v[17]), "=r"(v[18]), "=r"(v[19]), "=r"(v[20]), "=r"(v[21]), "=r"(v[22]), "=r"(v[23]),
        "=r"(v[24]), "=r"(v[25]), "=r"(v[26]), "=r"(v[27]), "=r"(v[28]), "=r"(v[29]), "=r"(v[30]), "=r"(v[31])
      : "r"(taddr));
}

// 写回累加器（online softmax 的 rescale 用）。
__device__ __forceinline__ void tmem_st32_x32(uint32_t taddr, const uint32_t (&v)[32]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], "
      "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
      "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};\n" ::"r"(taddr),
      "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7]), "r"(v[8]),
      "r"(v[9]), "r"(v[10]), "r"(v[11]), "r"(v[12]), "r"(v[13]), "r"(v[14]), "r"(v[15]), "r"(v[16]), "r"(v[17]),
      "r"(v[18]), "r"(v[19]), "r"(v[20]), "r"(v[21]), "r"(v[22]), "r"(v[23]), "r"(v[24]), "r"(v[25]), "r"(v[26]),
      "r"(v[27]), "r"(v[28]), "r"(v[29]), "r"(v[30]), "r"(v[31]));
}
__device__ __forceinline__ void tmem_st_wait() { asm volatile("tcgen05.wait::st.sync.aligned;\n"); }

// TMEM 地址：低 16 位列号、高 16 位 lane。lane 基址 = 32·warp（CUTLASS copy_traits 的断言）。
__device__ __forceinline__ uint32_t tmem_addr(uint32_t base, int warp, int col) {
  return base + ((uint32_t)(warp * 32) << 16) + (uint32_t)col;
}

// ------------------------------------------------- MN-major 的 128 B 描述符
// 用于 PV 的 B 操作数（V 在 smem 里按 [d 块][key][64 d] 放：d 是 MN、key 是 K）。
// 由 canonical ((8,n),(8,k)):((1,LBO),(8,SBO))（16 B 单位）推得：
//   一个 atom = 8 个 k 行 × 128 B；LBO = MN 块之间的距离，SBO = 8 个 k 行（一个 atom）之间的距离。
__device__ __forceinline__ uint64_t mn_major_128b_desc(uint32_t saddr, uint32_t lbo_bytes, uint32_t sbo_bytes) {
  return make_smem_desc(saddr, lbo_bytes, sbo_bytes, kSwz128B);
}


// softmax 的 exp2：SFU 近似，v6 实测比库函数快（r11a 第 4 轮）。
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ bool elect_one() {
  uint32_t pred = 0;
  asm volatile("{\n\t.reg .pred P;\n\telect.sync _|P, 0xffffffff;\n\tselp.b32 %0, 1, 0, P;\n\t}\n" : "+r"(pred));
  return pred != 0;
}

}  // namespace umma
