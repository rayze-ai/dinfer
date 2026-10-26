// 跨卡型能力常量（**唯一**允许的 arch 分叉点，见 CLAUDE.md「多卡型实现」规则）。
//
// 规则（硬性）：
//   1. kernel 体内**禁止** `#if __CUDA_ARCH__`；要分叉只读这里的 `kHas*` / `k*Bytes` 有名常量。
//   2. 这里只放「卡型固定的物理能力」，**不放调优数值**（groups/stages/切分等一律由 Rust 侧
//      `ArchProfile`（arch.rs）给，host 传参；环境变量只做覆盖）。
//   3. 不在 kernel 里写裸数字表达卡型差异。
#pragma once

namespace dinfer {

// 设备 pass 有 __CUDA_ARCH__；host pass 没有（host 侧卡型判断走 Rust `ArchProfile`，
// 这里的常量在 host pass 取保守默认值，不应被 host 代码使用）。
#if defined(__CUDA_ARCH__)
#define DINFER_ARCH_CC (__CUDA_ARCH__)
#else
#define DINFER_ARCH_CC (0)
#endif

// ---- 卡型（仅用于给下面的能力常量取值；kernel 体内不要再用它做逻辑分叉）----
// sm_80/86/89 = Ampere/Ada；sm_90 = Hopper；sm_100/103 = Blackwell 数据中心（B200/B300）；
// sm_120 = 消费级 Blackwell（RTX 5090）。
constexpr bool kArchSm8x = (DINFER_ARCH_CC >= 800 && DINFER_ARCH_CC < 900);
constexpr bool kArchSm90 = (DINFER_ARCH_CC >= 900 && DINFER_ARCH_CC < 1000);
constexpr bool kArchSm100 = (DINFER_ARCH_CC >= 1000 && DINFER_ARCH_CC < 1200);
constexpr bool kArchSm120 = (DINFER_ARCH_CC >= 1200);

/// 每 CTA 可用的 smem 上限（B）的**粗分档**，仅供 kernel 侧做保守上界判断；
/// host 侧精确值一律用 `cudaDeviceGetAttribute` 查（别在 kernel 里拿它算切分）。
constexpr int kSmemBytes = kArchSm100 ? (228 * 1024) : (kArchSm90 ? (227 * 1024) : (100 * 1024));

/// 是否有 TMA（`cp.async.bulk` / tensor map）。megakernel 第一版统一走 `cp.async.cg`，不依赖它。
constexpr bool kHasTma = kArchSm100 || kArchSm90;

/// fp4 块缩放 MMA（`mma.sync...kind::mxf4nvf4.block_scale`，NVFP4）是否可用。
/// sm120 需 `compute_120a` cubin（build.rs 已按 arch 单独编）。
constexpr bool kHasMmaFp4BlockScale = kArchSm100 || kArchSm120;

/// tcgen05（第五代 tensor core，仅数据中心 Blackwell）。
constexpr bool kHasTcgen05 = kArchSm100;

#undef DINFER_ARCH_CC

}  // namespace dinfer

