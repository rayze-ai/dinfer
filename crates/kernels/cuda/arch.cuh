// 跨卡型能力常量（**唯一**允许的 arch 分叉点，见 CLAUDE.md「多卡型实现」规则）。
//
// 规则（硬性）：
//   1. kernel 体内**禁止** `#if __CUDA_ARCH__`；要分叉只读这里的 `kHas*` / `k*Bytes` 有名常量。
//   2. 这里只放「卡型固定的物理能力」，**不放调优数值**（groups/stages/切分等一律由 Rust 侧
//      `ArchProfile`（arch.rs）给，host 传参；环境变量只做覆盖）。
//   3. 不在 kernel 里写裸数字表达卡型差异。
#pragma once

namespace dinfer {

#if defined(__CUDA_ARCH__)
// ---- 卡型（编译期选定，仅用于给下面的能力常量取值；kernel 体内不要再用它做逻辑分叉）----
// sm_80/86/89 = Ampere/Ada 消费/专业；sm_90 = Hopper；sm_100/103 = Blackwell 数据中心（B200/B300）；
// sm_120 = 消费级 Blackwell（RTX 5090）。
constexpr bool kArchSm8x = (__CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 900);
constexpr bool kArchSm90 = (__CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000);
constexpr bool kArchSm100 = (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200);
constexpr bool kArchSm120 = (__CUDA_ARCH__ >= 1200);

/// 每 CTA 可用的动态/静态 smem 上限（B）。仅作粗分档给 host 参考；精确值 host 用
/// `cudaDeviceGetAttribute` 查，别在 kernel 里拿它算切分。
constexpr int kSmemBytes = kArchSm100 ? (228 * 1024) : (kArchSm90 ? (227 * 1024) : (100 * 1024));

/// 是否有 TMA（`cp.async.bulk` / tensor map）。数据中心 Blackwell / Hopper 有；消费级 Blackwell
/// 与 Ampere 系第一版不用。megakernel 第一版统一走 `cp.async.cg`（不依赖 TMA）。
constexpr bool kHasTma = kArchSm100 || kArchSm90;

/// fp4 块缩放 MMA（`mma.sync...kind::mxf4nvf4.block_scale`，NVFP4）是否可用。
/// sm120 需 `compute_120a` cubin（build.rs 已按 arch 单独编）；sm100 系列同族。
constexpr bool kHasMmaFp4BlockScale = kArchSm100 || kArchSm120;

/// tcgen05（第五代 tensor core，仅数据中心 Blackwell）。
constexpr bool kHasTcgen05 = kArchSm100;
#else
// host 编译单元只看到 `#include`，不参与上面常量；host 侧的卡型判断走 Rust `ArchProfile`。
#endif

}  // namespace dinfer
