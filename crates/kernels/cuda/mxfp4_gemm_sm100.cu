// MXFP4 grouped GEMM 的 sm100（Blackwell）路径 —— **空壳 + TODO，本阶段不实现**。
// 规格与走法见 docs/loops/k3-moe.md「sm_103（B300）版的走法」。
//
// 为什么单开一个文件：CLAUDE.md 的多卡型规则要求按 arch 分文件、kernel 体内禁止
// `#if __CUDA_ARCH__`；host 侧按 ArchProfile 选实现（与 gqa / gqa_decode_v3 的既有做法一致）。
//
// TODO(sm100)：
//   1. 权重保持 packed（u8 [E, N, K/2]）直接进场，**不解包成 bf16**（896 专家 × 3 张量 ≈ 14 GB/层）。
//   2. block-scaled MMA：按 CUDA 13 头文件确认可用的 MXFP4 指令
//      （`tcgen05.mma` 的 scale-factor 通道，或 `mma.sync.m16n8k64` 的 e2m1 变体）。
//   3. 激活侧也要量化成 MXFP4（group 32 + e8m0），量化融合进 tile 载入。
//   4. 门禁：sm89 / sm103 两卡各跑一次 bench_gate，任一退化 > 3% 不合入。
//
// 在实现之前，host 侧对 sm100 直接回落到 sm8x 实现（mxfp4_gemm_sm8x.cu）。
#include "common.cuh"
#include "kernels.h"

extern "C" {

// 占位入口：显式调用即返回 cudaErrorNotSupported，提醒调用方回落到 sm8x。
int mxfp4_gemm_grouped_sm100(const void* xg, const void* wp, const void* ws, const int32_t* expert_off,
                             const int32_t* tile_e, const int32_t* tile_m, const int32_t* tile_n, void* out,
                             int64_t num_tiles, int64_t K, int64_t N, int64_t group, cudaStream_t stream) {
  (void)xg; (void)wp; (void)ws; (void)expert_off; (void)tile_e; (void)tile_m; (void)tile_n; (void)out;
  (void)num_tiles; (void)K; (void)N; (void)group; (void)stream;
  return (int)cudaErrorNotSupported;
}

}  // extern "C"
