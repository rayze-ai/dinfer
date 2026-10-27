// 模型无关的 dense MLP 支路 persistent megakernel（sm120，单卡）。
//
// 范围（第一版，最小可验证闭环）：把 dense MLP 支路的多个独立 kernel 压成**一次 launch**：
//   pre_ff_norm(rmsnorm) → gate GEMM(fp8) ‖ up GEMM(fp8) → gelu_tanh·mul(round2) → down GEMM(fp8) → post_ff_norm_1(rmsnorm)
// attention / MoE 仍走现有逐 kernel 路径（不在本 kernel 内）。
//
// **严格对齐 layer.rs 的真实语义**（L852-869）：
//   - `dense_gate` 与 `dense_up` 是**两个独立的 QLinear**（各自权重、各自 per-tensor scale），不是合并 [2I,H]；
//   - gelu 走 `gelu_tanh_mul_round2_bf16` 口径（gelu 先舍 bf16 再乘 up，vLLM act_and_mul 行为）；
//   - `dense_down` 吃 gelu 的 bf16 输出。
//
// 结构照 `crates/kernels/cuda/mega_sm100.cu` 的 persistent 框架；公共件在 `mega_common.cuh`。
// fp8 mma/ldmatrix 片段寻址照 `gemm_fp8_sm120.cu`（已验证正确的那份）。
#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "arch.cuh"
#include "common.cuh"
#include "kernels.h"
#include "mega_common.cuh"
// device 原语（fp8 mma / fp8_gemm_tile / rmsnorm_row_warp / gelu·mul）抽到共享头，B2 整层 megakernel 复用。
#include "mega_mlp_core.cuh"

namespace dinfer {
namespace mega_mlp {
// fp8_gemm_tile 三段 cycle 累加器 [载入, mma, 写出]（tid==0 across all GEMM tiles）。无条件定义：
// 符号永远在（C ABI 可链接），仅 MEGA_DIAG_GEMMTIME=1 时 fp8_gemm_tile 内才 atomicAdd，否则恒 0。诊断用，测完回退。
__device__ unsigned long long g_gemm_cyc[3] = {0, 0, 0};
}  // namespace mega_mlp
}  // namespace dinfer

extern "C" {
int mega_diag_gemmtime_read(unsigned long long* out3) {
  return (int)cudaMemcpyFromSymbol(out3, dinfer::mega_mlp::g_gemm_cyc, sizeof(unsigned long long) * 3);
}
int mega_diag_gemmtime_reset() {
  unsigned long long z[3] = {0, 0, 0};
  return (int)cudaMemcpyToSymbol(dinfer::mega_mlp::g_gemm_cyc, z, sizeof(unsigned long long) * 3);
}
}

namespace dinfer {
namespace mega_mlp {

using mega::kThreads;
using mega::kWarps;

enum Stage : int {
  S_NORM = 0,   // pre_ff_norm
  S_GATE = 1,   // dense_gate fp8 GEMM → gate_buf
  S_UP = 2,     // dense_up   fp8 GEMM → up_buf
  S_ACT = 3,    // gelu(gate)·up（round2）+ 量化 → act(fp8)
  S_DOWN = 4,   // dense_down fp8 GEMM → down_out
  S_POST = 5,   // post_ff_norm_1
  S_COUNT = 6,
};

/// host 填的一次 decode 步全部指针/形状（单卡，无卡间通信）。
struct Params {
  const bf16* x;      // [m, H] 输入残余流
  bf16* out;          // [m, H] 输出（post_ff_norm_1）
  const bf16* pre_w;  // [H]
  const bf16* post_w;  // [H]
  // gate: fp8 [I, H]；up: fp8 [I, H]；分别有自己的 per-tensor scale
  const uint8_t* gate_w;
  float gate_wscale;
  float gate_a_inv;
  const uint8_t* up_w;
  float up_wscale;
  float up_a_inv;
  // down: fp8 [H, I]
  const uint8_t* dn_w;
  float dn_wscale;
  float dn_a_inv;
  // 中间缓冲（调用方分配）
  bf16* norm1;    // [m, H]  bf16（对拍用）
  uint8_t* norm1_q;  // [m, H]  fp8（喂 gate/up 的 A）
  bf16* gate_buf;  // [m, I]
  bf16* up_buf;    // [m, I]
  bf16* act_bf;    // [m, I]  gelu·mul 结果
  uint8_t* act;    // [m, I]  量化后（喂 down）
  bf16* down_out;  // [m, H]
  int m, H, I;
  float eps;
  int pipe;
  int prof;
  const int* dep;
  const int* count;
  const int* start;
  int* ready;
  // split-K（每 GEMM 各一套；splits==1 时忽略 partial/sem）
  int sp_gu;        // gate/up 的 split 份数
  int sp_dn;        // down 的 split 份数
  float* part_gate; // [sp_gu][m][I]
  float* part_up;   // [sp_gu][m][I]
  float* part_dn;   // [sp_dn][m][H]
  int* sem_gate;    // [n_gu]
  int* sem_up;      // [n_gu]
  int* sem_dn;      // [n_dn]
};

// ---------------------------------------------------------------- Dispatch
struct MlpDenseDispatch {
  using Params = mega_mlp::Params;
  static constexpr int kBN = 128;  // 每 warp 16 行 N（b0 8 + b1 8）→ 8 warp × 16 = 128

  __device__ void run(Params& p, int shard, int stage, uint8_t* smem) const {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    switch (stage) {
      case S_NORM: {
        __shared__ float red[kWarps];
        {
          const bool live = warp < p.m;
          const int row = live ? warp : 0;
          bf16* n1 = p.norm1 + (long)row * p.H;
          rmsnorm_row_warp(p.x + (long)row * p.H, p.pre_w, n1, p.H, p.eps, lane, red, warp, live);
          if (live) {
            // 同时产出喂 fp8 GEMM 的量化副本
            uint8_t* q = p.norm1_q + (long)row * p.H;
            for (int i = lane; i < p.H; i += 32) {
              const float v = to_f32(n1[i]) * p.gate_a_inv;
              q[i] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
            }
          }
        }
        return;
      }
      case S_GATE: {
        // 口径（照 gemm_fp8_sm120.cu）：out = (A/a_scale · W) * (a_scale * w_scale)。
        // A 已按 a_inv=1/a_scale 量化，故 epilogue 需乘 (w_scale * a_scale) = w_scale / a_inv。
        // split-K：shard = tile·sp_gu + split
        const int tile = shard / p.sp_gu, split = shard - tile * p.sp_gu;
        fp8_gemm_tile<kBN>(p.norm1_q, p.gate_w, p.gate_buf, p.m, p.I, p.H, tile,
                           p.gate_wscale / p.gate_a_inv, smem, split, p.sp_gu, p.part_gate, p.sem_gate);
        return;
      }
      case S_UP: {
        const int tile = shard / p.sp_gu, split = shard - tile * p.sp_gu;
        fp8_gemm_tile<kBN>(p.norm1_q, p.up_w, p.up_buf, p.m, p.I, p.H, tile, p.up_wscale / p.up_a_inv, smem,
                           split, p.sp_gu, p.part_up, p.sem_up);
        return;
      }
      case S_ACT: {
        const long n = (long)p.m * p.I;
        for (long i = tid; i < n; i += kThreads) p.act_bf[i] = gelu_mul_round2(p.gate_buf[i], p.up_buf[i]);
        __syncthreads();
        for (long i = tid; i < n; i += kThreads) {
          const float v = to_f32(p.act_bf[i]) * p.dn_a_inv;
          p.act[i] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
        }
        return;
      }
      case S_DOWN: {
        const int tile = shard / p.sp_dn, split = shard - tile * p.sp_dn;
        fp8_gemm_tile<kBN>(p.act, p.dn_w, p.down_out, p.m, p.H, p.I, tile, p.dn_wscale / p.dn_a_inv, smem,
                           split, p.sp_dn, p.part_dn, p.sem_dn);
        return;
      }
      case S_POST: {
        __shared__ float red[kWarps];
        {
          const bool live = warp < p.m;
          const int row = live ? warp : 0;
          rmsnorm_row_warp(p.down_out + (long)row * p.H, p.post_w, p.out + (long)row * p.H, p.H, p.eps, lane, red,
                           warp, live);
        }
        return;
      }
    }
  }
};

}  // namespace mega_mlp
}  // namespace dinfer

// ---------------------------------------------------------------- host launch（C ABI）
extern "C" {

/// 一次 decode 步的 dense MLP 支路：pre_norm → gate ‖ up → gelu·mul → down → post_norm。
/// `dep/count/start/ready` 由调用方给：count[S_COUNT] 每阶段任务数、start 为前缀和、ready 长度 S_COUNT（**须先清零**）。
/// `n_gu` = gate/up 各自分几个 N tile（= ceil(I / BN)）；`n_dn` = down 的 N tile（= ceil(H / BN)）。
/// 返回 cudaError_t 数值（0 = 成功）。
int mega_mlp_dense_branch(const void* x, void* out, const void* pre_w, const void* post_w, const void* gate_w,
                         float gate_wscale, float gate_a_inv, const void* up_w, float up_wscale, float up_a_inv,
                         const void* dn_w, float dn_wscale, float dn_a_inv, void* norm1, void* norm1_q, void* gate_buf, void* up_buf,
                         void* act_bf, void* act, void* down_out, int m, int H, int I, float eps, int n_gu, int n_dn,
                         int* dep, int* count, int* start, int* ready, int* dep2, int pipe, int prof,
                         unsigned long long* wait_cyc, unsigned long long* exec_cyc,
                         int sp_gu, int sp_dn, void* part_gate, void* part_up, void* part_dn,
                         int* sem_gate, int* sem_up, int* sem_dn, int ctas, cudaStream_t stream) {
  using namespace dinfer;
  using namespace dinfer::mega_mlp;
  if (m <= 0 || m > 8 || H <= 0 || I <= 0) return (int)cudaErrorInvalidValue;
  if (sp_gu < 1) sp_gu = 1;
  if (sp_dn < 1) sp_dn = 1;
  constexpr int kBN = MlpDenseDispatch::kBN;
  // smem：sA(16×RS) + sB(BN×RS)
  const size_t smem = (size_t)(16 + kBN) * (128 + 16);

  // split-K 的 sem 须每次 launch 前清零（partial 不用：每 split 全写自己的区）。
  if (sp_gu > 1) {
    if (cudaMemsetAsync(sem_gate, 0, sizeof(int) * n_gu, stream) != cudaSuccess ||
        cudaMemsetAsync(sem_up, 0, sizeof(int) * n_gu, stream) != cudaSuccess)
      return (int)cudaGetLastError();
  }
  if (sp_dn > 1) {
    if (cudaMemsetAsync(sem_dn, 0, sizeof(int) * n_dn, stream) != cudaSuccess) return (int)cudaGetLastError();
  }

  // **注意**：dep/count/start/ready 是**设备指针**（Rust 侧 st.ptr_mut 传入）。表先在 host 本地算，
  // 再 cudaMemcpy 拷过去；直接写 *dep 会往设备内存解引用 → SIGSEGV。
  // dep/count/start/dep2 现由 **Rust 侧 MegaBufs::new 构造时预填到 device**（内容只依赖形状/split，层内每步不变）。
  // 不再 per-call 从栈数组 H2D memcpy——该 memcpy 在 CUDA graph capture 期录下**失效栈地址**，replay 读垃圾→dep 表乱→乱码
  // （kaon3-mega R45 根因）。dep/count/start/dep2 已是传入的常驻 device 指针，kernel 直接用；这里只按标量算总任务数 acc（标量进 graph 安全）。
  const int acc = 1 + n_gu * sp_gu + n_gu * sp_gu + 1 + n_dn * sp_dn + 1;

  Params p{};
  p.x = static_cast<const bf16*>(x);
  p.out = static_cast<bf16*>(out);
  p.pre_w = static_cast<const bf16*>(pre_w);
  p.post_w = static_cast<const bf16*>(post_w);
  p.gate_w = static_cast<const uint8_t*>(gate_w);
  p.gate_wscale = gate_wscale;
  p.gate_a_inv = gate_a_inv;
  p.up_w = static_cast<const uint8_t*>(up_w);
  p.up_wscale = up_wscale;
  p.up_a_inv = up_a_inv;
  p.dn_w = static_cast<const uint8_t*>(dn_w);
  p.dn_wscale = dn_wscale;
  p.dn_a_inv = dn_a_inv;
  p.norm1 = static_cast<bf16*>(norm1);
  p.norm1_q = static_cast<uint8_t*>(norm1_q);
  p.gate_buf = static_cast<bf16*>(gate_buf);
  p.up_buf = static_cast<bf16*>(up_buf);
  p.act_bf = static_cast<bf16*>(act_bf);
  p.act = static_cast<uint8_t*>(act);
  p.down_out = static_cast<bf16*>(down_out);
  p.m = m;
  p.H = H;
  p.I = I;
  p.eps = eps;
  p.pipe = pipe > 0 ? pipe : 2;
  p.prof = prof;
  p.dep = dep;
  p.count = count;
  p.start = start;
  p.ready = ready;
  p.sp_gu = sp_gu;
  p.sp_dn = sp_dn;
  p.part_gate = static_cast<float*>(part_gate);
  p.part_up = static_cast<float*>(part_up);
  p.part_dn = static_cast<float*>(part_dn);
  p.sem_gate = sem_gate;
  p.sem_up = sem_up;
  p.sem_dn = sem_dn;

  int dev = 0, sms = 1, max_smem = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  if ((int)smem > max_smem) return (int)cudaErrorInvalidValue;

  // ctas_per_sm：ctas>=2 用 OCC=2 变体（launch_bounds 放宽寄存器上限，允许 2 CTA 常驻），否则 OCC=1。
  // 实际常驻数再受 occupancy 查询约束（smem / 寄存器）。split-K 后任务数 > SM，多一波常驻能减少自旋空等。
  const bool want2 = ctas >= 2;
  auto kern1 = mega::mega_kernel<MlpDenseDispatch, 1>;
  auto kern2 = mega::mega_kernel<MlpDenseDispatch, 2>;
  const void* kptr = want2 ? (const void*)kern2 : (const void*)kern1;
  static thread_local int smem_set1 = 0, smem_set2 = 0;
  int& smem_set = want2 ? smem_set2 : smem_set1;
  if (smem_set < (int)smem) {
    if (const int rc = (int)cudaFuncSetAttribute(kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem))
      return rc;
    smem_set = (int)smem;
  }
  int occ = 0;
  if (const int rc = (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kptr, mega::kThreads, smem)) return rc;
  if (occ <= 0) return (int)cudaErrorInvalidValue;
  const int occ_cap = want2 ? 2 : 1;
  if (occ > occ_cap) occ = occ_cap;
  int grid = sms * occ;
  if (grid > acc) grid = acc;
  const mega::StagePlan sp{dep, count, start, ready, prof, wait_cyc, exec_cyc, dep2};
  if (want2)
    kern2<<<grid, mega::kThreads, smem, stream>>>(sp, p, MlpDenseDispatch{}, acc, S_COUNT, 0);
  else
    kern1<<<grid, mega::kThreads, smem, stream>>>(sp, p, MlpDenseDispatch{}, acc, S_COUNT, 0);
  return (int)cudaGetLastError();
}

}  // extern "C"


