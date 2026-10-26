// B1（kaon3-mega R36）：paged GQA decode attention 的 **persistent megakernel 版**（sm120 单卡）。
// 复用 `mega_common.cuh` 的 persistent 外壳 `mega::mega_kernel<Dispatch,OCC>`：一次 launch 常驻 SM，
// 持久 CTA 轮转领全局 task；每个 task = 一个 (b, hk, split)，task 体调 `gqa_v3::gqa_decode_v3_core`。
//
// 与现有 gqa_decode_v3.cu 的差别只在"怎么起 CTA"：
//   v3  : grid=(batch, Hkv, splits)，每 CTA 一个 (b,hk,split)，128 线程。
//   本版: 一次 launch grid = sms×occ 个持久 CTA（256 线程，kThreads），task=batch·Hkv·splits，靠 split 填 SM。
// **数值完全走同一个 core**（gqa_decode_v3_core），故应与 v3 逐位一致。core 是 128 线程算法：256 线程外壳下
// 只 tid<128 干活、其余陪跑 barrier（core 内 `active` 门控）。**代价**：attention stage 半数 lane 空转
//   —— B2 可考虑一 CTA 打包 2 个 shard（两个 128 线程半区各算一个），本步先不做。
//
// 单 stage（只有 attention 一个阶段，无依赖）：dep[0]=-1，count[0]=total_tasks。merge 仍单独 kernel（本步不进 persistent）。
// smem：core 用静态 __shared__（D=256/G=6 ≈32KB、D=128/G=4 ≈24KB，均 < 48KB 默认），动态 smem 传 0。
#include <cstdint>
#include <cuda_runtime.h>

#include "common.cuh"
#include "kernels.h"
#include "gqa_decode_v3_core.cuh"
#include "mega_common.cuh"

namespace dinfer {
namespace mega_attn {

using mega::kThreads;

/// host 填的一次 decode 步的 attention 全部指针/形状（单卡）。
struct Params {
  const bf16* q;
  const bf16* cache;
  const int32_t* block_table;
  const int32_t* seq_lens;
  const int32_t* cu_seqlens_q;  // 可为 nullptr（纯 decode）
  float* scratch;               // [b][H][splits][D+2]（splits>1）
  bf16* out_direct;             // splits==1 时非空，直接写输出
  int H, Hkv, block_size, max_blocks;
  float scale;
  int q_stride, splits, keys_per_split;
};

/// Dispatch：模板按 (GROUP, D) 实例化，task 体解出 (b, hk, split) 调 core。
template <int GROUP, int D>
struct AttnDispatch {
  using Params = mega_attn::Params;
  __device__ void run(Params& p, int shard, int /*stage*/, uint8_t* /*dyn_smem*/) const {
    // shard = (b*Hkv + hk)*splits + split（与 host 的 count/start 展平一致）
    const int split = shard % p.splits;
    const int bh = shard / p.splits;
    const int hk = bh % p.Hkv;
    const int b = bh / p.Hkv;
    gqa_v3::gqa_decode_v3_core<GROUP, D>(p.q, p.cache, p.block_table, p.seq_lens, p.cu_seqlens_q, p.scratch,
                                         p.out_direct, p.H, p.Hkv, p.block_size, p.max_blocks, p.scale, p.q_stride,
                                         p.splits, p.keys_per_split, b, hk, split, (int)threadIdx.x);
  }
};

}  // namespace mega_attn
}  // namespace dinfer

// ---------------------------------------------------------------- host launch（C ABI）
extern "C" {

/// persistent 版 decode attention（单 stage，B1）。dep/count/start/ready 为**设备指针**（host 本地算后 memcpy）。
/// 行为与 gqa_decode_v3_bf16 等价，只是一次 launch 的持久 CTA 调度。返回 cudaError_t 数值（0=成功）。
int mega_attn_decode(const void* q, const void* cache, const int32_t* block_table, const int32_t* seq_lens,
                     const int32_t* cu_seqlens_q, float* scratch, void* out_direct, int num_reqs, int heads,
                     int kv_heads, int head_dim, int block_size, int max_blocks, float scale, int q_stride, int splits,
                     int keys_per_split, int* dep, int* count, int* start, int* ready, unsigned long long* wait_cyc,
                     unsigned long long* exec_cyc, int prof, cudaStream_t stream) {
  using namespace dinfer;
  using namespace dinfer::mega_attn;
  const int group = heads / kv_heads;
  if (head_dim != 128 && head_dim != 256) return (int)cudaErrorInvalidValue;
  if (num_reqs <= 0 || kv_heads <= 0 || splits <= 0) return (int)cudaErrorInvalidValue;

  // 单 stage 依赖表（host 本地算 → memcpy 到设备）。
  const int total_tasks = num_reqs * kv_heads * splits;
  int h_dep = -1, h_count = total_tasks, h_start = 0;
  if (cudaMemcpyAsync(dep, &h_dep, sizeof(int), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemcpyAsync(count, &h_count, sizeof(int), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemcpyAsync(start, &h_start, sizeof(int), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemsetAsync(ready, 0, sizeof(int), stream) != cudaSuccess)
    return (int)cudaGetLastError();

  Params p{};
  p.q = static_cast<const bf16*>(q);
  p.cache = static_cast<const bf16*>(cache);
  p.block_table = block_table;
  p.seq_lens = seq_lens;
  p.cu_seqlens_q = cu_seqlens_q;
  p.scratch = scratch;
  p.out_direct = static_cast<bf16*>(out_direct);
  p.H = heads;
  p.Hkv = kv_heads;
  p.block_size = block_size;
  p.max_blocks = max_blocks;
  p.scale = scale;
  p.q_stride = q_stride > 0 ? q_stride : heads * head_dim;
  p.splits = splits;
  p.keys_per_split = keys_per_split;

  int dev = 0, sms = 1;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);

  const mega::StagePlan sp{dep, count, start, ready, prof, wait_cyc, exec_cyc, nullptr};

  // (group, D) → 实例化 mega_kernel<AttnDispatch<G,D>,1>，查 occupancy、起 sms×occ 个持久 CTA。
#define ATTN_LAUNCH(G, DD)                                                                                          \
  do {                                                                                                             \
    auto kern = mega::mega_kernel<mega_attn::AttnDispatch<G, DD>, 1>;                                              \
    int occ = 0;                                                                                                   \
    if (const int rc = (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*)kern, kThreads, 0))  \
      return rc;                                                                                                   \
    if (occ <= 0) return (int)cudaErrorInvalidValue;                                                              \
    int grid = sms * occ;                                                                                          \
    if (grid > total_tasks) grid = total_tasks;                                                                    \
    kern<<<grid, kThreads, 0, stream>>>(sp, p, mega_attn::AttnDispatch<G, DD>{}, total_tasks, 1, 0);              \
  } while (0)

  if (head_dim == 256) {
    switch (group) {
      case 1: ATTN_LAUNCH(1, 256); break;
      case 2: ATTN_LAUNCH(2, 256); break;
      case 4: ATTN_LAUNCH(4, 256); break;
      case 6: ATTN_LAUNCH(6, 256); break;
      default: return (int)cudaErrorInvalidValue;
    }
  } else {
    switch (group) {
      case 1: ATTN_LAUNCH(1, 128); break;
      case 2: ATTN_LAUNCH(2, 128); break;
      case 4: ATTN_LAUNCH(4, 128); break;
      default: return (int)cudaErrorInvalidValue;
    }
  }
#undef ATTN_LAUNCH
  return (int)cudaGetLastError();
}

}  // extern "C"
