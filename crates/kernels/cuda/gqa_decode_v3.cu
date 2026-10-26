// decode attention v3（B4 第 4 轮，docs/loops/b4-sm100.md "decode v3 设计"）。参考 FlashInfer `decode.cuh` 的结构重写，
// 读我们自己的 paged 布局 `[slot][Hkv][2][D]`，不引库。
//
// 针对哪代卡：sm80+ 通用（cp.async 16 B）；B300 上 v2 只到 KV 带宽 1/4，是这版的直接动机。
// 与 v2（gqa.cu::gqa_decode_v2_kernel）差在哪、为什么快：
//   1. D/8 lane / key（D=128 → 16、D=256 → 32）、每 lane 8 维一次 16 B（v2：8 lane / key，每 lane 两次 16 B）；
//      GQA 组内 GROUP 个 q 头共享同一份 K/V 读，点积对每个头各做一次 LPK lane shuffle
//      （v2 对两个头串行、且每轮还做跨 key 槽的 max/sum 归约）。
//   2. K/V 经 cp.async 进 smem，PIPE 级流水：每级 = 一个 tile（128/LPK 个 key：D=128 → 8、D=256 → 4）的 K+V，`wait_group<PIPE-1>` 只等最老一级，
//      等待粒度从"等自己的读"变成"等整组"，在飞字节数 = PIPE × tile（v2 是每 lane 自己预取一轮）。
//   3. 每线程的 key 槽各自维护 (m, l, acc)，循环结束后用 smem 合并一次（v2 每轮都合）。
//   4. split 规划：grid.z 仍按 batch 固定（CUDA graph），kernel 内 eff = ceil(n / keys_per_split)，多余 block 退出；
//      partial 布局与 v2 相同（[b][h][splits][D+2]），复用 gqa.cu 的 merge。work-item 展平留给 ArchProfile 轮。
// 对拍：ops_parity.sh（GQA_DECODE_V3=1 时 gqa_decode / gqa_mixed 走本实现），两卡都跑。
// 运行时可调参数只从 host 传入（keys_per_split）；模板参数 GROUP / D 由 host 按 heads/kv_heads / head_dim 选：
// D=128 支持 GROUP 1/2/4，D=256 支持 GROUP 1/2/4/6（Qwen3.5：24 q 头 / 4 kv 头）。
#include <cstdlib>
#include "common.cuh"
#include "kernels.h"

using namespace dinfer;

#include "gqa_decode_v3_core.cuh"

using namespace dinfer::gqa_v3;  // V3_THREADS / V3_LPK / gqa_decode_v3_core 等（B1 抽到 gqa_decode_v3_core.cuh）

namespace {

// 原入口：grid=(batch,Hkv,split)，128 线程，薄壳调 core（active 恒真 → 行为不变）。
template <int GROUP, int D>
__global__ void __launch_bounds__(V3_THREADS) gqa_decode_v3_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ cache, const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens, const int32_t* __restrict__ cu_seqlens_q, float* __restrict__ scratch,
    bf16* __restrict__ out_direct, int H, int Hkv, int block_size, int max_blocks, float scale, int q_stride, int splits,
    int keys_per_split) {
  gqa_decode_v3_core<GROUP, D>(q, cache, block_table, seq_lens, cu_seqlens_q, scratch, out_direct, H, Hkv, block_size,
                               max_blocks, scale, q_stride, splits, keys_per_split, blockIdx.x, blockIdx.y, blockIdx.z,
                               threadIdx.x);
}

}  // namespace

extern "C" int gqa_decode_v3_bf16(const void* q, const void* cache, const int32_t* block_table, const int32_t* seq_lens,
                                  const int32_t* cu_seqlens_q, float* scratch, void* out_direct, int64_t num_reqs,
                                  int64_t heads, int64_t kv_heads, int64_t head_dim, int64_t block_size,
                                  int64_t max_blocks, float scale, int64_t q_stride, int64_t splits,
                                  int64_t keys_per_split, cudaStream_t stream) {
  const int group = (int)(heads / kv_heads);
  if (head_dim != 128 && head_dim != 256) return (int)cudaErrorInvalidValue;
  dim3 grid((unsigned)num_reqs, (unsigned)kv_heads, (unsigned)splits);
#define V3_LAUNCH(G, DD)                                                                                               \
  gqa_decode_v3_kernel<G, DD><<<grid, V3_THREADS, 0, stream>>>(                                                        \
      static_cast<const bf16*>(q), static_cast<const bf16*>(cache), block_table, seq_lens, cu_seqlens_q, scratch,     \
      static_cast<bf16*>(out_direct), (int)heads, (int)kv_heads, (int)block_size, (int)max_blocks, scale,             \
      (int)q_stride, (int)splits, (int)keys_per_split)
  if (head_dim == 256) {
    switch (group) {
      case 1: V3_LAUNCH(1, 256); break;
      case 2: V3_LAUNCH(2, 256); break;
      case 4: V3_LAUNCH(4, 256); break;
      case 6: V3_LAUNCH(6, 256); break;
      default: return (int)cudaErrorInvalidValue;
    }
  } else {
    switch (group) {
      case 1: V3_LAUNCH(1, 128); break;
      case 2: V3_LAUNCH(2, 128); break;
      case 4: V3_LAUNCH(4, 128); break;
      default: return (int)cudaErrorInvalidValue;
    }
  }
#undef V3_LAUNCH
  return (int)cudaGetLastError();
}
