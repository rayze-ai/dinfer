// 小消息 one-shot all-reduce（对齐 vLLM csrc/custom_all_reduce.cuh 的 cross_device_reduce_1stage，design §11 "采用"）。
//
// 协议：每 rank 一块 cuda-ipc 设备区 = Signal（start/end flag [kMaxBlocks][kMaxRanks]，128 B 对齐）+ 输入镜像区。
// 调用方先把本 rank 的输入 D2D 进自己的镜像区，再 launch：
//   ① barrier_at_start：每 block 对每个 peer 写 start flag，自旋等 world 个 peer 到齐（volatile；跨设备可见性靠流序 + 自旋）；
//   ② 每线程按**固定 rank 顺序** 0..world 累加 8 个 bf16 的 packed 读（所有 rank 相同顺序 → 逐位一致，TP 采样一致性依赖这点）；
//   ③ 写 result；④ barrier_at_end（final_sync：volatile；保证所有 peer 读完镜像区之后本 rank 才能再写镜像区）。
// flag 用单调递增的 _flag[block] 计数器：CUDA graph 重放无需重置。
//
// 融合版 ar_oneshot_add_rmsnorm：② 的和不落回，直接 res += sum，再对该行做 RMSNorm 写 normed（一 block 一行；
// hidden 用 grid-stride 行覆盖）。替代 "all-reduce + fused_add_rmsnorm" 两步，省一次 t×h 读写和一次 launch。
#include "common.cuh"
#include "kernels.h"

using namespace dinfer;

namespace {

constexpr int kMaxBlocks = 36;
constexpr int kMaxRanks = 8;
using FlagType = uint32_t;

struct Signal {
  alignas(128) FlagType start[kMaxBlocks][kMaxRanks];
  alignas(128) FlagType end[kMaxBlocks][kMaxRanks];
  alignas(128) FlagType _flag[kMaxBlocks];
};

struct __align__(16) RankData {
  const void* ptrs[kMaxRanks];
};

struct __align__(16) RankSignals {
  Signal* signals[kMaxRanks];
};

// 8 × bf16 = 16 B，ld.128 / st.128
struct __align__(16) P8 {
  bf16 v[8];
};

__device__ __forceinline__ void st_flag_volatile(FlagType* p, FlagType f) {
  asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(f), "l"(p));
}
__device__ __forceinline__ FlagType ld_flag_volatile(FlagType* p) {
  FlagType f;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(f) : "l"(p));
  return f;
}

template <int NG>
__device__ __forceinline__ void barrier_at_start(const RankSignals& sg, Signal* self_sg, int rank) {
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < NG) {
    FlagType* peer = &sg.signals[threadIdx.x]->start[blockIdx.x][rank];
    FlagType* self = &self_sg->start[blockIdx.x][threadIdx.x];
    st_flag_volatile(peer, flag);
    while (ld_flag_volatile(self) != flag) {
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

// final_sync 版本（one-shot 之后不再有跨设备读，volatile 即可）
template <int NG>
__device__ __forceinline__ void barrier_at_end(const RankSignals& sg, Signal* self_sg, int rank) {
  __syncthreads();
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < NG) {
    FlagType* peer = &sg.signals[threadIdx.x]->end[blockIdx.x][rank];
    FlagType* self = &self_sg->end[blockIdx.x][threadIdx.x];
    st_flag_volatile(peer, flag);
    while (ld_flag_volatile(self) != flag) {
    }
  }
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

// 固定 rank 顺序 0..NG 累加（fp32），一次 8 个元素。
template <int NG>
__device__ __forceinline__ void packed_reduce(const RankData& dp, int64_t idx, float acc[8]) {
#pragma unroll
  for (int i = 0; i < 8; i++) acc[i] = 0.f;
#pragma unroll
  for (int r = 0; r < NG; r++) {
    P8 p = reinterpret_cast<const P8*>(dp.ptrs[r])[idx];
#pragma unroll
    for (int i = 0; i < 8; i++) acc[i] += to_f32(p.v[i]);
  }
}

template <int NG>
__global__ void __launch_bounds__(512, 1)
    ar_oneshot_kernel(RankData* _dp, RankSignals sg, Signal* self_sg, bf16* __restrict__ out, int rank,
                      int64_t packs) {
  RankData dp = *_dp;
  barrier_at_start<NG>(sg, self_sg, rank);
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < packs;
       idx += (int64_t)gridDim.x * blockDim.x) {
    float acc[8];
    packed_reduce<NG>(dp, idx, acc);
    P8 o;
#pragma unroll
    for (int i = 0; i < 8; i++) o.v[i] = to_bf16(acc[i]);
    reinterpret_cast<P8*>(out)[idx] = o;
  }
  barrier_at_end<NG>(sg, self_sg, rank);
}

// 一 block 一行：res[row] += Σ_r part_r[row]; normed[row] = rmsnorm(res[row]) · w。
// 行按 gridDim.x 步进（grid ≤ kMaxBlocks）。BLOCK 线程，每线程 hidden/8/BLOCK 个 pack（hidden 1024 / BLOCK 128 → 1 pack）。
template <int NG, int BLOCK>
__global__ void __launch_bounds__(BLOCK, 1)
    ar_oneshot_add_rmsnorm_kernel(RankData* _dp, RankSignals sg, Signal* self_sg, bf16* __restrict__ res,
                                  const bf16* __restrict__ w, bf16* __restrict__ normed, int rank, int tokens,
                                  int hidden, float eps) {
  RankData dp = *_dp;
  const int packs_per_row = hidden >> 3;
  barrier_at_start<NG>(sg, self_sg, rank);
  for (int row = blockIdx.x; row < tokens; row += gridDim.x) {
    bf16* rr = res + (int64_t)row * hidden;
    bf16* nr = normed + (int64_t)row * hidden;
    const int64_t base = (int64_t)row * packs_per_row;
    float ss = 0.f;
    // 第一遍：和 + 残差，写回 res，累平方和
    for (int p = threadIdx.x; p < packs_per_row; p += BLOCK) {
      float acc[8];
      packed_reduce<NG>(dp, base + p, acc);
      P8 r = reinterpret_cast<const P8*>(rr)[p];
      P8 o;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        // 与 fused_add_rmsnorm 同序：先 bf16(sum)，再 + res，再 bf16（all-reduce 结果本来就是 bf16）
        float v = to_f32(to_bf16(acc[i])) + to_f32(r.v[i]);
        bf16 vb = to_bf16(v);
        o.v[i] = vb;
        float vf = to_f32(vb);
        ss += vf * vf;
      }
      reinterpret_cast<P8*>(rr)[p] = o;
    }
    ss = block_reduce_sum<BLOCK>(ss);
    const float inv = rsqrtf(ss / hidden + eps);
    // 第二遍：归一化 × w
    for (int p = threadIdx.x; p < packs_per_row; p += BLOCK) {
      P8 r = reinterpret_cast<const P8*>(rr)[p];
      P8 wv = reinterpret_cast<const P8*>(w)[p];
      P8 o;
#pragma unroll
      for (int i = 0; i < 8; i++) o.v[i] = to_bf16(to_f32(r.v[i]) * inv * to_f32(wv.v[i]));
      reinterpret_cast<P8*>(nr)[p] = o;
    }
    __syncthreads();  // block_reduce_sum 的 smem 在下一行复用前要同步
  }
  barrier_at_end<NG>(sg, self_sg, rank);
}

inline bool fill(RankSignals& sg, const void* const* signals, int world) {
  for (int i = 0; i < kMaxRanks; i++) sg.signals[i] = nullptr;
  for (int i = 0; i < world; i++) {
    if (!signals[i]) return false;
    sg.signals[i] = reinterpret_cast<Signal*>(const_cast<void*>(signals[i]));
  }
  return true;
}

template <int NG>
int launch_oneshot(void* rank_data, RankSignals sg, Signal* self, bf16* out, int rank, int64_t n,
                   cudaStream_t s) {
  const int64_t packs = n >> 3;
  const int threads = 512;
  int blocks = (int)((packs + threads - 1) / threads);
  if (blocks < 1) blocks = 1;
  if (blocks > kMaxBlocks) blocks = kMaxBlocks;
  ar_oneshot_kernel<NG><<<blocks, threads, 0, s>>>(reinterpret_cast<RankData*>(rank_data), sg, self, out, rank,
                                                    packs);
  return last_err();
}

template <int NG>
int launch_fused(void* rank_data, RankSignals sg, Signal* self, bf16* res, const bf16* w, bf16* normed, int rank,
                 int64_t tokens, int64_t hidden, float eps, cudaStream_t s) {
  int blocks = (int)tokens;
  if (blocks > kMaxBlocks) blocks = kMaxBlocks;
  const int packs = (int)(hidden >> 3);
  auto rd = reinterpret_cast<RankData*>(rank_data);
  if (packs <= 128) {
    ar_oneshot_add_rmsnorm_kernel<NG, 128>
        <<<blocks, 128, 0, s>>>(rd, sg, self, res, w, normed, rank, (int)tokens, (int)hidden, eps);
  } else if (packs <= 256) {
    ar_oneshot_add_rmsnorm_kernel<NG, 256>
        <<<blocks, 256, 0, s>>>(rd, sg, self, res, w, normed, rank, (int)tokens, (int)hidden, eps);
  } else {
    ar_oneshot_add_rmsnorm_kernel<NG, 512>
        <<<blocks, 512, 0, s>>>(rd, sg, self, res, w, normed, rank, (int)tokens, (int)hidden, eps);
  }
  return last_err();
}

}  // namespace

extern "C" int64_t ar_signal_bytes(void) { return (int64_t)sizeof(Signal); }
extern "C" int64_t ar_rank_data_bytes(void) { return (int64_t)sizeof(RankData); }

// rank_data：设备上的 RankData（ptrs[r] = rank r 的镜像区设备地址，本进程已 IPC 打开）；
// signals：host 数组，signals[r] = rank r 的 Signal 设备地址（本 rank 用本地地址）；n 必须是 8 的倍数。
extern "C" int ar_oneshot_bf16(void* rank_data, const void* const* signals, void* out, int rank, int world,
                               int64_t n, cudaStream_t stream) {
  if (n <= 0) return 0;
  if ((n & 7) != 0 || rank < 0 || rank >= world || world > kMaxRanks) return err(cudaErrorInvalidValue);
  RankSignals sg;
  if (!fill(sg, signals, world)) return err(cudaErrorInvalidValue);
  Signal* self = sg.signals[rank];
  auto o = static_cast<bf16*>(out);
  switch (world) {
    case 2:
      return launch_oneshot<2>(rank_data, sg, self, o, rank, n, stream);
    case 4:
      return launch_oneshot<4>(rank_data, sg, self, o, rank, n, stream);
    case 8:
      return launch_oneshot<8>(rank_data, sg, self, o, rank, n, stream);
    default:
      return err(cudaErrorInvalidValue);
  }
}

// res[t,:] += Σ_r part_r[t,:]; normed[t,:] = rmsnorm(res[t,:]) · w。hidden 必须是 8 的倍数且 ≤ 4096。
extern "C" int ar_oneshot_add_rmsnorm_bf16(void* rank_data, const void* const* signals, void* res, const void* w,
                                           void* normed, int rank, int world, int64_t tokens, int64_t hidden,
                                           float eps, cudaStream_t stream) {
  if (tokens <= 0) return 0;
  if ((hidden & 7) != 0 || hidden > 4096 || rank < 0 || rank >= world || world > kMaxRanks)
    return err(cudaErrorInvalidValue);
  RankSignals sg;
  if (!fill(sg, signals, world)) return err(cudaErrorInvalidValue);
  Signal* self = sg.signals[rank];
  auto r = static_cast<bf16*>(res);
  auto wp = static_cast<const bf16*>(w);
  auto nm = static_cast<bf16*>(normed);
  switch (world) {
    case 2:
      return launch_fused<2>(rank_data, sg, self, r, wp, nm, rank, tokens, hidden, eps, stream);
    case 4:
      return launch_fused<4>(rank_data, sg, self, r, wp, nm, rank, tokens, hidden, eps, stream);
    case 8:
      return launch_fused<8>(rank_data, sg, self, r, wp, nm, rank, tokens, hidden, eps, stream);
    default:
      return err(cudaErrorInvalidValue);
  }
}
