// kaon3-mega B2：整层 persistent megakernel（attn 段 + dense 段 融一次 launch，sm120 单卡）。
//
// 把下面这条链编进**一个** persistent megakernel（一次 launch），中间残差流用 scratch global buffer 层内衔接：
//   input_norm → qkv(fp8 GEMM) → q/k/v_norm(逐头) → rope → kv_cache写 → attn(复用 B1 core)
//     → o_quant → o_proj(fp8) → +残差 → pre_ff_norm → gate‖up(fp8) → gelu·mul → down(fp8) → post_ff_norm_1
// （MoE 支路仍单独 launch，留 B3；不碰 cudagraph，留 B4。）
//
// 数值全部复用已验证的 device 函数：
//   - fp8 GEMM / rmsnorm / gelu·mul → `mega_mlp_core.cuh`（A 的 dense 段，R30 全配置逐位）
//   - attention flash-decode core → `gqa_decode_v3_core.cuh`（B1，8 配置逐位）
// 因此**融合不改数值**。对拍判据：同一份 kernel 代码，一次 launch(融合) vs 分 stage 多次 launch(分离) 逐位一致 0/N
//   （见 `mega_layer_parity.rs`）。这测的是"融合本身不破坏数值"；与 vLLM/eager 的 serve 级对拍留到接 serve 时。
//
// 适用层型：**sliding 层**（head_dim=256, nkv=8, heads=16 → GROUP=2, D=256，B1 core 支持的档）。
//   full 层 head_dim=512 超出 B1 core 的 D∈{128,256} —— 需扩 core，留后续（见交接）。
//
// 线程：mega 外壳固定 256 线程。GEMM/norm stage 用全 256；attention core 是 128 线程算法，
//   本步沿用 B1 的"256 外壳只 128 活"（半数空转）。一 CTA 打包 2 shard 的填满优化留作后续（见交接障碍①）。
#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "common.cuh"
#include "kernels.h"
#include "mega_common.cuh"
#include "mega_mlp_core.cuh"       // fp8_gemm_tile / rmsnorm_row_warp / gelu_mul_round2（dinfer::mega_mlp）
#include "gqa_decode_v3_core.cuh"  // gqa_decode_v3_core<GROUP,D>（dinfer::gqa_v3）

namespace dinfer {
namespace mega_layer {

using mega::kThreads;
using mega::kWarps;
using mega_mlp::fp8_gemm_tile;
using mega_mlp::gelu_mul_round2;
using mega_mlp::rmsnorm_row_warp;

// 15 个 stage（严格按数据依赖串）。
enum Stage : int {
  S_IN = 0,       // input_norm(rmsnorm x) → norm_bf，并量化 → norm_q（qkv 的 fp8 A）
  S_QKV = 1,      // 合并 qkv fp8 GEMM：norm_q[m,H] × qkv_w[ΣN,H] → qkv_buf[m,ΣN]
  S_QKNORM = 2,   // 逐头 rmsnorm：q(16 头,q_norm) / k(8 头,k_norm) / v(8 头,无权重)，原地改 qkv_buf 切片
  S_ROPE = 3,     // rope：q(16 头)+k(8 头)，就地
  S_KVWRITE = 4,  // 写 paged KV cache（可选 fp8 roundtrip q/k/v）
  S_ATTN = 5,     // gqa_decode_v3_core<GROUP,D>（splits=1，out_direct → attn_buf[m,heads*D]）
  S_OQUANT = 6,   // 量化 attn_buf → attn_q（o_proj 的 fp8 A）
  S_OPROJ = 7,    // o_proj fp8 GEMM：attn_q[m,heads*D] × o_w[H,heads*D] → o_buf[m,H]
  S_RES = 8,      // post_attn_norm(o_buf) + 残差：x2 = x + post_attn_norm(o)
  S_PRE_FF = 9,   // pre_ff_norm(x2) → norm1_bf，量化 → norm1_q（gate/up 的 fp8 A）
  S_GATE = 10,    // dense_gate fp8 GEMM → gate_buf
  S_UP = 11,      // dense_up   fp8 GEMM → up_buf
  S_ACT = 12,     // gelu(gate)·up（round2）+ 量化 → act_q
  S_DOWN = 13,    // dense_down fp8 GEMM → down_out
  S_POST = 14,    // post_ff_norm_1 → out
  S_COUNT = 15,
};

/// host 填的整层一次 decode 步全部指针/形状（单卡）。
struct Params {
  // --- 输入 / 输出 ---
  const bf16* x;    // [m, H] 残余流输入
  bf16* out;        // [m, H] 输出（dense 段 post_ff_norm_1；MoE 由后续现有逻辑合并）
  // --- 权重 ---
  const bf16* input_w;      // [H]
  const bf16* q_norm_w;     // [D]
  const bf16* k_norm_w;     // [D]
  const bf16* post_attn_w;  // [H]
  const bf16* pre_ff_w;     // [H]
  const bf16* post_ff1_w;   // [H]
  const uint8_t* qkv_w;     // fp8 [ΣN, H]
  float qkv_wscale, qkv_a_inv;
  const uint8_t* o_w;       // fp8 [H, heads*D]
  float o_wscale, o_a_inv;
  const uint8_t* gate_w;    // fp8 [I, H]
  float gate_wscale, gate_a_inv;
  const uint8_t* up_w;      // fp8 [I, H]
  float up_wscale, up_a_inv;
  const uint8_t* dn_w;      // fp8 [H, I]
  float dn_wscale, dn_a_inv;
  // --- 中间缓冲（host 预分配，层内复用；"scratch global 衔接"）---
  bf16* norm_bf;    // [m, H]
  uint8_t* norm_q;  // [m, H]
  bf16* qkv_buf;    // [m, ΣN]
  bf16* attn_buf;   // [m, heads*D]
  uint8_t* attn_q;  // [m, heads*D]
  bf16* o_buf;      // [m, H]
  bf16* x2;         // [m, H]  x + post_attn_norm(o)
  bf16* norm1_bf;   // [m, H]
  uint8_t* norm1_q; // [m, H]
  bf16* gate_buf;   // [m, I]
  bf16* up_buf;     // [m, I]
  uint8_t* act_q;   // [m, I]
  bf16* down_out;   // [m, H]
  bf16* act_scratch;  // [m, I]  gelu·mul 的 bf16 中间（给 round2 用）
  // --- attention paged KV ---
  bf16* kv_cache;             // [slot][Hkv][2][D]
  const int32_t* block_table; // [m, max_blocks]
  const int32_t* seq_lens;    // [m]
  const int32_t* slot_mapping; // [m]
  float* attn_scratch;        // gqa scratch（splits>1 用；splits=1 走 out_direct，可空）
  // --- rope ---
  const float* cos_sin;       // [max_pos, rope_dim]
  const int32_t* positions;   // [m]
  int rope_dim, rope_span, rope_neox;
  // --- 形状 ---
  int m, H, I;
  int heads, nkv, D;  // sliding: 16 / 8 / 256
  int block_size, max_blocks;
  float eps, attn_scale;
  int kv_fp8;         // 1 → attention 读量化后的 q/k/v（vLLM kv_cache_dtype=fp8_e4m3）
  int splits, keys_per_split;
  // --- split-K 份数（每 GEMM 各一套）---
  int sp_qkv, sp_o, sp_gu, sp_dn;
  float *part_qkv, *part_o, *part_gate, *part_up, *part_dn;
  int *sem_qkv, *sem_o, *sem_gate, *sem_up, *sem_dn;
  // 片上流水：1 → S_QKNORM 吸收 rope + kv 写（norm→rope→roundtrip→写 qkv_buf/cache 全程 head 留 smem，
  //   不落 global 中间量），S_ROPE/S_KVWRITE 置 0 task、S_ATTN 直接依赖 S_QKNORM。0 → 原分离三 stage（基线）。
  int fuse_glue;
};

// ---------------------------------------------------------------- 小工具
/// e4m3 量化：v = x * a_inv（round-to-nearest-even + 饱和），与 dense 段 S_NORM 的量化口径一致。
__device__ __forceinline__ uint8_t quant_e4m3(float x, float a_inv) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x * a_inv, __NV_SATFINITE, __NV_E4M3));
}
/// fp8 e4m3 roundtrip（量化再反量化，scale=1）：与 ops::fp8_e4m3_roundtrip_bf16_n 同口径。
__device__ __forceinline__ bf16 fp8_roundtrip(bf16 v) {
  const float f = to_f32(v);
  const uint8_t q = static_cast<uint8_t>(__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3));
  __nv_fp8_e4m3 e;
  e.__x = q;
  return to_bf16(static_cast<float>(e));
}

/// 逐头 rmsnorm（一 warp 一行×头，D 维）：out = bf16(x·rsqrt(mean(x²)+eps)·w)；w==nullptr 则不乘。
/// 与 rmsnorm_row_warp 同口径，只是 H→D、权重可空。red[widx] 做跨 lane 归约暂存。
__device__ __forceinline__ void head_rmsnorm_warp(bf16* __restrict__ x, const bf16* __restrict__ w, int D, float eps,
                                                  int lane, float* red, int widx, bool live) {
  float ss = 0.f;
  for (int i = lane; i < D; i += 32) {
    const float v = to_f32(x[i]);
    ss += v * v;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
  const float inv = rsqrtf(ss / (float)D + eps);
  if (!live) return;
  for (int i = lane; i < D; i += 32) {
    const float o = to_f32(x[i]) * inv * (w ? to_f32(w[i]) : 1.f);
    x[i] = to_bf16(o);
  }
}

// ---------------------------------------------------------------- 片上融合 glue（QKNORM+ROPE+KVWRITE 一个 CTA）
// 一个 warp 处理一个 (row, 头槽)：把该头 D 维 load 进 smem → rmsnorm（q:q_norm / k:k_norm / v:无权重）
//   → rope（仅 q/k）→ fp8 roundtrip（kv_fp8 时，整头 D）→ 写回 qkv_buf（全部头槽）+ 写 KV cache（k/v 头）。
// 全程 head 留 smem，**不落 global 中间量**（省 QKNORM 写回 / ROPE 读写 / KVWRITE 从 qkv_buf 读的 global 往返）。
// 数值与"分离三 stage"**逐位一致**：同一套 rmsnorm（shfl 归约）/ rope（每步舍 bf16）/ roundtrip 口径，次序相同。
__device__ __forceinline__ void fused_qkv_glue(Params& p, int lane, int warp, uint8_t* smem) {
  const int sigmaN = p.heads * p.D + 2 * p.nkv * p.D;
  const int qN = p.heads * p.D, kN = p.nkv * p.D;
  const int slots_per_row = p.heads + 2 * p.nkv;
  const int slots = p.m * slots_per_row;
  // 每 warp 一段 smem 暂存一个头（D 个 bf16）。D=256 → 512 B/warp ×8 = 4KB（< GEMM 动态 smem，复用首段）。
  bf16* sh = reinterpret_cast<bf16*>(smem) + (long)warp * p.D;
  const int half = p.rope_dim / 2;
  for (int s = warp; s < ((slots + kWarps - 1) / kWarps) * kWarps; s += kWarps) {
    const bool live = s < slots;
    const int ss = live ? s : 0;
    const int r = ss / slots_per_row;
    const int hh = ss % slots_per_row;
    // 头槽 → qkv_buf 内的列基址 + norm 权重 + 是否 rope + k/v 类别（0=q,1=k,2=v）
    bf16* base;
    const bf16* w;
    int kvkind;   // 0 q, 1 k, 2 v
    int hk = 0;   // k/v 的 kv 头号
    if (hh < p.heads) { base = p.qkv_buf + (long)r * sigmaN + (long)hh * p.D; w = p.q_norm_w; kvkind = 0; }
    else if (hh < p.heads + p.nkv) { hk = hh - p.heads; base = p.qkv_buf + (long)r * sigmaN + qN + (long)hk * p.D; w = p.k_norm_w; kvkind = 1; }
    else { hk = hh - p.heads - p.nkv; base = p.qkv_buf + (long)r * sigmaN + qN + kN + (long)hk * p.D; w = nullptr; kvkind = 2; }
    const bool do_rope = live && (kvkind == 0 || kvkind == 1);
    // 1) load head → smem
    if (live) for (int i = lane; i < p.D; i += 32) sh[i] = base[i];
    __syncwarp();
    // 2) rmsnorm（就地改 smem）——与 head_rmsnorm_warp 同口径（f32 平方和、shfl 归约、末尾舍一次）
    float ss2 = 0.f;
    if (live) for (int i = lane; i < p.D; i += 32) { const float v = to_f32(sh[i]); ss2 += v * v; }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss2 += __shfl_xor_sync(0xffffffffu, ss2, off);
    const float inv = rsqrtf(ss2 / (float)p.D + p.eps);
    if (live) for (int i = lane; i < p.D; i += 32)
      sh[i] = to_bf16(to_f32(sh[i]) * inv * (w ? to_f32(w[i]) : 1.f));
    __syncwarp();
    // 3) rope（仅 q/k，就地改 smem）——NEOX 每步舍 bf16，与 rope.cu / S_ROPE 同口径
    if (do_rope) {
      const float* cs = p.cos_sin + (long)p.positions[r] * p.rope_dim;
      for (int i = lane; i < half; i += 32) {
        const float c = cs[i], sv = cs[half + i];
        int a, b;
        if (p.rope_neox) { a = i; b = i + p.rope_span; }
        else { a = 2 * i; b = 2 * i + 1; }
        const float x0 = to_f32(sh[a]), x1 = to_f32(sh[b]);
        const float q0 = to_f32(to_bf16(x0 * c)), q1 = to_f32(to_bf16(x1 * sv));
        const float q2 = to_f32(to_bf16(x1 * c)), q3 = to_f32(to_bf16(x0 * sv));
        sh[a] = to_bf16(q0 - q1);
        sh[b] = to_bf16(q2 + q3);
      }
      __syncwarp();
    }
    // 4) fp8 roundtrip（kv_fp8 时，整头 D；q/k/v 都做，与 S_KVWRITE 口径一致）
    if (live && p.kv_fp8) for (int i = lane; i < p.D; i += 32) sh[i] = fp8_roundtrip(sh[i]);
    __syncwarp();
    // 5) 写回 qkv_buf（所有头槽；attention 的 q 从这里读）+ 写 KV cache（k/v 头）
    if (live) {
      for (int i = lane; i < p.D; i += 32) base[i] = sh[i];
      if (kvkind != 0) {
        const int64_t slot = p.slot_mapping[r];
        bf16* dst = p.kv_cache + (slot * p.nkv * 2 + hk * 2 + (kvkind == 2 ? 1 : 0)) * p.D;
        for (int i = lane; i < p.D; i += 32) dst[i] = sh[i];
      }
    }
    __syncwarp();
  }
}

// ---------------------------------------------------------------- Dispatch
template <int GROUP, int DIM>
struct LayerDispatch {
  using Params = mega_layer::Params;
  static constexpr int kBN = 128;

  __device__ void run(Params& p, int shard, int stage, uint8_t* smem) const {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int sigmaN = p.heads * p.D + 2 * p.nkv * p.D;  // qkv 合并列数
    const int qN = p.heads * p.D, kN = p.nkv * p.D;      // q / k 列偏移
    switch (stage) {
      case S_IN: {
        // input_norm + 量化 → norm_q（qkv 的 fp8 A）。一 warp 一行。
        __shared__ float red[kWarps];
        const bool live = warp < p.m;
        const int row = live ? warp : 0;
        bf16* nb = p.norm_bf + (long)row * p.H;
        rmsnorm_row_warp(p.x + (long)row * p.H, p.input_w, nb, p.H, p.eps, lane, red, warp, live);
        if (live) {
          uint8_t* q = p.norm_q + (long)row * p.H;
          for (int i = lane; i < p.H; i += 32) q[i] = quant_e4m3(to_f32(nb[i]), p.qkv_a_inv);
        }
        return;
      }
      case S_QKV: {
        const int tile = shard / p.sp_qkv, split = shard - tile * p.sp_qkv;
        fp8_gemm_tile<kBN>(p.norm_q, p.qkv_w, p.qkv_buf, p.m, sigmaN, p.H, tile, p.qkv_wscale / p.qkv_a_inv, smem,
                           split, p.sp_qkv, p.part_qkv, p.sem_qkv);
        return;
      }
      case S_QKNORM: {
        if (p.fuse_glue) {  // 片上流水：吸收 rope + kv 写，不落 global 中间量（见 fused_qkv_glue）
          fused_qkv_glue(p, lane, warp, smem);
          return;
        }
        // 逐头 rmsnorm：行号 r∈[0,m)，头 h。q:16 头(q_norm)、k:8 头(k_norm)、v:8 头(无权重)。
        // 一个 warp 处理一个 (r, 头槽)。头槽总数 = m*(heads + 2*nkv)。8 warp 轮转。
        __shared__ float red[kWarps];
        const int slots = p.m * (p.heads + 2 * p.nkv);
        for (int s = warp; s < ((slots + kWarps - 1) / kWarps) * kWarps; s += kWarps) {
          const bool live = s < slots;
          const int ss = live ? s : 0;
          const int r = ss / (p.heads + 2 * p.nkv);
          const int hh = ss % (p.heads + 2 * p.nkv);
          bf16* base = p.qkv_buf + (long)r * sigmaN;
          bf16* x;
          const bf16* w;
          if (hh < p.heads) { x = base + (long)hh * p.D; w = p.q_norm_w; }
          else if (hh < p.heads + p.nkv) { x = base + qN + (long)(hh - p.heads) * p.D; w = p.k_norm_w; }
          else { x = base + qN + kN + (long)(hh - p.heads - p.nkv) * p.D; w = nullptr; }
          head_rmsnorm_warp(x, w, p.D, p.eps, lane, red, warp, live);
        }
        return;
      }
      case S_ROPE: {
        // rope：q(heads) + k(nkv)，就地。一个线程一个 (row, head, pair)。NEOX 口径同 rope.cu。
        const int half = p.rope_dim / 2;
        const int th = p.heads + p.nkv;  // 参与 rope 的头数（q+k）
        const long total = (long)p.m * th * half;
        for (long idx = tid; idx < total; idx += kThreads) {
          const int pr = idx / (th * half);
          const int rem = idx - (long)pr * th * half;
          const int h = rem / half;
          const int i = rem - h * half;
          bf16* base;
          if (h < p.heads) base = p.qkv_buf + (long)pr * sigmaN + (long)h * p.D;
          else base = p.qkv_buf + (long)pr * sigmaN + qN + (long)(h - p.heads) * p.D;
          const float* cs = p.cos_sin + (long)p.positions[pr] * p.rope_dim;
          const float c = cs[i], s = cs[half + i];
          int a, b;
          if (p.rope_neox) { a = i; b = i + p.rope_span; }
          else { a = 2 * i; b = 2 * i + 1; }
          const float x0 = to_f32(base[a]), x1 = to_f32(base[b]);
          const float p0 = to_f32(to_bf16(x0 * c)), p1 = to_f32(to_bf16(x1 * s));
          const float p2 = to_f32(to_bf16(x1 * c)), p3 = to_f32(to_bf16(x0 * s));
          base[a] = to_bf16(p0 - p1);
          base[b] = to_bf16(p2 + p3);
        }
        return;
      }
      case S_KVWRITE: {
        // 可选 fp8 roundtrip（q/k/v），再写 paged KV cache。cache 布局 [slot][Hkv][2][D]。
        // roundtrip 必须先于写 cache（attention 读的是量化后的 K/V；q 也量化，见 layer.rs §4a）。
        if (p.kv_fp8) {
          // q: heads*D；k/v: nkv*D 各一段。逐元素 roundtrip（就地）。
          const long nq = (long)p.m * p.heads * p.D;
          for (long i = tid; i < nq; i += kThreads) {
            const int r = i / (p.heads * p.D), c = i % (p.heads * p.D);
            bf16* e = p.qkv_buf + (long)r * sigmaN + c;
            *e = fp8_roundtrip(*e);
          }
          const long nkv = (long)p.m * p.nkv * p.D;
          for (long i = tid; i < nkv; i += kThreads) {
            const int r = i / (p.nkv * p.D), c = i % (p.nkv * p.D);
            bf16* ek = p.qkv_buf + (long)r * sigmaN + qN + c;
            bf16* ev = p.qkv_buf + (long)r * sigmaN + qN + kN + c;
            *ek = fp8_roundtrip(*ek);
            *ev = fp8_roundtrip(*ev);
          }
        }
        __syncthreads();
        // 写 cache：每 (row, kv头, 维) 写 K 和 V。slot = slot_mapping[row]。
        const long nkv = (long)p.m * p.nkv * p.D;
        for (long i = tid; i < nkv; i += kThreads) {
          const int r = i / (p.nkv * p.D);
          const int rem = i % (p.nkv * p.D);
          const int hk = rem / p.D, d = rem % p.D;
          const int64_t slot = p.slot_mapping[r];
          bf16* kdst = p.kv_cache + (slot * p.nkv * 2 + hk * 2) * p.D + d;
          bf16* vdst = p.kv_cache + (slot * p.nkv * 2 + hk * 2 + 1) * p.D + d;
          *kdst = p.qkv_buf[(long)r * sigmaN + qN + (long)hk * p.D + d];
          *vdst = p.qkv_buf[(long)r * sigmaN + qN + kN + (long)hk * p.D + d];
        }
        return;
      }
      case S_ATTN: {
        // shard = (b*nkv + hk)*splits + split（与 host count 展平一致）。q 从 qkv_buf 读（q_stride=sigmaN）。
        const int split = shard % p.splits;
        const int bh = shard / p.splits;
        const int hk = bh % p.nkv;
        const int b = bh / p.nkv;
        gqa_v3::gqa_decode_v3_core<GROUP, DIM>(
            p.qkv_buf, p.kv_cache, p.block_table, p.seq_lens, /*cu_seqlens_q=*/nullptr, p.attn_scratch, p.attn_buf,
            p.heads, p.nkv, p.block_size, p.max_blocks, p.attn_scale, sigmaN, p.splits, p.keys_per_split, b, hk, split,
            tid);
        return;
      }
      case S_OQUANT: {
        const long n = (long)p.m * p.heads * p.D;
        for (long i = tid; i < n; i += kThreads) p.attn_q[i] = quant_e4m3(to_f32(p.attn_buf[i]), p.o_a_inv);
        return;
      }
      case S_OPROJ: {
        const int tile = shard / p.sp_o, split = shard - tile * p.sp_o;
        fp8_gemm_tile<kBN>(p.attn_q, p.o_w, p.o_buf, p.m, p.H, p.heads * p.D, tile, p.o_wscale / p.o_a_inv, smem,
                           split, p.sp_o, p.part_o, p.sem_o);
        return;
      }
      case S_RES: {
        // x2 = x + post_attn_norm(o_buf)。一 warp 一行。
        __shared__ float red[kWarps];
        const bool live = warp < p.m;
        const int row = live ? warp : 0;
        bf16* x2 = p.x2 + (long)row * p.H;
        rmsnorm_row_warp(p.o_buf + (long)row * p.H, p.post_attn_w, x2, p.H, p.eps, lane, red, warp, live);
        if (live) {
          const bf16* xin = p.x + (long)row * p.H;
          for (int i = lane; i < p.H; i += 32) x2[i] = to_bf16(to_f32(x2[i]) + to_f32(xin[i]));
        }
        return;
      }
      case S_PRE_FF: {
        __shared__ float red[kWarps];
        const bool live = warp < p.m;
        const int row = live ? warp : 0;
        bf16* nb = p.norm1_bf + (long)row * p.H;
        rmsnorm_row_warp(p.x2 + (long)row * p.H, p.pre_ff_w, nb, p.H, p.eps, lane, red, warp, live);
        if (live) {
          uint8_t* q = p.norm1_q + (long)row * p.H;
          for (int i = lane; i < p.H; i += 32) q[i] = quant_e4m3(to_f32(nb[i]), p.gate_a_inv);
        }
        return;
      }
      case S_GATE: {
        const int tile = shard / p.sp_gu, split = shard - tile * p.sp_gu;
        fp8_gemm_tile<kBN>(p.norm1_q, p.gate_w, p.gate_buf, p.m, p.I, p.H, tile, p.gate_wscale / p.gate_a_inv, smem,
                           split, p.sp_gu, p.part_gate, p.sem_gate);
        return;
      }
      case S_UP: {
        const int tile = shard / p.sp_gu, split = shard - tile * p.sp_gu;
        fp8_gemm_tile<kBN>(p.norm1_q, p.up_w, p.up_buf, p.m, p.I, p.H, tile, p.up_wscale / p.up_a_inv, smem, split,
                           p.sp_gu, p.part_up, p.sem_up);
        return;
      }
      case S_ACT: {
        const long n = (long)p.m * p.I;
        for (long i = tid; i < n; i += kThreads) p.act_scratch[i] = gelu_mul_round2(p.gate_buf[i], p.up_buf[i]);
        __syncthreads();
        for (long i = tid; i < n; i += kThreads) p.act_q[i] = quant_e4m3(to_f32(p.act_scratch[i]), p.dn_a_inv);
        return;
      }
      case S_DOWN: {
        const int tile = shard / p.sp_dn, split = shard - tile * p.sp_dn;
        fp8_gemm_tile<kBN>(p.act_q, p.dn_w, p.down_out, p.m, p.H, p.I, tile, p.dn_wscale / p.dn_a_inv, smem, split,
                           p.sp_dn, p.part_dn, p.sem_dn);
        return;
      }
      case S_POST: {
        __shared__ float red[kWarps];
        const bool live = warp < p.m;
        const int row = live ? warp : 0;
        rmsnorm_row_warp(p.down_out + (long)row * p.H, p.post_ff1_w, p.out + (long)row * p.H, p.H, p.eps, lane, red,
                         warp, live);
        return;
      }
    }
  }
};

}  // namespace mega_layer
}  // namespace dinfer

// ---------------------------------------------------------------- host launch（C ABI）
extern "C" {

/// 整层 megakernel 一次（或分段多次）launch。
/// `stage_lo..stage_hi`（含）指定本次 launch 覆盖哪些 stage 的 task：
///   - 融合：stage_lo=0, stage_hi=14 → 一次 launch 跑全链。
///   - 分离（对拍/计时基线）：对每个 s 调一次 stage_lo=stage_hi=s（流序列化 + ready 累加 → 等价于分别 launch）。
/// `setup`（stage_lo==0 时由调用方置 1）：memcpy dep/count/start 表 + 清零 ready/sem。分段调用首个置 1，其余 0。
/// dep/count/start/ready 为**设备指针**（host 本地算后 memcpy）。返回 cudaError_t（0=成功）。
int mega_layer_forward(const dinfer::mega_layer::Params* p_in, int stage_lo, int stage_hi, int setup,
                       int* dep, int* count, int* start, int* ready, int* dep2,
                       unsigned long long* wait_cyc, unsigned long long* exec_cyc, int prof, cudaStream_t stream) {
  using namespace dinfer;
  using namespace dinfer::mega_layer;
  Params p = *p_in;
  if (p.m <= 0 || p.m > 8) return (int)cudaErrorInvalidValue;
  if (p.D != 256 || p.heads / p.nkv != 2) return (int)cudaErrorInvalidValue;  // 本步仅 sliding 档（GROUP=2,D=256）
  if (p.sp_qkv < 1) p.sp_qkv = 1;
  if (p.sp_o < 1) p.sp_o = 1;
  if (p.sp_gu < 1) p.sp_gu = 1;
  if (p.sp_dn < 1) p.sp_dn = 1;
  constexpr int kBN = LayerDispatch<2, 256>::kBN;
  if (p.splits < 1) p.splits = 1;

  const int sigmaN = p.heads * p.D + 2 * p.nkv * p.D;
  const int n_qkv = (sigmaN + kBN - 1) / kBN;
  const int n_o = (p.H + kBN - 1) / kBN;
  const int n_gu = (p.I + kBN - 1) / kBN;
  const int n_dn = (p.H + kBN - 1) / kBN;

  int h_count[S_COUNT], h_dep[S_COUNT], h_start[S_COUNT], h_dep2[S_COUNT];
  for (int i = 0; i < S_COUNT; ++i) h_dep2[i] = -1;
  h_count[S_IN] = 1;
  h_count[S_QKV] = n_qkv * p.sp_qkv;
  h_count[S_QKNORM] = 1;
  h_count[S_ROPE] = p.fuse_glue ? 0 : 1;      // 片上融合：rope 已吸进 QKNORM，置 0 task
  h_count[S_KVWRITE] = p.fuse_glue ? 0 : 1;   // 片上融合：kv 写已吸进 QKNORM，置 0 task
  h_count[S_ATTN] = p.m * p.nkv * p.splits;
  h_count[S_OQUANT] = 1;
  h_count[S_OPROJ] = n_o * p.sp_o;
  h_count[S_RES] = 1;
  h_count[S_PRE_FF] = 1;
  h_count[S_GATE] = n_gu * p.sp_gu;
  h_count[S_UP] = n_gu * p.sp_gu;
  h_count[S_ACT] = 1;
  h_count[S_DOWN] = n_dn * p.sp_dn;
  h_count[S_POST] = 1;
  h_dep[S_IN] = -1;
  h_dep[S_QKV] = S_IN;
  h_dep[S_QKNORM] = S_QKV;
  h_dep[S_ROPE] = S_QKNORM;
  h_dep[S_KVWRITE] = S_ROPE;
  h_dep[S_ATTN] = p.fuse_glue ? S_QKNORM : S_KVWRITE;  // 融合时 ROPE/KVWRITE count=0，ATTN 直接等 QKNORM
  h_dep[S_OQUANT] = S_ATTN;
  h_dep[S_OPROJ] = S_OQUANT;
  h_dep[S_RES] = S_OPROJ;
  h_dep[S_PRE_FF] = S_RES;
  h_dep[S_GATE] = S_PRE_FF;
  h_dep[S_UP] = S_PRE_FF;
  h_dep[S_ACT] = S_GATE;
  h_dep2[S_ACT] = S_UP;
  h_dep[S_DOWN] = S_ACT;
  h_dep[S_POST] = S_DOWN;
  int acc = 0;
  for (int s = 0; s < S_COUNT; ++s) { h_start[s] = acc; acc += h_count[s]; }

  if (setup) {
    if (cudaMemcpyAsync(count, h_count, sizeof(h_count), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(dep, h_dep, sizeof(h_dep), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(start, h_start, sizeof(h_start), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(dep2, h_dep2, sizeof(h_dep2), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemsetAsync(ready, 0, sizeof(int) * S_COUNT, stream) != cudaSuccess)
      return (int)cudaGetLastError();
    if (p.sp_qkv > 1 && cudaMemsetAsync(p.sem_qkv, 0, sizeof(int) * n_qkv, stream) != cudaSuccess)
      return (int)cudaGetLastError();
    if (p.sp_o > 1 && cudaMemsetAsync(p.sem_o, 0, sizeof(int) * n_o, stream) != cudaSuccess)
      return (int)cudaGetLastError();
    if (p.sp_gu > 1 && (cudaMemsetAsync(p.sem_gate, 0, sizeof(int) * n_gu, stream) != cudaSuccess ||
                        cudaMemsetAsync(p.sem_up, 0, sizeof(int) * n_gu, stream) != cudaSuccess))
      return (int)cudaGetLastError();
    if (p.sp_dn > 1 && cudaMemsetAsync(p.sem_dn, 0, sizeof(int) * n_dn, stream) != cudaSuccess)
      return (int)cudaGetLastError();
  }

  // 本次 launch 的 task 范围 [task_lo, task_hi)。
  const int task_lo = h_start[stage_lo];
  const int task_hi = h_start[stage_hi] + h_count[stage_hi];
  const int n_tasks = task_hi - task_lo;
  if (n_tasks <= 0) return (int)cudaErrorInvalidValue;

  // smem：GEMM tile 的 sA(16×RS)+sB(BN×RS)；RS=128+16。
  const size_t smem = (size_t)(16 + kBN) * (128 + 16);
  int dev = 0, sms = 1, max_smem = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  if ((int)smem > max_smem) return (int)cudaErrorInvalidValue;

  auto kern = mega::mega_kernel<LayerDispatch<2, 256>, 1>;
  static thread_local int smem_set = 0;
  if (smem_set < (int)smem) {
    if (const int rc = (int)cudaFuncSetAttribute((const void*)kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 (int)smem))
      return rc;
    smem_set = (int)smem;
  }
  int occ = 0;
  if (const int rc = (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*)kern, mega::kThreads, smem))
    return rc;
  if (occ <= 0) return (int)cudaErrorInvalidValue;
  int grid = sms * occ;
  if (grid > n_tasks) grid = n_tasks;

  const mega::StagePlan sp{dep, count, start, ready, prof, wait_cyc, exec_cyc, dep2};
  // 持久 CTA 从 task_lo 起领到 task_hi：mega_kernel 的循环是 task = blockIdx + start_offset; task < total_tasks。
  kern<<<grid, mega::kThreads, smem, stream>>>(sp, p, LayerDispatch<2, 256>{}, task_hi, S_COUNT, task_lo);
  return (int)cudaGetLastError();
}

}  // extern "C"
