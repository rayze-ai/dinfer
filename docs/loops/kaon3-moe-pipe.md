# loop: kaon3-moe-pipe —— MoE grouped NVFP4 GEMM 主循环 cp.async 软件流水

> 第一刀（R71/R72 定位）：`moe_grouped_sm120.cu` 的 `grouped_gemm_kernel` K 循环是
> `load sA/sfa/sB/sfb → __syncthreads → mma → __syncthreads` 纯串行、无 cp.async、无多 stage。
> 实测大 M（每专家 M≈132）下仍只 68 TFLOP/s = NVFP4 峰值 502 的 **13%**，根因 = load 停等 mma（无重叠）。
> vLLM 走 TRT-LLM fused MoE（flashinfer，日志 `trtllm::fused_moe::gemm2`+autotune），主循环 TMA/cp.async 多 stage 流水。

## 目标
`gemm3` 有效带宽从 ~300GB/s 往峰值（1.7TB/s）走，gemm3 从 13% 峰值往上；**M 无关**（decode+prefill 通吃）。
不动数值：cp.async 只改 global→smem 拷贝的时机，不改落到 smem 的字节与 mma 归约序 → **逐位一致是硬门槛**。

## 环境
- 5090（sm_120）容器编译（本机 nvcc 12.4 编不了 sm_120a）。本地写 → `put` 单文件 → 容器 `nvcc`/`cargo`。
- kernel：`crates/kernels/cuda/moe_grouped_sm120.cu`；FFI `crates/kernels/src/ffi.rs`；调用 `crates/kernels/src/ops.rs` + `models/kaon3/src/layer.rs` moe_branch。

## 做法（分步，每步过门禁再下一步）
1. **保留旧 kernel 当 oracle**：`grouped_gemm_kernel<BK>` 不删，作为逐位基准。
2. **加流水版** `grouped_gemm_kernel_pipe<BK,STAGES>`：双/三缓冲 smem，`cp.async` 预取下一 K 块 +
   `cp.async.commit_group`/`wait_group` 错位，当前块跑 mma。smem 布局 / mma 读法 / 归约序**与旧版逐字相同**。
   - 越界行（m_blk+r≥M / n_blk+r≥N）默认填值（sA=0x11、sB=0、sfa=0x38383838、sfb=0）：
     边界块少，这些行**不走 cp.async，直接 smem 写默认**（cp.async 后、wait_group 前）。
   - alpha/epilogue/量化 kernel 全不动。
3. **env 开关选版本**：`MOE_GEMM_PIPE=1` 走流水版，默认旧版（serve/对拍可 A/B）。

## 停止条件（机器可判定）
- ① **逐位门禁**：新测 `moe_pipe_parity`（self-contained，随机 A/B/scale/offsets，跑旧版+流水版，断言
  **逐字节相等 bad=0**）。`scripts/verify/kaon3-moe-pipe.sh` exit 0 = 过。**不过不接 serve**。
- ② gemm3 墙钟降：隔离微基准（或 serve KAON3_MOE_TIME）gemm3 从 ~0.096ms（decode）/~2.96ms（大M prefill）下降。
- ③ 端到端：serve bench（绑核 60-91，seed42/全共享1024/128out/rate8）吞吐升、进 CSV 趋势（CHANGE）。

## 风险
- cp.async 双缓冲同 buffer 读写竞态（compute(kb) 读 buf 与 load(kb+1) 写同 buf）：需 wait_group + __syncthreads 顺序正确。
- NVFP4 scale（sfa/sfb）也要流水预取，布局别错位。
- cp.async 要求 16B 对齐 src（旧版已 uint4=16B 拷贝，天然对齐）；越界行默认值不能用 cp.async 的 zfill（0x11/0x38 非 0）。
- 逐位是硬门槛：流水只改时机不改值，若 bad≠0 说明顺序/布局引入了差异，回退重查。

## 迭代上限 / 记录
每轮 put→编→对拍→（过了）bench 进 CSV，写进 loop 末尾 R 编号续 `kaon3-fp8gemm.md`（同一性能线）。
