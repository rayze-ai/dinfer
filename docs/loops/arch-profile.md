# 多卡型兼容：一套接口、多套实现、按卡选 + 两卡门禁（B4 前置）

> 用户 2026-10-19："不能 4090 优、B300 差"；"同意一套接口多套实现，但 `#if __CUDA_ARCH__ >= 1000` 这类 case 一定要明确、可读性一定要好"。

## 可读性硬规则（写进 CLAUDE.md，子代理 brief 必带）

1. **分文件，不分段**：按 arch 的实现放独立 `.cu`（`gqa_sm8x.cu` / `gqa_sm100.cu`），C ABI 符号带后缀（`gqa_prefill_sm8x` / `gqa_prefill_sm100`），Rust 侧按 `ArchProfile` 选。**kernel 体内禁止 `#if __CUDA_ARCH__`**。
2. **能力名代替数字**：共享头里确需分叉的（smem 上限、cp.async.bulk / TMA、tcgen05、集群），只允许在 `crates/kernels/cuda/arch.cuh` 一处定义有名字的 `constexpr bool` / 常量，文件头放一张"arch → 能力"表；使用处 `if constexpr (kHasTma)`，不出现 `>= 1000` 这类裸数字。
3. **实现文件头三件事**：针对哪代卡（sm 范围）、与基线实现差在哪、为什么快；对拍方式（`ops_parity.sh` 两卡都跑）。
4. 访存型 kernel（norm / silu / 采样 / pack）各 arch 共用一份；只有 attention / GEMM 周边 / 通信这类算力或互联敏感的才分实现。新增 arch 实现必须附带两卡 bench：**基线卡零变化，目标卡有提升**。

## 目标

1. **fat binary**：`CUDA_ARCHS` 默认 `86,89,90,103`（build.rs），一个二进制两台机器都跑；远端 env.sh 不再单独设。
2. **`ArchProfile`**（`crates/kernels/src/arch.rs`，模型无关）：启动时由 `Device::compute_capability()` 得 `Sm8x | Sm90 | Sm100`，字段：attention 实现选择、prefill NBUF、decode splits 上限、sample block、是否启用 custom all-reduce 等。现在靠环境变量调的量全部变成 profile 默认值，环境变量只做覆盖并打 info 日志"覆盖了哪项"。
3. **`OpSet` 多实现**：`Sm8xOps`（现在的 DefaultOps 改名）/ `Sm100Ops`（先全部转发 sm8x，B4 逐个替换 attention）；`OPS=` 环境变量强制指定用于对拍。
4. **启动自调优 + 缓存**（vLLM 没有，§11 记"改进"）：decode split-K splits、prefill tile 等数值参数，首次在某卡启动时对 3 个代表形状各跑候选取最快，写 `$XDG_CACHE_HOME/<repo>/tune-<gpu>-<git短哈希>.json`；预算 ≤ 1 s，`TUNE=0` 关。
5. **两卡门禁 `scripts/verify/bench_gate.sh`**：对本轮改动在 4090 与 B300 各跑主线四组，与各自上一轮 CSV 比；**任一卡任一组 TPOT 退化 > 3% → exit 1**，打印两卡 Δ 表。CLAUDE.md "每轮性能改动必须进 CSV" 升级为"必须过两卡门禁"。
6. `ops_parity.sh` 加 `--gpu-tag`，两卡都跑，结果目录按卡分。

## 环境

4090（`scripts/dev/remote4090.sh`，GPU 3–7 可用）+ B300（`scripts/dev/remote_b300.sh`，GPU 0–6；7 是 vLLM 基线）。

## 观察方法

- `cargo build` 一次产物在两台机器 `q1.sh` / `q5_tp.sh` 都过。
- `OPS=sm8x` 与 `OPS=sm100` 在 B300 上输出一致（对拍口径同 q5_tp）。
- `bench_gate.sh` 两卡 Δ 表。

## 停止条件

`bench_gate.sh` exit 0 且 4090 四组与 bench23 持平 ±2%、B300 四组与 bench_b300_1 持平 ±2%（本 loop 只搭机制不调性能；B4 在此之上做 sm100 attention）。

## 迭代上限

6 轮。

## 记录

| 轮 | 改动 | 结果 |
|---|---|---|

## 状态

- 规格写好；等 B2 all-reduce 子代理收口后开工（它在改 kernels / comm，避免同时动 build.rs 与 opset）。
