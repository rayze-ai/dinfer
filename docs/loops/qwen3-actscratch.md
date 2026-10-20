# Qwen3 激活缓冲：每层一套 → 全模型一套（ActScratch）

> 状态：**进行中（2026-10-21 派出）**。同一 bug 的第四例（qwen35 `models/qwen35/src/scratch.rs`、kimik3 `models/kimik3/src/scratch.rs` 已改完；glm53 同构未改）。Qwen3-0.6B 单卡不疼，疼在**大预算 / 多卡**：`max_tokens = 16384` 时 28 层 × 每 token 30 KB ≈ **14 GB** 白占，把 KV pool 挤到起不来（pitfalls 2026-10-20 两条例）。

## 现状（代码事实）

- `models/qwen3/src/model.rs:175-176`：`Attention::load` 每层分配 `qkv_buf [t, (H+2Hkv)·D]` + `attn [t, H·D]`。
- `crates/layers/src/nn.rs:220-236`：`Mlp::load_tp(max_tokens > 0)` 每层分配 `gate_up [t, 2I]` + `act [t, I]`；**`max_tokens == 0` 时不自带**，走 `forward_with_scratch`（共用缓冲）——已有现成路径，qwen35 就是这么用的。
- 0.6B 单卡：`(H+2Hkv)·D = (16+16)·128 = 4096` 列、`H·D = 2048`、`2I = 6144`、`I = 3072`，合计 **15360 列 × 2 B = 30 KB/token/层**。
- 生命周期：层在同一条流上串行执行，层 i 的中间量在层 i+1 开始前已消费完，任何时刻只有一层在用 → 一套足够。

## 环境

- 改 / 编译 / 单测：3090 本机（`cargo build --release --features qwen3 -p dinfer`、`cargo test -p dinfer-qwen3`、`scripts/verify/q1.sh` 的 CPU 段）。
- GPU：**3090 GPU 0 与 r11b 的 parity 共卡** → 跑 GPU 前先 `nvidia-smi --query-compute-apps`，有他人进程就等；`CUDA_VISIBLE_DEVICES=0`。**不许上 B300**（卡归主会话调度）。
- 文件归属：`models/qwen3/src/{model.rs,scratch.rs(新),executor.rs}`、必要时 `crates/layers/src/nn.rs`（**只加注释 / 不改 ABI**）。
- **禁改**（r11b 工作区在改）：`crates/kernels/cuda/{gemm_small_m.cu,gqa.cu}`, `crates/kernels/src/{arch.rs,ops.rs,ffi.rs}`, `crates/kernels/cuda/kernels.h`。
- 提交纪律：只 `git add` 自己那几个文件 + `git commit`，**禁止 push**（主会话统一推）。

## 改法

1. 新建 `models/qwen3/src/scratch.rs::ActScratch { qkv, attn, gate_up, act, max_tokens }`，照 qwen35 的三件套：`cols(cfg)` 给列数、`bytes(cfg, max_tokens)` 能在分配前算（日志 / 预算用）、`new(stream, cfg, max_tokens)`。
   - `qkv` / `attn` 列宽用 **本 rank 的** cfg（TP 后头数已除），`gate_up` / `act` 用 `inter`（TP 后已除）。
2. `Attention::load` 去掉两块 `Tensor::zeros`（`max_tokens` 参数可保留但置 0 或删掉，调用点同步）；`forward` 收 `&mut ActScratch` 替代 `self.qkv_buf` / `self.attn`。注意 `forward` 里 `q_view` / `k_view` / `v_view` 是对 `qkv_buf` 的 slice，改动要保证行跨度 `qkv_w` = `(H+2Hkv)·D` 不变（共享缓冲是**更宽**的块，必须用 `ActScratch.qkv` 的列数当 stride，不能再用 `numel/h` 推）。
3. `Mlp::load_tp(..., 0, ...)` + `forward_with_scratch(..., &mut sc.gate_up, &mut sc.act)`；`DecoderLayer.mlp` 从 `&mut self` 调用改为 `&self`（`forward_with_scratch` 已 `&self`）。
4. `ModelScratch` / `LayerScratch` / `alloc_scratch` / `swap_scratch`：第二套（跨流 `XSTREAM`）就是一个 `ActScratch`，`LayerScratch` 整个删掉。`swap_scratch` 变成 `std::mem::swap(&mut self.act, &mut s.act)`。
5. 日志：现有 `load timing: per-layer activation buffers (Tensor::zeros inside Attention/Mlp::load)` 应归零 → 改成 `load timing: act scratch rows=… bytes=…`（启动一行）；`startup done` 字段不变。

## 观察方法

- 启动日志的 `num_blocks`（KV 池）与 `free`：同一 `--max-num-batched-tokens` 下应上升 ≈ 27 层份额。
- `zeros_mib` 从 ~14 GB 级降到 ~0.5 GB 级。
- `test/qwen3/<日期>/` 存两次启动日志摘要（改前 / 改后），结论先行。

## 停止条件（机器可判定）

1. `cargo test -p dinfer-qwen3` 全过（含 tiny golden 11 项对拍）。
2. `scripts/verify/q1.sh` exit 0（3090 CPU 段；GPU 段若卡被占可跳过并注明）。
3. 预算 16384 下 `served` 起得来且 `num_blocks` ≥ 改前（启动 3 次取中位）。
4. 3090 `bench_compare.sh` 四组不退化 > 3%（卡被 r11b 占着就交给主会话）。
5. 迭代上限 4 轮 / 半天；到上限回主会话。

## 轮次

| 轮 | 改动 | 结果 | 结论 |
|---|---|---|---|
| 1 | 新建 `models/qwen3/src/scratch.rs::ActScratch`；`Attention` 去掉 `qkv_buf`/`attn` 两个字段与 `swap_bufs`；`Attention::forward` 收 `&mut ActScratch`，GEMM 的 `ld` / `qkv_post` 行宽 / attention 的 `q_stride` 一律用本层 `qkv_w`（显式，不从 `numel` 反推），加 `numel ≥ t × 行宽` 断言；`Mlp::load_tp(..., 0, ...)` + `forward_with_scratch`；删 `LayerScratch`，`ModelScratch.act: ActScratch`，`swap_scratch` 变成一次 `mem::swap`；启动日志新增 `act_scratch_rows/bytes/mib` | `cargo build --release --features qwen3 -p dinfer` OK（49.9 s）；`cargo test --release -p dinfer-qwen3` 11 项全过（8 单测 + 1 guided + 2 lmhead_colsplit） | 编译与 CPU 侧通过；`compute-apps` 空，下一轮跑 q1 + 真权重启动对比 |
| 2 | 真权重启动 A/B（旧二进制 = `33528b9~1` 在 /tmp 工作树编出；同 `--max-num-batched-tokens 16384`、同模型、各 3 次） | **改前 3/3 起不来**：`zeros_mib=14013`（28 层 × 15360 列 × 2 B）、`num_blocks=3050` 之后就 `CUDA_ERROR_OUT_OF_MEMORY`（在 xstream 分配第二套 scratch 时炸）。**改后 3/3 起来**：`act_scratch_rows=16400`、`act_scratch_bytes=503808000`（480 MiB）、`num_blocks=10808`（3 次一致）、`startup done total_ms=1227` | 病根与规格一致：每层一套 13.7 GiB → 一套 0.47 GiB；KV 3050 → 10808 块（差 3.5×，改前那次还是在 OOM 之前的读数）。**规格里的停止条件 3 已满足** |
