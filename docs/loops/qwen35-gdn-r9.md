# Qwen3.5 r9：GDN chunk prefill 大 T 伸缩（wy / ho / conv）

> 状态：**进行中（2026-10-20 开）**。来源 `docs/loops/engine-gap.md` 8c–8h：长 prompt TTFT 斜率 dinfer 51.5 vs vLLM ~39 µs/token，差 12.5，其中 GDN 三段约 8（fla 同形状 1 K 持平、8 K 只涨 2.4× 而我们线性 5.3×）。用户 2026-10-20 决定：**按当前方案优化（修自己的 kernel），实在不行再谈替代**。

## 环境

- 开发 / parity / 微基准：3090 本机（`cargo test -p dinfer-kernels --release --features ref-tests --test gdn_parity`，`gdn_chunk_bench --ignored`）。
- 验收：B300-50 单卡（C-6 结束后给 GPU 6；`CUDA_ARCHS=103`），sm_103。4090 回归门禁（`scripts/verify/bench_gate.sh`，任一卡退化 > 3% 不合入）。
- 文件归属：`crates/kernels/cuda/gdn_chunk.cu`、`crates/kernels/src/ops.rs` 的 `gdn_chunk_*`、`crates/kernels/src/arch.rs` 的 `gdn_chunk_*`、`crates/kernels/tests/gdn_parity.rs`。**不动 `models/qwen35`**（r8 在改 executor / gdn.rs / service.rs），kernel ABI 若必须改，加新入口不删旧。
- 对手口径：vLLM `third_party/flash_linear_attention` fla `chunk_gated_delta_rule`（Triton）：`solve_tril` 分块三角逆（16×16 子块逆 + 矩阵乘拼 64×64，全 tensor core）；`chunk_fwd_h` 只递推 h 并把每 chunk 的 h 写回全局；`chunk_fwd_o` 对所有 (chunk, head) 全并行算 o = q·h + A·v。B300 实测：1×1024 359 µs、1×3009 349、1×8192 853（含 l2norm）。

## 观察方法

1. `gdn_chunk_bench` **加形状** 1×4096、1×8192（单长序列是 ho 并行度最差情形，现有三形状都没覆盖），每轮记 prep / wy / ho / conv 各段 µs 进 `test/qwen35/<日期>/probe_gdn_r9_N/probe.csv`（group `gdn_scale`，side dinfer；fla 数字作 side vllm 对照），`python3 scripts/dev/probe_trend.py --root test/qwen35`。
2. nsys `gdn_chunk_bench`（`cuda_gpu_kern_sum`）看 wy / ho 的 min / max 与 SM 占用；B300 上再对 6 K 单发 prefill 做一次 nsys 分解（engine-gap 待办）。
3. 正确性：`gdn_parity` 14/14（3090 + B300），`scripts/verify/qwen35_golden.sh`（主会话在 B300 跑，子代理不跑）。

## 改法（按序，每步独立可测）

1. **ho 装载 cp.async 双缓冲**：算 chunk c 时预取 c+1 的 q / w / kg / Aq（`gdn_decode.cuh` 已有 `cp_async16_smem` / `cp_async_commit_g`，共享头抽到 `arch.cuh` 或新 `.cuh`）。预期 ho 每 chunk 10 → 6–7 µs。
2. **ho 拆两段**：`h-pass`（每 (seq, head, V 分片) 一 block 串行递推 h，只做 h = h·g + v'ᵀ·kg，每 chunk 的 h bf16 写回全局 `[chunks, HV, V, K]`）+ `o-pass`（grid (chunk, head)，全并行 o = (q·hᵀ)·dec + Aq·v'）。串行段算量减半，并行段吃满 SM。中间 h 缓冲走 `GdnChunk` 工作区（启动一次分配，不在 kernel 内算尺寸）。
3. **wy 三角逆分块**：64 步串行前代 → 4 个 16×16 子块各自求逆（迭代 / 小前代）+ 矩阵乘拼 64×64，全部 mma；去掉逐步 `__syncthreads`。
4. **conv1d 行数**：每 block 行数由 `ArchProfile` 给（按 T 与 SM 数算），目标 8×1024 222 → < 80 µs（60 MB / 8 TB/s ≈ 8 µs 下限）。
5. 若 1–4 后仍差：wy 的 (chunk, head) 一 block 4 warp 太重 → 升 8 warp 或拆 V 分片。

## 硬性规则

- kernel 内不写卡型 / 形状调优数值（BV、行数、warp 数、双缓冲深度 → `ArchProfile` + 自调优），`#if __CUDA_ARCH__` 禁止在 kernel 体内；按 arch 分文件（`*_sm8x.cu` / `*_sm100.cu`）只在确有分叉时。
- 旧 kernel 保留为回退（环境变量 `GDN_CHUNK_HO=legacy` 之类只做覆盖）。
- 每轮 commit + 记本页轮次表 + probe.csv。

## 停止条件（机器可判定）

- `gdn_parity` 14/14（3090 + B300 sm_103）；
- `gdn_chunk_bench`（B300）：**8×1024 全链 ≤ 800 µs**（现 1662）、**1×8192 wy+ho ≤ 1000 µs**（fla 853 含 l2norm；现估 ~2500）；
- 单发 6 K TTFT 斜率（probe ttft_len 复测）**≤ 45 µs/token**（现 51.5；GDN 8 µs/token 拿回大半）；
- 4090 `bench_gate.sh` 四组不退化 > 3%；B300 `bench_compare.sh` 四组不退化。
- 迭代上限：8 轮或 2 天；到上限未达标 → 回主会话与用户讨论替代方案（cuDNN SDPA 只对注意力；GDN 没有现成 C 库）。

## 轮次

| 轮 | 改动 | 3090 1×1024 / 8×1024 / 1×8192（wy / ho / conv µs） | B300 | parity | 结论 |
|---|---|---|---|---|---|
| 0 | 基线（第 5A 轮 mma 版）；bench 加 1×4096 / 1×8192 | 3090：1×1024 **542 / 381 / 74**，8×1024 **4219 / 2460 / 427**，1×8192 **4205 / 2961 / 424**（ho 取 bv32、conv 取各档最优；全链 956 / 6787 / 7352）；B300 1×1024 134 / 160 / 41，8×1024 912 / 664 / 222 | 全链 311 / 1662 | 3090 15/15 | 起点。3090 上 wy / ho 都 ~8× 线性涨（1 K → 8 K），conv 5.7×；ho 1 block/SM（smem 75 KB）× 4 warp 纯延迟链；wy 的 64 步前代（每步 2 次 `__syncthreads`）占大头。`probe_gdn_r9_0` |
