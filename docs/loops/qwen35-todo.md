# Qwen3.5 待办总表（2026-10-21 汇总）

> 本文件是**唯一索引**：所有进行中 / 待做的 qwen35 事项在这里列一行，细节留在各自的 loop 规格里。
> 状态口径：`待做` / `进行中` / `代码已写未验证` / `已收口`（收口的不再列，除非要回访）。
> 顺序 = 预期收益 ÷ 风险。

## 目标与现状

- 目标（用户 2026-10-20）：**qwen35 全面超过 vLLM 1.2×**；TTFT / TPOT 不得明显下降。
- 四组闭环（B300，2026-10-21 bench_merge_21）：**1.10 / 1.10 / 1.05 / 1.04×** —— 回归门禁，不是超越判据。
- 6 K 单发 TTFT：**dinfer 282.0 ms vs vLLM 257.5 ms**（差 9.5%）。这是唯一可信的端到端硬数字。
- **分段归属目前没有可信数据**：`qwen35-gemm-gap.md` 里那张 GDN/attn/elementwise/MLP 四行对照，
  四行中三行已被自己的复核推翻（MLP、attention、elementwise），只剩 GDN（101 vs 143 ms）成立。
  根因：从手工挑的 nsys 窗口里按 kernel 名求和，窗口一换答案就变。
  **重做方式**：vLLM 端改成 torch profiler + 单请求（`VLLM_TORCH_PROFILER_DIR`，一次只有 1 个请求在飞）；
  dinfer 端继续用 CUDA events（`LAYER_PROFILE=2` / `GDN_PROFILE=1` / `MLP_PROFILE=1`）。

## A. 性能（按收益排序）

| # | 事项 | 依据 | 状态 | 规格 |
|---|---|---|---|---|
| A1 | **attn 搬 tcgen05**（`gqa_fa_sm100.cu`），目标 T=6019 单层 ≤ 800 µs | attn 6 K 单层 1558 µs = 286 TF = mma.sync 实测峰值 575 的 50%；700 µs 目标物理不可达 | 进行中：工具链里程碑 1 完成（数值 rel 1.2e-4） | `qwen35-tcgen05.md` |
| A2 | **GDN `ho` 搬 tcgen05**（BV = V = 128） | ho 现 51 TF/s = 峰值 9%；51.6 GFLOP 按 30% 效率 = 77 µs | **代码已写未验证**（上一次卡在 token 过期；REG 168 / smem 172 KB，能跑无错，数值未对拍） | `qwen35-gdn-r11d.md` |
| A3 | **GDN 接 FlashInfer**（AOT cubin + Rust 加载，路线 A） | FlashInfer 49 / 156 / 253 µs vs dinfer 208 / 770 / 969 = **3.8–4.9×** | 进行中：轮 1 已落出独立 `.sm_103a.cubin`（235 KB）；轮 2 参数块对齐待做 | `qwen35-gdn-fi.md` |
| A4 | **6 K TTFT 差额归因**（dinfer 282 vs vLLM 257.5，24.5 ms） | 上面「分段归属」 | 待做：vLLM 单请求 torch profiler 重做 | `qwen35-gemm-gap.md` |
| A5 | **MLP 生产比 autotune 慢 29%**（1.899 vs 1.468 ms/层 = 23.7 ms） | `MLP_PROFILE=1` 实测 | 待定：归因为「权重 168 MB > L2 126 MB，autotune 反复读同一份权重」，属测量偏差；**若 A1–A3 做完仍有缺口，再回来查有没有第二解释** | `qwen35-gemm-gap.md` §真实原因 |

## B. 工程 / 正确性 / 能力

| # | 事项 | 状态 | 规格 |
|---|---|---|---|
| B1 | **4090 两卡门禁**（r11 合并后的自调优是否选了 `small_m`） | 待做 | `scripts/verify/bench_gate.sh` |
| B2 | **`build.rs` 默认 arch 按本机卡探测**（现在默认 86,89,90，导致 sm100 文件静默不编译，编出来的二进制「看着成功」） | 待做（已加 `scripts/verify/remote_sync.sh` 兜文件缺失，但没解决 arch 探测） | — |
| B3 | **glm53 ActScratch**（第五例：模型级 scratch 复用） | 待做 | `qwen35-gdn-r10.md` 系列同型 |
| B4 | **executor 公共件抽取 + async scheduling**：async D2H 两槽胶水 / CUDA graph 桶捕获 / 状态槽表 / plan 上传，从 qwen35 内联抽到 `crates/core` | 待做（perf 第 8 轮，用户 2026-10-20 定） | `qwen35-perf.md` 第 8 轮 |
| B5 | **GDN 状态按块缓存**（vLLM `mamba_cache_mode=align`）→ 重开 prefix cache，c8 / c8-long TTFT 回命中档 | 待做 | `qwen35-perf.md` 第 4 轮 |
| B6 | **跨流 prefill/decode**（`forward_sample_pair`）+ 混批分流 → c32 TPOT p99 | 待做 | `qwen35-perf.md` 第 3 轮 |

## C. 纪律（每次都要做）

- 每轮性能改动：**干净代码编译** → B300 实测 → 有提升才 commit；无提升不 commit，写进 loop 记录。
- 每轮改动进 `test/qwen35/perf-trend.csv`（`bench_compare.sh`），否则视为未完成。
- 性能改动必须过两卡门禁（任一卡退化 > 3% 不合入）。
- 远端代码齐备校验：`B300_TAG=50 scripts/verify/remote_sync.sh`（缺文件会 exit 1）。
- 子代理禁止 `remote_*.sh push`；只 `put` 单文件。

## 已收口（不再列细节，需要时回访规格）

`qwen35-attn.md`（里程碑 3）、`qwen35-attn-r11a.md`（v6 cfg 3464）、`qwen35-gdn-r9.md`、`qwen35-gdn-r10.md`（8 轮）、
`qwen35-decode-r11b.md`（GEMM 路线物理关闭）、`qwen35-model.md`。
