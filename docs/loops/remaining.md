# 离目标还差什么（跨模型索引，2026-10-22 盘点）

> 这是**唯一的总入口**：`design.md §12` 是 2026-10-18 的旧盘点，已被本文件取代（§12 保留作历史）。
> 每条只写一行结论 + 指向细节规格；状态口径 `待做 / 进行中 / 代码已写未验证 / 已收口`。
> 收功时更新本文件，不要新建同类索引。

## 三个目标（用户口径，不要混用）

| 目标 | 判定 | 当前 |
|---|---|---|
| **引擎层面超越 vLLM**（最终判据，`requirements.md` 验收 v2） | 同 SLO 下 goodput 最大 req/s ≥ vLLM，p99 曲线不劣，1 h 长跑无漂移 | **最大缺口**：最近一次 head-to-head `engine_2` dinfer 4 vs vLLM 12 req/s（0.33×） |
| Qwen3.5 全面 1.2× | 四组闭环 ≥ 1.2×（现 1.10/1.10/1.05/1.04×） | 门禁 1.0× 上下，未到 1.2 |
| Kaon-V3 5090 单卡 ≥ 基线 1.2× | `kaon_v3.sh` exit 0 + out tok/s 与 TTFT 双 ≥ 1.2× | 阶段 3（数值 parity）未过，性能一个数都没测 |

## A. Kaon-V3（当前主线，5090）

| # | 事项 | 状态 | 规格 |
|---|---|---|---|
| K1 | **parity 第 8 轮收尾**：RoPE 表已退回 f32（`86ec5e9`），**embed normalizer 仍 bf16 未验证**；同一快照两卡 A/B | 待做（P2） | `loops/kaon-v3.md` 末节计划 P1–P9 |
| K2 | **pre2 真 bug**：vLLM 强判 1852 @0.986，我们判 5976，BI 开 / 关都一样 → 与噪声无关 | 待做（定位中） | 同上 |
| K3 | **短 prompt 翻转率 / 共有 Δ 仍超下限**（25.25% / 1.80 vs 下限 13.95% / 1.19） | 待做 | 同上 |
| K4 | **判据对齐生产**：oracle 目标改 `oracle_nobi`（生产不开 BI），verify 仍比对 `oracle/` | 待做（脚本改一行 + 重算下限） | `loops/kaon3-parity.md` 末节 |
| K5 | **高效 D=512 attention**：full 层现在走 `gqa_prefill_ref_kernel` O(T²) 慢路径，只能当参照 | 待做 | `loops/kaon-v3.md` 阶段 3 续 |
| K6 | **decode 侧滑窗**（`gqa_decode_v3.cu`），滑窗只在 prefill 生效 | 待做 | 同上 |
| K7 | **NVFP4 GEMM 真实形状 141 TF/s vs 探针 330**：512×2816×2816 只有 704 block / 148 SM，需改 tile 形状或 split-K；剩余是两路动态 smem scale 读的寄存器压力（48 reg） | 待做 | `loops/kaon-v3.md` 阶段 3 续 |
| K8 | **阶段 4 服务**：serve / API / bench 全未接 Kaon | 待做 | `loops/kaon-v3.md` |
| K9 | 性能正式测（c1/c8/c32/6K TTFT 对基线 1.2×） | 阻塞于 K1–K4 | 同上 |

## B. Qwen3.5-27B（B300）

细节与逐条状态在 **`loops/qwen35-todo.md`**（A1–A5/B1–B6），此处只列对"离目标"影响最大的：

| # | 事项 | 状态 | 为什么算欠账 |
|---|---|---|---|
| B5 | GDN 状态按块缓存 → 重开 prefix cache | 待做 | 引擎口径最大单点：多轮共享前缀负载下我们 TTFT 落一档（cache hit 0.49 vs vLLM） |
| A3 | GDN 接 FlashInfer（AOT cubin + Rust 加载） | 进行中（轮 2 参数块对齐） | 3.8–4.9× 的确定收益 |
| A1 | attn 搬 tcgen05 | 进行中（里程碑 1 过） | 6 K 单层 1558 µs = 实测 mma.sync 峰值的 50% |
| A2 | GDN `ho` 搬 tcgen05 | **代码已写未验证**（数值未对拍） | 已投入，缺一次对拍收口 |
| A4 | 6 K TTFT 24.5 ms 差额归因 | 待做（vLLM 单请求 torch profiler） | 唯一可信硬数字的分段归属还空着 |
| B6 | 跨流 prefill/decode + 混批分流 | 待做 | c32 TPOT p99 的唯一手段 |
| B4 | executor 公共件抽取（async D2H / graph 桶 / 槽表 / plan 上传）+ async scheduling | 待做 | 用户 2026-10-20 定；现在 qwen35 内联，别的模型要复制 |

## C. 引擎层面（真正的"超越"判据，全部模型共用）

| # | 事项 | 状态 | 规格 |
|---|---|---|---|
| E1 | **goodput 差 3× 的归因**：`engine_2` 4 vs 12 req/s，三次误读后结论是"差在无 prefix cache 的 prefill 工作量"→ 按块缓存（= B5）是唯一已定位的抓手 | 待做 | `knowledge/perf-method.md`、`loops/engine-bench.md` |
| E2 | **1 h 长跑不漂移**（显存 / TPOT p99） | 未做过 | `requirements.md` 验收 v2-4 |
| E3 | **多卡 TP / PD 的 goodput 扩展效率** | 未做过（PD 只测过功能 + 单点 TPOT） | 验收 v2-5 |
| E4 | vLLM 侧也用同一套 engine 口径跑满（rate 4/8/12 已抓，需与 dinfer 同轮同 seed 对齐） | 部分 | `loops/engine-bench.md` |

## D. 工程 / 能力欠账（不分模型）

| # | 事项 | 状态 | 备注 |
|---|---|---|---|
| D1 | **`scripts/verify/bench_gate.sh` 不存在** | **待做** | CLAUDE.md 硬性规则要求"性能改动必须过两卡门禁"，脚本没有 → 规则目前无法执行 |
| D2 | `build.rs` 默认 arch 按本机卡探测 | 待做 | 现在默认 86,89,90，sm100 文件静默不编译，"编出来看着成功" |
| D3 | **`--features no-nvcc` 已坏**（stub 签名 24 处不符） | 待做 | 本地 CPU 编不了，无卡环境不能验 |
| D4 | 服务可生产：优雅停机 / API key / 限流 / 服务侧多卡 TP（现在 `--device` 单卡、无 NCCL）/ >8k 长上下文 / `/metrics` | 待做 | `design.md §12-#4` |
| D5 | API：`/v1/embeddings`、structured output（范围待用户定）、`tool_choice` 强制、`logit_bias`、投机解码抽到通用层、量化 | 待做 | `design.md §12-#5`、§9-3 |
| D6 | KV 分层：PinnedHostTier 默认开（需低水位 + 异步 fetch 收回 -5%）；RemoteTier（Mooncake/LMCache/NIXL） | 待做 | `design.md §12-#2` |
| D7 | PD 续：NIXL/CUDA IPC 传输后端、D 侧 unpack 移到 copy stream、多 P/D 路由与健康探测 | 待做 | `design.md §12-#1` |
| D8 | glm53 ActScratch（第五例 scratch 复用） | 待做 | `loops/qwen35-todo.md` B3 |
| D9 | 知识库欠账：`perf-method.md` 126 行超 120 上限，章节编号有两组重复 7/8/9；CLAUDE.md 引用了「§7」 | 待做 | 重排前先核所有 `§N` 引用 |
| D10 | K3 乱文根因（14–92 层 MLA / 93 层链式漂移） | 挂起（用户 2026-10-20） | `loops/k3-serve.md` 末行 |

## E. 待用户决策（阻塞上面多条）

- 目标生产机器（卡型 / 卡数 / 单机多机）→ 阻塞 D4 多卡、D5 部分项（§9-1）。
- structured output 是否在范围内（§9-3）→ 阻塞 D5。
- 基线配置：vLLM 侧开不开 MTP / EP / FP8 KV（§9-5）→ 影响所有对比口径。

## 建议顺序

1. **K1–K4 收口 Kaon parity**（同一条 loop，几天内可判定）→ 才有资格谈 K9 性能。
2. **B5（GDN 按块缓存）+ E1** —— 引擎口径最大单点，也是唯一已定位的 3× 缺口抓手。
3. **D1 + D3 两件小工程**（门禁脚本、no-nvcc），它们卡着"改动能不能合入"和"没卡能不能验"。
4. **A3 → A1 → A2** 收口 qwen35 的确定收益，再回 A4/A5 补归因。
5. **E2/E3 长跑 + 多卡口径**，最后才有"引擎层面超越"的判定数字。
