# loop 规格：kaon3-parity

- 状态：进行中
- 创建：2026-10-22
- 验证脚本：`scripts/verify/kaon3-parity.sh`
- 迭代记录：`test/kaon3/2026-10-22/kaon3-parity.md`
- 上游：`docs/loops/kaon-v3.md`（阶段 3）。用户 2026-10-22：Rust 实现、Rust 验证，不用 Python 参考；loop 不达标不停。

## 目标

`models/kaon3` 的 Rust 前向（5090 单卡，真实 checkpoint，不反量化权重）在固定 prompt 集上与 vLLM 逐位置 logprob 一致。

## 环境

- 机器：5090（`scripts/dev/remote_5090.sh`，ControlMaster；断了要新 token）。编译 / 运行都在基线镜像容器里：
  `GPU=0 scripts/dev/remote_5090.sh ctr '<cmd>'`（cwd=`/w/dinfer`）。同步：`scripts/dev/remote_5090.sh sync`（无 `--delete`）。
- **oracle = 私有 vLLM 副本**：`kaon-oracle` 容器，GPU 1，`:28100`，与基线同镜像同参数（`VLLM_BATCH_INVARIANT=1`、
  block 32、max-model-len 16384），只去掉 PegaKV 连接器。**实验性请求一律打私有副本；共享基线 `:28067`（GPU 7）只做短 prompt
  交叉核对**——2026-10-22 长 prompt 的 `prompt_logprobs` 打崩过基线两次（见 pitfalls）。
  重启私有副本：见 `docs/loops/kaon-v3.md` 本日节的 `docker run` 命令。
- oracle 文件：`scripts/dev/kaon3_oracle.sh <case> ids|chat|file …` → `test/kaon3/oracle/<case>.{ids,json}`（入库）。
- 用例：`bos1`（T=1，隔离 attention / RoPE）、`raw9`（T=9 裸 id）、`chat_en` / `chat_zh`（chat 模板）、
  `long_zh`（>1024 token，滑窗；待私有副本就绪后抓）。

## 观察方法

| 看什么 | 在哪里 | 命令 |
|---|---|---|
| 逐行比对 | 探针输出 `row N: oracle top1 … ours top1 … Δtop1` + `worst` 5 行 | `bash scripts/verify/kaon3-parity.sh [case]` |
| 判定 | `PARITY PASS/FAIL` 一行（可判定行 top1 不一致数 / top5 Δ 均值 / top1 Δ 最大 / 实际 token Δ） | 同上 |
| 中间量 | `--dump-layer N`（第 N 层残差流前 8 个 + 范数） | `ctr './target/release/kaon3_probe $M --forward --dump-layer N <ids>'` |

## 停止条件（必须可机器判定）

**2026-10-22 修订**（原判据"可判定行 top1 零翻转 + Δ 均值 ≤ 0.15"作废）：标定发现 **vLLM 自己对自己**（同镜像，
开 / 关 `VLLM_BATCH_INVARIANT`，只换 kernel）在 long_zh 上就有 11.3% 的可判定行 top1 翻转、chat_en 22%、raw9 50%；
我们两个都正确的 attention kernel（v5 / 慢路径）互比也有 29% 翻转，逐层追踪显示差异从 L0 的 1.2% 放大到 L29 的 23%——
W4A4 / W8A8 激活量化让微小扰动跨过舍入边界后放大，逐行对齐在原理上做不到。新判据以 vLLM 自身噪声为基准：

`scripts/verify/kaon3-parity.sh` exit 0，即：
1. kernel 对拍 `gqa_window_parity`（滑窗 / 大分数）与 `gemm_lowp_parity`（FP8 静态 scale、NVFP4 gs）全过；
2. 汇总用例（`test/kaon3/oracle` 里 ≤512 行的全部，当前 27 个：24 条 prompts.tsv + raw9 / chat_en / chat_zh），
   记 FLOOR = vLLM(开 BI) 对 vLLM(关 BI)，OURS = vLLM(开 BI) 对我们：
   - 可判定行 top1 翻转率 OURS ≤ 1.5 × FLOOR + 3 个百分点；
   - （第 7 轮后改为只作参考、不判定）短 prompt 的平均 NLL 约 13 nat/token，由 chat 模板与用户消息开头那些
     logprob ≈ −20 的远尾 token 主导，下限本身就有 −0.25；质量判定改由自然长文本的 ΔNLL 承担（第 3 条）；
   - 共有 top-5 token 的 |Δlogprob| 均值 OURS ≤ 1.5 × FLOOR；
3. 长序列 long_zh（1252 token，过滑窗 1024）的 |ΔNLL| ≤ max(3 × 下限, 0.01)；
4. 单 token（bos1，无 attention）top1 一致。

## 上限

- 最多迭代 30 轮（用户要求不达标不停；上限只作为"方法错了"的报警）。
- 中途唯一允许停下的情况：ControlMaster 断了需要用户给新 token。

## 定位手段（只用 Rust + vLLM 输出）

1. `bos1` 单 token：attention 退化成 V 本身、RoPE 在位置 0 是恒等 → 只考 embed / norm / proj / MLP / MoE / lm_head / softcap。
2. 语义逐项核对 vLLM `gemma4.py`（镜像内 0.28 版本为准）与 `layer.rs` / `model.rs`。
3. kernel 级：`crates/kernels/tests/` 的 Rust GPU 测试（CPU 参考也用 Rust）。
4. 假设开关：一轮只改一处，结果进迭代记录。

## 卡点（到上限或放弃时填）

- 现象：
- 已排除的假设：
- 建议：

## 最新一次结果

- 日期：2026-10-22
- 第 0 轮：未跑（规格建立）

## P1 参数对齐（2026-10-22 晚，用户纠正后定稿）

用户 2026-10-22：「生产没有但是不代表不能配置，我只是要求你在对拍的时候，把参数对齐」。
生产命令（`vllm serve novita/Kaon-V3-nvpf4 --port 8081 --max_num_batched_tokens 4096`，无 `VLLM_BATCH_INVARIANT`）
用的是 vLLM 默认值；**对拍不看生产默认，看探针侧的假设**，逐项显式写出、不许留隐式默认：

| 参数 | 探针侧假设 | oracle 取值（显式传） | 生产实际 |
|---|---|---|---|
| block size | 32（`kaon3_probe.rs:23 BLOCK_SIZE`） | `--block-size 32` | 16（默认） |
| prefill 分块 | 一次 prefill 整串 | `--max-num-batched-tokens 4096`（> 最长用例 1252 → 等效不分块） | 4096 |
| KV dtype | FP8 e4m3（config `kv_cache_quant_algo`） | checkpoint 自带 `fp8_e4m3` | 同 |
| max_model_len | 无（单序列） | `--max-model-len 16384` | 取 config |
| max_num_seqs | 1 条请求 | `--max-num-seqs 64` | 256 |
| BI | 无此概念 | **BI=1 作判定参照 / BI=0 作噪声下限** | 不设 |

**判定参照口径（用户 2026-10-22 选定）**：`oracle/`（BI=1，确定性）作参照，`oracle_nobi/`（BI=0）只用于定噪声下限。
理由：BI=1 同输入两次完全一致，任何不一致都能归因到我们的实现；BI=0 抖动大，只当 floor。
**P1 出口**：用显式参数重启 oracle，重抓同一批用例，`kaon3_oracle_diff` 对旧文件 **flips=0 且 mean_common_d=0**
（证明没有隐式默认造成的漂移），然后跑一次 verify 把第 7 轮状态在新口径下的 `POOLED` 行写进迭代记录。

## 生产配置(用户 2026-10-22,**改 oracle 基线**)

生产启动命令只有:`vllm serve novita/Kaon-V3-nvpf4 --port 8081 --max_num_batched_tokens 4096`。
**没有 `VLLM_BATCH_INVARIANT=1`** → 生产用默认(非确定性 / batch-variant)kernel。含义:
- parity 的**判定参照 = BI 开的 vLLM**(`test/kaon3/oracle/`，确定性，任何不一致都可归因到我们)；
  **BI 关的 `test/kaon3/oracle_nobi/` 只用于定噪声下限**。用户 2026-10-22 晚选定，理由见上面 P1 参数对齐一节。
- 生产本身 run-to-run / 跨 batch 就不确定;逐 token 完全一致本就无法达到,噪声下限(BI 开 vs 关)正是这个量。
- pre2 bug 与 BI 无关:两种配置都强判 1852,我们判 5976。
