# loop 规格：kaon3-layer1

- 状态：进行中（2026-10-24 创建）
- 创建：2026-10-24
- 验证脚本：`scripts/verify/kaon3-layer1.sh`
- 迭代记录：`test/kaon3/2026-10-24/kaon3-layer1.md`
- 上游：`docs/loops/kaon3-fp4dense.md`（MoE 段已排除；已有干净起点见下）

## 目标

用**截层对拍**定位 compile 口径下差异从**第几层**开始，并先只做 **1 层**。
1 层通过与否，决定往上一层还是往层内查。

## 为什么用截层而不是 hook（关键）

上一轮所有逐层尝试都坏在同一处：**hook 在 inductor 图里不生效**，
而且 eager 抓的数据不能当 compile 判据（`docs/knowledge/segment-isolation.md` §3）。

截层对拍绕开这个：vLLM 用 `--hf-overrides '{"text_config":{"num_hidden_layers":N}}'`
**真的只建 N 层**，我们侧 `kaon3_probe --layers N` **真的只跑 N 层**。
两侧都是 compile、都是真实代码路径，不需要 hook。

## 起点（2026-10-24 实测，可信）

compile 口径、中文 prompt、`kaon3_oracle_diff`：

| 口径 | flip | \|Δlp\| |
|---|---|---|
| **FLOOR**（同一 oracle 抓两次，BI=1） | **0.00%** | **0.0000** |
| 30 层全量（`MOE_COMBINE_F32=1`） | 14.29% (7/49) | 1.1599 |

**FLOOR = 0** → vLLM compile + BI=1 逐位可复现，判据是"逐位"，没有容差可言。

## 环境（每次从干净状态起）

- 机器 5090，`scripts/dev/remote_5090.sh`。**GPU 1 = 私有 oracle，GPU 7 = 基线禁用**。
- 用前 `nvidia-smi` 核显存，别假设空闲。
- oracle（**compile，不开 `enforce_eager`**）：
  ```bash
  ORACLE_GPU=1 ORACLE_PORT=28101 ORACLE_NAME=kaon-oracle-1 \
    scripts/dev/kaon3_oracle_server.sh 1        # 参数 = 截到 N 层
  ```
- 我们侧（compile 口径的 norm）：
  ```bash
  CUBLASLT_WORKSPACE_SIZE=1 RMSNORM_MODE=0 \
    ./target/release/kaon3_probe $M --batch <oracle目录> --layers 1 --out <ours>
  ```
- 对拍：`./target/release/kaon3_oracle_diff <oracle目录> <ours目录>`

## 观察方法

| 看什么 | 命令 |
|---|---|
| 端到端差值 | `kaon3_oracle_diff <oracle> <ours>` 的 `POOLED` 行 |
| FLOOR（必做） | 同一 oracle 抓两次，互相比；**必须为 0**，否则判据不可用 |
| 每层收敛情况 | `flip_pct` 与 `mean_common_d` 随 N 变化的曲线 |

## 停止条件（可机器判定，见 verify 脚本）

1. **FLOOR 必须为 0**（同一 oracle 两次抓取逐位一致）。不为 0 则整轮作废。
2. 参数逐项核对：block 32 / max-num-batched-tokens 4096 / BI=1 / compile 模式 /
   截层数两侧一致。任何一项不符则整轮作废。
3. 每层记录 `flip_pct`、`mean_common_d`，直到找到**第一个 flip > 0 的层号**。
4. 1 层时若已 flip > 0 → 差异在**第 0 层内部**，转层内定位；
   若 1 层 flip = 0 → 逐层往上加，找到第一层开始偏。

## 上限

每轮 ≤ 10 次迭代；单层定位超过 3 轮仍无结论，停下写卡点。

---

## 数据布局（用户 2026-10-24 要求）

产物按 `用途_日期时间_轮次` 存：`$REMOTE_ROOT/runs/<用途>_<YYYYMMDD-HHMM>_<rNN>/`，
目录内 `README.txt` 记口径与环境变量。工具 `scripts/dev/mkrun.sh`，规范见 `docs/data-layout.md`。

## 已排除（不要重复）

- **dense NVFP4 GEMM**：自洽合成判据，`ndiff=0/11264`，三条独立路径互证。
- **`MOE_COMBINE_F32` 是主因**：compile 口径下两种口径都差 ~1.1，改它方向还反了。
- **eager 抓的子模块数据当判据**：规格禁止；且参照 `L*.bin` 两次抓取自身差 0.0077。
