# 远端数据目录规范（5090 / B300 通用）

用户 2026-10-24 要求：**下一次按 `用途_日期时间_轮次` 存数据，避免对比的数据找错**。

## 规则

远端产物一律落在 `$REMOTE_ROOT/runs/<用途>_<YYYYMMDD-HHMM>_<轮次>/`，例如：

```
/hf_data/yaochen/runs/moe-seg_20261024-1430_r07/
/hf_data/yaochen/runs/hidden_20261024-1445_r08/
/hf_data/yaochen/runs/e2e_20261024-1500_r08/
```

- **用途**：小写 kebab，固定几个值：`moe-seg`（抓参照分段）、`hidden`（逐层残差流）、
  `e2e`（端到端探针输出）、`fp4`（FP4 定标/合成数据）、`oracle`（参照服务抓取）。
- **日期时间**：`YYYYMMDD-HHMM`，创建时刻（不是结束时刻）。
- **轮次**：`rNN`，与 `docs/loops/<feature>.md` 的轮次号一致。
- **不覆盖**：同一轮重跑就加后缀 `_b`、`_c`，不要删掉前一份。
- **口径写进目录**：目录内必须有 `README.txt`，写明
  - 用途、轮次、创建时间
  - **参照口径**（compile / eager）、关键环境变量（`RMSNORM_MODE`、`MOE_COMBINE_F32`、
    `CUBLASLT_WORKSPACE_SIZE` 等）
  - 数据来源脚本与命令行
  - **非零校验结果**（每个文件的行数 / nz 比例）

## 为什么

2026-10-24 同一天里，`/w/tmp/moe1`（eager）、`/w/tmp/moe3`（全零 dummy）、
`/w/hid/fn`（eager）、`/w/hid/our_e2e`（历史某轮）混在同一层目录下，
光看名字分不清哪份是哪份、什么口径 —— 差点拿 eager 的数据当 compile 判据，
也曾经把全零 dummy 当成真实数据报出"逐位一致"。

## 历史数据（2026-10-24 之前）

`/w/hid/`、`/w/tmp/` 下的老数据没有规范，**口径基本未知**。
已在远端建索引 `/w/runs/_legacy-index.txt`（列出目录 + 文件数）。
用老数据前必须：① 回去翻 `docs/loops` / `test/` 找它是哪一轮抓的；
② 按 `docs/data-layout.md` 的非零校验核一遍；③ 在记录里注明"来源为旧目录 xxx"。

已知口径的几个（2026-10-24 核过）：
| 目录 | 口径 | 内容 |
|---|---|---|
| `/w/hid/fn` | **eager**（`kaon3_vllm_hidden.py:58`） | 逐层残差流参照，p02 的 24 token |
| `/w/tmp/moe1` | **eager**（`moe_capture.py`） | layer0 experts 的 x/y + router logits，24 token |
| `/w/tmp/moe3`、`moe3b` | **全零 dummy，作废** | 抓成了 vLLM 显存 profile 那一次 |

## 怎么用

```bash
# 造目录 + README
RUN=$(bash scripts/dev/mkrun.sh moe-seg r07 "参照分段，compile 口径")
# $RUN = /hf_data/yaochen/runs/moe-seg_20261024-1430_r07
```

脚本里 **不要硬编码** `/w/tmp/xxx`；一律从 `MK_RUN` 拿到本次目录。
