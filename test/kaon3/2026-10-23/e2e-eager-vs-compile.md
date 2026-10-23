# 2026-10-23 Kaon-V3 端到端复测 + 参照口径定位

## 结论
- p02 对 **eager** vLLM（BI=1）整跑 **L0–L29 逐位 0**；但 30 用例端到端无实质改善。
- 原因：端到端判据 `oracle_clean` 来自服务实例（**compile 模式**）；vLLM 自身 eager vs compile 在 p02 上差 4/23 翻转、|Δlp| 1.5725。
- 下一步换靶子到 compile 模式（待用户确认 BI=1 还是 BI=0）。

## 端到端（`PROBE_ENV="CUBLASLT_WORKSPACE_SIZE=1" bash scripts/verify/kaon3-parity.sh`）

| run | 状态 | flip | \|Δlp\| | 短 ΔNLL | 长 ΔNLL |
|---|---|---|---|---|---|
| 1023-151951-1421909 | router 归零后 | 15.53% | 1.2593 | +0.0875 | +0.0432 |
| 1023-165244-1500999 | p02 30 层逐位 0 后 | 16.56% | 1.1994 | −0.1625 | +0.1466 |
| FLOOR（oracle_clean_b） | — | 0.83% | 0.0276 | −0.0025 | −0.0817 |

p02 单例（run 1023-165244）：rows=24 decisive=22 flips=3（13.64%）|Δlp| 1.2109。

## eager vs compile（`/w/tmp/eager_vs_srv.py <M> <eager> p02`，参数同服务：max_model_len 16384 / max_num_seqs 64 / batched 4096 / block 32 / BI=1）

```
RESULT eager=1 case=p02 rows=23 top1_flips=4 mean|dlp(actual)|=1.5725
RESULT eager=0 case=p02 rows=23 top1_flips=0 mean|dlp(actual)|=0.0000
```

## p02 整跑逐层（`kaon3_probe --hidden-all --hidden-cmp /w/hid2/ref_p02`，ref = eager 抓取）
L0–L29 mean/max 全部 `0.000e0`（§57）。修复历程见 `D-16-REBASE.md` §51–§57。
