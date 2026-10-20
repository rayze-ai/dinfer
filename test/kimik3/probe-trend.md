# probe trend — kimik3

> 探针测量（TTFT-长度 / kernel 分解 / 微基准伸缩 / 第三方对照），与 perf-trend（四组闭环）、engine-trend（开环泊松）并列。
> 数据源 `test/<model>/<日期>/probe_*/probe.csv`；本文件与 `probe-trend.{csv,html}` 由 `scripts/dev/probe_trend.py` 生成。

## probe_c6_tp8_1（2026-10-20，B300）

### c6_closed_loop（ms）

| x | completed dinfer | completed vllm | out_tok_per_s dinfer | out_tok_per_s vllm | tpot_p50_ms dinfer | tpot_p50_ms vllm | ttft_p50_ms dinfer | ttft_p50_ms vllm |
|---|---|---|---|---|---|---|---|---|
| c1 | 16 | 16 | 5.51 | 98.73 | 168.78 | 8.70 | 1822.35 | 192.59 |
| c8 | 32 | 32 | 21.07 | 502.98 | 324.45 | 13.09 | 7313.95 | 368.42 |
| c32 | 32 | 128 | 20.88 | 1432.39 | 817.28 | 19.15 | 32958.10 | 420.56 |
| c8-long |  | 32 |  | 469.43 |  | 13.11 |  | 508.34 |

- c1 tpot_p50_ms dinfer：C-6 首轮 tp8 全 93 层，预算 512，graph 关；vLLM = bench_b300_vllm1 tp8 预算 16384
- c1 ttft_p50_ms dinfer：C-6 首轮 tp8 全 93 层，预算 512，graph 关；vLLM = bench_b300_vllm1 tp8 预算 16384
- c1 out_tok_per_s dinfer：C-6 首轮 tp8 全 93 层，预算 512，graph 关；vLLM = bench_b300_vllm1 tp8 预算 16384
- c1 completed dinfer：C-6 首轮 tp8 全 93 层，预算 512，graph 关；vLLM = bench_b300_vllm1 tp8 预算 16384
- c32 tpot_p50_ms dinfer：n=32（脚本 max(c×4,16)）
- c32 ttft_p50_ms dinfer：n=32（脚本 max(c×4,16)）
- c32 out_tok_per_s dinfer：n=32（脚本 max(c×4,16)）
- c32 completed dinfer：n=32（脚本 max(c×4,16)）

## probe_c6_tp8_2（2026-10-20，B300）

### c6_closed_loop（ms）

| x | completed dinfer | out_tok_per_s dinfer | out_tok_per_s vllm | tpot_p50_ms dinfer | tpot_p50_ms vllm | ttft_p50_ms dinfer | ttft_p50_ms vllm |
|---|---|---|---|---|---|---|---|
| c1 | 16 | 5.50 | 98.73 | 169.13 | 8.70 | 1810.27 | 192.59 |
| c8 | 32 | 23.12 | 502.98 | 263.28 | 13.09 | 10881.14 | 368.42 |
| c32 | 128 | 40.62 | 1432.39 | 623.74 | 19.15 | 21732.56 | 420.56 |

- c1 tpot_p50_ms dinfer：第三轮：ActScratch f3962d7 + thread_local 08fd4ec，预算 2048，graph 关；smoke 三条 greedy 乱文待查
