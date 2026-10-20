# probe trend — qwen35

> 探针测量（TTFT-长度 / kernel 分解 / 微基准伸缩 / 第三方对照），与 perf-trend（四组闭环）、engine-trend（开环泊松）并列。
> 数据源 `test/<model>/<日期>/probe_*/probe.csv`；本文件与 `probe-trend.{csv,html}` 由 `scripts/dev/probe_trend.py` 生成。

## probe_gdn_r9_0（2026-10-20，3090）

### gdn_scale（us）

| x | chain_us dinfer | conv_rows0_us dinfer | conv_rows128_us dinfer | conv_rows256_us dinfer | conv_rows32_us dinfer | conv_rows64_us dinfer | ho_bv16_us dinfer | ho_bv32_us dinfer | ho_legacy_us dinfer | prep_us dinfer | wy_legacy_us dinfer | wy_us dinfer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x1024 | 956 | 551 | 133 | 248 | 74 | 77 | 605 | 381 | 2946 | 32 | 1201 | 542 |
| 8x1024 | 6787 | 934 | 442 | 427 | 510 | 462 | 4633 | 2460 | 18604 | 173 | 9165 | 4219 |
| 32x128 | 3496 | 418 | 236 | 285 | 266 | 244 | 2304 | 1424 | 9235 | 93 | 4610 | 1970 |
| 1x4096 | 3561 | 2199 | 228 | 273 | 261 | 239 | 2386 | 1488 | 11984 | 93 | 4601 | 1977 |
| 1x8192 | 7352 | 4396 | 444 | 424 | 508 | 459 | 4761 | 2961 | 24103 | 172 | 9226 | 4205 |

- 1x1024 chain_us dinfer：r9 第 0 轮基线（第 5A 轮 mma 版）；27B 形状 H=16 HV=48 K=V=128；gdn_chunk_bench 5 次均值
- 8x1024 chain_us dinfer：r9 第 0 轮基线（第 5A 轮 mma 版）；27B 形状 H=16 HV=48 K=V=128；gdn_chunk_bench 5 次均值
- 32x128 chain_us dinfer：r9 第 0 轮基线（第 5A 轮 mma 版）；27B 形状 H=16 HV=48 K=V=128；gdn_chunk_bench 5 次均值
- 1x4096 chain_us dinfer：r9 第 0 轮基线（第 5A 轮 mma 版）；27B 形状 H=16 HV=48 K=V=128；gdn_chunk_bench 5 次均值
- 1x8192 chain_us dinfer：r9 第 0 轮基线（第 5A 轮 mma 版）；27B 形状 H=16 HV=48 K=V=128；gdn_chunk_bench 5 次均值

## probe_gdn_r9_0（2026-10-20，B300）

### gdn_scale（us）

| x | chain_us dinfer | chain_us vllm | conv_rows128_us dinfer | conv_rows32_us dinfer | ho_bv16_us dinfer | ho_bv32_us dinfer | wy_us dinfer |
|---|---|---|---|---|---|---|---|
| 1x1024 | 311 | 359 |  | 41 | 160 |  | 134 |
| 8x1024 | 1662 |  | 222 |  |  | 664 | 912 |
| 1x8192 |  | 853 |  |  |  |  |  |

- 1x1024 wy_us dinfer：B300 GPU 6 sm_103（engine-gap 8e / 8g）
- 1x1024 chain_us vllm：fla chunk_gated_delta_rule 含 l2norm（engine-gap 8d）

## probe_gdn_r9_1（2026-10-20，3090）

### gdn_scale（us）

| x | chain_us dinfer | conv_rows0_us dinfer | conv_rows128_us dinfer | conv_rows256_us dinfer | conv_rows32_us dinfer | conv_rows64_us dinfer | ho_bv16_us dinfer | ho_bv16pf_us dinfer | ho_bv32_us dinfer | ho_bv32pf_us dinfer | ho_legacy_us dinfer | prep_us dinfer | wy_legacy_us dinfer | wy_us dinfer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x1024 | 845 | 551 | 133 | 248 | 75 | 77 | 346 | 440 | 278 | 441 | 2906 | 32 | 1201 | 542 |
| 8x1024 | 6156 | 986 | 442 | 425 | 510 | 463 | 2698 | 3494 | 1857 | 3148 | 18619 | 173 | 9202 | 4196 |
| 32x128 | 3229 | 427 | 238 | 294 | 270 | 245 | 1400 | 1897 | 1176 | 1907 | 9227 | 93 | 4604 | 1972 |
| 1x4096 | 3095 | 2280 | 227 | 279 | 262 | 237 | 1351 | 1712 | 1037 | 1779 | 12030 | 93 | 4609 | 1980 |
| 1x8192 | 6429 | 4501 | 440 | 425 | 508 | 458 | 2736 | 3467 | 2043 | 3473 | 24022 | 172 | 9224 | 4198 |

- 1x1024 chain_us dinfer：r9 第 1 轮：ho 装载改寄存器向量（PF=0）+ 可选预取（PF=1，bv*pf 列）；默认 PF=0
- 8x1024 chain_us dinfer：r9 第 1 轮：ho 装载改寄存器向量（PF=0）+ 可选预取（PF=1，bv*pf 列）；默认 PF=0
- 32x128 chain_us dinfer：r9 第 1 轮：ho 装载改寄存器向量（PF=0）+ 可选预取（PF=1，bv*pf 列）；默认 PF=0
- 1x4096 chain_us dinfer：r9 第 1 轮：ho 装载改寄存器向量（PF=0）+ 可选预取（PF=1，bv*pf 列）；默认 PF=0
- 1x8192 chain_us dinfer：r9 第 1 轮：ho 装载改寄存器向量（PF=0）+ 可选预取（PF=1，bv*pf 列）；默认 PF=0

## probe_prefill_1（2026-10-20，B300）

### ttft_len（ms）

| x | slope_us_per_token dinfer | slope_us_per_token vllm | ttft_p50_ms dinfer | ttft_p50_ms vllm | ttft_p50_repeat_ms dinfer | ttft_p50_repeat_ms vllm |
|---|---|---|---|---|---|---|
| 510 |  |  | 30.0 | 44.7 |  |  |
| 1021 |  |  | 55.5 | 86.7 | 55.4 | 44.7 |
| 2029 |  |  | 104.7 | 105.1 |  |  |
| 3009 |  |  | 155.2 | 141.6 |  |  |
| 4059 |  |  | 210.2 | 186.5 |  |  |
| 6019 |  |  | 313.8 | 259.7 |  |  |
| 1k-6k | 51.7 | 34.6 |  |  |  |  |

- 510 ttft_p50_ms dinfer：单发 max_tokens=1 换 seed 5 次中位；dinfer :8012 GPU 0
- 510 ttft_p50_ms vllm：vLLM :8013 GPU 1 容器 nightly
- 2029 ttft_p50_ms vllm：拐点
- 1021 ttft_p50_repeat_ms dinfer：同 prompt 重复第 2–5 次（无 prefix cache 不变）
- 1021 ttft_p50_repeat_ms vllm：同 prompt 重复第 2–5 次（prefix cache 命中）
- 1k-6k slope_us_per_token dinfer：1 K→6 K 线性拟合

### kernel_1k（ms）

| x | step_ms dinfer |
|---|---|
| gemm | 34.8 |
| gdn_wy_mma | 8.08 |
| gdn_ho_mma | 8.05 |
| conv1d_prefill | 2.30 |
| gqa_fa_v5 | 1.95 |
| rmsnorm | 1.42 |
| silu_and_mul | 1.41 |
| gated_rmsnorm | 0.69 |
| cumsum | 0.58 |
| norm | 0.49 |
| other | 0.5 |
| total_kernel | 60.3 |

- gemm step_ms dinfer：q35_mix.sqlite 稳态 11 个纯 prefill 步均值（T≈1024）；nvjet 17 种 ≈257 次；≈1.6 PFLOP/s = 70% 峰值
- gdn_wy_mma step_ms dinfer：48 层 × 168 µs
- gdn_ho_mma step_ms dinfer：48 层 × 168 µs
- conv1d_prefill step_ms dinfer：48 × 48 µs
- gqa_fa_v5 step_ms dinfer：16 层 × 122 µs
- rmsnorm step_ms dinfer：129 × 11 µs
- silu_and_mul step_ms dinfer：64 × 22 µs
- other step_ms dinfer：qkv_post / mul_sigmoid / gate / splitKreduce / argmax
- total_kernel step_ms dinfer：span 62.2；851 kernel/步；H2D 3 次 5 µs

### gdn_scale（us）

| x | chain_us dinfer | conv_us dinfer | fla_chunk_gdr_us vllm | ho_us dinfer | wy_us dinfer |
|---|---|---|---|---|---|
| 1x1024 | 311 | 41 | 359 | 160 | 134 |
| 8x1024 | 1662 | 222 |  | 664 | 912 |
| 32x128 | 845 |  |  |  |  |
| 1x3009 |  |  | 349 |  |  |
| 1x8192 |  |  | 853 |  |  |

- 1x1024 chain_us dinfer：gdn_chunk_bench GPU 6 sm_103；prep 18 / wy 134 / ho bv16 160 / conv rows32 41
- 8x1024 chain_us dinfer：wy 912 / ho bv32 664 / conv rows32 222（5.3× 线性涨）
- 32x128 chain_us dinfer：wy 461 / ho bv32 341 / conv 137
- 1x1024 ho_us dinfer：bv16
- 8x1024 ho_us dinfer：bv32
- 1x1024 conv_us dinfer：rows32
- 8x1024 conv_us dinfer：rows32
- 1x1024 fla_chunk_gdr_us vllm：vLLM 容器 third_party fla chunk_gated_delta_rule 同形状 HV48 K=V=128 含 l2norm 50 次均值
- 1x8192 fla_chunk_gdr_us vllm：1 K→8 K 只 2.4×
