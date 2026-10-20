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

## probe_gdn_r9_2（2026-10-20，3090）

### gdn_scale（us）

| x | chain_us dinfer | conv_rows0_us dinfer | conv_rows128_us dinfer | conv_rows256_us dinfer | conv_rows32_us dinfer | conv_rows64_us dinfer | ho_bv16_us dinfer | ho_bv16pf_us dinfer | ho_bv16pfsp_us dinfer | ho_bv16sp_us dinfer | ho_bv32_us dinfer | ho_bv32pf_us dinfer | ho_bv32pfsp_us dinfer | ho_bv32sp_us dinfer | ho_legacy_us dinfer | prep_us dinfer | wy_legacy_us dinfer | wy_us dinfer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x1024 | 983 | 551 | 132 | 249 | 73 | 77 | 347 | 442 | 326 | 340 | 278 | 441 | 316 | 318 | 2972 | 33 | 1363 | 603 |
| 8x1024 | 6483 | 1027 | 443 | 427 | 512 | 464 | 2678 | 3536 | 2539 | 2489 | 1816 | 3142 | 2147 | 2111 | 18632 | 173 | 9258 | 4203 |
| 32x128 | 3191 | 444 | 238 | 311 | 269 | 250 | 1407 | 1851 | 1342 | 1324 | 1156 | 1878 | 1235 | 1196 | 9261 | 93 | 4606 | 1977 |
| 1x4096 | 3346 | 2321 | 230 | 283 | 260 | 236 | 1362 | 1730 | 1282 | 1344 | 1035 | 1744 | 1280 | 1300 | 11993 | 93 | 4609 | 1974 |
| 1x8192 | 6943 | 4716 | 436 | 427 | 513 | 463 | 2792 | 3415 | 2605 | 2696 | 2045 | 3498 | 2654 | 2634 | 23781 | 173 | 9252 | 4206 |

- 1x1024 chain_us dinfer：r9 第 2 轮：ho 两段版 h-pass + o-pass（*sp 列）；3090 默认仍融合版（ArchProfile 门限待 B300 数字）
- 8x1024 chain_us dinfer：r9 第 2 轮：ho 两段版 h-pass + o-pass（*sp 列）；3090 默认仍融合版（ArchProfile 门限待 B300 数字）
- 32x128 chain_us dinfer：r9 第 2 轮：ho 两段版 h-pass + o-pass（*sp 列）；3090 默认仍融合版（ArchProfile 门限待 B300 数字）
- 1x4096 chain_us dinfer：r9 第 2 轮：ho 两段版 h-pass + o-pass（*sp 列）；3090 默认仍融合版（ArchProfile 门限待 B300 数字）
- 1x8192 chain_us dinfer：r9 第 2 轮：ho 两段版 h-pass + o-pass（*sp 列）；3090 默认仍融合版（ArchProfile 门限待 B300 数字）

## probe_gdn_r9_3（2026-10-20，3090）

### gdn_scale（us）

| x | chain_us dinfer | conv_rows0_us dinfer | conv_rows128_us dinfer | conv_rows256_us dinfer | conv_rows32_us dinfer | conv_rows64_us dinfer | ho_bv16_us dinfer | ho_bv16pf_us dinfer | ho_bv16pfsp_us dinfer | ho_bv16sp_us dinfer | ho_bv32_us dinfer | ho_bv32pf_us dinfer | ho_bv32pfsp_us dinfer | ho_bv32sp_us dinfer | ho_legacy_us dinfer | prep_us dinfer | wy_legacy_us dinfer | wy_serial_us dinfer | wy_us dinfer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x1024 | 591 | 560 | 135 | 253 | 75 | 78 | 352 | 448 | 330 | 343 | 281 | 448 | 320 | 322 | 2942 | 32 | 1219 | 342 | 274 |
| 8x1024 | 3878 | 1039 | 443 | 456 | 511 | 463 | 2699 | 3468 | 2524 | 2466 | 1916 | 3113 | 2135 | 2116 | 18746 | 173 | 9247 | 2287 | 1906 |
| 32x128 | 2263 | 448 | 240 | 311 | 269 | 249 | 1407 | 1869 | 1364 | 1354 | 1170 | 1931 | 1270 | 1202 | 9261 | 92 | 4607 | 1220 | 1010 |
| 1x4096 | 2125 | 2354 | 225 | 290 | 260 | 238 | 1370 | 1713 | 1319 | 1372 | 1029 | 1767 | 1287 | 1274 | 12001 | 93 | 4618 | 1218 | 1002 |
| 1x8192 | 4140 | 4775 | 435 | 424 | 513 | 462 | 2781 | 3552 | 2684 | 2687 | 2052 | 3558 | 2688 | 2607 | 23891 | 173 | 9414 | 2284 | 1910 |

- 1x1024 chain_us dinfer：r9 第 3 轮：wy 分块三角逆（16×16 对角块 warp 内前代 + 非对角 fp32 乘）+ sM f32 复用 sK∪sQ 省 16 KB smem；wy_serial 列 = 64 步前代回退
- 8x1024 chain_us dinfer：r9 第 3 轮：wy 分块三角逆（16×16 对角块 warp 内前代 + 非对角 fp32 乘）+ sM f32 复用 sK∪sQ 省 16 KB smem；wy_serial 列 = 64 步前代回退
- 32x128 chain_us dinfer：r9 第 3 轮：wy 分块三角逆（16×16 对角块 warp 内前代 + 非对角 fp32 乘）+ sM f32 复用 sK∪sQ 省 16 KB smem；wy_serial 列 = 64 步前代回退
- 1x4096 chain_us dinfer：r9 第 3 轮：wy 分块三角逆（16×16 对角块 warp 内前代 + 非对角 fp32 乘）+ sM f32 复用 sK∪sQ 省 16 KB smem；wy_serial 列 = 64 步前代回退
- 1x8192 chain_us dinfer：r9 第 3 轮：wy 分块三角逆（16×16 对角块 warp 内前代 + 非对角 fp32 乘）+ sM f32 复用 sK∪sQ 省 16 KB smem；wy_serial 列 = 64 步前代回退

## probe_gdn_r9_4（2026-10-20，3090）

### gdn_scale（us）

| x | chain_us dinfer | conv_rows0_us dinfer | conv_rows128_us dinfer | conv_rows16_us dinfer | conv_rows256_us dinfer | conv_rows32_us dinfer | conv_rows64_us dinfer | conv_rows8_us dinfer | ho_bv16_us dinfer | ho_bv16pf_us dinfer | ho_bv16pfsp_us dinfer | ho_bv16sp_us dinfer | ho_bv32_us dinfer | ho_bv32pf_us dinfer | ho_bv32pfsp_us dinfer | ho_bv32sp_us dinfer | ho_legacy_us dinfer | prep_us dinfer | wy_legacy_us dinfer | wy_serial_us dinfer | wy_us dinfer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x1024 | 579 | 552 | 133 | 84 | 249 | 73 | 77 | 105 | 348 | 441 | 344 | 354 | 275 | 442 | 358 | 335 | 2947 | 32 | 1202 | 339 | 276 |
| 8x1024 | 3852 | 1021 | 441 | 591 | 426 | 513 | 465 | 764 | 2689 | 3475 | 2909 | 2863 | 1959 | 3127 | 2516 | 2497 | 18713 | 173 | 9320 | 2273 | 1904 |
| 32x128 | 2272 | 448 | 237 | 310 | 308 | 269 | 248 | 394 | 1409 | 1854 | 1513 | 1506 | 1167 | 1935 | 1398 | 1348 | 9271 | 93 | 4643 | 1221 | 1008 |
| 1x4096 | 2123 | 2326 | 227 | 303 | 287 | 260 | 236 | 389 | 1388 | 1743 | 1443 | 1471 | 1038 | 1738 | 1436 | 1414 | 12057 | 93 | 4637 | 1217 | 1002 |
| 1x8192 | 4144 | 4729 | 436 | 592 | 426 | 512 | 462 | 765 | 2808 | 3550 | 3011 | 3044 | 2075 | 3563 | 3003 | 2954 | 24244 | 177 | 9444 | 2284 | 1914 |

- 1x1024 chain_us dinfer：r9 第 4 轮 A：o-pass VT=64（2 block/SM）+ conv 6 block/SM 档位（rows8/16 列）——两者 3090 都退化，见 loop 页；*sp 列是 VT=64 的数
- 8x1024 chain_us dinfer：r9 第 4 轮 A：o-pass VT=64（2 block/SM）+ conv 6 block/SM 档位（rows8/16 列）——两者 3090 都退化，见 loop 页；*sp 列是 VT=64 的数
- 32x128 chain_us dinfer：r9 第 4 轮 A：o-pass VT=64（2 block/SM）+ conv 6 block/SM 档位（rows8/16 列）——两者 3090 都退化，见 loop 页；*sp 列是 VT=64 的数
- 1x4096 chain_us dinfer：r9 第 4 轮 A：o-pass VT=64（2 block/SM）+ conv 6 block/SM 档位（rows8/16 列）——两者 3090 都退化，见 loop 页；*sp 列是 VT=64 的数
- 1x8192 chain_us dinfer：r9 第 4 轮 A：o-pass VT=64（2 block/SM）+ conv 6 block/SM 档位（rows8/16 列）——两者 3090 都退化，见 loop 页；*sp 列是 VT=64 的数

## probe_gdn_r9_5（2026-10-20，3090）

### gdn_scale（us）

| x | chain_us dinfer | conv_rows0_us dinfer | conv_rows128_us dinfer | conv_rows16_us dinfer | conv_rows256_us dinfer | conv_rows32_us dinfer | conv_rows64_us dinfer | conv_rows8_us dinfer | ho_bv16_us dinfer | ho_bv16pf_us dinfer | ho_bv16pfsp_us dinfer | ho_bv16sp_us dinfer | ho_bv32_us dinfer | ho_bv32pf_us dinfer | ho_bv32pfsp_us dinfer | ho_bv32sp_us dinfer | ho_legacy_us dinfer | prep_us dinfer | wy_legacy_us dinfer | wy_serial_us dinfer | wy_us dinfer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x1024 | 654 | 550 | 133 | 85 | 248 | 75 | 77 | 105 | 348 | 441 | 326 | 337 | 273 | 441 | 316 | 317 | 3012 | 33 | 1365 | 375 | 302 |
| 8x1024 | 3842 | 1041 | 442 | 590 | 427 | 510 | 463 | 767 | 2667 | 3482 | 2508 | 2489 | 1892 | 3137 | 2129 | 2112 | 18598 | 173 | 9265 | 2283 | 1908 |
| 32x128 | 2252 | 439 | 238 | 311 | 309 | 267 | 249 | 392 | 1406 | 1899 | 1387 | 1351 | 1141 | 1918 | 1248 | 1198 | 9340 | 93 | 4658 | 1225 | 1013 |
| 1x4096 | 2126 | 2370 | 228 | 303 | 282 | 259 | 236 | 390 | 1371 | 1712 | 1321 | 1369 | 1035 | 1784 | 1305 | 1305 | 11994 | 93 | 4614 | 1219 | 1003 |
| 1x8192 | 4137 | 4762 | 436 | 593 | 424 | 514 | 462 | 764 | 2771 | 3640 | 2690 | 2700 | 2053 | 3590 | 2678 | 2587 | 24176 | 173 | 9446 | 2290 | 1910 |

- 1x1024 chain_us dinfer：r9 第 5 轮（3090 收口）：o-pass VT=128 为首选档（8 warp）、conv 回 2 波规则；默认路径 = wy Mma(分块逆) + ho 融合 bv32 PF=0
- 8x1024 chain_us dinfer：r9 第 5 轮（3090 收口）：o-pass VT=128 为首选档（8 warp）、conv 回 2 波规则；默认路径 = wy Mma(分块逆) + ho 融合 bv32 PF=0
- 32x128 chain_us dinfer：r9 第 5 轮（3090 收口）：o-pass VT=128 为首选档（8 warp）、conv 回 2 波规则；默认路径 = wy Mma(分块逆) + ho 融合 bv32 PF=0
- 1x4096 chain_us dinfer：r9 第 5 轮（3090 收口）：o-pass VT=128 为首选档（8 warp）、conv 回 2 波规则；默认路径 = wy Mma(分块逆) + ho 融合 bv32 PF=0
- 1x8192 chain_us dinfer：r9 第 5 轮（3090 收口）：o-pass VT=128 为首选档（8 warp）、conv 回 2 波规则；默认路径 = wy Mma(分块逆) + ho 融合 bv32 PF=0

## probe_gdn_r9_6（2026-10-20，B300）

### gdn_scale（us）

| x | chain_us dinfer | conv_rows0_us dinfer | conv_rows128_us dinfer | conv_rows16_us dinfer | conv_rows256_us dinfer | conv_rows32_us dinfer | conv_rows64_us dinfer | conv_rows8_us dinfer | ho_bv16_us dinfer | ho_bv16pf_us dinfer | ho_bv16pfsp_us dinfer | ho_bv16sp_us dinfer | ho_bv32_us dinfer | ho_bv32pf_us dinfer | ho_bv32pfsp_us dinfer | ho_bv32sp_us dinfer | ho_legacy_us dinfer | prep_us dinfer | wy_legacy_us dinfer | wy_serial_us dinfer | wy_us dinfer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8x1024 | 1110 | 808 | 310 | 220 | 321 | 222 | 247 | 249 | 694 | 755 | 793 | 981 | 498 | 649 | 671 | 648 | 5102 | 82 | 2955 | 685 | 531 |
| 32x128 | 593 | 217 | 172 | 128 | 301 | 137 | 168 | 137 | 403 | 418 | 455 | 503 | 265 | 350 | 366 | 348 | 2519 | 43 | 1504 | 371 | 284 |
| 1x4096 | 786 | 3063 | 162 | 122 | 296 | 133 | 160 | 131 | 483 | 548 | 509 | 641 | 334 | 441 | 407 | 512 | 3390 | 44 | 1505 | 371 | 284 |
| 1x8192 | 1538 | 6155 | 311 | 219 | 315 | 222 | 246 | 249 | 946 | 1095 | 999 | 1264 | 660 | 874 | 792 | 1001 | 6933 | 82 | 2956 | 685 | 532 |

- 8x1024 chain_us dinfer：r9 第 6 轮：B300-50 GPU 6 sm_103 首次复测（r5 代码，默认门限还是 3090 的：ho bv16 / conv rows128）；chunk_qwen parity 过，decode_b32 bv0_legacy 失败待查是否既有
- 32x128 chain_us dinfer：r9 第 6 轮：B300-50 GPU 6 sm_103 首次复测（r5 代码，默认门限还是 3090 的：ho bv16 / conv rows128）；chunk_qwen parity 过，decode_b32 bv0_legacy 失败待查是否既有
- 1x4096 chain_us dinfer：r9 第 6 轮：B300-50 GPU 6 sm_103 首次复测（r5 代码，默认门限还是 3090 的：ho bv16 / conv rows128）；chunk_qwen parity 过，decode_b32 bv0_legacy 失败待查是否既有
- 1x8192 chain_us dinfer：r9 第 6 轮：B300-50 GPU 6 sm_103 首次复测（r5 代码，默认门限还是 3090 的：ho bv16 / conv rows128）；chunk_qwen parity 过，decode_b32 bv0_legacy 失败待查是否既有

## probe_gdn_r9_7（2026-10-20，B300）

### gdn_scale（us）

| x | chain_us dinfer | conv_rows0_us dinfer | conv_rows128_us dinfer | conv_rows16_us dinfer | conv_rows256_us dinfer | conv_rows32_us dinfer | conv_rows64_us dinfer | conv_rows8_us dinfer | ho_bv16_us dinfer | ho_bv16pf_us dinfer | ho_bv16pfsp_us dinfer | ho_bv16sp_us dinfer | ho_bv32_us dinfer | ho_bv32pf_us dinfer | ho_bv32pfsp_us dinfer | ho_bv32sp_us dinfer | ho_legacy_us dinfer | prep_us dinfer | wy_legacy_us dinfer | wy_serial_us dinfer | wy_us dinfer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x1024 | 208 | 486 | 114 | 38 | 217 | 41 | 62 | 37 | 115 | 142 | 132 | 166 | 87 | 115 | 106 | 133 | 851 | 18 | 434 | 136 | 102 |
| 8x1024 | 1109 | 809 | 310 | 219 | 320 | 222 | 247 | 250 | 694 | 755 | 792 | 984 | 498 | 648 | 672 | 642 | 5092 | 82 | 2957 | 685 | 532 |
| 32x128 | 593 | 218 | 172 | 127 | 300 | 136 | 166 | 138 | 403 | 418 | 455 | 503 | 265 | 350 | 366 | 348 | 2523 | 43 | 1503 | 371 | 283 |
| 1x4096 | 660 | 3063 | 162 | 122 | 296 | 132 | 161 | 131 | 459 | 549 | 510 | 640 | 334 | 442 | 405 | 511 | 3385 | 43 | 1503 | 372 | 283 |
| 1x8192 | 1271 | 6158 | 309 | 219 | 316 | 221 | 247 | 249 | 943 | 1086 | 1001 | 1265 | 659 | 875 | 792 | 1004 | 6926 | 82 | 2956 | 684 | 533 |

- 1x1024 chain_us dinfer：r9 第 7 轮：B300-50 GPU 6，新 ArchProfile 门限（ho bv 恒 32、conv Sm100 8 块/SM → rows16/32）；默认路径数
- 8x1024 chain_us dinfer：r9 第 7 轮：B300-50 GPU 6，新 ArchProfile 门限（ho bv 恒 32、conv Sm100 8 块/SM → rows16/32）；默认路径数
- 32x128 chain_us dinfer：r9 第 7 轮：B300-50 GPU 6，新 ArchProfile 门限（ho bv 恒 32、conv Sm100 8 块/SM → rows16/32）；默认路径数
- 1x4096 chain_us dinfer：r9 第 7 轮：B300-50 GPU 6，新 ArchProfile 门限（ho bv 恒 32、conv Sm100 8 块/SM → rows16/32）；默认路径数
- 1x8192 chain_us dinfer：r9 第 7 轮：B300-50 GPU 6，新 ArchProfile 门限（ho bv 恒 32、conv Sm100 8 块/SM → rows16/32）；默认路径数

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

## probe_prefill_2（2026-10-20，B300）

### ttft_len（ms）

| x | slope_us_per_token dinfer | ttft_p50_ms dinfer | ttft_p50_ms dinfer_legacy_kernel | ttft_p50_ms dinfer_nosnap |
|---|---|---|---|---|
| 510 |  | 28.4 |  | 28.7 |
| 1021 |  | 52.0 | 63.6 | 52.3 |
| 2029 |  | 107.0 |  | 106.9 |
| 3009 |  | 163.0 | 198.5 | 163.1 |
| 4059 |  | 223.6 |  | 223.7 |
| 6019 |  | 349.1 | 426.1 | 346.3 |
| 1k-6k | 58.3 |  |  |  |

- 510 ttft_p50_ms dinfer：r8+r9 合入后二进制（1873e55），snapshot 默认开（10% KV 预算）；同口径复测 probe_prefill_1
- 1021 ttft_p50_ms dinfer：r8+r9 合入后二进制（1873e55），snapshot 默认开（10% KV 预算）；同口径复测 probe_prefill_1
- 2029 ttft_p50_ms dinfer：r8+r9 合入后二进制（1873e55），snapshot 默认开（10% KV 预算）；同口径复测 probe_prefill_1
- 3009 ttft_p50_ms dinfer：r8+r9 合入后二进制（1873e55），snapshot 默认开（10% KV 预算）；同口径复测 probe_prefill_1
- 4059 ttft_p50_ms dinfer：r8+r9 合入后二进制（1873e55），snapshot 默认开（10% KV 预算）；同口径复测 probe_prefill_1
- 6019 ttft_p50_ms dinfer：r8+r9 合入后二进制（1873e55），snapshot 默认开（10% KV 预算）；同口径复测 probe_prefill_1
- 1k-6k slope_us_per_token dinfer：r9 前 51.5（probe_prefill_1 重拟合）；变差，6 K +35 ms，疑 r8 每 1024 对齐边界存快照
- 510 ttft_p50_ms dinfer_nosnap：--gdn-snapshot-gib 0，池关但 1024 对齐切步仍在
- 1021 ttft_p50_ms dinfer_nosnap：--gdn-snapshot-gib 0，池关但 1024 对齐切步仍在
- 2029 ttft_p50_ms dinfer_nosnap：--gdn-snapshot-gib 0，池关但 1024 对齐切步仍在
- 3009 ttft_p50_ms dinfer_nosnap：--gdn-snapshot-gib 0，池关但 1024 对齐切步仍在
- 4059 ttft_p50_ms dinfer_nosnap：--gdn-snapshot-gib 0，池关但 1024 对齐切步仍在
- 6019 ttft_p50_ms dinfer_nosnap：--gdn-snapshot-gib 0，池关但 1024 对齐切步仍在
- 1021 ttft_p50_ms dinfer_legacy_kernel：GDN_CHUNK_WY=legacy GDN_CHUNK_HO_BV=0（r9 前 kernel）+ 池关 + 切步
- 3009 ttft_p50_ms dinfer_legacy_kernel：GDN_CHUNK_WY=legacy GDN_CHUNK_HO_BV=0（r9 前 kernel）+ 池关 + 切步
- 6019 ttft_p50_ms dinfer_legacy_kernel：GDN_CHUNK_WY=legacy GDN_CHUNK_HO_BV=0（r9 前 kernel）+ 池关 + 切步
