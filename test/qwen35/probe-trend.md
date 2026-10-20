# probe trend — qwen35

> 探针测量（TTFT-长度 / kernel 分解 / 微基准伸缩 / 第三方对照），与 perf-trend（四组闭环）、engine-trend（开环泊松）并列。
> 数据源 `test/<model>/<日期>/probe_*/probe.csv`；本文件与 `probe-trend.{csv,html}` 由 `scripts/dev/probe_trend.py` 生成。

## probe_attn_r11a_0（2026-10-20，B300）

### attn_scale（us）

| x | fa_nbuf2_us dinfer | fa_us dinfer | tflops dinfer | tflops_nbuf2 dinfer |
|---|---|---|---|---|
| 1024 | 99 | 91 | 142.1 | 129.8 |
| 3009 | 565 | 527 | 211.1 | 196.8 |
| 6019 | 1917 | 1818 | 244.9 | 232.2 |

- 1024 fa_us dinfer：r11a 第 0 轮基线：gqa_fa_v5<256> TILE 32 NBUF 1；B300-50 GPU 4；ncu 255 reg 有 spill、2 block/SM、tensor pipe 45%、DRAM 0.6%

## probe_attn_r11a_1（2026-10-20，B300）

### attn_scale（us）

| x | fa_cfg0_us dinfer | fa_cfg432_us dinfer | fa_cfg832_us dinfer | fa_cfg864_us dinfer | fa_us dinfer | tflops dinfer | tflops_cfg0 dinfer | tflops_cfg432 dinfer | tflops_cfg832 dinfer | tflops_cfg864 dinfer |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 91 | 88 | 107 | 97 | 83 | 154.5 | 142.0 | 146.2 | 121.0 | 133.3 |
| 3009 | 524 | 534 | 634 | 561 | 482 | 230.9 | 212.5 | 208.5 | 175.5 | 198.5 |
| 6019 | 1815 | 1870 | 2197 | 1916 | 1645 | 270.6 | 245.2 | 238.1 | 202.6 | 232.4 |

- 1024 fa_us dinfer：r11a 第 1 轮：v6（Q 进 smem、NW/TILE 模板化、block 反序）cfg 464 = 4 warp × tile 64；B300-50 GPU 4

## probe_attn_r11a_2（2026-10-20，B300）

### attn_scale（us）

| x | fa_cfg1432_us dinfer | fa_cfg1464_us dinfer | fa_cfg1832_us dinfer | fa_cfg1864_us dinfer | fa_us dinfer | tflops dinfer | tflops_cfg1432 dinfer | tflops_cfg1464 dinfer | tflops_cfg1832 dinfer | tflops_cfg1864 dinfer |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 106 | 91 | 102 | 90 | 83 | 155.3 | 121.1 | 142.1 | 126.1 | 142.7 |
| 3009 | 700 | 577 | 609 | 528 | 481 | 231.2 | 158.8 | 192.8 | 182.6 | 210.7 |
| 6019 | 2615 | 2078 | 2143 | 1802 | 1650 | 269.9 | 170.2 | 214.2 | 207.7 | 247.0 |

- 1024 fa_us dinfer：r11a 第 2 轮：v6 cfg 464 复现（默认路径）；同轮 v7 warp 对分工 1864/1832/1464/1432 全更慢

## probe_gdn_r10_1（2026-10-20，B300）

### gdn_scale（us）

| x | chain_us dinfer | conv_rows0_us dinfer | conv_rows128_us dinfer | conv_rows16_us dinfer | conv_rows256_us dinfer | conv_rows32_us dinfer | conv_rows64_us dinfer | conv_rows8_us dinfer | ho_bv16_us dinfer | ho_bv16pf_us dinfer | ho_bv16pfsp_us dinfer | ho_bv16sp_us dinfer | ho_bv32_us dinfer | ho_bv32pf_us dinfer | ho_bv32pfsp_us dinfer | ho_bv32sp_us dinfer | ho_legacy_us dinfer | prep_us dinfer | wy_legacy_us dinfer | wy_serial_us dinfer | wy_us dinfer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x1024 | 185 | 488 | 114 | 39 | 218 | 41 | 62 | 37 | 70 | 78 | 89 | 110 | 65 | 66 | 114 | 100 | 789 | 18 | 421 | 111 | 76 |
| 8x1024 | 988 | 809 | 310 | 221 | 320 | 222 | 247 | 249 | 542 | 543 | 622 | 725 | 405 | 379 | 633 | 543 | 5021 | 82 | 2931 | 656 | 505 |
| 32x128 | 544 | 218 | 172 | 127 | 301 | 137 | 168 | 137 | 317 | 325 | 346 | 386 | 230 | 221 | 343 | 302 | 2512 | 42 | 1485 | 359 | 272 |
| 1x4096 | 661 | 3065 | 162 | 122 | 296 | 132 | 161 | 131 | 321 | 387 | 351 | 439 | 352 | 263 | 458 | 395 | 3306 | 43 | 1488 | 359 | 272 |
| 1x8192 | 1271 | 6161 | 308 | 219 | 314 | 221 | 247 | 248 | 634 | 769 | 683 | 865 | 690 | 520 | 899 | 777 | 6638 | 82 | 2921 | 657 | 505 |

- 1x1024 chain_us dinfer：r10 第 1 轮：aq/u/w/kg f32→bf16，ho uint4 直载；B300-50 GPU 7 target_r10；默认路径（ArchProfile）
- 8x1024 chain_us dinfer：r10 第 1 轮：aq/u/w/kg f32→bf16，ho uint4 直载；B300-50 GPU 7 target_r10；默认路径（ArchProfile）
- 32x128 chain_us dinfer：r10 第 1 轮：aq/u/w/kg f32→bf16，ho uint4 直载；B300-50 GPU 7 target_r10；默认路径（ArchProfile）
- 1x4096 chain_us dinfer：r10 第 1 轮：aq/u/w/kg f32→bf16，ho uint4 直载；B300-50 GPU 7 target_r10；默认路径（ArchProfile）
- 1x8192 chain_us dinfer：r10 第 1 轮：aq/u/w/kg f32→bf16，ho uint4 直载；B300-50 GPU 7 target_r10；默认路径（ArchProfile）

## probe_gdn_r10_2（2026-10-20，B300）

### gdn_scale（us）

| x | chain_us dinfer | conv_rows0_us dinfer | conv_rows128_us dinfer | conv_rows16_us dinfer | conv_rows256_us dinfer | conv_rows32_us dinfer | conv_rows64_us dinfer | conv_rows8_us dinfer | ho_bv16_us dinfer | ho_bv16pf_us dinfer | ho_bv16pfsp_us dinfer | ho_bv16sp_us dinfer | ho_bv32_us dinfer | ho_bv32pf_us dinfer | ho_bv32pfsp_us dinfer | ho_bv32sp_us dinfer | ho_legacy_us dinfer | prep_us dinfer | wy_legacy_us dinfer | wy_serial_us dinfer | wy_us dinfer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x1024 | 178 | 488 | 115 | 38 | 218 | 41 | 62 | 37 | 70 | 78 | 89 | 109 | 65 | 66 | 114 | 100 | 790 | 18 | 421 | 111 | 76 |
| 8x1024 | 967 | 811 | 310 | 220 | 320 | 221 | 248 | 249 | 541 | 544 | 624 | 723 | 410 | 378 | 634 | 543 | 5020 | 82 | 2931 | 657 | 505 |
| 32x128 | 535 | 218 | 172 | 128 | 301 | 137 | 169 | 137 | 317 | 325 | 346 | 386 | 230 | 221 | 343 | 302 | 2510 | 43 | 1486 | 359 | 272 |
| 1x4096 | 576 | 3067 | 162 | 122 | 296 | 133 | 160 | 131 | 320 | 390 | 351 | 440 | 352 | 264 | 459 | 396 | 3303 | 42 | 1488 | 359 | 272 |
| 1x8192 | 1103 | 6160 | 307 | 220 | 314 | 221 | 245 | 249 | 633 | 775 | 684 | 866 | 690 | 520 | 900 | 778 | 6626 | 82 | 2918 | 657 | 505 |

- 1x1024 chain_us dinfer：r10 第 2 轮：ArchProfile Sm100 开 ho 寄存器预取（默认路径 = bv32pf）；B300-50 GPU 7
- 8x1024 chain_us dinfer：r10 第 2 轮：ArchProfile Sm100 开 ho 寄存器预取（默认路径 = bv32pf）；B300-50 GPU 7
- 32x128 chain_us dinfer：r10 第 2 轮：ArchProfile Sm100 开 ho 寄存器预取（默认路径 = bv32pf）；B300-50 GPU 7
- 1x4096 chain_us dinfer：r10 第 2 轮：ArchProfile Sm100 开 ho 寄存器预取（默认路径 = bv32pf）；B300-50 GPU 7
- 1x8192 chain_us dinfer：r10 第 2 轮：ArchProfile Sm100 开 ho 寄存器预取（默认路径 = bv32pf）；B300-50 GPU 7

## probe_gdn_r10_3（2026-10-20，B300）

### gdn_scale（us）

| x | chain_us dinfer | conv_rows0_us dinfer | conv_rows128_us dinfer | conv_rows16_us dinfer | conv_rows256_us dinfer | conv_rows32_us dinfer | conv_rows64_us dinfer | conv_rows8_us dinfer | ho_bv16_us dinfer | ho_bv16pf_us dinfer | ho_bv16pfsp_us dinfer | ho_bv16sp_us dinfer | ho_bv32_us dinfer | ho_bv32pf_us dinfer | ho_bv32pfsp_us dinfer | ho_bv32sp_us dinfer | ho_legacy_us dinfer | prep_us dinfer | wy_legacy_us dinfer | wy_mma8_us dinfer | wy_serial_us dinfer | wy_us dinfer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x1024 | 161 | 487 | 114 | 39 | 218 | 41 | 62 | 37 | 70 | 79 | 89 | 109 | 65 | 65 | 114 | 99 | 791 | 18 | 422 | 65 | 111 | 76 |
| 8x1024 | 898 | 809 | 311 | 220 | 319 | 222 | 247 | 250 | 542 | 545 | 624 | 726 | 408 | 377 | 633 | 543 | 5018 | 82 | 2926 | 435 | 656 | 506 |
| 32x128 | 494 | 217 | 172 | 129 | 301 | 137 | 169 | 138 | 318 | 325 | 345 | 386 | 231 | 221 | 343 | 302 | 2502 | 42 | 1488 | 229 | 358 | 273 |
| 1x4096 | 534 | 3063 | 163 | 122 | 295 | 132 | 164 | 131 | 322 | 392 | 351 | 440 | 350 | 264 | 458 | 396 | 3302 | 43 | 1486 | 229 | 360 | 271 |
| 1x8192 | 1034 | 6173 | 308 | 219 | 316 | 220 | 247 | 249 | 634 | 782 | 683 | 865 | 692 | 521 | 901 | 778 | 6649 | 82 | 2926 | 435 | 657 | 506 |

- 1x1024 chain_us dinfer：r10 第 3 轮：wy 8 warp（Sm100 默认 Mma8）；B300-50 GPU 7
- 8x1024 chain_us dinfer：r10 第 3 轮：wy 8 warp（Sm100 默认 Mma8）；B300-50 GPU 7
- 32x128 chain_us dinfer：r10 第 3 轮：wy 8 warp（Sm100 默认 Mma8）；B300-50 GPU 7
- 1x4096 chain_us dinfer：r10 第 3 轮：wy 8 warp（Sm100 默认 Mma8）；B300-50 GPU 7
- 1x8192 chain_us dinfer：r10 第 3 轮：wy 8 warp（Sm100 默认 Mma8）；B300-50 GPU 7

## probe_gdn_r10_4（2026-10-20，B300）

### gdn_scale（us）

| x | chain_us dinfer | conv_rows0_us dinfer | conv_rows128_us dinfer | conv_rows16_us dinfer | conv_rows256_us dinfer | conv_rows32_us dinfer | conv_rows64_us dinfer | conv_rows8_us dinfer | ho_bv16_us dinfer | ho_bv16pf_us dinfer | ho_bv16pfsp_us dinfer | ho_bv16sp_us dinfer | ho_bv32_us dinfer | ho_bv32pf_us dinfer | ho_bv32pfsp_us dinfer | ho_bv32sp_us dinfer | ho_legacy_us dinfer | prep_us dinfer | wy_legacy_us dinfer | wy_mma8_us dinfer | wy_serial_us dinfer | wy_us dinfer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x1024 | 148 | 487 | 114 | 38 | 218 | 40 | 63 | 38 | 72 | 79 | 97 | 117 | 64 | 65 | 85 | 113 | 785 | 18 | 420 | 57 | 92 | 60 |
| 8x1024 | 822 | 810 | 311 | 221 | 319 | 221 | 248 | 249 | 548 | 548 | 653 | 760 | 373 | 380 | 573 | 570 | 5006 | 82 | 2926 | 360 | 509 | 372 |
| 32x128 | 452 | 218 | 171 | 128 | 301 | 136 | 167 | 137 | 326 | 324 | 360 | 399 | 213 | 221 | 323 | 309 | 2501 | 42 | 1487 | 187 | 275 | 194 |
| 1x4096 | 494 | 3064 | 162 | 122 | 296 | 131 | 164 | 131 | 329 | 391 | 380 | 473 | 252 | 267 | 336 | 461 | 3290 | 44 | 1488 | 187 | 276 | 194 |
| 1x8192 | 963 | 6172 | 309 | 219 | 316 | 221 | 246 | 249 | 643 | 785 | 744 | 929 | 497 | 524 | 659 | 904 | 6617 | 82 | 2924 | 360 | 509 | 371 |

- 1x1024 chain_us dinfer：r10 第 4 轮：wy stage 16 B 向量装载 + kg 16 B 写；B300-50 GPU 7，合并树（r8b 已并入）
- 8x1024 chain_us dinfer：r10 第 4 轮：wy stage 16 B 向量装载 + kg 16 B 写；B300-50 GPU 7，合并树（r8b 已并入）
- 32x128 chain_us dinfer：r10 第 4 轮：wy stage 16 B 向量装载 + kg 16 B 写；B300-50 GPU 7，合并树（r8b 已并入）
- 1x4096 chain_us dinfer：r10 第 4 轮：wy stage 16 B 向量装载 + kg 16 B 写；B300-50 GPU 7，合并树（r8b 已并入）
- 1x8192 chain_us dinfer：r10 第 4 轮：wy stage 16 B 向量装载 + kg 16 B 写；B300-50 GPU 7，合并树（r8b 已并入）

## probe_gdn_r10_5（2026-10-20，B300）

### gdn_scale（us）

| x | chain_us dinfer | conv_rows0_us dinfer | conv_rows128_us dinfer | conv_rows16_us dinfer | conv_rows256_us dinfer | conv_rows32_us dinfer | conv_rows64_us dinfer | conv_rows8_us dinfer | ho_bv16_us dinfer | ho_bv16pf_us dinfer | ho_bv16pfsp_us dinfer | ho_bv16sp_us dinfer | ho_bv32_us dinfer | ho_bv32pf_us dinfer | ho_bv32pfsp_us dinfer | ho_bv32sp_us dinfer | ho_legacy_us dinfer | prep_us dinfer | wy_legacy_us dinfer | wy_mma8_us dinfer | wy_serial_us dinfer | wy_us dinfer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x1024 | 147 | 487 | 115 | 39 | 218 | 41 | 62 | 38 | 74 | 77 | 105 | 112 | 63 | 65 | 85 | 102 | 786 | 18 | 421 | 57 | 92 | 60 |
| 8x1024 | 833 | 808 | 310 | 221 | 319 | 222 | 246 | 250 | 509 | 549 | 700 | 741 | 384 | 389 | 575 | 555 | 5005 | 82 | 2924 | 360 | 509 | 372 |
| 32x128 | 453 | 218 | 172 | 128 | 301 | 135 | 169 | 137 | 305 | 324 | 374 | 393 | 217 | 222 | 324 | 302 | 2497 | 42 | 1485 | 187 | 275 | 195 |
| 1x4096 | 497 | 3065 | 162 | 123 | 295 | 132 | 164 | 131 | 369 | 394 | 420 | 450 | 256 | 268 | 338 | 409 | 3294 | 43 | 1486 | 187 | 276 | 194 |
| 1x8192 | 966 | 6174 | 308 | 219 | 315 | 221 | 248 | 249 | 720 | 804 | 828 | 880 | 504 | 525 | 661 | 802 | 6615 | 82 | 2923 | 360 | 508 | 371 |

- 1x1024 chain_us dinfer：r10 第 5 轮：ho h16 B 操作数 ldmatrix.x4；B300-50 GPU 7
- 8x1024 chain_us dinfer：r10 第 5 轮：ho h16 B 操作数 ldmatrix.x4；B300-50 GPU 7
- 32x128 chain_us dinfer：r10 第 5 轮：ho h16 B 操作数 ldmatrix.x4；B300-50 GPU 7
- 1x4096 chain_us dinfer：r10 第 5 轮：ho h16 B 操作数 ldmatrix.x4；B300-50 GPU 7
- 1x8192 chain_us dinfer：r10 第 5 轮：ho h16 B 操作数 ldmatrix.x4；B300-50 GPU 7

## probe_gdn_r10_6（2026-10-20，B300）

### gdn_scale（us）

| x | chain_us dinfer | conv_rows0_us dinfer | conv_rows128_us dinfer | conv_rows16_us dinfer | conv_rows256_us dinfer | conv_rows32_us dinfer | conv_rows64_us dinfer | conv_rows8_us dinfer | ho_bv16_us dinfer | ho_bv16pf_us dinfer | ho_bv16pfsp_us dinfer | ho_bv16sp_us dinfer | ho_bv32_us dinfer | ho_bv32pf_us dinfer | ho_bv32pfsp_us dinfer | ho_bv32sp_us dinfer | ho_bv64_us dinfer | ho_bv64pf_us dinfer | ho_legacy_us dinfer | prep_us dinfer | wy_legacy_us dinfer | wy_mma8_us dinfer | wy_serial_us dinfer | wy_us dinfer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x1024 | 147 | 487 | 114 | 38 | 218 | 41 | 63 | 38 | 74 | 77 | 106 | 112 | 63 | 65 | 85 | 102 | 59 | 65 | 786 | 18 | 421 | 57 | 91 | 60 |
| 8x1024 | 829 | 810 | 311 | 221 | 320 | 222 | 248 | 250 | 510 | 549 | 700 | 741 | 382 | 388 | 573 | 556 | 308 | 327 | 5006 | 82 | 2922 | 360 | 508 | 372 |
| 32x128 | 454 | 218 | 171 | 128 | 301 | 137 | 167 | 138 | 305 | 324 | 375 | 392 | 216 | 222 | 324 | 302 | 180 | 190 | 2509 | 43 | 1484 | 187 | 276 | 194 |
| 1x4096 | 496 | 3066 | 162 | 121 | 297 | 132 | 162 | 131 | 365 | 395 | 420 | 447 | 256 | 266 | 338 | 408 | 302 | 319 | 3296 | 43 | 1489 | 187 | 276 | 195 |
| 1x8192 | 968 | 6161 | 308 | 220 | 314 | 221 | 246 | 249 | 727 | 780 | 827 | 883 | 504 | 525 | 660 | 798 | 595 | 632 | 6605 | 82 | 2926 | 360 | 508 | 371 |

- 1x1024 chain_us dinfer：r10 第 6 轮：ho BV=64 档（bench 列 bv64/bv64pf；默认档仍 32）；B300-50 GPU 7
- 8x1024 chain_us dinfer：r10 第 6 轮：ho BV=64 档（bench 列 bv64/bv64pf；默认档仍 32）；B300-50 GPU 7
- 32x128 chain_us dinfer：r10 第 6 轮：ho BV=64 档（bench 列 bv64/bv64pf；默认档仍 32）；B300-50 GPU 7
- 1x4096 chain_us dinfer：r10 第 6 轮：ho BV=64 档（bench 列 bv64/bv64pf；默认档仍 32）；B300-50 GPU 7
- 1x8192 chain_us dinfer：r10 第 6 轮：ho BV=64 档（bench 列 bv64/bv64pf；默认档仍 32）；B300-50 GPU 7

## probe_gdn_r10_7（2026-10-20，B300）

### gdn_scale（us）

| x | chain_us dinfer | conv_rows0_us dinfer | conv_rows128_us dinfer | conv_rows16_us dinfer | conv_rows256_us dinfer | conv_rows32_us dinfer | conv_rows64_us dinfer | conv_rows8_us dinfer | ho_bv16_us dinfer | ho_bv16pf_us dinfer | ho_bv16pfsp_us dinfer | ho_bv16sp_us dinfer | ho_bv32_us dinfer | ho_bv32pf_us dinfer | ho_bv32pfsp_us dinfer | ho_bv32sp_us dinfer | ho_bv64_us dinfer | ho_bv64pf_us dinfer | ho_legacy_us dinfer | prep_us dinfer | wy_legacy_us dinfer | wy_mma8_us dinfer | wy_serial_us dinfer | wy_us dinfer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x1024 | 147 | 487 | 114 | 38 | 217 | 40 | 62 | 37 | 74 | 77 | 106 | 111 | 63 | 65 | 85 | 102 | 59 | 65 | 786 | 18 | 421 | 57 | 94 | 60 |
| 8x1024 | 770 | 809 | 309 | 220 | 320 | 221 | 247 | 249 | 509 | 549 | 700 | 738 | 382 | 388 | 575 | 556 | 308 | 327 | 5008 | 82 | 2927 | 360 | 508 | 372 |
| 32x128 | 423 | 218 | 171 | 127 | 300 | 136 | 169 | 137 | 304 | 323 | 375 | 392 | 216 | 222 | 325 | 302 | 180 | 190 | 2502 | 43 | 1486 | 187 | 276 | 194 |
| 1x4096 | 495 | 3066 | 162 | 122 | 296 | 132 | 164 | 131 | 356 | 397 | 421 | 449 | 256 | 265 | 339 | 408 | 302 | 318 | 3289 | 42 | 1485 | 187 | 275 | 195 |
| 1x8192 | 969 | 6168 | 308 | 219 | 314 | 221 | 248 | 249 | 720 | 788 | 828 | 883 | 504 | 523 | 661 | 801 | 595 | 631 | 6613 | 82 | 2925 | 360 | 508 | 370 |

- 1x1024 chain_us dinfer：r10 第 7 轮：ArchProfile 按 block 铺满选 BV 64/32（Sm100）；默认路径；B300-50 GPU 7
- 8x1024 chain_us dinfer：r10 第 7 轮：ArchProfile 按 block 铺满选 BV 64/32（Sm100）；默认路径；B300-50 GPU 7
- 32x128 chain_us dinfer：r10 第 7 轮：ArchProfile 按 block 铺满选 BV 64/32（Sm100）；默认路径；B300-50 GPU 7
- 1x4096 chain_us dinfer：r10 第 7 轮：ArchProfile 按 block 铺满选 BV 64/32（Sm100）；默认路径；B300-50 GPU 7
- 1x8192 chain_us dinfer：r10 第 7 轮：ArchProfile 按 block 铺满选 BV 64/32（Sm100）；默认路径；B300-50 GPU 7

## probe_gdn_r10_8（2026-10-20，B300）

### gdn_scale（us）

| x | chain_us dinfer | conv_rows0_us dinfer | conv_rows128_us dinfer | conv_rows16_us dinfer | conv_rows256_us dinfer | conv_rows32_us dinfer | conv_rows64_us dinfer | conv_rows8_us dinfer | ho_bv16_us dinfer | ho_bv16pf_us dinfer | ho_bv16pfsp_us dinfer | ho_bv16sp_us dinfer | ho_bv32_us dinfer | ho_bv32pf_us dinfer | ho_bv32pfsp_us dinfer | ho_bv32sp_us dinfer | ho_bv64_us dinfer | ho_bv64pf_us dinfer | ho_legacy_us dinfer | prep_us dinfer | wy_legacy_us dinfer | wy_mma8_us dinfer | wy_serial_us dinfer | wy_us dinfer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x1024 | 147 | 486 | 114 | 38 | 218 | 40 | 63 | 37 | 74 | 76 | 106 | 112 | 63 | 65 | 85 | 102 | 59 | 65 | 786 | 18 | 421 | 57 | 93 | 60 |
| 8x1024 | 769 | 810 | 312 | 220 | 320 | 222 | 247 | 250 | 508 | 550 | 700 | 738 | 382 | 384 | 576 | 555 | 308 | 328 | 5012 | 82 | 2924 | 360 | 508 | 371 |
| 32x128 | 421 | 218 | 171 | 128 | 301 | 137 | 167 | 138 | 305 | 324 | 375 | 392 | 216 | 222 | 325 | 303 | 180 | 190 | 2496 | 43 | 1486 | 187 | 276 | 194 |
| 1x4096 | 495 | 3065 | 162 | 122 | 296 | 132 | 162 | 131 | 362 | 394 | 421 | 448 | 256 | 267 | 338 | 409 | 302 | 318 | 3289 | 42 | 1487 | 187 | 276 | 194 |
| 1x8192 | 967 | 6162 | 307 | 220 | 314 | 221 | 247 | 249 | 726 | 790 | 828 | 884 | 504 | 523 | 661 | 797 | 595 | 632 | 6598 | 82 | 2922 | 360 | 509 | 370 |

- 1x1024 chain_us dinfer：r10 第 8 轮：Sm8x wy 8 warp（B300 默认不变，回归确认）；B300-50 GPU 7
- 8x1024 chain_us dinfer：r10 第 8 轮：Sm8x wy 8 warp（B300 默认不变，回归确认）；B300-50 GPU 7
- 32x128 chain_us dinfer：r10 第 8 轮：Sm8x wy 8 warp（B300 默认不变，回归确认）；B300-50 GPU 7
- 1x4096 chain_us dinfer：r10 第 8 轮：Sm8x wy 8 warp（B300 默认不变，回归确认）；B300-50 GPU 7
- 1x8192 chain_us dinfer：r10 第 8 轮：Sm8x wy 8 warp（B300 默认不变，回归确认）；B300-50 GPU 7

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

## probe_gdn_r9_0b300（2026-10-20，B300）

### gdn_scale（us）

| x | chain_us dinfer | conv_us dinfer | fla_chunk_gdr_us vllm | ho_us dinfer | wy_us dinfer |
|---|---|---|---|---|---|
| 1x1024 | 311 | 41 | 359 | 160 | 134 |
| 8x1024 | 1662 | 222 |  | 664 | 912 |
| 32x128 | 845 |  |  |  |  |
| 1x3009 |  |  | 349 |  |  |
| 1x8192 |  |  | 853 |  |  |

- 1x1024 chain_us dinfer：gdn_chunk_bench GPU 6 sm_103；prep 18 / wy 134 / ho bv16 160 / conv rows32 41；r9 起点（第 5A 轮 mma 版）B300 基线，原误放在 probe_prefill_1
- 8x1024 chain_us dinfer：wy 912 / ho bv32 664 / conv rows32 222（5.3× 线性涨）；r9 起点（第 5A 轮 mma 版）B300 基线，原误放在 probe_prefill_1
- 32x128 chain_us dinfer：wy 461 / ho bv32 341 / conv 137；r9 起点（第 5A 轮 mma 版）B300 基线，原误放在 probe_prefill_1
- 1x1024 wy_us dinfer：r9 起点（第 5A 轮 mma 版）B300 基线，原误放在 probe_prefill_1
- 8x1024 wy_us dinfer：r9 起点（第 5A 轮 mma 版）B300 基线，原误放在 probe_prefill_1
- 1x1024 ho_us dinfer：bv16；r9 起点（第 5A 轮 mma 版）B300 基线，原误放在 probe_prefill_1
- 8x1024 ho_us dinfer：bv32；r9 起点（第 5A 轮 mma 版）B300 基线，原误放在 probe_prefill_1
- 1x1024 conv_us dinfer：rows32；r9 起点（第 5A 轮 mma 版）B300 基线，原误放在 probe_prefill_1
- 8x1024 conv_us dinfer：rows32；r9 起点（第 5A 轮 mma 版）B300 基线，原误放在 probe_prefill_1
- 1x1024 fla_chunk_gdr_us vllm：vLLM 容器 third_party fla chunk_gated_delta_rule 同形状 HV48 K=V=128 含 l2norm 50 次均值；早期 fla 单 kernel 总时 benchmark，正式分解见 probe_gdn_r9_8
- 1x3009 fla_chunk_gdr_us vllm：早期 fla 单 kernel 总时 benchmark，正式分解见 probe_gdn_r9_8
- 1x8192 fla_chunk_gdr_us vllm：1 K→8 K 只 2.4×；早期 fla 单 kernel 总时 benchmark，正式分解见 probe_gdn_r9_8

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

| x | chain_nol2_us vllm | chain_us dinfer | chain_us vllm | conv_rows0_us dinfer | conv_rows128_us dinfer | conv_rows16_us dinfer | conv_rows256_us dinfer | conv_rows32_us dinfer | conv_rows64_us dinfer | conv_rows8_us dinfer | ho_bv16_us dinfer | ho_bv16pf_us dinfer | ho_bv16pfsp_us dinfer | ho_bv16sp_us dinfer | ho_bv32_us dinfer | ho_bv32pf_us dinfer | ho_bv32pfsp_us dinfer | ho_bv32sp_us dinfer | ho_legacy_us dinfer | prep_us dinfer | wy_legacy_us dinfer | wy_serial_us dinfer | wy_us dinfer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x1024 | 289 | 208 | 360 | 486 | 114 | 38 | 217 | 41 | 62 | 37 | 115 | 142 | 132 | 166 | 87 | 115 | 106 | 133 | 851 | 18 | 434 | 136 | 102 |
| 8x1024 | 684 | 1109 | 745 | 809 | 310 | 219 | 320 | 222 | 247 | 250 | 694 | 755 | 792 | 984 | 498 | 648 | 672 | 642 | 5092 | 82 | 2957 | 685 | 532 |
| 32x128 | 380 | 593 | 410 | 218 | 172 | 127 | 300 | 136 | 166 | 138 | 403 | 418 | 455 | 503 | 265 | 350 | 366 | 348 | 2523 | 43 | 1503 | 371 | 283 |
| 1x4096 | 416 | 660 | 447 | 3063 | 162 | 122 | 296 | 132 | 161 | 131 | 459 | 549 | 510 | 640 | 334 | 442 | 405 | 511 | 3385 | 43 | 1503 | 372 | 283 |
| 1x8192 | 792 | 1271 | 853 | 6158 | 309 | 219 | 316 | 221 | 247 | 249 | 943 | 1086 | 1001 | 1265 | 659 | 875 | 792 | 1004 | 6926 | 82 | 2956 | 684 | 533 |

- 1x1024 chain_us dinfer：r9 第 7 轮：B300-50 GPU 6，新 ArchProfile 门限（ho bv 恒 32、conv Sm100 8 块/SM → rows16/32）；默认路径数
- 8x1024 chain_us dinfer：r9 第 7 轮：B300-50 GPU 6，新 ArchProfile 门限（ho bv 恒 32、conv Sm100 8 块/SM → rows16/32）；默认路径数
- 32x128 chain_us dinfer：r9 第 7 轮：B300-50 GPU 6，新 ArchProfile 门限（ho bv 恒 32、conv Sm100 8 块/SM → rows16/32）；默认路径数
- 1x4096 chain_us dinfer：r9 第 7 轮：B300-50 GPU 6，新 ArchProfile 门限（ho bv 恒 32、conv Sm100 8 块/SM → rows16/32）；默认路径数
- 1x8192 chain_us dinfer：r9 第 7 轮：B300-50 GPU 6，新 ArchProfile 门限（ho bv 恒 32、conv Sm100 8 块/SM → rows16/32）；默认路径数
- 1x1024 chain_us vllm：vLLM fla chunk_gated_delta_rule（nightly 0.26.1rc1 / torch 2.13 cu130，B300 GPU 6 容器，50 次均值）；chain_us 含 l2norm = vLLM 实际路径，chain_nol2_us 不含 ≈ 可比 dinfer wy+ho

## probe_gdn_r9_8（2026-10-20，B300）

### gdn_scale（us）

| x | fla_cumsum_nol2_us vllm | fla_cumsum_us vllm | fla_fill_nol2_us vllm | fla_fill_us vllm | fla_fwd_h_nol2_us vllm | fla_fwd_h_us vllm | fla_fwd_o_nol2_us vllm | fla_fwd_o_us vllm | fla_kkt_nol2_us vllm | fla_kkt_us vllm | fla_l2norm_us vllm | fla_total_nol2_us vllm | fla_total_us vllm | fla_tril_merge_nol2_us vllm | fla_tril_merge_us vllm | fla_wu_nol2_us vllm | fla_wu_us vllm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x1024 | 3 | 3 | 3 | 3 | 32 | 32 | 24 | 24 | 8 | 8 | 11 | 117 | 128 | 32 | 32 | 15 | 15 |
| 8x1024 | 6 | 6 | 14 | 14 | 165 | 165 | 167 | 167 | 46 | 47 | 59 | 685 | 746 | 189 | 189 | 98 | 98 |
| 1x8192 | 6 | 6 | 14 | 14 | 273 | 273 | 167 | 167 | 46 | 47 | 59 | 793 | 853 | 189 | 189 | 98 | 98 |

- 1x1024 fla_total_us vllm：fla per-kernel 分解：B300-50 GPU 6，vllm/vllm-openai:nightly 容器，torch.profiler 20 次均值，HV48 K=V128；对照见 loops/qwen35-gdn-r9.md "fla 分解"

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

## probe_prefill_3（2026-10-20，B300）

### ttft_len（ms）

| x | slope_us_per_token dinfer | slope_us_per_token dinfer_nosnap | ttft_p50_ms dinfer | ttft_p50_ms dinfer_nosnap |
|---|---|---|---|---|
| 510 |  |  |  | 28.0 |
| 1021 |  |  | 51.9 | 52.1 |
| 2029 |  |  |  | 96.7 |
| 3009 |  |  | 163.7 | 143.3 |
| 4059 |  |  |  | 192.3 |
| 6019 |  |  | 347.5 | 290.5 |
| 1k-6k | 59.3 | 47.5 |  |  |

- 6019 ttft_p50_ms dinfer_nosnap：bbdf147 无池不切步 + r9 kernel；6 K 步数 1
- 6019 ttft_p50_ms dinfer：默认池开 119 槽，6 K 仍 6 步（已知权衡，r8b 消）
- 1k-6k slope_us_per_token dinfer_nosnap：r9 前 51.5；目标 45
- 1k-6k slope_us_per_token dinfer：池开切步，3 点拟合

## probe_prefill_4（2026-10-20，B300）

### ttft_len（ms）

| x | slope_us_per_token dinfer | slope_us_per_token vllm | ttft_p50_ms dinfer | ttft_p50_ms vllm |
|---|---|---|---|---|
| 510 |  |  | 28.2 |  |
| 1021 |  |  | 52.3 | 83.0 |
| 2029 |  |  | 107.4 |  |
| 3009 |  |  | 163.1 | 138.6 |
| 4059 |  |  | 223.7 |  |
| 6019 |  |  | 346.8 | 260.0 |
| 1k-6k | 57.9 | 35.8 |  |  |

- 6019 ttft_p50_ms dinfer：b17 二进制（r8+r9+公共件），默认池开 119 槽，6 K 6 步
- 1021 ttft_p50_ms vllm：vLLM 容器 :8013 同时段
- 3009 ttft_p50_ms vllm：vLLM 容器 :8013 同时段
- 6019 ttft_p50_ms vllm：vLLM 容器 :8013 同时段
- 1k-6k slope_us_per_token dinfer：池开切步；8n 池关 47.5
- 1k-6k slope_us_per_token vllm：三点拟合，固定 40.6 ms

## probe_prefill_5（2026-10-20，B300）

### ttft_len（ms）

| x | engine_steps_per_request dinfer | ttft_min_ms dinfer | ttft_p50_ms dinfer |
|---|---|---|---|
| 3009 | 1 |  | 142.5 |
| 6019 | 1 | 284.7 | 291.8 |

- 3009 ttft_p50_ms dinfer：r8b 单发 1 次（池开 119 槽）
- 6019 ttft_p50_ms dinfer：r8b 5 次中位（284.7 289.6 291.8 292.3 291.8；池开 119 槽）
- 6019 ttft_min_ms dinfer：r8b
- 3009 engine_steps_per_request dinfer：r8b /metrics engine_steps_total 差值（r8 池开为 3）
- 6019 engine_steps_per_request dinfer：r8b /metrics engine_steps_total 差值（r8 池开为 6）

## probe_prefill_6（2026-10-20，B300）

### ttft_len（ms）

| x | slope_us_per_token dinfer | slope_us_per_token vllm | ttft_p50_ms dinfer | ttft_p50_ms vllm |
|---|---|---|---|---|
| 510 |  |  | 27.5 |  |
| 1021 |  |  | 50.3 | 83.4 |
| 2029 |  |  | 94.5 |  |
| 3009 |  |  | 138.1 | 139.0 |
| 4059 |  |  | 188.5 |  |
| 6019 |  |  | 283.9 | 260.6 |
| 1k-6k | 46.4 | 35.9 |  |  |

- 6019 ttft_p50_ms dinfer：合体 cb7ee9a（r8b + r10 r1–4），默认池开，6 K 1 步
- 1k-6k slope_us_per_token dinfer：r9 前 51.5；8n 池关 47.5；8p 池开切步 57.9

## probe_prefill_7（2026-10-20，B300）

### ttft_len（ms）

| x | slope_us_per_token dinfer | slope_us_per_token vllm | ttft_p50_ms dinfer | ttft_p50_ms vllm |
|---|---|---|---|---|
| 510 |  |  | 27.1 |  |
| 1021 |  |  | 50.1 | 82.8 |
| 2029 |  |  | 93.9 |  |
| 3009 |  |  | 139.5 | 138.0 |
| 4059 |  |  | 189.3 |  |
| 6019 |  |  | 284.9 | 260.6 |
| 1k-6k | 46.7 | 36.0 |  |  |

- 6019 ttft_p50_ms dinfer：第 12 轮合体定稿（r8b + r10 全 8 轮，2b0cc54），6 K 1 步
- 1k-6k slope_us_per_token dinfer：b18 46.4；r9 前 51.5
