#!/usr/bin/env bash
# b56（D221）：同二进制 ABAB —— A = XSTREAM=1（pair 步：decode 主流图 + prefill 副流 eager，decode 等 prefill）
#   vs B = XSTREAM=0（单流 Paged 混批步：prefill 行 + decode 行一个 eager 前向）。看哪种结构对 c16–c64 TPOT/TTFT 更好。
# 前置：GPU0 空（nvidia-smi < 500 MiB）。结果 tmp_b56ab_RESULT.txt，标记 B56AB_DONE。
TAG=b56ab AENV="XSTREAM=1" BENV="XSTREAM=0" PORT=8352 bash /w/envab_c.sh
