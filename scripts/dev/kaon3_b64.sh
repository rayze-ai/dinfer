#!/usr/bin/env bash
# b64（E7，备用）：prefill 副流中间优先级。同二进制 ABAB：A = 默认（prefill least=0 / decode greatest=-5）  B = PREFILL_STREAM_PRIO=-2。
# 目标：换回 c32 TTFT 一部分而 TPOT 少丢。结果 tmp_b64ab_RESULT.txt，标记 B64AB_DONE。
TAG=b64ab AENV="PREFILL_ASYNC=1" BENV="PREFILL_ASYNC=1 PREFILL_STREAM_PRIO=-2" PORT=8363 bash /w/envab_c.sh
