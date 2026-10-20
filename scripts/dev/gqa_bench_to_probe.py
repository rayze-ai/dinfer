#!/usr/bin/env python3
"""把 `gqa_fa_bench --nocapture` 的输出行转成 probe.csv 长表（group attn_scale，side dinfer）。

用法：python3 scripts/dev/gqa_bench_to_probe.py <bench.log> <probe名> <gpu> <日期> [备注] [metric后缀] >> test/qwen35/<日期>/<probe名>/probe.csv
每个 T 一组 x（如 6019），metric：fa_us（默认路径）/ tflops；`metric后缀`（如 nbuf2）用于同轮的变体行：fa_nbuf2_us。
首次调用请先写表头（--header）。
"""
import re, sys

COLS = "probe,date,gpu,model,group,metric,side,x,value,unit,note"
if len(sys.argv) >= 2 and sys.argv[1] == "--header":
    print(COLS); sys.exit(0)
log, probe, gpu, date = sys.argv[1:5]
note = sys.argv[5] if len(sys.argv) > 5 else ""
suffix = sys.argv[6] if len(sys.argv) > 6 else ""
pat = re.compile(r"\[attn T=(\d+)\] fa_us=(\d+) tflops=([\d.]+)")
first = True
for line in open(log, encoding="utf-8"):
    m = pat.search(line.strip())
    if not m:
        continue
    x = m.group(1)
    mid = f"_{suffix}" if suffix else ""
    print(f"{probe},{date},{gpu},qwen35,attn_scale,fa{mid}_us,dinfer,{x},{m.group(2)},us,{note if first else ''}")
    print(f"{probe},{date},{gpu},qwen35,attn_scale,tflops{mid},dinfer,{x},{m.group(3)},TFLOP/s,")
    first = False
