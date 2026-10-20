#!/usr/bin/env python3
"""把 `gdn_chunk_bench --nocapture` 的输出行转成 probe.csv 长表（group gdn_scale，side dinfer）。

用法：python3 scripts/dev/gdn_bench_to_probe.py <bench.log> <probe名> <gpu> <日期> [备注] >> test/qwen35/<日期>/<probe名>/probe.csv
每个形状一组 x（如 1x8192），metric：chain_us / prep_us / wy_us（mma）/ wy_legacy_us / ho_bv16_us / ho_bv32_us / ho_legacy_us /
conv_rows{0,32,64,128,256}_us。首次调用请先写表头（--header）。
"""
import re, sys

COLS = "probe,date,gpu,model,group,metric,side,x,value,unit,note"
if len(sys.argv) >= 2 and sys.argv[1] == "--header":
    print(COLS); sys.exit(0)
log, probe, gpu, date = sys.argv[1:5]
note = sys.argv[5] if len(sys.argv) > 5 else ""
pat = re.compile(r"\[(\d+)×(\d+)\].*?chunk 全链 (\d+) µs.*?prep (\d+) \| wy old (\d+) mma (\d+) \| ho (.*?) \| conv (.*)$")
for line in open(log, encoding="utf-8"):
    m = pat.search(line.strip())
    if not m:
        continue
    x = f"{m.group(1)}x{m.group(2)}"
    rows = [("chain_us", m.group(3)), ("prep_us", m.group(4)), ("wy_legacy_us", m.group(5)), ("wy_us", m.group(6))]
    for kv in m.group(7).split():
        k, v = kv.split("=")
        rows.append((("ho_legacy_us" if k == "bv0" else f"ho_{k}_us"), v))
    for kv in m.group(8).split():
        k, v = kv.split("=")
        rows.append((f"conv_{k}_us", v))
    for i, (metric, v) in enumerate(rows):
        n = note if i == 0 else ""
        print(f"{probe},{date},{gpu},qwen35,gdn_scale,{metric},dinfer,{x},{v},us,{n}")
