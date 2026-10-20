#!/usr/bin/env python3
"""bench_engine.sh 的一行 summary：bench JSON + 服务 /metrics 前后快照 → Markdown 表格行（客户端测得 + 服务端自报 + 压测端开销）。
  python3 -I scripts/dev/engine_row.py <bench.json> <rate> <side> <metrics.before> <metrics.after>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I 不读 PYTHONPATH，自己加同目录
import metrics_delta as md  # noqa: E402

d = json.load(open(sys.argv[1]))
rate, side = sys.argv[2], sys.argv[3]
f = lambda x: f"{x:.1f}"


def four(p):
    return f"{f(p['p50'])}/{f(p['p90'])}/{f(p['p99'])}/{f(p['max'])}"


gp = d.get("goodput_rps")
gp = "-" if gp is None else f"{gp:.2f} ({d['good_requests']}/{d['completed']})"
ch = d.get("prefix_cache_hit_ratio")
ch = "n/a" if ch is None else f"{ch:.3f}"
try:
    srv = md.summarize(sys.argv[4], sys.argv[5])
except Exception:  # 服务没有 /metrics 或快照为空
    srv = {}
pc = srv.get("prefix_cache")
pc = "-" if not pc else (f"{pc['hit_ratio']:.3f}" if pc["queries"] else "0q")
pre = srv.get("preemptions"); pre = "-" if pre is None else str(pre)
c = d.get("client") or {}
cl = "-" if not c else f"{c.get('cpu_cores_avg', 0):.2f}/{c.get('max_rss_mib', 0):.0f}/{c.get('parse_ms_per_request', 0):.2f}"
print(f"| {rate} | {side} | {d['completed']}/{d['failed']} | {gp} | {d['request_throughput_rps']:.2f} | {f(d['output_token_throughput_tps'])} | "
      f"{four(d['ttft_ms'])} | {f(d['ttft_first_chunk_ms']['p50'])} | {four(d['tpot_ms'])} | {ch} | {d['finished_by_length']}/{d['completed']} | "
      f"{md.fmt_hist(srv.get('ttft'))} | {md.fmt_hist(srv.get('tpot'))} | {pc} | {pre} | {md.fmt_steps(srv.get('steps'))} | {cl} |")
