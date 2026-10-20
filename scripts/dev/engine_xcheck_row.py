#!/usr/bin/env python3
"""bench_engine_vllm_tool.sh 的对账行：我们的 bench JSON vs vllm bench serve JSON（同 rate / seed / 长度）→ Markdown 行，差 > 阈值标 ⚠。
  python3 -I scripts/dev/engine_xcheck_row.py <ours.json> <tool.json> <rate> <side> <thresh_pct>
"""
import json
import sys

o, t = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
rate, lab, th = sys.argv[3], sys.argv[4], float(sys.argv[5])


def cmp(a, b):
    if b in (None, 0) or a is None:
        return f"{a}/{b} (-)"
    d = (a - b) / b * 100
    return f"{a:.1f}/{b:.1f} ({d:+.0f}%{' ⚠' if abs(d) > th else ''})"


tool_prompt = (t.get("total_input_tokens") or 0) / max(t.get("completed") or 1, 1)
print(f"| {rate} | {lab} | {cmp(o['ttft_first_chunk_ms']['p50'], t.get('p50_ttft_ms'))} | {cmp(o['ttft_first_chunk_ms']['p99'], t.get('p99_ttft_ms'))} | "
      f"{cmp(o['tpot_ms']['p50'], t.get('p50_tpot_ms'))} | {cmp(o['tpot_ms']['p99'], t.get('p99_tpot_ms'))} | "
      f"{o['request_throughput_rps']:.2f}/{(t.get('request_throughput') or 0):.2f} | {o['prompt_tokens_mean']:.0f}/{tool_prompt:.0f} |")
