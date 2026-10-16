#!/usr/bin/env python3
"""性能趋势：汇总 test/qwen3/*/bench*/ 的每轮 bench_compare 结果，输出
  test/qwen3/perf-trend.md    每轮一行：吞吐 / TTFT / TPOT（每组）、相对 vLLM 比值、相对上一轮的提升
  test/qwen3/perf-trend.json  同内容的机器可读版
  test/qwen3/perf-trend.svg   曲线（纯 SVG，不依赖 matplotlib）：每组 out tok/s 相对 vLLM 的比值 + TTFT / TPOT
每轮的"改了什么"读自 bench 目录里的 CHANGE 文件（一行文字）；没有就用目录名。
用法：python3 -I scripts/dev/bench_trend.py [--root test/qwen3]
"""
import glob, json, os, re, sys

ROOT = sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else "test/qwen3"
GROUPS = ["c1_p128_o128", "c8_p128_o128", "c32_p128_o128", "c8_p1024_o128"]
SHORT = {"c1_p128_o128": "c1", "c8_p128_o128": "c8", "c32_p128_o128": "c32", "c8_p1024_o128": "c8-long"}

def load_round(d):
    r = {"dir": d, "label": os.path.basename(d), "date": os.path.basename(os.path.dirname(d)), "groups": {}}
    chg = os.path.join(d, "CHANGE")
    r["change"] = open(chg).read().strip() if os.path.exists(chg) else ""
    for g in GROUPS:
        fa, fb = os.path.join(d, f"dinfer_{g}.json"), os.path.join(d, f"vllm_{g}.json")
        if not (os.path.exists(fa) and os.path.exists(fb)):
            return None
        a, b = json.load(open(fa)), json.load(open(fb))
        r["groups"][g] = {
            "tps": a["output_token_throughput_tps"], "vllm_tps": b["output_token_throughput_tps"],
            "ttft": a["ttft_ms"]["p50"], "vllm_ttft": b["ttft_ms"]["p50"],
            "tpot": a["tpot_ms"]["p50"], "vllm_tpot": b["tpot_ms"]["p50"],
        }
        r["groups"][g]["ratio"] = r["groups"][g]["tps"] / r["groups"][g]["vllm_tps"]
    return r

def round_key(d):
    m = re.search(r"bench(\d*)$", os.path.basename(d))
    return (os.path.basename(os.path.dirname(d)), int(m.group(1) or 0))

dirs = sorted((d for d in glob.glob(os.path.join(ROOT, "*", "bench*")) if os.path.isdir(d)), key=round_key)
rounds = [r for r in (load_round(d) for d in dirs) if r]
if not rounds:
    sys.exit("no bench rounds found")

def pct(new, old, lower_better=False):
    if old == 0: return "—"
    v = (old - new) / old * 100 if lower_better else (new - old) / old * 100
    return f"{v:+.0f}%"

lines = ["# Qwen3-0.6B 性能趋势（dinfer vs vLLM v0.31.0，4090，`bench_compare.sh` 四组）", "",
         "每轮一行；括号内是相对上一轮的变化（吞吐 ↑ 为好，TTFT/TPOT ↓ 为好）。比值 = dinfer out tok/s ÷ vLLM。vLLM 各轮自己也有 ±3% 抖动。", ""]
hdr = "| 轮 | 日期 | 改动 | " + " | ".join(f"{SHORT[g]} tok/s (×vLLM)" for g in GROUPS) + " | " + " | ".join(f"{SHORT[g]} TTFT ms" for g in GROUPS) + " | " + " | ".join(f"{SHORT[g]} TPOT ms" for g in GROUPS) + " |"
lines += [hdr, "|" + "---|" * (3 + 3 * len(GROUPS))]
prev = None
for r in rounds:
    cells = [r["label"], r["date"], r["change"] or "—"]
    for g in GROUPS:
        x = r["groups"][g]; d = f" ({pct(x['tps'], prev['groups'][g]['tps'])})" if prev else ""
        cells.append(f"{x['tps']:.0f} ({x['ratio']:.2f}×){d}")
    for g in GROUPS:
        x = r["groups"][g]; d = f" ({pct(x['ttft'], prev['groups'][g]['ttft'], True)})" if prev else ""
        cells.append(f"{x['ttft']:.1f} / {x['vllm_ttft']:.1f}{d}")
    for g in GROUPS:
        x = r["groups"][g]; d = f" ({pct(x['tpot'], prev['groups'][g]['tpot'], True)})" if prev else ""
        cells.append(f"{x['tpot']:.2f} / {x['vllm_tpot']:.2f}{d}")
    lines.append("| " + " | ".join(cells) + " |")
    prev = r
first, last = rounds[0], rounds[-1]
lines += ["", "## 首轮 → 最新", "", "| 组 | out tok/s | ×vLLM | TTFT ms | TPOT ms |", "|---|---|---|---|---|"]
for g in GROUPS:
    a, b = first["groups"][g], last["groups"][g]
    lines.append(f"| {SHORT[g]} | {a['tps']:.0f} → {b['tps']:.0f} ({pct(b['tps'], a['tps'])}) | {a['ratio']:.2f} → {b['ratio']:.2f} | {a['ttft']:.1f} → {b['ttft']:.1f} ({pct(b['ttft'], a['ttft'], True)}) | {a['tpot']:.2f} → {b['tpot']:.2f} ({pct(b['tpot'], a['tpot'], True)}) |")
lines += ["", "![trend](perf-trend.svg)", ""]
open(os.path.join(ROOT, "perf-trend.md"), "w").write("\n".join(lines))
json.dump(rounds, open(os.path.join(ROOT, "perf-trend.json"), "w"), indent=1, ensure_ascii=False)

# ---- SVG：三张小图（ratio、TTFT、TPOT），x = 轮次 ----
W, H, PAD = 980, 260, 48
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
def panel(y0, title, key, lower_better, ref_key=None, ylabel=""):
    n = len(rounds)
    vals = {g: [r["groups"][g][key] for r in rounds] for g in GROUPS}
    allv = [v for g in GROUPS for v in vals[g]] + ([r["groups"][g][ref_key] for r in rounds for g in GROUPS] if ref_key else [])
    ymax = max(allv) * 1.1 or 1
    def X(i): return PAD + i * (W - 2 * PAD) / max(n - 1, 1)
    def Y(v): return y0 + H - PAD - (v / ymax) * (H - 2 * PAD)
    out = [f'<text x="{PAD}" y="{y0 + 18}" font-size="14" font-weight="bold">{title}</text>',
           f'<line x1="{PAD}" y1="{Y(0)}" x2="{W - PAD}" y2="{Y(0)}" stroke="#888"/>',
           f'<line x1="{PAD}" y1="{y0 + PAD}" x2="{PAD}" y2="{Y(0)}" stroke="#888"/>',
           f'<text x="{PAD - 6}" y="{y0 + PAD + 4}" font-size="10" text-anchor="end">{ymax:.2f}</text>',
           f'<text x="{PAD - 6}" y="{Y(0) + 4}" font-size="10" text-anchor="end">0</text>']
    if key == "ratio":
        out.append(f'<line x1="{PAD}" y1="{Y(1.0)}" x2="{W - PAD}" y2="{Y(1.0)}" stroke="#444" stroke-dasharray="4 3"/>')
        out.append(f'<text x="{W - PAD + 4}" y="{Y(1.0) + 4}" font-size="10">vLLM = 1.0</text>')
    for ci, g in enumerate(GROUPS):
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals[g]))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{COLORS[ci]}" stroke-width="2"/>')
        for i, v in enumerate(vals[g]):
            out.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3" fill="{COLORS[ci]}"/>')
        if ref_key:
            pts = " ".join(f"{X(i):.1f},{Y(r['groups'][g][ref_key]):.1f}" for i, r in enumerate(rounds))
            out.append(f'<polyline points="{pts}" fill="none" stroke="{COLORS[ci]}" stroke-width="1" stroke-dasharray="3 3" opacity="0.6"/>')
        out.append(f'<text x="{W - PAD + 4}" y="{Y(vals[g][-1]) + 4}" font-size="10" fill="{COLORS[ci]}">{SHORT[g]}</text>')
    for i, r in enumerate(rounds):
        out.append(f'<text x="{X(i):.1f}" y="{Y(0) + 14}" font-size="9" text-anchor="middle">{r["label"]}</text>')
    return out
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{3 * H}" font-family="sans-serif">',
       f'<rect width="{W}" height="{3 * H}" fill="white"/>']
svg += panel(0, "out tok/s ÷ vLLM（越高越好；虚线 = 1.0）", "ratio", False)
svg += panel(H, "TTFT p50 ms（实线 dinfer，虚线 vLLM；越低越好）", "ttft", True, "vllm_ttft")
svg += panel(2 * H, "TPOT p50 ms（实线 dinfer，虚线 vLLM；越低越好）", "tpot", True, "vllm_tpot")
svg.append("</svg>")
open(os.path.join(ROOT, "perf-trend.svg"), "w").write("\n".join(svg))
print(f"{len(rounds)} rounds → {ROOT}/perf-trend.{{md,json,svg}}")
print("\n".join(lines[4:4 + 2 + len(rounds)]))
