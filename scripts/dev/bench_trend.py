#!/usr/bin/env python3
"""性能趋势：汇总 test/qwen3/*/ 下的每轮结果，输出
  test/qwen3/perf-trend.md    single（主线 benchN，dinfer vs vLLM）每轮一行 + pd / tp / dp / other 各一节（同 HTML 页签名）
  test/qwen3/perf-trend.json  主线轮次的机器可读版
  test/qwen3/perf-trend.svg   主线三张曲线 + 每类 ab 组一行柱图（纯 SVG，不依赖 matplotlib）
  test/qwen3/perf-trend.csv   长表；列 kind = main|pd|tp|dp|other，baseline / variant 写清谁比谁
  test/qwen3/perf-trend.html  Chart.js 面板，切页：single（默认）/ pd / tp / dp / other，URL hash（#pd）直达
数据来源：
  bench<N>/           主线：dinfer_<group>.json vs vllm_<group>.json（四组），CHANGE 一行改动说明
  bench_<tag>/        多卡（bench_tp2 / bench_dp2）：同上文件名，但不进主线曲线，按 tag 归 tp / dp 页
  ab_<name>/          A/B：<variant>__<group>.json；有 single 以它作基线，否则字典序第一个 variant（pd vs single / tier16 vs tier00）
  poisson*/           开环泊松：dinfer_xs1 / dinfer_xs0 vs vllm，组 = r<rate>；归 other 页
类别判定：目录名前缀 / CHANGE 文本（bench\\d* → single；ab_pd* 或含 "pd" → pd；tp → tp；dp → dp；其余 other）。
用法：python3 -I scripts/dev/bench_trend.py [--root test/qwen3]
"""
import csv, glob, json, os, re, sys

ROOT = sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else "test/qwen3"
GROUPS = ["c1_p128_o128", "c8_p128_o128", "c32_p128_o128", "c8_p1024_o128"]
SHORT = {"c1_p128_o128": "c1", "c8_p128_o128": "c8", "c32_p128_o128": "c32", "c8_p1024_o128": "c8-long"}
CATS = ["pd", "tp", "dp", "other"]
CAT_HINT = {"single": "基线 = vLLM v0.31.0 单卡（同机同配置）", "pd": "基线 = single（单机 :8012，同卡同配置的 dinfer 单实例）；变量 = PD 分离的传输方式 / 卡分布",
            "tp": "基线 = vLLM 单卡；变量 = dinfer --tp N（多卡张量并行）", "dp": "基线 = vLLM 单卡；变量 = dinfer --dp N（同进程数据并行）",
            "other": "基线 = 每个目录字典序第一个 variant（见 baseline 列）"}


def read_change(d):
    c = os.path.join(d, "CHANGE")
    return open(c).read().strip().replace("\n", " ") if os.path.exists(c) else ""


def category(label, change=""):
    if re.fullmatch(r"bench\d*", label):
        return "single"
    s = (label + " " + change).lower()
    if label.startswith("ab_pd") or re.search(r"\bpd\b|pd 分离|pd1", s):
        return "pd"
    if re.search(r"\btp\s*\d|--tp\b|_tp\d", s):
        return "tp"
    if re.search(r"\bdp\s*\d|--dp\b|_dp\d", s):
        return "dp"
    return "other"


def load_round(d):
    r = {"dir": d, "label": os.path.basename(d), "date": os.path.basename(os.path.dirname(d)), "groups": {}}
    r["change"] = read_change(d)
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
    r["kind"] = category(r["label"], r["change"])
    return r


def round_key(d):
    # benchN（数字轮次）或 bench_<tag>（如 bench_tp2 / bench_dp2，多卡轮次）；带 tag 的排在同日数字轮之后
    b = os.path.basename(d)
    m = re.search(r"bench(\d*)$", b)
    if m:
        return (os.path.basename(os.path.dirname(d)), 0, int(m.group(1) or 0), "")
    return (os.path.basename(os.path.dirname(d)), 1, 0, b)


dirs = sorted((d for d in glob.glob(os.path.join(ROOT, "*", "bench*")) if os.path.isdir(d)), key=round_key)
all_rounds = [r for r in (load_round(d) for d in dirs) if r]
rounds = [r for r in all_rounds if r["kind"] == "single"]           # 主线
multi_rounds = [r for r in all_rounds if r["kind"] != "single"]     # bench_tp2 / bench_dp2 …
if not rounds:
    sys.exit("no bench rounds found")


def pct(new, old, lower_better=False):
    if old == 0: return "—"
    v = (old - new) / old * 100 if lower_better else (new - old) / old * 100
    return f"{v:+.0f}%"


# ---------------------------------------------------------------------------
# 附加实验（ab_* / bench_<tag> / poisson*）统一成 entries：每条 = 一个目录 × 一组，带基线与若干 variant
#   {"kind", "round", "date", "change", "group", "baseline", "base": metrics, "variants": [(name, metrics)]}
# 用户要求（2026-10-17）：所有结果都要能在 CSV / 曲线上看到，不能只留 JSON。
# ---------------------------------------------------------------------------
def _m(j):
    return {"out_tok_per_s": j["output_token_throughput_tps"], "ttft_p50_ms": j["ttft_ms"]["p50"],
            "ttft_p99_ms": j["ttft_ms"]["p99"], "tpot_p50_ms": j["tpot_ms"]["p50"], "failed": j.get("failed", 0)}


BETTER = {"out_tok_per_s": "higher", "ttft_p50_ms": "lower", "ttft_p99_ms": "lower", "tpot_p50_ms": "lower", "failed": "lower"}


def extra_entries():
    es = []
    for r in multi_rounds:  # bench_tp2 / bench_dp2：dinfer(多卡) vs vLLM 单卡
        for g in GROUPS:
            a, b = json.load(open(os.path.join(r["dir"], f"dinfer_{g}.json"))), json.load(open(os.path.join(r["dir"], f"vllm_{g}.json")))
            es.append({"kind": r["kind"], "round": r["label"], "date": r["date"], "change": r["change"], "group": SHORT[g],
                       "baseline": "vllm", "base": _m(b), "variants": [("dinfer", _m(a))]})
    for d in sorted(glob.glob(os.path.join(ROOT, "*", "poisson*"))):
        if not os.path.isdir(d):
            continue
        rnd, date, chg = os.path.basename(d), os.path.basename(os.path.dirname(d)), read_change(d) or os.path.basename(d)
        rates = sorted({re.sub(r".*_r(\d+)\.json$", r"\1", f) for f in glob.glob(os.path.join(d, "vllm_r*.json"))}, key=int)
        for rate in rates:
            v = json.load(open(os.path.join(d, f"vllm_r{rate}.json")))
            vs = []
            for side in ("dinfer_xs1", "dinfer_xs0"):
                f = os.path.join(d, f"{side}_r{rate}.json")
                if os.path.exists(f):
                    vs.append((side, _m(json.load(open(f)))))
            if vs:
                es.append({"kind": category(rnd, chg), "round": rnd, "date": date, "change": chg, "group": f"r{rate}",
                           "baseline": "vllm", "base": _m(v), "variants": vs})
    for d in sorted(glob.glob(os.path.join(ROOT, "*", "ab_*"))):
        if not os.path.isdir(d):
            continue
        rnd, date, chg = os.path.basename(d), os.path.basename(os.path.dirname(d)), read_change(d) or os.path.basename(d)
        files = glob.glob(os.path.join(d, "*.json"))
        variants = sorted({os.path.basename(f).split("__")[0] for f in files if "__" in os.path.basename(f)})
        if len(variants) < 2:
            continue
        # 基线：有 single（单机）就以它为基线（pd vs single 等），否则字典序第一个
        base = "single" if "single" in variants else variants[0]
        variants = [base] + [v for v in variants if v != base]
        groups = sorted({os.path.basename(f).split("__")[1][:-5] for f in files if "__" in os.path.basename(f)})
        for g in groups:
            fb = os.path.join(d, f"{base}__{g}.json")
            if not os.path.exists(fb):
                continue
            vs = [(var, _m(json.load(open(os.path.join(d, f"{var}__{g}.json"))))) for var in variants[1:]
                  if os.path.exists(os.path.join(d, f"{var}__{g}.json"))]
            if vs:
                es.append({"kind": category(rnd, chg), "round": rnd, "date": date, "change": chg, "group": SHORT.get(g, g),
                           "baseline": base, "base": _m(json.load(open(fb))), "variants": vs})
    return es


extras = extra_entries()
by_cat = {c: [e for e in extras if e["kind"] == c] for c in CATS}

# ---------------------------------------------------------------------------
# MD：single 主表 + 首轮→最新 + 每类一节
# ---------------------------------------------------------------------------
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
main_md_len = len(lines)
for c in CATS:
    es = by_cat[c]
    if not es:
        continue
    lines += [f"## {c}", "", CAT_HINT[c] + "。每行 = 一个目录 × 一组 × 一个 variant；括号内 = 相对基线（↑ 为好 / TTFT、TPOT ↓ 为好）。", "",
              "| 目录 | 日期 | 改动 | 组 | 基线 | variant | out tok/s | TTFT p50 ms | TPOT p50 ms |", "|---|---|---|---|---|---|---|---|---|"]
    for e in es:
        b = e["base"]
        lines.append(f"| {e['round']} | {e['date']} | {e['change'] or '—'} | {e['group']} | {e['baseline']} | {e['baseline']}（基线） | {b['out_tok_per_s']:.0f} | {b['ttft_p50_ms']:.1f} | {b['tpot_p50_ms']:.2f} |")
        for name, a in e["variants"]:
            lines.append(f"| | | | {e['group']} | {e['baseline']} | {name} | {a['out_tok_per_s']:.0f} ({pct(a['out_tok_per_s'], b['out_tok_per_s'])}) | "
                         f"{a['ttft_p50_ms']:.1f} ({pct(a['ttft_p50_ms'], b['ttft_p50_ms'], True)}) | {a['tpot_p50_ms']:.2f} ({pct(a['tpot_p50_ms'], b['tpot_p50_ms'], True)}) |")
    lines.append("")
open(os.path.join(ROOT, "perf-trend.md"), "w").write("\n".join(lines))
json.dump(rounds, open(os.path.join(ROOT, "perf-trend.json"), "w"), indent=1, ensure_ascii=False)

# ---------------------------------------------------------------------------
# SVG：主线三张曲线（只画 benchN）+ 每类一行三张柱图（tok/s / TTFT / TPOT，基线灰、variant 彩）
# ---------------------------------------------------------------------------
W, H, PAD = 980, 260, 48
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def panel(y0, title, key, lower_better, ref_key=None, ylabel=""):
    n = len(rounds)
    vals = {g: [r["groups"][g][key] for r in rounds] for g in GROUPS}
    allv = [v for g in GROUPS for v in vals[g]] + ([r["groups"][g][ref_key] for r in rounds for g in GROUPS] if ref_key else [])
    ymax = max(allv) * 1.1 or 1
    def X(i): return PAD + i * (W - 2 * PAD) / max(n - 1, 1)
    def Y(v): return y0 + H - PAD - (v / ymax) * (H - 2 * PAD)
    out = [f'<text x="{PAD}" y="{y0 + 18}" font-size="14" font-weight="bold">{esc(title)}</text>',
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
        out.append(f'<text x="{X(i):.1f}" y="{Y(0) + 14}" font-size="9" text-anchor="middle">{esc(r["label"])}</text>')
    return out


def bar_panel(x0, y0, w, h, title, labels, series, fmt):
    """series: [(name, color, [v per label])]；每个 label 一簇柱。"""
    pad = 34
    n, m = len(labels), len(series)
    ymax = max((v for _, _, vs in series for v in vs if v is not None), default=1) * 1.18 or 1
    gw = (w - 2 * pad) / max(n, 1)
    bw = gw / (m + 1)
    def Y(v): return y0 + h - pad - (v / ymax) * (h - 2 * pad - 14)
    out = [f'<text x="{x0 + pad}" y="{y0 + 14}" font-size="12" font-weight="bold">{esc(title)}</text>',
           f'<line x1="{x0 + pad}" y1="{Y(0)}" x2="{x0 + w - pad}" y2="{Y(0)}" stroke="#888"/>']
    for si, (name, color, vs) in enumerate(series):
        for i, v in enumerate(vs):
            if v is None:
                continue
            x = x0 + pad + i * gw + (si + 0.5) * bw
            out.append(f'<rect x="{x:.1f}" y="{Y(v):.1f}" width="{bw * 0.9:.1f}" height="{Y(0) - Y(v):.1f}" fill="{color}"/>')
            out.append(f'<text x="{x + bw * 0.45:.1f}" y="{Y(v) - 2:.1f}" font-size="8" text-anchor="middle">{fmt(v)}</text>')
        out.append(f'<rect x="{x0 + pad + si * 70:.1f}" y="{y0 + 20}" width="8" height="8" fill="{color}"/>')
        out.append(f'<text x="{x0 + pad + si * 70 + 11:.1f}" y="{y0 + 28}" font-size="9">{esc(name[:12])}</text>')
    for i, lab in enumerate(labels):
        cx = x0 + pad + (i + 0.5) * gw
        for li, part in enumerate(lab.split("\n")[:2]):
            out.append(f'<text x="{cx:.1f}" y="{Y(0) + 10 + li * 9}" font-size="7.5" text-anchor="middle">{esc(part[:22])}</text>')
    return out


def cat_svg(y0, c, es):
    out = [f'<text x="{PAD}" y="{y0 + 20}" font-size="15" font-weight="bold">{c}</text>',
           f'<text x="{PAD + 40}" y="{y0 + 20}" font-size="11" fill="#555">{esc(CAT_HINT[c])}</text>']
    labels = [f"{e['round']}\n{e['group']}" for e in es]
    vnames = []
    for e in es:
        for name, _ in e["variants"]:
            if name not in vnames:
                vnames.append(name)
    base_name = "/".join(sorted({e["baseline"] for e in es}))
    pw = W / 3
    for pi, (key, title, fmt) in enumerate([("out_tok_per_s", "out tok/s（↑ 好）", lambda v: f"{v:.0f}"),
                                             ("ttft_p50_ms", "TTFT p50 ms（↓ 好）", lambda v: f"{v:.0f}"),
                                             ("tpot_p50_ms", "TPOT p50 ms（↓ 好）", lambda v: f"{v:.2f}")]):
        series = [(f"{base_name}（基线）", "#9a9a9a", [e["base"][key] for e in es])]
        for vi, vn in enumerate(vnames):
            series.append((vn, COLORS[vi % len(COLORS)], [dict(e["variants"]).get(vn, {}).get(key) for e in es]))
        out += bar_panel(pi * pw, y0 + 26, pw, H - 26, title, labels, series, fmt)
    return out


cats_with_data = [c for c in CATS if by_cat[c]]
total_h = 3 * H + 24 + len(cats_with_data) * H
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{total_h}" font-family="sans-serif">',
       f'<rect width="{W}" height="{total_h}" fill="white"/>',
       f'<text x="{PAD}" y="{14}" font-size="11" fill="#555">single：主线 benchN，dinfer vs vLLM 单卡（下方 pd / tp / dp / other 各一行，基线见各节）</text>']
svg += panel(24, "out tok/s ÷ vLLM（越高越好；虚线 = 1.0）", "ratio", False)
svg += panel(24 + H, "TTFT p50 ms（实线 dinfer，虚线 vLLM；越低越好）", "ttft", True, "vllm_ttft")
svg += panel(24 + 2 * H, "TPOT p50 ms（实线 dinfer，虚线 vLLM；越低越好）", "tpot", True, "vllm_tpot")
for i, c in enumerate(cats_with_data):
    svg += cat_svg(24 + 3 * H + i * H, c, by_cat[c])
svg.append("</svg>")
open(os.path.join(ROOT, "perf-trend.svg"), "w").write("\n".join(svg))

# ---------------------------------------------------------------------------
# CSV（长表，用户要求：每轮每指标的提升 + 改动说明）。kind / baseline / variant 三列说明谁比谁。
# ---------------------------------------------------------------------------
with open(os.path.join(ROOT, "perf-trend.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["round", "date", "change", "group", "metric", "dinfer", "vllm", "ratio_dinfer_over_vllm", "delta_pct_vs_prev_round", "better_is",
                "kind", "baseline", "variant"])
    prev = None
    for r in rounds:
        for g in GROUPS:
            x = r["groups"][g]
            for metric, key, vkey, lower in (("out_tok_per_s", "tps", "vllm_tps", False), ("ttft_p50_ms", "ttft", "vllm_ttft", True), ("tpot_p50_ms", "tpot", "vllm_tpot", True)):
                d = ""
                if prev:
                    o, nv = prev["groups"][g][key], x[key]
                    d = f"{((o - nv) / o * 100 if lower else (nv - o) / o * 100):.1f}" if o else ""
                ratio = x[key] / x[vkey] if x[vkey] else ""
                w.writerow([r["label"], r["date"], r["change"], SHORT[g], metric, f"{x[key]:.3f}", f"{x[vkey]:.3f}", f"{ratio:.3f}" if ratio != "" else "", d,
                            "lower" if lower else "higher", "main", "vllm", "dinfer"])
        prev = r
    # 附加实验：dinfer 列 = variant，vllm 列 = 基线
    for e in extras:
        for name, a in e["variants"]:
            for k, av in a.items():
                bv = e["base"][k]
                w.writerow([e["round"], e["date"], e["change"], e["group"], k, f"{av:.3f}", f"{bv:.3f}", f"{(av / bv):.3f}" if bv else "", "", BETTER[k],
                            e["kind"], e["baseline"], name])

# ---------------------------------------------------------------------------
# HTML 面板：静态页，fetch 同目录 perf-trend.csv，每 10 s 重画；Chart.js 走 cdnjs；切页纯 JS
# ---------------------------------------------------------------------------
html = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>dinfer perf trend</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--card:#f6f7f9;--line:#e3e5e8;--acc:#1f77b4}
@media (prefers-color-scheme: dark){:root:not([data-theme=light]){--bg:#111;--fg:#eee;--muted:#9a9a9a;--card:#1b1d20;--line:#2a2d31}}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
h1{font-size:18px;margin:0 0 4px}.sub{color:var(--muted);margin-bottom:12px}
.tabs{display:flex;gap:4px;border-bottom:1px solid var(--line);margin:8px 0 12px;flex-wrap:wrap}
.tabs button{background:none;border:1px solid transparent;border-bottom:none;color:var(--muted);padding:6px 14px;font:inherit;cursor:pointer;border-radius:6px 6px 0 0}
.tabs button.on{color:var(--fg);border-color:var(--line);background:var(--card);font-weight:600}
.tabs button[hidden]{display:none}
.page[hidden]{display:none}
.base{padding:6px 10px;margin-bottom:12px;border-left:3px solid var(--acc);background:var(--card);font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px}
.card h2{font-size:14px;margin:0 0 6px}
table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:right;vertical-align:top}
th:first-child,td:first-child,td.change,th.change,td.l,th.l{text-align:left}
.up{color:#1a7f37}.down{color:#c62828}.chg{max-width:420px;white-space:normal;color:var(--muted)}
.updated{color:var(--muted);font-size:12px}
</style></head><body>
<h1>dinfer perf trend — Qwen3-0.6B，4090</h1>
<div class="sub">数据：<code>perf-trend.csv</code>（每 10 s 重读）。页签：single = 主线 benchN（dinfer vs vLLM）；pd / tp / dp / other = A/B 组，每页顶部写明基线。URL hash 直达（如 <code>#pd</code>）。<span id="updated" class="updated"></span></div>
<nav class="tabs" id="tabs">
 <button data-tab="single" class="on">single</button><button data-tab="pd" hidden>pd</button><button data-tab="tp" hidden>tp</button><button data-tab="dp" hidden>dp</button><button data-tab="other" hidden>other</button>
</nav>

<section class="page" id="page-single">
<div class="base">基线 = vLLM v0.31.0 单卡（同机同配置）。比值 = dinfer out tok/s ÷ vLLM；虚线 1.0 = 追平。TTFT/TPOT 越低越好。只含 benchN 主线轮次。</div>
<div class="grid">
 <div class="card"><h2>out tok/s ÷ vLLM</h2><canvas id="ratio"></canvas></div>
 <div class="card"><h2>TTFT p50 (ms)，实线 dinfer / 虚线 vLLM</h2><canvas id="ttft"></canvas></div>
 <div class="card"><h2>TPOT p50 (ms)，实线 dinfer / 虚线 vLLM</h2><canvas id="tpot"></canvas></div>
 <div class="card"><h2>out tok/s，实线 dinfer / 虚线 vLLM</h2><canvas id="tps"></canvas></div>
</div>
<div class="card" id="detail" style="margin-top:14px;display:none"><h2 id="detail-title">轮次详情</h2><div id="detail-change" class="chg" style="margin-bottom:8px"></div><div id="detail-table"></div></div>
<div class="card" style="margin-top:14px"><h2>每轮改动与提升（相对上一轮；绿色 = 变好；点击轮次看详情）</h2><div id="table"></div></div>
</section>
<section class="page" id="page-pd" hidden><div class="base" id="base-pd"></div><div class="grid" id="grid-pd"></div><div class="card" style="margin-top:14px"><div id="table-pd"></div></div></section>
<section class="page" id="page-tp" hidden><div class="base" id="base-tp"></div><div class="grid" id="grid-tp"></div><div class="card" style="margin-top:14px"><div id="table-tp"></div></div></section>
<section class="page" id="page-dp" hidden><div class="base" id="base-dp"></div><div class="grid" id="grid-dp"></div><div class="card" style="margin-top:14px"><div id="table-dp"></div></div></section>
<section class="page" id="page-other" hidden><div class="base" id="base-other"></div><div class="grid" id="grid-other"></div><div class="card" style="margin-top:14px"><div id="table-other"></div></div></section>

<script>
const GROUPS=["c1","c8","c32","c8-long"], COLORS=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"];
const CATS=["pd","tp","dp","other"];
const HINT={pd:"基线 = single（单机 :8012，同卡同配置的 dinfer 单实例）。变量 = PD 分离的传输方式 / 卡分布（tcp / shm / ipc / 跨卡 / tp）。柱 = 各 variant 并列，灰柱基线。",
  tp:"基线 = vLLM 单卡。变量 = dinfer --tp N（多卡张量并行，无 P2P 走 NCCL SHM）。",
  dp:"基线 = vLLM 单卡。变量 = dinfer --dp N（同进程数据并行，按在飞数路由）。",
  other:"基线 = 各目录字典序第一个 variant（见表中基线列）。"};
const charts={};
function parseCSV(t){const lines=t.trim().split(/\r?\n/);const hdr=lines[0].split(",");return lines.slice(1).map(l=>{const cells=[];let cur="",q=false;for(const ch of l){if(ch=='"'){q=!q;continue}if(ch==','&&!q){cells.push(cur);cur="";continue}cur+=ch}cells.push(cur);const o={};hdr.forEach((h,i)=>o[h]=cells[i]);return o;});}
const fg=()=>getComputedStyle(document.body).color;
function upsert(id,cfg){if(charts[id]){charts[id].data=cfg.data;charts[id].options=cfg.options;charts[id].update();}else charts[id]=new Chart(document.getElementById(id),cfg);}
// ---- single（主线）----
function buildMain(rows){
  const rounds=[...new Set(rows.map(r=>r.round))];
  const change={};rows.forEach(r=>change[r.round]=r.change);
  const get=(metric,g,field)=>rounds.map(rd=>{const x=rows.find(r=>r.round==rd&&r.group==g&&r.metric==metric);return x?parseFloat(x[field]):null;});
  const mk=(id,metric,field,withV,refLine)=>{
    const ds=[];GROUPS.forEach((g,i)=>{ds.push({label:g,data:get(metric,g,field),borderColor:COLORS[i],backgroundColor:COLORS[i],tension:.15,pointRadius:3});
      if(withV)ds.push({label:g+" vLLM",data:get(metric,g,"vllm"),borderColor:COLORS[i],borderDash:[4,4],pointRadius:0,borderWidth:1});});
    if(refLine)ds.push({label:"vLLM = 1.0",data:rounds.map(()=>1),borderColor:"#888",borderDash:[6,3],pointRadius:0,borderWidth:1});
    upsert(id,{type:"line",data:{labels:rounds,datasets:ds},options:{animation:false,responsive:true,interaction:{mode:"index",intersect:false},
      onClick:(evt,els,chart)=>{const pts=chart.getElementsAtEventForMode(evt,"index",{intersect:false},true);if(pts.length){showRound(rounds[pts[0].index]);}},
      plugins:{legend:{labels:{boxWidth:10,font:{size:11}}},tooltip:{callbacks:{afterTitle:(it)=>"改动: "+(change[it[0].label]||"")}}},
      scales:{y:{beginAtZero:true,ticks:{color:fg()}},x:{ticks:{color:fg()}}}}});
  };
  mk("ratio","out_tok_per_s","ratio_dinfer_over_vllm",false,true);
  mk("ttft","ttft_p50_ms","dinfer",true,false);
  mk("tpot","tpot_p50_ms","dinfer",true,false);
  mk("tps","out_tok_per_s","dinfer",true,false);
  let h="<table><tr><th>轮</th><th class=change>改动</th>";GROUPS.forEach(g=>h+=`<th>${g} tok/s<br>(×vLLM, Δ)</th>`);GROUPS.forEach(g=>h+=`<th>${g} TTFT ms<br>(Δ)</th>`);GROUPS.forEach(g=>h+=`<th>${g} TPOT ms<br>(Δ)</th>`);h+="</tr>";
  rounds.forEach(rd=>{h+=`<tr><td><a href="#" onclick="showRound('${rd}');return false;">${rd}</a></td><td class="change chg">${change[rd]||"—"}</td>`;
    for(const m of ["out_tok_per_s","ttft_p50_ms","tpot_p50_ms"])GROUPS.forEach(g=>{const x=rows.find(r=>r.round==rd&&r.group==g&&r.metric==m);if(!x){h+="<td>—</td>";return;}
      const d=x.delta_pct_vs_prev_round===""?"":parseFloat(x.delta_pct_vs_prev_round);const cls=d===""?"":(d>0.5?"up":(d<-0.5?"down":""));const ds=d===""?"":` <span class="${cls}">${d>0?"+":""}${d.toFixed(0)}%</span>`;
      const v=parseFloat(x.dinfer);h+=m=="out_tok_per_s"?`<td>${v.toFixed(0)} (${parseFloat(x.ratio_dinfer_over_vllm).toFixed(2)}×)${ds}</td>`:`<td>${v.toFixed(m=="tpot_p50_ms"?2:1)} / ${parseFloat(x.vllm).toFixed(m=="tpot_p50_ms"?2:1)}${ds}</td>`;});
    h+="</tr>";});
  document.getElementById("table").innerHTML=h+"</table>";
  window.__rows=rows; window.__rounds=rounds; window.__change=change;
  if(window.__selected) showRound(window.__selected);
}
function showRound(rd){
  const rows=window.__rows||[], rounds=window.__rounds||[], change=window.__change||{};
  const i=rounds.indexOf(rd); if(i<0) return; window.__selected=rd;
  const prev=i>0?rounds[i-1]:null;
  document.getElementById("detail").style.display="block";
  document.getElementById("detail-title").textContent=`轮次 ${rd}（第 ${i+1}/${rounds.length} 轮）`+(prev?`，相对 ${prev}`:"，基线");
  document.getElementById("detail-change").textContent="改动："+(change[rd]||"—");
  const M=[["out_tok_per_s","out tok/s",0,false],["ttft_p50_ms","TTFT p50 ms",1,true],["tpot_p50_ms","TPOT p50 ms",2,true]];
  let h="<table><tr><th>组</th><th>指标</th><th>dinfer</th><th>vLLM（基线）</th><th>dinfer ÷ vLLM</th><th>相对上一轮</th><th>相对首轮</th></tr>";
  GROUPS.forEach(g=>M.forEach(([m,name,dec,lower])=>{const x=rows.find(r=>r.round==rd&&r.group==g&&r.metric==m); if(!x) return;
    const f=rows.find(r=>r.round==rounds[0]&&r.group==g&&r.metric==m); const v=parseFloat(x.dinfer), vv=parseFloat(x.vllm);
    const d=x.delta_pct_vs_prev_round===""?"—":(parseFloat(x.delta_pct_vs_prev_round)>0?"+":"")+parseFloat(x.delta_pct_vs_prev_round).toFixed(1)+"%";
    const d0=f?(((lower?(parseFloat(f.dinfer)-v)/parseFloat(f.dinfer):(v-parseFloat(f.dinfer))/parseFloat(f.dinfer))*100)):null;
    const cls=(s)=>{const n=parseFloat(s);return isNaN(n)?"":(n>0.5?"up":(n<-0.5?"down":""));};
    const ratio=lower?(vv/v):(v/vv);
    h+=`<tr><td>${g}</td><td>${name}</td><td>${v.toFixed(dec)}</td><td>${vv.toFixed(dec)}</td><td>${ratio.toFixed(2)}×${lower?"（vLLM÷dinfer，>1 = 我们更快）":""}</td><td class="${cls(d)}">${d}</td><td class="${d0===null?"":cls(d0)}">${d0===null?"—":(d0>0?"+":"")+d0.toFixed(0)+"%"}</td></tr>`;}));
  document.getElementById("detail-table").innerHTML=h+"</table>";
  document.getElementById("detail").scrollIntoView({behavior:"smooth",block:"nearest"});
}
// ---- ab 页（pd / tp / dp / other）：每个目录×组 一簇柱，灰 = 基线，彩 = variant ----
function buildCat(cat,rows){
  const tab=document.querySelector(`.tabs button[data-tab="${cat}"]`);
  if(!rows.length){tab.hidden=true;return;}
  tab.hidden=false;
  const keys=[...new Set(rows.map(r=>r.round+"\u0000"+r.group))].map(k=>k.split("\u0000"));
  const variants=[...new Set(rows.map(r=>r.variant))];
  const bases=[...new Set(rows.map(r=>r.baseline))];
  document.getElementById("base-"+cat).innerHTML=`<b>${cat}</b>：基线 = <b>${bases.join(" / ")}</b>。${HINT[cat]||""}`;
  const grid=document.getElementById("grid-"+cat);
  const M=[["out_tok_per_s","out tok/s（↑ 好）",0],["ttft_p50_ms","TTFT p50 ms（↓ 好）",1],["tpot_p50_ms","TPOT p50 ms（↓ 好）",2]];
  if(!grid.children.length) M.forEach(([m])=>{const c=document.createElement("div");c.className="card";c.innerHTML=`<h2 id="h-${cat}-${m}"></h2><canvas id="c-${cat}-${m}"></canvas>`;grid.appendChild(c);});
  const labels=keys.map(([rd,g])=>[rd,g]);
  const chg={};rows.forEach(r=>chg[r.round]=r.change);
  M.forEach(([m,title])=>{
    document.getElementById(`h-${cat}-${m}`).textContent=title+`，${variants.join(" / ")} vs ${bases.join("/")}（灰柱）`;
    const ds=[{label:bases.join("/")+"（基线）",backgroundColor:"#9a9a9a",data:keys.map(([rd,g])=>{const x=rows.find(r=>r.round==rd&&r.group==g&&r.metric==m);return x?parseFloat(x.vllm):null;})}];
    variants.forEach((vn,i)=>ds.push({label:vn,backgroundColor:COLORS[i%COLORS.length],data:keys.map(([rd,g])=>{const x=rows.find(r=>r.round==rd&&r.group==g&&r.metric==m&&r.variant==vn);return x?parseFloat(x.dinfer):null;})}));
    upsert(`c-${cat}-${m}`,{type:"bar",data:{labels,datasets:ds},options:{animation:false,responsive:true,
      plugins:{legend:{labels:{boxWidth:10,font:{size:11}}},tooltip:{callbacks:{afterTitle:(it)=>"改动: "+(chg[labels[it[0].dataIndex][0]]||"")}}},
      scales:{y:{beginAtZero:true,ticks:{color:fg()}},x:{ticks:{color:fg(),font:{size:10}}}}}});
  });
  let h=`<table><tr><th class=l>目录</th><th class=l>改动</th><th>组</th><th class=l>基线</th><th class=l>variant</th><th>out tok/s（vs 基线）</th><th>TTFT p50 ms</th><th>TTFT p99 ms</th><th>TPOT p50 ms</th><th>failed</th></tr>`;
  const cell=(x,dec,lower)=>{if(!x)return"<td>—</td>";const a=parseFloat(x.dinfer),b=parseFloat(x.vllm);const d=b?((lower?(b-a)/b:(a-b)/b)*100):null;const cls=d===null?"":(d>0.5?"up":(d<-0.5?"down":""));
    return `<td>${a.toFixed(dec)} / ${b.toFixed(dec)}${d===null?"":` <span class="${cls}">${d>0?"+":""}${d.toFixed(0)}%</span>`}</td>`;};
  keys.forEach(([rd,g])=>variants.forEach(vn=>{const f=(m)=>rows.find(r=>r.round==rd&&r.group==g&&r.metric==m&&r.variant==vn);if(!f("out_tok_per_s"))return;
    h+=`<tr><td class=l>${rd}</td><td class="l chg">${chg[rd]||"—"}</td><td>${g}</td><td class=l>${f("out_tok_per_s").baseline}</td><td class=l>${vn}</td>${cell(f("out_tok_per_s"),0,false)}${cell(f("ttft_p50_ms"),1,true)}${cell(f("ttft_p99_ms"),1,true)}${cell(f("tpot_p50_ms"),2,true)}${cell(f("failed"),0,true)}</tr>`;}));
  document.getElementById("table-"+cat).innerHTML=h+"</table><div class=updated>单元格 = variant / 基线，百分比 = variant 相对基线（绿色 = 变好）。</div>";
}
function build(rows){
  buildMain(rows.filter(r=>r.kind=="main"||r.kind===undefined));
  CATS.forEach(c=>buildCat(c,rows.filter(r=>r.kind==c)));
  document.getElementById("updated").textContent=" 更新于 "+new Date().toLocaleTimeString();
  showTab(current,false);
}
// ---- 切页：纯 JS + URL hash ----
let current=(location.hash||"#single").slice(1);
function showTab(t,push){
  if(!document.getElementById("page-"+t)) t="single";
  current=t;
  document.querySelectorAll(".page").forEach(p=>p.hidden=(p.id!="page-"+t));
  document.querySelectorAll(".tabs button").forEach(b=>b.classList.toggle("on",b.dataset.tab==t));
  if(push&&location.hash!="#"+t) history.replaceState(null,"","#"+t);
  requestAnimationFrame(()=>Object.values(charts).forEach(c=>{try{c.resize();}catch(e){}}));
}
document.getElementById("tabs").addEventListener("click",e=>{const b=e.target.closest("button[data-tab]");if(b)showTab(b.dataset.tab,true);});
window.addEventListener("hashchange",()=>showTab((location.hash||"#single").slice(1),false));
async function refresh(){try{const t=await (await fetch("perf-trend.csv?ts="+Date.now(),{cache:"no-store"})).text();build(parseCSV(t));}catch(e){document.getElementById("updated").textContent=" 读取 CSV 失败: "+e;}}
refresh();setInterval(refresh,10000);
</script></body></html>
"""
open(os.path.join(ROOT, "perf-trend.html"), "w").write(html)

print(f"{len(rounds)} rounds → {ROOT}/perf-trend.{{md,json,csv,svg,html}}")
print("\n".join(lines[4:4 + 2 + len(rounds)]))
if extras:
    print(f"extra entries: {len(extras)} (" + ", ".join(f"{c} {len(by_cat[c])}" for c in CATS if by_cat[c]) + ")", file=sys.stderr)
