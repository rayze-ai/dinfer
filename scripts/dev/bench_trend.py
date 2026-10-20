#!/usr/bin/env python3
"""性能趋势：汇总 test/qwen3/*/ 下的每轮结果，输出
  test/qwen3/perf-trend.md    按卡型分大节（# 4090 / # B300），节内 single（主线 benchN）每轮一行 + pd / tp / dp / other 各一小节（同 HTML 页签名）
  test/qwen3/perf-trend.json  主线轮次的机器可读版（每轮带 gpu）
  test/qwen3/perf-trend.svg   每张卡：主线三张曲线 + 每类 ab 组一行柱图（纯 SVG，不依赖 matplotlib）
  test/qwen3/perf-trend.csv   长表；列 kind = main|pd|tp|dp|other，gpu = 卡型，baseline / variant 写清谁比谁
  test/qwen3/perf-trend.html  Chart.js 面板：顶层卡型按钮（默认数据最多的卡）→ 页签 single / pd / tp / dp / other；URL hash `#B300/tp`
数据来源：
  bench<N>/           主线：dinfer_<group>.json [+ vllm_<group>.json]（四组），CHANGE 一行改动说明；没有 vLLM 的卡只画 dinfer 自己的曲线
  bench_<tag>/        多卡（bench_tp2 / bench_dp2）：同上文件名，但不进主线曲线，按 tag 归 tp / dp 页
  ab_<name>/          A/B：<variant>__<group>.json；有 single 以它作基线，否则字典序第一个 variant（pd vs single / tp2 vs single / tier16 vs tier00）
  poisson*/           开环泊松：dinfer_xs1 / dinfer_xs0 vs vllm，组 = r<rate>；归 other 页
  GPU                 每个目录一行卡型（bench_compare.sh / bench_tp_ab.sh 写）；没有的历史目录一律视为 4090（2026-10-19 之前全是 4090）
类别判定：目录名前缀 / CHANGE 文本（bench\\d* → single；ab_pd* 或含 "pd" → pd；tp → tp；dp → dp；其余 other）。
用法：python3 -I scripts/dev/bench_trend.py [--root test/qwen3]
"""
import csv, glob, json, math, os, re, sys

ROOT = sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else "test/qwen3"
# 模型维度（用户 2026-10-19：图表维度多了 → URL 带模型、加索引页）：模型名 = ROOT 最后一段（test/qwen3 → qwen3），
# 进 CSV `model` 列、HTML 标题与 hash（#<model>/<gpu>/<tab>，旧 #<gpu>/<tab> 仍兼容），索引页 test/index.html 由 bench_index.py 汇总。
MODEL = os.path.basename(os.path.normpath(ROOT))
GROUPS = ["c1_p128_o128", "c8_p128_o128", "c32_p128_o128", "c8_p1024_o128"]
SHORT = {"c1_p128_o128": "c1", "c8_p128_o128": "c8", "c32_p128_o128": "c32", "c8_p1024_o128": "c8-long"}
CATS = ["pd", "tp", "dp", "other"]
CAT_HINT = {"single": "基线 = vLLM v0.31.0 单卡（同机同配置）", "pd": "基线 = single（单机 :8012，同卡同配置的 dinfer 单实例）；变量 = PD 分离的传输方式 / 卡分布",
            "tp": "基线 = single（同卡 dinfer 单实例）或 vLLM 单卡（见 baseline 列）；变量 = dinfer --tp N（多卡张量并行）",
            "dp": "基线 = vLLM 单卡；变量 = dinfer --dp N（同进程数据并行）",
            "other": "基线 = 每个目录字典序第一个 variant（见 baseline 列）"}
DEFAULT_GPU = "4090"
NAN = float("nan")


def read_change(d):
    c = os.path.join(d, "CHANGE")
    return open(c).read().strip().replace("\n", " ") if os.path.exists(c) else ""


def read_gpu(d):
    g = os.path.join(d, "GPU")
    if os.path.exists(g):
        s = open(g).read().strip().split("\n")[0].strip()
        if s:
            return s
    return DEFAULT_GPU


def category(label, change=""):
    # benchN（4090 主线）与 bench_<卡型>_N（如 bench_b300_1，别的卡上的主线 dinfer vs vLLM）都是 single；bench_tp2 / bench_dp2 走下面的 tp / dp
    if re.fullmatch(r"bench\d*", label) or (re.fullmatch(r"bench_[a-z0-9]+_(vllm)?\d+", label) and not re.search(r"_(tp|dp)\d*_", label)):
        return "single"
    s = (label + " " + change).lower()
    if label.startswith("ab_pd") or re.search(r"\bpd\b|pd 分离|pd1", s):
        return "pd"
    if re.search(r"\btp\s*\d|--tp\b|_tp\d|_tp_", s):
        return "tp"
    if re.search(r"\bdp\s*\d|--dp\b|_dp\d|_dp_", s):
        return "dp"
    return "other"


def load_round(d):
    """dinfer_<g>.json 四组齐才算一轮；vllm_<g>.json 可缺（无 vLLM 基线的卡，如 B300）。"""
    r = {"dir": d, "label": os.path.basename(d), "date": os.path.basename(os.path.dirname(d)), "groups": {}, "gpu": read_gpu(d)}
    r["change"] = read_change(d)
    has_v = False
    for g in GROUPS:
        fa, fb = os.path.join(d, f"dinfer_{g}.json"), os.path.join(d, f"vllm_{g}.json")
        if not os.path.exists(fa):
            # 只有 vLLM 的轮（基线先行，dinfer 侧还没有该模型：如 K3 的 bench_b300_vllm1）→ kind=baseline，dinfer 列 NaN
            if not os.path.exists(fb):
                return None
            r["baseline_only"] = True
            x = {"tps": NAN, "ttft": NAN, "tpot": NAN, "vllm_tps": NAN, "vllm_ttft": NAN, "vllm_tpot": NAN, "ratio": NAN}
            b = json.load(open(fb))
            x.update({"vllm_tps": b["output_token_throughput_tps"], "vllm_ttft": b["ttft_ms"]["p50"], "vllm_tpot": b["tpot_ms"]["p50"]})
            r["groups"][g] = x
            has_v = True
            continue
        a = json.load(open(fa))
        x = {"tps": a["output_token_throughput_tps"], "ttft": a["ttft_ms"]["p50"], "tpot": a["tpot_ms"]["p50"],
             "vllm_tps": NAN, "vllm_ttft": NAN, "vllm_tpot": NAN, "ratio": NAN}
        if os.path.exists(fb):
            b = json.load(open(fb))
            x.update({"vllm_tps": b["output_token_throughput_tps"], "vllm_ttft": b["ttft_ms"]["p50"], "vllm_tpot": b["tpot_ms"]["p50"]})
            x["ratio"] = x["tps"] / x["vllm_tps"] if x["vllm_tps"] else NAN
            has_v = True
        r["groups"][g] = x
    r["has_vllm"] = has_v
    r["kind"] = "baseline" if r.get("baseline_only") else category(r["label"], r["change"])
    return r


def round_key(d):
    # benchN（数字轮次）或 bench_<tag>（如 bench_tp2 / bench_dp2，多卡轮次）；带 tag 的排在同日数字轮之后
    b = os.path.basename(d)
    m = re.search(r"bench(\d*)$", b) or re.fullmatch(r"bench_[a-z0-9]+_(?:vllm)?(\d+)", b)
    if m:
        return (os.path.basename(os.path.dirname(d)), 0, int(m.group(1) or 0), "")
    return (os.path.basename(os.path.dirname(d)), 1, 0, b)


def isnan(v):
    return isinstance(v, float) and math.isnan(v)


def fmt(v, dec):
    return "—" if v is None or isnan(v) else f"{v:.{dec}f}"


def pct(new, old, lower_better=False):
    if old is None or isnan(old) or isnan(new) or old == 0: return "—"
    v = (old - new) / old * 100 if lower_better else (new - old) / old * 100
    return f"{v:+.0f}%"


dirs = sorted((d for d in glob.glob(os.path.join(ROOT, "*", "bench*")) if os.path.isdir(d)), key=round_key)
all_rounds = [r for r in (load_round(d) for d in dirs) if r]
single_rounds = [r for r in all_rounds if r["kind"] == "single"]       # 主线（所有卡）
multi_rounds = [r for r in all_rounds if r["kind"] != "single"]        # bench_tp2 / bench_dp2 …
if not single_rounds:
    sys.exit("no bench rounds found")

# ---------------------------------------------------------------------------
# 附加实验（ab_* / bench_<tag> / poisson*）统一成 entries：每条 = 一个目录 × 一组，带基线与若干 variant
#   {"kind", "gpu", "round", "date", "change", "group", "baseline", "base": metrics, "variants": [(name, metrics)]}
# 用户要求（2026-10-17）：所有结果都要能在 CSV / 曲线上看到，不能只留 JSON。
# ---------------------------------------------------------------------------
def _m(j):
    return {"out_tok_per_s": j["output_token_throughput_tps"], "ttft_p50_ms": j["ttft_ms"]["p50"],
            "ttft_p99_ms": j["ttft_ms"]["p99"], "tpot_p50_ms": j["tpot_ms"]["p50"], "failed": j.get("failed", 0)}


BETTER = {"out_tok_per_s": "higher", "ttft_p50_ms": "lower", "ttft_p99_ms": "lower", "tpot_p50_ms": "lower", "failed": "lower"}
known_gpus = set()  # 出现过 GPU 文件 / 轮次的卡型（即使目录还没有 json，HTML 也要出按钮）


def extra_entries():
    es = []
    for r in multi_rounds:  # bench_tp2 / bench_dp2：dinfer(多卡) vs vLLM 单卡（没有 vllm 文件的跳过）
        for g in GROUPS:
            fb = os.path.join(r["dir"], f"vllm_{g}.json")
            if not os.path.exists(fb):
                continue
            a, b = json.load(open(os.path.join(r["dir"], f"dinfer_{g}.json"))), json.load(open(fb))
            es.append({"kind": r["kind"], "gpu": r["gpu"], "round": r["label"], "date": r["date"], "change": r["change"], "group": SHORT[g],
                       "baseline": "vllm", "base": _m(b), "variants": [("dinfer", _m(a))]})
    for d in sorted(glob.glob(os.path.join(ROOT, "*", "poisson*"))):
        if not os.path.isdir(d):
            continue
        rnd, date, chg, gpu = os.path.basename(d), os.path.basename(os.path.dirname(d)), read_change(d) or os.path.basename(d), read_gpu(d)
        known_gpus.add(gpu)
        rates = sorted({re.sub(r".*_r(\d+)\.json$", r"\1", f) for f in glob.glob(os.path.join(d, "vllm_r*.json"))}, key=int)
        for rate in rates:
            v = json.load(open(os.path.join(d, f"vllm_r{rate}.json")))
            vs = []
            for side in ("dinfer_xs1", "dinfer_xs0"):
                f = os.path.join(d, f"{side}_r{rate}.json")
                if os.path.exists(f):
                    vs.append((side, _m(json.load(open(f)))))
            if vs:
                es.append({"kind": category(rnd, chg), "gpu": gpu, "round": rnd, "date": date, "change": chg, "group": f"r{rate}",
                           "baseline": "vllm", "base": _m(v), "variants": vs})
    for d in sorted(glob.glob(os.path.join(ROOT, "*", "ab_*"))):
        if not os.path.isdir(d):
            continue
        rnd, date, chg, gpu = os.path.basename(d), os.path.basename(os.path.dirname(d)), read_change(d) or os.path.basename(d), read_gpu(d)
        known_gpus.add(gpu)
        files = glob.glob(os.path.join(d, "*.json"))
        variants = sorted({os.path.basename(f).split("__")[0] for f in files if "__" in os.path.basename(f)})
        if len(variants) < 2:
            continue
        # 基线：有 single（单机）就以它为基线（pd vs single / tpN vs single），否则字典序第一个
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
                es.append({"kind": category(rnd, chg), "gpu": gpu, "round": rnd, "date": date, "change": chg, "group": SHORT.get(g, g),
                           "baseline": base, "base": _m(json.load(open(fb))), "variants": vs})
    return es


extras = extra_entries()
for r in all_rounds:
    known_gpus.add(r["gpu"])
for d in glob.glob(os.path.join(ROOT, "*", "*", "GPU")):
    known_gpus.add(read_gpu(os.path.dirname(d)))

# 卡型顺序：主线轮次多的在前（HTML 默认选第一张）；同数按名字
gpu_count = {g: sum(1 for r in single_rounds if r["gpu"] == g) for g in known_gpus}
GPUS = sorted(known_gpus, key=lambda g: (-gpu_count.get(g, 0), g))
DEFAULT = GPUS[0]
rounds_of = {g: [r for r in single_rounds if r["gpu"] == g] for g in GPUS}
cat_of = {g: {c: [e for e in extras if e["kind"] == c and e["gpu"] == g] for c in CATS} for g in GPUS}
rounds = rounds_of[DEFAULT]  # 兼容：stdout 末尾打印默认卡的主线表


# ---------------------------------------------------------------------------
# MD：每张卡一大节；节内 single 主表 + 首轮→最新 + 每类一小节
# ---------------------------------------------------------------------------
def main_table_lines(rs):
    out = []
    hdr = "| 轮 | 日期 | 改动 | " + " | ".join(f"{SHORT[g]} tok/s (×vLLM)" for g in GROUPS) + " | " + " | ".join(f"{SHORT[g]} TTFT ms" for g in GROUPS) + " | " + " | ".join(f"{SHORT[g]} TPOT ms" for g in GROUPS) + " |"
    out += [hdr, "|" + "---|" * (3 + 3 * len(GROUPS))]
    prev = None
    for r in rs:
        cells = [r["label"], r["date"], r["change"] or "—"]
        for g in GROUPS:
            x = r["groups"][g]; d = f" ({pct(x['tps'], prev['groups'][g]['tps'])})" if prev else ""
            cells.append(f"{x['tps']:.0f} ({fmt(x['ratio'], 2)}×){d}")
        for g in GROUPS:
            x = r["groups"][g]; d = f" ({pct(x['ttft'], prev['groups'][g]['ttft'], True)})" if prev else ""
            cells.append(f"{x['ttft']:.1f} / {fmt(x['vllm_ttft'], 1)}{d}")
        for g in GROUPS:
            x = r["groups"][g]; d = f" ({pct(x['tpot'], prev['groups'][g]['tpot'], True)})" if prev else ""
            cells.append(f"{x['tpot']:.2f} / {fmt(x['vllm_tpot'], 2)}{d}")
        out.append("| " + " | ".join(cells) + " |")
        prev = r
    return out


lines = ["# Qwen3-0.6B 性能趋势（dinfer vs vLLM v0.31.0，`bench_compare.sh` 四组；按卡型分节：" + " / ".join(GPUS) + "）", "",
         "每轮一行；括号内是相对上一轮的变化（吞吐 ↑ 为好，TTFT/TPOT ↓ 为好）。比值 = dinfer out tok/s ÷ vLLM（无 vLLM 基线的卡为 —）。vLLM 各轮自己也有 ±3% 抖动。", ""]
main_print = None
for gpu in GPUS:
    rs = rounds_of[gpu]
    lines += [f"# {gpu}", ""]
    if rs:
        lines += ["## single", "", CAT_HINT["single"] + ("" if any(r["has_vllm"] for r in rs) else "；**本卡无 vLLM 基线，只有 dinfer 自己的数**") + "。", ""]
        tbl = main_table_lines(rs)
        if gpu == DEFAULT:
            main_print = tbl
        lines += tbl
        first, last = rs[0], rs[-1]
        lines += ["", "### 首轮 → 最新", "", "| 组 | out tok/s | ×vLLM | TTFT ms | TPOT ms |", "|---|---|---|---|---|"]
        for g in GROUPS:
            a, b = first["groups"][g], last["groups"][g]
            lines.append(f"| {SHORT[g]} | {a['tps']:.0f} → {b['tps']:.0f} ({pct(b['tps'], a['tps'])}) | {fmt(a['ratio'], 2)} → {fmt(b['ratio'], 2)} | {a['ttft']:.1f} → {b['ttft']:.1f} ({pct(b['ttft'], a['ttft'], True)}) | {a['tpot']:.2f} → {b['tpot']:.2f} ({pct(b['tpot'], a['tpot'], True)}) |")
        lines.append("")
    else:
        lines += ["## single", "", "（本卡暂无主线 benchN 轮次）", ""]
    for c in CATS:
        es = cat_of[gpu][c]
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
lines += ["![trend](perf-trend.svg)", ""]
open(os.path.join(ROOT, "perf-trend.md"), "w").write("\n".join(lines))
json.dump([{k: v for k, v in r.items()} for r in single_rounds], open(os.path.join(ROOT, "perf-trend.json"), "w"), indent=1, ensure_ascii=False)

# ---------------------------------------------------------------------------
# SVG：每张卡一大节：主线三张曲线（只画 benchN）+ 每类一行三张柱图（tok/s / TTFT / TPOT，基线灰、variant 彩）
# ---------------------------------------------------------------------------
W, H, PAD = 980, 260, 48
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def panel(y0, rs, title, key, lower_better, ref_key=None):
    n = len(rs)
    vals = {g: [r["groups"][g][key] for r in rs] for g in GROUPS}
    allv = [v for g in GROUPS for v in vals[g] if not isnan(v)]
    if ref_key:
        allv += [r["groups"][g][ref_key] for r in rs for g in GROUPS if not isnan(r["groups"][g][ref_key])]
    out = [f'<text x="{PAD}" y="{y0 + 18}" font-size="14" font-weight="bold">{esc(title)}</text>']
    if not allv:
        out.append(f'<text x="{W / 2}" y="{y0 + H / 2}" font-size="13" text-anchor="middle" fill="#888">无 vLLM 基线（本卡只有 dinfer 结果）</text>')
        return out
    ymax = max(allv) * 1.1 or 1
    def X(i): return PAD + i * (W - 2 * PAD) / max(n - 1, 1)
    def Y(v): return y0 + H - PAD - (v / ymax) * (H - 2 * PAD)
    out += [f'<line x1="{PAD}" y1="{Y(0)}" x2="{W - PAD}" y2="{Y(0)}" stroke="#888"/>',
            f'<line x1="{PAD}" y1="{y0 + PAD}" x2="{PAD}" y2="{Y(0)}" stroke="#888"/>',
            f'<text x="{PAD - 6}" y="{y0 + PAD + 4}" font-size="10" text-anchor="end">{ymax:.2f}</text>',
            f'<text x="{PAD - 6}" y="{Y(0) + 4}" font-size="10" text-anchor="end">0</text>']
    if key == "ratio":
        out.append(f'<line x1="{PAD}" y1="{Y(1.0)}" x2="{W - PAD}" y2="{Y(1.0)}" stroke="#444" stroke-dasharray="4 3"/>')
        out.append(f'<text x="{W - PAD + 4}" y="{Y(1.0) + 4}" font-size="10">vLLM = 1.0</text>')
    for ci, g in enumerate(GROUPS):
        pts = [(i, v) for i, v in enumerate(vals[g]) if not isnan(v)]
        out.append(f'<polyline points="{" ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in pts)}" fill="none" stroke="{COLORS[ci]}" stroke-width="2"/>')
        for i, v in pts:
            out.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3" fill="{COLORS[ci]}"/>')
        if ref_key:
            rp = [(i, r["groups"][g][ref_key]) for i, r in enumerate(rs) if not isnan(r["groups"][g][ref_key])]
            if rp:
                out.append(f'<polyline points="{" ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in rp)}" fill="none" stroke="{COLORS[ci]}" stroke-width="1" stroke-dasharray="3 3" opacity="0.6"/>')
        if pts:
            out.append(f'<text x="{W - PAD + 4}" y="{Y(pts[-1][1]) + 4}" font-size="10" fill="{COLORS[ci]}">{SHORT[g]}</text>')
    for i, r in enumerate(rs):
        out.append(f'<text x="{X(i):.1f}" y="{Y(0) + 14}" font-size="9" text-anchor="middle">{esc(r["label"])}</text>')
    return out


def bar_panel(x0, y0, w, h, title, labels, series, f):
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
            out.append(f'<text x="{x + bw * 0.45:.1f}" y="{Y(v) - 2:.1f}" font-size="8" text-anchor="middle">{f(v)}</text>')
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
    for pi, (key, title, f) in enumerate([("out_tok_per_s", "out tok/s（↑ 好）", lambda v: f"{v:.0f}"),
                                           ("ttft_p50_ms", "TTFT p50 ms（↓ 好）", lambda v: f"{v:.0f}"),
                                           ("tpot_p50_ms", "TPOT p50 ms（↓ 好）", lambda v: f"{v:.2f}")]):
        series = [(f"{base_name}（基线）", "#9a9a9a", [e["base"][key] for e in es])]
        for vi, vn in enumerate(vnames):
            series.append((vn, COLORS[vi % len(COLORS)], [dict(e["variants"]).get(vn, {}).get(key) for e in es]))
        out += bar_panel(pi * pw, y0 + 26, pw, H - 26, title, labels, series, f)
    return out


svg_body, y = [], 0
for gpu in GPUS:
    rs = rounds_of[gpu]
    cats = [c for c in CATS if cat_of[gpu][c]]
    if not rs and not cats:
        continue
    svg_body.append(f'<text x="{PAD}" y="{y + 20}" font-size="18" font-weight="bold">{esc(gpu)}</text>')
    svg_body.append(f'<text x="{PAD + 80}" y="{y + 20}" font-size="11" fill="#555">single：主线 benchN，dinfer vs vLLM 单卡（下方 pd / tp / dp / other 各一行，基线见各节）</text>')
    y += 28
    if rs:
        svg_body += panel(y, rs, "out tok/s ÷ vLLM（越高越好；虚线 = 1.0）", "ratio", False)
        svg_body += panel(y + H, rs, "TTFT p50 ms（实线 dinfer，虚线 vLLM；越低越好）", "ttft", True, "vllm_ttft")
        svg_body += panel(y + 2 * H, rs, "TPOT p50 ms（实线 dinfer，虚线 vLLM；越低越好）", "tpot", True, "vllm_tpot")
        y += 3 * H
    for c in cats:
        svg_body += cat_svg(y, c, cat_of[gpu][c])
        y += H
    y += 16
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{max(y, 40)}" font-family="sans-serif">',
       f'<rect width="{W}" height="{max(y, 40)}" fill="white"/>'] + svg_body + ["</svg>"]
open(os.path.join(ROOT, "perf-trend.svg"), "w").write("\n".join(svg))

# ---------------------------------------------------------------------------
# CSV（长表，用户要求：每轮每指标的提升 + 改动说明）。kind / gpu / baseline / variant 四列说明谁比谁、哪张卡。
# ---------------------------------------------------------------------------
with open(os.path.join(ROOT, "perf-trend.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["round", "date", "change", "group", "metric", "dinfer", "vllm", "ratio_dinfer_over_vllm", "delta_pct_vs_prev_round", "better_is",
                "kind", "gpu", "baseline", "variant", "model"])
    for gpu in GPUS:
        prev = None
        for r in rounds_of[gpu]:
            for g in GROUPS:
                x = r["groups"][g]
                for metric, key, vkey, lower in (("out_tok_per_s", "tps", "vllm_tps", False), ("ttft_p50_ms", "ttft", "vllm_ttft", True), ("tpot_p50_ms", "tpot", "vllm_tpot", True)):
                    d = ""
                    if prev:
                        o, nv = prev["groups"][g][key], x[key]
                        d = f"{((o - nv) / o * 100 if lower else (nv - o) / o * 100):.1f}" if o else ""
                    vv = x[vkey]
                    ratio = (x[key] / vv) if (not isnan(vv) and vv) else ""
                    w.writerow([r["label"], r["date"], r["change"], SHORT[g], metric, f"{x[key]:.3f}", "" if isnan(vv) else f"{vv:.3f}",
                                f"{ratio:.3f}" if ratio != "" else "", d, "lower" if lower else "higher", "main", gpu, "vllm", "dinfer", MODEL])
            prev = r
    # 附加实验：dinfer 列 = variant，vllm 列 = 基线
    for e in extras:
        for name, a in e["variants"]:
            for k, av in a.items():
                bv = e["base"][k]
                w.writerow([e["round"], e["date"], e["change"], e["group"], k, f"{av:.3f}", f"{bv:.3f}", f"{(av / bv):.3f}" if bv else "", "", BETTER[k],
                            e["kind"], e["gpu"], e["baseline"], name, MODEL])

# ---------------------------------------------------------------------------
# HTML 面板：静态页，fetch 同目录 perf-trend.csv，每 10 s 重画；Chart.js 走 cdnjs；卡型按钮 + 切页纯 JS，hash = #<gpu>/<tab>
# ---------------------------------------------------------------------------
html = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>perf trend — __MODEL__</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--card:#f6f7f9;--line:#e3e5e8;--acc:#1f77b4}
@media (prefers-color-scheme: dark){:root:not([data-theme=light]){--bg:#111;--fg:#eee;--muted:#9a9a9a;--card:#1b1d20;--line:#2a2d31}}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
h1{font-size:18px;margin:0 0 4px}.sub{color:var(--muted);margin-bottom:12px}
.gpus{display:flex;gap:6px;align-items:center;margin:6px 0 2px;flex-wrap:wrap}.gpus span{color:var(--muted);font-size:12px;margin-right:4px}
.gpus button{border:1px solid var(--line);background:var(--card);color:var(--fg);padding:5px 14px;font:inherit;font-weight:600;border-radius:999px;cursor:pointer}
.gpus button.on{background:var(--acc);border-color:var(--acc);color:#fff}
.tabs{display:flex;gap:4px;border-bottom:1px solid var(--line);margin:8px 0 12px;flex-wrap:wrap}
.tabs button{background:none;border:1px solid transparent;border-bottom:none;color:var(--muted);padding:6px 14px;font:inherit;cursor:pointer;border-radius:6px 6px 0 0}
.tabs button.on{color:var(--fg);border-color:var(--line);background:var(--card);font-weight:600}
.tabs button[hidden]{display:none}
.page[hidden]{display:none}
.base{padding:6px 10px;margin-bottom:12px;border-left:3px solid var(--acc);background:var(--card);font-size:13px}
.empty{padding:24px;text-align:center;color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;position:relative}
.card h2{font-size:14px;margin:0 0 6px}
.nov{position:absolute;inset:40px 10px 10px;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:13px;pointer-events:none}
table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:right;vertical-align:top}
th:first-child,td:first-child,td.change,th.change,td.l,th.l{text-align:left}
.up{color:#1a7f37}.down{color:#c62828}.chg{max-width:420px;white-space:normal;color:var(--muted)}
.updated{color:var(--muted);font-size:12px}
</style></head><body>
<h1>perf trend — __MODEL__ <a href="../index.html" style="font-size:13px;font-weight:normal;margin-left:12px">← 全部模型索引</a></h1>
<div class="sub">数据：<code>perf-trend.csv</code>（每 10 s 重读）。先选卡型，再选页签：single = 主线 benchN（dinfer vs vLLM）；pd / tp / dp / other = A/B 组，每页顶部写明基线。URL hash 直达（如 <code>#B300/tp</code>）。<span id="updated" class="updated"></span></div>
<div class="gpus" id="gpus"><span>卡型</span></div>
<nav class="tabs" id="tabs">
 <button data-tab="single" class="on">single</button><button data-tab="pd" hidden>pd</button><button data-tab="tp" hidden>tp</button><button data-tab="dp" hidden>dp</button><button data-tab="other" hidden>other</button>
</nav>

<section class="page" id="page-single">
<div class="base" id="base-single">基线 = vLLM v0.31.0 单卡（同机同配置）。比值 = dinfer out tok/s ÷ vLLM；虚线 1.0 = 追平。TTFT/TPOT 越低越好。只含 benchN 主线轮次。</div>
<div id="single-body">
<div class="grid">
 <div class="card"><h2>out tok/s ÷ vLLM</h2><canvas id="ratio"></canvas><div class="nov" id="nov-ratio" hidden>无 vLLM 基线（本卡只有 dinfer 结果）</div></div>
 <div class="card"><h2>TTFT p50 (ms)，实线 dinfer / 虚线 vLLM</h2><canvas id="ttft"></canvas></div>
 <div class="card"><h2>TPOT p50 (ms)，实线 dinfer / 虚线 vLLM</h2><canvas id="tpot"></canvas></div>
 <div class="card"><h2>out tok/s，实线 dinfer / 虚线 vLLM</h2><canvas id="tps"></canvas></div>
</div>
<div class="card" id="detail" style="margin-top:14px;display:none"><h2 id="detail-title">轮次详情</h2><div id="detail-change" class="chg" style="margin-bottom:8px"></div><div id="detail-table"></div></div>
<div class="card" style="margin-top:14px"><h2>每轮改动与提升（相对上一轮；绿色 = 变好；点击轮次看详情）</h2><div id="table"></div></div>
</div>
<div class="empty" id="single-empty" hidden>本卡暂无主线 benchN 轮次</div>
</section>
<section class="page" id="page-pd" hidden><div class="base" id="base-pd"></div><div class="grid" id="grid-pd"></div><div class="card" style="margin-top:14px"><div id="table-pd"></div></div></section>
<section class="page" id="page-tp" hidden><div class="base" id="base-tp"></div><div class="grid" id="grid-tp"></div><div class="card" style="margin-top:14px"><div id="table-tp"></div></div></section>
<section class="page" id="page-dp" hidden><div class="base" id="base-dp"></div><div class="grid" id="grid-dp"></div><div class="card" style="margin-top:14px"><div id="table-dp"></div></div></section>
<section class="page" id="page-other" hidden><div class="base" id="base-other"></div><div class="grid" id="grid-other"></div><div class="card" style="margin-top:14px"><div id="table-other"></div></div></section>

<script>
const KNOWN_GPUS=__GPUS__, DEFAULT_GPU=__DEFAULT__, MODEL=__MODEL__;
const GROUPS=["c1","c8","c32","c8-long"], COLORS=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"];
const CATS=["pd","tp","dp","other"];
const HINT={single:"基线 = vLLM v0.31.0 单卡（同机同配置）。比值 = dinfer out tok/s ÷ vLLM；虚线 1.0 = 追平。TTFT/TPOT 越低越好。只含 benchN 主线轮次。",
  pd:"基线 = single（单机 :8012，同卡同配置的 dinfer 单实例）。变量 = PD 分离的传输方式 / 卡分布（tcp / shm / ipc / 跨卡 / tp）。柱 = 各 variant 并列，灰柱基线。",
  tp:"变量 = dinfer --tp N（多卡张量并行）。基线 = single（同卡 dinfer 单实例，bench_tp_ab.sh）或 vLLM 单卡（bench_tp2），以表中基线列为准。",
  dp:"基线 = vLLM 单卡。变量 = dinfer --dp N（同进程数据并行，按在飞数路由）。",
  other:"基线 = 各目录字典序第一个 variant（见表中基线列）。"};
const charts={};
let ALL=[];
function parseCSV(t){const lines=t.trim().split(/\r?\n/);const hdr=lines[0].split(",");return lines.slice(1).map(l=>{const cells=[];let cur="",q=false;for(const ch of l){if(ch=='"'){q=!q;continue}if(ch==','&&!q){cells.push(cur);cur="";continue}cur+=ch}cells.push(cur);const o={};hdr.forEach((h,i)=>o[h]=cells[i]);return o;});}
const fg=()=>getComputedStyle(document.body).color;
const num=(s)=>{if(s===undefined||s==="")return null;const v=parseFloat(s);return isNaN(v)?null:v;};
const f=(v,d)=>v===null||v===undefined||isNaN(v)?"—":v.toFixed(d);
function upsert(id,cfg){if(charts[id]){charts[id].data=cfg.data;charts[id].options=cfg.options;charts[id].update();}else charts[id]=new Chart(document.getElementById(id),cfg);}
// ---- single（主线）----
function buildMain(rows){
  const empty=!rows.length;
  document.getElementById("single-body").hidden=empty; document.getElementById("single-empty").hidden=!empty;
  const hasV=rows.some(r=>r.vllm!=="");
  document.getElementById("base-single").textContent=HINT.single+(hasV?"":" 本卡无 vLLM 基线：只有 dinfer 自己的 TTFT / TPOT / tok/s。");
  document.getElementById("nov-ratio").hidden=hasV||empty;
  const rounds=[...new Set(rows.map(r=>r.round))];
  const change={};rows.forEach(r=>change[r.round]=r.change);
  const get=(metric,g,field)=>rounds.map(rd=>{const x=rows.find(r=>r.round==rd&&r.group==g&&r.metric==metric);return x?num(x[field]):null;});
  const mk=(id,metric,field,withV,refLine)=>{
    const ds=[];GROUPS.forEach((g,i)=>{ds.push({label:g,data:get(metric,g,field),borderColor:COLORS[i],backgroundColor:COLORS[i],tension:.15,pointRadius:3,spanGaps:true});
      if(withV&&hasV)ds.push({label:g+" vLLM",data:get(metric,g,"vllm"),borderColor:COLORS[i],borderDash:[4,4],pointRadius:0,borderWidth:1,spanGaps:true});});
    if(refLine&&hasV)ds.push({label:"vLLM = 1.0",data:rounds.map(()=>1),borderColor:"#888",borderDash:[6,3],pointRadius:0,borderWidth:1});
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
      const d=num(x.delta_pct_vs_prev_round);const cls=d===null?"":(d>0.5?"up":(d<-0.5?"down":""));const ds=d===null?"":` <span class="${cls}">${d>0?"+":""}${d.toFixed(0)}%</span>`;
      const v=num(x.dinfer), dec=m=="tpot_p50_ms"?2:1;h+=m=="out_tok_per_s"?`<td>${f(v,0)} (${f(num(x.ratio_dinfer_over_vllm),2)}×)${ds}</td>`:`<td>${f(v,dec)} / ${f(num(x.vllm),dec)}${ds}</td>`;});
    h+="</tr>";});
  document.getElementById("table").innerHTML=h+"</table>";
  window.__rows=rows; window.__rounds=rounds; window.__change=change;
  if(window.__selected&&rounds.includes(window.__selected)) showRound(window.__selected); else document.getElementById("detail").style.display="none";
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
    const fr=rows.find(r=>r.round==rounds[0]&&r.group==g&&r.metric==m); const v=num(x.dinfer), vv=num(x.vllm);
    const dn=num(x.delta_pct_vs_prev_round); const d=dn===null?"—":(dn>0?"+":"")+dn.toFixed(1)+"%";
    const f0=fr?num(fr.dinfer):null; const d0=f0?(((lower?(f0-v)/f0:(v-f0)/f0)*100)):null;
    const cls=(s)=>{const n=parseFloat(s);return isNaN(n)?"":(n>0.5?"up":(n<-0.5?"down":""));};
    const ratio=vv===null?null:(lower?(vv/v):(v/vv));
    h+=`<tr><td>${g}</td><td>${name}</td><td>${f(v,dec)}</td><td>${f(vv,dec)}</td><td>${ratio===null?"—":ratio.toFixed(2)+"×"+(lower?"（vLLM÷dinfer，>1 = 我们更快）":"")}</td><td class="${cls(d)}">${d}</td><td class="${d0===null?"":cls(d0)}">${d0===null?"—":(d0>0?"+":"")+d0.toFixed(0)+"%"}</td></tr>`;}));
  document.getElementById("detail-table").innerHTML=h+"</table>";
  document.getElementById("detail").scrollIntoView({behavior:"smooth",block:"nearest"});
}
// ---- ab 页（pd / tp / dp / other）：每个目录×组 一簇柱，灰 = 基线，彩 = variant ----
function buildCat(cat,rows){
  const tab=document.querySelector(`.tabs button[data-tab="${cat}"]`);
  tab.hidden=!rows.length;
  if(!rows.length) return;
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
    const ds=[{label:bases.join("/")+"（基线）",backgroundColor:"#9a9a9a",data:keys.map(([rd,g])=>{const x=rows.find(r=>r.round==rd&&r.group==g&&r.metric==m);return x?num(x.vllm):null;})}];
    variants.forEach((vn,i)=>ds.push({label:vn,backgroundColor:COLORS[i%COLORS.length],data:keys.map(([rd,g])=>{const x=rows.find(r=>r.round==rd&&r.group==g&&r.metric==m&&r.variant==vn);return x?num(x.dinfer):null;})}));
    upsert(`c-${cat}-${m}`,{type:"bar",data:{labels,datasets:ds},options:{animation:false,responsive:true,
      plugins:{legend:{labels:{boxWidth:10,font:{size:11}}},tooltip:{callbacks:{afterTitle:(it)=>"改动: "+(chg[labels[it[0].dataIndex][0]]||"")}}},
      scales:{y:{beginAtZero:true,ticks:{color:fg()}},x:{ticks:{color:fg(),font:{size:10}}}}}});
  });
  let h=`<table><tr><th class=l>目录</th><th class=l>改动</th><th>组</th><th class=l>基线</th><th class=l>variant</th><th>out tok/s（vs 基线）</th><th>TTFT p50 ms</th><th>TTFT p99 ms</th><th>TPOT p50 ms</th><th>failed</th></tr>`;
  const cell=(x,dec,lower)=>{if(!x)return"<td>—</td>";const a=num(x.dinfer),b=num(x.vllm);const d=b?((lower?(b-a)/b:(a-b)/b)*100):null;const cls=d===null?"":(d>0.5?"up":(d<-0.5?"down":""));
    return `<td>${f(a,dec)} / ${f(b,dec)}${d===null?"":` <span class="${cls}">${d>0?"+":""}${d.toFixed(0)}%</span>`}</td>`;};
  keys.forEach(([rd,g])=>variants.forEach(vn=>{const q=(m)=>rows.find(r=>r.round==rd&&r.group==g&&r.metric==m&&r.variant==vn);if(!q("out_tok_per_s"))return;
    h+=`<tr><td class=l>${rd}</td><td class="l chg">${chg[rd]||"—"}</td><td>${g}</td><td class=l>${q("out_tok_per_s").baseline}</td><td class=l>${vn}</td>${cell(q("out_tok_per_s"),0,false)}${cell(q("ttft_p50_ms"),1,true)}${cell(q("ttft_p99_ms"),1,true)}${cell(q("tpot_p50_ms"),2,true)}${cell(q("failed"),0,true)}</tr>`;}));
  document.getElementById("table-"+cat).innerHTML=h+"</table><div class=updated>单元格 = variant / 基线，百分比 = variant 相对基线（绿色 = 变好）。</div>";
}
// ---- 卡型 + 切页：纯 JS + URL hash（#<gpu>/<tab>）----
let curGpu=DEFAULT_GPU, curTab="single";
function parseHash(){const h=(location.hash||"").slice(1);if(!h)return;let parts=h.split("/");if(parts.length===3){if(decodeURIComponent(parts[0])!==MODEL)return;parts=parts.slice(1);}const [a,b]=parts;if(b!==undefined){curGpu=decodeURIComponent(a)||curGpu;curTab=b||"single";}else if(["single",...CATS].includes(a)){curTab=a;}else{curGpu=decodeURIComponent(a);}}
function gpuList(rows){const s=new Set(KNOWN_GPUS);rows.forEach(r=>{if(r.gpu)s.add(r.gpu);});return [...s];}
function renderGpus(rows){
  const gpus=gpuList(rows); if(!gpus.includes(curGpu)) curGpu=gpus[0]||DEFAULT_GPU;
  const box=document.getElementById("gpus"); box.querySelectorAll("button").forEach(b=>b.remove());
  gpus.forEach(g=>{const b=document.createElement("button");b.textContent=g;b.dataset.gpu=g;b.classList.toggle("on",g==curGpu);b.onclick=()=>{curGpu=g;rebuild();show(true);};box.appendChild(b);});
}
function rebuild(){
  const rows=ALL.filter(r=>(r.gpu||DEFAULT_GPU)==curGpu);
  buildMain(rows.filter(r=>r.kind=="main"||r.kind===undefined));
  CATS.forEach(c=>buildCat(c,rows.filter(r=>r.kind==c)));
  document.querySelectorAll(".gpus button").forEach(b=>b.classList.toggle("on",b.dataset.gpu==curGpu));
}
function show(push){
  const tab=document.querySelector(`.tabs button[data-tab="${curTab}"]`);
  if(!tab||tab.hidden) curTab="single";
  document.querySelectorAll(".page").forEach(p=>p.hidden=(p.id!="page-"+curTab));
  document.querySelectorAll(".tabs button").forEach(b=>b.classList.toggle("on",b.dataset.tab==curTab));
  const want="#"+encodeURIComponent(MODEL)+"/"+encodeURIComponent(curGpu)+"/"+curTab;
  if(push&&location.hash!=want) history.replaceState(null,"",want);
  requestAnimationFrame(()=>Object.values(charts).forEach(c=>{try{c.resize();}catch(e){}}));
}
function build(rows){ALL=rows;renderGpus(rows);rebuild();document.getElementById("updated").textContent=" 更新于 "+new Date().toLocaleTimeString();show(true);}
document.getElementById("tabs").addEventListener("click",e=>{const b=e.target.closest("button[data-tab]");if(b){curTab=b.dataset.tab;show(true);}});
window.addEventListener("hashchange",()=>{parseHash();if(ALL.length){rebuild();}show(false);});
parseHash();
async function refresh(){try{const t=await (await fetch("perf-trend.csv?ts="+Date.now(),{cache:"no-store"})).text();build(parseCSV(t));}catch(e){document.getElementById("updated").textContent=" 读取 CSV 失败: "+e;}}
refresh();setInterval(refresh,10000);
</script></body></html>
"""
html = html.replace("__GPUS__", json.dumps(GPUS, ensure_ascii=False)).replace("__DEFAULT__", json.dumps(DEFAULT, ensure_ascii=False)).replace("__MODEL__", json.dumps(MODEL, ensure_ascii=False)).replace("<title>perf trend — \"" + MODEL + "\"</title>", "<title>perf trend — " + MODEL + "</title>").replace("<h1>perf trend — \"" + MODEL + "\" ", "<h1>perf trend — " + MODEL + " ")
open(os.path.join(ROOT, "perf-trend.html"), "w").write(html)

print(f"{len(single_rounds)} rounds ({', '.join(f'{g} {gpu_count.get(g, 0)}' for g in GPUS)}) → {ROOT}/perf-trend.{{md,json,csv,svg,html}}")
if main_print:
    print("\n".join(main_print))  # 默认卡主线表（bench_compare.sh 取 tail -3）
if extras:
    print(f"extra entries: {len(extras)} (" + ", ".join(f"{g}:{c} {len(cat_of[g][c])}" for g in GPUS for c in CATS if cat_of[g][c]) + ")", file=sys.stderr)
