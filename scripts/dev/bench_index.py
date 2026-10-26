#!/usr/bin/env python3
"""汇总所有模型的 perf-trend 成一页索引 test/index.html（用户 2026-10-19：维度多了——模型 × 卡型——要一个索引页）。
扫 test/*/perf-trend.json（每个模型目录一份，bench_trend.py 产出），每个 (模型, 卡型) 一行：最新主线轮次、日期、四组 TPOT 与
×vLLM、改动一句话，链接到 <model>/perf-trend.html#<model>/<gpu>/single。纯静态，不依赖外部库。
  python3 -I scripts/dev/bench_index.py [--root test]
"""
import glob, html, json, os, sys

ROOT = sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else "test"
GROUPS = [("c1_p128_o128", "c1"), ("c8_p128_o128", "c8"), ("c32_p128_o128", "c32"), ("c8_p1024_o128", "c8-long")]

rows = []
perf_models = set()
# 每个模型的质量对比详情页（优先最新一轮 qa_zh2，否则 qa_zh）：model -> 相对路径
qa_report = {}
for sub in ("qa_zh2", "qa_zh"):
    for rp in sorted(glob.glob(os.path.join(ROOT, "*", sub, "report.html"))):
        m = rp.split(os.sep)[-3]
        qa_report.setdefault(m, os.path.relpath(rp, ROOT))
def qa_cell(m):
    rel = qa_report.get(m)
    return f'<a href="{html.escape(rel)}">质量详情</a>' if rel else "—"


for jf in sorted(glob.glob(os.path.join(ROOT, "*", "perf-trend.json"))):
    model = os.path.basename(os.path.dirname(jf))
    try:
        rounds = json.load(open(jf))
    except Exception:
        continue
    perf_models.add(model)
    by_gpu = {}
    for r in rounds:
        by_gpu.setdefault(r.get("gpu", "?"), []).append(r)
    for gpu, rs in sorted(by_gpu.items()):
        last = rs[-1]
        cells = []
        for g, short in GROUPS:
            m = last.get("groups", {}).get(g) or {}
            tpot, vt = m.get("tpot"), m.get("vllm_tpot")
            isnan = lambda v: v is None or (isinstance(v, float) and v != v)
            if isnan(tpot) and not isnan(vt):
                cells.append(f"<span class=r>vLLM {vt:.2f}</span>"); continue   # 只有 vLLM 基线（dinfer 侧未就位）
            if isnan(tpot):
                cells.append("—"); continue
            ratio = f" <span class=r>({vt/tpot:.2f}×)</span>" if vt and not isnan(vt) else ""
            cells.append(f"{tpot:.2f}{ratio}")
        chg = last.get("change", "")
        if last.get("kind") == "baseline":
            chg = "【仅 vLLM 基线，dinfer 侧未就位】" + chg
        rows.append((model, gpu, len(rs), last.get("label", ""), last.get("date", ""), cells, chg))

# 只有质量对比、还没进 perf-trend 的模型（如 kaon3）：也进同一张总表，一行，perf 单元格留空。
for m in sorted(set(qa_report) - perf_models):
    rows.append((m, "—", 0, "仅端到端质量对比", "", ["—"] * len(GROUPS), "尚未接入四组 perf bench"))

def td(x): return f"<td>{x}</td>"
body = "\n".join(
    (f'<tr><td><a href="{html.escape(m)}/perf-trend.html#{html.escape(m)}/{html.escape(g)}/single"><b>{html.escape(m)}</b></a></td>'
     if g != "—" else f'<tr><td><b>{html.escape(m)}</b></td>')
    + (f'<td><a href="{html.escape(m)}/perf-trend.html#{html.escape(m)}/{html.escape(g)}/single">{html.escape(g)}</a></td>'
       if g != "—" else "<td>—</td>")
    + f"{td(n)}{td(html.escape(lbl))}{td(date)}" + "".join(td(c) for c in cells)
    + f"{td(qa_cell(m))}" + f'<td class=chg>{html.escape(chg)}</td></tr>'
    for m, g, n, lbl, date, cells, chg in rows)
qa_section = ""

page = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>dinfer perf index</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="5">
<style>
:root{{--fg:#1b1f23;--muted:#6a737d;--line:#e1e4e8;--bg:#fff;--hl:#f6f8fa}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--fg:#e6edf3;--muted:#8b949e;--line:#30363d;--bg:#0d1117;--hl:#161b22}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}}
h1{{font-size:20px;margin:0 0 4px}} .sub{{color:var(--muted);font-size:13px;margin-bottom:12px}}
table{{border-collapse:collapse;width:100%}} th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top;white-space:nowrap}}
th{{background:var(--hl);position:sticky;top:0}} td.chg{{white-space:normal;color:var(--muted);font-size:12px;max-width:520px}}
.r{{color:var(--muted);font-size:12px}} a{{color:inherit}}
</style></head><body>
<h1>dinfer perf index</h1>
<div class="sub">每个（模型 × 卡型）一行：最新主线轮次与四组 TPOT p50（ms，括号 = vLLM/dinfer 倍数，&gt;1 为我们快）。点模型或卡型进该模型的趋势页（URL 带 #模型/卡型/页签；页签含 single / pd / tp / dp / other / engine / probe）。数据来自各模型目录的 <code>perf-trend.json</code>。</div>
<table><thead><tr><th>模型</th><th>卡型</th><th>轮次数</th><th>最新轮</th><th>日期</th><th>c1</th><th>c8</th><th>c32</th><th>c8-long</th><th>质量详情</th><th>最新改动</th></tr></thead>
<tbody>
{body}
</tbody></table>
{qa_section}
</body></html>
"""
out = os.path.join(ROOT, "index.html")
open(out, "w").write(page)
print(f"{out}: {len(rows)} (模型 × 卡型) 行")
