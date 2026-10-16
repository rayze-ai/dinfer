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

# ---- CSV（长表，用户要求：每轮每指标的提升 + 改动说明）----
import csv
with open(os.path.join(ROOT, "perf-trend.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["round", "date", "change", "group", "metric", "dinfer", "vllm", "ratio_dinfer_over_vllm", "delta_pct_vs_prev_round", "better_is"])
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
                w.writerow([r["label"], r["date"], r["change"], SHORT[g], metric, f"{x[key]:.3f}", f"{x[vkey]:.3f}", f"{ratio:.3f}" if ratio != "" else "", d, "lower" if lower else "higher"])
        prev = r

# ---- HTML 面板：静态页，fetch 同目录 perf-trend.csv，每 10 s 重画；Chart.js 走 cdnjs ----
html = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>dinfer perf trend</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--card:#f6f7f9;--line:#e3e5e8}
@media (prefers-color-scheme: dark){:root:not([data-theme=light]){--bg:#111;--fg:#eee;--muted:#9a9a9a;--card:#1b1d20;--line:#2a2d31}}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
h1{font-size:18px;margin:0 0 4px}.sub{color:var(--muted);margin-bottom:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px}
.card h2{font-size:14px;margin:0 0 6px}
table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:right;vertical-align:top}
th:first-child,td:first-child,td.change,th.change{text-align:left}
.up{color:#1a7f37}.down{color:#c62828}.chg{max-width:420px;white-space:normal;color:var(--muted)}
.updated{color:var(--muted);font-size:12px}
</style></head><body>
<h1>dinfer vs vLLM v0.31.0 — Qwen3-0.6B，4090（每轮 bench_compare 四组）</h1>
<div class="sub">数据：<code>perf-trend.csv</code>（每 10 s 重读）。比值 = dinfer out tok/s ÷ vLLM；虚线 1.0 = 追平。TTFT/TPOT 越低越好。<span id="updated" class="updated"></span></div>
<div class="grid">
 <div class="card"><h2>out tok/s ÷ vLLM</h2><canvas id="ratio"></canvas></div>
 <div class="card"><h2>TTFT p50 (ms)，实线 dinfer / 虚线 vLLM</h2><canvas id="ttft"></canvas></div>
 <div class="card"><h2>TPOT p50 (ms)，实线 dinfer / 虚线 vLLM</h2><canvas id="tpot"></canvas></div>
 <div class="card"><h2>out tok/s，实线 dinfer / 虚线 vLLM</h2><canvas id="tps"></canvas></div>
</div>
<div class="card" style="margin-top:14px"><h2>每轮改动与提升（相对上一轮；绿色 = 变好）</h2><div id="table"></div></div>
<script>
const GROUPS=["c1","c8","c32","c8-long"], COLORS=["#1f77b4","#ff7f0e","#2ca02c","#d62728"];
const charts={};
function parseCSV(t){const lines=t.trim().split(/\r?\n/);const hdr=lines[0].split(",");return lines.slice(1).map(l=>{const cells=[];let cur="",q=false;for(const ch of l){if(ch=='"'){q=!q;continue}if(ch==','&&!q){cells.push(cur);cur="";continue}cur+=ch}cells.push(cur);const o={};hdr.forEach((h,i)=>o[h]=cells[i]);return o;});}
function build(rows){
  const rounds=[...new Set(rows.map(r=>r.round))];
  const change={};rows.forEach(r=>change[r.round]=r.change);
  const get=(metric,g,field)=>rounds.map(rd=>{const x=rows.find(r=>r.round==rd&&r.group==g&&r.metric==metric);return x?parseFloat(x[field]):null;});
  const mk=(id,metric,field,withV,refLine)=>{
    const ds=[];GROUPS.forEach((g,i)=>{ds.push({label:g,data:get(metric,g,field),borderColor:COLORS[i],backgroundColor:COLORS[i],tension:.15,pointRadius:3});
      if(withV)ds.push({label:g+" vLLM",data:get(metric,g,"vllm"),borderColor:COLORS[i],borderDash:[4,4],pointRadius:0,borderWidth:1});});
    if(refLine)ds.push({label:"vLLM = 1.0",data:rounds.map(()=>1),borderColor:"#888",borderDash:[6,3],pointRadius:0,borderWidth:1});
    const cfg={type:"line",data:{labels:rounds,datasets:ds},options:{animation:false,responsive:true,interaction:{mode:"index",intersect:false},
      plugins:{legend:{labels:{boxWidth:10,font:{size:11}}},tooltip:{callbacks:{afterTitle:(it)=>"改动: "+(change[it[0].label]||"")}}},
      scales:{y:{beginAtZero:true,ticks:{color:getComputedStyle(document.body).color}},x:{ticks:{color:getComputedStyle(document.body).color}}}}};
    if(charts[id]){charts[id].data=cfg.data;charts[id].update();}else charts[id]=new Chart(document.getElementById(id),cfg);
  };
  mk("ratio","out_tok_per_s","ratio_dinfer_over_vllm",false,true);
  mk("ttft","ttft_p50_ms","dinfer",true,false);
  mk("tpot","tpot_p50_ms","dinfer",true,false);
  mk("tps","out_tok_per_s","dinfer",true,false);
  // 表
  let h="<table><tr><th>轮</th><th class=change>改动</th>";GROUPS.forEach(g=>h+=`<th>${g} tok/s<br>(×vLLM, Δ)</th>`);GROUPS.forEach(g=>h+=`<th>${g} TTFT ms<br>(Δ)</th>`);GROUPS.forEach(g=>h+=`<th>${g} TPOT ms<br>(Δ)</th>`);h+="</tr>";
  rounds.forEach(rd=>{h+=`<tr><td>${rd}</td><td class="change chg">${change[rd]||"—"}</td>`;
    for(const m of ["out_tok_per_s","ttft_p50_ms","tpot_p50_ms"])GROUPS.forEach(g=>{const x=rows.find(r=>r.round==rd&&r.group==g&&r.metric==m);if(!x){h+="<td>—</td>";return;}
      const d=x.delta_pct_vs_prev_round===""?"":parseFloat(x.delta_pct_vs_prev_round);const cls=d===""?"":(d>0.5?"up":(d<-0.5?"down":""));const ds=d===""?"":` <span class="${cls}">${d>0?"+":""}${d.toFixed(0)}%</span>`;
      const v=parseFloat(x.dinfer);h+=m=="out_tok_per_s"?`<td>${v.toFixed(0)} (${parseFloat(x.ratio_dinfer_over_vllm).toFixed(2)}×)${ds}</td>`:`<td>${v.toFixed(m=="tpot_p50_ms"?2:1)} / ${parseFloat(x.vllm).toFixed(m=="tpot_p50_ms"?2:1)}${ds}</td>`;});
    h+="</tr>";});
  document.getElementById("table").innerHTML=h+"</table>";
  document.getElementById("updated").textContent=" 更新于 "+new Date().toLocaleTimeString();
}
async function refresh(){try{const t=await (await fetch("perf-trend.csv?ts="+Date.now(),{cache:"no-store"})).text();build(parseCSV(t));}catch(e){document.getElementById("updated").textContent=" 读取 CSV 失败: "+e;}}
refresh();setInterval(refresh,10000);
</script></body></html>
"""
open(os.path.join(ROOT, "perf-trend.html"), "w").write(html)

print(f"{len(rounds)} rounds → {ROOT}/perf-trend.{{md,json,csv,svg,html}}")
print("\n".join(lines[4:4 + 2 + len(rounds)]))
