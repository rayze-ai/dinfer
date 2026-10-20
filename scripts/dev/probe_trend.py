#!/usr/bin/env python3
"""探针趋势（不是四组闭环、不是开环泊松的那些测量：TTFT-长度曲线、kernel 分解、微基准伸缩、第三方 kernel 对照）。
用户 2026-10-20：所有测试结果都要进 CSV、页面刷新可见，不能只留 markdown。

输入：test/<model>/<日期>/probe_*/probe.csv（长表，列固定：
  probe,date,gpu,model,group,metric,side,x,value,unit,note
  group 自定义（ttft_len / kernel_1k / gdn_scale …），同 group 内同 metric 的 x 成一条曲线，side 区分 dinfer / vllm）
输出：test/<model>/probe-trend.csv（所有 probe 拼接）、probe-trend.md（按 probe → group 的表）、probe-trend.html（Chart.js：
  选 probe → 每个 group 一张图：x 数值型画折线（dinfer 蓝实 / vLLM 橙虚），x 类别型画分组柱）。
用法：python3 scripts/dev/probe_trend.py [--root test/qwen35]
"""
import csv, glob, html, json, os, sys

ROOT = sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else "test/qwen35"
MODEL = os.path.basename(ROOT.rstrip("/"))
COLS = ["probe", "date", "gpu", "model", "group", "metric", "side", "x", "value", "unit", "note"]

rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "*", "probe_*", "probe.csv"))):
    with open(f, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("probe") and r.get("value", "") != "":
                rows.append({c: r.get(c, "") for c in COLS})
if not rows:
    print("no probe.csv under", ROOT); sys.exit(0)

with open(os.path.join(ROOT, "probe-trend.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)

# ---- markdown：probe → group → 表（行 = x，列 = metric/side）----
lines = [f"# probe trend — {MODEL}", "",
         "> 探针测量（TTFT-长度 / kernel 分解 / 微基准伸缩 / 第三方对照），与 perf-trend（四组闭环）、engine-trend（开环泊松）并列。",
         "> 数据源 `test/<model>/<日期>/probe_*/probe.csv`；本文件与 `probe-trend.{csv,html}` 由 `scripts/dev/probe_trend.py` 生成。", ""]
by = {}
for r in rows:
    by.setdefault((r["probe"], r["date"], r["gpu"]), {}).setdefault(r["group"], []).append(r)
for (p, d, g), groups in sorted(by.items(), key=lambda kv: kv[0][1]):
    lines += [f"## {p}（{d}，{g}）", ""]
    for grp, rs in groups.items():
        cols = sorted({(r["metric"], r["side"]) for r in rs})
        xs = []
        for r in rs:
            if r["x"] not in xs: xs.append(r["x"])
        unit = next((r["unit"] for r in rs if r["unit"]), "")
        lines += [f"### {grp}（{unit}）", "", "| x | " + " | ".join(f"{m} {s}" for m, s in cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for x in xs:
            cells = []
            for m, s in cols:
                v = [r["value"] for r in rs if r["x"] == x and r["metric"] == m and r["side"] == s]
                cells.append(v[0] if v else "")
            lines.append(f"| {x} | " + " | ".join(cells) + " |")
        notes = [f"- {r['x']} {r['metric']} {r['side']}：{r['note']}" for r in rs if r["note"]]
        lines += [""] + notes + [""]
open(os.path.join(ROOT, "probe-trend.md"), "w", encoding="utf-8").write("\n".join(lines))

# ---- HTML（Chart.js，内嵌数据；x 全数值 → 折线，否则分组柱）----
page = """<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>probe trend — __MODEL__</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>body{font-family:system-ui,sans-serif;margin:16px;max-width:1200px}h2{margin:24px 0 4px}.chart{height:320px;margin:8px 0 24px}
select{font-size:14px;padding:4px}.note{color:#666;font-size:13px;margin:0 0 8px}a{color:#06c}</style></head><body>
<h1>probe trend — __MODEL__ <small><a href="probe-trend.md">md</a> · <a href="probe-trend.csv">csv</a> · <a href="perf-trend.html">perf</a> · <a href="engine-trend.html">engine</a> · <a href="../index.html">索引</a></small></h1>
<p class="note">探针测量（TTFT-长度曲线 / kernel 分解 / 微基准伸缩 / 第三方 kernel 对照）。dinfer 蓝实线 / vLLM 橙虚线。</p>
<label>probe <select id="probe"></select></label>
<div id="charts"></div>
<script>
const ROWS=__ROWS__;
const probes=[...new Set(ROWS.map(r=>r.probe+"|"+r.date+"|"+r.gpu))];
const sel=document.getElementById("probe");probes.forEach(p=>{const o=document.createElement("option");o.value=p;o.textContent=p.replace(/\\|/g,"  ·  ");sel.appendChild(o)});
const COLOR={dinfer:"#1f77b4",vllm:"#ff7f0e"};
function render(){const [p,d,g]=sel.value.split("|");const rs=ROWS.filter(r=>r.probe===p&&r.date===d&&r.gpu===g);
 const box=document.getElementById("charts");box.innerHTML="";
 const groups=[...new Set(rs.map(r=>r.group))];
 for(const grp of groups){const gr=rs.filter(r=>r.group===grp);const unit=(gr.find(r=>r.unit)||{}).unit||"";
  const h=document.createElement("h2");h.textContent=grp+(unit?"（"+unit+"）":"");box.appendChild(h);
  const notes=gr.filter(r=>r.note).map(r=>r.x+" "+r.metric+" "+r.side+"："+r.note);
  if(notes.length){const n=document.createElement("p");n.className="note";n.textContent=notes.join("；");box.appendChild(n)}
  const div=document.createElement("div");div.className="chart";const cv=document.createElement("canvas");div.appendChild(cv);box.appendChild(div);
  const xs=[...new Set(gr.map(r=>r.x))];const numeric=xs.every(x=>/^-?\\d+(\\.\\d+)?$/.test(x));
  const series=[...new Set(gr.map(r=>r.metric+"|"+r.side))];
  const ds=series.map((s,i)=>{const [m,side]=s.split("|");const pts=gr.filter(r=>r.metric===m&&r.side===side);
   const c=COLOR[side]||"hsl("+(i*47%360)+",60%,45%)";
   return numeric?{label:m+" "+side,data:pts.map(r=>({x:+r.x,y:+r.value})),borderColor:c,backgroundColor:c,borderDash:side==="vllm"?[6,4]:[],tension:0,pointRadius:4,showLine:true}
                 :{label:m+" "+side,data:xs.map(x=>{const r=pts.find(q=>q.x===x);return r?+r.value:null}),backgroundColor:c,borderColor:c}});
  new Chart(cv,{type:numeric?"line":"bar",data:numeric?{datasets:ds}:{labels:xs,datasets:ds},
   options:{responsive:true,maintainAspectRatio:false,interaction:{mode:"nearest",intersect:false},
    scales:numeric?{x:{type:"linear",title:{display:true,text:"x（"+grp+"）"}},y:{title:{display:true,text:unit},beginAtZero:true}}:{y:{title:{display:true,text:unit},beginAtZero:true}},
    plugins:{legend:{position:"bottom"},tooltip:{callbacks:{label:c=>c.dataset.label+": "+(c.parsed.y??c.raw)+" "+unit}}}}});}}
sel.onchange=render;if(location.hash){const v=decodeURIComponent(location.hash.slice(1));if(probes.includes(v))sel.value=v}
if(probes.length)render();
</script></body></html>"""
open(os.path.join(ROOT, "probe-trend.html"), "w", encoding="utf-8").write(
    page.replace("__MODEL__", html.escape(MODEL)).replace("__ROWS__", json.dumps(rows, ensure_ascii=False)))
print(f"probe-trend: {len(rows)} rows, {len(by)} probes → {ROOT}/probe-trend.{{csv,md,html}}")
