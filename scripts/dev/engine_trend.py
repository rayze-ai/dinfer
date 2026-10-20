#!/usr/bin/env python3
"""引擎层面趋势（docs/design.md §15）：汇总 test/<model>/<日期>/engine*/ 下每轮 bench_engine.sh 的结果，输出
  test/<model>/engine-trend.csv   长表：round,date,gpu,change,side,rate,metric,value（metric ∈ goodput_rps / request_rps / out_tps /
                                  ttft_p50|p90|p99|max / ttft_1st_p50 / tpot_p50|p90|p99|max / cache_hit / completed / failed / max_good_rate）
  test/<model>/engine-trend.md    每轮一节：SLO、负载一行 + 表（每率两端各一行）+ "SLO 下最大 req/s"（good/completed ≥ 0.95 的最大 rate）
  test/<model>/engine-trend.html  Chart.js：选轮次 → goodput vs rate、TTFT p99 vs rate、TPOT p99 vs rate、cache hit vs rate（两端对比），
                                  再一张"各轮 SLO 下最大 req/s"柱图（dinfer vs vLLM）
数据来源：每个 engine*/ 目录里 {dinfer,vllm}_r<rate>.json（bench --out）、CHANGE、GPU。独立于 bench_trend.py（不同口径，不耦合）。
用法：python3 -I scripts/dev/engine_trend.py [--root test/qwen35]
"""
import csv, glob, json, os, re, sys

ROOT = sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else "test/qwen35"
MODEL = os.path.basename(os.path.normpath(ROOT))
SIDES = ["dinfer", "vllm"]
METRICS = [  # (key, 取值函数, 表头, 小数位)
    ("goodput_rps", lambda d: d.get("goodput_rps"), "goodput", 2),
    ("request_rps", lambda d: d["request_throughput_rps"], "req/s", 2),
    ("out_tps", lambda d: d["output_token_throughput_tps"], "out tok/s", 0),
    ("ttft_p50", lambda d: d["ttft_ms"]["p50"], "TTFT p50", 1),
    ("ttft_p90", lambda d: d["ttft_ms"]["p90"], "TTFT p90", 1),
    ("ttft_p99", lambda d: d["ttft_ms"]["p99"], "TTFT p99", 1),
    ("ttft_max", lambda d: d["ttft_ms"]["max"], "TTFT max", 1),
    ("ttft_1st_p50", lambda d: d.get("ttft_first_chunk_ms", {}).get("p50"), "TTFT1st p50", 1),
    ("tpot_p50", lambda d: d["tpot_ms"]["p50"], "TPOT p50", 1),
    ("tpot_p90", lambda d: d["tpot_ms"]["p90"], "TPOT p90", 1),
    ("tpot_p99", lambda d: d["tpot_ms"]["p99"], "TPOT p99", 1),
    ("tpot_max", lambda d: d["tpot_ms"]["max"], "TPOT max", 1),
    ("cache_hit", lambda d: d.get("prefix_cache_hit_ratio"), "cache hit", 3),
    ("completed", lambda d: d["completed"], "完成", 0),
    ("failed", lambda d: d["failed"], "失败", 0),
]


def read1(d, name, default=""):
    p = os.path.join(d, name)
    return open(p).read().strip() if os.path.exists(p) else default


def load_round(d):
    """engine*/ 目录 → {label, date, gpu, change, slo, load, data: {side: {rate: json}}}"""
    data = {s: {} for s in SIDES}
    for f in glob.glob(os.path.join(d, "*_r*.json")):
        m = re.match(r"(dinfer|vllm)_r([\d.]+)\.json$", os.path.basename(f))
        if not m:
            continue
        try:
            j = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        if "request_throughput_rps" not in j or "ttft_ms" not in j:
            continue  # 不是 bins/bench 的输出（如 vllm bench serve 的 JSON）
        data[m.group(1)][float(m.group(2))] = j
    if not any(data.values()):
        return None
    any_json = next(j for s in SIDES for j in data[s].values())
    a = any_json.get("args", {})
    slo = a.get("slo", {})
    load = (f"prompt mix {a.get('prompt_len_mix') or a.get('max_tokens')}；out mix {a.get('output_len_mix') or a.get('max_tokens')}；"
            f"{'ignore_eos' if a.get('ignore_eos') else '不 ignore_eos'}；prefix_repeat {a.get('prefix_repeat')}；multi_turn {a.get('multi_turn', 1)}；"
            f"每率 {a.get('num_requests_requested', any_json.get('num_requests'))} 条")
    date = os.path.basename(os.path.dirname(d))
    return {"label": f"{date}/{os.path.basename(d)}", "date": date, "gpu": read1(d, "GPU", "?"), "change": read1(d, "CHANGE"),
            "mtime": os.path.getmtime(d),
            "slo": f"TTFT ≤ {slo.get('ttft_ms', 0):.0f} ms 且 TPOT ≤ {slo.get('tpot_ms', 0):.0f} ms", "load": load, "data": data}


def round_key(r):
    m = re.search(r"engine_?r?(\d*)$", r["label"])  # engine_2 / engine_r8 都按尾数排
    # 无尾数的具名轮（engine_final / engine_merge / engine_r8b）按"有尾数之后"排，内部按 mtime——否则会和 engine_1 同分被字母序挪到前面
    n = int(m.group(1)) if (m and m.group(1)) else None
    return (r["date"], 0 if n is not None else 1, n if n is not None else 0, r.get("mtime", 0), r["label"])


GOOD_FRAC = 0.95


def max_good_rate(series):
    """SLO 下最大 req/s：good_requests / completed ≥ 0.95 的最大到达率（没有 → 0）。
    不用 goodput ≥ 0.95×rate：短跑（每率 96 条）的实际 req/s 因收尾长尾总低于名义 rate，会把全好的点也判掉。"""
    best = 0.0
    for rate, d in sorted(series.items()):
        g, c = d.get("good_requests"), d.get("completed", 0)
        if g is not None and c > 0 and g / c >= GOOD_FRAC:
            best = max(best, rate)
    return best


def fmt(v, dec):
    if v is None or (isinstance(v, float) and v != v):
        return "-"
    return f"{v:.{dec}f}" if dec else f"{v:.0f}"


rounds = [r for r in (load_round(d) for d in glob.glob(os.path.join(ROOT, "*", "engine*")) if os.path.isdir(d)) if r]
rounds.sort(key=round_key)
if not rounds:
    print(f"no engine*/ rounds under {ROOT}")
    sys.exit(0)

# ---- CSV（长表）----
os.makedirs(ROOT, exist_ok=True)
with open(os.path.join(ROOT, "engine-trend.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["round", "date", "gpu", "change", "side", "rate", "metric", "value"])
    for r in rounds:
        for s in SIDES:
            for rate, d in sorted(r["data"][s].items()):
                for key, get, _, dec in METRICS:
                    v = get(d)
                    if v is not None:
                        w.writerow([r["label"], r["date"], r["gpu"], r["change"], s, rate, key, f"{v:.4f}"])
            if r["data"][s]:
                w.writerow([r["label"], r["date"], r["gpu"], r["change"], s, "", "max_good_rate", f"{max_good_rate(r['data'][s]):.2f}"])

# ---- Markdown ----
lines = [f"# 引擎层面趋势 — {MODEL}", "",
         "口径：泊松开环到达率扫描；goodput = 满足 SLO（TTFT 且 TPOT）的完成数 / 墙钟；**SLO 下最大 req/s** = goodput ≥ 0.95×rate 的最大到达率；"
         "cache hit = Σcached_tokens / Σprompt_tokens；TTFT = 首个带内容 chunk。详见 docs/design.md §15、docs/loops/engine-bench.md。", ""]
lines += ["| 轮 | 卡 | 改动 | SLO 下最大 req/s dinfer | vLLM | 比 |", "|---|---|---|---|---|---|"]
for r in rounds:
    a, b = max_good_rate(r["data"]["dinfer"]), max_good_rate(r["data"]["vllm"])
    ratio = f"{a / b:.2f}×" if b else "-"
    lines.append(f"| {r['label']} | {r['gpu']} | {r['change'] or '-'} | {fmt(a, 0)} | {fmt(b, 0) if r['data']['vllm'] else '-'} | {ratio} |")
lines.append("")
for r in rounds:
    lines += [f"## {r['label']}（{r['gpu']}）{('— ' + r['change']) if r['change'] else ''}", "", f"SLO：{r['slo']}。负载：{r['load']}", "",
              "| rate | 端 | 完成/失败 | goodput | req/s | out tok/s | TTFT p50/p90/p99/max | TPOT p50/p90/p99/max | cache hit |", "|---|---|---|---|---|---|---|---|---|"]
    rates = sorted(set(k for s in SIDES for k in r["data"][s]))
    for rate in rates:
        for s in SIDES:
            d = r["data"][s].get(rate)
            if not d:
                continue
            gp = d.get("goodput_rps")
            gp = "-" if gp is None else f"{gp:.2f} ({d.get('good_requests')}/{d['completed']})"
            t, p = d["ttft_ms"], d["tpot_ms"]
            lines.append(f"| {rate:g} | {s} | {d['completed']}/{d['failed']} | {gp} | {d['request_throughput_rps']:.2f} | {d['output_token_throughput_tps']:.0f} | "
                         f"{t['p50']:.1f}/{t['p90']:.1f}/{t['p99']:.1f}/{t['max']:.1f} | {p['p50']:.1f}/{p['p90']:.1f}/{p['p99']:.1f}/{p['max']:.1f} | {fmt(d.get('prefix_cache_hit_ratio'), 3)} |")
    lines.append("")
open(os.path.join(ROOT, "engine-trend.md"), "w").write("\n".join(lines))

# ---- HTML（Chart.js，fetch 同目录 CSV；两端固定两色：dinfer 蓝 / vLLM 橙，legend 常显，hover 十字线 + tooltip）----
html = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>engine trend — __MODEL__</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#fcfcfb;--fg:#0b0b0b;--muted:#52514e;--card:#f3f3f1;--line:#e3e5e8;--s1:#2a78d6;--s2:#eb6834}
@media (prefers-color-scheme: dark){:root:not([data-theme=light]){--bg:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--card:#232322;--line:#2f2f2d;--s1:#3987e5;--s2:#d95926}}
:root[data-theme=dark]{--bg:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--card:#232322;--line:#2f2f2d;--s1:#3987e5;--s2:#d95926}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
h1{font-size:18px;margin:0 0 4px}.sub{color:var(--muted);margin-bottom:12px;font-size:13px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:8px 0}
select{font:inherit;padding:4px 8px;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px}
.card h2{font-size:14px;margin:0 0 6px;font-weight:600}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin-top:14px}th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:right}
th:first-child,td:first-child,td.l{text-align:left}.muted{color:var(--muted)}
</style></head><body>
<h1>engine trend — __MODEL__ <a href="../index.html" style="font-size:13px;font-weight:normal;margin-left:12px">← 全部模型索引</a></h1>
<div class="sub">引擎层面口径（design §15）：泊松到达率扫描；goodput = 满足 SLO 的完成数/墙钟；SLO 下最大 req/s = good/completed ≥ 0.95 的最大到达率。数据 <code>engine-trend.csv</code>，每 10 s 重读。</div>
<div class="row"><label>轮次 <select id="round"></select></label><span id="meta" class="muted"></span></div>
<div class="grid">
 <div class="card"><h2>goodput（req/s）vs 到达率</h2><canvas id="c-goodput"></canvas></div>
 <div class="card"><h2>TTFT p99（ms）vs 到达率</h2><canvas id="c-ttft"></canvas></div>
 <div class="card"><h2>TPOT p99（ms）vs 到达率</h2><canvas id="c-tpot"></canvas></div>
 <div class="card"><h2>prefix cache 命中率 vs 到达率</h2><canvas id="c-cache"></canvas></div>
 <div class="card" style="grid-column:1/-1"><h2>各轮 SLO 下最大 req/s</h2><canvas id="c-max"></canvas></div>
</div>
<div id="tbl"></div>
<script>
const COL={dinfer:getComputedStyle(document.documentElement).getPropertyValue('--s1').trim(),vllm:getComputedStyle(document.documentElement).getPropertyValue('--s2').trim()};
const charts={};
function parseCSV(t){const L=t.trim().split(/\r?\n/);const h=L.shift().split(",");return L.map(l=>{const c=[];let cur="",q=false;for(const ch of l){if(ch=='"'){q=!q}else if(ch==','&&!q){c.push(cur);cur=""}else cur+=ch}c.push(cur);const o={};h.forEach((k,i)=>o[k]=c[i]);return o;});}
function upsert(id,cfg){if(charts[id]){charts[id].data=cfg.data;charts[id].options=cfg.options;charts[id].update();}else charts[id]=new Chart(document.getElementById(id),cfg);}
function lineCfg(rates,series,yLabel,yMax){
  const ds=Object.entries(series).map(([s,m])=>({label:s,data:rates.map(r=>m[r]??null),borderColor:COL[s],backgroundColor:COL[s],borderWidth:2,pointRadius:4,pointHoverRadius:6,spanGaps:true,tension:0}));
  return {type:"line",data:{labels:rates.map(String),datasets:ds},options:{animation:false,responsive:true,interaction:{mode:"index",intersect:false},
    plugins:{legend:{display:true,position:"bottom"},tooltip:{enabled:true}},
    scales:{x:{title:{display:true,text:"到达率 req/s"},grid:{color:"rgba(128,128,128,.15)"}},y:{beginAtZero:true,suggestedMax:yMax,title:{display:true,text:yLabel},grid:{color:"rgba(128,128,128,.15)"}}}}};
}
function build(rows){
  const byRound={};rows.forEach(r=>{(byRound[r.round]??={gpu:r.gpu,change:r.change,rows:[]}).rows.push(r)});
  const rounds=Object.keys(byRound);
  const sel=document.getElementById("round");const prev=sel.value||location.hash.slice(1)||rounds[rounds.length-1];
  sel.innerHTML=rounds.map(r=>`<option ${r===prev?"selected":""}>${r}</option>`).join("");
  sel.onchange=()=>{location.hash=sel.value;draw(byRound,sel.value)};
  draw(byRound,sel.value);
  // 各轮最大 req/s 柱图
  const mg=rounds.map(r=>{const o={};byRound[r].rows.filter(x=>x.metric==="max_good_rate").forEach(x=>o[x.side]=+x.value);return o;});
  upsert("c-max",{type:"bar",data:{labels:rounds,datasets:["dinfer","vllm"].map(s=>({label:s,data:mg.map(o=>o[s]??null),backgroundColor:COL[s],borderRadius:4,borderSkipped:"bottom",maxBarThickness:28}))},
    options:{animation:false,responsive:true,plugins:{legend:{position:"bottom"}},scales:{y:{beginAtZero:true,title:{display:true,text:"req/s"}},x:{grid:{display:false}}}}});
}
function draw(byRound,rk){
  const R=byRound[rk];if(!R)return;
  document.getElementById("meta").textContent=`${R.gpu} · ${R.change||"（无改动说明）"}`;
  const rates=[...new Set(R.rows.filter(x=>x.rate).map(x=>+x.rate))].sort((a,b)=>a-b);
  const pick=m=>{const o={dinfer:{},vllm:{}};R.rows.filter(x=>x.metric===m).forEach(x=>o[x.side][+x.rate]=+x.value);for(const s in o)if(!Object.keys(o[s]).length)delete o[s];return o;};
  const g=pick("goodput_rps");g.ideal=Object.fromEntries(rates.map(r=>[r,r]));
  const cfg=lineCfg(rates,g,"req/s");cfg.data.datasets.find(d=>d.label==="ideal").borderDash=[4,4];cfg.data.datasets.find(d=>d.label==="ideal").borderColor="rgba(128,128,128,.6)";cfg.data.datasets.find(d=>d.label==="ideal").pointRadius=0;
  upsert("c-goodput",cfg);
  upsert("c-ttft",lineCfg(rates,pick("ttft_p99"),"ms"));
  upsert("c-tpot",lineCfg(rates,pick("tpot_p99"),"ms"));
  upsert("c-cache",lineCfg(rates,pick("cache_hit"),"命中率",1));
  // 表
  const M=["goodput_rps","request_rps","out_tps","ttft_p50","ttft_p99","ttft_max","tpot_p50","tpot_p99","tpot_max","cache_hit","completed","failed"];
  const H=["goodput","req/s","out tok/s","TTFT p50","TTFT p99","TTFT max","TPOT p50","TPOT p99","TPOT max","cache hit","完成","失败"];
  let h=`<table><tr><th>rate</th><th>端</th>${H.map(x=>`<th>${x}</th>`).join("")}</tr>`;
  for(const r of rates)for(const s of ["dinfer","vllm"]){const rr=R.rows.filter(x=>x.side===s&&+x.rate===r);if(!rr.length)continue;const v=m=>{const x=rr.find(y=>y.metric===m);return x?(+x.value).toFixed(m.startsWith("cache")?3:(m==="completed"||m==="failed"||m==="out_tps")?0:m.includes("rps")?2:1):"-"};
    h+=`<tr><td class="l">${r}</td><td class="l">${s}</td>${M.map(m=>`<td>${v(m)}</td>`).join("")}</tr>`}
  document.getElementById("tbl").innerHTML=h+"</table>";
}
async function refresh(){try{const t=await (await fetch("engine-trend.csv?ts="+Date.now(),{cache:"no-store"})).text();build(parseCSV(t));}catch(e){console.error(e)}}
refresh();setInterval(refresh,10000);
</script></body></html>
"""
open(os.path.join(ROOT, "engine-trend.html"), "w").write(html.replace("__MODEL__", MODEL))
print(f"engine-trend: {len(rounds)} round(s) → {ROOT}/engine-trend.{{csv,md,html}}")
last = rounds[-1]
print(f"latest {last['label']} ({last['gpu']}): max good rate dinfer {max_good_rate(last['data']['dinfer']):g} / vllm {max_good_rate(last['data']['vllm']):g} req/s")
