#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kaon-V3 端到端四配置压测对比页（dinfer eager/mega vs vLLM eager/非eager）。
读 test/kaon3/<date>/bench<N>/<TAG>_<group>.json，出一张 Chart.js 对比页（output tok/s / TTFT / TPOT / goodput，
按负载组分 x 轴，四配置四条线）。进 Dashboard 的"质量/性能详情"。
用法：python3 -I scripts/dev/kaon3_bench_chart.py <bench_dir>
"""
import sys, os, json, glob, html

TAGS = [("R0_dinfer_eager", "dinfer eager"), ("R1_dinfer_mega", "dinfer mega(非eager)"),
        ("Ve_vllm_eager", "vLLM eager"), ("Vn_vllm_ne", "vLLM 非eager")]
GROUPS = [("c1_p128_o128", "c1 p128"), ("c8_p128_o128", "c8 p128"),
          ("c16_p128_o128", "c16 p128"), ("c8_p1024_o128", "c8 p1024(长)")]
METRICS = [("output_token_throughput_tps", "输出 tok/s（越高越好）"),
           ("ttft_p50", "TTFT p50 ms（越低越好）"), ("tpot_p50", "TPOT p50 ms（越低越好）"),
           ("goodput_rps", "goodput req/s（SLO 内，越高越好）")]

def getm(d, key):
    # ttft_ms / tpot_ms 是嵌套 {mean,p50,...}
    if key in ("ttft_p50", "tpot_p50"):
        base = d.get(key.split("_")[0] + "_ms", {})
        return base.get("p50") if isinstance(base, dict) else None
    return d.get(key)

def main():
    d = sys.argv[1]
    data = {}  # tag -> group -> json
    for tag, _ in TAGS:
        for g, _ in GROUPS:
            fp = os.path.join(d, f"{tag}_{g}.json")
            if os.path.exists(fp):
                try: data.setdefault(tag, {})[g] = json.load(open(fp))
                except Exception: pass
    # 每个指标一个数据集：labels = 负载组；每配置一条线
    charts = []
    for mkey, mlabel in METRICS:
        series = []
        for tag, tlabel in TAGS:
            vals = [getm(data.get(tag, {}).get(g, {}), mkey) for g, _ in GROUPS]
            series.append({"label": tlabel, "data": vals})
        charts.append({"title": mlabel, "labels": [gl for _, gl in GROUPS], "series": series})
    # client 开销核对（验收 v2：压测端 < 核数）
    client = []
    for tag, tlabel in TAGS:
        for g, gl in GROUPS:
            j = data.get(tag, {}).get(g)
            if j:
                cc = j.get("client_cpu_cores") or j.get("client", {}).get("cores")
                client.append(f"{tlabel}/{gl}: client {cc if cc is not None else '?'} cores")
    doc = f"""<!doctype html><html lang=zh><head><meta charset=utf-8>
<title>Kaon-V3 四配置压测</title><meta name=viewport content="width=device-width,initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{{--bg:#fff;--fg:#1a1a1a;--line:#e0e0e0;--muted:#777}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#16181c;--fg:#e6e6e6;--line:#2c2f36;--muted:#999}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.6 system-ui,sans-serif;margin:0;padding:16px}}
h1{{font-size:20px}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;max-width:1100px}}
@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
.card{{border:1px solid var(--line);border-radius:8px;padding:12px}} .note{{color:var(--muted);font-size:12px;margin-top:10px}}
canvas{{max-height:260px}}
</style></head><body>
<h1>Kaon-V3 端到端压测 · dinfer vs vLLM（5090 单卡，各独占 GPU + 绑核 12C，压测端 16C）</h1>
<div class=grid>{''.join(f'<div class=card><canvas id="c{i}"></canvas></div>' for i in range(len(charts)))}</div>
<p class=note>口径：temp=0 贪心、ignore_eos、同 chat 模板。eager=逐 kernel / enforce_eager；非eager=cudagraph+融合(vLLM) 或 megakernel(dinfer)。<br>
client 开销（应 &lt; 16 核）：{html.escape(' | '.join(client[:8]))}</p>
<script>
const COLORS=['#2563eb','#16a34a','#b8860b','#c0392b'];
const charts={json.dumps(charts, ensure_ascii=False)};
charts.forEach((ch,i)=>{{
  new Chart(document.getElementById('c'+i),{{type:'bar',
    data:{{labels:ch.labels,datasets:ch.series.map((s,j)=>({{label:s.label,data:s.data,backgroundColor:COLORS[j]}}))}},
    options:{{responsive:true,plugins:{{title:{{display:true,text:ch.title}},legend:{{position:'bottom'}}}},
      scales:{{y:{{beginAtZero:true}}}}}}}});
}});
</script></body></html>"""
    out = os.path.join(d, "chart.html")
    open(out, "w", encoding="utf-8").write(doc)
    print(f"写 {out}")

    # CSV 长表（每行：日期,轮次目录,配置,eager?,负载组,并发,prompt,指标,值）。项目硬规则：每轮进 CSV 趋势。
    import csv, datetime
    date = datetime.date.today().isoformat()
    runtag = os.path.basename(d.rstrip("/"))
    csv_cols = ["output_token_throughput_tps", "ttft_p50", "tpot_p50", "goodput_rps",
                "request_throughput_rps", "completed", "failed"]
    rows = []
    for tag, tlabel in TAGS:
        eager = "eager" if tag in ("R0_dinfer_eager", "Ve_vllm_eager") else "non-eager"
        for g, gl in GROUPS:
            j = data.get(tag, {}).get(g)
            if not j: continue
            c = j.get("concurrency"); pl = j.get("prompt_len_target")
            for mk in csv_cols:
                v = getm(j, mk) if mk in ("ttft_p50", "tpot_p50") else j.get(mk)
                rows.append([date, runtag, tlabel, eager, gl, c, pl, mk, v])
    csv_path = os.path.join(d, "bench.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "run", "config", "mode", "group", "concurrency", "prompt_len", "metric", "value"])
        w.writerows(rows)
    print(f"写 {csv_path}（{len(rows)} 行）")
    # 同时追加到模型级累计 CSV（趋势用）
    acc = os.path.join(os.path.dirname(os.path.dirname(d)), "perf-bench.csv")
    newfile = not os.path.exists(acc)
    with open(acc, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if newfile:
            w.writerow(["date", "run", "config", "mode", "group", "concurrency", "prompt_len", "metric", "value"])
        w.writerows(rows)
    print(f"追加 {acc}")

if __name__ == "__main__":
    main()
