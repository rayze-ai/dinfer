#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dinfer vs vLLM 端到端生成质量 + 性能对比（Kaon-V3）。
对 test/kaon3/qa_zh/prompts.jsonl 的每条中文问答，分别打两边的 /v1/chat/completions，
抓：生成文本、TTFT、tok/s、是否乱码、是否命中 expect 关键词（答非所问初筛）。
产物：
  test/kaon3/qa_zh/result_<tag>.jsonl   每条一行（原始）
  test/kaon3/qa_zh/report.html          并排对比页（进 dashboard，含乱码/命中/性能）
用法：
  python3 -I scripts/dev/kaon3_qa_compare.py run  --tag dinfer --url http://127.0.0.1:PORT --model NAME
  python3 -I scripts/dev/kaon3_qa_compare.py run  --tag vllm   --url http://127.0.0.1:PORT --model NAME
  python3 -I scripts/dev/kaon3_qa_compare.py report               # 合并两边 → report.html
判定口径（机器可判，非主观）：
  乱码：① 非 CJK/ASCII/常用标点 占比 >15%；② 任一字符连续重复 >20 次；③ 空输出。
  答非所问（初筛）：expect 关键词不在输出里（仅初筛，命中不代表答对，但不命中强提示跑偏）。
  性能：TTFT（首字节到首 token delta）、tok/s（completion_tokens / 总耗时）。
"""
import sys, os, json, time, re, html, urllib.request

ROOT = os.path.join("test", "kaon3", os.environ.get("QA_DIR", "qa_zh"))
PROMPTS = os.path.join(ROOT, "prompts.jsonl")

def load_prompts():
    return [json.loads(l) for l in open(PROMPTS, encoding="utf-8") if l.strip()]

def is_garbled(text):
    if not text or not text.strip():
        return True, "空输出"
    # 连续重复单字符 >20
    if re.search(r"(.)\1{20,}", text):
        return True, "字符重复>20"
    # 非 CJK / ASCII可见 / 常用标点 占比
    ok = 0
    for ch in text:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF: ok += 1          # CJK
        elif 0x20 <= o <= 0x7E: ok += 1             # ASCII 可见
        elif ch in " \n\t，。！？、；：""''（）《》…—·%": ok += 1
        elif 0x3000 <= o <= 0x303F: ok += 1         # CJK 标点
        elif 0xFF00 <= o <= 0xFFEF: ok += 1         # 全角
    bad = len(text) - ok
    if len(text) > 0 and bad / len(text) > 0.15:
        return True, f"异常字符占比{bad/len(text):.0%}"
    return False, ""

def call(url, model, q, max_tokens=512, timeout=120):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": q}],
        "temperature": 0.0, "max_tokens": max_tokens, "stream": True,
    }).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; chunks = []; ntok = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": break
            try: obj = json.loads(data)
            except Exception: continue
            delta = (obj.get("choices") or [{}])[0].get("delta", {}).get("content")
            if delta:
                if ttft is None: ttft = time.time() - t0
                chunks.append(delta); ntok += 1
    dt = time.time() - t0
    text = "".join(chunks)
    return {"text": text, "ttft": ttft, "elapsed": dt, "ntok": ntok,
            "tok_s": (ntok / dt) if dt > 0 else 0.0}

def run(tag, url, model):
    out = os.path.join(ROOT, f"result_{tag}.jsonl")
    # warmup：先打一个一次性请求，消除首请求的懒初始化偏差（两边对齐）。
    try:
        call(url, model, "你好", max_tokens=8)
        print(f"[{tag}] warmup done", flush=True)
    except Exception as e:
        print(f"[{tag}] warmup failed: {e}", flush=True)
    rows = []
    for p in load_prompts():
        mt = int(p.get("max_tokens", 512))
        try:
            r = call(url, model, p["q"], max_tokens=mt)
            garb, why = is_garbled(r["text"])
            hit = p["expect"] in r["text"]
            rec = {**p, "tag": tag, **r, "garbled": garb, "garble_why": why, "kw_hit": hit}
        except Exception as e:
            rec = {**p, "tag": tag, "text": "", "error": str(e), "garbled": True,
                   "garble_why": f"请求失败:{e}", "kw_hit": False, "ttft": None, "tok_s": 0, "ntok": 0, "elapsed": 0}
        rows.append(rec)
        print(f"[{tag}] {p['id']:8s} garb={rec['garbled']} hit={rec.get('kw_hit')} "
              f"tok/s={rec.get('tok_s',0):.1f} ttft={rec.get('ttft')} :: {rec['text'][:60]!r}", flush=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"写 {out}（{len(rows)} 条）")

def esc(s): return html.escape(s or "")

def _all_tags():
    tags = []
    for fp in sorted(glob.glob(os.path.join(ROOT, "result_*.jsonl"))):
        t = os.path.basename(fp)[len("result_"):-len(".jsonl")]
        tags.append(t)
    # 固定展示序：dinfer 先，其余按名
    order = {"dinfer": 0, "vllm_eager": 1, "vllm_compile": 2}
    return sorted(tags, key=lambda t: (order.get(t, 9), t))

def report():
    prompts = load_prompts()
    expect = {p["id"]: p["expect"] for p in prompts}
    tags = _all_tags()
    data = {}
    for tag in tags:
        fp = os.path.join(ROOT, f"result_{tag}.jsonl")
        d = {}
        for l in open(fp, encoding="utf-8"):
            if not l.strip(): continue
            r = json.loads(l)
            r["kw_hit"] = expect.get(r["id"], "") in (r.get("text") or "")
            d[r["id"]] = r
        data[tag] = d
    def agg(tag):
        rs = list(data.get(tag, {}).values())
        if not rs: return None
        n = len(rs)
        return {"n": n, "garb": sum(r["garbled"] for r in rs), "hit": sum(r.get("kw_hit") for r in rs),
                "tok_s": sum(r.get("tok_s", 0) for r in rs) / n,
                "ttft": sum((r.get("ttft") or 0) for r in rs) / n}
    A = {t: agg(t) for t in tags}
    LABEL = {"dinfer": "dinfer (eager)", "vllm_eager": "vLLM (eager)", "vllm_compile": "vLLM (compile)"}
    lab = lambda t: LABEL.get(t, t)
    rows_html = []
    for p in prompts:
        cells = [f"<td class=cat>{esc(p['cat'])}</td><td class=q>{esc(p['q'])}<br><span class=perf>max={p.get('max_tokens',512)}</span></td>"]
        for tag in tags:
            r = data.get(tag, {}).get(p["id"])
            if not r:
                cells.append("<td class=na>—</td>"); continue
            flags = []
            if r["garbled"]: flags.append(f"<span class=bad>乱码:{esc(r['garble_why'])}</span>")
            if not r.get("kw_hit"): flags.append("<span class=warn>未命中</span>")
            perf = f"<span class=perf>{r.get('tok_s',0):.1f} tok/s · TTFT {('%.0fms'%(r['ttft']*1000)) if r.get('ttft') else '—'} · {r.get('ntok',0)}tok</span>"
            cells.append(f"<td><div class=txt>{esc(r['text'][:500])}</div>{' '.join(flags)} {perf}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    def sumrow(tag):
        a = A.get(tag)
        if not a: return f"{lab(tag)}: 无数据"
        return (f"{lab(tag)}: {a['n']} 条 · 乱码 {a['garb']} · 命中 {a['hit']}/{a['n']} "
                f"· 均 {a['tok_s']:.1f} tok/s · 均 TTFT {a['ttft']*1000:.0f}ms")
    thead = "<th>类</th><th>问题</th>" + "".join(f"<th>{esc(lab(t))}</th>" for t in tags)
    sumhtml = "<br>".join(esc(sumrow(t)) for t in tags)
    doc = f"""<!doctype html><html lang=zh><head><meta charset=utf-8>
<title>Kaon-V3 生成对比</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{{--bg:#fff;--fg:#1a1a1a;--line:#e0e0e0;--bad:#c0392b;--warn:#b8860b;--perf:#2563eb;--muted:#777}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#16181c;--fg:#e6e6e6;--line:#2c2f36;--bad:#ff6b6b;--warn:#e0b050;--perf:#6aa8ff;--muted:#999}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.6 system-ui,sans-serif;margin:0;padding:16px}}
h1{{font-size:20px}} .sum{{margin:8px 0;padding:10px;border:1px solid var(--line);border-radius:8px}}
table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid var(--line);padding:8px;vertical-align:top;text-align:left}}
th{{position:sticky;top:0;background:var(--bg)}} .cat{{white-space:nowrap;color:var(--muted)}} .q{{max-width:200px}}
.txt{{white-space:pre-wrap;max-width:380px;font-size:13px}} .bad{{color:var(--bad);font-weight:600}} .warn{{color:var(--warn)}}
.perf{{color:var(--perf);font-size:12px}} .na{{color:var(--muted)}}
</style></head><body>
<h1>Kaon-V3 端到端生成对比 · dinfer vs vLLM</h1>
<div class=sum><b>汇总</b>（temp=0 贪心；dinfer/vLLM-eager 对齐：均无 cudagraph/compile + warmup 一次）<br>{sumhtml}</div>
<table><thead><tr>{thead}</tr></thead>
<tbody>{''.join(rows_html)}</tbody></table>
<p style="color:var(--muted);font-size:12px">乱码=机器判定(异常字符/重复/空)；未命中=expect 关键词不在输出(答非所问初筛,命中不等于答对但不命中强提示跑偏)；性能=该条 tok/s / TTFT / 输出 token 数。</p>
</body></html>"""
    out = os.path.join(ROOT, "report.html")
    open(out, "w", encoding="utf-8").write(doc)
    print(f"写 {out}")
    for t in tags: print(sumrow(t))


def dump():
    """把 prompt + 各模式完整 output 导成可读 markdown（带回给人看）。"""
    tags = _all_tags()
    LABEL = {"dinfer": "dinfer (eager)", "vllm_eager": "vLLM (eager)", "vllm_compile": "vLLM (compile)"}
    data = {}
    for tag in tags:
        fp = os.path.join(ROOT, f"result_{tag}.jsonl")
        data[tag] = {json.loads(l)["id"]: json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()}
    lines = ["# Kaon-V3 生成对比（prompt + 完整输出）", f"\n模式：{', '.join(LABEL.get(t,t) for t in tags)}\n"]
    for p in load_prompts():
        lines.append(f"\n## [{p['cat']}] {p['id']}  (max_tokens={p.get('max_tokens',512)})\n")
        lines.append(f"**Prompt：**\n\n```\n{p['q']}\n```\n")
        for tag in tags:
            r = data.get(tag, {}).get(p["id"])
            if not r:
                lines.append(f"**{LABEL.get(tag,tag)}：** (无数据)\n"); continue
            flags = []
            if r.get("garbled"): flags.append(f"⚠乱码:{r.get('garble_why')}")
            flags.append(f"{r.get('tok_s',0):.1f} tok/s")
            if r.get("ttft") is not None: flags.append(f"TTFT {r['ttft']*1000:.0f}ms")
            flags.append(f"{r.get('ntok',0)} tok")
            lines.append(f"**{LABEL.get(tag,tag)}** ({' · '.join(flags)})：\n\n```\n{r.get('text','')}\n```\n")
    out = os.path.join(ROOT, "dump.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines))
    print(f"写 {out}")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    def arg(k, d=None):
        return sys.argv[sys.argv.index(k)+1] if k in sys.argv else d
    if cmd == "run":
        run(arg("--tag"), arg("--url"), arg("--model"))
    elif cmd == "dump":
        dump()
    else:
        report()
