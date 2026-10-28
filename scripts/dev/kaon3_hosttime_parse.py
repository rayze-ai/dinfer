#!/usr/bin/env python3
"""解析 dinfer serve 日志（HOST_TIME=1）的一个窗口 [lo, hi) 行，按 prefill / decode 分类给各段均值（us）。
  python3 -I kaon3_hosttime_parse.py <log> <lo> <hi>
输出一行：
  render=<chat模板+tokenize us> | prefill: n=.. sched exec publish | upd build fwd | upload launch collect | decode: 同
行格式（见 crates/core/engine.rs / runner.rs、models/kaon3/executor.rs）：
  [render-time] render(模板+tokenize)=123.4us prompt_ids=1560
  [host-time] tokens=T nreq=N sched=X exec(begin+finish)=Y publish=Z total=W us
  [begin-split] tokens=T async=false update_states=Xus build_plan=Yus forward_sample=Zus
  [fwd-sample] ENTER mode=Decode t=T n=N nrows=R        ← 给紧随其后的 [kaon3-host] 定类
  [kaon3-host] upload=Xus launch(submit)=Yus collect(dtoh_sync)=Zus
"""
import re, sys
from collections import defaultdict

log, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ansi = re.compile(r"\x1b\[[0-9;]*m")
num = lambda s, k: float(re.search(k + r"=([0-9.]+)", s).group(1))

acc = defaultdict(lambda: defaultdict(list))  # cls -> field -> [vals]
render = []
cur_cls = "decode"
with open(log, errors="replace") as f:
    for i, line in enumerate(f):
        if i < lo:
            continue
        if i >= hi:
            break
        s = ansi.sub("", line)
        if "[render-time]" in s:
            render.append(num(s, r"tokenize\)"))
        elif "[fwd-sample] ENTER" in s:
            t, n = num(s, " t"), num(s, " n")
            cur_cls = "decode" if t == n else "prefill"
        elif "[kaon3-host]" in s:
            a = acc[cur_cls]
            a["upload"].append(num(s, "upload"))
            a["launch"].append(num(s, r"launch\(submit\)"))
            a["collect"].append(num(s, r"collect\(dtoh_sync\)"))
        elif "[begin-split]" in s:
            t = num(s, "tokens")
            a = acc[cur_cls]
            a["upd"].append(num(s, "update_states"))
            a["build"].append(num(s, "build_plan"))
            a["fwd"].append(num(s, "forward_sample"))
        elif "[host-time]" in s:
            t, n = num(s, "tokens"), num(s, "nreq")
            a = acc["decode" if t == n else "prefill"]
            a["sched"].append(num(s, "sched"))
            a["exec"].append(num(s, r"exec\(begin\+finish\)"))
            a["publish"].append(num(s, "publish"))
            a["total"].append(num(s, "total"))
            a["toks"].append(t)
            a["nreq"].append(n)

def mean(v):
    return sum(v) / len(v) if v else float("nan")

out = [f"render={mean(render):.0f}us(n={len(render)})"]
for cls in ("prefill", "decode"):
    a = acc[cls]
    if not a["exec"]:
        out.append(f"{cls}: none")
        continue
    out.append(
        f"{cls}: steps={len(a['exec'])} toks/step={mean(a['toks']):.0f} nreq={mean(a['nreq']):.1f} "
        f"sched={mean(a['sched']):.0f} exec={mean(a['exec']):.0f} publish={mean(a['publish']):.0f} total={mean(a['total']):.0f} | "
        f"upd={mean(a['upd']):.0f} build={mean(a['build']):.0f} fwd={mean(a['fwd']):.0f} | "
        f"htod={mean(a['upload']):.0f} launch={mean(a['launch']):.0f} wait={mean(a['collect']):.0f}"
    )
print(" || ".join(out))
