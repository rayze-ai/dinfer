#!/usr/bin/env python3
"""nsys sqlite → 非 graph（eager）prefill / 混批步的 kernel 清单（loop kaon3-decode-opt D201）。
  python3 -I nsys_prefill_steps.py <sqlite> [min_tokens_hint_kernels=200] [top=25]
做法：取 graphId==0（eager）的 kernel，按时间间隙 > 2 ms 切成"步"；步内 kernel 数 ≥ 阈值的当 prefill 步；
对最长的 3 个步给 span / kernel 和 / 空隙，再按 kernel 名聚合（次数、总时长、均值、占比）与按家族汇总。"""
import sqlite3, sys, collections, re
db = sys.argv[1]; thr = int(sys.argv[2]) if len(sys.argv) > 2 else 200; top = int(sys.argv[3]) if len(sys.argv) > 3 else 25; GAP = int(float(sys.argv[4]) * 1e6) if len(sys.argv) > 4 else 2_000_000
c = sqlite3.connect(db)
K = "CUPTI_ACTIVITY_KIND_KERNEL"
cols = {r[1] for r in c.execute(f"pragma table_info({K})")}
gcol = "graphId" if "graphId" in cols else None
q = f"select k.start, k.end, s.value, k.streamId{', k.'+gcol if gcol else ''} from {K} k join StringIds s on k.shortName = s.id order by k.start"
rows = c.execute(q).fetchall()
# --stream <id>：只看该流（E6 后 prefill 在副流；先用 --streams 列出各流 kernel 数）
if "--streams" in sys.argv:
    import collections as _c
    cnt = _c.Counter(r[3] for r in rows); print("streams (id: kernels):", dict(cnt.most_common(8))); sys.exit(0)
want = int(sys.argv[sys.argv.index("--stream") + 1]) if "--stream" in sys.argv else None
rows = [r for r in rows if want is None or r[3] == want]
eager = [(a, b, n) for a, b, n, _sid, *g in rows if not gcol or (g and g[0] in (0, None))]
# 稳态：最后一次大 H2D 之后（权重加载）
try:
    big = [r[0] for r in c.execute("select end from CUPTI_ACTIVITY_KIND_MEMCPY where bytes >= 8000000 and copyKind=1").fetchall()]
    t0 = max(big) if big else 0
except Exception:
    t0 = 0
eager = [e for e in eager if e[0] > t0]
steps, cur = [], []
for e in eager:
    if cur and e[0] - cur[-1][1] > GAP:
        steps.append(cur); cur = []
    cur.append(e)
if cur: steps.append(cur)
pre = [s for s in steps if len(s) >= thr]
print(f"eager kernels={len(eager)} eager-steps={len(steps)} prefill-like(>= {thr} kernels)={len(pre)}")
FAM = [("attn", ("gqa", "flash", "fa_", "paged", "decode_win", "splitk_attn")), ("moe_gemm", ("grouped_gemm", "pipe4", "moe_gemm", "expert")),
       ("moe_route", ("route",)), ("moe_misc", ("gather", "scatter", "combine", "bucket", "sort", "scan", "perm")),
       ("gemm", ("nvjet", "cublas", "gemm", "gemv", "mma", "Lt")), ("norm", ("norm", "rms")), ("quant", ("quant", "fp8", "fp4", "cvt")),
       ("rope", ("rope",)), ("kv", ("cache_write", "kv_")), ("elem", ("elementwise", "add", "mul", "gelu", "silu", "copy", "fill", "memset"))]
def fam(n):
    for f, keys in FAM:
        if any(k.lower() in n.lower() for k in keys): return f
    return "other"
for s in sorted(pre, key=lambda s: -(s[-1][1] - s[0][0]))[:3]:
    span = (s[-1][1] - s[0][0]) / 1e3; ksum = sum(b - a for a, b, _ in s) / 1e3
    print(f"\n== step kernels={len(s)} span={span/1e3:.2f} ms kernel_sum={ksum/1e3:.2f} ms gaps={(span-ksum)/1e3:.2f} ms")
    agg = collections.defaultdict(lambda: [0, 0.0])
    for a, b, n in s:
        agg[n][0] += 1; agg[n][1] += (b - a) / 1e3
    famagg = collections.defaultdict(float)
    for n, (cnt, us) in agg.items(): famagg[fam(n)] += us
    print("  family: " + " ".join(f"{f}={us/1e3:.2f}ms({100*us/ksum:.0f}%)" for f, us in sorted(famagg.items(), key=lambda x: -x[1])))
    for n, (cnt, us) in sorted(agg.items(), key=lambda x: -x[1][1])[:top]:
        print(f"  {us/1e3:7.2f} ms {100*us/ksum:5.1f}%  n={cnt:4d}  {us/cnt:8.1f} us/call  {n[:110]}")
