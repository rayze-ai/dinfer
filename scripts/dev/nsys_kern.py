#!/usr/bin/env python3
"""nsys sqlite → decode 步 kernel 清单。
  python3 nsys_kern.py <sqlite>
找 graph 启动（graphId>0 的 kernel 按 graphExecId / 时间聚类），取"最常见 kernel 数"的那类图实例当 decode 步；
对一个稳态实例按 kernel 名聚合（次数、总时长、占比），再给整步 span / kernel 和 / 空隙。
"""
import sqlite3, sys, collections, re

c = sqlite3.connect(sys.argv[1])
tabs = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
K = "CUPTI_ACTIVITY_KIND_KERNEL"
if K not in tabs:
    print("no kernel table; tables:", sorted(t for t in tabs if "CUPTI" in t)); sys.exit(1)
cols = [r[1] for r in c.execute(f"pragma table_info({K})")]
print("cols:", [x for x in cols if re.search("graph|Name|grid|block|start|end|stream", x, re.I)])
name_col = "demangledName" if "demangledName" in cols else "shortName"
strs = dict(c.execute("select id, value from StringIds"))
gcol = next((x for x in ("graphExecId", "graphId") if x in cols), None)
rows = list(c.execute(f"select start, end, {name_col}, {'gridX,gridY,gridZ,blockX' if 'gridX' in cols else '0,0,0,0'}, {gcol if gcol else 0} from {K} order by start"))
print("kernels total:", len(rows), "graph col:", gcol)

# 分组：graph 内 kernel 按 gcol 值；gcol 为 0 的（eager/prefill）按时间间隙 >200us 切段
groups = collections.OrderedDict()
seg, last_end, gid = 0, None, None
for st, en, nm, gx, gy, gz, bx, g in rows:
    if g and g > 0:
        key = ("g", g)
    else:
        if last_end is None or st - last_end > 200_000:
            seg += 1
        key = ("e", seg)
    groups.setdefault(key, []).append((st, en, strs.get(nm, str(nm)), gx, gy, gz, bx))
    last_end = en
sizes = collections.Counter(len(v) for k, v in groups.items() if k[0] == "g")
print("graph instances by kernel-count:", sizes.most_common(6))
esizes = collections.Counter(len(v) for k, v in groups.items() if k[0] == "e")
print("eager segments by kernel-count:", esizes.most_common(6))

def report(inst, title):
    span = (inst[-1][1] - inst[0][0]) / 1e3
    busy = sum(en - st for st, en, *_ in inst) / 1e3
    print(f"\n== {title}: kernels={len(inst)} span={span:.0f}us kernel_sum={busy:.0f}us gaps={span-busy:.0f}us")
    agg = collections.defaultdict(lambda: [0, 0.0, None])
    for st, en, nm, gx, gy, gz, bx in inst:
        a = agg[nm]; a[0] += 1; a[1] += (en - st) / 1e3; a[2] = a[2] or (gx, gy, gz, bx)
    tot = sum(v[1] for v in agg.values())
    for nm, (n, t, grid) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:40]:
        print(f"{t:8.0f}us {t/tot*100:5.1f}% n={n:4d} avg={t/n*1e3:6.1f}us grid={grid} {nm[:90]}")

if sizes:
    want = sizes.most_common(1)[0][0]
    insts = [v for k, v in groups.items() if k[0] == "g" and len(v) == want]
    report(insts[len(insts) // 2], f"decode graph instance (kernel-count={want}, {len(insts)} instances, middle one)")
if esizes:
    big = max((v for k, v in groups.items() if k[0] == "e"), key=len)
    report(big, "largest eager segment (prefill?)")
