#!/usr/bin/env python3
"""nsys sqlite → 按 graphId 组的"每步"统计 + kernel 时长直方图 + 间隙直方图（只读 sqlite）。
  python3 nsys_hist.py <sqlite> [graphId ...]   # 不给 id 则对每个 graph 组都报
步数 = route_bucket 类 kernel 次数 / 30（kaon3 30 层每层 1 次）；找不到则按 lm_head gemvx 次数。
"""
import sqlite3, sys, collections, re
c = sqlite3.connect(sys.argv[1])
K = "CUPTI_ACTIVITY_KIND_KERNEL"
cols = [r[1] for r in c.execute(f"pragma table_info({K})")]
name_col = "demangledName" if "demangledName" in cols else "shortName"
strs = dict(c.execute("select id, value from StringIds"))
gcol = next((x for x in ("graphExecId", "graphId") if x in cols), None)
rows = list(c.execute(f"select start, end, {name_col}, gridX,gridY,gridZ, {gcol} from {K} order by start"))
groups = collections.OrderedDict()
for st, en, nm, gx, gy, gz, g in rows:
    if g and g > 0:
        groups.setdefault(g, []).append((st, en, strs.get(nm, str(nm)), gx*gy*gz))
want = [int(x) for x in sys.argv[2:] if x.isdigit()]
BINS = [(0,3,"<3us"),(3,8,"3-8us"),(8,20,"8-20us"),(20,1e9,">20us")]
GBINS = [(0,1,"<1us"),(1,2,"1-2"),(2,5,"2-5"),(5,20,"5-20"),(20,200,"20-200"),(200,1e12,">200(步间)")]
for g, inst in groups.items():
    if want and g not in want: continue
    if len(inst) < 500: continue
    names = collections.Counter(nm for *_, nm, _ in inst)
    nroute = sum(n for nm, n in names.items() if "route_bucket" in nm)
    nlm = sum(n for nm, n in names.items() if "gemvx" in nm or "lm_head" in nm)
    steps = nroute // 30 if nroute >= 30 else max(1, nlm)
    span = (inst[-1][1] - inst[0][0]) / 1e3
    busy = sum(en - st for st, en, *_ in inst) / 1e3
    gaps = [max(0, inst[i][0] - inst[i-1][1]) / 1e3 for i in range(1, len(inst))]
    inter = sum(x for x in gaps if x > 200); intra = sum(x for x in gaps if x <= 200)
    print(f"\n== graph id={g} kernels={len(inst)} steps={steps} (route n={nroute}, lm n={nlm})")
    print(f"每步: kernels={len(inst)/steps:.0f} kernel_sum={busy/steps:.0f}us span={span/steps:.0f}us "
          f"gaps_all={(span-busy)/steps:.0f}us  gaps_步内(<=200us)={intra/steps:.0f}us 步间(>200us)={inter/steps:.0f}us")
    # kernel 时长直方图
    print("kernel 时长分布（每步）:")
    tot = busy
    for lo, hi, lab in BINS:
        sel = [(en-st)/1e3 for st, en, *_ in inst if lo <= (en-st)/1e3 < hi]
        print(f"  {lab:7s} n/步={len(sel)/steps:6.1f}  时长/步={sum(sel)/steps:7.0f}us  占 kernel 和 {sum(sel)/tot*100:5.1f}%")
    print("步内间隙分布:")
    for lo, hi, lab in GBINS:
        sel = [x for x in gaps if lo <= x < hi]
        print(f"  {lab:10s} n/步={len(sel)/steps:6.1f}  合计/步={sum(sel)/steps:7.0f}us")
    # 每类 kernel：n/步、avg、归档到哪个 bin
    agg = collections.defaultdict(lambda: [0, 0.0, 0])
    for st, en, nm, grid in inst:
        a = agg[nm]; a[0] += 1; a[1] += (en-st)/1e3; a[2] = grid
    print("按 kernel 名（n/步, avg us, 每步 us, grid）—— 全部，按 avg 升序:")
    for nm, (n, t, grid) in sorted(agg.items(), key=lambda kv: kv[1][1]/kv[1][0]):
        print(f"  {n/steps:6.1f}/步 avg={t/n:6.1f}us 每步={t/steps:6.0f}us grid={grid:6d} {nm[:70]}")
