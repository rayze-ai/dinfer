#!/usr/bin/env bash
# b54（D217）：四刀（B46/B48/B48b/B49）后重抓 c16 混批步 kernel 表（nsysp16 同口径，TAG=nsysp16b），看新的大头。
# 容器内：docker exec -d kaon3-full bash -c 'bash /w/b54_nsys.sh > /w/dinfer/tmp_b54_run.log 2>&1'；结果 tmp_nsysp16b_RESULT.txt，标记 NSYSP16_DONE。
TAG=nsysp16b PORT=8349 bash /w/nsysp16.sh
# 追加：按 grid 聚合的一步（~958 kernel 片段）top 列表
NSYS_SQL=/w/dinfer/tmp_nsysp16b.sqlite
taskset -c 100-127 python3 -I - $NSYS_SQL <<'PY' >> /w/dinfer/tmp_nsysp16b_RESULT.txt 2>&1
import sqlite3,sys,collections
c=sqlite3.connect(sys.argv[1]); K="CUPTI_ACTIVITY_KIND_KERNEL"
rows=c.execute(f"select k.start,k.end,s.value,d.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.graphId from {K} k join StringIds s on k.shortName=s.id join StringIds d on k.demangledName=d.id order by k.start").fetchall()
big=[r[0] for r in c.execute("select end from CUPTI_ACTIVITY_KIND_MEMCPY where bytes>=8000000 and copyKind=1")]
t0=max(big) if big else 0
eager=[r for r in rows if r[0]>t0 and (r[8] in (0,None))]
steps=[];cur=[]
for e in eager:
    if cur and e[0]-cur[-1][1]>2_000_000: steps.append(cur);cur=[]
    cur.append(e)
if cur: steps.append(cur)
st=[s for s in steps if 800<=len(s)<=1200]
print("GRID-AGG steps(800-1200 kernels):",len(st))
if st:
    s=st[len(st)//2]; span=(s[-1][1]-s[0][0])/1e6; ks=sum(b-a for a,b,*_ in s)/1e6
    print(f"step kernels={len(s)} span={span:.2f}ms kernel_sum={ks:.2f}ms")
    agg=collections.defaultdict(lambda:[0,0.0,None])
    for a,b,sn,dn,gx,gy,gz,bx,_ in s:
        key=(sn,gx,gy,gz,bx); agg[key][0]+=1; agg[key][1]+=(b-a)/1e3; agg[key][2]=dn
    for (sn,gx,gy,gz,bx),(n,us,dn) in sorted(agg.items(),key=lambda x:-x[1][1])[:16]:
        print(f"{us/1e3:7.2f}ms n={n:3d} {us/n:7.1f}us grid=({gx},{gy},{gz}) blk={bx}  {dn[:120]}")
PY
