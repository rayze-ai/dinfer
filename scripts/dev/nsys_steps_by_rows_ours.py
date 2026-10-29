#!/usr/bin/env python3
# loop kaon3-prefill-gemm G8：我们副流(stream 15) eager 步按 nvjet 行数分桶，每桶每步 MoE/fp8/attn/其它 ms（与 vLLM 版同口径）。用法：python3 -I <this> <ours.sqlite>
import sqlite3,sys,collections,statistics as st
c=sqlite3.connect(sys.argv[1]); K="CUPTI_ACTIVITY_KIND_KERNEL"
rows=c.execute(f"select k.start,k.end,s.value,d.value,k.gridX,k.gridY,k.gridZ,k.graphId from {K} k join StringIds s on k.shortName=s.id join StringIds d on k.demangledName=d.id where k.streamId=15 order by k.start").fetchall()
eag=[r for r in rows if r[7] in (0,None)]
steps=[];cur=[]
for r in eag:
    if cur and r[0]-cur[-1][1]>1_500_000: steps.append(cur);cur=[]
    cur.append(r)
if cur: steps.append(cur)
def rowsof(s):
    g=[r[5] for r in s if "nvjet" in r[2] and "128x128" in r[2]]
    return max(g)*128 if g else 0
def fam(sn,dn):
    if "p64" in sn or "grouped_gemm" in sn: return "moe"
    if "nvjet" in sn or "xmma" in sn: return "fp8"
    if "gqa" in sn: return "attn"
    if "Kernel2" in sn or "small_m" in sn: return "bf16"
    return "other"
byb=collections.defaultdict(list)
for s in steps:
    r=rowsof(s)
    if r==0: continue
    b = 128 if r<=128 else 256 if r<=256 else 512 if r<=512 else 1024 if r<=1024 else 2048 if r<=2048 else 4096
    byb[b].append(s)
print("OURS rows bucket: n_steps | kernels/step | span | kernel_sum | moe ms (n, us) | fp8 (n) | attn (n, us) | bf16 | other | moe grids")
for b in sorted(byb):
    ss=byb[b]; n=len(ss)
    f=collections.defaultdict(lambda:[0,0.0]); grids=collections.Counter()
    for s in ss:
        for a,e,sn,dn,gx,gy,gz,_ in s:
            k=fam(sn,dn); f[k][0]+=1; f[k][1]+=(e-a)/1e6
            if k=="moe": grids[(gx,gy,gz)]+=1
    span=st.median((s[-1][1]-s[0][0])/1e6 for s in ss); ks=st.median(sum(e-a for a,e,*_ in s)/1e6 for s in ss)
    m=f["moe"]; p=f["fp8"]; at=f["attn"]
    print(f"  {b:5d}: {n:4d} | {st.median(len(s) for s in ss):4.0f} | {span:6.1f} | {ks:6.1f} | {m[1]/n:6.2f} ({m[0]/n:4.1f}, {1e3*m[1]/max(m[0],1):5.0f}us) | {p[1]/n:5.2f} ({p[0]/n:4.1f}) | {at[1]/n:5.2f} ({at[0]/n:3.1f}, {1e3*at[1]/max(at[0],1):4.0f}us) | {f['bf16'][1]/n:4.2f} | {f['other'][1]/n:4.2f} | {grids.most_common(2)}")
