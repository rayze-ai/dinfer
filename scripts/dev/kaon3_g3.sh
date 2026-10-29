#!/usr/bin/env bash
# g3（G3a 机理核）：nsysp16 口径抓两次 c16（A 默认 / B FP8_LT_PREFER_SMALL_TILE=1），比副流（prefill）上 nvjet 按 grid 的实测时长。
# 结果 tmp_g3_RESULT.txt，标记 G3_DONE。
R=/w/dinfer/tmp_g3_RESULT.txt; : > $R; date +%T >> $R
for side in A B; do
  if [ $side = A ]; then EX=""; else EX="FP8_LT_PREFER_SMALL_TILE=1"; fi
  EXTRA_ENV="$EX GEMM_FP8_DEBUG=1" TAG=nsysg3$side PORT=$((8368 + ([ $side = B ] && echo 1 || echo 0))) bash /w/nsysp16.sh
  echo "== $side ($EX)" >> $R
  grep -E "throughput|host-time:" /w/dinfer/tmp_nsysg3${side}_RESULT.txt | sed -E "s/client:.*//" | cut -c1-160 >> $R
  taskset -c 100-127 python3 -I - /w/dinfer/tmp_nsysg3${side}.sqlite <<'PY' >> $R 2>&1
import sqlite3,sys,collections
c=sqlite3.connect(sys.argv[1]); K="CUPTI_ACTIVITY_KIND_KERNEL"
rows=c.execute(f"select k.start,k.end,d.value,k.gridX,k.gridY,k.gridZ,k.streamId from {K} k join StringIds d on k.demangledName=d.id order by k.start").fetchall()
big=[r[0] for r in c.execute("select end from CUPTI_ACTIVITY_KIND_MEMCPY where bytes>=8000000 and copyKind=1")]
t0=max(big) if big else 0
rows=[r for r in rows if r[0]>t0]
sid=collections.Counter(r[6] for r in rows if "nvjet" in r[2]).most_common(1)[0][0]
print("alt stream id (most nvjet):",sid)
agg=collections.defaultdict(lambda:[0,0.0])
tot=0.0
for a,b,n,gx,gy,gz,s in rows:
    if s!=sid: continue
    us=(b-a)/1e3
    fam="nvjet" if "nvjet" in n else ("p64" if "p64" in n else ("fa" if "gqa_fa" in n else "other"))
    key=(fam, n[:34] if fam=="nvjet" else fam, gx, gy) if fam=="nvjet" else (fam,fam,0,0)
    agg[key][0]+=1; agg[key][1]+=us; tot+=us
print(f"alt-stream kernel total {tot/1e3:.1f} ms")
for (fam,n,gx,gy),(cnt,us) in sorted(agg.items(),key=lambda x:-x[1][1])[:12]:
    print(f"  {us/1e3:7.1f}ms {100*us/tot:5.1f}% n={cnt:5d} {us/cnt:7.1f}us grid=({gx},{gy}) {n}")
PY
done
echo G3_DONE >> $R
