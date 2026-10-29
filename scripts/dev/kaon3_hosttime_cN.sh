#!/usr/bin/env bash
# D170：c8 每步 host 分段 + 步间隔定性。服务开 HOST_TIME=1，跑 bN 同口径 c1/c8 bench，解析 [host-time] 行按 (tokens,nreq) 分组给 p50/p90，
# 并用日志时间戳算 nreq=8 decode 步的步间隔（cadence）。用法：BIN=... TAG=hosttime bash /w/hosttime.sh（docker exec -d 时输出必须重定向）
NV=/usr/local/lib/python3.12/dist-packages/nvidia
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
TAG=${TAG:-hosttime_cN}; PORT=${PORT:-8397}; BIN=${BIN:-./target/release/dinfer}
R=/w/dinfer/tmp_${TAG}_RESULT.txt; : > $R; L=/w/dinfer/tmp_${TAG}_serve.log
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
echo "bin=$(stat -c %y $BIN | cut -c1-16) env=[$AENV] $(date +%T)" >> $R
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=0
taskset -c ${SERVE_TASKSET:-0-31} env HOST_TIME=1 $AENV $BIN serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 >$L 2>&1 &
sp=$!
for i in $(seq 1 150); do sed "s/\x1b\[[0-9;]*m//g" $L | grep -q "listening addr" && break; sleep 1; done
sed "s/\x1b\[[0-9;]*m//g" $L | grep -q "listening addr" || { echo "NOTUP" >> $R; kill -KILL $sp 2>/dev/null; echo DONE >> $R; exit 1; }
for c in 16 32 64; do
  echo "== c$c mark $(date +%T.%N | cut -c1-12)" >> $R
  echo -n "c$c: " >> $R
  taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:$PORT --concurrency $c --num-requests $((c*8)) --prompt-len 400 --max-tokens 64 --prefix-repeat 0 --warmup 2 --seed 7 2>&1 | grep -E "throughput|TTFT |TPOT " | tr '\n' ' ' >> $R
  echo >> $R
done
curl -s -m10 http://127.0.0.1:$PORT/metrics | grep -E "step_time_(bucket|sum|count)" | tr '\n' ' ' | cut -c1-600 >> $R; echo >> $R
kill -INT $sp 2>/dev/null; sleep 2; kill -KILL $sp 2>/dev/null; sleep 2
sed "s/\x1b\[[0-9;]*m//g" $L | grep -E "host-time" > /w/dinfer/tmp_${TAG}_ht.txt
python3 -I - /w/dinfer/tmp_${TAG}_ht.txt >> $R <<'PY'
import sys,re,datetime,statistics as st
rows=[]
for l in open(sys.argv[1]):
    m=re.search(r'T(\d\d:\d\d:\d\d\.\d{6})',l); f=dict(re.findall(r'(\w+(?:\(begin\+finish\))?)=([\d.]+)',l))
    if not m or 'total' not in f: continue
    t=datetime.datetime.strptime(m.group(1),'%H:%M:%S.%f')
    rows.append((t,int(f['tokens']),int(f['nreq']),float(f['sched']),float(f.get('exec(begin+finish)',f.get('exec',0))),float(f['publish']),float(f['total'])))
def q(v,p): v=sorted(v); return v[min(len(v)-1,int(p*len(v)))]
groups={}
for r in rows: groups.setdefault((r[1],r[2]),[]).append(r)
print("host-time by (tokens,nreq): n | total p50/p90 | sched p50 | exec p50/p90 | publish p50/p90 (us)")
for k in sorted(groups,key=lambda k:-len(groups[k]))[:8]:
    g=groups[k]; tot=[r[6] for r in g]; sc=[r[3] for r in g]; ex=[r[4] for r in g]; pb=[r[5] for r in g]
    print(f"  {k}: {len(g)} | {q(tot,.5):.0f}/{q(tot,.9):.0f} | {q(sc,.5):.0f} | {q(ex,.5):.0f}/{q(ex,.9):.0f} | {q(pb,.5):.0f}/{q(pb,.9):.0f}")
# cadence: 连续两步同 (tokens,nreq) 的时间戳差
for key in [(16,16),(32,32),(64,64)]:
    d=[]
    for a,b in zip(rows,rows[1:]):
        if (a[1],a[2])==key and (b[1],b[2])==key: d.append((b[0]-a[0]).total_seconds()*1e3)
    if d: print(f"cadence {key}: n={len(d)} p50={q(d,.5):.2f} p90={q(d,.9):.2f} mean={st.mean(d):.2f} ms  max={max(d):.1f}")
PY
echo DONE >> $R
