#!/usr/bin/env bash
# D178b：量 MoE 每层各段 GPU 时间（KAON3_MOE_TIME=1：每层 event + 同步，eager；只看 gemm3 段的相对值）。
# 跑 c8×24 bench，解析日志 "moe-time" 按 tokens 分组给 active_experts 的 p50/mean/max。
# 用法：BIN=... bash /w/expstats.sh（docker exec -d 时输出必须重定向）
NV=/usr/local/lib/python3.12/dist-packages/nvidia
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
TAG=${TAG:-moetime}; PORT=${PORT:-8398}; BIN=${BIN:-./target/release/dinfer}
R=/w/dinfer/tmp_${TAG}_RESULT.txt; : > $R; L=/w/dinfer/tmp_${TAG}_serve.log
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
echo "bin=$(stat -c %y $BIN | cut -c1-16) md5=$(md5sum $BIN | cut -c1-12) marker=$(strings $BIN | grep -c MOE_EXPERT_STATS) $(date +%T)" >> $R
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=0
taskset -c ${SERVE_TASKSET:-0-31} env KAON3_MOE_TIME=1 $BIN serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 >$L 2>&1 &
sp=$!
for i in $(seq 1 150); do sed "s/\x1b\[[0-9;]*m//g" $L | grep -q "listening addr" && break; sleep 1; done
sed "s/\x1b\[[0-9;]*m//g" $L | grep -q "listening addr" || { echo "NOTUP" >> $R; kill -KILL $sp 2>/dev/null; echo DONE >> $R; exit 1; }
for c in 1 8; do
  echo -n "c$c: " >> $R
  taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:$PORT --concurrency $c --num-requests $((c*3)) --prompt-len 400 --max-tokens 64 --prefix-repeat 0 --warmup 1 --seed 11 2>&1 | grep -E "TPOT " | tr '\n' ' ' >> $R
  echo >> $R
done
kill -INT $sp 2>/dev/null; sleep 2; kill -KILL $sp 2>/dev/null; sleep 2
sed "s/\x1b\[[0-9;]*m//g" $L | grep -E "moe-time" > /w/dinfer/tmp_${TAG}_stats.txt
python3 -I - /w/dinfer/tmp_${TAG}_stats.txt >> $R <<'PY'
import sys,re,collections
g=collections.defaultdict(list)
for l in open(sys.argv[1]):
    m=re.search(r'r=(\d+) router=([\d.]+) sm\+bucket=([\d.]+) norm\+gather\+quant=([\d.]+) \| gemm3=([\d.]+) scatter=([\d.]+)',l)
    if m: g[int(m.group(1))].append(tuple(float(x) for x in m.groups()[1:]))
def q(v,p): v=sorted(v); return v[min(len(v)-1,int(p*len(v)))]
print("moe-time by r (rows=tokens*8): n | router | sm+bucket | norm+gather+quant | gemm3 p50/p90 | scatter  (ms, per layer, eager+sync)")
for r in sorted(g):
    v=g[r]; cols=list(zip(*v))
    print(f"  r={r}: {len(v)} | {q(cols[0],.5):.3f} | {q(cols[1],.5):.3f} | {q(cols[2],.5):.3f} | {q(cols[3],.5):.3f}/{q(cols[3],.9):.3f} | {q(cols[4],.5):.3f}")
PY
echo DONE >> $R
