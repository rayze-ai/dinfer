#!/usr/bin/env bash
# D178：数 M=8 decode 步每层激活的专家数。**必须 NO_GRAPH=1**：graph 回放不执行 Rust 层代码，诊断只在 eager 步生效（b40 第一次只抓到启动期 4 条）。服务开 MOE_EXPERT_STATS=1（每 200 次 MoE 调用 D2H 一次，会拖慢，数字不做 bench），
# 跑 c8×24 bench，解析日志 "moe expert stats" 按 tokens 分组给 active_experts 的 p50/mean/max。
# 用法：BIN=... bash /w/expstats.sh（docker exec -d 时输出必须重定向）
NV=/usr/local/lib/python3.12/dist-packages/nvidia
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
TAG=${TAG:-expstats}; PORT=${PORT:-8398}; BIN=${BIN:-./target/release/dinfer}
R=/w/dinfer/tmp_${TAG}_RESULT.txt; : > $R; L=/w/dinfer/tmp_${TAG}_serve.log
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
echo "bin=$(stat -c %y $BIN | cut -c1-16) md5=$(md5sum $BIN | cut -c1-12) marker=$(strings $BIN | grep -c MOE_EXPERT_STATS) $(date +%T)" >> $R
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=0
taskset -c ${SERVE_TASKSET:-0-31} env NO_GRAPH=1 MOE_EXPERT_STATS=1 $BIN serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 >$L 2>&1 &
sp=$!
for i in $(seq 1 150); do sed "s/\x1b\[[0-9;]*m//g" $L | grep -q "listening addr" && break; sleep 1; done
sed "s/\x1b\[[0-9;]*m//g" $L | grep -q "listening addr" || { echo "NOTUP" >> $R; kill -KILL $sp 2>/dev/null; echo DONE >> $R; exit 1; }
for c in 1 8; do
  echo -n "c$c: " >> $R
  taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:$PORT --concurrency $c --num-requests $((c*3)) --prompt-len 400 --max-tokens 64 --prefix-repeat 0 --warmup 1 --seed 11 2>&1 | grep -E "TPOT " | tr '\n' ' ' >> $R
  echo >> $R
done
kill -INT $sp 2>/dev/null; sleep 2; kill -KILL $sp 2>/dev/null; sleep 2
sed "s/\x1b\[[0-9;]*m//g" $L | grep -E "moe expert stats" > /w/dinfer/tmp_${TAG}_stats.txt
python3 -I - /w/dinfer/tmp_${TAG}_stats.txt >> $R <<'PY'
import sys,re,collections
g=collections.defaultdict(list)
for l in open(sys.argv[1]):
    f=dict(re.findall(r'(\w+)=(\d+)',l))
    if 'active_experts' in f: g[int(f['tokens'])].append((int(f['active_experts']),int(f['max_rows_per_expert'])))
for t in sorted(g):
    a=sorted(x[0] for x in g[t]); m=max(x[1] for x in g[t])
    print(f"tokens={t}: n={len(a)} active_experts p50={a[len(a)//2]} mean={sum(a)/len(a):.1f} min={a[0]} max={a[-1]} | max_rows_per_expert={m}")
PY
echo DONE >> $R
