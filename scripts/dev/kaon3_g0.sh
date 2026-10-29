#!/usr/bin/env bash
# g0（loop kaon3-prefill-gemm Step A）：编 HEAD → ① c16/c32/c64 HOST_TIME 跑（late-begin/late-harvest 墙钟 → 副流 prefill tok/ms）
# → ② nsys c16（nsysp16 口径，TAG=nsysp16c）→ 按流分 kernel 表。结果 tmp_g0_RESULT.txt，标记 G0_DONE。
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LIBRARY_PATH=/tmp/stub LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_ARCHS=120 CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0
cd /w/dinfer
R=/w/dinfer/tmp_g0_RESULT.txt; : > $R; date +%T >> $R
cargo build --release -p dinfer --features kaon3 > /w/dinfer/tmp_g0_build.log 2>&1; brc=$?
echo "BUILD EXIT=$brc md5=$(md5sum target/release/dinfer | cut -c1-12) probe=$(strings target/release/dinfer | grep -c 'late-harvest') $(date +%T)" >> $R
[ $brc -ne 0 ] && { grep -E "^error" -A8 /w/dinfer/tmp_g0_build.log | head -30 | cut -c1-200 >> $R; echo G0_DONE >> $R; exit 1; }
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 HOST_TIME=1
L=/w/dinfer/tmp_g0_serve.log
taskset -c 0-31 ./target/release/dinfer serve --model "$M" --tokenizer "$M" --port 8364 --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 >$L 2>&1 &
SP=$!
for i in $(seq 1 240); do sed "s/\x1b\[[0-9;]*m//g" $L | grep -q "listening addr" && break; sleep 1; done
sed "s/\x1b\[[0-9;]*m//g" $L | grep -q "listening addr" || { echo NOTUP >> $R; kill -KILL $SP; echo G0_DONE >> $R; exit 1; }
taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:8364 --concurrency 16 --num-requests 32 --prompt-len 400 --max-tokens 16 --prefix-repeat 0 --warmup 0 --seed 1 >/dev/null 2>&1
for c in 16 32 64; do
  echo -n "c$c: " >> $R
  taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:8364 --concurrency $c --num-requests $((c*8)) --prompt-len 400 --max-tokens 64 --prefix-repeat 0 --warmup 2 --seed $((15000+c)) 2>&1 | grep -E "throughput|TTFT |TPOT " | tr '\n' ' ' | sed -E 's/ throughput: [0-9.]+ req.s +([0-9.]+) output tok.s +[0-9.]+ total tok.s +TTFT +mean +[0-9.]+ +p50 +([0-9.]+) +p90 +[0-9.]+ +p99 +([0-9.]+)[^T]*TPOT +mean +[0-9.]+ +p50 +([0-9.]+) +p90 +[0-9.]+ +p99 +([0-9.]+).*/tps=\1 ttft_p50=\2 ttft_p99=\3 tpot_p50=\4 tpot_p99=\5/' | cut -c1-200 >> $R
done
kill -INT $SP 2>/dev/null; sleep 3; kill -KILL $SP 2>/dev/null; sleep 3
sed "s/\x1b\[[0-9;]*m//g" $L | grep -E "late-begin|late-harvest" > /w/dinfer/tmp_g0_late.txt
python3 -I - /w/dinfer/tmp_g0_late.txt >> $R <<'PY'
import sys,re,statistics as st,collections
# 配对：每个 late-begin 后的第一个 late-harvest
b=None; pairs=[]
for l in open(sys.argv[1]):
    m=re.search(r'late-begin\] prefill_tokens=(\d+) prefill_reqs=(\d+) decode_reqs=(\d+)',l)
    if m: b=(int(m.group(1)),int(m.group(2)),int(m.group(3))); continue
    m=re.search(r'late-harvest\] reqs=(\d+) tokens=(\d+) wall_ms=([\d.]+)',l)
    if m and b: pairs.append((b[0],b[1],b[2],float(m.group(3)))); b=None
g=collections.defaultdict(list)
for t,pr,dr,w in pairs: g[(t//500*500)].append((t,w))
print(f"late pairs={len(pairs)}")
for k in sorted(g):
    v=g[k]; toks=st.median(x[0] for x in v); w=st.median(x[1] for x in v)
    print(f"  prefill tok~{k}: n={len(v)} tokens_med={toks:.0f} wall_med={w:.1f}ms -> {toks/w:.1f} tok/ms (上界: 含等下一步 finish 的离散)")
PY
# ② nsys c16（副流 kernel 表）
TAG=nsysp16c PORT=8365 bash /w/nsysp16.sh
echo "== nsys streams" >> $R
taskset -c 100-127 python3 -I /w/nsys_prefill_steps.py /w/dinfer/tmp_nsysp16c.sqlite --streams >> $R 2>&1
echo G0_DONE >> $R
