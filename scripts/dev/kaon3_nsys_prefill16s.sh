#!/usr/bin/env bash
# D201：抓 c16 混批（pair 步 prefill 副流）的 eager kernel 清单。stepA4 口径（400 tok prompt / 64 out / c16 / 128 请求），
# nsys --cuda-graph-trace=node；结束后 nsys_prefill_steps.py 按 20 ms 间隙切步、取最长 3 步聚合。结果 tmp_nsysp16_RESULT.txt，标记 NSYSP16_DONE。
# 容器内：docker exec -d kaon3-full bash -c 'bash /w/nsysp16.sh > /w/dinfer/tmp_nsysp16_run.log 2>&1'
TAG=${TAG:-nsysp16}; C=${C:-16}; PORT=${PORT:-8341}
R=/w/dinfer/tmp_${TAG}_RESULT.txt; : > $R
NSYS=$(ls /opt/nvidia/nsight-systems/2026*/target-linux-x64/nsys 2>/dev/null | head -1)
[ -z "$NSYS" ] && { echo "no nsys" >> $R; echo NSYSP16_DONE >> $R; exit 1; }
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
echo "bin=$(stat -c %y target/release/dinfer | cut -c1-16) md5=$(md5sum target/release/dinfer | cut -c1-12) c=$C env=[${EXTRA_ENV:-}]" >> $R
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=0 HOST_TIME=1 ${EXTRA_ENV:-}
rm -f /w/dinfer/tmp_${TAG}.nsys-rep /w/dinfer/tmp_${TAG}.sqlite
taskset -c 0-31 "$NSYS" profile ${NSYS_DELAY:+--delay $NSYS_DELAY} -o /w/dinfer/tmp_${TAG} --force-overwrite true -t cuda --cuda-graph-trace=node --cuda-flush-interval=1000 --sample none --cpuctxsw none \
  ./target/release/dinfer serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 >/w/dinfer/tmp_${TAG}_serve.log 2>&1 &
NP=$!
for i in $(seq 1 180); do sed "s/\x1b\[[0-9;]*m//g" /w/dinfer/tmp_${TAG}_serve.log | grep -q "listening addr" && break; sleep 1; done
sed "s/\x1b\[[0-9;]*m//g" /w/dinfer/tmp_${TAG}_serve.log | grep -q "listening addr" || { echo "serve NOTUP" >> $R; tail -3 /w/dinfer/tmp_${TAG}_serve.log >> $R; echo NSYSP16_DONE >> $R; kill -9 $NP; exit 1; }
taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:$PORT --concurrency $C --num-requests $((C*${NREQ_MULT:-8})) --prompt-len 400 --max-tokens 64 --prefix-repeat 0 --warmup 2 --seed 4016 2>&1 | grep -E "throughput|TTFT |TPOT " | tr '\n' ' ' | cut -c1-300 >> $R; echo >> $R
kill -INT $(pgrep -x dinfer) 2>/dev/null; wait $NP 2>/dev/null; sleep 2; ls -la /w/dinfer/tmp_${TAG}.nsys-rep | cut -c1-100 >> $R 2>&1
"$NSYS" export --type sqlite --force-overwrite true -o /w/dinfer/tmp_${TAG}.sqlite /w/dinfer/tmp_${TAG}.nsys-rep >/dev/null 2>&1
taskset -c 100-127 python3 -I /w/nsys_prefill_steps.py /w/dinfer/tmp_${TAG}.sqlite 400 30 20 >> $R 2>&1
echo NSYSP16_DONE >> $R
# host-time 对账：混批步（tokens != nreq）数与均值
sed "s/\x1b\[[0-9;]*m//g" /w/dinfer/tmp_${TAG}_serve.log | grep -E "host-time" | python3 -I -c '
import sys,re
mixed=[];dec=[]
for l in sys.stdin:
    f=dict(re.findall(r"(\w+(?:\(begin\+finish\))?)=([\d.]+)",l))
    if "total" not in f: continue
    t,n,e=int(f["tokens"]),int(f["nreq"]),float(f["exec(begin+finish)"])
    (dec if t==n else mixed).append((t,n,e))
import statistics as st
print(f"host-time: decode steps={len(dec)} exec_mean={st.mean([e for *_,e in dec])/1e3 if dec else 0:.2f}ms | mixed steps={len(mixed)} toks_mean={st.mean([t for t,_,_ in mixed]) if mixed else 0:.0f} nreq_mean={st.mean([n for _,n,_ in mixed]) if mixed else 0:.1f} exec_mean={st.mean([e for *_,e in mixed])/1e3 if mixed else 0:.1f}ms exec_max={max([e for *_,e in mixed])/1e3 if mixed else 0:.1f}ms")
' >> $R
