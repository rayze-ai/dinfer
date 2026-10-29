#!/usr/bin/env bash
# 高并发对照（用户 2026-10-29："测 c16/c32/c64 有优势的数据"）：dinfer（当前 target）vs vLLM 基线 :28201，同 bench 同参数，
# c ∈ {8,16,32,64}，每档两轮交错（d/v/d/v），每档换 seed（避免同服务 prefix cache 命中）。两端 max-num-seqs 64。
# 钉核：serve 0-31、bench 60-91（vLLM 容器在 GPU 2/3，自己的核）。结果 tmp_cN_RESULT.txt，唯一结束标记 CN_DONE。
NV=/usr/local/lib/python3.12/dist-packages/nvidia
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
TAG=${TAG:-cN}; PORT=${PORT:-8399}; VPORT=28201
R=/w/dinfer/tmp_${TAG}_RESULT.txt; : > $R; L=/w/dinfer/tmp_${TAG}_serve.log
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
echo "bin=$(stat -c %y target/release/dinfer | cut -c1-16) md5=$(md5sum target/release/dinfer | cut -c1-12) $(date +%T)" >> $R
echo "vllm_models=$(curl -s -m5 http://127.0.0.1:$VPORT/v1/models | python3 -c 'import sys,json;print([m["id"] for m in json.load(sys.stdin)["data"]])' 2>&1)" >> $R
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=0
taskset -c 0-31 ./target/release/dinfer serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 >$L 2>&1 &
sp=$!
for i in $(seq 1 150); do sed "s/\x1b\[[0-9;]*m//g" $L | grep -q "listening addr" && break; sleep 1; done
sed "s/\x1b\[[0-9;]*m//g" $L | grep -q "listening addr" || { echo "NOTUP" >> $R; kill -KILL $sp 2>/dev/null; echo CN_DONE >> $R; exit 1; }
echo "dinfer_models=$(curl -s -m5 http://127.0.0.1:$PORT/v1/models | python3 -c 'import sys,json;print([m["id"] for m in json.load(sys.stdin)["data"]])' 2>&1)" >> $R
taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:$PORT --concurrency 16 --num-requests 32 --prompt-len 400 --max-tokens 16 --prefix-repeat 0 --warmup 0 --seed 1 >/dev/null 2>&1
for round in 1 2; do
  for c in 8 16 32 64; do
    seed=$((100 * round + c))
    for side in d v; do
      if [ $side = d ]; then url=http://127.0.0.1:$PORT; else url=http://127.0.0.1:$VPORT; fi
      echo -n "r$round c$c $side: " >> $R
      taskset -c 60-91 ./target/release/bench --url $url --concurrency $c --num-requests $((c*4)) --prompt-len 400 --max-tokens 64 --prefix-repeat 0 --warmup 2 --seed $seed 2>&1 | grep -E "throughput|TTFT |TPOT " | tr '\n' ' ' | sed 's/  */ /g' >> $R
      echo >> $R
    done
  done
done
kill -INT $sp 2>/dev/null; sleep 2; kill -KILL $sp 2>/dev/null; sleep 2
echo CN_DONE >> $R
