#!/usr/bin/env bash
# b50（D210）：B48 MoE prefill 大 tile（BM64×BN64 p64 kernel，R > MOE_DEC_RMAX 时走）。
# 编 → 微基准 moe_gemm_timing（MOE_PREFILL_P64=0/1 两进程：P424/P2148 的 us + CHECKSUM 必须逐位同）→ 门禁过才起 ABAB（A=P64 关，B=默认开）c8–c64 两轮。
# 结果 tmp_b50_RESULT.txt，标记 B50_DONE。容器内：docker exec -d kaon3-full bash -c 'bash /w/b50.sh > /w/dinfer/tmp_b50_run.log 2>&1'
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LIBRARY_PATH=/tmp/stub LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_ARCHS=120 CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0
cd /w/dinfer
TAG=${TAG:-b50}; PORT=${PORT:-8344}; CS=${CS:-"8 16 32 64"}
R=/w/dinfer/tmp_${TAG}_RESULT.txt; : > $R; date +%T >> $R
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
cargo test --release -p dinfer-kernels --test moe_gemm_timing --no-run > /w/dinfer/tmp_${TAG}_tbuild.log 2>&1; rc=$?; echo "TEST BUILD EXIT=$rc" >> $R
[ $rc -ne 0 ] && { grep -E "error" -A8 /w/dinfer/tmp_${TAG}_tbuild.log | head -40 | cut -c1-200 >> $R; echo B50_DONE >> $R; exit 1; }
for v in 0 1; do
  MOE_PREFILL_P64=$v cargo test --release -p dinfer-kernels --test moe_gemm_timing -- --ignored --nocapture 2>&1 | grep -E "MOE_GEMM_TIMING (P424|P2148|M=8).*(dual|down grouped|CHECKSUM)|panicked" | sed "s/^/P64=$v /" | cut -c1-200 >> $R
done
python3 -I - $R <<'PY' >> $R
import sys,re
ck={}
for l in open(sys.argv[1]):
    m=re.match(r'P64=(\d) MOE_GEMM_TIMING (\S+) rows=\d+ CHECKSUM (.*)',l)
    if m: ck[(m.group(2),m.group(1))]=m.group(3)
bad=[n for n in {k[0] for k in ck} if ck.get((n,'0'))!=ck.get((n,'1'))]
print("GATE="+("0" if not bad and ck else "1")+" mismatch="+",".join(bad))
PY
grep -q "GATE=0" $R || { echo B50_DONE >> $R; exit 1; }
cargo build --release -p dinfer --features kaon3 > /w/dinfer/tmp_${TAG}_build.log 2>&1; brc=$?
echo "BUILD EXIT=$brc md5=$(md5sum target/release/dinfer | cut -c1-12) marker=$(strings target/release/dinfer | grep -c MOE_PREFILL_P64) $(date +%T)" >> $R
[ $brc -ne 0 ] && { grep -E "^error" -A8 /w/dinfer/tmp_${TAG}_build.log | head -30 | cut -c1-200 >> $R; echo B50_DONE >> $R; exit 1; }
echo "serve_taskset=0-31 bench_taskset=60-91 A=MOE_PREFILL_P64=0 B=default" >> $R
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 HOST_TIME=1
strip() { sed "s/\x1b\[[0-9;]*m//g"; }
run_var() {  # $1=round $2=tag $3=env
  local r=$1 tag=$2 L=/w/dinfer/tmp_${TAG}_r${1}_$2.log
  env $3 taskset -c 0-31 ./target/release/dinfer serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 >$L 2>&1 &
  local sp=$!
  for i in $(seq 1 180); do strip < $L | grep -q "listening addr" && break; sleep 1; done
  strip < $L | grep -q "listening addr" || { echo "r$r $tag NOTUP" >> $R; strip < $L | grep -iE "error|panic" | tail -2 | cut -c1-200 >> $R; kill -KILL $sp 2>/dev/null; return 1; }
  taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:$PORT --concurrency 16 --num-requests 32 --prompt-len 400 --max-tokens 16 --prefix-repeat 0 --warmup 0 --seed 1 >/dev/null 2>&1
  local vi=0; [ $tag = B ] && vi=1
  for c in $CS; do
    local seed=$((11000 + 100 * r + c + vi * 13)) LO=$(wc -l < $L)
    echo -n "r$r $tag c$c: " >> $R
    taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:$PORT --concurrency $c --num-requests $((c*8)) --prompt-len 400 --max-tokens 64 --prefix-repeat 0 --warmup 2 --seed $seed 2>&1 | grep -E "throughput|TTFT |TPOT " | tr '\n' ' ' | sed -E 's/ throughput: [0-9.]+ req.s +([0-9.]+) output tok.s +[0-9.]+ total tok.s +TTFT +mean +[0-9.]+ +p50 +([0-9.]+) +p90 +[0-9.]+ +p99 +([0-9.]+)[^T]*TPOT +mean +[0-9.]+ +p50 +([0-9.]+) +p90 +[0-9.]+ +p99 +([0-9.]+).*/tps=\1 ttft_p50=\2 ttft_p99=\3 tpot_p50=\4 tpot_p99=\5/' | cut -c1-200 >> $R
    local HI=$(wc -l < $L)
    echo " || $(python3 -I /w/kaon3_hosttime_parse.py $L $LO $HI 2>&1 | sed -E 's/render=[^|]*\|\| //; s/ publish=[0-9]+ total=[0-9]+ \| upd=[^|]*\| htod=[^|]*//g' | cut -c1-200)" >> $R
  done
  echo -n "r$r $tag out: " >> $R
  curl -s -m30 http://127.0.0.1:$PORT/v1/completions -H "content-type: application/json" -d '{"model":"kaon3","prompt":"The capital of France is","max_tokens":24,"temperature":0}' | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'][:100].replace(chr(10),' '))" >> $R 2>&1
  echo -n "r$r $tag out2: " >> $R
  curl -s -m60 http://127.0.0.1:$PORT/v1/chat/completions -H "content-type: application/json" -d '{"model":"kaon3","messages":[{"role":"user","content":"用三句话介绍一下长城的历史。"}],"max_tokens":60,"temperature":0}' | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'][:100].replace(chr(10),' '))" >> $R 2>&1
  kill -INT $sp 2>/dev/null; sleep 2; kill -KILL $sp 2>/dev/null; sleep 3
}
for r in 1 2; do run_var $r A "MOE_PREFILL_P64=0"; run_var $r B "MOE_PREFILL_P64=1"; done
echo B50_DONE >> $R
