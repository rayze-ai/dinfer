#!/usr/bin/env bash
# D127 二分：自然语言 10 条，同 prompt ×2（第 2 次 prefix cache 命中 → 10–16 行 chunk 走 Paged 步）。
# 组：NOPC（--no-prefix-caching，确定性基线）/ plain / FP8_GEMV=0 / MOE_ROUTE_FUSED=0 / MOE_DECODE_GEMV=0 / KAON3_MEGA=0 / MOE_GROUPED=0(+MEGA=0)
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
R=/w/dinfer/tmp_cacheprobe4_RESULT.txt; : > $R
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=0 FP8_GEMV=1
PORT=8385; BIN=${BIN:-./target/release/dinfer}
echo "bin=$(stat -c %y $BIN | cut -c1-16) $(date +%T)" >> $R
run_group() { # $1 tag  $2 extra-env  $3 extra-args
( # subshell
  local tag=$1 ev=$2 xa=$3
  LOG=/w/dinfer/tmp_cacheprobe4_$tag.log
  env $ev taskset -c 0-31 $BIN serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 $xa >$LOG 2>&1 &
  SP=$!
  for i in $(seq 1 150); do sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -q "listening addr" && break; sleep 1; done
  sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -q "listening addr" || { echo "== $tag NOTUP" >> $R; sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -iE "error|panic" | head -2 | cut -c1-140 >> $R; kill -9 $SP; exit 0; }
  echo "== $tag [$ev $xa]" >> $R
  timeout 600 python3 - "$PORT" >> $R 2>&1 <<'PY'
import sys, json, urllib.request
port = sys.argv[1]; url = f"http://127.0.0.1:{port}/v1/chat/completions"
def one(p, mt=48):
    body = json.dumps({"model":"kaon3","messages":[{"role":"user","content":p}],"max_tokens":mt,"temperature":0}).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request(url, body, {"content-type":"application/json"}), timeout=300))["choices"][0]["message"]["content"]
nat = ["用三句话介绍一下长城的历史。", "What is the capital of France? Answer in one sentence.", "Explain what a hash table is to a beginner.",
       "写一首关于秋天的四行小诗。", "List three differences between TCP and UDP.", "太阳系有几颗行星？分别是什么？",
       "Summarize the plot of Romeo and Juliet in two sentences.", "How do I reverse a linked list in Python? Give code.",
       "请用中文解释什么是量子纠缠，面向高中生。", "Write a haiku about the ocean."]
res = []
for p in nat:
    a = one(p); b = one(p)
    res.append("SAME" if a == b else f"DIFF@{next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)))}")
print(f"  {sum('DIFF' in r for r in res)}/10 differ [{' '.join(res)}]")
PY
  kill -INT $SP 2>/dev/null; sleep 2; kill -KILL $SP 2>/dev/null; sleep 2
)
}
run_group NOPC "" "--no-prefix-caching"
run_group plain "" ""
run_group fp8gemv0 "FP8_GEMV=0" ""
run_group routefused0 "MOE_ROUTE_FUSED=0" ""
run_group decodegemv0 "MOE_DECODE_GEMV=0" ""
run_group mega0 "KAON3_MEGA=0" ""
run_group grouped0 "MOE_GROUPED=0 KAON3_MEGA=0" ""
echo DONE >> $R
