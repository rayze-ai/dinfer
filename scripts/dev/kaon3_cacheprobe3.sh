#!/usr/bin/env bash
# D121：自然语言版 cacheprobe。同 prompt（chat）三次：第 2/3 次 prefix cache 命中 → 带历史 chunk 走 Paged 步。env KV_FP8_STORE=1 / 默认。
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
R=/w/dinfer/tmp_cacheprobe3_RESULT.txt; : > $R
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=0 FP8_GEMV=1
PORT=8387; BIN=${BIN:-./target/release/dinfer}
echo "bin=$(stat -c %y $BIN | cut -c1-16) $(date +%T)" >> $R
for ENVSET in "KV_FP8_STORE=1" ""; do
( # subshell
  LOG=/w/dinfer/tmp_cacheprobe3_${ENVSET:-plain}.log
  env $ENVSET taskset -c 0-31 $BIN serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 >$LOG 2>&1 &
  SP=$!
  for i in $(seq 1 150); do sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -q "listening addr" && break; sleep 1; done
  sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -q "listening addr" || { echo "== env=[${ENVSET:-plain}] NOTUP" >> $R; sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -iE "error|panic" | head -2 | cut -c1-140 >> $R; kill -9 $SP; exit 0; }
  echo "== env=[${ENVSET:-plain}]" >> $R
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
nd = 0
for i, p in enumerate(nat):
    a = one(p); b = one(p); c = one(p)
    if a == b == c: print(f"  nat{i} SAME")
    else:
        nd += 1; k = next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)))
        print(f"  nat{i} DIFF: 1st vs 2nd first diff char {k}: {a[k:k+30]!r} / {b[k:k+30]!r}; 2nd==3rd {b==c}")
print(f"  total DIFF {nd}/{len(nat)}")
PY
  kill -INT $SP 2>/dev/null; sleep 2; kill -KILL $SP 2>/dev/null; sleep 2
)
done
echo DONE >> $R
