#!/usr/bin/env bash
# D143：纯并发对照（排除 cache 命中混杂）：--no-prefix-caching，同 8 条自然语言 prompt，串行 vs c2 vs c8（都 fresh）。
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
R=/w/dinfer/tmp_concprobe_RESULT.txt; : > $R
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=0 FP8_GEMV=1
PORT=8382; BIN=${BIN:-./target/release/dinfer}
echo "bin=$(stat -c %y $BIN | cut -c1-16) $(date +%T)" >> $R
( LOG=/w/dinfer/tmp_concprobe.log
  taskset -c 0-31 $BIN serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 --no-prefix-caching >$LOG 2>&1 &
  SP=$!
  for i in $(seq 1 150); do sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -q "listening addr" && break; sleep 1; done
  sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -q "listening addr" || { echo "NOTUP" >> $R; kill -9 $SP; exit 0; }
  timeout 900 python3 - "$PORT" >> $R 2>&1 <<'PY'
import sys, json, urllib.request, concurrent.futures as cf
port = sys.argv[1]; url = f"http://127.0.0.1:{port}/v1/chat/completions"
def one(p, mt=64):
    body = json.dumps({"model":"kaon3","messages":[{"role":"user","content":p}],"max_tokens":mt,"temperature":0}).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request(url, body, {"content-type":"application/json"}), timeout=300))["choices"][0]["message"]["content"]
def cmp(a, b):
    if a == b: return "SAME"
    return "DIFF@%d" % next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)))
nat = ["用三句话介绍一下长城的历史。", "What is the capital of France? Answer in one sentence.", "Explain what a hash table is to a beginner.",
       "写一首关于秋天的四行小诗。", "List three differences between TCP and UDP.", "太阳系有几颗行星？分别是什么？",
       "Summarize the plot of Romeo and Juliet in two sentences.", "How do I reverse a linked list in Python? Give code."]
ser = [one(p) for p in nat]; ser2 = [one(p) for p in nat]
print(" serial×2 (no cache): %d/8 differ" % sum(a != b for a, b in zip(ser, ser2)))
with cf.ThreadPoolExecutor(2) as ex: c2 = list(ex.map(one, nat))
with cf.ThreadPoolExecutor(8) as ex: c8 = list(ex.map(one, nat))
with cf.ThreadPoolExecutor(8) as ex: c8b = list(ex.map(one, nat))
r2 = [cmp(s, c) for s, c in zip(ser, c2)]; r8 = [cmp(s, c) for s, c in zip(ser, c8)]; r88 = [cmp(s, c) for s, c in zip(c8, c8b)]
print(" c2 vs serial: %d/8 [%s]" % (sum("DIFF" in r for r in r2), " ".join(r2)))
print(" c8 vs serial: %d/8 [%s]" % (sum("DIFF" in r for r in r8), " ".join(r8)))
print(" c8 vs c8(repeat): %d/8 [%s]" % (sum("DIFF" in r for r in r88), " ".join(r88)))
for i, (s, c) in enumerate(zip(ser, c8)):
    if s != c: print("   nat%d serial=%r\n         c8    =%r" % (i, s[:60], c[:60]))
PY
  kill -INT $SP 2>/dev/null; sleep 2; kill -KILL $SP 2>/dev/null )
echo DONE >> $R
