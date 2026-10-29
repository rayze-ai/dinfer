#!/usr/bin/env bash
# 通用 env A/B 自然语言输出比对（串行，--no-prefix-caching；10 条 chat + 1 条 raw 补全）：TAG=<名> AENV="K=V" BENV="K=V" bash /w/envnat.sh
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
TAG=${TAG:-envnat}; PORT=${PORT:-8381}; BIN=${BIN:-./target/release/dinfer}
R=/w/dinfer/tmp_${TAG}_RESULT.txt; : > $R
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=0
echo "bin=$(stat -c %y $BIN | cut -c1-16) A=[$AENV] B=[$BENV] $(date +%T)" >> $R
OUT=/w/dinfer/tmp_${TAG}_out; mkdir -p $OUT
run_one() { # $1 tag $2 env
( local tag=$1; shift
  LOG=/w/dinfer/tmp_${TAG}_$tag.log
  setsid env $* taskset -c 0-31 $BIN serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 --no-prefix-caching >$LOG 2>&1 &
  SP=$!
  for i in $(seq 1 150); do sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -q "listening addr" && break; sleep 1; done
  sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -q "listening addr" || { echo "$tag NOTUP" >> $R; kill -9 $SP; exit 0; }
  timeout 600 python3 - "$PORT" "$OUT/$tag.json" >> $R 2>&1 <<'PY'
import sys, json, urllib.request
port, outp = sys.argv[1], sys.argv[2]
def chat(p, mt=48):
    body = json.dumps({"model":"kaon3","messages":[{"role":"user","content":p}],"max_tokens":mt,"temperature":0}).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"content-type":"application/json"}), timeout=300))["choices"][0]["message"]["content"]
def raw(p, mt=24):
    body = json.dumps({"model":"kaon3","prompt":p,"max_tokens":mt,"temperature":0}).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body, {"content-type":"application/json"}), timeout=300))["choices"][0]["text"]
nat = ["用三句话介绍一下长城的历史。", "What is the capital of France? Answer in one sentence.", "Explain what a hash table is to a beginner.",
       "写一首关于秋天的四行小诗。", "List three differences between TCP and UDP.", "太阳系有几颗行星？分别是什么？",
       "Summarize the plot of Romeo and Juliet in two sentences.", "How do I reverse a linked list in Python? Give code.",
       "请用中文解释什么是量子纠缠，面向高中生。", "Write a haiku about the ocean."]
outs = [chat(p) for p in nat] + [raw("The capital of France is"), raw("Once upon a time, in a small village by the sea,")]
json.dump(outs, open(outp, "w"), ensure_ascii=False); print(f"  {len(outs)} saved")
PY
  kill -INT $SP 2>/dev/null; sleep 2; kill -KILL $SP 2>/dev/null; sleep 2 )
}
run_one A $AENV
run_one B $BENV
python3 - "$OUT" >> $R 2>&1 <<'PY'
import sys, json
d = sys.argv[1]; a = json.load(open(f"{d}/A.json")); b = json.load(open(f"{d}/B.json"))
res = []
for x, y in zip(a, b):
    res.append("SAME" if x == y else "DIFF@%d" % next((j for j,(p,q) in enumerate(zip(x,y)) if p!=q), min(len(x),len(y))))
print("  A vs B: %d/%d differ [%s]" % (sum("DIFF" in r for r in res), len(res), " ".join(res)))
for i, (x, y) in enumerate(zip(a, b)):
    if x != y: print("   #%d A=%r\n      B=%r" % (i, x[:70], y[:70]))
PY
echo DONE >> $R
