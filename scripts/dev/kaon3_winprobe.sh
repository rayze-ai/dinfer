#!/usr/bin/env bash
# D136：hist>0 Paged 步 bug 是否在 v6 PAGED D=256 window 路径。--no-prefix-caching；每 env 组：base4096 vs chunk16。
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
R=/w/dinfer/tmp_winprobe_RESULT.txt; : > $R
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=0 FP8_GEMV=1
PORT=8383; BIN=${BIN:-./target/release/dinfer}; OUT=/w/dinfer/tmp_winprobe_out; mkdir -p $OUT
echo "bin=$(stat -c %y $BIN | cut -c1-16) $(date +%T)" >> $R
run_group() { # $1 tag $2 env $3 budget
( local tag=$1 ev=$2 budget=$3; LOG=/w/dinfer/tmp_winprobe_$tag.log
  env $ev taskset -c 0-31 $BIN serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens $budget --no-prefix-caching >$LOG 2>&1 &
  SP=$!
  for i in $(seq 1 150); do sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -q "listening addr" && break; sleep 1; done
  sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -q "listening addr" || { echo "== $tag NOTUP" >> $R; kill -9 $SP; exit 0; }
  echo "== $tag [$ev budget=$budget]" >> $R
  timeout 600 python3 - "$PORT" "$OUT/$tag.json" >> $R 2>&1 <<'PY'
import sys, json, urllib.request
port, outp = sys.argv[1], sys.argv[2]; url = f"http://127.0.0.1:{port}/v1/chat/completions"
def one(p, mt=48):
    body = json.dumps({"model":"kaon3","messages":[{"role":"user","content":p}],"max_tokens":mt,"temperature":0}).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request(url, body, {"content-type":"application/json"}), timeout=300))["choices"][0]["message"]["content"]
nat = ["用三句话介绍一下长城的历史。", "What is the capital of France? Answer in one sentence.", "Explain what a hash table is to a beginner.",
       "写一首关于秋天的四行小诗。", "List three differences between TCP and UDP.", "太阳系有几颗行星？分别是什么？",
       "Summarize the plot of Romeo and Juliet in two sentences.", "How do I reverse a linked list in Python? Give code.",
       "请用中文解释什么是量子纠缠，面向高中生。", "Write a haiku about the ocean."]
json.dump([one(p) for p in nat], open(outp, "w"), ensure_ascii=False); print("  saved")
PY
  kill -INT $SP 2>/dev/null; sleep 2; kill -KILL $SP 2>/dev/null; sleep 2 )
}
run_group v6_base "" 4096
run_group v6_chunk16 "" 16
run_group v5_base "GQA_PREFILL_CFG=0" 4096
run_group v5_chunk16 "GQA_PREFILL_CFG=0" 16
run_group d512ref_chunk16 "GQA_D512_REF=1" 16
python3 - "$OUT" >> $R 2>&1 <<'PY'
import sys, json, os
d = sys.argv[1]
def cmp(a, b):
    r = []
    for x, y in zip(a, b):
        r.append("SAME" if x == y else "DIFF@%d" % next((j for j,(p,q) in enumerate(zip(x,y)) if p!=q), min(len(x),len(y))))
    return "%d/10 [%s]" % (sum("DIFF" in t for t in r), " ".join(r))
L = lambda t: json.load(open(f"{d}/{t}.json")) if os.path.exists(f"{d}/{t}.json") else None
b6, c6, b5, c5, dr = (L(t) for t in ["v6_base","v6_chunk16","v5_base","v5_chunk16","d512ref_chunk16"])
if b6 and c6: print("  v6: chunk16 vs base:", cmp(b6, c6))
if b5 and c5: print("  v5(GQA_PREFILL_CFG=0): chunk16 vs base:", cmp(b5, c5))
if b6 and b5: print("  v5_base vs v6_base:", cmp(b6, b5))
if b6 and dr: print("  d512ref chunk16 vs v6_base:", cmp(b6, dr))
PY
echo DONE >> $R
