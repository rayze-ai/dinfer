#!/usr/bin/env bash
# D130：Paged 步本身 vs prefix cache 复用。--no-prefix-caching 下，用小 --max-num-batched-tokens 把 prompt 切成多个 chunk（后续 chunk = 带历史 Paged 步），
# 与 4096 预算（单步 Prefill）的输出比。相同 → Paged kernel 路径干净，bug 在 cache 复用；不同 → Paged 步（hist>0）本身错。
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
R=/w/dinfer/tmp_chunkprobe_RESULT.txt; : > $R
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=0 FP8_GEMV=1 HOST_TIME=1
PORT=8384; BIN=${BIN:-./target/release/dinfer}
echo "bin=$(stat -c %y $BIN | cut -c1-16) $(date +%T)" >> $R
OUT=/w/dinfer/tmp_chunkprobe_out
mkdir -p $OUT
run_group() { # $1 tag  $2 budget  $3 extra-args
( # subshell
  local tag=$1 budget=$2 xa=$3
  LOG=/w/dinfer/tmp_chunkprobe_$tag.log
  taskset -c 0-31 $BIN serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens $budget $xa >$LOG 2>&1 &
  SP=$!
  for i in $(seq 1 150); do sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -q "listening addr" && break; sleep 1; done
  sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -q "listening addr" || { echo "== $tag NOTUP" >> $R; sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -iE "error|panic" | head -2 | cut -c1-140 >> $R; kill -9 $SP; exit 0; }
  echo "== $tag [budget=$budget $xa]" >> $R
  lo=$(wc -l < $LOG)
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
outs = [one(p) for p in nat]
json.dump(outs, open(outp, "w"), ensure_ascii=False)
print(f"  saved {len(outs)}")
PY
  echo "  steps: $(sed -n "$((lo+1)),\$p" $LOG | grep -oE "EXIT mode=[A-Za-z]+ t=[0-9]+" | sed "s/EXIT mode=//;s/ t=/:/" | sort | uniq -c | sort -rn | head -6 | awk '{printf "%s×%s ", $2, $1}')" >> $R
  kill -INT $SP 2>/dev/null; sleep 2; kill -KILL $SP 2>/dev/null; sleep 2
)
}
run_group base4096 4096 "--no-prefix-caching"
run_group chunk16 16 "--no-prefix-caching"
run_group chunk32 32 "--no-prefix-caching"
run_group chunk8 8 "--no-prefix-caching"
run_group pc4096 4096 ""
python3 - "$OUT" >> $R 2>&1 <<'PY'
import sys, json, os
d = sys.argv[1]; base = json.load(open(f"{d}/base4096.json"))
for tag in ["chunk16", "chunk32", "chunk8", "pc4096"]:
    p = f"{d}/{tag}.json"
    if not os.path.exists(p): print(f"  {tag}: missing"); continue
    o = json.load(open(p))
    res = ["SAME" if a == b else f"DIFF@{next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)))}" for a, b in zip(base, o)]
    print(f"  {tag} vs base4096: {sum('DIFF' in r for r in res)}/10 differ [{' '.join(res)}]")
PY
echo DONE >> $R
