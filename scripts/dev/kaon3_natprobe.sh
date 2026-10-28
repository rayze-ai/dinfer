#!/usr/bin/env bash
# D111：区分"并发 bug"与"批尺寸数值噪声翻转贪心"。自然语言 prompt（分布尖）串行 vs c2/c8 应 SAME；
# 词沙拉 prompt（分布平）作对照。同一服务；每组换 prompt 避免 cache。env 组 plain / NO_GRAPH=1。
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
R=/w/dinfer/tmp_natprobe_RESULT.txt; : > $R
trap 'echo DONE >> $R' EXIT
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=0 FP8_GEMV=1
PORT=8388; BIN=${BIN:-./target/release/dinfer}
echo "bin=$(stat -c %y $BIN | cut -c1-16) $(date +%T)" >> $R
#!/usr/bin/env bash
# D111：区分"并发 bug"与"批尺寸数值噪声翻转贪心"。自然语言 prompt（分布尖）串行 vs c2/c8 应 SAME；
# 词沙拉 prompt（分布平）作对照。同一服务；每组换 prompt 避免 cache。env 组 plain / NO_GRAPH=1。
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
R=/w/dinfer/tmp_natprobe_RESULT.txt; : > $R
trap 'echo DONE >> $R' EXIT
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=0 FP8_GEMV=1
PORT=8388; BIN=${BIN:-./target/release/dinfer}
echo "bin=$(stat -c %y $BIN | cut -c1-16) $(date +%T)" >> $R
for ENVSET in "" "NO_GRAPH=1"; do
( # subshell：子进程被杀 / timeout 不影响外层循环（D113）
  pkill -x dinfer; sleep 3
  LOG=/w/dinfer/tmp_natprobe_${ENVSET:-plain}.log
  env $ENVSET taskset -c 0-31 $BIN serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 >$LOG 2>&1 &
  SP=$!
  for i in $(seq 1 150); do sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -q "listening addr" && break; sleep 1; done
  sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -q "listening addr" || { echo "== env=[${ENVSET:-plain}] NOTUP" >> $R; sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -iE "error|panic" | head -2 | cut -c1-140 >> $R; kill -9 $SP; continue; }
  echo "== env=[${ENVSET:-plain}]" >> $R
  timeout 900 python3 - "$PORT" >> $R 2>&1 <<'PY'
import sys, json, urllib.request, concurrent.futures as cf, random
port = sys.argv[1]; url = f"http://127.0.0.1:{port}/v1/chat/completions"
def one(p, mt=64):
    body = json.dumps({"model":"kaon3","messages":[{"role":"user","content":p}],"max_tokens":mt,"temperature":0}).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request(url, body, {"content-type":"application/json"}), timeout=300))["choices"][0]["message"]["content"]
def cmp(a, b):
    if a == b: return "SAME"
    k = next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)))
    return f"DIFF@{k}"
nat = ["用三句话介绍一下长城的历史。", "What is the capital of France? Answer in one sentence.", "Explain what a hash table is to a beginner.",
       "写一首关于秋天的四行小诗。", "List three differences between TCP and UDP.", "太阳系有几颗行星？分别是什么？",
       "Summarize the plot of Romeo and Juliet in two sentences.", "How do I reverse a linked list in Python? Give code."]
words = "the of and to in is was for on that with as by at from it an be this which or are were but not have has had his her they their one two time year people way day man thing world life hand part child eye woman place work week case point government company number group problem fact".split()
def salad(seed): r = random.Random(seed); return " ".join(r.choice(words) for _ in range(300))
for tag, prompts in [("natural", nat), ("salad", [salad(900 + i) for i in range(8)])]:
    ser = [one(p) for p in prompts]
    with cf.ThreadPoolExecutor(2) as ex: c2 = list(ex.map(one, prompts))
    with cf.ThreadPoolExecutor(8) as ex: c8 = list(ex.map(one, prompts))
    r2 = [cmp(s, c) for s, c in zip(ser, c2)]; r8 = [cmp(s, c) for s, c in zip(ser, c8)]
    print(f" {tag}: c2 {sum('DIFF' in r for r in r2)}/8 differ [{' '.join(r2)}] | c8 {sum('DIFF' in r for r in r8)}/8 differ [{' '.join(r8)}]")
    if tag == "natural":
        for i, (s, c) in enumerate(zip(ser, c8)):
            if s != c: print(f"   nat{i} serial={s[:60]!r}\n          c8    ={c[:60]!r}")
PY
  kill -INT $SP 2>/dev/null; sleep 2; kill -KILL $SP 2>/dev/null; sleep 2
)
done
