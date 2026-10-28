#!/usr/bin/env bash
# D125 二分：A = 固定自然语言 prompt（64 tok）。对照：
#  S  串行 ×2（确定性基线）
#  P1 A+B 同时发，B max_tokens=1（只共享 1 个 Prefill 步 + ≤1 个 Decode 步）
#  P2 A+B 同时发，B 64 tok（共享 Prefill + 全程 Decode 2 行）
#  P3 B 在 A 发出 150 ms 后到（A 单独 Prefill；B 单独 Prefill(作为混批/Paged)；之后共 Decode）
# env 组：plain / MOE_ROUTE_FUSED=0 / MOE_DECODE_GEMV=0 / FP8_GEMV=0 / NO_GRAPH=1 / KAON3_MEGA=0
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
R=/w/dinfer/tmp_natprobe2_RESULT.txt; : > $R
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=0 FP8_GEMV=1
PORT=8386; BIN=${BIN:-./target/release/dinfer}
echo "bin=$(stat -c %y $BIN | cut -c1-16) $(date +%T)" >> $R
for ENVSET in "" "MOE_ROUTE_FUSED=0" "MOE_DECODE_GEMV=0" "FP8_GEMV=0" "NO_GRAPH=1" "KAON3_MEGA=0"; do
( # subshell
  LOG=/w/dinfer/tmp_natprobe2_${ENVSET:-plain}.log
  env $ENVSET taskset -c 0-31 $BIN serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 >$LOG 2>&1 &
  SP=$!
  for i in $(seq 1 150); do sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -q "listening addr" && break; sleep 1; done
  sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -q "listening addr" || { echo "== env=[${ENVSET:-plain}] NOTUP" >> $R; sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -iE "error|panic" | head -2 | cut -c1-140 >> $R; kill -9 $SP; exit 0; }
  echo "== env=[${ENVSET:-plain}]" >> $R
  timeout 600 python3 - "$PORT" >> $R 2>&1 <<'PY'
import sys, json, urllib.request, threading, time
port = sys.argv[1]; url = f"http://127.0.0.1:{port}/v1/chat/completions"
def one(p, mt=64):
    body = json.dumps({"model":"kaon3","messages":[{"role":"user","content":p}],"max_tokens":mt,"temperature":0}).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request(url, body, {"content-type":"application/json"}), timeout=300))["choices"][0]["message"]["content"]
def cmp(a, b):
    if a == b: return "SAME"
    k = next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)))
    return f"DIFF@{k}"
A = "用三句话介绍一下长城的历史。"
Bs = ["Explain what a hash table is to a beginner.", "太阳系有几颗行星？分别是什么？", "Write a haiku about the ocean."]
sa = one(A); sa2 = one(A)
print(f" S  serial×2: {cmp(sa, sa2)}")
def pair(bprompt, bmt, delay):
    out = {}
    ta = threading.Thread(target=lambda: out.__setitem__("a", one(A))); ta.start()
    if delay: time.sleep(delay)
    out["b"] = one(bprompt, bmt); ta.join()
    return out["a"]
for i, bp in enumerate(Bs):
    r1 = cmp(sa, pair(bp, 1, 0)); r2 = cmp(sa, pair(bp, 64, 0)); r3 = cmp(sa, pair(bp, 64, 0.15))
    print(f" B{i}: P1(同发,B 1tok)={r1}  P2(同发,B 64tok)={r2}  P3(B 晚 150ms)={r3}")
PY
  kill -INT $SP 2>/dev/null; sleep 2; kill -KILL $SP 2>/dev/null; sleep 2
)
done
echo DONE >> $R
