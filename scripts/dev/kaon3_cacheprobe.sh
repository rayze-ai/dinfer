#!/usr/bin/env bash
# D90：排除并发——c1 串行，同 prompt 发两次（第 2 次 prefix cache 命中 → 带历史 chunk 走 Paged 步），比两次输出。
# env 组：plain / KV_FP8_STORE=0 / GQA_PAGED_D512_V6_OFF=1。GPU 由 GPU 变量指定（默认 4）。
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
R=/w/dinfer/tmp_cacheprobe_RESULT.txt; : > $R
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=${GPU:-4} FP8_GEMV=1
PORT=8391; BIN=${BIN:-./target/release/dinfer}
echo "bin=$(stat -c %y $BIN | cut -c1-16) gpu=$CUDA_VISIBLE_DEVICES $(date +%T)" >> $R
for ENVSET in "" "KV_FP8_STORE=0" "GQA_PAGED_D512_V6_OFF=1"; do
  LOG=/w/dinfer/tmp_cacheprobe_${ENVSET:-plain}.log
  env $ENVSET taskset -c 92-119 $BIN serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 >$LOG 2>&1 &
  SP=$!
  for i in $(seq 1 150); do sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -q "listening addr" && break; sleep 1; done
  echo "== env=[${ENVSET:-plain}]" >> $R
  python3 - "$PORT" >> $R 2>&1 <<'PY'
import sys, json, urllib.request, random
port = sys.argv[1]; url = f"http://127.0.0.1:{port}/v1/completions"
words = "the of and to in is was for on that with as by at from it an be this which or are were but not have has had his her they their one two time year people way day man thing world life hand part child eye woman place work week case point government company number group problem fact".split()
def mk(seed, n):
    r = random.Random(seed); return " ".join(r.choice(words) for _ in range(n))
def one(p, mt=48):
    body = json.dumps({"model":"kaon3","prompt":p,"max_tokens":mt,"temperature":0}).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request(url, body, {"content-type":"application/json"}), timeout=300))["choices"][0]["text"]
nd = 0
for i, n in enumerate([400, 413, 1200, 90, 33, 64, 65]):
    p = mk(500 + i, n)
    a = one(p); b = one(p); c = one(p)
    tag = "SAME" if a == b == c else "DIFF"
    if a != b or b != c:
        nd += 1
        k = next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)))
        print(f"  prompt{i} ({n} words) {tag}: 1st vs 2nd first diff char {k}: {a[k:k+30]!r} / {b[k:k+30]!r}; 2nd==3rd {b==c}")
    else:
        print(f"  prompt{i} ({n} words) SAME")
print(f"  total DIFF {nd}/7")
PY
  kill -INT $SP 2>/dev/null; sleep 2; kill -KILL $SP 2>/dev/null; sleep 2
done
echo DONE >> $R
