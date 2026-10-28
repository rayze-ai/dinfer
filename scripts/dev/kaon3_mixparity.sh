#!/usr/bin/env bash
# D87：混批正确性核查。同服务、同 prompt 集（python 直发 /v1/completions，greedy，64 tok）：串行逐条 vs 2 路/8 路并发，逐请求 text 比对。
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
R=/w/dinfer/tmp_mixparity_RESULT.txt; : > $R
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=0 FP8_GEMV=1
PORT=8393; LOG=/w/dinfer/tmp_mixparity.log
echo "bin=$(stat -c %y target/release/dinfer | cut -c1-16) $(date +%T) ${EXTRA_ENV:-}" >> $R
env ${EXTRA_ENV:-} taskset -c 0-31 ./target/release/dinfer serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 >$LOG 2>&1 &
SP=$!
for i in $(seq 1 150); do sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -q "listening addr" && break; sleep 1; done
python3 - "$PORT" >> $R 2>&1 <<'PY'
import sys, json, urllib.request, concurrent.futures as cf, random
port = sys.argv[1]
url = f"http://127.0.0.1:{port}/v1/completions"
random.seed(3)
words = "the of and to in is was for on that with as by at from it an be this which or are were but not have has had his her they their one two time year people way day man thing world life hand part child eye woman place work week case point government company number group problem fact".split()
prompts = [" ".join(random.choice(words) for _ in range(300 + 7*i)) for i in range(8)]
def one(p):
    body = json.dumps({"model":"kaon3","prompt":p,"max_tokens":64,"temperature":0}).encode()
    req = urllib.request.Request(url, body, {"content-type":"application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["text"]
# 预热 + 让 prefix cache 吃掉 prompt（之后三种并发度的 prefill 都走 cache → 只剩 decode 行差异）；再额外做一组不预热的
base = [one(p) for p in prompts]
for c in (2, 8):
    with cf.ThreadPoolExecutor(c) as ex: outs = list(ex.map(one, prompts))
    nd = sum(a != b for a, b in zip(base, outs))
    print(f"c{c} vs serial: {nd}/8 requests differ (cache 态)")
    for i,(a,b) in enumerate(zip(base,outs)):
        if a != b:
            k = next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)))
            print(f"  req{i}: first diff at char {k}: serial={a[k:k+40]!r} conc={b[k:k+40]!r}")
# 新 prompt（无 cache）：先并发 8 路，再串行，比对
prompts2 = [" ".join(random.choice(words) for _ in range(330 + 5*i)) for i in range(8)]
with cf.ThreadPoolExecutor(8) as ex: outs = list(ex.map(one, prompts2))
base2 = [one(p) for p in prompts2]
nd = sum(a != b for a, b in zip(base2, outs))
print(f"c8(fresh) vs serial: {nd}/8 requests differ")
for i,(a,b) in enumerate(zip(base2,outs)):
    if a != b:
        k = next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)))
        print(f"  req{i}: first diff at char {k}: serial={a[k:k+40]!r} conc={b[k:k+40]!r}")
PY
kill -INT $SP 2>/dev/null; sleep 2; kill -KILL $SP 2>/dev/null
echo DONE >> $R
