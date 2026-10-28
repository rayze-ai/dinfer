#!/usr/bin/env bash
# D89：定位混批错误在哪一侧。A 先发（200 tok out），50 ms 后 B 发（新 prompt，1/64 tok）→ B 必进 Paged 步（A 的 decode 行同步）。
# 比对：B 的首 token/全文 vs B 串行；A 全文 vs A 串行。三组 env：plain / CUT_IN=0 / ATTN_DECODE_OLD=1。GPU 由 GPU 环境变量指定。
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
R=/w/dinfer/tmp_mixprobe_RESULT.txt; : > $R
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=${GPU:-1} FP8_GEMV=1
PORT=8392; BIN=${BIN:-./target/release/dinfer}
echo "bin=$(stat -c %y $BIN | cut -c1-16) gpu=$CUDA_VISIBLE_DEVICES $(date +%T)" >> $R
for ENVSET in "" "CUT_IN=0" "ATTN_DECODE_OLD=1"; do
  LOG=/w/dinfer/tmp_mixprobe_${ENVSET:-plain}.log
  env $ENVSET taskset -c 32-59 $BIN serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 >$LOG 2>&1 &
  SP=$!
  for i in $(seq 1 150); do sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -q "listening addr" && break; sleep 1; done
  echo "== env=[${ENVSET:-plain}]" >> $R
  python3 - "$PORT" >> $R 2>&1 <<'PY'
import sys, json, urllib.request, threading, time, random
port = sys.argv[1]; url = f"http://127.0.0.1:{port}/v1/completions"
words = "the of and to in is was for on that with as by at from it an be this which or are were but not have has had his her they their one two time year people way day man thing world life hand part child eye woman place work week case point government company number group problem fact".split()
def mk(seed, n):
    r = random.Random(seed); return " ".join(r.choice(words) for _ in range(n))
def one(p, mt):
    body = json.dumps({"model":"kaon3","prompt":p,"max_tokens":mt,"temperature":0}).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request(url, body, {"content-type":"application/json"}), timeout=300))["choices"][0]["text"]
def cmp(tag, a, b):
    if a == b: print(f"  {tag}: SAME"); return
    k = next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)))
    print(f"  {tag}: DIFF at char {k}: serial={a[k:k+30]!r} conc={b[k:k+30]!r}")
# 每组用新 seed（避免 cache），先串行基线（A 200 tok、B 64 tok、B 1 tok），再并发
for case, (na, nb, mtb) in enumerate([(400, 400, 64), (400, 400, 1), (400, 60, 64), (1500, 400, 64)]):
    s0 = 1000 + case * 10 + hash(port) % 7
    pa, pb = mk(s0, na), mk(s0 + 1, nb)
    sa = one(pa, 200); sb = one(pb, mtb)
    out = {}
    ta = threading.Thread(target=lambda: out.__setitem__("a", one(pa, 200))); ta.start()
    time.sleep(0.08)  # A 已在 decode（prefill ~30 ms + 几步）
    out["b"] = one(pb, mtb); ta.join()
    print(f" case{case} A={na}tok/200out B={nb}tok/{mtb}out:")
    cmp("A(decode in flight while B prefill)", sa, out["a"]); cmp("B(prefill in Paged step)", sb, out["b"])
    # 再来一次 cache 态（同 prompt）：B 的 prefill 变成 带历史 chunk（t≈13）
    out2 = {}
    ta = threading.Thread(target=lambda: out2.__setitem__("a", one(pa, 200))); ta.start()
    time.sleep(0.08); out2["b"] = one(pb, mtb); ta.join()
    cmp("A cache-态", sa, out2["a"]); cmp("B cache-态", sb, out2["b"])
PY
  kill -INT $SP 2>/dev/null; sleep 2; kill -KILL $SP 2>/dev/null; sleep 2
done
echo DONE >> $R
