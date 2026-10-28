#!/usr/bin/env bash
# D100：两请求并发 decode 正确性。A/B 同时发（新 prompt，64 out）vs 各自串行；环境组 plain / CUT_IN=0 / ATTN_DECODE_OLD=1 / NO_GRAPH=1。
# 另：B max_tokens=1 对照；8 路同时发对照。都在 KV_FP8_STORE=0（当前远端二进制默认开 → 显式关）。
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
R=/w/dinfer/tmp_pair2_RESULT.txt; : > $R
# DONE 只在脚本真正结尾写（trap EXIT 会在 timeout 子进程退出时提前触发，D113）
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=0 FP8_GEMV=1 KV_FP8_STORE=0
PORT=8389; BIN=${BIN:-./target/release/dinfer}
echo "bin=$(stat -c %y $BIN | cut -c1-16) $(date +%T)" >> $R
#!/usr/bin/env bash
# D100：两请求并发 decode 正确性。A/B 同时发（新 prompt，64 out）vs 各自串行；环境组 plain / CUT_IN=0 / ATTN_DECODE_OLD=1 / NO_GRAPH=1。
# 另：B max_tokens=1 对照；8 路同时发对照。都在 KV_FP8_STORE=0（当前远端二进制默认开 → 显式关）。
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
R=/w/dinfer/tmp_pair2_RESULT.txt; : > $R
# DONE 只在脚本真正结尾写（trap EXIT 会在 timeout 子进程退出时提前触发，D113）
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=0 FP8_GEMV=1 KV_FP8_STORE=0
PORT=8389; BIN=${BIN:-./target/release/dinfer}
echo "bin=$(stat -c %y $BIN | cut -c1-16) $(date +%T)" >> $R
for ENVSET in "" "CUT_IN=0" "ATTN_DECODE_OLD=1" "NO_GRAPH=1"; do
( # subshell：子进程被杀 / timeout 不影响外层循环（D113）
  LOG=/w/dinfer/tmp_pair2_${ENVSET:-plain}.log
  env $ENVSET taskset -c 0-31 $BIN serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 >$LOG 2>&1 &
  SP=$!
  for i in $(seq 1 150); do sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -q "listening addr" && break; sleep 1; done
  echo "== env=[${ENVSET:-plain}]" >> $R
  timeout 600 python3 - "$PORT" >> $R 2>&1 <<'PY'
import sys, json, urllib.request, concurrent.futures as cf, random
port = sys.argv[1]; url = f"http://127.0.0.1:{port}/v1/completions"
words = "the of and to in is was for on that with as by at from it an be this which or are were but not have has had his her they their one two time year people way day man thing world life hand part child eye woman place work week case point government company number group problem fact".split()
def mk(seed, n):
    r = random.Random(seed); return " ".join(r.choice(words) for _ in range(n))
def one(p, mt=64):
    body = json.dumps({"model":"kaon3","prompt":p,"max_tokens":mt,"temperature":0}).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request(url, body, {"content-type":"application/json"}), timeout=300))["choices"][0]["text"]
def cmp(tag, a, b):
    if a == b: return f"{tag}:SAME"
    k = next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)))
    return f"{tag}:DIFF@{k}"
seed = 7000 + (hash(port) % 100)
# ① 2 路同时发（新 prompt），再串行
for case, (na, nb, mtb) in enumerate([(400, 60, 64), (1500, 400, 64), (400, 400, 64), (400, 400, 1)]):
    pa, pb = mk(seed + case * 2, na), mk(seed + case * 2 + 1, nb)
    with cf.ThreadPoolExecutor(2) as ex:
        fa = ex.submit(one, pa, 64); fb = ex.submit(one, pb, mtb); ca, cb = fa.result(), fb.result()
    sa, sb = one(pa, 64), one(pb, mtb)
    print(f" pair case{case} A={na}w B={nb}w/{mtb}: {cmp('A', sa, ca)} {cmp('B', sb, cb)}")
# ② 8 路同时发（新 prompt 各 400 词）vs 串行
ps = [mk(seed + 100 + i, 400) for i in range(8)]
with cf.ThreadPoolExecutor(8) as ex: conc = list(ex.map(one, ps))
ser = [one(p) for p in ps]
res = [cmp(f"r{i}", s, c) for i, (s, c) in enumerate(zip(ser, conc))]
print(" c8 fresh:", " ".join(res), f"| {sum('DIFF' in r for r in res)}/8 differ")
PY
  kill -INT $SP 2>/dev/null; sleep 2; kill -KILL $SP 2>/dev/null; sleep 2
)
done
echo DONE >> $R
