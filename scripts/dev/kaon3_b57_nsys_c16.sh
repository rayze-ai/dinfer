#!/usr/bin/env bash
# b57（D224）：五刀二进制上抓 c16 纯 decode 步（graph node）kernel 表，回答 "M=16 decode 步 8.74 vs M=8 6.56 多在哪"。结果 tmp_nsys_c16d_RESULT.txt。
# decode 图内 kernel 清单（loop kaon3-decode-opt D52）：nsys --cuda-graph-trace=node 抓 c1 decode，
# 导出每个 kernel 的 (名, 次数, 总/均时长)，算一步 decode 的 kernel 数与各段占比。
# 容器内：docker exec -d kaon3-full bash /w/nsys2.sh；结果 /w/dinfer/tmp_nsys_c16d_RESULT.txt
R=/w/dinfer/tmp_nsys_c16d_RESULT.txt; : > $R
NSYS=$(ls /opt/nvidia/nsight-systems/2026*/target-linux-x64/nsys 2>/dev/null | head -1)
if [ -z "$NSYS" ]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get install -y -qq nsight-systems-2026.3.2 >/tmp/ni.log 2>&1
  NSYS=$(ls /opt/nvidia/nsight-systems/2026*/target-linux-x64/nsys 2>/dev/null | head -1)
fi
echo "NSYS=$NSYS" >> $R
[ -z "$NSYS" ] && { echo "install FAIL"; tail -3 /tmp/ni.log; echo DONE; } >> $R && exit 1
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=0 ${EXTRA_ENV:-}
rm -f /w/dinfer/tmp_nsys_c16d.nsys-rep /w/dinfer/tmp_nsys_c16d.sqlite
"$NSYS" profile -o /w/dinfer/tmp_nsys_c16d --force-overwrite true -t cuda --cuda-graph-trace=node --cuda-flush-interval=1000 --sample none --cpuctxsw none \
  ./target/release/dinfer serve --model "$M" --tokenizer "$M" --port 8353 --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 >/w/dinfer/tmp_nsys_c16d_serve.log 2>&1 &
NP=$!
for i in $(seq 1 400); do sed "s/\x1b\[[0-9;]*m//g" /w/dinfer/tmp_nsys_c16d_serve.log | grep -q "listening addr" && break; sleep 1; done
sed "s/\x1b\[[0-9;]*m//g" /w/dinfer/tmp_nsys_c16d_serve.log | grep -q "listening addr" || { echo "serve NOTUP" >> $R; tail -3 /w/dinfer/tmp_nsys_c16d_serve.log >> $R; echo DONE >> $R; kill -9 $NP; exit 1; }
# c1，400 词 prompt，64 out，3 请求（含预热）→ 约 190 个 decode 步
taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:8353 --concurrency 16 --num-requests 64 --prompt-len 400 --max-tokens 64 --prefix-repeat 0 --warmup 1 --seed 7016 2>&1 | grep -E "TPOT " >> $R
kill -INT $(fuser 8353/tcp 2>/dev/null | tr -d " " | head -1) 2>/dev/null; wait $NP 2>/dev/null; sleep 2; ls -la /w/dinfer/tmp_nsys_c16d.nsys-rep >> $R 2>&1
"$NSYS" export --type sqlite --force-overwrite true -o /w/dinfer/tmp_nsys_c16d.sqlite /w/dinfer/tmp_nsys_c16d.nsys-rep >/dev/null 2>&1
taskset -c 100-127 python3 /w/nsys_kern.py /w/dinfer/tmp_nsys_c16d.sqlite --all >> $R 2>&1
echo DONE >> $R
