#!/usr/bin/env bash
# b55（D214）：B48d p64 STAGES=4。编 → 微基准（MOE_PREFILL_BN=64/128 两进程：P424/P2148 us + CHECKSUM 门禁）→ envab_c ABAB（A=BN64 默认，B=BN=128）。
# 结果 tmp_b55_RESULT.txt，标记 B55_DONE。前置：b52 已结束（GPU0 空）。
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LIBRARY_PATH=/tmp/stub LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_ARCHS=120 CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0
cd /w/dinfer
R=/w/dinfer/tmp_b55_RESULT.txt; : > $R; date +%T >> $R
cargo test --release -p dinfer-kernels --test moe_gemm_timing --no-run > /w/dinfer/tmp_b55_tbuild.log 2>&1; rc=$?; echo "TEST BUILD EXIT=$rc" >> $R
[ $rc -ne 0 ] && { grep -E "error" -A8 /w/dinfer/tmp_b55_tbuild.log | head -40 | cut -c1-200 >> $R; echo B55_DONE >> $R; exit 1; }
for v in 3 4; do
  MOE_PREFILL_STAGES=$v cargo test --release -p dinfer-kernels --test moe_gemm_timing -- --ignored --nocapture 2>&1 | grep -E "MOE_GEMM_TIMING (P424|P2148|M=8).*(dual|down grouped|CHECKSUM)|panicked" | sed "s/^/ST=$v /" | cut -c1-200 >> $R
done
python3 -I - $R <<'PY' >> $R
import sys,re
ck={}
for l in open(sys.argv[1]):
    m=re.match(r'ST=(\d+) MOE_GEMM_TIMING (\S+) rows=\d+ CHECKSUM (.*)',l)
    if m: ck[(m.group(2),m.group(1))]=m.group(3)
bad=[n for n in {k[0] for k in ck} if ck.get((n,"3"))!=ck.get((n,"4"))]
print("GATE="+("0" if not bad and ck else "1")+" mismatch="+",".join(bad))
PY
grep -q "GATE=0" $R || { echo B55_DONE >> $R; exit 1; }
cargo build --release -p dinfer --features kaon3 > /w/dinfer/tmp_b55_build.log 2>&1; brc=$?
echo "BUILD EXIT=$brc md5=$(md5sum target/release/dinfer | cut -c1-12) marker=$(strings target/release/dinfer | grep -c MOE_PREFILL_STAGES) $(date +%T)" >> $R
[ $brc -ne 0 ] && { grep -E "^error" -A8 /w/dinfer/tmp_b55_build.log | head -30 | cut -c1-200 >> $R; echo B55_DONE >> $R; exit 1; }
TAG=b55ab AENV="MOE_PREFILL_STAGES=3" BENV="MOE_PREFILL_STAGES=4" PORT=8351 bash /w/envab_c.sh
grep -E "^r[12] [AB] (c|out|startup)|md5" /w/dinfer/tmp_b55ab_RESULT.txt >> $R
echo B55_DONE >> $R
