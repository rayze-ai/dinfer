#!/usr/bin/env bash
# b51（D212）：B48b（p64 加 BK=64 实例，down 走大 tile）。编 → 微基准门禁（CHECKSUM）→ envab_c ABAB（A=MOE_PREFILL_P64=0，B=默认）
# → 紧接 b52：同二进制 FP8_LT_TUNE=0/1 ABAB（B49 候选）。结果 tmp_b51_RESULT.txt / tmp_b52_RESULT.txt；标记 B51_DONE / B52_DONE。
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LIBRARY_PATH=/tmp/stub LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_ARCHS=120 CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0
cd /w/dinfer
R=/w/dinfer/tmp_b51_RESULT.txt; : > $R; date +%T >> $R
cargo test --release -p dinfer-kernels --test moe_gemm_timing --no-run > /w/dinfer/tmp_b51_tbuild.log 2>&1; rc=$?; echo "TEST BUILD EXIT=$rc" >> $R
[ $rc -ne 0 ] && { grep -E "error" -A8 /w/dinfer/tmp_b51_tbuild.log | head -40 | cut -c1-200 >> $R; echo B51_DONE >> $R; exit 1; }
for v in 0 1; do
  MOE_PREFILL_P64=$v cargo test --release -p dinfer-kernels --test moe_gemm_timing -- --ignored --nocapture 2>&1 | grep -E "MOE_GEMM_TIMING (P424|P2148|M=8).*(dual|down grouped|CHECKSUM)|panicked" | sed "s/^/P64=$v /" | cut -c1-200 >> $R
done
python3 -I - $R <<'PY' >> $R
import sys,re
ck={}
for l in open(sys.argv[1]):
    m=re.match(r'P64=(\d) MOE_GEMM_TIMING (\S+) rows=\d+ CHECKSUM (.*)',l)
    if m: ck[(m.group(2),m.group(1))]=m.group(3)
bad=[n for n in {k[0] for k in ck} if ck.get((n,'0'))!=ck.get((n,'1'))]
print("GATE="+("0" if not bad and ck else "1")+" mismatch="+",".join(bad))
PY
grep -q "GATE=0" $R || { echo B51_DONE >> $R; exit 1; }
cargo build --release -p dinfer --features kaon3 > /w/dinfer/tmp_b51_build.log 2>&1; brc=$?
echo "BUILD EXIT=$brc md5=$(md5sum target/release/dinfer | cut -c1-12) $(date +%T)" >> $R
[ $brc -ne 0 ] && { grep -E "^error" -A8 /w/dinfer/tmp_b51_build.log | head -30 | cut -c1-200 >> $R; echo B51_DONE >> $R; exit 1; }
TAG=b51ab AENV="MOE_PREFILL_P64=0" BENV="MOE_PREFILL_P64=1" PORT=8346 bash /w/envab_c.sh
cat /w/dinfer/tmp_b51ab_RESULT.txt >> $R
echo B51_DONE >> $R
# b52：FP8_LT_TUNE（m ≥ 64 自调优，只影响 prefill / 混批的 cuBLASLt fp8 GEMM）
TAG=b52 AENV="FP8_LT_TUNE=0" BENV="FP8_LT_TUNE=1" PORT=8347 bash /w/envab_c.sh
