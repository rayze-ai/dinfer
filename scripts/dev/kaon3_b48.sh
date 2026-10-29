#!/usr/bin/env bash
# b48（D197 ①：MOE_DEC_RMAX 让 R ≤ 512 走 BM16/BK256 dual pipe4）：stage48 入树 → 门禁（R=264：ref(旧体) vs RMAX=512 dump cmp）→ 编译 → envab_cN A=默认(128) / B=MOE_DEC_RMAX=512，c16/32/64。
# 唯一结束标记 B48_DONE。
cd /w/dinfer
cp -f /w/stage48/moe_grouped_sm120.cu crates/kernels/cuda/; cp -f /w/stage48/moe_bk_dump.rs crates/kernels/tests/
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LIBRARY_PATH=/tmp/stub LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_ARCHS=120 CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0
R=/w/dinfer/tmp_b48par_RESULT.txt; : > $R; date +%T >> $R
cargo test --release -p dinfer-kernels --test moe_bk_dump --no-run > /w/dinfer/tmp_b48par_build.log 2>&1; rc=$?; echo "TEST BUILD EXIT=$rc" >> $R
[ $rc -ne 0 ] && { grep -E "^error|error\[" -A8 /w/dinfer/tmp_b48par_build.log | head -30 | cut -c1-200 >> $R; echo B48_DONE >> $R; exit 1; }
rm -f /tmp/moe_bk_*.bin
MOE_BK_MULT=12 MOE_BK_TAG=ref264 MOE_GEMM_PIPE=0 cargo test --release -p dinfer-kernels --test moe_bk_dump -- --ignored --nocapture 2>&1 | grep -E "MOE_BK_DUMP|panicked" | cut -c1-160 >> $R
MOE_BK_MULT=12 MOE_BK_TAG=old264 cargo test --release -p dinfer-kernels --test moe_bk_dump -- --ignored --nocapture 2>&1 | grep -E "MOE_BK_DUMP|panicked" | cut -c1-160 >> $R
MOE_BK_MULT=12 MOE_BK_TAG=rmax264 MOE_DEC_RMAX=512 cargo test --release -p dinfer-kernels --test moe_bk_dump -- --ignored --nocapture 2>&1 | grep -E "MOE_BK_DUMP|panicked" | cut -c1-160 >> $R
gate=0
for t in old264 rmax264; do if cmp -s /tmp/moe_bk_ref264.bin /tmp/moe_bk_$t.bin; then echo "$t vs ref264: SAME" >> $R; else echo "$t vs ref264: DIFF" >> $R; gate=1; fi; done
echo "GATE=$gate" >> $R
[ $gate -ne 0 ] && { echo B48_DONE >> $R; exit 1; }
cargo build --release -p dinfer --features kaon3 > /w/dinfer/tmp_b48_build.log 2>&1; brc=$?; echo "BUILD EXIT=$brc md5=$(md5sum target/release/dinfer | cut -c1-12)" >> $R
[ $brc -ne 0 ] && { grep -E "^error" -A6 /w/dinfer/tmp_b48_build.log | head -20 | cut -c1-200 >> $R; echo B48_DONE >> $R; exit 1; }
cd /w; TAG=b48 AENV="" BENV="MOE_DEC_RMAX=512" bash /w/envab_cN.sh >/w/dinfer/tmp_b48_run.log 2>&1
echo B48_DONE >> $R
