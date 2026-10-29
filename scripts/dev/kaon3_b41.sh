#!/usr/bin/env bash
# b41 链（B40：decode pipe4 BK=256 + K 尾谓词）：stage41 入树 → 门禁（BK dump ×4 进程 cmp）→ moebench（BK 扫）→ bN TAG=b41
# A = 编译前 target（b40 = b39 + 诊断，诊断默认关 ⇒ 等价 b39）；B = BK=256 默认。唯一结束标记 B41_DONE（bN 自身 DONE 在 tmp_b41_RESULT）。
cd /w/dinfer
cp -f /w/stage41/moe_grouped_sm120.cu crates/kernels/cuda/
cp -f /w/stage41/moe_gemm_timing.rs /w/stage41/moe_bk_dump.rs crates/kernels/tests/
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LIBRARY_PATH=/tmp/stub LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_ARCHS=120 CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0
[ -f /w/dinfer_prev_b41 ] || cp -f target/release/dinfer /w/dinfer_prev_b41
R=/w/dinfer/tmp_b41par_RESULT.txt; : > $R; date +%T >> $R
echo "A_md5=$(md5sum /w/dinfer_prev_b41 | cut -c1-12)" >> $R
cargo test --release -p dinfer-kernels --test moe_bk_dump --test moe_gemm_timing --test moe_pipe_parity --no-run > /w/dinfer/tmp_b41par_build.log 2>&1; rc=$?; echo "TEST BUILD EXIT=$rc" >> $R
[ $rc -ne 0 ] && { grep -E "^error|error\[" -A8 /w/dinfer/tmp_b41par_build.log | head -30 | cut -c1-200 >> $R; echo B41_DONE >> $R; exit 1; }
rm -f /tmp/moe_bk_*.bin
MOE_BK_TAG=ref MOE_GEMM_PIPE=0 cargo test --release -p dinfer-kernels --test moe_bk_dump -- --ignored --nocapture 2>&1 | grep -E "MOE_BK_DUMP|panicked" | cut -c1-160 >> $R
for bk in 64 128 256; do MOE_BK_TAG=$bk MOE_GEMM_BK=$bk cargo test --release -p dinfer-kernels --test moe_bk_dump -- --ignored --nocapture 2>&1 | grep -E "MOE_BK_DUMP|panicked" | cut -c1-160 >> $R; done
gate=0
for bk in 64 128 256; do if cmp -s /tmp/moe_bk_ref.bin /tmp/moe_bk_$bk.bin; then echo "BK=$bk vs ref: SAME" >> $R; else echo "BK=$bk vs ref: DIFF" >> $R; gate=1; fi; done
MOE_GEMM_BK=256 cargo test --release -p dinfer-kernels --test moe_pipe_parity -- --ignored --nocapture moe_dual_matches 2>&1 | grep -E "test result|panicked" | cut -c1-120 >> $R
echo "GATE=$gate" >> $R
[ $gate -ne 0 ] && { echo B41_DONE >> $R; echo "PARITY_FAIL" >> /w/dinfer/tmp_b41_RESULT.txt; echo DONE >> /w/dinfer/tmp_b41_RESULT.txt; exit 1; }
bash /w/moebench.sh >/w/dinfer/tmp_moebench2_run.log 2>&1; cp /w/dinfer/tmp_moebench_RESULT.txt /w/dinfer/tmp_moebench2_RESULT.txt
cd /w; TAG=b41 MARKER="MOE_GEMM_BK" BENV="" bash -x /w/bN_keep3.sh >/w/dinfer/tmp_b41_bN_trace.log 2>&1
echo B41_DONE >> $R
