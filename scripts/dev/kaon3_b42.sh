#!/usr/bin/env bash
# b42 链（B41：decode pipe4 BN_T=32 四 warp CTA；默认仍 64，B 侧用 env MOE_GEMM_BN=32）：stage42 入树 → 门禁（ref/256/bn32 dump cmp + dual parity）→ moebench（BN 扫）→ bN TAG=b42
# A = 编译前 target（b41 二进制）；B = 新二进制 + env MOE_GEMM_BN=32（默认不变，所以 A/B 二进制差异只在新模板实例）。唯一结束标记 B42_DONE（bN 自身 DONE 在 tmp_b42_RESULT）。
cd /w/dinfer
cp -f /w/stage42/moe_grouped_sm120.cu crates/kernels/cuda/
cp -f /w/stage42/moe_gemm_timing.rs /w/stage42/moe_bk_dump.rs crates/kernels/tests/
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LIBRARY_PATH=/tmp/stub LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_ARCHS=120 CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0
[ -f /w/dinfer_prev_b42 ] || cp -f target/release/dinfer /w/dinfer_prev_b42
R=/w/dinfer/tmp_b42par_RESULT.txt; : > $R; date +%T >> $R
echo "A_md5=$(md5sum /w/dinfer_prev_b42 | cut -c1-12)" >> $R
cargo test --release -p dinfer-kernels --test moe_bk_dump --test moe_gemm_timing --test moe_pipe_parity --no-run > /w/dinfer/tmp_b42par_build.log 2>&1; rc=$?; echo "TEST BUILD EXIT=$rc" >> $R
[ $rc -ne 0 ] && { grep -E "^error|error\[" -A8 /w/dinfer/tmp_b42par_build.log | head -30 | cut -c1-200 >> $R; echo B42_DONE >> $R; exit 1; }
rm -f /tmp/moe_bk_*.bin
MOE_BK_TAG=ref MOE_GEMM_PIPE=0 cargo test --release -p dinfer-kernels --test moe_bk_dump -- --ignored --nocapture 2>&1 | grep -E "MOE_BK_DUMP|panicked" | cut -c1-160 >> $R
for bk in 256; do MOE_BK_TAG=$bk MOE_GEMM_BK=$bk cargo test --release -p dinfer-kernels --test moe_bk_dump -- --ignored --nocapture 2>&1 | grep -E "MOE_BK_DUMP|panicked" | cut -c1-160 >> $R; done
gate=0
MOE_BK_TAG=bn32 MOE_GEMM_BN=32 cargo test --release -p dinfer-kernels --test moe_bk_dump -- --ignored --nocapture 2>&1 | grep -E "MOE_BK_DUMP|panicked" | cut -c1-160 >> $R
for bk in 256 bn32; do if cmp -s /tmp/moe_bk_ref.bin /tmp/moe_bk_$bk.bin; then echo "BK=$bk vs ref: SAME" >> $R; else echo "BK=$bk vs ref: DIFF" >> $R; gate=1; fi; done
MOE_GEMM_BN=32 cargo test --release -p dinfer-kernels --test moe_pipe_parity -- --ignored --nocapture moe_dual_matches 2>&1 | grep -E "test result|panicked" | cut -c1-120 >> $R
echo "GATE=$gate" >> $R
[ $gate -ne 0 ] && { echo B42_DONE >> $R; echo "PARITY_FAIL" >> /w/dinfer/tmp_b42_RESULT.txt; echo DONE >> /w/dinfer/tmp_b42_RESULT.txt; exit 1; }
bash /w/moebench.sh >/w/dinfer/tmp_moebench3_run.log 2>&1; cp /w/dinfer/tmp_moebench_RESULT.txt /w/dinfer/tmp_moebench3_RESULT.txt
cd /w; TAG=b42 MARKER="MOE_GEMM_BN" BENV="MOE_GEMM_BN=32" bash -x /w/bN_keep3.sh >/w/dinfer/tmp_b42_bN_trace.log 2>&1
echo B42_DONE >> $R
