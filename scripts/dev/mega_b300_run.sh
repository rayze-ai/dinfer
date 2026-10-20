#!/usr/bin/env bash
# mega 实验（docs/loops/mega-sm100.md）在 B300-50 上的一键流程：put 相关文件 → 远端重编 → 默认参数带 prof 跑
# → 参数扫描（只计时）。token 窗口短（~2 h），所以一条命令跑完、日志落 /tmp/mega_b300_<tag>.log，本机只看摘要。
# 用法：scripts/dev/mega_b300_run.sh <tag> [GPU=4] [SWEEP=1]
set -uo pipefail
cd "$(dirname "$0")/../.."
TAG="${1:-run}"; GPU="${2:-4}"; SWEEP="${3:-1}"
R=/data/yaochen/infer/dinfer
export B300_TAG=50
FILES=(crates/kernels/cuda/mega_sm100.cu crates/kernels/cuda/gdn_decode.cuh crates/kernels/cuda/gdn.cu crates/kernels/cuda/kernels.h
       crates/kernels/build.rs crates/kernels/src/ffi.rs crates/kernels/src/arch.rs crates/kernels/src/lib.rs crates/kernels/src/mega.rs
       crates/kernels/tests/mega_parity.rs)
for f in "${FILES[@]}"; do
  timeout 60 scripts/dev/remote_b300.sh put "$f" "$R/$f" 2>&1 | grep -vE "^master" | grep -E "error|code 12" && { echo "put 失败：$f（token 过期？）"; exit 2; }
done
echo "put ok $(date +%H:%M)"
LOG=/tmp/mega_b300_$TAG.log
SWEEP_PARAMS='groups=4,unroll=4,stages=3,ctas=1 groups=4,unroll=4,stages=2,ctas=1 groups=4,unroll=2,stages=4,ctas=1 groups=2,unroll=4,stages=4,ctas=1 groups=2,unroll=4,stages=3,ctas=2 groups=2,unroll=2,stages=4,ctas=2 groups=2,unroll=4,stages=2,ctas=2 groups=4,unroll=4,stages=3,ctas=1,gdn_waves=1 groups=4,unroll=4,stages=3,ctas=1,gdn_waves=4'
timeout 2400 scripts/dev/remote_b300.sh run "source /data/yaochen/infer/env.sh; cd $R && rm -f target/.rustc_info.json && \
  cargo build --release -p dinfer-kernels --features ref-tests --tests 2>&1 | grep -E '^error|-->|Finished|could not compile' | head -20; \
  export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU MEGA_DEBUG=1; \
  echo '== default (prof)'; MEGA_PROF=1 cargo test --release -p dinfer-kernels --features ref-tests --test mega_parity -- --ignored --nocapture 2>&1 \
    | grep -E '^error|invalid|GPU|splits|parity|grid=|speedup_B|stage |^  [a-z_(+0-9]+ |panic|test result' | cut -c1-160; \
  if [ '$SWEEP' = 1 ]; then for P in $SWEEP_PARAMS; do echo \"== \$P\"; MEGA_PARAMS=\$P MEGA_SKIP_PARITY=1 MEGA_ITERS=300 \
    cargo test --release -p dinfer-kernels --features ref-tests --test mega_parity -- --ignored --nocapture 2>&1 \
    | grep -E 'invalid|splits|grid=|speedup_B|panic' | cut -c1-160; done; fi" > "$LOG" 2>&1
echo "exit=$? log=$LOG lines=$(grep -c '' "$LOG")"
grep -E "^==|speedup_B|invalid|panic|^error" "$LOG" | cut -c1-160
