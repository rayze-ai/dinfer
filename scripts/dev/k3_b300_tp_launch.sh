#!/usr/bin/env bash
# K3 C-5 B300 launcher：$1 = single | tp2 | build | smoke。全部 nohup 后台，日志在 $K。
source /data/yaochen/infer/env.sh
cd /data/yaochen/infer/dinfer
K=/data/yaochen/infer/bootstrap/k3tp; mkdir -p "$K"
EXE=$(ls -t target/release/deps/model_parity-* | grep -v '\.d$' | head -1)
COMMON="KIMIK3_DIR=/data/models/moonshotai/Kimi-K3 KIMIK3_REFS=test/kimik3/2026-10-20/model KIMIK3_LAYERS=5 RUST_LOG=info CUDA_DEVICE_ORDER=PCI_BUS_ID"
case "$1" in
  single) nohup bash -c "$COMMON CUDA_VISIBLE_DEVICES=2 KIMIK3_TP=1 KIMIK3_DUMP=$K/single $EXE --ignored --nocapture; echo SINGLE_RC=\$?" > "$K/single.log" 2>&1 < /dev/null & ;;
  tp2)    nohup bash -c "$COMMON CUDA_VISIBLE_DEVICES=2,3 KIMIK3_TP=2 KIMIK3_DUMP=$K/tp2 $EXE --ignored --nocapture; echo TP2_RC=\$?" > "$K/tp2.log" 2>&1 < /dev/null & ;;
  build)  nohup bash -c "cargo build --release -p dinfer -p dinfer-bench --features kimik3 2>&1 | grep -E '^error|Finished|Blocking'; echo BIN2_RC=\${PIPESTATUS[0]}; echo BUILD2_DONE" > "$K/build2.log" 2>&1 < /dev/null & ;;
  smoke)  nohup bash -c "K3_GPU=2,3 K3_TP=2 OUT=$K/smoke5_tp2 bash scripts/dev/k3_b300_smoke.sh; echo SMOKE_RC=\$?" > "$K/smoke.log" 2>&1 < /dev/null & ;;
  *) echo "usage: single|tp2|build|smoke"; exit 2 ;;
esac
echo "LAUNCHED $1 exe=$EXE $(date +%T)"
