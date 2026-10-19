#!/usr/bin/env bash
# Qwen3 逐算子对拍（A5；docs/requirements.md 第 30 行 "每个 kernel 与 PyTorch 逐算子对拍"）。exit 0 = 全过。
#   1) kernels.sh：torch 生成期望 → ref_tests.rs（已有覆盖的算子不重写，一起跑）
#   2) tests/ops_parity.rs：随机输入 → 跑 kernel → dump .npy 到 target/ops_parity/
#   3) ops_parity_ref.py：torch 参考 + 容差，打印一张汇总表（算子 / 形状 / 最大误差 / 通过），并把 1) 的结果并入
# 环境：SKIP_REF_TESTS=1 跳过 1)；OPS_PARITY_DIR 改 dump 目录；GPU_WAIT_S 等其它 dinfer 进程释放显存的上限（默认 600）。
set -uo pipefail
cd "$(dirname "$0")/../.."

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
DUMP="$(realpath -m "${OPS_PARITY_DIR:-target/ops_parity}")"
REF_LOG=/tmp/dinfer_kernels.log
PARITY_LOG=/tmp/dinfer_ops_parity.log

step() { rm -f target/.rustc_info.json; printf '\n==> %s\n' "$*"; }

# 其它 dinfer 进程占着 GPU 时等待（每 30 s 一次，最多 GPU_WAIT_S）
wait_gpu() {
  local waited=0 lim="${GPU_WAIT_S:-600}"
  while :; do
    local busy
    busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read -r p; do
      [ -n "$p" ] && tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q dinfer && echo "$p"; done | head -1)
    [ -z "$busy" ] && return 0
    [ "$waited" -ge "$lim" ] && { echo "GPU 仍被 dinfer 进程 $busy 占用，超过 ${lim}s，继续尝试"; return 0; }
    echo "GPU 被 dinfer 进程 $busy 占用，等待 30 s（已等 ${waited}s）"; sleep 30; waited=$((waited + 30))
  done
}

ref_rc=0
if [ "${SKIP_REF_TESTS:-0}" != "1" ]; then
  step "1/3 ref_tests.rs（kernels.sh）"
  wait_gpu
  scripts/verify/kernels.sh >/tmp/dinfer_kernels_sh.log 2>&1 || ref_rc=$?
  grep -E '^test |test result' "$REF_LOG" 2>/dev/null | tail -3
  [ "$ref_rc" -ne 0 ] && echo "kernels.sh exit $ref_rc（详见 $REF_LOG）"
else
  REF_LOG=/dev/null
fi

step "2/3 ops_parity.rs → $DUMP"
rm -rf "$DUMP"; mkdir -p "$DUMP"
wait_gpu
OPS_PARITY_DIR="$DUMP" cargo test -p dinfer-kernels --test ops_parity -- --ignored --test-threads=1 </dev/null 2>&1 \
  | tee "$PARITY_LOG" | grep -E '^test |test result|panicked|error' | cut -c1-200
dump_rc=${PIPESTATUS[0]}
grep -qE 'test result: ok\. [1-9]' "$PARITY_LOG" || { echo "ops_parity dump 失败（$PARITY_LOG）"; exit 1; }

step "3/3 torch 参考 + 容差"
python3 -I scripts/verify/ops_parity_ref.py --dir "$DUMP" --ref-log "$REF_LOG"
py_rc=$?

echo
if [ "$py_rc" -eq 0 ] && [ "$ref_rc" -eq 0 ] && [ "$dump_rc" -eq 0 ]; then
  echo "ops_parity PASS"; exit 0
fi
echo "ops_parity FAIL (ref_tests=$ref_rc dump=$dump_rc compare=$py_rc)"; exit 1
