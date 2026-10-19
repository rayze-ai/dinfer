#!/usr/bin/env bash
# B300 机器（8 × B300 SXM6 NVLink，JumpServer 跳板）。JumpServer 连接 token **一次性 / 当天过期**：所有操作复用一条
# 常驻 ControlMaster 连接（`up` 建立，断了自动重建；重建需要新 token → 向用户要）。用法同 remote4090.sh：
#   scripts/dev/remote_b300.sh up | status | ssh | run '<cmd>' | push | pull <远端> <本地>
# 凭据：token 用户名 ~/.ssh/jms_b300_user；token 密码 ~/.ssh/jms_b300_pass。不进仓库。远端只用 $REMOTE_ROOT。
set -euo pipefail
JUMP_HOST="${JUMP_HOST:-111.48.159.177}"
JUMP_PORT="${JUMP_PORT:-2222}"
CM="$HOME/.ssh/cm_b300"
REMOTE_ROOT="${REMOTE_ROOT:-/data/yaochen/infer}"   # 用户约束（2026-10-19）：远程只能用这个目录
REMOTE_DIR="${REMOTE_DIR:-$REMOTE_ROOT/dinfer}"
cd "$(dirname "$0")/../.."
USER_AT="$(tr -d '[:space:]' < "$HOME/.ssh/jms_b300_user")@$JUMP_HOST"
BASE=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=25 -p "$JUMP_PORT")
MUX=(-o ControlPath="$CM")

alive() { ssh "${MUX[@]}" -O check "$USER_AT" >/dev/null 2>&1; }
up() {
  alive && { echo "master alive"; return 0; }
  rm -f "$CM"
  SSHPASS="$(cat "$HOME/.ssh/jms_b300_pass")"; export SSHPASS
  setsid nohup sshpass -e ssh "${BASE[@]}" -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 \
    -o ControlMaster=yes -o ControlPath="$CM" -o ControlPersist=yes -o ServerAliveInterval=20 -o ServerAliveCountMax=30 \
    -N "$USER_AT" > /tmp/cm_b300.log 2>&1 < /dev/null &
  for _ in $(seq 1 30); do sleep 1; alive && { echo "master up"; return 0; }; done
  echo "master FAILED (token 过期 / 已用过？向用户要新 token 写入 ~/.ssh/jms_b300_user 与 jms_b300_pass):"; cat /tmp/cm_b300.log; return 1
}

case "${1:-}" in
  up)     up ;;
  status) alive && echo alive || { echo down; exit 1; } ;;
  ssh)    up >/dev/null; exec ssh "${BASE[@]}" "${MUX[@]}" -tt "$USER_AT" ;;
  run)    shift; up >/dev/null; exec ssh "${BASE[@]}" "${MUX[@]}" "$USER_AT" "$*" ;;
  push)   up >/dev/null; exec rsync -az --delete -e "ssh ${BASE[*]} ${MUX[*]}" \
            --exclude target/ --exclude .git/ --exclude 'test/qwen3_golden/' --exclude 'test/*fixture*' --exclude 'test/**/bench*/' --exclude 'test/**/ab_*/' --exclude 'test/**/stage*/' \
            ./ "$USER_AT:$REMOTE_DIR/" ;;
  pull)   shift; up >/dev/null; exec rsync -az -e "ssh ${BASE[*]} ${MUX[*]}" "$USER_AT:$1" "$2" ;;
  *) echo "usage: $0 up|status|ssh|run '<cmd>'|push|pull <remote> <local>"; exit 2 ;;
esac
