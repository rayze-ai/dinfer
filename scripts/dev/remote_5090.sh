#!/usr/bin/env bash
# RTX 5090 ×8 机器（sm_120，host-172-16-20-59，JumpServer 跳板）。token 一次性 / 约 2 分钟过期：所有操作复用一条
# 常驻 ControlMaster 连接（`up` 建立，断了需新 token → 向用户要）。用法：
#   scripts/dev/remote_5090.sh up | status | ssh | run '<cmd>' | push | pull <远端> <本地> | put <本地> <远端>
# 凭据：~/.ssh/jms_5090_{user,pass}。不进仓库。
set -euo pipefail
JUMP_HOST="${JUMP_HOST:-jump.aicloud.paigod.work}"
JUMP_PORT="${JUMP_PORT:-2222}"
CM="$HOME/.ssh/cm_5090"
REMOTE_ROOT="${REMOTE_ROOT:-/data/yaochen/infer}"
REMOTE_DIR="${REMOTE_DIR:-$REMOTE_ROOT/dinfer}"
cd "$(dirname "$0")/../.."
USER_AT="$(tr -d '[:space:]' < "$HOME/.ssh/jms_5090_user")@$JUMP_HOST"
BASE=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=25 -p "$JUMP_PORT")
MUX=(-o ControlPath="$CM")

alive() { ssh "${MUX[@]}" -O check "$USER_AT" >/dev/null 2>&1; }
up() {
  alive && { echo "master alive"; return 0; }
  rm -f "$CM"
  SSHPASS="$(cat "$HOME/.ssh/jms_5090_pass")"; export SSHPASS
  setsid nohup sshpass -e ssh "${BASE[@]}" -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 \
    -o ControlMaster=yes -o ControlPath="$CM" -o ControlPersist=yes -o ServerAliveInterval=20 -o ServerAliveCountMax=30 \
    -N "$USER_AT" > /tmp/cm_5090.log 2>&1 < /dev/null &
  for _ in $(seq 1 30); do sleep 1; alive && { echo "master up"; return 0; }; done
  echo "master FAILED (token 过期 / 已用过？向用户要新 token 写入 ~/.ssh/jms_5090_{user,pass}):"; cat /tmp/cm_5090.log; return 1
}

case "${1:-}" in
  up)     up ;;
  status) alive && echo alive || { echo down; exit 1; } ;;
  ssh)    up >/dev/null; exec ssh "${BASE[@]}" "${MUX[@]}" -tt "$USER_AT" ;;
  run)    shift; up >/dev/null; exec ssh "${BASE[@]}" "${MUX[@]}" "$USER_AT" "$*" ;;
  push)   up >/dev/null; exec rsync -az --delete -e "ssh ${BASE[*]} ${MUX[*]}" \
            --exclude target/ --exclude .git/ ./ "$USER_AT:$REMOTE_DIR/" ;;
  pull)   shift; up >/dev/null; exec rsync -az -e "ssh ${BASE[*]} ${MUX[*]}" "$USER_AT:$1" "$2" ;;
  put)    shift; up >/dev/null; exec rsync -az -e "ssh ${BASE[*]} ${MUX[*]}" "$1" "$USER_AT:$2" ;;
  *) echo "usage: $0 up|status|ssh|run '<cmd>'|push|pull <remote> <local>|put <local> <remote>"; exit 2 ;;
esac
