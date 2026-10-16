#!/usr/bin/env bash
# 4090 测试机（JumpServer 跳板）。所有操作复用一条常驻 ControlMaster 连接（`up` 建立，断了自动重建）。
#   scripts/dev/remote4090.sh up            # 建立/确认常驻连接（公钥 ~/.ssh/id_ed25519_gpu4090，密码兜底）
#   scripts/dev/remote4090.sh status        # 连接是否活着
#   scripts/dev/remote4090.sh ssh           # 交互登录
#   scripts/dev/remote4090.sh run '<cmd>'   # 远端执行
#   scripts/dev/remote4090.sh push          # rsync 仓库到 $REMOTE_DIR（排除 target/、.git/、golden、fixture、以及远端产出的 bench*/ stage/ perf-trend.*）
#   scripts/dev/remote4090.sh pull <远端> <本地>
# 凭据：token 用户名 ~/.ssh/jms_4090_user；密码 ~/.ssh/jms_4090_pass（仅公钥失败时用）。不进仓库。
set -euo pipefail
JUMP_HOST="${JUMP_HOST:-jump.aicloud.paigod.work}"
JUMP_PORT="${JUMP_PORT:-2222}"
KEY="${JMS_KEY:-$HOME/.ssh/id_ed25519_gpu4090}"
CM="$HOME/.ssh/cm_4090"
REMOTE_ROOT="${REMOTE_ROOT:-/data/yaochen/infer}"   # 用户约束：远程只能用这个目录
REMOTE_DIR="${REMOTE_DIR:-$REMOTE_ROOT/dinfer}"
cd "$(dirname "$0")/../.."
USER_AT="$(tr -d '[:space:]' < "$HOME/.ssh/jms_4090_user")@$JUMP_HOST"
BASE=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -p "$JUMP_PORT")
MUX=(-o ControlPath="$CM")

alive() { ssh "${MUX[@]}" -O check "$USER_AT" >/dev/null 2>&1; }
up() {
  alive && { echo "master alive"; return 0; }
  rm -f "$CM"
  SSHPASS="$(cat "$HOME/.ssh/jms_4090_pass" 2>/dev/null || true)"; export SSHPASS
  setsid nohup sshpass -e ssh "${BASE[@]}" -o IdentitiesOnly=yes -i "$KEY" -o PreferredAuthentications=publickey,password \
    -o ControlMaster=yes -o ControlPath="$CM" -o ControlPersist=yes -o ServerAliveInterval=20 -o ServerAliveCountMax=30 \
    -N "$USER_AT" > /tmp/cm_4090.log 2>&1 < /dev/null &
  for _ in $(seq 1 30); do sleep 1; alive && { echo "master up"; return 0; }; done
  echo "master FAILED:"; cat /tmp/cm_4090.log; return 1
}

case "${1:-}" in
  up)     up ;;
  status) alive && echo alive || { echo down; exit 1; } ;;
  ssh)    up >/dev/null; exec ssh "${BASE[@]}" "${MUX[@]}" -tt "$USER_AT" ;;
  run)    shift; up >/dev/null; exec ssh "${BASE[@]}" "${MUX[@]}" "$USER_AT" "$*" ;;
  push)   up >/dev/null; exec rsync -az --delete -e "ssh ${BASE[*]} ${MUX[*]}" \
            --exclude target/ --exclude .git/ --exclude 'test/kernel_refs/' --exclude 'test/qwen3_golden/' --exclude 'test/tiny_glm*/' --exclude 'test/qwen3/*/bench*/' --exclude 'test/qwen3/*/stage/' --exclude 'test/qwen3/perf-trend.*' \
            ./ "$USER_AT:$REMOTE_DIR/" ;;
  pull)   shift; up >/dev/null; exec rsync -az -e "ssh ${BASE[*]} ${MUX[*]}" "$USER_AT:$1" "$2" ;;
  *) sed -n 2,9p "$0"; exit 2 ;;
esac
