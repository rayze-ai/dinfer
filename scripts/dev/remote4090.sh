#!/usr/bin/env bash
# 4090 测试机（JumpServer 跳板，临时账号）。用法：
#   scripts/dev/remote4090.sh ssh                 # 交互登录
#   scripts/dev/remote4090.sh run '<cmd>'         # 远端执行一条命令
#   scripts/dev/remote4090.sh push                # rsync 仓库到远端 $REMOTE_DIR（排除 target/、test/ 大文件）
#   scripts/dev/remote4090.sh pull <远端路径> <本地路径>
# 凭据：JumpServer 的 token 用户名放 $JMS_USER_FILE（默认 ~/.ssh/jms_4090_user），密码放 $JMS_PASS_FILE
# （默认 ~/.ssh/jms_4090_pass，chmod 600）。账号过期后只需更新这两个文件。不要把密码写进仓库。
set -euo pipefail
JUMP_HOST="${JUMP_HOST:-jump.aicloud.paigod.work}"
JUMP_PORT="${JUMP_PORT:-2222}"
JMS_USER_FILE="${JMS_USER_FILE:-$HOME/.ssh/jms_4090_user}"
JMS_PASS_FILE="${JMS_PASS_FILE:-$HOME/.ssh/jms_4090_pass}"
REMOTE_DIR="${REMOTE_DIR:-/data/code/dinfer}"
cd "$(dirname "$0")/../.."

[ -r "$JMS_USER_FILE" ] || { echo "missing $JMS_USER_FILE (JumpServer token 用户名，形如 JMS-xxxx)"; exit 2; }
[ -r "$JMS_PASS_FILE" ] || { echo "missing $JMS_PASS_FILE (token 密码)"; exit 2; }
USER_AT="$(tr -d '[:space:]' < "$JMS_USER_FILE")@$JUMP_HOST"
export SSHPASS; SSHPASS="$(tr -d '\n' < "$JMS_PASS_FILE")"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -p "$JUMP_PORT")

case "${1:-}" in
  ssh)  exec sshpass -e ssh -tt "${SSH_OPTS[@]}" "$USER_AT" ;;
  run)  shift; exec sshpass -e ssh -tt "${SSH_OPTS[@]}" "$USER_AT" "$*; exit" 2>&1 | sed 's/\x1b\[[0-9;?]*[a-zA-Z]//g' ;;
  push)
    exec sshpass -e rsync -az --delete -e "ssh ${SSH_OPTS[*]}" \
      --exclude target/ --exclude .git/ --exclude 'test/kernel_refs/' --exclude 'test/qwen3_golden/' --exclude 'test/tiny_glm*/' \
      ./ "$USER_AT:$REMOTE_DIR/" ;;
  pull) shift; exec sshpass -e rsync -az -e "ssh ${SSH_OPTS[*]}" "$USER_AT:$1" "$2" ;;
  *) sed -n 2,8p "$0"; exit 2 ;;
esac
