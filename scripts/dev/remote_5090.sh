#!/usr/bin/env bash
# RTX 5090 ×8 机器（sm_120，host-172-16-20-59，JumpServer 跳板）。token 一次性 / 约 2 分钟过期：所有操作复用一条
# 常驻 ControlMaster 连接（`up` 建立，断了需新 token → 向用户要）。用法：
#   scripts/dev/remote_5090.sh up | status | ssh | run '<cmd>' | push | sync | pull <远端> <本地> | put <本地> <远端>
#   GPU=0 scripts/dev/remote_5090.sh ctr '<cmd>'   # 在基线镜像容器里跑（cwd=/w/dinfer，含 stub 软链接 / LD_LIBRARY_PATH / 代理）
#   GPU 7 是基线服务，ctr 拒绝。宿主无 nvcc / rust，编译与 GPU 运行都走 ctr。
#   sync = 把本地 crates/ models/ bins/ scripts/ test/kaon3/ Cargo.* 增量推到远端
#          （**不带 --delete**，本地为准；SYNC_DRY=1 只列差异）。
# 跳板会间歇拒绝新会话：打印 "new ssh client failed: ssh: handshake failed: EOF" 且 rc=0（命令没到远端），
# run / ctr 识别这句后重放 stdin 重试，rsync 按非零 rc 重试（2026-10-22 实测约 1/4 会话失败）。
# 凭据：~/.ssh/jms_5090_{user,pass}。不进仓库。
set -euo pipefail
JUMP_HOST="${JUMP_HOST:-jump.aicloud.paigod.work}"
JUMP_PORT="${JUMP_PORT:-2222}"
CM="$HOME/.ssh/cm_5090"
REMOTE_ROOT="${REMOTE_ROOT:-/hf_data/yaochen}"
REMOTE_DIR="${REMOTE_DIR:-$REMOTE_ROOT/dinfer}"
cd "$(dirname "$0")/../.."
USER_AT="$(tr -d '[:space:]' < "$HOME/.ssh/jms_5090_user")@$JUMP_HOST"
BASE=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=25 -p "$JUMP_PORT")
MUX=(-o ControlPath="$CM")
IMAGE="${IMAGE:-image.paigpu.com/library/vllm-openai:v0.28.0-pegaflow-pr455-51261c6-cu130}"
NV=/usr/local/lib/python3.12/dist-packages/nvidia
# 容器前置：镜像只有 libnccl.so.2 / libnvrtc.so.13，链接要未版本化软链接；运行期要 LD_LIBRARY_PATH，否则 127。
CTR_PRE="set -e; mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; \
ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so; export LIBRARY_PATH=/tmp/stub; \
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib:\${LD_LIBRARY_PATH:-}; set +e"

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
# sshr <远端命令>：SSHR_IN=1 时 stdin 整体缓存（Bash 工具的 stdin 不会 EOF，不能默认读），跳板拒绝时重放重试；成功后原样输出 stdout / stderr、返回远端 rc。
sshr() {
  local in="" rc=0 i o e
  if [ -n "${SSHR_IN:-}" ]; then in=$(cat; printf x); in=${in%x}; fi
  o=$(mktemp); e=$(mktemp)
  for i in 1 2 3 4 5 6 7 8; do
    rc=0; printf '%s' "$in" | ssh "${BASE[@]}" "${MUX[@]}" "$USER_AT" "$1" >"$o" 2>"$e" || rc=$?
    if grep -qs '^new ssh client failed' "$o" "$e"; then sleep $((i < 4 ? 2 : 5)); continue; fi
    break
  done
  cat "$o"; cat "$e" >&2; rm -f "$o" "$e"; return $rc
}
rsyncr() {
  local i
  for i in 1 2 3 4 5 6 7 8; do rsync -az -e "ssh ${BASE[*]} ${MUX[*]}" "$@" 2>/tmp/rsync_5090.err && return 0; sleep 3; done
  cat /tmp/rsync_5090.err >&2; return 1
}
ctr() {
  local gpu="${GPU:-0}" f=".ctr_$$_$RANDOM.sh"
  [ "$gpu" = 7 ] && { echo "GPU 7 是基线服务，禁止占用"; exit 2; }
  printf '%s\n%s\n' "$CTR_PRE" "$*" | SSHR_IN=1 sshr "cat > $REMOTE_ROOT/$f && docker run --rm --entrypoint bash --gpus device=$gpu --ipc=host \
      -v $REMOTE_ROOT:/w -v /hf_data/hub:/hf_data/hub:ro -v /root/.cargo:/root/.cargo -v /root/.rustup:/root/.rustup \
      -w /w/dinfer -e PATH=/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
      -e CUDA_ARCHS=120 -e CUDA_HOME=/usr/local/cuda \
      -e https_proxy=http://172.17.0.1:1081 -e http_proxy=http://172.17.0.1:1081 \
      $IMAGE /w/$f; rc=\$?; rm -f $REMOTE_ROOT/$f; exit \$rc"
}

case "${1:-}" in
  up)     up ;;
  status) alive && echo alive || { echo down; exit 1; } ;;
  ssh)    up >/dev/null; exec ssh "${BASE[@]}" "${MUX[@]}" -tt "$USER_AT" ;;
  run)    shift; up >/dev/null; sshr "$*" ;;
  runin)  shift; up >/dev/null; SSHR_IN=1 sshr "$*" ;;   # stdin 转给远端命令
  ctr)    shift; up >/dev/null; ctr "$*" ;;
  push)   up >/dev/null; exec rsync -az --delete -e "ssh ${BASE[*]} ${MUX[*]}" \
            --exclude target/ --exclude .git/ ./ "$USER_AT:$REMOTE_DIR/" ;;
  sync)   up >/dev/null; rsyncr ${SYNC_DRY:+-n -i} -R --exclude target/ crates models bins scripts test/kaon3 Cargo.toml Cargo.lock "$USER_AT:$REMOTE_DIR/" ;;
  pull)   shift; up >/dev/null; rsyncr "$USER_AT:$1" "$2" ;;
  put)    shift; up >/dev/null; rsyncr "$1" "$USER_AT:$2" ;;
  *) echo "usage: $0 up|status|ssh|run '<cmd>'|ctr '<cmd>'|push|sync|pull <remote> <local>|put <local> <remote>"; exit 2 ;;
esac
