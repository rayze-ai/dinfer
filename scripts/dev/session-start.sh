#!/usr/bin/env bash
# SessionStart hook：每次会话开始把"上次做到哪"注入上下文。
# 这是不依赖 tmux 的跨会话记忆：机器重启、迁移、换工具都不影响。
# stdout 会被 Claude Code 作为上下文附加到会话开头，所以只打印最关键的内容。
set -uo pipefail

ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
PROGRESS="$ROOT/docs/progress.md"
PITFALLS="$ROOT/docs/pitfalls.md"
LOOPS="$ROOT/docs/loops"

[ -f "$PROGRESS" ] || exit 0

echo "<project-skill-resume>"
echo "## 上次交接区（docs/progress.md）"
# 打印从"## 当前状态"到"## 日志"之间的内容
awk '/^## 当前状态/{p=1} /^## 日志/{p=0} p' "$PROGRESS" | sed '/^<!--/d' | head -40

if [ -f "$PITFALLS" ]; then
  echo
  echo "## 最近踩坑（docs/pitfalls.md，仅标题）"
  grep -E '^## ' "$PITFALLS" | head -3 || true
fi

if [ -d "$LOOPS" ]; then
  ACTIVE=$(grep -l '^- 状态：进行中' "$LOOPS"/*.md 2>/dev/null || true)
  if [ -n "$ACTIVE" ]; then
    echo
    echo "## 进行中的 loop 规格"
    for f in $ACTIVE; do echo "- ${f#$ROOT/}"; done
  fi
fi

if command -v git >/dev/null && git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  DIRTY=$(git -C "$ROOT" status --short | wc -l | tr -d ' ')
  echo
  echo "## git：未提交改动 $DIRTY 个文件；最近提交：$(git -C "$ROOT" log --oneline -1 2>/dev/null)"
fi

echo
echo "请先用几行复述现状再动手；结束前执行 /project-skill wrap。"
echo "</project-skill-resume>"
