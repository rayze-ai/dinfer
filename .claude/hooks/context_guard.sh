#!/usr/bin/env bash
# 上下文纪律硬约束（PreToolUse）。本项目已 4 次撞 `API Error: 400 Input is too long`。
# 1. 每个会话计工具调用次数：≥ 40 次起每 10 次注入提醒 → 写 progress / 提交 / 建议 /clear。
# 2. 拒绝明显会打爆上下文的调用：Read 大文件不带 limit；Bash 里 cat/git diff/git show/git log -p 不带截断管道。
set -u
IN=$(cat)
j() { printf '%s' "$IN" | python3 -I -c "import sys,json; d=json.load(sys.stdin); v=d$1; print(v if not isinstance(v,(dict,list)) else json.dumps(v))" 2>/dev/null; }
SID=$(j "['session_id']"); TOOL=$(j "['tool_name']")
CNT_FILE="/tmp/claude_ctx_guard_${SID:-none}.cnt"
N=$(( $(cat "$CNT_FILE" 2>/dev/null || echo 0) + 1 )); echo "$N" > "$CNT_FILE"
deny() { printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"%s"}}\n' "$1"; exit 0; }
case "$TOOL" in
  Read)
    FP=$(j "['tool_input']['file_path']"); LIM=$(j "['tool_input'].get('limit','')")
    if [ -z "$LIM" ] && [ -f "$FP" ] && [ "$(wc -l < "$FP")" -gt 200 ]; then
      deny "上下文约束：$FP 超过 200 行，Read 必须带 offset/limit（≤ 200 行），或用 grep -n 定位后只读片段。"
    fi ;;
  Bash)
    CMD=$(j "['tool_input']['command']")
    # 整文件 cat / 不截断的 diff、show、log -p
    if printf '%s' "$CMD" | grep -qE '(^|[;&|] *)(cat|git (diff|show)|git log -p)( |$)' \
       && ! printf '%s' "$CMD" | grep -qE '\|\s*(head|tail|grep|cut|wc|sed -n|awk|tee)|--stat|--name-only|--name-status|--oneline|<<'; then
      deny "上下文约束：cat / git diff / git show 必须经 head/tail/grep/cut 截断（或用 --stat/--name-only）。"
    fi ;;
esac
if [ "$N" -ge 40 ] && [ $(( N % 10 )) -eq 0 ]; then
  MSG="【上下文纪律】本会话已 $N 次工具调用。立即：把当前进展写进 docs/loops 或 docs/progress.md 并 commit；若正等待长任务，用 run_in_background；超过 60 次应向用户建议 /clear 后按交接区恢复。"
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","additionalContext":%s}}\n' "$(printf '%s' "$MSG" | python3 -I -c 'import sys,json;print(json.dumps(sys.stdin.read()))')"
fi
exit 0
