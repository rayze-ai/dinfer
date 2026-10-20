#!/usr/bin/env bash
# 上下文纪律硬约束（PreToolUse）。本项目已 5 次撞 `API Error: 400 Input is too long`（第 5 次 2026-10-18：多段拼接各自合规、合计超限）。
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
    # 按 ; && || 拆段逐段检查：2026-10-18 第 5 次撞 400 就是 "status; diff --stat; diff | head -c 6000; tail -30" 四段各自合规、合计 11 KB。
    # 先剥掉 heredoc 正文（<<TAG ... TAG），正文是数据不是命令；再按 ; && || 拆段
    printf '%s' "$CMD" | python3 -I -c '
import sys,re
c=sys.stdin.read()
c=re.sub(r"<<-?\s*[\x27\"]?(\w+)[\x27\"]?[^\n]*\n.*?\n\1[ \t]*(\n|$)", "<<HEREDOC\n", c, flags=re.S)
print(re.sub(r"(&&|\|\||;)", "\n", c))' | while IFS= read -r SEG || [ -n "$SEG" ]; do
      # cat > / >> 文件（含 heredoc）是写文件不是输出，放行
      if printf '%s' "$SEG" | grep -qE '(^|[|] *)\s*(cat|git (diff|show)|git log -p)( |$)' \
         && ! printf '%s' "$SEG" | grep -qE 'cat\s*>|\|\s*(head|tail|grep|cut|wc|sed -n|awk|tee)|--stat|--name-only|--name-status|--oneline|<<'; then
        echo "DENY1"
      fi
      # 截断量硬上限：head -c > 3000、head/tail -n > 60（无 cut -c 时）都算没截断
      C=$(printf '%s' "$SEG" | grep -oE 'head +-c *[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1)
      [ -n "${C:-}" ] && [ "$C" -gt 3000 ] && echo "DENY2"
      L=$(printf '%s' "$SEG" | grep -oE '(head|tail) +-?n? *-?[0-9]+' | grep -oE '[0-9]+$' | sort -n | tail -1)
      [ -n "${L:-}" ] && [ "$L" -gt 60 ] && ! printf '%s' "$SEG" | grep -qE 'cut -c' && echo "DENY2"
      # sed -n 'A,Bp' 单次跨度 >150 行（2026-10-21）：子代理用 sed -n '1,1626p' 整文件读 gdn_chunk.cu，
      # 74 次调用把上下文堆到 36 万 token，每轮响应从 15 s 退化到 99 s、80 次调用一行代码没写出来。
      printf '%s' "$SEG" | python3 -I -c '
import sys,re
for a,b in re.findall(r"sed\s+-n\s+.\s*([0-9]+)\s*,\s*([0-9]+)\s*p", sys.stdin.read()):
    if int(b)-int(a) > 150:
        print("DENY3"); break
'
    done > /tmp/claude_ctx_guard_${SID:-none}.seg
    grep -q DENY1 /tmp/claude_ctx_guard_${SID:-none}.seg && deny "上下文约束：cat / git diff / git show 每一段都必须经 head/tail/grep/cut 截断（或用 --stat/--name-only）。"
    grep -q DENY2 /tmp/claude_ctx_guard_${SID:-none}.seg && deny "上下文约束：单条命令截断上限 head -c 3000 / head|tail -n 60（超 60 行需配 cut -c）。多段拼接时每段都算，合计别超 8 KB；要看大文件分多次调用。"
    grep -q DENY3 /tmp/claude_ctx_guard_${SID:-none}.seg && deny "上下文约束：sed -n 单次读取跨度上限 150 行。用 grep -n 先定位再看，或分多次小段读；整文件读会把上下文堆爆（2026-10-21 子代理实例：36 万 token、每轮 99 s、零产出）。"
    ;;
esac
if [ "$N" -ge 40 ] && [ $(( N % 10 )) -eq 0 ]; then
  MSG="【上下文纪律】本会话已 $N 次工具调用。立即：把当前进展写进 docs/loops 或 docs/progress.md 并 commit；若正等待长任务，用 run_in_background；超过 60 次应向用户建议 /clear 后按交接区恢复。"
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","additionalContext":%s}}\n' "$(printf '%s' "$MSG" | python3 -I -c 'import sys,json;print(json.dumps(sys.stdin.read()))')"
fi
exit 0
