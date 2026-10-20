#!/usr/bin/env bash
# 引擎层面的 dinfer vs vLLM 对比（docs/requirements.md "验收口径 v2" / docs/design.md §15 / docs/loops/engine-bench.md）：开环泊松到达，
# 到达率从低扫到饱和，prompt / 输出长度按分布抽样、不 ignore_eos、一半请求共享长前缀（prefix cache），指标 = SLO 下 goodput、
# TTFT/TPOT 的 p50/p90/p99/max、prefix cache 命中率、失败数；每率同时抓两边服务 /metrics 前后差值（服务端自报的 TTFT/TPOT/cache/步模式）
# 与客户端测得的放同一行对账，并记压测进程自身开销（CPU 核数 / RSS / 解析耗时）。
#   bash scripts/verify/bench_engine.sh [OUT_DIR] ["本轮改动一句话"]    # 默认 test/qwen35/$(date +%F)/engine；改动写 OUT/CHANGE，卡型写 OUT/GPU
# 环境变量（都可改）：
#   SERVE_URL / VLLM_URL   默认 http://127.0.0.1:8012 / :8013；VLLM_URL 置空（VLLM_URL=）或 /v1/models 不通 → 只跑 dinfer 一端
#   RATES="2 4 8 12 16 24"  到达率 req/s           REQS=96   每率请求数（另 +2 预热）        SEED 默认当前秒
#   MIX="128:0.5,1024:0.35,4096:0.15"  prompt 长度分布     OUT_MIX="128:0.5,512:0.35,1024:0.15"  输出 max_tokens 分布
#   PREFIX_REPEAT=0.5  共享长前缀比例      MULTI_TURN=1  每会话轮数（>1 = 真多轮：messages 带历史）
#   SLO_TTFT=500  SLO_TPOT=20  （ms；goodput 判定）     ALLOW_EOS=1（默认不 ignore_eos；=0 则 ignore_eos）
#   PROMPT_LEN=1024  共享前缀的词数（--prompt-len 同时是共享前缀长度）
#   BENCH_CPUS       压测进程钉到的核（taskset 列表，如 "56-63"）；默认 nproc 的最后 8 核。脚本把服务进程（按端口找 pid）与压测进程的
#                    CPU 亲和集合都写进 summary 头，重叠会标 "⚠"——与服务不重叠由使用者保证（服务起时自己 taskset）。
# 产物：OUT/{dinfer,vllm}_r<rate>.{json,log}、OUT/{dinfer,vllm}_r<rate>.metrics.{before,after}（服务 /metrics 整份快照）、OUT/summary.md；
#       并重生成 test/<model>/engine-trend.*（scripts/dev/engine_trend.py）。
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY   # bench(reqwest) 不走代理
cd "$(dirname "$0")/../.."
OUT="${1:-test/qwen35/$(date +%F)/engine}"; mkdir -p "$OUT"
[ -n "${2:-${CHANGE:-}}" ] && printf '%s\n' "${2:-$CHANGE}" > "$OUT/CHANGE"
gpu_tag() { local n; n=$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader 2>/dev/null); case "$n" in *4090*) echo 4090;; *B300*) echo B300;; *3090*) echo 3090;; *) echo "${n// /}";; esac; }
[ -n "$(gpu_tag)" ] && gpu_tag > "$OUT/GPU"
A="${SERVE_URL:-http://127.0.0.1:8012}"
B="${VLLM_URL-http://127.0.0.1:8013}"
RATES="${RATES:-2 4 8 12 16 24}"; REQS="${REQS:-96}"; SEED="${SEED:-$(date +%s)}"
MIX="${MIX:-128:0.5,1024:0.35,4096:0.15}"; OUT_MIX="${OUT_MIX:-128:0.5,512:0.35,1024:0.15}"
PREFIX_REPEAT="${PREFIX_REPEAT:-0.5}"; MULTI_TURN="${MULTI_TURN:-1}"; PROMPT_LEN="${PROMPT_LEN:-1024}"
SLO_TTFT="${SLO_TTFT:-500}"; SLO_TPOT="${SLO_TPOT:-20}"; ALLOW_EOS="${ALLOW_EOS:-1}"
EOS_FLAG=(); [ "$ALLOW_EOS" = 1 ] && EOS_FLAG=(--allow-eos)
BENCH=./target/release/bench
[ -x "$BENCH" ] || cargo build --release -p dinfer-bench
"$BENCH" --self-check > "$OUT/self_check.log" 2>&1 || { tail -3 "$OUT/self_check.log"; echo "bench self-check FAILED"; exit 1; }
echo "self-check PASS: $(grep 'measurement error' "$OUT/self_check.log" | sed 's/^ *//')"
model_of() { curl -s -m 3 "$1/v1/models" 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true; }
SIDES=(dinfer); if [ -n "$B" ] && [ -n "$(model_of "$B")" ]; then SIDES+=(vllm); else echo "vLLM 端不可用（VLLM_URL='${B}'）→ 只跑 dinfer"; fi
[ -n "$(model_of "$A")" ] || { echo "dinfer $A /v1/models 不通"; exit 1; }
vtag=$( { curl -s -m 3 "$B/version" 2>/dev/null | python3 -c 'import sys,json;print("vllm-"+json.load(sys.stdin)["version"])' 2>/dev/null; } || true )

# 服务 /metrics 整份快照（差值由 scripts/dev/metrics_delta.py 算）
snap() { curl -s -m 5 "$1/metrics" > "$2" 2>/dev/null || : > "$2"; }
# 压测进程 CPU 钉核：默认 nproc 的最后 8 核
NCPU=$(nproc); LO=$(( NCPU > 8 ? NCPU - 8 : 0 )); BENCH_CPUS="${BENCH_CPUS:-$LO-$((NCPU - 1))}"
TASKSET=(); command -v taskset >/dev/null && TASKSET=(taskset -c "$BENCH_CPUS")
# 按端口找服务进程 pid → Cpus_allowed_list（容器 --net host 的 vLLM 在宿主机也能看到）
cpus_of_port() { local port=${1##*:} pid; pid=$(ss -ltnpH "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2); if [ -n "$pid" ]; then awk '/Cpus_allowed_list/{print $2}' "/proc/$pid/status" 2>/dev/null || echo "?"; else echo "?"; fi; }
expand() { python3 -c 'import sys;o=set()
for part in sys.argv[1].split(","):
    if "-" in part: a,b=part.split("-"); o.update(range(int(a),int(b)+1))
    elif part.strip().isdigit(): o.add(int(part))
print("\n".join(map(str,sorted(o))))' "$1" 2>/dev/null | sort; }
overlap() { comm -12 <(expand "$1") <(expand "$2") | tr '\n' ' '; }
CPUS_D=$(cpus_of_port "$A"); CPUS_V="?"; [ -n "$B" ] && CPUS_V=$(cpus_of_port "$B")
OV=""; for c in "$CPUS_D" "$CPUS_V"; do if [ "$c" != "?" ] && [ -n "$(overlap "$c" "$BENCH_CPUS")" ]; then OV="⚠ 压测核与服务核重叠：$(overlap "$c" "$BENCH_CPUS")"; fi; done

MD="$OUT/summary.md"
{
  echo "# 引擎层面：dinfer vs vLLM，泊松到达扫描（$(date '+%F %T')，$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader 2>/dev/null)）"
  echo
  echo "dinfer: $A（$(model_of "$A")）；vLLM: ${B:-无}（${vtag:-unknown}）"
  echo "seed=$SEED  每率 $REQS 条（+2 预热）  rates=$RATES"
  echo "负载：泊松开环；prompt mix=$MIX；输出 max_tokens mix=$OUT_MIX，$([ "$ALLOW_EOS" = 1 ] && echo '不 ignore_eos（实际输出 = min(eos, 抽样)）' || echo 'ignore_eos')；prefix_repeat=$PREFIX_REPEAT（共享前缀 $PROMPT_LEN 词 + 16 随机词）；multi_turn=$MULTI_TURN；temperature 0；enable_thinking=false"
  echo "SLO：TTFT ≤ $SLO_TTFT ms 且 TPOT ≤ $SLO_TPOT ms → good；goodput = good/墙钟（vLLM --goodput 同义）。TTFT = 首个带内容 chunk（TTFT1st = 任意首 chunk，对账用）。cache hit = Σcached_tokens/Σprompt_tokens（usage.prompt_tokens_details）。"
  echo "CPU：dinfer 服务进程 $CPUS_D；vLLM 服务进程 $CPUS_V；压测进程 taskset $BENCH_CPUS（nproc=$NCPU）。$OV"
  echo "压测工具自校验：$(grep 'measurement error' "$OUT/self_check.log" | sed 's/^ *//')"
  echo "服务端口径（每率 bench 前后 /metrics 差值，scripts/dev/metrics_delta.py）：srv TTFT/TPOT = 服务自报直方图增量的 mean/p50/p99 ms（桶内插值，只对量级），与客户端列的差 = 网络 + 工具开销；"
  echo "step = dinfer engine_step_execute_{prefill,decode,mixed} count 增量（均值 ms）——**async 调度下量的是含 finish 等待的 host 侧步时，不等于 GPU 步时间**；混批步占比 = M/(P+D+M)。两边单卡默认都是混批（vLLM 单卡无 prefill throttle）。srv cache = 服务自报 prefix_cache hits/queries（dinfer *_total / vllm:prefix_cache_*）；preempt = num_preemptions[_total] 增量。"
  echo "client = 压测进程开销（bench JSON client 段）：平均占用核数 / 峰值 RSS MiB / SSE+JSON 解析 ms 每请求。"
  echo
  echo "| rate | 端 | 完成/失败 | goodput | req/s | out tok/s | TTFT p50/p90/p99/max | TTFT1st p50 | TPOT p50/p90/p99/max | cache hit | len-stop | srv TTFT mean/p50/p99 | srv TPOT mean/p50/p99 | srv cache | preempt | step P/D/M (mean ms) | client cores/RSS/parse |"
  echo "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
} > "$MD"

row() { # json rate side metrics_before metrics_after
  python3 -I scripts/dev/engine_row.py "$1" "$2" "$3" "$4" "$5" >> "$MD"
}

run() { # side url rate
  local side=$1 url=$2 rate=$3
  local f="$OUT/${side}_r${rate}.json" before="$OUT/${side}_r${rate}.metrics.before" after="$OUT/${side}_r${rate}.metrics.after"
  snap "$url" "$before"
  "${TASKSET[@]}" "$BENCH" --url "$url" --label "$side" --request-rate "$rate" --num-requests "$REQS" --warmup 2 --seed "$SEED" \
    --prompt-len "$PROMPT_LEN" --prompt-len-mix "$MIX" --output-len-mix "$OUT_MIX" --prefix-repeat "$PREFIX_REPEAT" --multi-turn "$MULTI_TURN" \
    --slo-ttft-ms "$SLO_TTFT" --slo-tpot-ms "$SLO_TPOT" "${EOS_FLAG[@]}" --out "$f" > "$OUT/${side}_r${rate}.log" 2>&1 \
    || { echo "bench $side rate=$rate failed:"; tail -3 "$OUT/${side}_r${rate}.log"; exit 1; }
  snap "$url" "$after"
  row "$f" "$rate" "$side" "$before" "$after"
  grep -E 'throughput|goodput|TTFT |TPOT|client:' "$OUT/${side}_r${rate}.log" | sed "s/^/  [$side r=$rate] /" | cut -c1-200
}
for r in $RATES; do for side in "${SIDES[@]}"; do
  url=$A; [ "$side" = vllm ] && url=$B
  run "$side" "$url" "$r"
done; done
echo; cat "$MD"
# 趋势：多轮 engine_N 目录汇成 test/<model>/engine-trend.{csv,md,html}
python3 -I scripts/dev/engine_trend.py --root "$(dirname "$(dirname "$OUT")")" | tail -3
