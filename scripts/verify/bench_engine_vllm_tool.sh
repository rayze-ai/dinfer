#!/usr/bin/env bash
# 第三方口径交叉验证：用 vLLM 自带的 `vllm bench serve`（vllm/benchmarks/serve.py）分别打 dinfer :8012 与 vLLM :8013，
# 用来核我们 `bins/bench` 的数字与第三方工具一致（同 rate 下 p50/p99 差 < 10%）。**在 B300 上跑**：bench 在 vLLM 容器里执行
# （容器 --net host，可直接打宿主机端口），本机没有 vllm 环境、也没有容器，不要在本机跑。
#   bash scripts/verify/bench_engine_vllm_tool.sh [OUT_DIR] ["一句话"]     # 默认 test/qwen35/$(date +%F)/engine_vllm_tool
# 环境变量：
#   CONTAINER=vllm-qwen35-b300  nerdctl 容器名（k8s.io 命名空间；sudo -n）     MODEL=Qwen3.8-27B（served-model-name）
#   TOKENIZER=/data/models/Qwen/Qwen3.8-27B（容器内路径，/data/models 只读挂载）
#   RATES="4 8 16"   REQS=96   IN_LEN=1024  OUT_LEN=256  PREFIX_LEN=512（random 数据集：所有请求共享 PREFIX_LEN 个随机 token 前缀）
#   SLO_TTFT=500 SLO_TPOT=20（--goodput ttft:.. tpot:..）   SEED 默认当前秒   PORTS="8012 8013"（端口 → 标签 dinfer / vllm）
# 口径差异（对账时注意，详见 docs/loops/engine-bench.md）：
#   - vllm bench serve 的 TTFT = 任意首个 SSE 事件（dinfer 首 chunk 是空 role delta → 它测出 ~5 ms 伪像）；对应我们 JSON 里的
#     ttft_first_chunk_ms，而我们的 ttft_ms = 首个带内容 chunk。对账 TTFT 时用 ttft_first_chunk_ms。
#   - 它的输出长度固定 OUT_LEN、默认 ignore_eos=False（random 数据集给的 max_tokens，模型可能提前 eos）；prompt 长度固定 IN_LEN+PREFIX_LEN，
#     不是分布；所以只在"同 rate、同长度"的点上对账，用 `bench --prompt-len $((IN_LEN+PREFIX_LEN)) --max-tokens OUT_LEN --prefix-repeat 1`
#     跑我们的一端做对照（本脚本末尾会打印这条命令）。
#   - goodput 定义与我们一致（DistServe：TTFT ≤ 且 TPOT ≤ 的请求数 / 墙钟；输出 ≤ 1 token 的 TPOT 记 0）。
# 产物：OUT/<label>_r<rate>.json（vllm bench serve --save-result 的 JSON，含 p50/p99 与 request_goodput）、OUT/<label>_r<rate>.log、OUT/summary.md
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-test/qwen35/$(date +%F)/engine_vllm_tool}"; mkdir -p "$OUT"
[ -n "${2:-}" ] && printf '%s\n' "$2" > "$OUT/CHANGE"
CONTAINER="${CONTAINER:-vllm-qwen35-b300}"; MODEL="${MODEL:-Qwen3.8-27B}"; TOKENIZER="${TOKENIZER:-/data/models/Qwen/Qwen3.8-27B}"
RATES="${RATES:-4 8 16}"; REQS="${REQS:-96}"; IN_LEN="${IN_LEN:-1024}"; OUT_LEN="${OUT_LEN:-256}"; PREFIX_LEN="${PREFIX_LEN:-512}"
SLO_TTFT="${SLO_TTFT:-500}"; SLO_TPOT="${SLO_TPOT:-20}"; SEED="${SEED:-$(date +%s)}"; PORTS="${PORTS:-8012 8013}"
EXEC=(sudo -n nerdctl -n k8s.io exec "$CONTAINER")
"${EXEC[@]}" true 2>/dev/null || { echo "容器 $CONTAINER 不在跑（sudo -n nerdctl -n k8s.io ps）"; exit 1; }
label_of() { case "$1" in 8012) echo dinfer;; 8013) echo vllm;; *) echo "p$1";; esac; }
ABS_OUT="$(realpath "$OUT")"
# 容器内写不到宿主机 OUT：结果先写容器 /tmp，再 nerdctl cp 出来
MD="$OUT/summary.md"
{
  echo "# vllm bench serve 交叉验证（$(date '+%F %T')）"; echo
  echo "容器 $CONTAINER；model=$MODEL；random 数据集 in=$IN_LEN + prefix=$PREFIX_LEN，out=$OUT_LEN（不 ignore_eos）；seed=$SEED；每率 $REQS 条；goodput ttft:$SLO_TTFT tpot:$SLO_TPOT"
  echo "注意：此工具的 TTFT = 任意首 chunk（对应我们的 ttft_first_chunk_ms）。"; echo
  echo "| rate | 端 | 完成 | goodput req/s | req/s | out tok/s | TTFT mean/p50/p99 (ms) | TPOT mean/p50/p99 (ms) | ITL p50/p99 (ms) |"
  echo "|---|---|---|---|---|---|---|---|---|"
} > "$MD"
for r in $RATES; do for port in $PORTS; do
  lab=$(label_of "$port"); f="${lab}_r${r}.json"
  "${EXEC[@]}" bash -c "cd /tmp && vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --base-url http://127.0.0.1:$port \
      --model '$MODEL' --tokenizer '$TOKENIZER' --dataset-name random --random-input-len $IN_LEN --random-output-len $OUT_LEN --random-prefix-len $PREFIX_LEN \
      --request-rate $r --num-prompts $REQS --seed $SEED --goodput ttft:$SLO_TTFT tpot:$SLO_TPOT --percentile-metrics ttft,tpot,itl,e2el \
      --metric-percentiles 50,90,99 --save-result --result-dir /tmp/bench_out --result-filename '$f'" > "$OUT/${lab}_r${r}.log" 2>&1 \
    || { echo "vllm bench serve $lab rate=$r failed:"; tail -5 "$OUT/${lab}_r${r}.log"; exit 1; }
  sudo -n nerdctl -n k8s.io cp "$CONTAINER:/tmp/bench_out/$f" "$ABS_OUT/$f"
  python3 - "$OUT/$f" "$r" "$lab" >> "$MD" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); rate, lab = sys.argv[2], sys.argv[3]
g = d.get("request_goodput"); g = "-" if g is None else f"{g:.2f}"
f = lambda k: f"{d.get(k, float('nan')):.1f}"
print(f"| {rate} | {lab} | {d.get('completed')}/{d.get('num_prompts')} | {g} | {d.get('request_throughput', 0):.2f} | {d.get('output_throughput', 0):.0f} | "
      f"{f('mean_ttft_ms')}/{f('p50_ttft_ms')}/{f('p99_ttft_ms')} | {f('mean_tpot_ms')}/{f('p50_tpot_ms')}/{f('p99_tpot_ms')} | {f('p50_itl_ms')}/{f('p99_itl_ms')} |")
PY
  grep -E "Request throughput|goodput|Mean TTFT|P99 TTFT|Mean TPOT|P99 TPOT" "$OUT/${lab}_r${r}.log" | sed "s/^/  [$lab r=$r] /"
done; done
echo; cat "$MD"

# ---- 对账：我们的 bench 在同端、同 rate、同 seed、同长度（prompt = IN_LEN+PREFIX_LEN 词、全部共享前缀 ≈ random-prefix-len、max_tokens = OUT_LEN、
# 不 ignore_eos）各跑一组，比 p50 / p99：TTFT 用 ttft_first_chunk_ms（vllm 工具口径 = 任意首 chunk），TPOT 用 tpot_ms。
# 差 = (ours − tool) / tool，> THRESH%（默认 10）标 ⚠。OURS=0 跳过。压测进程 taskset 到 BENCH_CPUS（默认 nproc 最后 8 核）。
OURS="${OURS:-1}"; THRESH="${THRESH:-10}"
if [ "$OURS" = 1 ]; then
  BENCH=./target/release/bench; [ -x "$BENCH" ] || cargo build --release -p dinfer-bench
  NCPU=$(nproc); LO=$(( NCPU > 8 ? NCPU - 8 : 0 )); BENCH_CPUS="${BENCH_CPUS:-$LO-$((NCPU - 1))}"
  TASKSET=(); command -v taskset >/dev/null && TASKSET=(taskset -c "$BENCH_CPUS")
  {
    echo; echo "## 对账：bins/bench（同 rate / seed / 长度，taskset $BENCH_CPUS）vs vllm bench serve；差 = (ours − tool) / tool，阈值 ${THRESH}%"; echo
    echo "| rate | 端 | TTFT p50 ours/tool (差) | TTFT p99 ours/tool (差) | TPOT p50 ours/tool (差) | TPOT p99 ours/tool (差) | req/s ours/tool | prompt tok ours/tool |"
    echo "|---|---|---|---|---|---|---|---|"
  } >> "$MD"
  for r in $RATES; do for port in $PORTS; do
    lab=$(label_of "$port"); f="$OUT/ours_${lab}_r${r}.json"
    "${TASKSET[@]}" "$BENCH" --url "http://127.0.0.1:$port" --label "$lab" --request-rate "$r" --num-requests "$REQS" --warmup 2 --seed "$SEED" \
      --prompt-len $((IN_LEN + PREFIX_LEN)) --max-tokens "$OUT_LEN" --prefix-repeat 1 --allow-eos --slo-ttft-ms "$SLO_TTFT" --slo-tpot-ms "$SLO_TPOT" \
      --out "$f" > "$OUT/ours_${lab}_r${r}.log" 2>&1 || { echo "ours $lab rate=$r failed:"; tail -3 "$OUT/ours_${lab}_r${r}.log"; exit 1; }
    python3 -I scripts/dev/engine_xcheck_row.py "$f" "$OUT/${lab}_r${r}.json" "$r" "$lab" "$THRESH" >> "$MD"
  done; done
  echo; tail -n +"$(grep -n '^## 对账' "$MD" | cut -d: -f1)" "$MD"
  echo "（⚠ = 差 > ${THRESH}%：先核两边 prompt 实际 token 数与 len-stop 比例是否一致，再怀疑计时口径。）"
fi
