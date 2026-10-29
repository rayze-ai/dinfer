#!/usr/bin/env bash
# Step A v4（loop kaon3-decode-opt D199）：逐环节对标 vLLM。ROUNDS 轮 × c ∈ {1,2,4,8,16}（主）+ {32,64}（附加），
# 每窗口 dinfer / vLLM 交错、同窗口同 seed、跨窗口换 seed（seed=1000*r+c）。dinfer 非 eager + HOST_TIME=1；
# 每窗口记：bench 三指标、nvidia-smi 采样（dinfer GPU0 / vLLM GPU2）、dinfer host 各段均值、两端 /metrics 增量。
# 钉核：serve 0-31、bench 60-91。结果 /w/dinfer/tmp_${TAG}_RESULT.txt，唯一结束标记 STEPA4_DONE。
# 用法（容器内）：docker exec -d kaon3-full bash -c 'bash /w/stepA4.sh > /w/dinfer/tmp_stepA4_run.log 2>&1'
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
TAG=${TAG:-stepA4}; PORT=${PORT:-8331}; ROUNDS=${ROUNDS:-5}; VPORT=${VPORT:-28201}; VGPU=${VGPU:-2}
CS=${CS:-"1 2 4 8 16 32 64"}
R=/w/dinfer/tmp_${TAG}_RESULT.txt; LOG=/w/dinfer/tmp_${TAG}_serve.log; : > $R
strip() { sed "s/\x1b\[[0-9;]*m//g"; }
echo "serve_taskset=0-31 bench_taskset=60-91 vllm=container(GPU$VGPU) $(date +%T)" >> $R
echo "bin=$(stat -c %y target/release/dinfer | cut -c1-16) md5=$(md5sum target/release/dinfer | cut -c1-12) probes=$(strings target/release/dinfer | grep -cE 'kaon3-host|begin-split|render-time')" >> $R
echo "vllm_models=$(curl -s -m5 http://127.0.0.1:$VPORT/v1/models | python3 -c 'import sys,json;print([m["id"] for m in json.load(sys.stdin)["data"]])' 2>&1)" >> $R
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=0 HOST_TIME=1 ${EXTRA_ENV:-}
T0=$(date +%s.%N)
taskset -c 0-31 ./target/release/dinfer serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 >$LOG 2>&1 &
SP=$!
for i in $(seq 1 180); do strip < $LOG | grep -q "listening addr" && break; sleep 1; done
strip < $LOG | grep -q "listening addr" || { echo NOTUP >> $R; strip < $LOG | grep -iE "error|panic" | tail -3 | cut -c1-200 >> $R; echo STEPA4_DONE >> $R; kill -9 $SP; exit 1; }
echo "startup: wall=$(awk -v a=$T0 -v b=$(date +%s.%N) "BEGIN{printf \"%.1f\", b-a}")s $(strip < $LOG | grep -oE 'model_load_ms=[0-9]+' | head -1) $(strip < $LOG | grep -oiE 'num_blocks[=: ]+[0-9]+' | head -1)" >> $R
echo "dinfer_models=$(curl -s -m5 http://127.0.0.1:$PORT/v1/models | python3 -c 'import sys,json;print([m["id"] for m in json.load(sys.stdin)["data"]])' 2>&1)" >> $R

# /metrics 快照 → 单行 k=v（只取 _total / _sum / _count 计数器）
snap() { curl -s -m5 "$1/metrics" | python3 -I -c '
import sys,re
out={}
for l in sys.stdin:
    if l.startswith("#"): continue
    m=re.match(r"([a-zA-Z_:]+)(\{[^}]*\})?\s+([0-9.eE+-]+)",l)
    if not m: continue
    k,lab,v=m.group(1),m.group(2) or "",float(m.group(3))
    if k.startswith("dinfer:"): k=k[7:]
    if "_bucket" in k or "_created" in k: continue
    if k.endswith(("_total","_sum","_count")): out[k]=out.get(k,0)+v
print(" ".join(f"{k}={v:.6g}" for k,v in sorted(out.items())))'; }
# 两快照差 → 关心的字段
diffm() { python3 -I - "$1" "$2" <<'PY'
import sys
def parse(s): return {k:float(v) for k,v in (x.split("=") for x in s.split() if "=" in x)}
a,b=parse(sys.argv[1]),parse(sys.argv[2]); d={k:b[k]-a.get(k,0) for k in b}
def g(k): return d.get(k,0.0)
o=[]
if "vllm:decode_steps_total" in d:
    ds,ps=g("vllm:decode_steps_total"),g("vllm:prefill_steps_total")
    n=g("vllm:request_success_total") or 1
    o.append(f"steps prefill={ps:.0f} decode={ds:.0f}")
    o.append(f"per_req queue={1e3*g('vllm:request_queue_time_seconds_sum')/n:.1f}ms prefill={1e3*g('vllm:request_prefill_time_seconds_sum')/n:.1f}ms decode={1e3*g('vllm:request_decode_time_seconds_sum')/n:.1f}ms infer={1e3*g('vllm:request_inference_time_seconds_sum')/n:.1f}ms")
    if ds: o.append(f"decode_step={1e3*g('vllm:request_decode_time_seconds_sum')/ds:.2f}ms(req-sum/steps)")
    sc=g("vllm:scheduler_time_seconds_sum"); scn=g("vllm:scheduler_time_seconds_count") or 1
    o.append(f"sched={1e6*sc/scn:.0f}us/call n={scn:.0f}")
    it=g("vllm:iteration_tokens_total_sum"); itn=g("vllm:iteration_tokens_total_count") or 1
    o.append(f"iter_tokens_mean={it/itn:.1f}")
    o.append(f"read_bytes={g('vllm:estimated_read_bytes_per_gpu_total')/1e9:.1f}GB gen_tok={g('vllm:generation_tokens_total'):.0f} prefix_hit={g('vllm:prefix_cache_hits_total'):.0f}/{g('vllm:prefix_cache_queries_total'):.0f} preempt={g('vllm:num_preemptions_total'):.0f}")
else:
    st=g("engine_steps_total") or 1; n=g("request_success_total") or 1
    o.append(f"steps={st:.0f} step_ms={1e3*g('engine_step_seconds_sum')/st:.2f} sched={1e6*g('engine_step_schedule_seconds_sum')/st:.0f}us exec={1e3*g('engine_step_execute_seconds_sum')/st:.2f}ms")
    o.append(f"per_req ttft={1e3*g('time_to_first_token_seconds_sum')/n:.1f}ms e2e={1e3*g('e2e_request_latency_seconds_sum')/n:.1f}ms tpot={1e3*g('time_per_output_token_seconds_sum')/(g('time_per_output_token_seconds_count') or 1):.2f}ms")
    o.append(f"gen_tok={g('generation_tokens_total'):.0f} prompt_tok={g('prompt_tokens_total'):.0f} prefix_hit={g('prefix_cache_hits_total'):.0f}/{g('prefix_cache_queries_total'):.0f} preempt={g('num_preemptions_total'):.0f}")
print(" | ".join(o))
PY
}
smi_start() { nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits -i $1 -lms 200 > /tmp/smi_$TAG.txt 2>/dev/null & echo $!; }
smi_stop() { kill $1 2>/dev/null; wait $1 2>/dev/null; awk -F, 'NR>3{s+=$1;m+=$2;n++} END{if(n) printf "sm=%.0f%% mem=%.0f%% n=%d", s/n, m/n, n}' /tmp/smi_$TAG.txt; }
bench() { taskset -c 60-91 ./target/release/bench --url $1 --concurrency $2 --num-requests $(( $2 * 8 )) --prompt-len 400 --max-tokens 64 --prefix-repeat 0 --warmup 2 --seed $3 2>&1 | grep -E "throughput|TTFT |TPOT |client" | tr '\n' ' ' | tr -s ' ' | cut -c1-400; }

# 预热：每个 graph 桶都捕获过
bench http://127.0.0.1:$PORT 16 1 >/dev/null
for r in $(seq 1 $ROUNDS); do
  for c in $CS; do
    seed=$((1000 * r + c))
    echo "=== round=$r c=$c seed=$seed $(date +%T) ===" >> $R
    LO=$(wc -l < $LOG); S0=$(snap http://127.0.0.1:$PORT); P=$(smi_start 0)
    echo "  [dinfer] $(bench http://127.0.0.1:$PORT $c $seed)" >> $R
    echo "  [dinfer-gpu] $(smi_stop $P)" >> $R
    HI=$(wc -l < $LOG); S1=$(snap http://127.0.0.1:$PORT)
    echo "r$r c$c dinfer S0 $S0" >> /w/dinfer/tmp_${TAG}_snaps.txt; echo "r$r c$c dinfer S1 $S1" >> /w/dinfer/tmp_${TAG}_snaps.txt
    echo "  [dinfer-metrics] $(diffm "$S0" "$S1")" >> $R
    echo "  [dinfer-host] $(python3 -I /w/kaon3_hosttime_parse.py $LOG $LO $HI 2>&1 | cut -c1-900)" >> $R
    V0=$(snap http://127.0.0.1:$VPORT); P=$(smi_start $VGPU)
    echo "  [vllm] $(bench http://127.0.0.1:$VPORT $c $seed)" >> $R
    echo "  [vllm-gpu] $(smi_stop $P)" >> $R
    V1=$(snap http://127.0.0.1:$VPORT)
    echo "r$r c$c vllm S0 $V0" >> /w/dinfer/tmp_${TAG}_snaps.txt; echo "r$r c$c vllm S1 $V1" >> /w/dinfer/tmp_${TAG}_snaps.txt
    echo "  [vllm-metrics] $(diffm "$V0" "$V1")" >> $R
  done
done
kill -INT $SP 2>/dev/null; sleep 2; kill -KILL $SP 2>/dev/null
echo STEPA4_DONE >> $R
