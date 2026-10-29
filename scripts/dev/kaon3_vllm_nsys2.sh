#!/usr/bin/env bash
# Step A（loop kaon3-prefill-gemm）：在**宿主**跑。用基线镜像起临时容器 vllm-nsys（GPU3），bind kaon3-full 的 nsys 目录，
# 容器内 `nsys profile vllm serve …`（参数照 vllm-noneager），用 kaon3-full 里的 bench 打 c16 同口径，INT 停，报告落 /hf_data/yaochen/dinfer/tmp_vnsys.nsys-rep，
# 再在 kaon3-full 里 export sqlite + 两张 kernel 表。结果 /hf_data/yaochen/dinfer/tmp_vnsys_RESULT.txt，标记 VNSYS_DONE。
# 用法（宿主）：nohup bash /hf_data/yaochen/vllm_nsys2.sh > /hf_data/yaochen/dinfer/tmp_vnsys_run.log 2>&1 &
set -u
W=/hf_data/yaochen; R=$W/dinfer/tmp_vnsys_RESULT.txt; : > $R; date +%T >> $R
IMG=image.paigpu.com/library/vllm-openai:v0.28.0-pegaflow-pr455-51261c6-cu130
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
NSYS_DIR=$(docker exec kaon3-full bash -c 'ls -d /opt/nvidia/nsight-systems/2026*/' | head -1)
# nsys 目录在 kaon3-full 的 overlay 里，宿主路径不直观 → 先拷到 /hf_data/yaochen/nsys（一次，1.2G）
[ -d $W/nsys/target-linux-x64 ] || { docker cp kaon3-full:${NSYS_DIR%/} $W/nsys_tmp && mv $W/nsys_tmp $W/nsys; }
docker rm -f vllm-nsys >/dev/null 2>&1
docker run -d --name vllm-nsys --runtime nvidia --gpus '"device=3"' --ipc=host --net host --cap-add=SYS_ADMIN \
  -v /hf_data/hub:/hf_data/hub:ro -v $W:/w -e HF_HUB_OFFLINE=1 -e VLLM_LOGGING_LEVEL=INFO --entrypoint bash $IMG -c \
  "/w/nsys/target-linux-x64/nsys profile -o /w/dinfer/tmp_vnsys --force-overwrite true -t cuda --cuda-graph-trace=node --cuda-flush-interval=1000 --sample none --cpuctxsw none \
   vllm serve $M --port 28301 --served-model-name kaon-vllm --max-model-len 16384 --max-num-batched-tokens 4096 --block-size 32 --gpu-memory-utilization 0.85 --chat-template $M/chat_template.jinja" >> $R 2>&1
for i in $(seq 1 900); do curl -s -m2 http://127.0.0.1:28301/v1/models 2>/dev/null | grep -q kaon-vllm && break; sleep 2; done
curl -s -m2 http://127.0.0.1:28301/v1/models | grep -q kaon-vllm || { echo "vllm NOTUP" >> $R; docker logs vllm-nsys 2>&1 | tail -8 | cut -c1-200 >> $R; docker rm -f vllm-nsys >/dev/null 2>&1; echo VNSYS_DONE >> $R; exit 1; }
echo "vllm up $(date +%T)" >> $R
docker exec kaon3-full bash -c "cd /w/dinfer && taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:28301 --concurrency 16 --num-requests 32 --prompt-len 400 --max-tokens 16 --prefix-repeat 0 --warmup 0 --seed 1 >/dev/null 2>&1; taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:28301 --concurrency 16 --num-requests 128 --prompt-len 400 --max-tokens 64 --prefix-repeat 0 --warmup 2 --seed 4016 2>&1 | grep -E 'throughput|TTFT |TPOT ' | tr '\n' ' ' | cut -c1-300" >> $R; echo >> $R
# 停：给容器内 vllm 根进程 INT（nsys 等它退出后落盘），最多等 3 min
docker exec vllm-nsys bash -c 'for d in /proc/[0-9]*; do tr "\0" " " < $d/cmdline 2>/dev/null | grep -qE "^python.*vllm serve|^vllm serve|/bin/vllm serve" && kill -INT $(basename $d) 2>/dev/null; done' 2>/dev/null
for i in $(seq 1 90); do docker ps --format '{{.Names}}' | grep -q '^vllm-nsys$' || break; sleep 2; done
docker ps --format '{{.Names}}' | grep -q '^vllm-nsys$' && { docker stop -t 60 vllm-nsys >/dev/null 2>&1; }
ls -la $W/dinfer/tmp_vnsys.nsys-rep | cut -c1-100 >> $R 2>&1
docker rm -f vllm-nsys >/dev/null 2>&1
docker exec kaon3-full bash -c "NSYS=\$(ls /opt/nvidia/nsight-systems/2026*/target-linux-x64/nsys | head -1); \$NSYS export --type sqlite --force-overwrite true -o /w/dinfer/tmp_vnsys.sqlite /w/dinfer/tmp_vnsys.nsys-rep >/dev/null 2>&1; echo '== decode graph kernels'; taskset -c 100-127 python3 /w/nsys_kern.py /w/dinfer/tmp_vnsys.sqlite --all 2>&1 | head -60 | cut -c1-200; echo '== eager (prefill) steps'; taskset -c 100-127 python3 -I /w/nsys_prefill_steps.py /w/dinfer/tmp_vnsys.sqlite 100 30 2 2>&1 | head -60 | cut -c1-170" >> $R 2>&1
echo VNSYS_DONE >> $R
