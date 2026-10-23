#!/usr/bin/env bash
# 判定 FlashInfer 的 fp8 GEMM 在**跨进程**时是否逐位一致。
# 两次独立加载模型（各自 JIT autotune）跑同一份输入，比较该层 GEMM 的输出。
# 结论决定"与 vLLM 逐位对齐"这个目标是否良定义。
set -u
cd /w/dinfer || exit 1
MODEL=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
IDS=$(cat test/kaon3/oracle_clean/p02.ids)

for r in r1 r2; do
  rm -rf /root/.cache/vllm/flashinfer_autotune_cache
  echo "=== 进程 $r ==="
  OUT=/w/d17_$r MODEL=$MODEL python3 -I scripts/dev/kaon3_vllm_fp8det.py $IDS 2>&1 \
    | grep -aE '确定性|自复现|vs 前向真实值|DONE'
done

echo "=== 两个独立进程的该 GEMM 输出逐位对比 ==="
python3 -I - <<'PY'
import numpy as np
a = np.fromfile('/w/d17_r1/fwd_ref_bf16.bin', dtype=np.int16)
b = np.fromfile('/w/d17_r2/fwd_ref_bf16.bin', dtype=np.int16)
if a.size != b.size:
    print('SIZE MISMATCH', a.size, b.size)
else:
    d = (a != b)
    print('cross-process diff', int(d.sum()), '/', a.size)
PY
echo "=== END ==="
