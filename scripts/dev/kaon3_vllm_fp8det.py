#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D-17b：vLLM 的 fp8 GEMM 算子自身是否**逐位确定**，并和它自己的前向输出比。

修 D-17 的类型 bug：`flashinfer_scaled_fp8_mm` 的 scale_a/scale_b 必须是 **0 维 tensor**
（内部 `assert scale_a.numel() == 1`），我上一轮传了 Python float。

判定三件事（全部同入参、同进程、同 GPU）：
  1. **确定性**：用抓到的同一份入参重复调用 N 次，输出互相是否逐位相同；
  2. **自复现**：重复调用的输出 vs 该次前向里那份**真实返回值**（vLLM 自己算的），是否逐位相同；
  3. **变体下限**：同入参下 flashinfer vs cutlass 的差异（复核 D-13）。

跑法（容器内）：
  OUT=/w/hid/d17b MODEL=<snapshot> python3 -I scripts/dev/kaon3_vllm_fp8det.py $(cat test/kaon3/oracle_clean/p02.ids)
"""
import os
import sys

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import numpy as np
import torch

import vllm.model_executor.kernels.linear.scaled_mm.flashinfer as FIK
import vllm.model_executor.models.gemma4 as g4
from vllm import LLM, SamplingParams

model_dir = os.environ["MODEL"]
ids = [int(x) for x in sys.argv[1:]]
out = os.environ["OUT"]
os.makedirs(out, exist_ok=True)
N_REPEAT = int(os.environ.get("N_REPEAT", "6"))

captured = []
orig_mm = FIK.flashinfer_scaled_fp8_mm


def _rec(A, B, out_dtype, scale_a, scale_b, bias=None):
    r = orig_mm(A, B, out_dtype=out_dtype, scale_a=scale_a, scale_b=scale_b, bias=bias)
    if len(captured) < 1:
        captured.append(
            dict(A=A.detach().clone(), B=B.detach().clone(), out_dtype=out_dtype,
                 scale_a=scale_a.detach().clone(), scale_b=scale_b.detach().clone(),
                 bias=None if bias is None else bias.detach().clone(),
                 fwd=r.detach().clone(), shapeA=tuple(A.shape), shapeB=tuple(B.shape))
        )
    return r


FIK.flashinfer_scaled_fp8_mm = _rec


def _cmp(x, y, label):
    """逐位比较 bf16/fp8 张量，返回 (不同元素数, max abs diff in f32)。"""
    a = x.flatten().view(torch.int16)
    b = y.flatten().view(torch.int16)
    if a.shape != b.shape:
        print(f"       {label}: 形状不同 {tuple(x.shape)} vs {tuple(y.shape)}", flush=True)
        return -1, None
    n = int((a != b).sum().item())
    mx = float((x.float().flatten() - y.float().flatten()).abs().max().item()) if n else 0.0
    print(f"       {label}: diff {n}/{a.numel()}  max {mx:.4e}", flush=True)
    return n, mx


llm = LLM(model=model_dir, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.85,
          dtype="auto", max_model_len=4096, enable_prefix_caching=False, block_size=32, max_num_seqs=1)
captured.clear()

print("[d17b] 前向一次以抓样本 ...", flush=True)
llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)
if not captured:
    print("[d17b] FAIL 没抓到 flashinfer 调用", flush=True)
    sys.exit(2)

c = captured[0]
A, B, sa, sb = c["A"], c["B"], c["scale_a"], c["scale_b"]
print(f"[d17b] A={c['shapeA']} stride={A.stride()}  B={c['shapeB']} stride={B.stride()}", flush=True)
print(f"[d17b] scale_a={sa.tolist()} ({sa.dtype})  scale_b={sb.tolist()} ({sb.dtype}) out={c['out_dtype']}", flush=True)

# 存盘，供 Rust 侧灌同一份字节
np.ascontiguousarray(A.view(torch.uint8).cpu().numpy()).tofile(f"{out}/A_fp8.bin")
np.ascontiguousarray(B.view(torch.uint8).cpu().numpy()).tofile(f"{out}/B_fp8.bin")
# 参考输出存 **f32**：bf16 值精确扩展，Rust 侧 `bf16::to_f32()` 比得上且不受二次舍入影响
np.ascontiguousarray(c["fwd"].float().cpu().numpy().astype(np.float32)).tofile(f"{out}/ref_f32.bin")
with open(f"{out}/meta.txt", "w") as f:
    f.write(f"shapeA={c['shapeA']} shapeB={c['shapeB']} strideB={B.stride()} "
            f"scale_a={float(sa)} scale_b={float(sb)} out={c['out_dtype']}\n")

# ---- 1. 确定性：同入参重复 N 次 ----
print(f"[d17b] === 1. flashinfer 同入参重复 {N_REPEAT} 次（自身确定性）===", flush=True)
outs = []
for _ in range(N_REPEAT):
    outs.append(orig_mm(A, B, out_dtype=c["out_dtype"], scale_a=sa, scale_b=sb, bias=c["bias"]).detach().clone())
det = True
for i in range(1, N_REPEAT):
    n, _ = _cmp(outs[0], outs[i], f"run0 vs run{i}")
    det = det and (n == 0)
print(f"[d17b] 确定性: {'DETERMINISTIC（逐位一致）' if det else 'NON-DETERMINISTIC'}", flush=True)

# ---- 2. 自复现：重复调用 vs 那次前向的真实返回值 ----
print("[d17b] === 2. 重新调用 vs 前向里的真实返回值（vLLM 自复现）===", flush=True)
n_self, mx_self = _cmp(outs[0], c["fwd"], "run0 vs 前向真实值")

# ---- 3. 变体下限：cutlass ----
print("[d17b] === 3. flashinfer vs cutlass（同入参）===", flush=True)
try:
    from vllm._custom_ops import cutlass_scaled_mm

    oc = cutlass_scaled_mm(A, B.t().contiguous().t(), sa, sb, out_dtype=torch.bfloat16).detach().clone()
    _cmp(outs[0], oc, "flashinfer vs cutlass")
    _cmp(oc, c["fwd"], "cutlass vs 前向真实值")
except Exception as e:  # noqa: BLE001
    print(f"       cutlass 跳过: {e}", flush=True)

print(f"[d17b] 入参/参考输出已存 {out}/", flush=True)
print("[d17b] DONE", flush=True)
