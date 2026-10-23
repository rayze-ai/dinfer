#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读出 vLLM 的 fp8 GEMM 在**本模型本形状**上命中的具体 tactic（决定 K 方向累加序）。

背景：实测该 GEMM 有唯一确定输出（跨进程逐位一致），我们与它差 8/196608 个 1-ulp，
差异纯属累加树形状。要复刻，必须先知道它选了哪套 tile/unroll/split 配置。

做法：包一层 `flashinfer.bmm_fp8`，先跑一次前向把 autotune 定型，然后
  1. 打印该形状的 tactic 选择（autotuner 内部状态）；
  2. 逐个枚举候选 config，用同一入参逐一执行，看哪个输出与"已被选中的"输出逐位相同
     —— 命中者就是我们要复刻的那套累加序。
"""
import os
import sys

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch

import vllm.model_executor.kernels.linear.scaled_mm.flashinfer as FIK
import vllm.model_executor.models.gemma4 as g4
from vllm import LLM, SamplingParams

model_dir = os.environ["MODEL"]
ids = [int(x) for x in sys.argv[1:]]

captured = []
orig_mm = FIK.flashinfer_scaled_fp8_mm


def _rec(A, B, out_dtype, scale_a, scale_b, bias=None):
    r = orig_mm(A, B, out_dtype=out_dtype, scale_a=scale_a, scale_b=scale_b, bias=bias)
    if not captured:
        captured.append(dict(A=A.detach().clone(), B=B.detach().clone(), out_dtype=out_dtype,
                             scale_a=scale_a.detach().clone(), scale_b=scale_b.detach().clone(),
                             fwd=r.detach().clone()))
    return r


FIK.flashinfer_scaled_fp8_mm = _rec

llm = LLM(model=model_dir, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.85,
          dtype="auto", max_model_len=4096, enable_prefix_caching=False, block_size=32, max_num_seqs=1)
captured.clear()
llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)

if not captured:
    print("FAIL 没抓到调用", flush=True)
    sys.exit(2)
c = captured[0]
A, B = c["A"], c["B"]
print(f"形状 A={tuple(A.shape)} B={tuple(B.shape)} scale_a={float(c['scale_a'])} scale_b={float(c['scale_b'])}",
      flush=True)

# ---- 1. 看 autotuner 对该形状的 tactic ----
try:
    from flashinfer.autotuner import autotune

    print(f"autotune 对象: {autotune!r}", flush=True)
    for attr in ("tuned", "_tuned", "cache", "_cache"):
        if hasattr(autotune, attr):
            v = getattr(autotune, attr)
            print(f"  autotune.{attr} = {str(v)[:1200]}", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"读 autotuner 状态失败: {e}", flush=True)

# ---- 2. 枚举 bmm_fp8 的候选配置，找输出与"已选中输出"逐位相同的那个 ----
from flashinfer import bmm_fp8 as bmm_fp8_


def run_with(backend="auto"):
    """按 vLLM 的调用形态（unsqueeze(0) 成 3D）跑一次同入参。"""
    out = torch.empty(A.shape[0], B.shape[1], dtype=torch.bfloat16, device=A.device)
    try:
        return bmm_fp8_(A.unsqueeze(0), B.unsqueeze(0), c["scale_a"], c["scale_b"],
                        torch.bfloat16, out.unsqueeze(0), backend)
    except Exception as e:  # noqa: BLE001
        return ("ERR", str(e)[:160])


def same(x, y):
    a = x.view(torch.int16).flatten()
    b = y.view(torch.int16).flatten()
    return int((a != b).sum().item())


print("=== 基线（当前 tactic）vs 前向真实值 ===", flush=True)
base = run_with()
if isinstance(base, tuple):
    print("基线失败:", base, flush=True)
    sys.exit(3)
print(f"  diff {same(base, c['fwd'])}/{base.numel()}", flush=True)

print("=== 枚举 backend（真实签名：cudnn/cublas/cutlass/cute-dsl/auto）===", flush=True)
for be in ["auto", "cudnn", "cublas", "cutlass", "cute-dsl"]:
    r = run_with(backend=be)
    if isinstance(r, tuple):
        print(f"  {be:<10} 不可用: {r[1][:110]}", flush=True)
    else:
        print(f"  {be:<10} diff vs 前向 {same(r, c['fwd'])}/{r.numel()}   vs base {same(r, base)}", flush=True)

print("DONE", flush=True)
