#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对模型里**每一种** fp8 GEMM 形状，找出与 vLLM 逐位一致的 cuBLASLt algo 下标。

背景（D-17 §31）：algo[2] 只在 L0 `qkv_proj`（m=24,n=8192,k=2816）上定标过。
模型里 120 个 fp8 层的 (n,k) 各不相同，**不同形状命中的"正确 algo"未必都是下标 2**。

做法：抓 vLLM 前向里**所有** fp8 GEMM 的入参（按 (n,k) 去重，每形状留一份），
对每份枚举该形状的启发式候选，与"该次前向的真实输出"逐位比，打印每个形状的答案。

输出形如：  (n, k) -> algo idx=K  diff=0
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

seen = {}  # (n,k) -> dict
orig_mm = FIK.flashinfer_scaled_fp8_mm


def _rec(A, B, out_dtype, scale_a, scale_b, bias=None):
    r = orig_mm(A, B, out_dtype=out_dtype, scale_a=scale_a, scale_b=scale_b, bias=bias)
    key = (int(B.shape[1]), int(A.shape[1]))  # (n, k)
    if key not in seen:
        seen[key] = dict(A=A.detach().clone(), B=B.detach().clone(), scale_a=scale_a.detach().clone(),
                         scale_b=scale_b.detach().clone(), out=r.detach().clone(), m=int(A.shape[0]))
    return r


FIK.flashinfer_scaled_fp8_mm = _rec

llm = LLM(model=model_dir, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.85,
          dtype="auto", max_model_len=4096, enable_prefix_caching=False, block_size=32, max_num_seqs=1)
# 必须清空：LLM 构造期的 profiling run 会跑 m=max_num_batched_tokens(=4096) 的假批量，
# 那些调用会先占住每个 (n,k) 的 key，把真实 prefill（m=24）挡在门外（本轮踩过，读数全 0 且 m 全是 4096）。
seen.clear()
print(f"[shape-scan] 前向一次以抓全部形状 ...", flush=True)
llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)

print(f"[shape-scan] 共 {len(seen)} 种 (n,k) 形状", flush=True)

from flashinfer import bmm_fp8 as bmm_fp8_


def same_bf16(x, y):
    a = x.view(torch.int16).flatten()
    b = y.view(torch.int16).flatten()
    if a.numel() != b.numel():
        return -1
    return int((a != b).sum().item())


for (n, k), c in sorted(seen.items()):
    A, B = c["A"], c["B"]
    m = c["m"]
    # FlashInfer 的 auto 就是 vLLM 的真实路径：作为"该形状的正确输出"
    got = {"auto": None, "cublas": None, "cudnn": None, "cutlass": None}
    for be in list(got):
        try:
            o = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
            bmm_fp8_(A.unsqueeze(0), B.unsqueeze(0), c["scale_a"], c["scale_b"], torch.bfloat16,
                     o.unsqueeze(0), be)
            got[be] = same_bf16(o, c["out"])
        except Exception:  # noqa: BLE001
            got[be] = None
    best = min((v for v in got.values() if v is not None), default=None)
    print(f"  n={n:<6} k={k:<6} m={m:<4} " + " ".join(f"{b}={v}" for b, v in got.items()), flush=True)

print("DONE", flush=True)
