#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为模型里每一种 fp8 GEMM 形状，标定出**与 vLLM 逐位一致**的 cuBLASLt algo 下标。

为什么需要：cuBLAS 对全部 6 种 (n,k) 都能给出 0 差异（形状扫描已证），
但**具体是哪一个启发式 algo** 因形状而异（我们的 Lt 默认第 0 个在 (8192,2816) 上差 8 处）。
写死某个下标会在别的形状上失效（踩过）。

做法：抓 vLLM 前向里每种 (n,k) 的真实入参 → 对每种形状枚举 flashinfer 侧能复现的 backend，
并把该形状的 (n,k,auto 输出) 存盘，供 Rust 侧标定 algo 下标。

输出：/w/d17b/shapes/<n>x<k>.{A_fp8,B_fp8,ref_bf16}.bin + meta.txt
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
out = os.environ.get("OUT", "/w/d17b/shapes")
os.makedirs(out, exist_ok=True)

seen = {}
orig_mm = FIK.flashinfer_scaled_fp8_mm


def _rec(A, B, out_dtype, scale_a, scale_b, bias=None):
    r = orig_mm(A, B, out_dtype=out_dtype, scale_a=scale_a, scale_b=scale_b, bias=bias)
    key = (int(B.shape[1]), int(A.shape[1]))  # (n, k)
    if key not in seen:
        seen[key] = dict(A=A.detach().clone(), B=B.detach().clone(),
                         scale_a=scale_a.detach().clone(), scale_b=scale_b.detach().clone(),
                         out=r.detach().clone(), m=int(A.shape[0]))
    return r


FIK.flashinfer_scaled_fp8_mm = _rec

llm = LLM(model=model_dir, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.85,
          dtype="auto", max_model_len=4096, enable_prefix_caching=False, block_size=32, max_num_seqs=1)
seen.clear()  # 必须清：profile run 的假批量（m=4096）会抢占 key
llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)

print(f"[calib] 抓到 {len(seen)} 种 (n,k)", flush=True)
for (n, k), c in sorted(seen.items()):
    d = f"{out}/{n}x{k}"
    os.makedirs(d, exist_ok=True)
    A, B = c["A"], c["B"]
    np.ascontiguousarray(A.view(torch.uint8).cpu().numpy()).tofile(f"{d}/A_fp8.bin")
    # B 在 vLLM 里是 [K,N] 逻辑序（np.ascontiguousarray 会按逻辑序重排），与 D-17 §17 一致
    np.ascontiguousarray(B.view(torch.uint8).cpu().numpy()).tofile(f"{d}/B_fp8.bin")
    np.ascontiguousarray(c["out"].float().cpu().numpy().astype(np.float32)).tofile(f"{d}/ref_f32.bin")
    with open(f"{d}/meta.txt", "w") as f:
        f.write(f"shapeA=({c['m']}, {k}) shapeB=({k}, {n}) "
                f"scale_a={float(c['scale_a'])} scale_b={float(c['scale_b'])} out=torch.bfloat16\n")
    print(f"  {n}x{k}: m={c['m']} 已存 {d}", flush=True)

print("DONE", flush=True)
