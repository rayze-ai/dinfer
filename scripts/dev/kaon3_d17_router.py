#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定位 L2 router 的 bf16 GEMM 1-ulp 差异：vLLM 的 F.linear 到底走哪条 cuBLAS 路径。

背景（D-16-REBASE §46）：L2 隔离下 `router_in` 逐位相同，`router_logits` 有 4 个元素差 1-ulp
（token 1: 2.625 vs 2.640625）。router 是 bf16 权重，我们走 cuBLASLt 启发式 h[0]，
vLLM 走 PyTorch `F.linear`。

本脚本抓 vLLM 前向里 router 的真实入参（`hidden_states` 进 router 前 + 输出），
再用多种 PyTorch 表达方式算同一个 GEMM，看出哪种复现 vLLM 的输出：
  - F.linear(x, w)
  - x @ w.t()
  - torch.mm(x, w.t())
  - F.linear(x.half(), ...)（若权重是 bf16）
"""
import os
import sys

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import numpy as np
import torch

import vllm.model_executor.models.gemma4 as g4
from vllm import LLM, SamplingParams

model_dir = sys.argv[1]
ids = [int(x) for x in sys.argv[2:]]
out = os.environ.get("OUT", "/w/d17b/router")
os.makedirs(out, exist_ok=True)
TARGET_LAYER = 2

cap = {}
# vLLM 的 Gemma4Router.forward 是：
#   x = self.norm(x); x = x * self.root_size.to(dtype); x = x * self.scale.to(dtype)
#   router_logits, _ = self.proj(x)
# 所以真正喂进 GEMM 的是**预处理之后**的 x。抓 proj 的入参而不是 router 的入参。
_orig_forward = g4.Gemma4Router.forward


def _patched_router(self, x, *a, **k):
    # 复刻 vLLM 的预处理，拿到进 proj 前的真实张量（逐位同口径）
    xp = self.norm(x)
    xp = xp * self.root_size.to(xp.dtype)
    xp = xp * self.scale.to(xp.dtype)
    if len(cap) < 1:
        cap["x_in"] = x.detach().clone()
        cap["xp"] = xp.detach().clone()
        cap["w"] = self.proj.weight.detach().clone()
    return _orig_forward(self, x, *a, **k)


g4.Gemma4Router.forward = _patched_router

llm = LLM(model=model_dir, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.85,
          dtype="auto", max_model_len=4096, enable_prefix_caching=False, block_size=32, max_num_seqs=1)
cap.clear()
llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)

if not cap:
    print("FAIL 没抓到 router 调用")
    sys.exit(2)

x, out_ref, w = cap["xp"], None, cap["w"]
print(f"router: 进 proj 的 x={tuple(x.shape)} {x.dtype}  w={tuple(w.shape)} {w.dtype}", flush=True)

# 用 vLLM 自己的 forward 再跑一遍，作为 out_ref（同进程、同权重、确定性）
with torch.no_grad():
    out_ref = torch.nn.functional.linear(x, w)

np.ascontiguousarray(x.float().cpu().numpy().astype(np.float32)).tofile(f"{out}/x_f32.bin")
np.ascontiguousarray(out_ref.float().cpu().numpy().astype(np.float32)).tofile(f"{out}/out_ref_f32.bin")
np.ascontiguousarray(w.view(torch.int16).cpu().numpy()).tofile(f"{out}/w_bf16.bin")


def same(a, b):
    aa = a.flatten().view(torch.int16)
    bb = b.flatten().view(torch.int16)
    if aa.numel() != bb.numel():
        return -1
    return int((aa != bb).sum().item())


print("=== 用同一份入参，试不同 PyTorch 表达式 ===", flush=True)
if w is not None:
    for label, fn in [
        ("F.linear(x, w)", lambda: torch.nn.functional.linear(x, w)),
        ("x @ w.t()", lambda: x @ w.t()),
        ("torch.mm(x, w.t())", lambda: torch.mm(x, w.t())),
        ("F.linear(x, w.contiguous())", lambda: torch.nn.functional.linear(x, w.contiguous())),
        ("F.linear(x, w.t().t())", lambda: torch.nn.functional.linear(x, w.t().t())),
    ]:
        try:
            r = fn()
            print(f"  {label:<32} vs vLLM out: {same(r, out_ref)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {label:<32} 失败 {str(e)[:80]}", flush=True)

print("DONE", flush=True)
