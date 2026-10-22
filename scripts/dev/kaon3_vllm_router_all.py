#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 抓 **每一层** router 的输入（proj 之前）与输出（router_logits），一次前向抓全 30 层。
# 只挂 hook、不做任何计算，用于 `scripts/verify/kaon3_router_truth_all.py` 的 fp64 真值扫描。
# 跑法（容器内）：OUT=<dir> python3 -I scripts/dev/kaon3_vllm_router_all.py <model_dir> <id...>
# 产物：<OUT>/L{l}_router_in.bin、<OUT>/L{l}_router_logits.bin（f32 LE，[T, ·]）
import sys, os
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import numpy as np
from vllm import LLM, SamplingParams

model_dir = sys.argv[1]
ids = [int(x) for x in sys.argv[2:]]
t = len(ids)
out = os.environ["OUT"]
os.makedirs(out, exist_ok=True)

llm = LLM(model=model_dir, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.85,
          dtype="auto", max_model_len=4096, enable_prefix_caching=False, block_size=32, max_num_seqs=1)
m = llm.llm_engine.model_executor.driver_worker.model_runner.model
cand = m
while not hasattr(cand, "layers"):
    cand = next(getattr(cand, n) for n in ("language_model", "model", "text_model") if hasattr(cand, n))
captured = {}
for l, layer in enumerate(cand.layers):
    rt = getattr(layer, "router", None)
    if rt is None or getattr(rt, "proj", None) is None:
        continue
    def pre(mod, inp, l=l):
        x = inp[0] if inp else None
        if x is not None and x.shape[0] == t and f"L{l}_router_in" not in captured:
            captured[f"L{l}_router_in"] = x.detach().float().cpu().numpy().copy()
    def post(mod, inp, outp, l=l):
        o = outp[0] if isinstance(outp, tuple) else outp
        if o is not None and o.shape[0] == t and f"L{l}_router_logits" not in captured:
            captured[f"L{l}_router_logits"] = o.detach().float().cpu().numpy().copy()
    rt.proj.register_forward_pre_hook(pre)
    rt.register_forward_hook(post)

llm.generate(prompts=[{"prompt_token_ids": ids}], sampling_params=SamplingParams(max_tokens=1, temperature=0))
for name, v in captured.items():
    v[:t].astype("<f4").tofile(f"{out}/{name}.bin")
print(f"写 {len(captured)} 个张量到 {out}（{len(captured)//2} 层）", flush=True)
