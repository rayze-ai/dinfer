#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 抓 **SUB_LAYER 层 MoE 专家内部**的中间量（VLLM_CUTLASS NvFp4 路径），只挂 patch、不做计算。
# 用于子 loop C-5：与 kaon3_probe SUB_LAYER=<l> 写出的 moe_e{eid}_{gate,up,act,down}.bin 按 (token, 专家) 对齐比较。
# 跑法（容器内）：OUT=<dir> SUB_LAYER=11 python3 -I scripts/dev/kaon3_vllm_moe_experts.py <model_dir> <id...>
# 产物：topk_ids.bin(i32 [T,K]) c1.bin([R,2I] gate|up) c2.bin([R,I] 激活) c3s.bin([T*K,H] down 输出, (t,k) 序) c_map.bin(i32 [T*K])
#   vLLM 行 r = c_map[t*K + k]（k 为 topk_ids[t] 里的槽位）；均为 f32 LE。
import sys, os
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import numpy as np
import vllm.model_executor.models.gemma4 as g4
import vllm.model_executor.layers.fused_moe.experts.cutlass_moe as cm
from vllm import _custom_ops as ops
from vllm import LLM, SamplingParams

model_dir = sys.argv[1]
ids = [int(x) for x in sys.argv[2:]]
t = len(ids)
out = os.environ["OUT"]
want = int(os.environ.get("SUB_LAYER", "0"))
os.makedirs(out, exist_ok=True)
cap = {}
st = {"layer": -1, "on": False, "mm": 0}

_orig_layer = g4.Gemma4DecoderLayer.forward
def _layer(self, *a, **k):
    st["layer"] = self.layer_idx
    try:
        return _orig_layer(self, *a, **k)
    finally:
        st["layer"] = -1
g4.Gemma4DecoderLayer.forward = _layer

np_ = lambda x: x.detach().float().cpu().numpy().copy()

_orig_run = cm.run_cutlass_moe_fp4
def _run(*a, **k):
    a_ = k.get("a", a[1] if len(a) > 1 else None)
    on = st["layer"] == want and a_ is not None and a_.shape[0] == t and "c3s" not in cap
    if on:
        tk = k.get("topk_ids")
        if tk is not None:
            cap["topk_ids"] = tk.detach().cpu().numpy().astype("<i4")
        st["on"], st["mm"] = True, 0
    try:
        return _orig_run(*a, **k)
    finally:
        st["on"] = False
cm.run_cutlass_moe_fp4 = _run

_orig_mm = ops.cutlass_fp4_moe_mm
def _mm(c, *a, **k):
    r = _orig_mm(c, *a, **k)
    if st["on"]:
        cap["c1" if st["mm"] == 0 else "c3"] = np_(c)
        st["mm"] += 1
    return r
ops.cutlass_fp4_moe_mm = _mm

_orig_act = cm.apply_moe_activation
def _act(act, c2, c1, *a, **k):
    r = _orig_act(act, c2, c1, *a, **k)
    if st["on"]:
        cap["c2"] = np_(c2)
    return r
cm.apply_moe_activation = _act

_orig_sh = ops.shuffle_rows
def _sh(x, cmap, *a, **k):
    r = _orig_sh(x, cmap, *a, **k)
    if st["on"]:
        cap["c3s"] = np_(r)
        cap["c_map"] = cmap.detach().cpu().numpy().astype("<i4")
    return r
ops.shuffle_rows = _sh

llm = LLM(model=model_dir, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.85,
          dtype="auto", max_model_len=4096, enable_prefix_caching=False, block_size=32, max_num_seqs=1)
llm.generate(prompts=[{"prompt_token_ids": ids}], sampling_params=SamplingParams(max_tokens=1, temperature=0))
for n, v in cap.items():
    (v.astype("<i4") if v.dtype.kind == "i" else v.astype("<f4")).tofile(f"{out}/{n}.bin")
    print(f"  {n:<9} shape {v.shape} dtype {v.dtype}", flush=True)
print(f"写 {len(cap)} 个张量到 {out}；缺 {[n for n in ('topk_ids','c1','c2','c3s','c_map') if n not in cap]}", flush=True)
