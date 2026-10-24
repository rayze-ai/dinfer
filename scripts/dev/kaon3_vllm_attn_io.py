#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compile 口径起 vLLM，monkeypatch `unified_attention_with_output`（graph-break custom op，compile 下命中），
# dump layer 0 attention 的真实输入 q/k/v 与输出 output。用途：用户 2026-10-24 两步硬对拍——
#   ① 比我方 dump 的 attention 输入是否与 vLLM 逐位一致；② 把 vLLM 的 q/k/v 喂我们 Rust gqa 比输出。
# 跑法（容器内，-I）：OUT=<目录> python3 -I scripts/dev/kaon3_vllm_attn_io.py <model_dir> <id...>
import sys, os
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # 单进程 patch 才生效
import numpy as np
import torch
import vllm.model_executor.layers.attention.attention as attn
from vllm import LLM, SamplingParams

model_dir = sys.argv[1]
ids = [int(x) for x in sys.argv[2:]]
out = os.environ["OUT"]
os.makedirs(out, exist_ok=True)

grabbed = {}
GEN = {"on": False}  # 只在真实 generate 阶段抓，warmup 阶段不抓
_orig = attn.unified_attention_with_output
def _patched(query, key, value, output, layer_name, *a, **k):
    name = attn._resolve_layer_name(layer_name) if hasattr(attn, "_resolve_layer_name") else str(layer_name)
    _orig(query, key, value, output, layer_name, *a, **k)  # 先算（output 就地写）
    # 第 0 层：generate 期间每次都覆盖保存（warmup 在 clear 前已丢），并打印 shape。
    # 只接受 token 数 <= max_want 的（排除 4096 warmup padding 批；真实 prefill token 数小）。
    max_want = int(os.environ.get("MAX_T", "512"))
    got_t = query.shape[0]
    if "layers.0." in str(name) and got_t <= max_want and GEN["on"]:
        def sv(nm, t):
            arr = t.detach().float().cpu().numpy().copy()
            arr.tofile(f"{out}/vllm_{nm}.bin")
            return tuple(t.shape), str(t.dtype)
        info = {
            "query": sv("q", query), "key": sv("k", key),
            "value": sv("v", value), "output": sv("out", output),
        }
        grabbed["l0"] = info
        print("[attn_io] L0 抓到:", info, flush=True)
attn.unified_attention_with_output = _patched
# torch.ops 注册的也要换（compile 图走的是 torch.ops.vllm.unified_attention_with_output）
try:
    import vllm.model_executor.layers.attention.attention as _a
    torch.library.impl("vllm::unified_attention_with_output", "cuda")(_patched)
except Exception as e:
    print("[attn_io] torch.library.impl patch skip:", e, flush=True)

# compile 口径（不设 enforce_eager）。与 oracle server 参数对齐：block 32 / max_num_batched_tokens 4096 / 1 层截断可选
llm = LLM(model=model_dir, tensor_parallel_size=1, gpu_memory_utilization=0.85,
          dtype="auto", max_model_len=16384, enable_prefix_caching=False, block_size=32,
          max_num_seqs=64, max_num_batched_tokens=4096,
          hf_overrides={"text_config": {"num_hidden_layers": 1}})
grabbed.clear()
GEN["on"] = True
llm.generate(prompts=[{"prompt_token_ids": ids}],
             sampling_params=SamplingParams(max_tokens=1, temperature=0, logprobs=5))
print("[attn_io] done, grabbed:", list(grabbed.keys()), flush=True)
