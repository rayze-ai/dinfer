#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 从 vLLM **自身**前向抓每层残差流(不是自写参考前向):monkey-patch Gemma4DecoderLayer.forward,
# 存它返回的 hidden_states(已 * layer_scalar,与我们 kaon3_probe 的 sc.x 同定义)。
# 用途:和 Rust 探针 --hidden-out 逐层比,定位哪层哪算子开始偏。
# 跑法(容器内,-I 隔离): OUT=<目录> python3 -I scripts/dev/kaon3_vllm_hidden.py <model_dir> <id...>
import sys, os
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # 单进程,patch 才在同进程生效
import numpy as np
import vllm.model_executor.models.gemma4 as g4
from vllm import LLM, SamplingParams

model_dir = sys.argv[1]
ids = [int(x) for x in sys.argv[2:]]
out = os.environ["OUT"]
os.makedirs(out, exist_ok=True)

cap = []
_orig = g4.Gemma4DecoderLayer.forward
def _patched(self, *a, **k):
    r = _orig(self, *a, **k)
    hs = r[0] if isinstance(r, tuple) else r
    cap.append(hs.detach().float().cpu().numpy().copy())
    return r
g4.Gemma4DecoderLayer.forward = _patched

llm = LLM(model=model_dir, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.85,
          dtype="auto", max_model_len=4096, enable_prefix_caching=False, block_size=32, max_num_seqs=1)
cap.clear()  # 丢掉 warmup / profile run
llm.generate(prompts=[{"prompt_token_ids": ids}], sampling_params=SamplingParams(max_tokens=1, temperature=0, logprobs=5))
t = len(ids)
layers = cap[-30:] if len(cap) >= 30 else cap
print(f"captured {len(cap)} forward 调用, 取最后 {len(layers)} 层", flush=True)
for i, h in enumerate(layers):
    h = h[:t]  # 去掉 padding 行
    h.astype("<f4").tofile(f"{out}/L{i}.bin")
    if i < 3 or i >= len(layers) - 2:
        n = float(np.sqrt((h.astype(np.float64) ** 2).sum()))
        print(f"  L{i}: shape {h.shape}  首8 {h.flatten()[:8].round(4).tolist()}  ‖x‖ {n:.4f}", flush=True)
print(f"写 {len(layers)} 个 .bin 到 {out}", flush=True)
