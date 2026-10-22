#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 抓 **layer 0 内部**各子模块的输出（不是整层残差流），用于把我们 layer.rs 的中间量逐项对拍。
# 做法：patch Gemma4DecoderLayer.forward；当 self.layer_idx == 0 且 batch 行数 == t 时，
# 临时给 layer 0 的子模块挂 forward hook，跑完就摘掉。enforce_eager=True 保证不走图、hook 必触发。
# 跑法（容器内）：OUT=<dir> python3 -I scripts/dev/kaon3_vllm_sublayer.py <model_dir> <id...>
import sys, os
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import numpy as np
import vllm.model_executor.models.gemma4 as g4
from vllm import LLM, SamplingParams

model_dir = sys.argv[1]
ids = [int(x) for x in sys.argv[2:]]
t = len(ids)
out = os.environ["OUT"]
os.makedirs(out, exist_ok=True)

# 模块属性名 -> 输出文件名（与 models/kaon3/src/layer.rs 的 sub dump 同名）
HOOKS = {
    "input_layernorm": "input_norm",
    "self_attn": "o_proj",              # vLLM 的 self_attn 已含 o_proj
    "post_attention_layernorm": "post_attn_norm",
    "mlp": "dense",
    "router": "router_logits",
    "experts": "moe_y",
    "post_feedforward_layernorm_2": "ff2",
    "post_feedforward_layernorm": "post_ff_norm",
}

captured = {}
_orig = g4.Gemma4DecoderLayer.forward

def _patched(self, *a, **k):
    hs = a[1] if len(a) > 1 else k.get("hidden_states")
    if self.layer_idx != 0 or hs is None or hs.shape[0] != t or captured:
        return _orig(self, *a, **k)
    handles = []
    def mk(name):
        def hook(mod, inp, outp):
            o = outp[0] if isinstance(outp, tuple) else outp
            captured[name] = o.detach().float().cpu().numpy().copy()
        return hook
    for attr, name in HOOKS.items():
        m = getattr(self, attr, None)
        if m is not None:
            handles.append(m.register_forward_hook(mk(name)))
    try:
        r = _orig(self, *a, **k)
    finally:
        for h in handles:
            h.remove()
    return r

g4.Gemma4DecoderLayer.forward = _patched

llm = LLM(model=model_dir, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.85,
          dtype="auto", max_model_len=4096, enable_prefix_caching=False, block_size=32, max_num_seqs=1)
llm.generate(prompts=[{"prompt_token_ids": ids}], sampling_params=SamplingParams(max_tokens=1, temperature=0, logprobs=5))

for name, v in captured.items():
    v = v[:t]
    v.astype("<f4").tofile(f"{out}/{name}.bin")
    n = float(np.sqrt((v.astype(np.float64) ** 2).sum()))
    print(f"  {name:<16} shape {v.shape}  ‖x‖ {n:.4f}  首4 {v.flatten()[:4].round(4).tolist()}", flush=True)
missing = [n for n in HOOKS.values() if n not in captured]
print(f"写 {len(captured)} 个张量到 {out}" + (f"；缺 {missing}" if missing else ""), flush=True)
