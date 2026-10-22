#!/usr/bin/env python3
# 抓 vLLM 的**完整 logits**（pre/post softcap），用于和我们的 logits 直接比幅度与形状。
# patch Gemma4ForCausalLM.compute_logits。
import sys, os
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import numpy as np
import vllm.model_executor.models.gemma4 as g4
from vllm import LLM, SamplingParams

model_dir = sys.argv[1]; ids = [int(x) for x in sys.argv[2:]]; t = len(ids)
out = os.environ["OUT"]; os.makedirs(out, exist_ok=True)
cap = {}
_orig = g4.Gemma4ForCausalLM.compute_logits
def _patched(self, hidden_states, sampling_metadata=None):
    r = _orig(self, hidden_states, sampling_metadata)
    if "post" not in cap and r.shape[0] == t:
        cap["post"] = r.detach().float().cpu().numpy().copy()
    return r
g4.Gemma4ForCausalLM.compute_logits = _patched
# 同时抓 softcap 之前的：patch LogitsProcessor.forward 之前拦 _apply_head 返回值
import vllm.model_executor.layers.logits_processor as lpm
_orig_apply = lpm.LogitsProcessor._apply_head
def _patched_apply(self, lm_head, hidden_states, embedding_bias):
    r = _orig_apply(self, lm_head, hidden_states, embedding_bias)
    if "pre" not in cap and r.shape[0] == t:
        cap["pre"] = r.detach().float().cpu().numpy().copy()
    return r
lpm.LogitsProcessor._apply_head = _patched_apply

llm = LLM(model=model_dir, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.85,
          dtype="auto", max_model_len=4096, enable_prefix_caching=False, block_size=32, max_num_seqs=1)
llm.generate(prompts=[{"prompt_token_ids": ids}], sampling_params=SamplingParams(max_tokens=1, temperature=0, logprobs=5))
for k, v in cap.items():
    v = v[:t]
    v.astype("<f4").tofile(f"{out}/logits_{k}.bin")
    print(f"  logits_{k}: shape {v.shape} min={v.min():.4f} max={v.max():.4f} std={v.std():.4f}", flush=True)
print("OK", list(cap), flush=True)
