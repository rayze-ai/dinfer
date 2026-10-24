#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compile 口径起 vLLM，monkeypatch torch.ops.vllm.moe_forward（graph-break custom op），
# dump layer0 MoE 的真实输入（hidden=pre_ff2_norm, router_logits）与输出（moe_out, post_ff_norm_2 之前）。
# 用途：loop kaon3-moe 第 1 步——隔离 MoE 输出 vs 我方 moe_y。
# 跑：OUT=<目录> python3 -I scripts/dev/kaon3_vllm_moe_io.py <model_dir> <id...>
import sys, os
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import numpy as np
import torch
import vllm._custom_ops  # noqa
from vllm import LLM, SamplingParams

model_dir = sys.argv[1]
ids = [int(x) for x in sys.argv[2:]]
out = os.environ["OUT"]; os.makedirs(out, exist_ok=True)
MAX_T = int(os.environ.get("MAX_T", "512"))

GEN = {"on": False}
grabbed = {}

# patch MoERunner.forward（moe_forward op 背后的 eager 实现，moe_runner.py:714 self._forward_entry 所在）。
# 不碰 op 注册，无递归；graph-break op 内部 eager 跑，compile 下仍命中。
# patch MoERunner._forward_impl（moe_forward op 最终调的实例方法，moe_runner.py:830）。
# 实例调用动态查类方法 → 一定命中；op 背后 eager 跑，compile 下仍走它。
import vllm.model_executor.layers.fused_moe.runner.moe_runner as mr
_orig_impl = mr.MoERunner._forward_impl
def _patched_impl(self, hidden_states, router_logits, *a, **k):
    outp = _orig_impl(self, hidden_states, router_logits, *a, **k)
    if GEN["on"] and "l0" not in grabbed and hidden_states.shape[0] <= MAX_T:
        def sv(nm, t):
            t.detach().float().cpu().numpy().tofile(f"{out}/vllm_moe_{nm}.bin")
            return tuple(t.shape), str(t.dtype)
        grabbed["l0"] = {"hidden": sv("hidden", hidden_states), "logits": sv("logits", router_logits), "out": sv("out", outp)}
        print("[moe_io] L0 抓到:", grabbed["l0"], flush=True)
    return outp
mr.MoERunner._forward_impl = _patched_impl
print("[moe_io] patched MoERunner._forward_impl", flush=True)

llm = LLM(model=model_dir, tensor_parallel_size=1, gpu_memory_utilization=0.85,
          dtype="auto", max_model_len=16384, enable_prefix_caching=False, block_size=32,
          max_num_seqs=64, max_num_batched_tokens=4096,
          hf_overrides={"text_config": {"num_hidden_layers": 1}})
grabbed.clear(); GEN["on"] = True
llm.generate(prompts=[{"prompt_token_ids": ids}],
             sampling_params=SamplingParams(max_tokens=1, temperature=0, logprobs=5))
print("[moe_io] done grabbed:", list(grabbed.keys()), flush=True)
