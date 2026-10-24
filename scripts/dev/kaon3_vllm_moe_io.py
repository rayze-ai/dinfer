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
import vllm.model_executor.layers.fused_moe.runner.moe_runner as mr

def _grab(hidden_states, router_logits, outp):
    if GEN["on"] and "l0" not in grabbed and hidden_states.shape[0] <= MAX_T:
        def sv(nm, t):
            t.detach().float().cpu().numpy().tofile(f"{out}/vllm_moe_{nm}.bin")
            return tuple(t.shape), str(t.dtype)
        grabbed["l0"] = {"hidden": sv("hidden", hidden_states), "logits": sv("logits", router_logits), "out": sv("out", outp)}
        print("[moe_io] L0 抓到:", grabbed["l0"], flush=True)

def install():
    # 两处都 patch：① 重注册 op CUDA impl（compile 路径）；② _forward_impl 类方法（eager 路径）。
    _orig = mr._moe_forward
    @torch.library.impl("vllm::moe_forward", "cuda")
    def _impl(hidden_states, router_logits, shared_experts_input, input_ids, layer_name, hidden_dim_unpadded):
        outp = _orig(hidden_states, router_logits, shared_experts_input, input_ids, layer_name, hidden_dim_unpadded)
        _grab(hidden_states, router_logits, outp)
        return outp
    _oi = mr.MoERunner._forward_impl
    def _pi(self, hidden_states, router_logits, *a, **k):
        outp = _oi(self, hidden_states, router_logits, *a, **k)
        _grab(hidden_states, router_logits, outp)
        return outp
    mr.MoERunner._forward_impl = _pi
    print("[moe_io] installed op-impl + _forward_impl patches", flush=True)

# EAGER=1：enforce_eager，MoE patch 一定命中（无 compile 图吞）。MoE 数值 eager≈compile（无 FEXP 类差异）。
_eager = os.environ.get("EAGER", "0") == "1"
llm = LLM(model=model_dir, tensor_parallel_size=1, gpu_memory_utilization=0.85,
          dtype="auto", max_model_len=16384, enable_prefix_caching=False, block_size=32,
          max_num_seqs=64, max_num_batched_tokens=4096, enforce_eager=_eager,
          hf_overrides={"text_config": {"num_hidden_layers": 1}})
install()
grabbed.clear(); GEN["on"] = True
llm.generate(prompts=[{"prompt_token_ids": ids}],
             sampling_params=SamplingParams(max_tokens=1, temperature=0, logprobs=5))
print("[moe_io] done grabbed:", list(grabbed.keys()), flush=True)
