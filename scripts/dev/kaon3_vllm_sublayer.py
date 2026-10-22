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
    "pre_feedforward_layernorm": "pre_ff_norm",
    "router": "router_logits",
    "experts": "moe_y",
    "post_feedforward_layernorm_2": "ff2",
    "post_feedforward_layernorm": "post_ff_norm",
}

captured = {}
_orig = g4.Gemma4DecoderLayer.forward

# MoE 输出：MoERunner.forward 的返回值。`experts` 是 PluggableLayer，挂普通 forward hook 不触发，
# 所以直接 patch 类方法（模块级只做一次，避免每层重复包装）。
try:
    import vllm.model_executor.layers.fused_moe.runner.moe_runner as _mr
    _orig_moe = _mr.MoERunner.forward

    def _moe_patched(self_, *aa, **kk):
        out_ = _orig_moe(self_, *aa, **kk)
        t_ = out_[0] if isinstance(out_, tuple) else out_
        if t_ is not None and "moe_y" not in captured and t_.shape[0] == t:
            captured["moe_y"] = t_.detach().float().cpu().numpy().copy()
        return out_

    _mr.MoERunner.forward = _moe_patched
except Exception as _e:  # noqa: BLE001
    print(f"MoERunner patch 失败: {_e}", flush=True)


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
    sa = getattr(self, "self_attn", None)
    # rope 与 attention 的输出：rope 返回 (q, k)，attention 返回 o_proj 之前的 attn 输出
    if sa is not None:
        def rope_hook(mod, inp, outp):
            q_, k_ = outp[0], outp[1] if isinstance(outp, (tuple, list)) else (outp, None)
            captured["q_rope"] = q_.detach().float().cpu().numpy().copy()
            if k_ is not None:
                captured["k_rope"] = k_.detach().float().cpu().numpy().copy()
        r_ = getattr(sa, "rotary_emb", None)
        if r_ is not None:
            handles.append(r_.register_forward_hook(rope_hook))
        a_ = getattr(sa, "attn", None)
        if a_ is not None:
            handles.append(a_.register_forward_hook(mk("attn_out")))
    # down_proj 的**输入**（= gelu(gate)*up，FP8 激活量化的对象）
    mlp_ = getattr(self, "mlp", None)
    if mlp_ is not None:
        dp = getattr(mlp_, "down_proj", None)
        if dp is not None:
            def _dp_pre(mod, inp):
                if inp and inp[0] is not None:
                    captured["mlp_gelu"] = inp[0].detach().float().cpu().numpy().copy()
            handles.append(dp.register_forward_pre_hook(_dp_pre))
    # MLP 内部：gate_up 投影输出（gelu 之前）与 down 投影输出
    if mlp_ is not None:
        for attr, name in (("gate_up_proj", "mlp_gate_up"), ("down_proj", "mlp_down")):
            m_ = getattr(mlp_, attr, None)
            if m_ is not None:
                handles.append(m_.register_forward_hook(mk(name)))
    # router 的**真正输入**（norm + root_size + scale 之后、proj 之前）：用 pre-hook 抓
    rt = getattr(self, "router", None)
    pr = getattr(rt, "proj", None) if rt is not None else None
    if pr is not None:
        def pre_hook(mod, inp):
            if inp and inp[0] is not None:
                captured["router_in"] = inp[0].detach().float().cpu().numpy().copy()
        handles.append(pr.register_forward_pre_hook(pre_hook))
    # self_attn 内部：q_norm / k_norm 的输出（hook 到 norm 模块本身）
    for attr, name in (("q_norm", "q_norm"), ("k_norm", "k_norm"), ("v_norm", "v_norm")):
        m = getattr(sa, attr, None) if sa is not None else None
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
