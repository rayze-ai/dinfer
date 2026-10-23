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
# 端到端 kernel 变体下限（子 loop D-14）：FORCE_KERNEL=cutlass|flashinfer 强制 fp8 linear kernel。
# 必须在创建 LLM 之前 patch（实测：放在之后不生效）。
_FORCE = os.environ.get("FORCE_KERNEL")
if _FORCE:
    import vllm.model_executor.kernels.linear as _KL
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import CutlassFP8ScaledMMLinearKernel as _C
    from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import FlashInferFP8ScaledMMLinearKernel as _F
    _want = {"cutlass": _C, "flashinfer": _F}[_FORCE]
    _orig_init = _KL.init_fp8_linear_kernel
    _picked = {}
    def _patched_init(*a, **k):
        k.setdefault("force_kernel", _want)
        r = _orig_init(*a, **k)
        _picked[type(r).__name__] = _picked.get(type(r).__name__, 0) + 1
        return r
    _KL.init_fp8_linear_kernel = _patched_init
    import vllm.model_executor.layers.quantization.fp8 as _F8
    import vllm.model_executor.layers.quantization.modelopt as _MO
    for _m in (_F8, _MO):
        if hasattr(_m, "init_fp8_linear_kernel"):
            _m.init_fp8_linear_kernel = _patched_init
    print(f"[force_kernel] 请求 {_FORCE}", flush=True)
import vllm.model_executor.models.gemma4 as g4
from vllm import LLM, SamplingParams

model_dir = sys.argv[1]
ids = [int(x) for x in sys.argv[2:]]
out = os.environ["OUT"]
os.makedirs(out, exist_ok=True)

cap = []
embeds = []  # embed 输出（L-1：layer 0 之前的残差流）
_orig_embed = g4.Gemma4SelfDecoderLayers.embed_input_ids
def _patched_embed(self, input_ids):
    r = _orig_embed(self, input_ids)
    embeds.append(r.detach().float().cpu().numpy().copy())
    return r
g4.Gemma4SelfDecoderLayers.embed_input_ids = _patched_embed
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
# **只取 prefill 那一批**：decode 步每步也会调 30 次 DecoderLayer.forward，
# 直接 cap[-30:] 会取到最后一个 decode 步（单行）而不是 T 行的 prefill（2026-10-22 修）。
pre = [h for h in cap if h.shape[0] == t]
layers = pre[:30]
pe = [e for e in embeds if e.shape[0] == t]
if pe:
    pe[0][:t].astype("<f4").tofile(f"{out}/Lm1.bin")
    print(f"  写 Lm1.bin（embed 输出）shape {pe[0][:t].shape}", flush=True)
print(f"captured {len(cap)} 次 forward 调用，其中 T={t} 的 {len(pre)} 次；取前 {len(layers)} 层", flush=True)
assert len(layers) == 30, f"expected 30 prefill layers, got {len(layers)}"
for i, h in enumerate(layers):
    h = h[:t]
    h.astype("<f4").tofile(f"{out}/L{i}.bin")
    if i < 3 or i >= len(layers) - 2:
        n = float(np.sqrt((h.astype(np.float64) ** 2).sum()))
        print(f"  L{i}: shape {h.shape}  首8 {h.flatten()[:8].round(4).tolist()}  ‖x‖ {n:.4f}", flush=True)
print(f"写 {len(layers)} 个 .bin 到 {out}", flush=True)
