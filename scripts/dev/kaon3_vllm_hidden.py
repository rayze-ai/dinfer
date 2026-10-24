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
SUB = os.environ.get("SUB_LAYER")          # 设了才抓该层的子模块中间量
sub = {}
def _patched(self, *a, **k):
    r = _orig(self, *a, **k)
    hs = r[0] if isinstance(r, tuple) else r
    cap.append(hs.detach().float().cpu().numpy().copy())
    return r
g4.Gemma4DecoderLayer.forward = _patched

llm = LLM(model=model_dir, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.85,
          dtype="auto", max_model_len=4096, enable_prefix_caching=False, block_size=32, max_num_seqs=1)
cap.clear()  # 丢掉 warmup / profile run

# ---- 子模块级 hook：只为定位"层内哪一段"（SUB_LAYER=<n>）----
# 必须在 LLM 构造之后挂：模型实例那时才存在。用 named_modules 取真实模块，
# 不猜属性名 —— 找不到就打印候选名，不要静默跳过。
sub = {}
if SUB is not None:
    want = int(SUB)
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    # 只要语言模型那一层：`vision_tower.encoder.layers.0` 也会命中 "layers.0."，
    # 2026-10-24 实测因此抓到了 1152 宽的视觉层（形状对不上），必须限定前缀。
    cands = [(n, m) for n, m in model.named_modules()
             if n.startswith("language_model.model.layers.") and f".layers.{want}." in n]
    print(f"SUB_LAYER={want}: 候选模块 {len(cands)} 个", flush=True)
    picked = 0
    for n, m in cands:
        tag = None
        # 顺序重要：端到端匹配优先，长的放前面（`post_feedforward_layernorm_1`
        # 不能被 `post_feedforward_layernorm` 抢走）。
        for key, t in (("post_feedforward_layernorm_1", "post_ff_norm1"),
                       ("post_feedforward_layernorm_2", "post_ff_norm2"),
                       ("pre_feedforward_layernorm_2", "pre_ff_norm2"),
                       ("post_feedforward_layernorm", "post_ff_norm"),
                       ("input_layernorm", "input_norm"),
                       ("self_attn", "attn_out"),
                       ("pre_feedforward_layernorm", "pre_ff_norm"),
                       ("mlp", "dense"),
                       ("moe", "moe_out"),
                       ("router", "router_out")):
            if n.endswith("." + key):
                tag = t
                break
        if tag is None:
            continue

        def _mk(tg):
            def _h(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                if not hasattr(o, "detach") or o.shape[0] != len(ids):
                    return
                sub.setdefault(tg, o.detach().float().cpu().numpy().copy())
            return _h
        m.register_forward_hook(_mk(tag))
        picked += 1
        print(f"  hook {n} -> {tag}", flush=True)
    if picked == 0:
        print("  ⚠️ 一个都没挂上；候选名：", [n for n, _ in cands][:20], flush=True)
llm.generate(prompts=[{"prompt_token_ids": ids}], sampling_params=SamplingParams(max_tokens=1, temperature=0, logprobs=5))
t = len(ids)
# **只取 prefill 那一批**：decode 步每步也会调 30 次 DecoderLayer.forward，
# 直接 cap[-30:] 会取到最后一个 decode 步（单行）而不是 T 行的 prefill（2026-10-22 修）。
pre = [h for h in cap if h.shape[0] == t]
layers = pre[:30]
pe = [e for e in embeds if e.shape[0] == t]
for k_, v_ in sub.items():
    v_[:t].astype("<f4").tofile(f"{out}/{k_}.bin")
    print(f"  写 {k_}.bin shape {v_[:t].shape} nz={float((v_[:t]!=0).mean()):.4f}", flush=True)
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
