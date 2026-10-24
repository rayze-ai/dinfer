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
    _orig(query, key, value, output, layer_name, *a, **k)  # 先算（output 就地写、KV 已写进 fp8 cache）
    # 第 0 层：generate 期间每次都覆盖保存（warmup 在 clear 前已丢），并打印 shape。
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
        # 关键（用户 2026-10-24）：不从 bf16 重建 k/v，直接从 fp8 cache 读回 attention 实际用的那份字节。
        try:
            md, slf, kv_cache, slot = attn.get_attention_context(attn._resolve_layer_name(layer_name))
            import torch as _t
            T = got_t
            sm = slot.flatten()[:T].long()
            # kv_cache 形状含 fp8；view 成 fp8 后按 (block, off) gather。布局 [num_blocks, 2, block, nkv, D] 或 split。
            kc = kv_cache
            if kc.dtype == _t.uint8:
                kc = kc.view(_t.float8_e4m3fn)
            shp = tuple(kc.shape)
            saved = False
            # 实测布局 (num_blocks, nkv=8, block=16, 2*D=512)：最后一维前 D=k、后 D=v
            if len(shp) == 4 and shp[1] == 8 and shp[3] == 512:
                D2 = 256
                bs = shp[2]  # block size = cache 实际第 3 维（不猜，直接取）
                blk, off = sm // bs, sm % bs
                print(f"[attn_io] cache block_size={bs}, slot[:4]={sm[:4].tolist()}", flush=True)
                # gather：对每个 token 取 [nkv, 2D]，blk/off 索引到 (block, pos)
                kv = kc[blk, :, off, :]           # [T, nkv, 512] fp8
                kk = kv[:, :, :D2].to(_t.bfloat16)   # [T, nkv, 256]
                vv = kv[:, :, D2:].to(_t.bfloat16)
                kk.float().cpu().numpy().tofile(f"{out}/vllm_kcache.bin")
                vv.float().cpu().numpy().tofile(f"{out}/vllm_vcache.bin")
                info["kv_cache"] = ("4d_nkv_block_2D", shp, str(kv_cache.dtype), "blk*16+off")
                saved = True
            if not saved:
                info["kv_cache"] = ("shape_unhandled", shp, str(kv_cache.dtype))
                print("[attn_io] kv_cache 布局未处理:", shp, kv_cache.dtype, flush=True)
        except Exception as e:
            info["kv_cache_err"] = str(e)
            print("[attn_io] 读 fp8 cache 失败:", e, flush=True)
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
