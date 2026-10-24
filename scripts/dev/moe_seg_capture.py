#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓 vLLM **compile 口径**下 MoE 内部的三个分段中间量，逐段定位 1 ULP 差异。

为什么不用 hook：compile 下 forward hook 不触发（`moe_capture.py` 注释）。
所以改成 patch 模块函数，compile 会照常调用被 patch 的函数。

口径（用户 2026-10-23 明确）：**不开 enforce_eager**，与生产 / oracle 一致。

抓的分段（对应 `run_cutlass_moe_fp4`）：
  c1  = gate/up 的 GEMM 输出（bf16，silu/gelu **之前**）
  c2  = activation 之后（gate·up 门控后的 bf16）
  c3  = down 的 GEMM 输出、shuffle 回来之后（bf16）
  y   = combine 之后（MoE 最终输出）

同时抓 `rep_a_fp4 / rep_a_blockscale`（专家量化器的输出）与 `int_fp4/int_blockscale`，
用来判定"量化段"有没有差异。

产物：/w/tmp/moe3/
"""
import os
import sys
import numpy as np
import torch


def dump(name, t):
    d = os.environ.get("OUT", "/w/tmp/moe3")
    os.makedirs(d, exist_ok=True)
    arr = t.detach().cpu()
    if arr.dtype == torch.bfloat16:
        arr.view(torch.int16).numpy().tofile(f"{d}/{name}.bin")
    elif arr.dtype == torch.uint8:
        arr.numpy().tofile(f"{d}/{name}.bin")
    elif arr.dtype in (torch.float32, torch.int32):
        arr.numpy().tofile(f"{d}/{name}.bin")
    else:
        arr.float().numpy().tofile(f"{d}/{name}.bin")
    print(f"DUMP {name} shape={tuple(arr.shape)} dtype={arr.dtype}", flush=True)


def main():
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
    os.environ.setdefault("CUBLASLT_WORKSPACE_SIZE", "1")   # 逐层对拍必须，见 pitfalls 2026-10-23

    from vllm import LLM, SamplingParams
    from vllm import _custom_ops as ops
    import vllm.model_executor.layers.fused_moe.experts.cutlass_moe as cm
    from vllm.model_executor.layers.fused_moe.utils import (
        moe_kernel_quantize_input as real_q,
    )

    M = os.environ["M"]
    ids = [int(x) for x in os.environ["IDS"].split()]
    cap = {"n": 0}

    # ---- patch 1：activation 之后（c2 出口）；gelu 走 apply_moe_activation ----
    real_act = cm.apply_moe_activation

    def act_patched(activation, out, input_, **kw):
        r = real_act(activation, out, input_, **kw)
        if cap["n"] == 0:
            dump("c1_gateup", input_)
            dump("c2_act", out)
        return r

    cm.apply_moe_activation = act_patched

    # ---- patch 2：专家量化器（区分两次调用：gate/up 输入 vs down 输入）----
    real_eoq = ops.scaled_fp4_experts_quant

    def eoq_patched(a, gscale, eoff, boff, ntopk):
        r = real_eoq(a, gscale, eoff, boff, ntopk)
        if cap["n"] == 0:
            k = "a1" if "q1" not in cap else "a2"
            cap["q1"] = 1
            dump(f"{k}_in", a)
            dump(f"{k}_fp4", r[0])
            dump(f"{k}_sf", r[1])
            dump(f"{k}_eoff", eoff)
            dump(f"{k}_boff", boff)
        return r

    ops.scaled_fp4_experts_quant = eoq_patched

    # ---- patch 3：整个 run_cutlass_moe_fp4（拿 c3 与 combine 后的 output）----
    real_run = cm.run_cutlass_moe_fp4

    def run_patched(output, *a, **kw):
        r = real_run(output, *a, **kw)
        if cap["n"] == 0:
            dump("y_moe", output)
            cap["n"] = 1
        return r

    cm.run_cutlass_moe_fp4 = run_patched

    llm = LLM(model=M, tensor_parallel_size=1, gpu_memory_utilization=0.85,
              dtype="auto", max_model_len=4096, enable_prefix_caching=False,
              block_size=32, max_num_seqs=1)
    print("loaded", flush=True)
    llm.generate(prompts=[{"prompt_token_ids": ids}],
                 sampling_params=SamplingParams(max_tokens=1, temperature=0.0))

    # ---- 同时抓 layer 0 的 experts 输入 x（供我们侧复现）----
    print("DONE captures=", cap["n"], flush=True)


if __name__ == "__main__":
    main()
