#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 抓 vLLM **真实**的 MoE 子模块输入/输出（逐位核验专家 GEMM 用）。
# 做法：eager 模式下给 layers.0 的 `.experts` 挂 forward hook（compile 下 hook 不触发，所以必须 enforce_eager）。
# 跑法（**干净容器**，不要用被服务占着的卡）：
#   GPU=4 scripts/dev/remote_5090.sh ctr 'cd /w/dinfer && M=$(ls -d /hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/*/|head -1) \
#     IDS="<token ids>" python3 -u scripts/dev/moe_capture.py'
# 产物：/w/tmp/moe1/{x.bin,y.bin}（bf16 原始位）
import os, torch, sys
M=os.environ["M"]; ids=[int(x) for x in os.environ["IDS"].split()]
print("start", flush=True)
from vllm import LLM, SamplingParams
llm=LLM(model=M, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.85,
        dtype="auto", max_model_len=4096, enable_prefix_caching=False, block_size=32, max_num_seqs=1)
print("loaded", flush=True)
cap=[]
# 挂到 MoE 层的 experts 子模块上（eager 下 hook 必触发）
model=llm.llm_engine.model_executor.driver_worker.model_runner.model
n=0
for name,mod in model.named_modules():
    if name.endswith('.experts') and 'layers.0.' in name:
        def mk(nm):
            def h(m,i,o):
                if i and i[0].shape[0]==len(ids) and not cap:
                    cap.append((nm, i[0].detach().clone(), o.detach().clone() if isinstance(o,torch.Tensor) else None))
            return h
        mod.register_forward_hook(mk(name)); n+=1
print("hooks", n, flush=True)
llm.generate(prompts=[{"prompt_token_ids":ids}], sampling_params=SamplingParams(max_tokens=1,temperature=0))
print("gen done, caps", len(cap), flush=True)
if cap:
    nm,x,y=cap[0]
    d="/w/tmp/moe1"; os.makedirs(d,exist_ok=True)
    x.to(torch.bfloat16).view(torch.int16).cpu().numpy().tofile(f"{d}/x.bin")
    if y is not None: y.to(torch.bfloat16).view(torch.int16).cpu().numpy().tofile(f"{d}/y.bin")
    print("CAP",nm,x.shape, None if y is None else y.shape, flush=True)
