#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 抓参照**单个专家**的 gate/up/down 输出，用来判定"专家 GEMM 本身"与"combine"哪个在引入差异。
# 做法：给 layer0 的 experts 模块里的 experts 列表逐个挂 hook。
# 跑法：GPU=4 scripts/dev/remote_5090.sh ctr 'cd /w/dinfer && M=<ckpt> IDS="<ids>" python3 -u scripts/dev/moe_expert_capture.py'
import os, torch
def main():
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
    from vllm import LLM, SamplingParams
    M = os.environ["M"]; ids = [int(x) for x in os.environ["IDS"].split()]
    llm = LLM(model=M, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.85,
              dtype="auto", max_model_len=4096, enable_prefix_caching=False, block_size=32, max_num_seqs=1)
    m = llm.llm_engine.model_executor.driver_worker.model_runner.model
    cap = {}
    for name, mod in m.named_modules():
        if 'layers.0.' not in name:
            continue
        if name.endswith('.experts') and hasattr(mod, 'experts'):
            inner = getattr(mod, 'experts')
            lst = inner if isinstance(inner, (list, torch.nn.ModuleList)) else None
            print("EXPERTS_CONTAINER", name, type(inner).__name__, flush=True)
            if lst is not None:
                for i in (0, 1, 2):
                    if i < len(lst):
                        def mk(idx, nm):
                            def h(_m, inp, out):
                                k = f"e{idx}"
                                if k not in cap and inp and inp[0].shape[0] == len(ids):
                                    cap[k] = (inp[0].detach().clone(),
                                              out.detach().clone() if isinstance(out, torch.Tensor) else None)
                            return h
                        lst[i].register_forward_hook(mk(i, name))
                        print("HOOK", f"{name}.{i}", flush=True)
    llm.generate(prompts=[{"prompt_token_ids": ids}], sampling_params=SamplingParams(max_tokens=1, temperature=0))
    d = "/w/tmp/moe2"; os.makedirs(d, exist_ok=True)
    for k, (x, y) in cap.items():
        x.to(torch.bfloat16).view(torch.int16).cpu().numpy().tofile(f"{d}/{k}_x.bin")
        if y is not None:
            y.to(torch.bfloat16).view(torch.int16).cpu().numpy().tofile(f"{d}/{k}_y.bin")
        print("CAP", k, x.shape, None if y is None else y.shape, flush=True)
if __name__ == '__main__':
    main()
