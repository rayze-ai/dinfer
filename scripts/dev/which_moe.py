#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 查参照实际命中的 MoE 专家 kernel 实现（决定累加归约次序，逐位对拍必须先知道它）。
# 跑法：GPU=4 scripts/dev/remote_5090.sh ctr 'cd /w/dinfer && M=<ckpt> python3 -u scripts/dev/which_moe.py'
import os
def main():
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
    from vllm import LLM
    M = os.environ["M"]
    llm = LLM(model=M, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.85,
              dtype="auto", max_model_len=4096, enable_prefix_caching=False, block_size=32, max_num_seqs=1)
    m = llm.llm_engine.model_executor.driver_worker.model_runner.model
    for name, mod in m.named_modules():
        if name.endswith('.moe') and 'layers.0.' in name:
            print("MOE", name, type(mod).__name__, flush=True)
            for attr in ('experts', 'router', 'quant_method', 'moe_config'):
                v = getattr(mod, attr, None)
                if v is not None:
                    print(f"   {attr} = {type(v).__name__}", flush=True)
            r = getattr(mod, 'router', None)
            if r is not None:
                for attr in ('experts', 'impl', 'moe_kernel', 'quant_method'):
                    v = getattr(r, attr, None)
                    if v is not None:
                        print(f"   router.{attr} = {type(v).__name__}", flush=True)
                for attr in dir(r):
                    if 'impl' in attr or 'kernel' in attr or 'expert' in attr:
                        try:
                            v = getattr(r, attr)
                            if not callable(v):
                                print(f"   router.{attr} = {type(v).__name__}", flush=True)
                        except Exception:
                            pass
            break
if __name__ == '__main__':
    main()
