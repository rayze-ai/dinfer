#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 用**固定入参**独立调用参照的 cutlass FP4 MoE，产出"参照标准答案"，与我们的 Rust 实现逐位比。
# 入参由 cutlass_moe_probe.py 从前向里抓（/w/tmp/cmo/{a,topk_ids,topk_w}.bin + 权重取自 checkpoint）。
# 跑法：GPU=4 scripts/dev/remote_5090.sh ctr 'cd /w/dinfer && M=<ckpt> python3 -u scripts/dev/cutlass_moe_standalone.py'
import os, numpy as np, torch
def main():
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
    from vllm import LLM
    from vllm.model_executor.layers.fused_moe.experts import cutlass_moe as cm
    M = os.environ["M"]
    d_in = "/w/tmp/cmo"; d_out = "/w/tmp/cmo_out"; os.makedirs(d_out, exist_ok=True)
    cap = {}
    orig = cm.run_cutlass_moe_fp4
    def patched(output, a, a1_gscale, w1_fp4, w1_blockscale, w1_alphas, a2_gscale, w2_fp4,
                w2_blockscale, w2_alphas, topk_weights, topk_ids, activation, workspace13, workspace2,
                m, n, k, e, device, apply_router_weight_on_input=False, **kw):
        r = orig(output, a, a1_gscale, w1_fp4, w1_blockscale, w1_alphas, a2_gscale, w2_fp4,
                 w2_blockscale, w2_alphas, topk_weights, topk_ids, activation, workspace13, workspace2,
                 m, n, k, e, device, apply_router_weight_on_input, **kw)
        if 'w' not in cap:
            cap['w'] = (w1_fp4, w1_blockscale, w1_alphas, w2_fp4, w2_blockscale, w2_alphas)
            cap['g'] = (a1_gscale.detach().clone(), a2_gscale.detach().clone())
            cap['meta'] = (m, n, k, e, device, activation)
            print("GOT_WEIGHTS", flush=True)
        return r
    cm.run_cutlass_moe_fp4 = patched
    llm = LLM(model=M, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.85,
              dtype="auto", max_model_len=4096, enable_prefix_caching=False, block_size=32, max_num_seqs=1)
    llm.generate(prompts=[{"prompt_token_ids": [1, 2, 3]}], sampling_params=llm.get_default_sampling_params() if hasattr(llm,'get_default_sampling_params') else None)
    if 'w' not in cap:
        print("NO_CAPTURE", flush=True); return
    # 读固定入参
    a = torch.from_numpy(np.fromfile(f"{d_in}/a.bin", dtype=np.uint16)).cuda()
    a = (a.to(torch.int32) << 16).view(torch.float32).to(torch.bfloat16).reshape(24, 2816)
    ti = torch.from_numpy(np.fromfile(f"{d_in}/topk_ids.bin", dtype=np.int32)).cuda().reshape(24, 8)
    tw = torch.from_numpy(np.fromfile(f"{d_in}/topk_w.bin", dtype=np.float32)).cuda().reshape(24, 8)
    w1p, w1s, w1a, w2p, w2s, w2a = cap['w']
    a1g, a2g = cap['g']
    m, n, k, e, dev, act = cap['meta']
    out = torch.zeros(m, k, dtype=torch.bfloat16, device=dev)
    ws13 = torch.zeros(m * 2 * n * e, dtype=torch.bfloat16, device=dev)
    ws2 = torch.zeros(m * k * e, dtype=torch.bfloat16, device=dev)
    cm.run_cutlass_moe_fp4(out, a, a1g, w1p, w1s, w1a, a2g, w2p, w2s, w2a, tw, ti, act, ws13, ws2,
                           m, n, k, e, dev, False)
    out.to(torch.bfloat16).view(torch.int16).cpu().numpy().tofile(f"{d_out}/y.bin")
    # 与抓到的前向输出比（验证该入参组合可复现参照）
    ref = np.fromfile(f"{d_in}/y_ref.bin", dtype=np.uint16)
    o = np.fromfile(f"{d_out}/y.bin", dtype=np.uint16)
    print("STANDALONE vs REF ndiff=%d/%d" % (int((ref != o).sum()), ref.size), flush=True)
if __name__ == '__main__':
    main()
