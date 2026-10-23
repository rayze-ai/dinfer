#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 用**真实权重 + 真实激活**跑参照的 dense FP4 GEMM（ops.cutlass_scaled_fp4_mm），
# 导出 (A_bf16, 量化后的 A_fp4/scale, B_fp4/scale, alpha, 输出)，供 Rust 侧逐位比。
# 判据：我们的 gemm_nvfp4 与它逐位相同 → 专家 GEMM 没问题，差异在别处；不同 → 差异就是它。
import os, numpy as np, torch
def main():
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
    from vllm import LLM
    from vllm.model_executor.layers.fused_moe.experts import cutlass_moe as cm
    M = os.environ["M"]
    cap = {}
    orig = cm.run_cutlass_moe_fp4
    def patched(output, a, a1_gscale, w1_fp4, w1_blockscale, w1_alphas, a2_gscale, w2_fp4,
                w2_blockscale, w2_alphas, topk_weights, topk_ids, activation, workspace13, workspace2,
                m, n, k, e, device, apply_router_weight_on_input=False, **kw):
        r = orig(output, a, a1_gscale, w1_fp4, w1_blockscale, w1_alphas, a2_gscale, w2_fp4,
                 w2_blockscale, w2_alphas, topk_weights, topk_ids, activation, workspace13, workspace2,
                 m, n, k, e, device, apply_router_weight_on_input, **kw)
        if 'w' not in cap:
            cap['w'] = tuple(x.detach().clone() for x in (w1_fp4, w1_blockscale, w1_alphas,
                                                          w2_fp4, w2_blockscale, w2_alphas))
            cap['g'] = (a1_gscale.detach().clone(), a2_gscale.detach().clone())
            cap['meta'] = (m, n, k, e, device)
        return r
    cm.run_cutlass_moe_fp4 = patched
    llm = LLM(model=M, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.85,
              dtype="auto", max_model_len=4096, enable_prefix_caching=False, block_size=32, max_num_seqs=1)
    llm.generate(prompts=[{"prompt_token_ids": [1,2,3]}])
    if 'w' not in cap:
        print("NO_CAPTURE"); return
    w1p, w1s, w1a, w2p, w2s, w2a = cap['w']
    a1g, a2g = cap['g']
    m, n, k, e, dev = cap['meta']
    print("W1", tuple(w1p.shape), tuple(w1s.shape), tuple(w1a.shape), "W2", tuple(w2p.shape), flush=True)
    print("a1_gscale", a1g.shape, a1g[:4].tolist(), flush=True)
    # 取一个真实激活切片：用 moe1/x.bin 的前 8 行做 dense GEMM（专家 0 的 gate/up）
    a = torch.from_numpy(np.fromfile("/w/tmp/moe1/x.bin", dtype=np.uint16)).cuda()
    a = (a.to(torch.int32) << 16).view(torch.float32).to(torch.bfloat16).reshape(24, 2816)[:8]
    from vllm import _custom_ops as ops
    w1 = w1p[0][: n * 2].contiguous()   # expert 0 的 gate+up: [2n, k/2]
    s1 = w1s[0][: n * 2].contiguous()   # [2n, k/16]
    al1 = w1a[0]
    # 激活先量化成 fp4（与 MoE 里 `scaled_fp4_experts_quant` 同口径：n=1 的普通量化）
    # MoE 路径用的是 swizzled 布局（报错明确要求 "padded and swizzled"）
    aq, asq = ops.scaled_fp4_quant(a, a1g[0], True, "none", padded_n=n * 2)
    out = ops.cutlass_scaled_fp4_mm(aq, w1, asq, s1, al1, torch.bfloat16)[:, : n * 2]
    a_q, a_s = aq, asq
    d = "/w/tmp/fp4d"; os.makedirs(d, exist_ok=True)
    a.view(torch.int16).cpu().numpy().tofile(f"{d}/a.bin")
    a_q.view(torch.uint8).cpu().numpy().tofile(f"{d}/a_q.bin")
    a_s.view(torch.uint8).cpu().numpy().tofile(f"{d}/a_s.bin")
    w1.view(torch.uint8).cpu().numpy().tofile(f"{d}/w1.bin")
    s1.view(torch.uint8).cpu().numpy().tofile(f"{d}/s1.bin")
    out.view(torch.int16).cpu().numpy().tofile(f"{d}/y.bin")
    np.float32(al1.item()).tofile(f"{d}/alpha.bin")
    np.float32(a1g[0].item()).tofile(f"{d}/a_gscale.bin")
    open(f"{d}/meta.txt","w").write(f"m=8\nn={n*2}\nk={k}\n")
    print("DUMPED m=8 n=%d k=%d  |y|=%f" % (n*2, k, out.float().norm().item()), flush=True)
if __name__ == '__main__':
    main()
