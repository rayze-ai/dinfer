#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 直接调参照的 cutlass FP4 MoE（`run_cutlass_moe_fp4`），回答"MoE 差异是否出在这里"。
# 做法：hook 真实前向里的 run_cutlass_moe_fp4，抓它的全部入参与输出；
#       然后用同一组入参**再调一次**，与抓到的输出比 → 判定该 kernel 是否自复现；
#       同时与我们的 MoE 输出（/w/tmp/moe1/y.bin）比。
# 跑法：GPU=4 scripts/dev/remote_5090.sh ctr 'cd /w/dinfer && M=<ckpt> IDS="<ids>" python3 -u scripts/dev/cutlass_moe_probe.py'
import os, torch
def main():
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
    from vllm import LLM, SamplingParams
    import vllm.model_executor.layers.fused_moe.experts.cutlass_moe as cm
    M = os.environ["M"]; ids = [int(x) for x in os.environ["IDS"].split()]
    cap = {}
    orig = cm.run_cutlass_moe_fp4
    def patched(output, a, a1_gscale, w1_fp4, w1_blockscale, w1_alphas, a2_gscale, w2_fp4,
                w2_blockscale, w2_alphas, topk_weights, topk_ids, activation, workspace13, workspace2,
                m, n, k, e, device, apply_router_weight_on_input=False, **kw):
        r = orig(output, a, a1_gscale, w1_fp4, w1_blockscale, w1_alphas, a2_gscale, w2_fp4,
                 w2_blockscale, w2_alphas, topk_weights, topk_ids, activation, workspace13, workspace2,
                 m, n, k, e, device, apply_router_weight_on_input, **kw)
        if a.shape[0] == len(ids) and 'done' not in cap:
            cap['args'] = (a.detach().clone(), a1_gscale.detach().clone(), w1_fp4, w1_blockscale, w1_alphas,
                           a2_gscale.detach().clone(), w2_fp4, w2_blockscale, w2_alphas,
                           topk_weights.detach().clone(), topk_ids.detach().clone())
            cap['meta'] = (m, n, k, e, device, apply_router_weight_on_input)
            cap['out'] = output.detach().clone()
            cap['done'] = 1
            print("CAP m,n,k,e =", m, n, k, e, "| a", tuple(a.shape), "| topk", tuple(topk_ids.shape), flush=True)
        return r
    cm.run_cutlass_moe_fp4 = patched
    llm = LLM(model=M, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.85,
              dtype="auto", max_model_len=4096, enable_prefix_caching=False, block_size=32, max_num_seqs=1)
    llm.generate(prompts=[{"prompt_token_ids": ids}], sampling_params=SamplingParams(max_tokens=1, temperature=0))
    if 'done' not in cap:
        print("NO_CAPTURE", flush=True); return
    d = "/w/tmp/cmo"; os.makedirs(d, exist_ok=True)
    (a, a1g, w1p, w1s, w1a, a2g, w2p, w2s, w2a, tw, ti) = cap['args']
    m, n, k, e, dev, arwoi = cap['meta']
    # 复跑一次，看 kernel 自复现
    out2 = torch.zeros_like(cap['out'])
    ws13 = torch.zeros(max(m * 2 * n, 1) * e, dtype=a.dtype, device=dev)
    ws2 = torch.zeros(max(m * k, 1) * e, dtype=a.dtype, device=dev)
    try:
        from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import run_cutlass_moe_fp4 as R
        R(out2, a, a1g, w1p, w1s, w1a, a2g, w2p, w2s, w2a, tw, ti, cm.MoEActivation.SILU, ws13, ws2,
          m, n, k, e, dev, arwoi)
        same = torch.equal(out2.view(torch.int16), cap['out'].view(torch.int16))
        nd = int((out2.view(torch.int16) != cap['out'].view(torch.int16)).sum())
        print(f"REPLAY self-consistent={same} ndiff={nd}/{out2.numel()}", flush=True)
    except Exception as ex:
        print("REPLAY_FAIL", type(ex).__name__, str(ex)[:200], flush=True)
    cap['out'].to(torch.bfloat16).view(torch.int16).cpu().numpy().tofile(f"{d}/y_ref.bin")
    a.to(torch.bfloat16).view(torch.int16).cpu().numpy().tofile(f"{d}/a.bin")
    ti.to(torch.int32).cpu().numpy().tofile(f"{d}/topk_ids.bin")
    tw.to(torch.float32).cpu().numpy().tofile(f"{d}/topk_w.bin")
    # 与我们的 MoE 输出比
    ours = torch.from_numpy(__import__('numpy').fromfile('/w/tmp/moe1/y.bin', dtype='uint16')).cuda()
    ours_f = (ours.to(torch.int32) << 16).view(torch.float32).reshape(cap['out'].shape)
    o = cap['out'].to(torch.float32)
    nd = int((ours_f != o).sum())
    print(f"OURS_VS_CUTLASS ndiff={nd}/{o.numel()} max|d|={(ours_f-o).abs().max().item():.4e}", flush=True)
if __name__ == '__main__':
    main()
