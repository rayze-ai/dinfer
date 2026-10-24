#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第一步：确认进 attention 前 dinfer 与 vLLM 的 q/k/v 逐位一致。
vLLM 侧（kaon3_vllm_attn_io.py 抓）：vllm_q(fp8 值,存 f32) / vllm_k,v(bf16,存 f32)，T 可能 padding。
我方侧（kaon3_probe SUB_LAYER dump）：q_rope/k_rope(bf16,fp8 往返前) / v_norm。
对比：q → fp8 往返我方 q_rope 比 vllm_q；k/v → 我方 bf16 比 vllm bf16。
跑：python3 cmp_attn_in.py <vllm_dir> <ours_dir> <T>
"""
import sys
import numpy as np
import torch

vd, od, T = sys.argv[1], sys.argv[2], int(sys.argv[3])
NH, NKV, D = 16, 8, 256


def vl(nm, w):
    a = np.fromfile(f"{vd}/vllm_{nm}.bin", dtype=np.float32)
    return torch.from_numpy(a).cuda().reshape(-1, w)[:T]


def ou(nm, w):
    a = np.fromfile(f"{od}/{nm}.bin", dtype=np.float32)
    return torch.from_numpy(a).cuda().reshape(-1, w)[:T]


def cmp(tag, ref, our):
    nd = int((ref != our).sum())
    mx = (ref.float() - our.float()).abs().max().item()
    rel = (ref.float() - our.float()).norm().item() / max(ref.float().norm().item(), 1e-30)
    flag = "✅" if nd == 0 else ("~" if rel < 5e-3 else "❌")
    print(f"{tag:14s} {flag} 不同 {nd:6d}/{ref.numel():6d}  max|d|={mx:.4e}  rel={rel:.3e}  ‖ref‖={ref.float().norm():.2f} ‖our‖={our.float().norm():.2f}")


# vLLM q 是 fp8 值（存 f32）；我方 q_rope bf16 → fp8 往返再比
vq = vl("q", NH * D)
oq = ou("q_rope", NH * D).bfloat16().to(torch.float8_e4m3fn).to(torch.float32)
cmp("q(fp8)", vq, oq)
# k/v：vLLM bf16 vs 我方 bf16（我方 k_rope/v_norm 是 fp8 往返前；vLLM 进 attn 也是 bf16，但 vLLM 写 cache 前没 fp8——这里先比 bf16 原值）
cmp("k(bf16)", vl("k", NKV * D), ou("k_rope", NKV * D).bfloat16().float())
cmp("v(bf16)", vl("v", NKV * D), ou("v_norm", NKV * D).bfloat16().float())
