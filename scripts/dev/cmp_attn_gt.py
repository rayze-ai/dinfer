#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""地面真值对照：用 vLLM 自己的 q/k/v 做 fp32 softmax，看 vLLM 真实 output 离它多远。
若 vLLM output 也离 fp32 ~2%，说明 2.2% 是 fp8 attention 固有噪声，不是我方 kernel 错。
跑：docker exec kaon-oracle-1 python3 /cmp_attn_gt.py /vllm_attn_io 22
"""
import sys
import numpy as np
import torch

vd, T = sys.argv[1], int(sys.argv[2])
NH, NKV, D, WIN = 16, 8, 256, 1024


def vl(nm, w):
    a = np.fromfile(f"{vd}/vllm_{nm}.bin", dtype=np.float32)
    return torch.from_numpy(a).cuda().reshape(-1, w)[:T]


def fp8rt(x):
    return x.bfloat16().to(torch.float8_e4m3fn).to(torch.float32)


q = vl("q", NH * D)          # 已是 fp8 值（存 f32）
k = fp8rt(vl("k", NKV * D))  # vLLM 写 fp8 cache → dequant；复刻
v = fp8rt(vl("v", NKV * D))
out_vllm = vl("out", NH * D)

q = q.reshape(T, NH, D).float()
k = k.reshape(T, NKV, D).float()
v = v.reshape(T, NKV, D).float()
g = NH // NKV
gt = torch.zeros(T, NH, D, device="cuda")
pos = torch.arange(T, device="cuda")
mask = (pos[None, :] <= pos[:, None]) & (pos[:, None] - pos[None, :] < WIN)
for h in range(NH):
    qh, kh, vh = q[:, h], k[:, h // g], v[:, h // g]
    logits = (qh @ kh.t()).masked_fill(~mask, float("-inf"))
    p = torch.softmax(logits, dim=-1)
    gt[:, h] = p @ vh
gt = gt.reshape(T, NH * D)

for nm, a in [("vLLM_out", out_vllm)]:
    rel = (a.float() - gt).norm().item() / gt.norm().item()
    mx = (a.float() - gt).abs().max().item()
    print(f"{nm} vs fp32-gt: rel={rel:.3e} max|d|={mx:.4e}  ‖gt‖={gt.norm():.2f} ‖{nm}‖={a.float().norm():.2f}")
