#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用 FlashInfer bmm_fp8 参照：任意单权重 fp8 线性层。用真实输入 dump + checkpoint 权重。
存 A_fp8/W_fp8/ref_out/meta，Rust fp8_qkv_fi_parity（通用版）读同一份比我方 cuBLASLt。
跑：OUT=/w/fi_oproj WEIGHT=self_attn.o_proj INDUMP=/w/sd_now/attn_out.bin K=4096 \
     python3 -I scripts/dev/fi_lin_ref.py
"""
import os, glob
import numpy as np
import torch
import safetensors.torch as st
import flashinfer

out = os.environ["OUT"]; os.makedirs(out, exist_ok=True)
wname = os.environ["WEIGHT"]          # 如 self_attn.o_proj
indump = os.environ["INDUMP"]         # 我方 dump 的输入 .bin（f32，[T,K]）
K = int(os.environ["K"])              # 输入维
SNAP = glob.glob("/models/models--kaon*/snapshots/*/")[0]
P = f"model.language_model.layers.0.{wname}"

# 权重 + scale
W = Wsc = Xsc = None
for f in sorted(glob.glob(SNAP + "*.safetensors")):
    d = st.load_file(f)
    if P + ".weight" in d:
        W = d[P + ".weight"].cuda()                       # fp8 [N,K]
        Wsc = float(d[P + ".weight_scale"])
        Xsc = float(d[P + ".input_scale"])
        break
assert W is not None, wname
N = W.shape[0]
assert W.shape[1] == K, f"K 不符 {W.shape} vs {K}"

# 输入：我方 dump f32 [T,K] → bf16 → fp8（per-tensor Xsc）
x = torch.from_numpy(np.fromfile(indump, dtype=np.float32)).cuda().reshape(-1, K).bfloat16()
T = x.shape[0]
a_scale = torch.tensor([Xsc], dtype=torch.float32, device="cuda")
w_scale = torch.tensor([Wsc], dtype=torch.float32, device="cuda")
A = (x.float() / a_scale).clamp(-448, 448).to(torch.float8_e4m3fn)   # [T,K]

b = W.t().unsqueeze(0)                # [1,K,N] 列主序
ref = flashinfer.bmm_fp8(A.unsqueeze(0), b, a_scale, w_scale, torch.bfloat16).squeeze(0)  # [T,N]

A.view(torch.uint8).cpu().numpy().tofile(f"{out}/A_fp8.bin")
W.view(torch.uint8).cpu().numpy().tofile(f"{out}/W_qkv_fp8.bin")  # 复用 Rust 读的名
ref.float().cpu().numpy().tofile(f"{out}/ref_out.bin")
np.array([Xsc, Wsc, T, N, K], dtype=np.float32).tofile(f"{out}/meta.bin")
# 地面真值
gt = (A.to(torch.float32) * Xsc) @ (W.to(torch.float32) * Wsc).t()
rel = (ref.float() - gt).norm().item() / gt.norm().item()
print(f"[fi_lin {wname}] T={T} N={N} K={K} Xsc={Xsc:.5f} Wsc={Wsc:.6f} ‖ref‖={ref.float().norm():.1f} rel(ref→gt)={rel:.3e}")
