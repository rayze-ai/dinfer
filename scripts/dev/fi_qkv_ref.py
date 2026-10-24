#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""造 FlashInfer bmm_fp8 的 qkv GEMM 参照：用真实 input_norm + 合并 qkv 权重，调 flashinfer.bmm_fp8。
存：A_fp8.bin [T,2816] e4m3 字节、W_qkv_fp8.bin [8192,2816] e4m3 字节、scales、ref_out.bin [T,8192] bf16。
Rust 侧用同一份 A/W/scale 跑 forward_fp8_cublaslt，逐位比 ref_out（= attention T-sweep 的 qkv 版）。
跑：OUT=/w/fiqkv python3 -I scripts/dev/fi_qkv_ref.py <dump_dir(含 input_norm.bin)>
"""
import sys, os, glob
import numpy as np
import torch
import safetensors.torch as st
import flashinfer

dd = sys.argv[1]
out = os.environ.get("OUT", "/w/fiqkv")
os.makedirs(out, exist_ok=True)
SNAP = glob.glob("/models/models--kaon*/snapshots/*/")[0]
P = "model.language_model.layers.0.self_attn."
H = 2816

# 1) 真实 input_norm（我方 dump，f32）→ bf16 → fp8（input_scale=0.59375，per-tensor）
inn = np.fromfile(f"{dd}/input_norm.bin", dtype=np.float32)
T = inn.size // H
x = torch.from_numpy(inn).cuda().reshape(T, H).bfloat16()
a_scale = torch.tensor([0.59375], dtype=torch.float32, device="cuda")
A = (x.float() / a_scale).clamp(-448, 448).to(torch.float8_e4m3fn)  # [T,2816]

# 2) 合并 qkv 权重：按 requantize_with_max_scale 复刻（q,k,v 各自 scale → max）
def load(name):
    for f in sorted(glob.glob(SNAP + "*.safetensors")):
        d = st.load_file(f)
        if P + name + ".weight" in d:
            return d[P + name + ".weight"].cuda(), float(d[P + name + ".weight_scale"])
    raise SystemExit(name)
parts = [load("q_proj"), load("k_proj"), load("v_proj")]  # [(w_fp8[n,2816], scale)]
smax = max(s for _, s in parts)
w_scale = torch.tensor([smax], dtype=torch.float32, device="cuda")
rows = []
for w, s in parts:
    dq = (w.to(torch.float16) * np.float16(s))            # fp16 dequant（w8a8 口径）
    rq = (dq.to(torch.float32) / smax).clamp(-448, 448).to(torch.float8_e4m3fn)
    rows.append(rq)
W = torch.cat(rows, dim=0)  # [8192, 2816] fp8

# 3) FlashInfer bmm_fp8：batched 3D，a[1,T,2816] × b[1,2816,8192]（b 列主序）→ [1,T,8192] bf16
#    与 compile 图一致（flashinfer_scaled_fp8_mm 里 a.unsqueeze(0)/b.unsqueeze(0)）
b = W.t().unsqueeze(0)  # [1, 2816, 8192] 列主序（W 行主序[n,k] → .t() 即列主序[k,n]；不能 contiguous）
ref = flashinfer.bmm_fp8(A.unsqueeze(0), b, a_scale, w_scale, torch.bfloat16).squeeze(0)  # [T,8192]

# 存字节供 Rust 读
A.view(torch.uint8).cpu().numpy().tofile(f"{out}/A_fp8.bin")      # [T,2816] u8(e4m3)
W.view(torch.uint8).cpu().numpy().tofile(f"{out}/W_qkv_fp8.bin")  # [8192,2816] u8
ref.float().cpu().numpy().tofile(f"{out}/ref_out.bin")            # [T,8192] f32(bf16 值)
np.array([float(a_scale), smax, T], dtype=np.float32).tofile(f"{out}/meta.bin")
print(f"[fi_qkv] T={T} a_scale={float(a_scale)} w_scale={smax} ‖ref‖={ref.float().norm():.2f}")
print(f"  ref[0,:4]={ref[0,:4].float().tolist()}")
# 地面真值：dequant A/W 做 f32 矩阵乘，判 FlashInfer ref 是否合理（排除我造数据的布局/scale 错）
Adq = A.to(torch.float32) * a_scale        # [T,2816]
Wdq = W.to(torch.float32) * smax           # [8192,2816]
gt = (Adq @ Wdq.t())                        # [T,8192]
rel = (ref.float() - gt).norm().item() / gt.norm().item()
print(f"  [地面真值] ‖gt‖={gt.norm():.2f}  rel(ref→gt)={rel:.3e}  gt[0,:4]={gt[0,:4].tolist()}")
