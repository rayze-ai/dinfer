#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证"直接调 cuBLAS"能否逐位复现 vLLM 的 fp8 GEMM（用户 2026-10-23 思路）。

若成立，Rust 侧就照同一条 cuBLASLt 调用接（我们已有 `cudarc::cublaslt` 的 LtDescs/heuristics 实现）。
不加载 LLM——直接用已抓好的 A/B/scale + 参考输出，反复调 cuBLAS 比。

注意：vLLM 的真实路径是 flashinfer 的 `bmm_fp8` → `CublasFp8GemmRunner` →
`module.bmm_fp8(a, b, out, scale_a, scale_b, workspace, cublas_handle)`（tactic=-1 走启发式默认）。
本脚本用 `torch._scaled_mm`（PyTorch 内部同样是 cuBLASLt）作为独立第二实现来交叉验证。
"""
import os
import sys

import numpy as np
import torch

D = os.environ.get("D17_DIR", "/w/d17b")


def load(name, dtype, shape=None):
    a = np.fromfile(f"{D}/{name}")
    t = torch.from_numpy(a.view(dtype)).cuda()
    return t.view(shape) if shape else t


m, k, n = 24, 2816, 8192
A = load("A_fp8.bin", np.uint8, (m, k)).view(torch.float8_e4m3fn)
B = load("B_fp8.bin", np.uint8, (k, n)).view(torch.float8_e4m3fn)
ref = load("fwd_ref_bf16.bin", np.int16, (m, n)).view(torch.bfloat16)
scale_a = torch.tensor(0.59375, dtype=torch.float32, device="cuda")
scale_b = torch.tensor(0.0013166156131774187, dtype=torch.float32, device="cuda")


def cmp(x, y, label):
    a = x.flatten().view(torch.int16)
    b = y.flatten().view(torch.int16)
    d = int((a != b).sum().item())
    mx = float((x.float().flatten() - y.float().flatten()).abs().max().item())
    print(f"{label:<34} diff {d}/{a.numel()}  max {mx:.4e}", flush=True)
    return d


print(f"A={tuple(A.shape)} B={tuple(B.shape)} ref={tuple(ref.shape)}", flush=True)

# B 落盘是 **[K,N] 逻辑序**（见 D-16-REBASE §17）。flashinfer 的 `bmm_fp8` 内部要的是
# 物理 [N,K]、以 [K,N] 视图传入；所以传之前必须真的按 [N,K] 排一遍再转视图。
B_nk = B.t().contiguous()          # 物理 [N,K] 行主序
B_view = B_nk.t()                  # [K,N] 视图（stride=(1,K)），与 vLLM 的喂法一致

# 1. torch._scaled_mm：需要 b 是 [K,N] 且列主序（即物理 [N,K] 的转置视图）
try:
    r = torch._scaled_mm(A, B_nk.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
    cmp(r, ref, "torch._scaled_mm (cublas 路径)")
except Exception as e:  # noqa: BLE001
    print(f"torch._scaled_mm 失败: {type(e).__name__}: {str(e)[:200]}", flush=True)

# 2. 重复 3 次看确定性
try:
    rs = [torch._scaled_mm(A, B_nk.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
          for _ in range(3)]
    det = all(int((rs[0].flatten().view(torch.int16) != t.flatten().view(torch.int16)).sum()) == 0
              for t in rs[1:])
    print(f"torch._scaled_mm 重复 3 次确定性: {det}", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"重复测试失败: {e}", flush=True)

# 3. 对照：flashinfer 的 cublas backend（已知应为 0）
try:
    from flashinfer import bmm_fp8
    o = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    bmm_fp8(A.unsqueeze(0), B_view.unsqueeze(0), scale_a, scale_b, torch.bfloat16, o.unsqueeze(0), "cublas")
    cmp(o, ref, "flashinfer bmm_fp8(cublas)")
except Exception as e:  # noqa: BLE001
    print(f"flashinfer 失败: {type(e).__name__}: {str(e)[:200]}", flush=True)

print("DONE", flush=True)
