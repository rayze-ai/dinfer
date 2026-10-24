#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""attention(gqa) 隔离：喂我方 q_rope/k_rope/v_norm（attention 的输入），按 vLLM 口径
（q/k/v fp8 往返 scale=1.0 + GQA + 因果 + 滑窗1024 + fp32 softmax/累加）算参照，比我方 attn_out.bin。
目的：确认 L0 的主残差是否在 attention kernel 内部。
运行：docker exec kaon-oracle-1 python3 /seg_attn.py <dump_dir>
"""
import sys
import numpy as np
import torch

D, NH, NKV, WIN = 256, 16, 8, 1024


def rd(dd, name, width):
    v = np.fromfile(f"{dd}/{name}.bin", dtype=np.float32)
    return torch.from_numpy(v).cuda().reshape(-1, width)


def fp8rt(x):  # e4m3 往返，scale=1.0（vLLM kv_cache fp8，k/v 写 cache 量化、读 dequant；q 同）
    return x.to(torch.float8_e4m3fn).to(torch.bfloat16)


def main():
    dd = sys.argv[1]
    q = rd(dd, "q_rope", NH * D).bfloat16()   # [T,4096]
    k = rd(dd, "k_rope", NKV * D).bfloat16()  # [T,2048]
    v = rd(dd, "v_norm", NKV * D).bfloat16()  # [T,2048]
    T = q.shape[0]
    # vLLM：q 量化 fp8 进 kernel；k/v 写 fp8 cache 再 dequant 回 bf16。scale 全 1.0。
    q = fp8rt(q).reshape(T, NH, D)
    k = fp8rt(k).reshape(T, NKV, D)
    v = fp8rt(v).reshape(T, NKV, D)
    # GQA：16 q 头分 8 组，每组 2 个 q 共享一个 kv 头
    g = NH // NKV
    out = torch.zeros(T, NH, D, dtype=torch.float32, device="cuda")
    # 因果 + 滑窗（窗口 1024；T<1024 时只有因果）
    pos = torch.arange(T, device="cuda")
    mask = (pos[None, :] <= pos[:, None]) & (pos[:, None] - pos[None, :] < WIN)  # [T,T] True=可见
    for h in range(NH):
        kh = k[:, h // g, :].float()   # [T,D]
        vh = v[:, h // g, :].float()
        qh = q[:, h, :].float()        # [T,D]
        logits = (qh @ kh.t()) * 1.0   # scale=1.0（Gemma4 QK-norm 承担）
        logits = logits.masked_fill(~mask, float("-inf"))
        p = torch.softmax(logits, dim=-1)  # fp32
        out[:, h, :] = p @ vh
    ref = out.reshape(T, NH * D).bfloat16()
    ours = rd(dd, "attn_out", NH * D).bfloat16()
    nd = int((ref != ours).sum())
    mx = (ref.float() - ours.float()).abs().max().item()
    rel = (ref.float() - ours.float()).norm().item() / max(ref.float().norm().item(), 1e-30)
    print(f"attn_out: ref(fp32 softmax) vs ours  不同 {nd}/{ref.numel()}  max|d|={mx:.4e}  rel={rel:.3e}")
    print(f"  ‖ref‖={ref.float().norm():.3f}  ‖ours‖={ours.float().norm():.3f}  T={T}")
    print(f"  ref[0,:4]={ref[0,:4].float().tolist()}")
    print(f"  ours[0,:4]={ours[0,:4].float().tolist()}")


if __name__ == "__main__":
    main()
