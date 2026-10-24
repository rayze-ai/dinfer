#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""o_proj 隔离单测：喂我方 attn_out.bin，按 vLLM 口径量化 + FlashInfer bmm_fp8，比我方 o_proj.bin。
目的：判 o_proj GEMM 是不是又一个 cuBLASLt(我们) vs FlashInfer(vLLM) 的库差，从而把 attention 单独框出。
运行（oracle 容器内）：docker exec kaon-oracle-1 python3 /seg_oproj.py <dump_dir>
"""
import sys
import glob
import numpy as np
import torch
import safetensors.torch as st
from vllm.utils.flashinfer import flashinfer_scaled_fp8_mm
from vllm.model_executor.layers.quantization.utils.quant_utils import scaled_quantize  # noqa

SNAP = glob.glob("/models/models--kaon*/snapshots/*/")[0]
P = "model.language_model.layers.0.self_attn.o_proj"


def load_w():
    for f in sorted(glob.glob(SNAP + "*.safetensors")):
        d = st.load_file(f)
        if P + ".weight" in d:
            return (d[P + ".weight"].cuda(), d[P + ".weight_scale"].cuda(),
                    d[P + ".input_scale"].cuda())
    raise SystemExit("o_proj not found")


def main():
    dd = sys.argv[1]
    w, ws, xs = load_w()          # w: fp8 [2816,4096]；ws/xs: f32 标量
    attn = np.fromfile(f"{dd}/attn_out.bin", dtype=np.float32)
    T = attn.size // 4096
    x = torch.from_numpy(attn).cuda().reshape(T, 4096).bfloat16()
    # vLLM 静态量化：x_q = clamp(x/xs, ±448).to(fp8)（per-tensor）
    xq = (x.float() / xs).clamp(-448, 448).to(torch.float8_e4m3fn)
    # FlashInfer bmm_fp8：a[T,4096] fp8 × b[4096,2816] fp8（w 要转置成 [K,N]）
    b = w.t().contiguous()        # [4096, 2816]
    out = flashinfer_scaled_fp8_mm(xq, b, xs, ws, torch.bfloat16)  # [T, 2816]
    ours = torch.from_numpy(np.fromfile(f"{dd}/o_proj.bin", dtype=np.float32)).cuda().reshape(T, 2816).bfloat16()
    nd = int((out != ours).sum())
    mx = (out.float() - ours.float()).abs().max().item()
    rel = (out.float() - ours.float()).norm().item() / max(ours.float().norm().item(), 1e-30)
    print(f"o_proj: ref(FlashInfer) vs ours(cuBLASLt)  不同 {nd}/{out.numel()}  max|d|={mx:.4e}  rel={rel:.3e}")
    # 退化检查：各自范数、样本、输入是否合理
    print(f"  ‖ref‖={out.float().norm():.3f}  ‖ours‖={ours.float().norm():.3f}  ‖attn_out‖={x.float().norm():.3f}  xs={float(xs):.5f} ws={float(ws):.6f}")
    print(f"  ref[0,:4]={out[0,:4].float().tolist()}")
    print(f"  ours[0,:4]={ours[0,:4].float().tolist()}")
    print(f"  attn_out[0,:4]={x[0,:4].float().tolist()}  T={T}")
    # bf16 全精度地面真值：反量化权重到 bf16，x@w.T（不量化激活），判 ref/ours 谁贴它
    wdq = (w.float() * float(ws)).bfloat16()        # [2816,4096] fp8→f32×ws→bf16
    gt = (x.float() @ wdq.float().t()).bfloat16()   # [T,2816]
    def rel_to(a):
        return (a.float() - gt.float()).norm().item() / max(gt.float().norm().item(), 1e-30)
    print(f"  ‖gt(bf16)‖={gt.float().norm():.3f}  rel(ref→gt)={rel_to(out):.3e}  rel(ours→gt)={rel_to(ours):.3e}")
    print(f"  gt[0,:4]={gt[0,:4].float().tolist()}")


if __name__ == "__main__":
    main()
