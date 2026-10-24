#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐段复现 vLLM compile 口径的 L0，与我们 dump 的中间量逐段比。

依据：oracle 容器里的 `computation_graph.py`（ATen 层图），数据流已写进
`docs/loops/kaon3-layer1.md` 第 3 轮。本脚本按图**逐步复现**，每步与我们对应的 .bin 比。

运行（oracle 容器内）：
    docker exec kaon-oracle-1 python3 /seg_l0.py <our_dump_dir_in_container> <emb.bin>
"""
import sys
import glob
import numpy as np
import torch
import safetensors.torch as st
from vllm.ir.ops.layernorm import rms_norm

SNAP = glob.glob("/models/models--kaon*/snapshots/*/")[0]
P = "model.language_model.layers.0."
T, H = 24, 2816


def load_weights():
    W = {}
    for f in sorted(glob.glob(SNAP + "*.safetensors")):
        d = st.load_file(f)
        for k in d:
            if k.startswith(P):
                W[k[len(P):]] = d[k].cuda()
    return W


def ours(d, name, shape=None):
    v = np.fromfile(f"{d}/{name}.bin", dtype=np.float32)
    n = shape if shape else v.size // H
    return torch.from_numpy(v).cuda().reshape(n, H)


def cmp(tag, ref, our):
    if ref.shape != our.shape:
        print(f"{tag:22s} 形状不同 ref{tuple(ref.shape)} ours{tuple(our.shape)}")
        return
    nd = int((ref != our).sum())
    d = (ref.float() - our.float()).abs().max().item()
    rel = (ref.float() - our.float()).norm().item() / max(ref.float().norm().item(), 1e-30)
    flag = "✅" if nd == 0 else ("~" if rel < 5e-3 else "❌")
    print(f"{tag:22s} {flag} 不同 {nd:6d}/{ref.numel():6d}  max|d|={d:.4e}  rel={rel:.3e}")


def main():
    d = sys.argv[1]
    emb_path = sys.argv[2] if len(sys.argv) > 2 else f"{d}/Lm1.bin"
    W = load_weights()
    emb = ours(d, "Lm1").bfloat16()

    # ---- 1. input_norm ----
    inn = rms_norm(emb, W["input_layernorm.weight"], 1e-6)
    cmp("input_norm", inn, ours(d, "input_norm").bfloat16())

    # ---- 2. FFN 支路（与 attention 无关，先做能做的）----
    # 需要 add（= post_attn_norm(o) + embed），但我们没有参照的 o_proj，
    # 所以这里只能用**我方**的 add 当输入，测"给定同一 add，后续段是否一致"。
    # 这样能把"attention 段错"与"FFN 段错"分开。
    our_add = ours(d, "post_attn_norm") + emb          # 我方 add（bf16 相加）
    our_add = our_add.bfloat16()

    pre_ff1 = rms_norm(our_add, W["pre_feedforward_layernorm.weight"], 1e-6)
    cmp("pre_ff_norm", pre_ff1, ours(d, "pre_ff_norm").bfloat16())

    pre_ff2 = rms_norm(our_add, W["pre_feedforward_layernorm_2.weight"], 1e-6)
    cmp("pre_ff2_norm", pre_ff2, ours(d, "pre_ff2_norm").bfloat16())

    router_in = rms_norm(our_add, None, 1e-6)
    root_size = W["router.root_size"].float() if "router.root_size" in W else None
    if root_size is not None:
        rs = root_size.bfloat16()
        sc = W["router.scale"].bfloat16()
        r1 = (router_in * rs).bfloat16()
        r2 = (r1 * sc).bfloat16()
        cmp("router_in(3步)", r2, ours(d, "router_in").bfloat16())
        # 融合版：fp32 连乘后舍一次
        rf = ((router_in.float() * float(root_size)) * W["router.scale"].float()).bfloat16()
        cmp("router_in(融合)", rf, ours(d, "router_in").bfloat16())

    # ---- 3. post_ff_norm 的输入合成 ----
    # 图：sum2 = post_ff_norm_1(mlp_out) + post_ff_norm_2(moe_out)，再 rms_norm
    ff1 = ours(d, "ff1").bfloat16()
    ff2 = ours(d, "ff2").bfloat16()
    sum2 = (ff1 + ff2).bfloat16()
    cmp("ff1(vs 我方 ff1)", ff1, ours(d, "ff1").bfloat16())
    pfn_ref = rms_norm(sum2, W["post_feedforward_layernorm.weight"], 1e-6)
    cmp("post_ff_norm", pfn_ref, ours(d, "post_ff_norm").bfloat16())


if __name__ == "__main__":
    main()
