#!/usr/bin/env python3
"""生成 P2 算子对拍的输入与期望输出（.npy），供 `cargo test -p dinfer-kernels --features ref-tests` 读取。

只依赖 numpy + torch（CPU 即可）。bf16 用 torch.bfloat16 算再转 float32 存，
Rust 侧读 float32 再 cast 到 bf16，保证两边看到同一份 bf16 输入。

用法：python3 scripts/verify/gen_kernel_refs.py <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import torch

torch.manual_seed(1234)
out = Path(sys.argv[1] if len(sys.argv) > 1 else "test/kernel_refs")
out.mkdir(parents=True, exist_ok=True)


def save(name: str, t):
    arr = t.detach().cpu()
    if arr.dtype == torch.bfloat16:
        arr = arr.float()
    np.save(out / f"{name}.npy", arr.numpy())


def bf(x):
    return x.to(torch.bfloat16)


# ---- rmsnorm / fused_add_rmsnorm ----
T, H = 13, 1536
x = bf(torch.randn(T, H) * 2)
res = bf(torch.randn(T, H))
w = bf(torch.rand(H) + 0.5)
eps = 1e-5


def rmsnorm(v):
    v32 = v.float()
    return bf(v32 * torch.rsqrt(v32.pow(2).mean(-1, keepdim=True) + eps) * w.float())


save("rmsnorm_x", x)
save("rmsnorm_w", w)
save("rmsnorm_out", rmsnorm(x))
save("fused_res_in", res)
res_new = bf(x.float() + res.float())
save("fused_res_out", res_new)
save("fused_out", rmsnorm(res_new))

# ---- rope（interleave 与 neox）----
T, QH, KH, HD, RD, MAXPOS, THETA = 9, 4, 2, 128, 64, 4096, 8e6
pos = torch.randint(0, MAXPOS, (T,), dtype=torch.int32)
inv_freq = 1.0 / (THETA ** (torch.arange(0, RD, 2).float() / RD))
freqs = torch.outer(torch.arange(MAXPOS).float(), inv_freq)  # [MAXPOS, RD/2]
cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1)  # [MAXPOS, RD]
q = bf(torch.randn(T, QH, HD))
k = bf(torch.randn(T, KH, HD))


def rope(t, neox):
    t32 = t.float().clone()
    c = cos_sin[pos.long(), : RD // 2][:, None, :]  # [T,1,RD/2]
    s = cos_sin[pos.long(), RD // 2 :][:, None, :]
    r = t32[..., :RD]
    if neox:
        a, b = r[..., : RD // 2], r[..., RD // 2 :]
        na, nb = a * c - b * s, b * c + a * s
        t32[..., : RD // 2], t32[..., RD // 2 : RD] = na, nb
    else:
        a, b = r[..., 0::2], r[..., 1::2]
        na, nb = a * c - b * s, b * c + a * s
        t32[..., 0:RD:2], t32[..., 1:RD:2] = na, nb
    return bf(t32)


save("rope_pos", pos)
save("rope_cos_sin", cos_sin)
save("rope_q", q)
save("rope_k", k)
save("rope_q_out_interleave", rope(q, False))
save("rope_k_out_interleave", rope(k, False))
save("rope_q_out_neox", rope(q, True))
save("rope_k_out_neox", rope(k, True))

# ---- silu_and_mul ----
T, I = 11, 2048
x2 = bf(torch.randn(T, 2 * I))
save("silu_x", x2)
save("silu_out", bf(torch.nn.functional.silu(x2[:, :I].float()) * x2[:, I:].float()))

# ---- embedding ----
V, H = 1000, 256
table = bf(torch.randn(V, H))
ids = torch.randint(0, V, (17,), dtype=torch.int32)
save("emb_table", table)
save("emb_ids", ids)
save("emb_out", table[ids.long()])

# ---- gemm ----
M, N, K = 33, 96, 512
a = bf(torch.randn(M, K))
wt = bf(torch.randn(N, K) * 0.05)
bias = bf(torch.randn(N) * 0.1)
save("gemm_a", a)
save("gemm_w", wt)
save("gemm_bias", bias)
save("gemm_out", bf(a.float() @ wt.float().T + bias.float()))

# ---- argmax / topk_topp_mask ----
B, V = 6, 163840
logits = torch.randn(B, V) * 3
save("argmax_logits", logits)
save("argmax_out", logits.argmax(-1).to(torch.int32))

top_k = torch.tensor([0, 1, 50, 0, 5, 163840], dtype=torch.int32)
top_p = torch.tensor([1.0, 1.0, 1.0, 0.9, 0.5, 0.95], dtype=torch.float32)


def mask_ref(row, kk, pp):
    """与 vLLM `apply_top_k_top_p_pytorch` 同口径：升序排序，top-p 在 **top-k 之后** 的分布上算，
    保留 (1 - cumsum) <= p 的尾部（即累计到 p 为止，含跨过 p 的那个）。"""
    r = row.clone()
    if 0 < kk < r.numel():
        thresh = torch.topk(r, kk).values[-1]
        r[r < thresh] = float("-inf")
    if 0 < pp < 1:
        srt, idx = torch.sort(r, descending=False)
        probs = srt.softmax(-1)
        cum = probs.cumsum(-1)
        drop = cum <= 1 - pp
        drop[-1] = False  # 至少保留最大的那个
        r[idx[drop]] = float("-inf")
    return r


masked = torch.stack([mask_ref(logits[b], int(top_k[b]), float(top_p[b])) for b in range(B)])
save("mask_logits", logits)
save("mask_top_k", top_k)
save("mask_top_p", top_p)
save("mask_out_isinf", torch.isinf(masked).to(torch.int32))

print(f"wrote refs to {out}")
