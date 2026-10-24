#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用参照自己的 c3 与 y_moe 反推 combine 的确切口径（不猜，穷举 + 最小二乘）。

背景：MoE 内部四个分段（量化 → gate/up → gelu·mul → down）已全部逐位一致，
唯一剩下 `output = (c3.view(m,topk,k) * topk_weights.view(m,topk,1).to(bf16)).sum(dim=1)`。

`/w/tmp/moe3b` 里的 `c3_down.bin` 与 `y_moe.bin` **来自同一次运行**，所以：
  ① 先用最小二乘从 (c3, y) 反解每行的 topk 权重（2816 个方程解 8 个未知数，误差应逼近 0）
  ② 再用这些权重穷举 combine 的舍入/累加口径，看哪一种能逐位复现 y

判据：逐位命中率。100% 的那个口径就是答案。
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")


def main():
    d = os.environ.get("SEG_DIR", "/w/tmp/moe3b")
    rows, h, topk = 32768, 2816, 8
    m = rows // topk

    c3 = torch.from_numpy(np.fromfile(f"{d}/c3_down.bin", dtype=np.uint16)).cuda() \
        .view(torch.bfloat16).reshape(rows, h)
    y = torch.from_numpy(np.fromfile(f"{d}/y_moe.bin", dtype=np.uint16)).cuda() \
        .view(torch.bfloat16).reshape(m, h)
    print(f"c3={tuple(c3.shape)} y={tuple(y.shape)}", flush=True)

    c3v = c3.view(m, topk, h)
    yf = y.to(torch.float32)

    # ---- ① 最小二乘反解权重（每行独立，8×8 正规方程）----
    A = c3v.to(torch.float32)                     # [m, topk, h]
    G = torch.einsum("mth,msh->mts", A, A) + 1e-6 * torch.eye(topk, device="cuda")
    b = torch.einsum("mth,mh->mt", A, yf)
    w = torch.linalg.solve(G, b)                  # [m, topk]
    resid = (torch.einsum("mth,mt->mh", A, w) - yf).norm() / yf.norm()
    print(f"最小二乘残差 relerr={resid.item():.3e}（应接近 0）", flush=True)
    print("w[0] =", [round(v, 6) for v in w[0].tolist()], flush=True)
    print("w 行和 [0..4] =", [round(v, 6) for v in w.sum(1)[:5].tolist()], flush=True)

    # ---- ② 穷举 combine 口径 ----
    def hit(yv, tag):
        yb = yv.to(torch.bfloat16)
        same = (yb.view(torch.int16) == y.view(torch.int16)).float().mean().item()
        # 每个元素差几个 bf16 ULP
        dlt = (yb.to(torch.float32) - yf).abs()
        print(f"{tag:44s} 逐位命中 {same * 100:6.2f}%  mean|d|={dlt.mean().item():.3e}", flush=True)
        return same

    wf = w                                            # f32
    wb = w.to(torch.bfloat16)                         # bf16 权重
    res = {}

    # 口径 A：f32 权重、f32 乘积、f32 顺序累加
    res["A f32 w, f32 prod, seq sum"] = hit(
        torch.zeros_like(yf).index_add(0, torch.arange(m, device="cuda"),
                                       torch.einsum("mth,mt->mh", A, wf)), "A f32 w, f32 prod, seq(torch einsum)")

    # 口径 B：bf16 权重、bf16 乘积、f32 顺序累加（参照 `.to(out_dtype)` 的字面意思）
    prod_b = (c3v * wb.view(m, topk, 1)).to(torch.bfloat16)
    res["B bf16 w, bf16 prod, f32 sum"] = hit(prod_b.to(torch.float32).sum(1), "B bf16 w, bf16 prod, f32 sum")

    # 口径 C：bf16 权重、bf16 乘积、bf16 逐次累加
    acc = torch.zeros(m, h, dtype=torch.bfloat16, device="cuda")
    for t in range(topk):
        acc = (acc + prod_b[:, t]).to(torch.bfloat16)
    res["C bf16 w, bf16 prod, bf16 acc"] = hit(acc.to(torch.float32), "C bf16 w, bf16 prod, bf16 acc")

    # 口径 D：bf16 权重、f32 乘积（只权重舍 bf16）
    res["D bf16 w only"] = hit(
        torch.einsum("mth,mt->mh", A, wb.to(torch.float32)), "D bf16 w only, f32 prod")

    # 口径 E：直接用 torch 的 `(x * w).sum(1)`（照抄参照那行 Python，含其 kernel 行为）
    res["E torch: (bf16 * bf16).sum(1)"] = hit(
        (c3v * wb.view(m, topk, 1)).sum(1).to(torch.float32), "E torch (c3*w_bf16).sum(1)")
    res["F torch: (bf16 * f32cast).sum(1)"] = hit(
        (c3v * w.view(m, topk, 1).to(torch.bfloat16)).sum(1).to(torch.float32), "F torch (c3*w.to(bf16)).sum(1)")

    # 口径 G：权重按 f32 乘、但乘积不额外舍（对照，确认"是否必须舍 bf16"）
    res["G f32 w, prod rounding off"] = hit(torch.einsum("mth,mt->mh", A, wf), "G f32 w, prod rounding off")

    best = max(res, key=res.get)
    print(f"\n最佳口径: {best}  命中 {res[best] * 100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
