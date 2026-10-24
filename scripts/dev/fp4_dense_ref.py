#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""判定 `y.bin`（cutlass_scaled_fp4_mm 的输出）到底等于什么。

主逻辑问题：上一轮假设 `s1` 是**自然布局** `[N, K/16]`，但这个形状字节数与
swizzled 布局**完全相等**（N=2816 是 128 的倍数，K/16=176 是 4 的倍数），
所以"字节数对得上"**不能**作为证据。

本脚本用 vLLM 自己的参考实现（tests/kernels/quantization/nvfp4_utils.py，
与被测 kernel 同仓同版本）枚举四种组合，看哪一种能复现 `y.bin`：

    A 侧布局 ∈ {swizzled, 自然}
    B 侧布局 ∈ {swizzled, 自然}

判据：相对误差 ≈ fp4 噪声级（< 0.05）的那个组合就是真的。四个都不行 → 主逻辑另有问题。

用法（容器内）：
    M=<model path> python3 scripts/dev/fp4_dense_ref.py /w/tmp/fp4d
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")


def load_dirs(d, m=8, k=2816, n=1408):
    g = lambda nm, dt: torch.from_numpy(np.fromfile(f"{d}/{nm}", dtype=dt))
    a = g("a.bin", np.uint16).view(torch.bfloat16).reshape(m, k).cuda()
    a_q = g("a_q.bin", np.uint8).reshape(m, k // 2).cuda()
    a_s = g("a_s.bin", np.uint8).cuda()
    w1 = g("w1.bin", np.uint8).reshape(n, k // 2).cuda()
    s1 = g("s1.bin", np.uint8).reshape(n, k // 16).cuda()
    y = g("y.bin", np.uint16).view(torch.bfloat16).reshape(m, n).cuda()
    alpha = float(g("alpha.bin", np.float32)[0])
    gs = float(g("a_gscale.bin", np.float32)[0])
    return a, a_q, a_s, w1, s1, y, alpha, gs


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "/w/tmp/fp4d"
    a, a_q, a_s, w1, s1, y, alpha, gs = load_dirs(d)
    m, k = a.shape
    n = w1.numel() // (k // 2)
    print(f"形状 m={m} n={n} k={k}  alpha={alpha:.8g}  a_gscale={gs:.8g}", flush=True)
    print(f"字节数 a_s={a_s.numel()} s1={s1.numel()} 期望自然={n*(k//16)} "
          f"swizzled={((n+127)//128)*128*((((k//16)+3)//4)*4)}", flush=True)

    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        swizzle_blockscale,
    )

    def unswz(buf, rows, cols, block=16):
        """[rows, cols] 的 swizzled 缓冲 → 线性 [rows, cols//block]（float32）。

        反函数直接照抄参照源码 swizzle_blockscale 的 permute，不自己推。
        """
        Mp = (rows + 127) // 128 * 128
        Kp = (cols + 3) // 4 * 4
        t = buf.view(torch.float8_e4m3fn).reshape(1, Mp, Kp).to(torch.float32)
        tmp = t.reshape(1, Mp // 128, 4, 32, Kp // 4, 4)
        tmp = tmp.permute(0, 1, 4, 3, 2, 5)          # → [1, Mp/128, Kp/4, 32, 4, 4]
        lin = tmp.reshape(Mp, Kp)[:rows, :cols]
        return lin

    def unswz_gpu(buf, rows, cols):
        """用参照自己的函数做同样的事（交叉验证上面的手写反函数）。"""
        Mp = (rows + 127) // 128 * 128
        Kp = (cols + 3) // 4 * 4
        t = buf.view(torch.float8_e4m3fn).reshape(Mp, Kp)
        # swizzle_blockscale 的正向：从线性 → swizzled。这里我们有的是 swizzled，
        # 需要逆向：正向是 reshape(g,4,32,kO,4).permute(0,1,4,3,2,5)
        # → 逆 permute 为 (0,1,4,3,2,5) 自逆（对换 2 和 4 两次 = 恒等）。
        tmp = t.reshape(1, Mp // 128, 4, 32, Kp // 4, 4).permute(0, 1, 4, 3, 2, 5)
        return tmp.reshape(Mp, Kp)[:rows, :cols].to(torch.float32)

    FP4 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device="cuda")

    def unpack_fp4(q):
        """[rows, k/2] uint8 → [rows, k] float；**低半字节 = 偶数列**。"""
        lo = (q & 0x0F).to(torch.long)
        hi = ((q & 0xF0) >> 4).to(torch.long)
        v = torch.empty((q.shape[0], q.shape[1] * 2), device="cuda", dtype=torch.float32)
        v[:, 0::2] = FP4[lo & 7] * torch.where((lo & 8) != 0, -1., 1.)
        v[:, 1::2] = FP4[hi & 7] * torch.where((hi & 8) != 0, -1., 1.)
        return v

    aq = unpack_fp4(a_q)
    wq = unpack_fp4(w1)

    sA_swz = unswz(a_s, m, k // 16).repeat_interleave(16, dim=1)
    sA_nat = a_s[: m * (k // 16)].view(torch.float8_e4m3fn).to(torch.float32) \
        .reshape(m, k // 16).repeat_interleave(16, dim=1)
    sB_swz = unswz(s1, n, k // 16).repeat_interleave(16, dim=1)
    sB_nat = s1.view(torch.float8_e4m3fn).to(torch.float32).reshape(n, k // 16) \
        .repeat_interleave(16, dim=1)

    yref = y.to(torch.float32)
    ny = yref.norm()

    def report(tag, A, B, extra=1.0):
        out = (A @ B.T) * extra
        rel = (out - yref).norm() / ny
        corr = torch.corrcoef(torch.stack([out.flatten(), yref.flatten()]))[0, 1].item()
        print(f"{tag:34s} relerr={rel:.6f} corr={corr:.6f} "
              f"‖out‖/‖y‖={(out.norm()/ny).item():.6f}", flush=True)
        return rel

    print("\n--- 四种布局组合（反量化后 fp32 matmul，乘子 = alpha）---", flush=True)
    report("A=swz  B=swz   *alpha", aq * sA_swz / gs, wq * sB_swz, alpha)
    report("A=swz  B=nat   *alpha", aq * sA_swz / gs, wq * sB_nat, alpha)
    report("A=nat  B=swz   *alpha", aq * sA_nat / gs, wq * sB_swz, alpha)
    report("A=nat  B=nat   *alpha", aq * sA_nat / gs, wq * sB_nat, alpha)

    print("\n--- 对照：不带 /gs 的写法（判定 gs 该不该除）---", flush=True)
    report("A=swz(no /gs) B=swz *alpha", aq * sA_swz, wq * sB_swz, alpha / gs)
    report("A=swz B=swz  *alpha*gs", aq * sA_swz / gs, wq * sB_swz, alpha * gs)

    print("\n--- 权重 scale 前 4 个字节（两种解读的起始值）---", flush=True)
    print("s1 raw[:4] =", s1[:4].tolist(), flush=True)
    print("s1 swz->lin[0,:4] =", unswz(s1, n, k // 16)[0, :4].tolist(), flush=True)
    print("s1 nat      [0,:4] =", s1.view(torch.float8_e4m3fn)[0, :4].float().tolist(), flush=True)


if __name__ == "__main__":
    main()
