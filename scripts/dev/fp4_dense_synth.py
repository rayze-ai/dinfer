#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自洽合成数据：先把「参照 dense FP4 GEMM 到底等于什么」钉死，再拿去当我们的判据。

上一轮的教训：拿一份来源不清的 `y.bin`（corr 0.25）当判据，白耗一天。
本轮改成 vLLM CI 同款配方（`tests/kernels/quantization/test_nvfp4_scaled_mm.py`）：

    a, b 随机 → 两侧都用 ops.scaled_fp4_quant 量化（两侧 scale 都 swizzled）
    → 用参照自己的 dequantize_nvfp4_to_dtype 复算 → 与 cutlass_scaled_fp4_mm 比

CI 用 atol/rtol 1e-1 的容差；我们要的是**逐位**，所以先看两者差多少：
若 CI 自己也不是逐位，那"逐位"这个目标对 NVFP4 GEMM 就不成立，判据得改。

同时导出一份固定数据供 Rust 侧对拍（`/w/tmp/fp4s/`），并在下面用纯 numpy
按"已确证公式"独立复算一遍 —— 三方一致才算判据可信。

用法（容器内）：
    python3 scripts/dev/fp4_dense_synth.py [outdir]
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

# 形状取 Kaon 真实形状的一个可复现子集
M, N, K = 8, 1408, 2816


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "/w/tmp/fp4s"
    os.makedirs(d, exist_ok=True)
    torch.manual_seed(0)

    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        swizzle_blockscale,
    )

    FLOAT4_MAX = scalar_types.float4_e2m1f.max()          # 6.0
    FLOAT8_MAX = torch.finfo(torch.float8_e4m3fn).max     # 448.0

    a = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")

    a_gs = ((FLOAT8_MAX * FLOAT4_MAX) / torch.amax(a.flatten(), dim=-1)).to(torch.float32)
    b_gs = ((FLOAT8_MAX * FLOAT4_MAX) / torch.amax(b.flatten(), dim=-1)).to(torch.float32)
    alpha = 1.0 / (a_gs * b_gs)

    a_q, a_s = ops.scaled_fp4_quant(a, a_gs)              # 默认 padded/swizzled
    b_q, b_s = ops.scaled_fp4_quant(b, b_gs)
    print(f"形状 m={M} n={N} k={K}", flush=True)
    print(f"a_gs={a_gs.item():.8g} b_gs={b_gs.item():.8g} alpha={alpha.item():.8g}", flush=True)
    print(f"a_q={tuple(a_q.shape)} a_s={tuple(a_s.shape)} "
          f"b_q={tuple(b_q.shape)} b_s={tuple(b_s.shape)}", flush=True)

    y = ops.cutlass_scaled_fp4_mm(a_q, b_q, a_s, b_s, alpha, torch.bfloat16)

    # ---- 参照自己的复算（CI 同款；函数逐字复制自
    #      /data/ref/vllm/tests/kernels/quantization/nvfp4_utils.py，仅用于 oracle 侧诊断）----
    def convert_swizzled_to_linear(sf, m_, k_, block_size):
        m_tiles = (m_ + 128 - 1) // 128
        f = block_size * 4
        k_tiles = (k_ + f - 1) // f
        tmp = torch.reshape(sf, (1, m_tiles, k_tiles, 32, 4, 4))
        tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
        out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
        return out[0:m_, 0:k_]

    def break_fp4_bytes(a_, dtype):
        m_, n_ = a_.shape
        flat = a_.flatten()
        high = (flat & 0xF0) >> 4
        low = flat & 0x0F
        combined = torch.stack((low, high), dim=1).flatten()
        signs = (combined & 0x08).to(torch.bool)
        abs_vals = (combined & 0x07).to(torch.long)
        tab = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=a_.device)
        vals = tab[abs_vals] * torch.where(signs, -1.0, 1.0)
        return vals.reshape(m_, n_ * 2).to(dtype=dtype)

    def dequantize_nvfp4_to_dtype(tensor_fp4, tensor_sf, global_scale, dtype,
                                  device, block_size=16):
        m_, packed_k = tensor_fp4.shape
        k_ = packed_k * 2
        t_f32 = break_fp4_bytes(tensor_fp4, dtype)
        t_f32 = t_f32.reshape(m_, k_ // block_size, block_size)
        sf = tensor_sf.view(torch.float8_e4m3fn)
        sf = convert_swizzled_to_linear(sf, m_, k_, block_size)
        sf = sf.to(torch.float32) / global_scale
        return (t_f32 * sf.unsqueeze(-1)).reshape(m_, k_).to(dtype=dtype)
    a_deq = dequantize_nvfp4_to_dtype(a_q, a_s, a_gs, dtype=torch.float32,
                                      device="cuda", block_size=16)
    b_deq = dequantize_nvfp4_to_dtype(b_q, b_s, b_gs, dtype=torch.float32,
                                      device="cuda", block_size=16)
    y_ref = torch.matmul(a_deq, b_deq.t())

    yf = y.to(torch.float32)
    rel = (yf - y_ref).norm() / y_ref.norm()
    corr = torch.corrcoef(torch.stack([yf.flatten(), y_ref.flatten()]))[0, 1].item()
    ndiff = (y.view(torch.int16) != y_ref.to(torch.bfloat16).view(torch.int16)).sum().item()
    print(f"\n[1] cutlass vs 参照 dequant+matmul: relerr={rel.item():.6f} corr={corr:.8f} "
          f"bf16 逐位不同 {ndiff}/{M * N}", flush=True)
    print(f"    max|d|={(yf - y_ref).abs().max().item():.4e}  "
          f"‖y‖={y_ref.norm().item():.4f}", flush=True)

    # ---- 纯 numpy 独立复算（不调用 vLLM 反量化，验证公式本身）----
    def unswz_explicit(buf_u8, rows, cols):
        """显式逐元素按公式取，不依赖任何 permute 语义。"""
        Mp, Kp = (rows + 127) // 128 * 128, (cols + 3) // 4 * 4
        nKT = Kp // 4
        flat = buf_u8.reshape(-1)
        out = np.zeros((rows, cols), dtype=np.uint8)
        for row in range(rows):
            base = (row >> 7) * nKT * 512 + (row & 31) * 16 + ((row >> 5) & 3) * 4
            for kidx in range(cols):
                out[row, kidx] = flat[base + (kidx >> 2) * 512 + (kidx & 3)]
        return out

    def e4m3_dec(u8):
        sign = np.where((u8 >> 7) & 1, -1.0, 1.0)
        e = (u8 >> 3) & 0xF
        m = u8 & 7
        val = np.where(e == 0, m / 8.0 * 2.0**-6, (1 + m / 8.0) * 2.0 ** (e.astype(np.int32) - 7))
        return sign * val

    def fp4_dec(u8):
        tab = np.array([0, .5, 1., 1.5, 2., 3., 4., 6.])
        q = u8 & 0x7
        return tab[q] * np.where((u8 >> 3) & 1, -1.0, 1.0)

    a_s_u8 = a_s.view(torch.uint8).cpu().numpy()
    b_s_u8 = b_s.view(torch.uint8).cpu().numpy()
    a_q_u8 = a_q.view(torch.uint8).cpu().numpy()
    b_q_u8 = b_q.view(torch.uint8).cpu().numpy()

    a_s_lin = unswz_explicit(a_s_u8, M, K // 16)
    b_s_lin = unswz_explicit(b_s_u8, N, K // 16)
    print(f"\n[2] 显式反 swizzle 完成：a_s_lin{a_s_lin.shape} b_s_lin{b_s_lin.shape}", flush=True)

    def deq(q_u8, s_lin, gs):
        rows, kb = q_u8.shape[0], s_lin.shape[1]
        v = np.empty((rows, kb * 16), dtype=np.float32)
        v[:, 0::2] = fp4_dec(q_u8 & 0x0F)
        v[:, 1::2] = fp4_dec((q_u8 >> 4) & 0x0F)
        sc = (e4m3_dec(s_lin).astype(np.float32) / gs)         # 参照：sf / global_scale
        return v * np.repeat(sc, 16, axis=1)

    a_np = deq(a_q_u8, a_s_lin, a_gs.item())
    b_np = deq(b_q_u8, b_s_lin, b_gs.item())
    y_np = a_np @ b_np.T
    y_ref_np = y_ref.cpu().numpy()
    rel2 = np.linalg.norm(y_np - y_ref_np) / np.linalg.norm(y_ref_np)
    corr2 = np.corrcoef(y_np.ravel(), y_ref_np.ravel())[0, 1]
    print(f"[3] numpy 独立复算 vs 参照 dequant+matmul: relerr={rel2:.6f} corr={corr2:.8f}", flush=True)
    print(f"    ‖np‖/‖ref‖={np.linalg.norm(y_np)/np.linalg.norm(y_ref_np):.6f}", flush=True)

    # ---- 导出给 Rust 侧 ----
    a.view(torch.int16).cpu().numpy().tofile(f"{d}/a.bin")
    a_q_u8.tofile(f"{d}/a_q.bin")
    a_s_u8.tofile(f"{d}/a_s.bin")
    a_s_lin.tofile(f"{d}/a_s_nat.bin")
    b_q_u8.tofile(f"{d}/b_q.bin")
    b_s_u8.tofile(f"{d}/b_s.bin")
    b_s_lin.tofile(f"{d}/b_s_nat.bin")   # 反 swizzle 后的线性布局，供我们 kernel 的「自然读」路径
    y.view(torch.int16).cpu().numpy().tofile(f"{d}/y.bin")
    np.float32(alpha.item()).tofile(f"{d}/alpha.bin")
    np.float32(a_gs.item()).tofile(f"{d}/a_gs.bin")
    np.float32(b_gs.item()).tofile(f"{d}/b_gs.bin")
    open(f"{d}/meta.txt", "w").write(f"m={M}\nn={N}\nk={K}\n")
    print(f"\n已导出到 {d}", flush=True)


if __name__ == "__main__":
    main()
