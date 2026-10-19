#!/usr/bin/env python3
"""MXFP4（compressed-tensors `mxfp4-pack-quantized`）解包参考。格式见 `crates/kernels/cuda/mxfp4.cu` 顶部。

打包约定（与 torch `float4_e2m1fn_x2` / compressed-tensors 一致）：
  一个字节两个 E2M1，**低半字节是偶数下标**；scale 是 e8m0，值 = 2^(u8-127)，每 group=32 列一个。
本机 torch 2.6 没有 `float4_e2m1fn_x2`，所以这里按位构造，再与 GPU kernel 对拍。
真权重的约定核对在 B300 上做（vLLM 容器的 torch 版本更高，见 docs/loops/k3-moe.md）。

用法：python3 scripts/verify/gen_mxfp4_refs.py <out_dir>
"""
import os
import sys

import numpy as np
import torch

DIR = sys.argv[1] if len(sys.argv) > 1 else "test/kimik3/2026-10-19/mxfp4_refs"

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

CASES = [
    # (name, rows, cols, group)
    ("moE_w1_small", 64, 256, 32),
    ("moE_w2_small", 32, 512, 32),
    ("group32_odd_rows", 7, 96, 32),
]


def gen(name, rows, cols, group):
    d = os.path.join(DIR, name)
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(1234 + rows * 7 + cols)
    # packed：随机半字节，覆盖全部 16 种（符号位 = bit3）
    packed = rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)
    # scale：e8m0 指数，取常见范围 [117, 131]（±/2^10 附近），外加哨兵 127（=1.0）
    scale = rng.integers(117, 132, size=(rows, cols // group), dtype=np.uint8)
    scale[0, 0] = 127

    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    nib = np.empty((rows, cols), dtype=np.uint8)
    nib[:, 0::2] = lo
    nib[:, 1::2] = hi
    mag = E2M1[(nib & 7).astype(np.int64)]
    val = np.where((nib & 8) != 0, -mag, mag)
    # 每个 group 一个 e8m0 指数，逐列展开到 cols
    s_per_col = np.ldexp(np.ones((rows, cols // group), dtype=np.float32), scale.astype(np.int32) - 127)
    scl = np.repeat(s_per_col, group, axis=1)
    out = (val * scl).astype(np.float32)

    np.save(os.path.join(d, "packed.npy"), packed)
    np.save(os.path.join(d, "scale.npy"), scale)
    # 期望值按 bf16 量化：kernel 输出是 bf16
    want = torch.from_numpy(out).to(torch.bfloat16).float().numpy()
    np.save(os.path.join(d, "out.npy"), want)
    return name, f"rows={rows} cols={cols} group={group}"


# --------------------------------------------------------------------------- grouped GEMM 参考
# 语义见 crates/kernels/cuda/mxfp4_gemm_sm8x.cu 顶部：
#   out[r, n] = Σ_k dequant(w[e, n, k]) · xg[r, k]，r 属于专家 e 的排序后 token 段。
# 参考侧用 torch 做 fp32 matmul（解包 → 乘 → bf16）。分组用固定的 expert_off（每专家 token 数不同，
# 覆盖 M 不是 tile 整数倍、空专家）。


def dequant_ref(packed, scale, group):
    """packed u8 [N, cols/2] + scale u8 [N, cols/group] → f32 [N, cols]（与 mxfp4.cu 同式）。"""
    rows, half = packed.shape
    cols = half * 2
    nib = np.empty((rows, cols), dtype=np.uint8)
    nib[:, 0::2] = packed & 0x0F
    nib[:, 1::2] = (packed >> 4) & 0x0F
    mag = E2M1[(nib & 7).astype(np.int64)]
    val = np.where((nib & 8) != 0, -mag, mag).astype(np.float32)
    scl = np.repeat(np.ldexp(np.ones((rows, cols // group), dtype=np.float32),
                             scale.astype(np.int32) - 127), group, axis=1)
    return val * scl


def gen_gemm(name, e, n_out, k_in, group, ntok_per_expert, seed):
    d = os.path.join(DIR, name)
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(seed)
    packed = rng.integers(0, 256, size=(e, n_out, k_in // 2), dtype=np.uint8)
    # e8m0 指数取真权重的量级（2^-8 .. 2^-2）：乘积 ~O(0.01)，K 维累加后输出 O(1)，
    # 这样 bf16 输出的绝对误差才在 1e-2 量级（大指数会放大到 1e2，atol 失去意义）。
    scale = rng.integers(118, 126, size=(e, n_out, k_in // group), dtype=np.uint8)
    x = rng.standard_normal((sum(ntok_per_expert), k_in)).astype(np.float32) * 0.5

    off = np.concatenate([[0], np.cumsum(ntok_per_expert)]).astype(np.int32)
    out = np.zeros((x.shape[0], n_out), dtype=np.float32)
    for ei in range(e):
        r0, r1 = int(off[ei]), int(off[ei + 1])
        if r1 <= r0:
            continue
        w = dequant_ref(packed[ei], scale[ei], group)
        out[r0:r1] = x[r0:r1] @ w.T
    want = torch.from_numpy(out).to(torch.bfloat16).float().numpy()

    np.save(os.path.join(d, "packed.npy"), packed)
    np.save(os.path.join(d, "scale.npy"), scale)
    np.save(os.path.join(d, "x.npy"), x.astype(np.float32))
    np.save(os.path.join(d, "expert_off.npy"), off)
    np.save(os.path.join(d, "out.npy"), want)
    return name, f"E={e} N={n_out} K={k_in} group={group} tokens={ntok_per_expert}"


def gen_gemm_real_w1():
    # K3 的 w1/w3：3584 → 3072（K=3584 打包 1792）
    return gen_gemm("gemm_w1_like", 3, 3072, 3584, 32, [37, 8, 64], 9001)


def gen_gemm_real_w2():
    # K3 的 w2：3072 → 3584（K=3072 打包 1536）
    return gen_gemm("gemm_w2_like", 2, 3584, 3072, 32, [19, 33], 9002)


def gen_gemm_small():
    # 小形状 + 空专家 + 单 token（覆盖 block 内的 masked 行）
    return gen_gemm("gemm_small", 4, 128, 256, 32, [1, 0, 17, 64], 9003)


GEMM_CASES = [gen_gemm_real_w1, gen_gemm_real_w2, gen_gemm_small]


def main():
    os.makedirs(DIR, exist_ok=True)
    for name, rows, cols, group in CASES:
        n, note = gen(name, rows, cols, group)
        print(f"  {n:20s} {note}")
    for fn in GEMM_CASES:
        n, note = fn()
        print(f"  {n:20s} {note}")
    print(f"\n{len(CASES)} 个解包 case + {len(GEMM_CASES)} 个 GEMM case 写入 {DIR}")


if __name__ == "__main__":
    main()
