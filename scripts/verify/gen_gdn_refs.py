#!/usr/bin/env python3
"""GDN 算子对拍参考（Qwen3.5 线，模型无关的形状）。torch 生成输入 + 期望，Rust 侧 `gdn_parity.rs` 跑 kernel 断言。

数学口径见 `crates/kernels/cuda/gdn.cu` 顶部与 `docs/loops/qwen35-gdn.md`；与 vLLM
`layers/mamba/gdn/qwen_gdn_linear_attn.py` + `third_party/flash_linear_attention/ops/{fused_recurrent,chunk,l2norm}.py` 同式。
不依赖 vllm / fla 包（本机 venv 没有，B300 容器里才有）。

输出目录：`<dir>/<case>/`，张量一律 .npy。bf16 张量按原始位存（dtype `<u2`，名字带 `_bf16`），
与 `ops_parity.rs` / `kda_parity.rs` 口径一致，Rust 侧 `<<16` 还原 f32，逐位无损。

用法：python3 scripts/verify/gen_gdn_refs.py test/qwen35/<日期>/gdn_refs
"""
import os
import sys

import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

DIR = sys.argv[1] if len(sys.argv) > 1 else "test/qwen35/2026-10-19/gdn_refs"

# Qwen3.5-27B 的 GDN 真值 + 一个非整除小形状（group=2, K≠V）做回归
QWEN = dict(K=128, V=128, H=16, HV=48, width=4)
SMALL = dict(K=64, V=32, H=2, HV=4, width=3)
# 递归状态是 [S, HV, V, K] f32，HV=48/K=V=128 时每槽 3.1 MB（不能进仓库）：
# 用同样的 K=V=128 与 group=3，但 HV=12 缩小体积，group 映射与 head_dim 都被覆盖。
QWEN_REC = dict(QWEN, H=4, HV=12)


def save(name, t, out_dir):
    a = t.detach().cpu()
    if a.dtype == torch.bfloat16:
        bits = a.view(torch.uint16).numpy()
        np.save(os.path.join(out_dir, name + "_bf16.npy"), bits.astype(np.uint16))
    elif a.dtype == torch.float32:
        np.save(os.path.join(out_dir, name + ".npy"), a.numpy().astype(np.float32))
    elif a.dtype == torch.int32:
        np.save(os.path.join(out_dir, name + ".npy"), a.numpy().astype(np.int32))
    else:
        raise TypeError(a.dtype)


def bf16(t):
    """量化到 bf16 再回到 f32：参考侧与 kernel 的取值口径一致。"""
    return t.to(torch.bfloat16).float()


def out_dir(name):
    d = os.path.join(DIR, name)
    os.makedirs(d, exist_ok=True)
    return d


def softplus(x):
    # 与 kernel 同分支：x > 20 取 x（避免 exp 溢出）
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


# --------------------------------------------------------------------------- 1. conv1d
def gen_conv(cfg, tag):
    K, V, H, HV, W = cfg["K"], cfg["V"], cfg["H"], cfg["HV"], cfg["width"]
    dim = 2 * H * K + HV * V
    d = out_dir(f"conv1d_{tag}")
    lens = [7, 1, 12] if tag == "qwen" else [5, 9]
    cu = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    T = int(cu[-1])
    S = len(lens)  # conv 状态很小（[slots, dim, W-1]），每序列一个槽
    x = bf16(torch.randn(T, dim))
    w = bf16(torch.randn(dim, W) * 0.2)
    st = bf16(torch.randn(S, dim, W - 1) * 0.5)
    has_initial = np.array(([1, 0, 1] if len(lens) >= 3 else [1, 0])[: len(lens)], dtype=np.int32)
    # 参考：y[t] = silu(Σ_j w[d,j]·x[t-W+1+j])；t<0 取 state，列 0 最旧
    out = torch.empty_like(x)
    st_out = st.clone()
    for s in range(len(lens)):
        bos, eos = int(cu[s]), int(cu[s + 1])
        hist = [st[s, :, j].clone() if has_initial[s] else torch.zeros(dim) for j in range(W - 1)]
        for t in range(bos, eos):
            acc = w[:, W - 1] * x[t]
            for j in range(W - 1):
                acc = acc + w[:, j] * hist[j]
            out[t] = torch.nn.functional.silu(acc.to(torch.float32)).to(torch.bfloat16)
            if W > 1:
                hist = hist[1:] + [x[t].clone()]
        for j in range(W - 1):
            st_out[s, :, j] = hist[j].to(torch.bfloat16)
    save("x", x.to(torch.bfloat16), d)
    save("w", w.to(torch.bfloat16), d)
    save("state_in", st.to(torch.bfloat16), d)
    save("out", out.to(torch.bfloat16), d)
    save("state_out", st_out.to(torch.bfloat16), d)
    save("cu_seqlens", torch.from_numpy(cu), d)
    save("has_initial", torch.from_numpy(has_initial), d)
    save("state_indices", torch.arange(S, dtype=torch.int32), d)

    # update（decode 4 步）：每步 batch=3，槽 0..2 接着 conv 之后的状态继续
    du = out_dir(f"conv_update_{tag}")
    st_cur = st_out.clone()
    save("w", w.to(torch.bfloat16), du)
    save("state_in", st_cur.to(torch.bfloat16), du)
    save("state_indices", torch.arange(S, dtype=torch.int32), du)
    for step in range(4):
        xt = bf16(torch.randn(S, dim))
        acc = w[:, W - 1] * xt
        for j in range(W - 1):
            acc = acc + w[:, j] * st_cur[:, :, j]
        oc = torch.nn.functional.silu(acc.to(torch.float32)).to(torch.bfloat16)
        # 状态左移 + 追加
        st_new = st_cur.clone()
        for j in range(W - 2):
            st_new[:, :, j] = st_cur[:, :, j + 1]
        st_new[:, :, W - 2] = xt
        save(f"x{step}", xt, du)
        save(f"out{step}", oc, du)
        save(f"state{step}", st_new, du)
        st_cur = st_new


# --------------------------------------------------------------------------- 2. gate / beta
def gen_gate(cfg, tag):
    HV = cfg["HV"]
    d = out_dir(f"gate_{tag}")
    T = 9
    a = bf16(torch.randn(T, HV) * 3.0)
    b = bf16(torch.randn(T, HV) * 2.0)
    a_log = torch.randn(HV) * 0.5
    dt_bias = torch.randn(HV)
    x = a.numpy() + dt_bias.numpy()[None, :]
    g = -np.exp(a_log.numpy())[None, :] * softplus(x)
    beta = 1.0 / (1.0 + np.exp(-b.numpy()))
    save("a", a, d)
    save("b", b, d)
    save("a_log", a_log.float(), d)
    save("dt_bias", dt_bias.float(), d)
    save("g", torch.from_numpy(g).float(), d)
    save("beta", torch.from_numpy(beta).float(), d)


# --------------------------------------------------------------------------- 3. 递归核心
def gdn_ref(mixed, a, b, a_log, dt_bias, state, lens, K, V, H, HV, use_scale=True):
    """顺序递归参考。mixed: [T, 2HK+HVV] f32（bf16 量化后），a/b: [T,HV]，state: [S,HV,V,K]。"""
    T = mixed.shape[0]
    group = HV // H
    scale = K ** -0.5 if use_scale else 1.0
    out = torch.zeros(T, HV, V)
    state = state.clone()
    cu = np.concatenate([[0], np.cumsum(lens)])
    qkv_row = 2 * H * K + HV * V
    for s in range(len(lens)):
        S = state[s]
        for t in range(int(cu[s]), int(cu[s + 1])):
            row = mixed[t]
            q = row[: H * K].reshape(H, K)[:, :]
            k = row[H * K : 2 * H * K].reshape(H, K)[:, :]
            v = row[2 * H * K :].reshape(HV, V)
            x = a[t].numpy() + dt_bias.numpy()
            g = -np.exp(a_log.numpy()) * softplus(x)     # [HV]
            beta = 1.0 / (1.0 + np.exp(-b[t].numpy()))   # [HV]
            for hv in range(HV):
                qh = q[hv // group] / torch.sqrt((q[hv // group] ** 2).sum() + 1e-6) * scale
                kh = k[hv // group] / torch.sqrt((k[hv // group] ** 2).sum() + 1e-6)
                # S[hv] 形状 [V, K]
                S[hv] = S[hv] * float(np.exp(g[hv]))
                vp = (v[hv] - S[hv] @ kh) * float(beta[hv])
                S[hv] = S[hv] + vp[:, None] * kh[None, :]
                out[t, hv] = S[hv] @ qh
    return out, state


def gen_recurrent(cfg, tag, decode=False):
    K, V, H, HV = cfg["K"], cfg["V"], cfg["H"], cfg["HV"]
    d = out_dir(f"recurrent_{tag}" + ("_decode" if decode else ""))
    S = 2 if tag == "qwen" else 2
    torch.manual_seed(11)
    if decode:
        lens = [1, 1]
        steps = 2 if tag == "qwen" else 4
    else:
        lens = [6, 4] if tag == "qwen" else [5, 7]
        steps = 1
    T = int(np.sum(lens))
    qkv_row = 2 * H * K + HV * V
    mixed = bf16(torch.randn(T, qkv_row) * 0.5)
    a = bf16(torch.randn(T, HV) * 2.0)
    b = bf16(torch.randn(T, HV) * 2.0)
    a_log = torch.randn(HV) * 0.5
    dt_bias = torch.randn(HV)
    state = torch.randn(S, HV, V, K) * 0.05
    save("a_log", a_log.float(), d)
    save("dt_bias", dt_bias.float(), d)
    save("state_in", state.clone(), d)
    cu = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    save("cu_seqlens", torch.from_numpy(cu), d)
    if decode:
        state_cur = state.clone()
        for step in range(steps):
            m = bf16(torch.randn(T, qkv_row) * 0.5)
            aa = bf16(torch.randn(T, HV) * 2.0)
            bb = bf16(torch.randn(T, HV) * 2.0)
            o, state_cur = gdn_ref(m, aa, bb, a_log, dt_bias, state_cur, lens, K, V, H, HV)
            save(f"mixed{step}", m, d)
            save(f"a{step}", aa, d)
            save(f"b{step}", bb, d)
            save(f"out{step}", o.to(torch.bfloat16), d)
            save(f"state{step}", state_cur, d)
    else:
        save("mixed", mixed, d)
        save("a", a, d)
        save("b", b, d)
        o, st_out = gdn_ref(mixed, a, b, a_log, dt_bias, state, lens, K, V, H, HV)
        save("out", o.to(torch.bfloat16), d)
        save("state_out", st_out, d)


def proc_state(S, HV, V, K):
    """程序化初始状态（Rust 侧 gdn_parity.rs::proc_state 同式，逐位一致）：
    27B 形状 B=32 的 [S, HV, V, K] f32 有 100 MB，不能进仓库，两侧各按同一公式生成。
    值 = ((idx · 2654435761) mod 2^32 >> 8) / 2^24 · 0.1 − 0.05，idx 为平铺下标（u32 乘法回绕）。"""
    n = S * HV * V * K
    idx = np.arange(n, dtype=np.uint64)
    h = (idx * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)
    h = (h >> np.uint64(8)).astype(np.float32)
    v = h / np.float32(16777216.0) * np.float32(0.1) - np.float32(0.05)
    return torch.from_numpy(v.reshape(S, HV, V, K))


def gen_recurrent_decode_batch(cfg, name, B):
    """decode 一步、B 条序列、27B 真形状、非零初始状态：对拍寄存器版 decode kernel 的 batch 扩展。
    状态 100 MB 不落盘：state_in 程序化（proc_state），期望状态只存 K 维求和 `state_sum_k` [S,HV,V] 与
    V 维求和 `state_sum_v` [S,HV,K]（能抓到行 / 列 / 槽错位），out 存全量 bf16。"""
    K, V, H, HV = cfg["K"], cfg["V"], cfg["H"], cfg["HV"]
    d = out_dir(name)
    torch.manual_seed(31)
    lens = [1] * B
    qkv_row = 2 * H * K + HV * V
    mixed = bf16(torch.randn(B, qkv_row) * 0.5)
    a = bf16(torch.randn(B, HV) * 2.0)
    b = bf16(torch.randn(B, HV) * 2.0)
    a_log = torch.randn(HV) * 0.5
    dt_bias = torch.randn(HV)
    state = proc_state(B, HV, V, K)
    save("a_log", a_log.float(), d)
    save("dt_bias", dt_bias.float(), d)
    cu = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    save("cu_seqlens", torch.from_numpy(cu), d)
    save("mixed", mixed, d)
    save("a", a, d)
    save("b", b, d)
    o, st = gdn_ref(mixed, a, b, a_log, dt_bias, state, lens, K, V, H, HV)
    save("out", o.to(torch.bfloat16), d)
    save("state_sum_k", st.double().sum(dim=3).float(), d)
    save("state_sum_v", st.double().sum(dim=2).float(), d)


# --------------------------------------------------------------------------- 3b. chunk 形式 prefill（gdn_chunk.cu）
# 参考仍是顺序递归 gdn_ref：这正是 chunk 版与顺序版等价性的证明。
# 覆盖：非 BT(64) 整数倍长度、3 序列 varlen（其中一条带非零初始 state）、27B 真形状。
#   chunk         T=200 单序列，初始 state 非零
#   chunk_varlen  lens [130, 64, 7]，只有第 1 条带非零初始 state（其余零，对应 has_initial=0）
#   chunk_qwen    H=16 HV=48 K=V=128（27B 形状）T=256，初始 state 零；state_in 不落盘（3.1 MB/槽），
#                 Rust 侧缺 state_in 时按零初始化。
def gen_chunk(cfg, name, lens, init_mask, save_state_in=True, seed=23):
    K, V, H, HV = cfg["K"], cfg["V"], cfg["H"], cfg["HV"]
    d = out_dir(name)
    torch.manual_seed(seed)
    S = len(lens)
    T = int(np.sum(lens))
    qkv_row = 2 * H * K + HV * V
    mixed = bf16(torch.randn(T, qkv_row) * 0.5)
    a = bf16(torch.randn(T, HV) * 2.0)
    b = bf16(torch.randn(T, HV) * 2.0)
    a_log = torch.randn(HV) * 0.5
    dt_bias = torch.randn(HV)
    state = torch.randn(S, HV, V, K) * 0.05
    for s, m in enumerate(init_mask):
        if not m:
            state[s].zero_()
    save("a_log", a_log.float(), d)
    save("dt_bias", dt_bias.float(), d)
    if save_state_in:
        save("state_in", state.clone(), d)
    cu = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    save("cu_seqlens", torch.from_numpy(cu), d)
    # 输入已是 bf16 取值，按 bf16 位存（体积减半；Rust 侧 load 自动找 *_bf16.npy）
    save("mixed", mixed.to(torch.bfloat16), d)
    save("a", a.to(torch.bfloat16), d)
    save("b", b.to(torch.bfloat16), d)
    o, st_out = gdn_ref(mixed, a, b, a_log, dt_bias, state, lens, K, V, H, HV)
    save("out", o.to(torch.bfloat16), d)
    save("state_out", st_out, d)


# --------------------------------------------------------------------------- 4. gated RMSNorm
def gen_norm(cfg, tag):
    V = cfg["V"]
    for act in ("silu", "sigmoid"):
        d = out_dir(f"gated_rmsnorm_{act}_{tag}")
        rows = 7
        x = bf16(torch.randn(rows, V) * 1.5)
        w = bf16(torch.randn(V) * 0.5 + 1.0)
        z = bf16(torch.randn(rows, V) * 2.0)
        xf = x.float()
        rstd = 1.0 / torch.sqrt((xf * xf).sum(-1, keepdim=True) / V + 1e-6)
        y = xf * rstd * w.float()
        zf = z.float()
        gate = torch.sigmoid(zf) if act == "sigmoid" else zf * torch.sigmoid(zf)
        y = y * gate
        save("x", x, d)
        save("w", w, d)
        save("z", z, d)
        save("out", y.to(torch.bfloat16), d)


if __name__ == "__main__":
    for tag, cfg in (("qwen", QWEN), ("small", SMALL)):
        gen_conv(cfg, tag)
        gen_gate(cfg, tag)
        gen_recurrent(QWEN_REC if tag == "qwen" else cfg, tag, decode=False)
        gen_recurrent(QWEN_REC if tag == "qwen" else cfg, tag, decode=True)
        gen_norm(cfg, tag)
    gen_chunk(QWEN_REC, "chunk", [200], [True])
    gen_chunk(QWEN_REC, "chunk_varlen", [130, 64, 7], [True, False, False])
    gen_chunk(QWEN, "chunk_qwen", [256], [False], save_state_in=False)
    gen_recurrent_decode_batch(QWEN, "recurrent_qwen_decode_b32", 32)
    print("gdn refs ->", DIR)
