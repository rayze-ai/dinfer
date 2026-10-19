#!/usr/bin/env python3
"""KDA 算子对拍参考（Kimi-K3 线，模型无关的形状）。torch 生成输入 + 期望，Rust 侧 `kda_parity.rs` 跑 kernel 断言。

数学口径见 `crates/kernels/cuda/kda.cu` 顶部与 `docs/loops/k3-kda.md`；与 vLLM 的
`third_party/flash_linear_attention/ops/kda.py`（chunk/recurrent）及 K3 的 packed decode 同式。
不依赖 vllm / fla 包：本机 venv 没有，B300 容器里才有。

输出目录：`<dir>/<case>/`，张量一律 .npy。bf16 张量按原始位存（dtype `<u2`，名字带 `_bf16`），
与 `ops_parity.rs` 口径一致，Rust 侧 `<<16` 还原 f32，逐位无损。

用法：python3 scripts/verify/gen_kda_refs.py test/kimik3/<日期>/kda_refs
"""
import os
import sys

import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

DIR = sys.argv[1] if len(sys.argv) > 1 else "test/kimik3/2026-10-19/kda_refs"


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


def rnd(*shape, scale=1.0):
    return torch.randn(*shape) * scale


# --------------------------------------------------------------------------- 参考实现

def conv1d_ref(x, w, state, cu_seqlens, has_initial, width):
    """x/w/state 均为 bf16 量化后的 f32。x: [T, dim]，w: [dim, width]，state: [S, dim, width-1]。
    返回 (out, state_out)。state 列 0 最旧。"""
    T, dim = x.shape
    out = torch.empty_like(x)
    st = state.clone()
    for s in range(len(cu_seqlens) - 1):
        bos, eos = int(cu_seqlens[s]), int(cu_seqlens[s + 1])
        hist = []
        for j in range(width - 1):
            hist.append(st[s, :, j].clone() if has_initial[s] else torch.zeros(dim))
        for t in range(bos, eos):
            acc = w[:, width - 1] * x[t]
            for j in range(width - 1):
                acc = acc + w[:, j] * hist[j]
            out[t] = bf16(torch.nn.functional.silu(acc))
            if width > 1:
                hist = hist[1:] + [x[t].clone()]
        if width > 1:
            for j, h in enumerate(hist):
                st[s, :, j] = h
    return out, st


def conv1d_update_ref(x, w, state, width):
    """x: [B, dim]，state: [B, dim, width-1] → (out, state_out)。"""
    B, dim = x.shape
    out = torch.empty_like(x)
    st = state.clone()
    for b in range(B):
        acc = w[:, width - 1] * x[b]
        for j in range(width - 1):
            acc = acc + w[:, j] * st[b, :, j]
        out[b] = bf16(torch.nn.functional.silu(acc))
        cols = [st[b, :, j] for j in range(1, width - 1)] + [x[b].clone()]
        for j, c in enumerate(cols):
            st[b, :, j] = c
    return out, st


def gate_ref(g, a_log, dt_bias, lower_bound, heads, head_dim):
    """g: [T, heads, head_dim]（原始投影，bf16 量化后的 f32）→ gate: [T, heads, head_dim] f32。"""
    a = torch.exp(a_log)                       # [heads]
    x = g + dt_bias.view(heads, head_dim)      # [T, heads, head_dim]
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(a.view(1, -1, 1) * x)
    sp = torch.where(x > 20.0, x, torch.log1p(torch.exp(x)))
    return -a.view(1, -1, 1) * sp


def l2norm_ref(t):
    return t / torch.sqrt((t * t).sum(-1, keepdim=True) + 1e-6)


def recurrent_ref(q, k, v, raw_g, raw_beta, a_log, dt_bias, state, cu_seqlens, has_state, heads,
                  head_dim, scale, lower_bound, width):
    """逐 token 递归。q/k/v/raw_g: [T, heads, head_dim]；raw_beta: [T, heads]；
    state: [S, heads, head_dim(V), head_dim(K)] f32；返回 (o, state_out)。"""
    T = q.shape[0]
    o = torch.empty(T, heads, head_dim)
    st = state.clone()
    gate_all = gate_ref(raw_g, a_log, dt_bias, lower_bound, heads, head_dim)  # [T,heads,K]
    for s in range(len(cu_seqlens) - 1):
        bos, eos = int(cu_seqlens[s]), int(cu_seqlens[s + 1])
        S = st[s] if has_state[s] else torch.zeros(heads, head_dim, head_dim)
        for t in range(bos, eos):
            qq = l2norm_ref(q[t]) * scale       # [H, K]
            kk = l2norm_ref(k[t])               # [H, K]
            eg = torch.exp(gate_all[t])         # [H, K]
            beta = torch.sigmoid(raw_beta[t])   # [H]
            S = S * eg.unsqueeze(1)
            vp = (v[t] - (S * kk.unsqueeze(1)).sum(-1)) * beta.unsqueeze(-1)
            S = S + vp.unsqueeze(-1) * kk.unsqueeze(1)
            o[t] = (S * qq.unsqueeze(1)).sum(-1)
        st[s] = S
    return bf16(o), st


def gated_rmsnorm_ref(x, w, gate, eps=1e-5):
    """x/gate: [rows, head_dim] → y = rmsnorm(x)·sigmoid(gate)。"""
    rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return bf16(x * rstd * w * torch.sigmoid(gate))


# --------------------------------------------------------------------------- cases

def case(name):
    d = os.path.join(DIR, name)
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        os.remove(os.path.join(d, f))
    return d


def gen_conv_prefill():
    d = case("conv_prefill")
    dim, width, T = 384, 4, 97
    x = bf16(rnd(T, dim))
    w = bf16(rnd(dim, width, scale=0.5))
    st0 = bf16(rnd(1, dim, width - 1, scale=0.3))
    cu = torch.tensor([0, T], dtype=torch.int32)
    has_init = torch.tensor([1], dtype=torch.int32)
    out, st = conv1d_ref(x, w, st0, cu.tolist(), [True], width)
    save("x", x, d); save("w", w, d); save("state_in", st0, d); save("cu_seqlens", cu, d)
    save("has_initial", has_init, d); save("out", out, d); save("state_out", st, d)
    return "conv_prefill", f"T={T} dim={dim} width={width} 单序列带初态"


def gen_conv_prefill_varlen():
    d = case("conv_prefill_varlen")
    dim, width = 384, 4
    lens = [20, 45, 3]
    T = sum(lens)
    x = bf16(rnd(T, dim))
    w = bf16(rnd(dim, width, scale=0.5))
    st0 = bf16(rnd(3, dim, width - 1, scale=0.3))
    cu = torch.tensor([0] + list(np.cumsum(lens)), dtype=torch.int32)
    has_init = torch.tensor([1, 0, 1], dtype=torch.int32)
    out, st = conv1d_ref(x, w, st0, cu.tolist(), [True, False, True], width)
    save("x", x, d); save("w", w, d); save("state_in", st0, d); save("cu_seqlens", cu, d)
    save("has_initial", has_init, d); save("out", out, d); save("state_out", st, d)
    return "conv_prefill_varlen", f"varlen 3 段 {lens} dim={dim} 部分无初态"


def gen_conv_update():
    d = case("conv_update")
    dim, width, B, steps = 384, 4, 3, 4
    w = bf16(rnd(dim, width, scale=0.5))
    st = bf16(rnd(B, dim, width - 1, scale=0.3))
    save("w", w, d); save("state0", st, d)
    for s in range(steps):
        x = bf16(rnd(B, dim))
        out, st = conv1d_update_ref(x, w, st, width)
        save(f"x{s}", x, d); save(f"out{s}", out, d)
    save("state_final", st, d)
    return "conv_update", f"B={B} steps={steps} dim={dim} 连续 update"


def gen_gate():
    d = case("gate")
    heads, head_dim, T = 4, 128, 13
    a_log = torch.log(torch.tensor([1.0, 3.5, 9.0, 15.9]))
    dt_bias = rnd(heads * head_dim, scale=2.0)
    g = bf16(rnd(T, heads, head_dim, scale=2.0))
    g[0, 0, 0] = 40.0          # softplus 溢出侧（有 lower_bound 时走 sigmoid，仍要覆盖）
    g[1, 1, 1] = -40.0
    save("a_log", a_log, d); save("dt_bias", dt_bias, d); save("g", g, d)
    save("out_lb", gate_ref(g, a_log, dt_bias, -5.0, heads, head_dim), d)
    save("out_softplus", gate_ref(g, a_log, dt_bias, None, heads, head_dim), d)
    return "gate", f"heads={heads} head_dim={head_dim} T={T} 两种 gate 形式"


def _recurrent_case(name, lens, has_state, T_extra=0, width=4, seed_case=None):
    d = case(name)
    heads, head_dim = 4, 128
    T = sum(lens) + T_extra
    q = bf16(rnd(T, heads, head_dim, scale=0.8))
    k = bf16(rnd(T, heads, head_dim, scale=0.8))
    v = bf16(rnd(T, heads, head_dim, scale=0.8))
    raw_g = bf16(rnd(T, heads, head_dim, scale=1.5))
    raw_beta = rnd(T, heads, scale=2.0)
    a_log = torch.log(torch.tensor([1.0, 3.5, 9.0, 15.9]))
    dt_bias = rnd(heads * head_dim, scale=1.0)
    S = len(lens)
    st0 = rnd(S, heads, head_dim, head_dim, scale=0.05)
    # has_state=0 的槽由调用方保证是零（调度器语义）：参考与 kernel 都以零初态起步。
    for i, hs in enumerate(has_state):
        if not hs:
            st0[i] = 0.0
    cu = torch.tensor([0] + list(np.cumsum(lens)), dtype=torch.int32)
    scale = head_dim ** -0.5
    o, st = recurrent_ref(q, k, v, raw_g, raw_beta, a_log, dt_bias, st0, cu.tolist(), has_state,
                          heads, head_dim, scale, -5.0, width)
    save("q", q, d); save("k", k, d); save("v", v, d); save("raw_g", raw_g, d)
    save("raw_beta", raw_beta, d); save("a_log", a_log, d); save("dt_bias", dt_bias, d)
    save("cu_seqlens", cu, d); save("state_in", st0, d); save("out", o, d); save("state_out", st, d)
    return d, (q, k, v, raw_g, raw_beta, a_log, dt_bias, st, scale)


def gen_recurrent_prefill():
    _, extra = _recurrent_case("recurrent_prefill", [64], [0], seed_case=0)
    return "recurrent_prefill", "T=64 heads=4 head_dim=128 零初态单序列"


def gen_recurrent_varlen():
    _, extra = _recurrent_case("recurrent_varlen", [5, 40, 17], [1, 0, 1])
    return "recurrent_varlen", "varlen 3 段（5/40/17）带非零初态"


def gen_recurrent_decode():
    """decode：逐 token 前向，状态跨步累积，只比最后一步输出与末态。"""
    d = case("recurrent_decode")
    heads, head_dim, B, steps = 4, 128, 3, 4
    a_log = torch.log(torch.tensor([1.0, 3.5, 9.0, 15.9]))
    dt_bias = rnd(heads * head_dim, scale=1.0)
    state = rnd(B, heads, head_dim, head_dim, scale=0.05).to(torch.float32)
    save("a_log", a_log, d); save("dt_bias", dt_bias, d)
    save("state_in", state, d)
    scale = head_dim ** -0.5
    st_torch = state.clone()
    for s in range(steps):
        q = bf16(rnd(B, heads, head_dim, scale=0.8))
        k = bf16(rnd(B, heads, head_dim, scale=0.8))
        v = bf16(rnd(B, heads, head_dim, scale=0.8))
        rg = bf16(rnd(B, heads, head_dim, scale=1.5))
        rb = rnd(B, heads, scale=2.0)
        cu = torch.arange(0, B + 1, dtype=torch.int32)
        o, st_torch = recurrent_ref(q, k, v, rg, rb, a_log, dt_bias, st_torch, cu.tolist(),
                                    [True] * B, heads, head_dim, scale, -5.0, 4)
        save(f"q{s}", q, d); save(f"k{s}", k, d); save(f"v{s}", v, d); save(f"raw_g{s}", rg, d)
        save(f"raw_beta{s}", rb, d); save(f"cu_seqlens{s}", cu, d)
        save(f"out{s}", o, d); save(f"state{s}", st_torch, d)
    return "recurrent_decode", f"B={B} steps={steps} 递归状态跨步累积"


def gen_gated_rmsnorm():
    d = case("gated_rmsnorm")
    rows, head_dim = 20, 128
    x = bf16(rnd(rows, head_dim, scale=1.5))
    w = bf16(rnd(head_dim, scale=0.5) + 1.0)
    gate = bf16(rnd(rows, head_dim, scale=2.0))
    save("x", x, d); save("w", w, d); save("gate", gate, d)
    save("out", gated_rmsnorm_ref(x, w, gate), d)
    return "gated_rmsnorm", f"rows={rows} head_dim={head_dim} 输出门 gated RMSNorm"


CASES = [gen_conv_prefill, gen_conv_prefill_varlen, gen_conv_update, gen_gate, gen_recurrent_prefill,
         gen_recurrent_varlen, gen_recurrent_decode, gen_gated_rmsnorm]


def main():
    os.makedirs(DIR, exist_ok=True)
    rows = []
    for fn in CASES:
        name, note = fn()
        rows.append((name, note))
        print(f"  {name:24s} {note}")
    print(f"\n{len(rows)} 个 case 写入 {DIR}")


if __name__ == "__main__":
    main()
