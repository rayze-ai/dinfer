#!/usr/bin/env python3
"""Qwen3.5 tiny 模型 + 纯 torch 参考 golden（Qwen35 里程碑 4 对拍）。

产出一个小到能进仓库的权重（safetensors，bf16）与参考结果（.npy）：
  3 层 [linear, linear, full]、hidden 256、全注意力 2 q / 1 kv 头 × 256（partial RoPE 64、输出门）、
  GDN 2 k / 4 v 头 × 128（conv 4）、inter 512、vocab 512。

参考口径 = 我们的 kernel 口径（`docs/loops/qwen35-gdn.md` / `qwen35-attn.md`）：权重 bf16，每个 kernel 的输出边界
round 回 bf16，内部 f32 累加。GDN 参考与 `scripts/verify/gen_gdn_refs.py` 同式（q/k l2norm + scale、delta rule、
gated RMSNorm(silu)）；全注意力参考与 HF `modeling_qwen3_5.py` 同式（zero-centered q/k norm、partial neox RoPE、
per-head 交错 q|gate 的 `chunk(2, -1)`、attention 后 `* sigmoid(gate)`）。
mrope 在纯文本下三路位置相同 → 退化为普通 1D RoPE（与 vLLM 一致；见 design-qwen35 §3）。

用法：/data/ref/venv-tf519/bin/python -I scripts/dev/make_qwen35_tiny.py test/qwen35/golden [--gen 16]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--gen", type=int, default=16)
ap.add_argument("--seed", type=int, default=20261019)
args = ap.parse_args()
OUT = args.out
os.makedirs(OUT, exist_ok=True)
MODEL = os.path.join(OUT, "tiny")
os.makedirs(MODEL, exist_ok=True)

torch.manual_seed(args.seed)
g = torch.Generator().manual_seed(args.seed)

H, V, INTER = 256, 512, 512
NH, NKV, D = 2, 1, 256
ROPE_DIM, THETA = 64, 10000.0
HV, HK, KD, VD, W = 4, 2, 128, 128, 4
EPS = 1e-6
LAYERS = ["linear_attention", "linear_attention", "full_attention"]
CONV_DIM = 2 * HK * KD + HV * VD   # 1024
N_FULL = sum(1 for t in LAYERS if t == "full_attention")

cfg = {
    "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
    "vocab_size": V, "hidden_size": H, "intermediate_size": INTER, "num_hidden_layers": len(LAYERS),
    "layer_types": LAYERS, "full_attention_interval": 4,
    "num_attention_heads": NH, "num_key_value_heads": NKV, "head_dim": D,
    "partial_rotary_factor": ROPE_DIM / D, "attn_output_gate": True, "attention_bias": False,
    "rope_parameters": {"rope_theta": THETA, "mrope_section": [11, 11, 10], "mrope_interleaved": True},
    "linear_num_key_heads": HK, "linear_num_value_heads": HV,
    "linear_key_head_dim": KD, "linear_value_head_dim": VD, "linear_conv_kernel_dim": W,
    "mamba_ssm_dtype": "float32", "rms_norm_eps": EPS, "max_position_embeddings": 512,
    "tie_word_embeddings": False, "hidden_act": "silu", "mtp_num_hidden_layers": 0,
    "eos_token_id": V - 1, "bos_token_id": V - 1,
}
with open(os.path.join(MODEL, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)


def rnd(*shape, scale=None):
    if scale is None:
        scale = 1.0 / max(shape[-1], 1) ** 0.5
    return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)


W_ = {}
W_["model.embed_tokens.weight"] = rnd(V, H, scale=0.05)
for l, lt in enumerate(LAYERS):
    p = f"model.layers.{l}"
    W_[f"{p}.input_layernorm.weight"] = (torch.randn(H, generator=g) * 0.2).to(torch.bfloat16)  # zero-centered：(1+w)
    W_[f"{p}.post_attention_layernorm.weight"] = (torch.randn(H, generator=g) * 0.2).to(torch.bfloat16)  # zero-centered：(1+w)
    W_[f"{p}.mlp.gate_proj.weight"] = rnd(INTER, H)
    W_[f"{p}.mlp.up_proj.weight"] = rnd(INTER, H)
    W_[f"{p}.mlp.down_proj.weight"] = rnd(H, INTER)
    if lt == "linear_attention":
        q = f"{p}.linear_attn"
        W_[f"{q}.in_proj_qkv.weight"] = rnd(CONV_DIM, H)
        W_[f"{q}.in_proj_z.weight"] = rnd(HV * VD, H)
        W_[f"{q}.in_proj_a.weight"] = rnd(HV, H)
        W_[f"{q}.in_proj_b.weight"] = rnd(HV, H)
        W_[f"{q}.conv1d.weight"] = (torch.randn(CONV_DIM, 1, W, generator=g) * 0.5).to(torch.bfloat16)
        W_[f"{q}.A_log"] = torch.log(torch.empty(HV).uniform_(0.05, 4.0, generator=g)).to(torch.float32)
        W_[f"{q}.dt_bias"] = torch.zeros(HV, dtype=torch.float32)
        W_[f"{q}.norm.weight"] = torch.ones(VD, dtype=torch.bfloat16)
        W_[f"{q}.out_proj.weight"] = rnd(H, HV * VD)
    else:
        s = f"{p}.self_attn"
        W_[f"{s}.q_proj.weight"] = rnd(NH * D * 2, H)
        W_[f"{s}.k_proj.weight"] = rnd(NKV * D, H)
        W_[f"{s}.v_proj.weight"] = rnd(NKV * D, H)
        W_[f"{s}.o_proj.weight"] = rnd(H, NH * D)
        W_[f"{s}.q_norm.weight"] = (torch.randn(D, generator=g) * 0.02).to(torch.bfloat16)  # zero-centered：(1+w)
        W_[f"{s}.k_norm.weight"] = (torch.randn(D, generator=g) * 0.02).to(torch.bfloat16)
W_["model.norm.weight"] = (torch.randn(H, generator=g) * 0.2).to(torch.bfloat16)  # zero-centered：(1+w)
W_["lm_head.weight"] = rnd(V, H, scale=0.05)
save_file(W_, os.path.join(MODEL, "model.safetensors"))


# ---------------------------------------------------------------- 参考 forward（f32 内部 + bf16 边界）
def qb(x):
    return x.to(torch.bfloat16).float()


def rmsnorm(x, w, eps=EPS):
    """与 kernel 同序：f32 归一化 → 乘权重 → 落 bf16（`rmsnorm_bf16` / `gather_rmsnorm`）。"""
    v = x.float().pow(2).mean(-1, keepdim=True)
    return qb((x.float() * torch.rsqrt(v + eps)) * w.float())


def silu(x):
    return x * torch.sigmoid(x)


def softplus(x):
    return torch.where(x <= 20, torch.log1p(torch.exp(x.clamp(max=30))), x)


def rope_table():
    half = ROPE_DIM // 2
    inv = 1.0 / (THETA ** (torch.arange(half, dtype=torch.float64) * 2.0 / ROPE_DIM))
    return inv


INV = rope_table()


def rope(x, pos):
    """x: [T, heads, D]，只旋前 ROPE_DIM 维（pair i, i+half）。"""
    half = ROPE_DIM // 2
    x = x.clone()
    p = torch.tensor(pos, dtype=torch.float64)
    f = p[:, None] * INV[None, :]
    c, s = f.cos().float(), f.sin().float()
    a = x[..., :half]
    b = x[..., half:ROPE_DIM]
    x[..., :half] = a * c[:, None, :] - b * s[:, None, :]
    x[..., half:ROPE_DIM] = b * c[:, None, :] + a * s[:, None, :]
    return x


class GdnState:
    def __init__(self):
        self.conv = torch.zeros(CONV_DIM, W - 1, dtype=torch.float32)  # bf16 语义，但值都是 bf16
        self.rec = torch.zeros(HV, VD, KD, dtype=torch.float32)


def gdn_forward(x, Wd, st, p):
    """x: [T, H] f32（调用方保证已是 bf16 值）。返回 [T, H] bf16 值。"""
    mixed = qb(x @ Wd[f"{p}.in_proj_qkv.weight"].float().T)
    z = qb(x @ Wd[f"{p}.in_proj_z.weight"].float().T)
    a = qb(x @ Wd[f"{p}.in_proj_a.weight"].float().T)
    b = qb(x @ Wd[f"{p}.in_proj_b.weight"].float().T)
    cw = Wd[f"{p}.conv1d.weight"].float().reshape(CONV_DIM, W)
    # 短卷积（state 列 0 最旧）
    T = mixed.shape[0]
    hist = st.conv.clone()  # [dim, W-1]
    bufs = []
    for t in range(T):
        win = torch.cat([hist, mixed[t : t + 1].T], dim=1)  # [dim, W]
        acc = (win * cw).sum(1)
        bufs.append(silu(acc))
        hist = win[:, 1:]
    st.conv = hist.detach().clone()
    conv = qb(torch.stack(bufs))  # [T, conv_dim]
    g_ = -torch.exp(Wd[f"{p}.A_log"].float()) * softplus(a.float() + Wd[f"{p}.dt_bias"].float())
    beta = torch.sigmoid(b.float())
    q, k, v = conv[:, : HK * KD], conv[:, HK * KD : 2 * HK * KD], conv[:, 2 * HK * KD :]
    q = q.reshape(T, HK, KD)
    k = k.reshape(T, HK, KD)
    v = v.reshape(T, HV, VD)
    o = torch.zeros(T, HV, VD)
    group = HV // HK
    for t in range(T):
        for hv in range(HV):
            h = hv // group
            qv = q[t, h].float()
            kv = k[t, h].float()
            qv = qv / torch.sqrt((qv * qv).sum() + 1e-6) * KD ** -0.5
            kv = kv / torch.sqrt((kv * kv).sum() + 1e-6)
            S = st.rec[hv]
            S.mul_(torch.exp(g_[t, hv]))
            vp = (v[t, hv].float() - S @ kv) * beta[t, hv]
            S.add_(torch.outer(vp, kv))
            o[t, hv] = S @ qv
    o = qb(o.reshape(T, HV * VD))
    y = rmsnorm(o.reshape(-1, VD), Wd[f"{p}.norm.weight"].float()).reshape(T, HV * VD)
    y = qb(y * silu(z.float()))
    return qb(y @ Wd[f"{p}.out_proj.weight"].float().T)


def full_forward(x, Wd, p, pos, mask_cache=None):
    """标准 causal attention；返回 [T, H]。用到 KV 缓存时由调用方保证 mask_cache 是历史。"""
    T = x.shape[0]
    qg = qb(x @ Wd[f"{p}.q_proj.weight"].float().T)  # [T, NH*2*D]，per-head 交错
    kk = qb(x @ Wd[f"{p}.k_proj.weight"].float().T)
    vv = qb(x @ Wd[f"{p}.v_proj.weight"].float().T)
    qg = qg.reshape(T, NH, 2 * D)
    q = qg[..., :D]
    gate = qg[..., D:]
    q = rope(rmsnorm(q, 1.0 + Wd[f"{p}.q_norm.weight"].float()[None, None, :]), pos)
    k = rope(rmsnorm(kk.reshape(T, NKV, D), 1.0 + Wd[f"{p}.k_norm.weight"].float()[None, None, :]), pos)
    v = vv.reshape(T, NKV, D)
    group = NH // NKV
    qh = q.transpose(0, 1)  # [NH, T, D]
    kh = k.transpose(0, 1)  # [NKV, T, D]
    vh = v.transpose(0, 1)
    outs = []
    for h in range(NH):
        kvh = h // group
        sc = (qh[h] @ kh[kvh].T) * (D ** -0.5)
        idx = torch.arange(T)
        sc = sc.masked_fill(idx[None, :] > idx[:, None], float("-inf"))
        outs.append(torch.softmax(sc, -1) @ vh[kvh])
    o = qb(torch.stack(outs).transpose(0, 1))  # [T, NH, D]
    o = qb((o * torch.sigmoid(gate.float())).reshape(T, NH * D))
    return qb(o @ Wd[f"{p}.o_proj.weight"].float().T)


def forward(ids, Wd, states=None, pos0=0, cache=None):
    """整段前向（测试用：每次重算全序列）。states/cache 为 None 时本次从头开始。"""
    T = len(ids)
    x = qb(Wd["model.embed_tokens.weight"].float()[torch.tensor(ids)])
    pos = list(range(pos0, pos0 + T))
    st = states if states is not None else [GdnState() for _ in LAYERS]
    for l, lt in enumerate(LAYERS):
        p = f"model.layers.{l}"
        h = rmsnorm(x, 1.0 + Wd[f"{p}.input_layernorm.weight"].float())
        a = gdn_forward(h, Wd, st[l], f"{p}.linear_attn") if lt == "linear_attention" else full_forward(h, Wd, f"{p}.self_attn", pos)
        x = qb(x + a)
        h2 = rmsnorm(x, 1.0 + Wd[f"{p}.post_attention_layernorm.weight"].float())
        gu = qb(h2 @ Wd[f"{p}.mlp.gate_proj.weight"].float().T)
        uu = qb(h2 @ Wd[f"{p}.mlp.up_proj.weight"].float().T)
        act = qb(silu(gu) * uu)
        m = qb(act @ Wd[f"{p}.mlp.down_proj.weight"].float().T)
        x = qb(x + m)
    x = rmsnorm(x, 1.0 + Wd["model.norm.weight"].float())
    return qb(x @ Wd["lm_head.weight"].float().T)  # [T, V]


def save_npy(name, a):
    np.save(os.path.join(OUT, name), a)


def greedy(prompt, gen):
    """`step_logits[i]` 是**预测第 i 个生成 token**的那一行 logits（i=0 即 prefill 末行），与对拍口径一致。"""
    Wd = W_
    logits = forward(prompt, Wd)
    out = []
    seq = list(prompt)
    step_logits = []
    for i in range(gen):
        step_logits.append(logits[-1].numpy())
        nxt = int(torch.argmax(logits[-1]))
        out.append(nxt)
        seq.append(nxt)
        if i + 1 < gen:
            logits = forward(seq, Wd)
    return out, step_logits


prompts = {
    "a": [(i * 37 + 11) % (V - 2) + 1 for i in range(12)],
    "b": [(i * 53 + 7) % (V - 2) + 1 for i in range(7)],
    "c": [(i * 29 + 3) % (V - 2) + 1 for i in range(9)],
}
# 注：tiny 随机权重下 greedy 常出现平局（top-2 gap < 0.1），这是随机权重的性质，不是数值问题。
# 对拍口径（docs/pitfalls"硬判决不可逐位复现"）：只有双方都自信的行才断言 argmax，见 `golden_parity.rs`。
for name, p in prompts.items():
    lg = forward(p, W_)
    save_npy(f"prompt_{name}.npy", np.asarray(p, dtype=np.int32))
    save_npy(f"expect_logits_{name}.npy", lg.numpy().astype(np.float32))
    gr, steps = greedy(p, args.gen)
    save_npy(f"expect_greedy_{name}.npy", np.asarray(gr, dtype=np.int32))
    save_npy(f"expect_step_logits_{name}.npy", np.stack(steps).astype(np.float32))
    print(f"{name}: prompt {len(p)} tok, greedy {gr[:8]}", file=sys.stderr)

# 参考侧 logits 的 top-2 gap（对拍用例按 gap 决定哪些行可断言 argmax）
lg = forward(prompts["a"], W_)
top2 = torch.topk(lg, 2, dim=-1).values
print(f"top-2 gap: min {float(top2[:,0]-top2[:,1]).__round__(4) if False else (top2[:,0]-top2[:,1]).min():.4f}", file=sys.stderr)
print("ok", file=sys.stderr)
