#!/usr/bin/env python3
"""Kimi-K3 真权重单层 KDA 参考（里程碑 C-1，docs/loops/k3-kda.md）。

在 vLLM 容器里跑（有 torch + vLLM 自带的 fla kda ops；无 fla 包）：
  python3 -I gen_k3_kda_layer_refs.py <模型目录> <输出目录> [--layer 1] [--T 64] [--seed 0]

做什么：
  1. 用 safetensors 只读含该层的分片（不加载整模，1.56 TB）；
  2. x ~ N(0,1)·0.5 的 bf16 随机输入 [T, hidden]（seed 固定）；h = KimiRMSNorm(x)（w·norm(x)，非 zero-centered）；
  3. 严格按 HF modeling_kimi_linear.py::KimiDeltaAttention.forward 算 kda(h)：
     q/k/v_proj → ShortConvolution(silu, 因果 depthwise, 无 bias) → g = f_b(f_a(h))、beta = b_proj(h).float()
     → chunk_kda（vLLM 的 chunk_kda_with_fused_gate：A_log / dt_bias / lower_bound / qk l2norm / beta sigmoid 都在 kernel 内）
     → o_norm = FusedRMSNormGated(sigmoid)(o, g_proj(h)) → o_proj。
  4. 另算一份纯 torch f32 顺序递归（与 docs/loops/k3-kda.md 规格同式）做三方核对。
输出：x_bf16.npy、h_bf16.npy（<u2 位存）、y_fla.npy、n_fla.npy、y_torch.npy（f32）、meta.json。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

PREFIX = "language_model.model."


def save_bf16(path, t):
    np.save(path, t.detach().to(torch.bfloat16).view(torch.int16).cpu().numpy().view(np.uint16))


def save_f32(path, t):
    np.save(path, t.detach().float().cpu().numpy())


def load_layer(model_dir, layer, dev):
    from safetensors import safe_open

    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    pre = f"{PREFIX}layers.{layer}."
    want = {k: v for k, v in idx.items() if k.startswith(pre) and ".experts." not in k}
    files = sorted(set(want.values()))
    out = {}
    for f in files:
        with safe_open(os.path.join(model_dir, f), framework="pt", device="cpu") as sf:
            for k in want:
                if want[k] == f and k.startswith(pre + "self_attn.") or k == pre + "input_layernorm.weight":
                    if want[k] == f:
                        out[k[len(pre):]] = sf.get_tensor(k).to(dev)
    return out, files


def conv1d_causal(x, w, act_silu=True):
    """HF fla ShortConvolution（无 cache）：x [T, D] bf16，w [D, 1, W] → silu(causal depthwise conv)。
    fla 用 F.conv1d(x.T, w, padding=W-1)[..., :T]，再 silu；bf16 输入输出。"""
    T, D = x.shape
    W = w.shape[-1]
    y = F.conv1d(x.t().unsqueeze(0).to(w.dtype), w, padding=W - 1, groups=D)[0, :, :T].t()
    if act_silu:
        y = F.silu(y)
    return y.to(x.dtype)


def torch_recurrent(q, k, v, raw_g, raw_beta, a_log, dt_bias, lower_bound, scale):
    """纯 torch f32 顺序递归（docs/loops/k3-kda.md 规格）。q/k/v/raw_g [T,H,K]，raw_beta [T,H]。"""
    q, k, v, raw_g, raw_beta = (t.float() for t in (q, k, v, raw_g, raw_beta))
    T, H, K = q.shape
    a = torch.exp(a_log.float()).view(1, H, 1)
    gate = lower_bound * torch.sigmoid(a * (raw_g + dt_bias.float().view(1, H, K)))
    qn = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * scale
    kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    beta = torch.sigmoid(raw_beta)
    S = torch.zeros(H, K, K, device=q.device)  # [H, V, K]
    o = torch.empty(T, H, K, device=q.device)
    for t in range(T):
        S = S * torch.exp(gate[t]).unsqueeze(1)
        vp = (v[t] - (S * kn[t].unsqueeze(1)).sum(-1)) * beta[t].unsqueeze(-1)
        S = S + vp.unsqueeze(-1) * kn[t].unsqueeze(1)
        o[t] = (S * qn[t].unsqueeze(1)).sum(-1)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device("cuda")
    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))["text_config"]
    la = cfg["linear_attn_config"]
    H, K, W = la["num_heads"], la["head_dim"], la["short_conv_kernel_size"]
    hidden, eps, lb = cfg["hidden_size"], cfg["rms_norm_eps"], la["gate_lower_bound"]
    assert (a.layer + 1) in la["kda_layers"], f"layer {a.layer} 不是 KDA 层"

    t0 = time.time()
    w, files = load_layer(a.model_dir, a.layer, dev)
    print(f"loaded {len(w)} tensors from {files} in {time.time()-t0:.1f}s", flush=True)
    for name in ["q_proj", "k_proj", "v_proj", "g_proj", "f_a_proj", "f_b_proj", "b_proj", "o_proj"]:
        assert f"self_attn.{name}.weight" in w, name
    bf = torch.bfloat16
    # HF from_pretrained(dtype=bf16) 会把 conv1d(F32) / o_norm(F32) 转 bf16；A_log / dt_bias 在 HF 里是 bf16 参数但 fla kernel
    # 内 .float()；vLLM 把 A_log / dt_bias 保持 f32。这里按 vLLM 口径：A_log / dt_bias f32，conv / o_norm bf16（与 Rust 侧一致）。
    conv_w = [w[f"self_attn.{n}_conv1d.weight"].to(bf) for n in ("q", "k", "v")]
    a_log = w["self_attn.A_log"].float()
    assert a_log.shape[0] >= H and torch.all(a_log[H:] == 0), "A_log 尾部应为 0 填充"
    a_log = a_log[:H].contiguous()
    dt_bias = w["self_attn.dt_bias"].float()
    o_norm_w = w["self_attn.o_norm.weight"].to(bf)

    g = torch.Generator(device="cpu").manual_seed(a.seed)
    x = (torch.randn(a.T, hidden, generator=g) * 0.5).to(bf).to(dev)
    # KimiRMSNorm：w · (x.float() · rsqrt(mean(x²)+eps)).to(bf16)
    xf = x.float()
    h = (w["input_layernorm.weight"].to(bf) * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(bf))
    save_bf16(os.path.join(a.out_dir, "x_bf16.npy"), x)
    save_bf16(os.path.join(a.out_dir, "h_bf16.npy"), h)

    lin = lambda n, t: F.linear(t, w[f"self_attn.{n}.weight"].to(bf))
    q = conv1d_causal(lin("q_proj", h), conv_w[0])
    k = conv1d_causal(lin("k_proj", h), conv_w[1])
    v = conv1d_causal(lin("v_proj", h), conv_w[2])
    raw_g = lin("f_b_proj", lin("f_a_proj", h))  # [T, H·K] bf16
    raw_beta = lin("b_proj", h).float()  # [T, H]
    g2 = lin("g_proj", h)  # 输出门
    q3, k3, v3, g3 = (t.view(a.T, H, K) for t in (q, k, v, raw_g))
    scale = K ** -0.5

    # --- fla（vLLM 自带）chunk 路径
    sys.path.insert(0, "/")
    from vllm.third_party.flash_linear_attention.ops.kda import chunk_kda_with_fused_gate

    # vLLM 的 chunk_kda_with_fused_gate 不吃 HF 的 use_beta_sigmoid_in_kernel（**kwargs 吞掉）：beta 要先 sigmoid
    # （vLLM kda.py 在投影融合 kernel 里做 sigmoid；只有 flashinfer 路径有 beta_is_logit=True）。第一版漏了 → cos 0.23。
    o_fla, _ = chunk_kda_with_fused_gate(
        q=q3.unsqueeze(0), k=k3.unsqueeze(0), v=v3.unsqueeze(0), raw_g=g3.unsqueeze(0),
        beta=torch.sigmoid(raw_beta).unsqueeze(0),
        A_log=a_log, g_bias=dt_bias, scale=scale, initial_state=None, output_final_state=True,
        lower_bound=lb, use_qk_l2norm_in_kernel=True, cu_seqlens=None,
    )
    o_fla = o_fla[0]  # [T, H, K] bf16
    # o_norm：与 vLLM FusedRMSNormGated.forward_native（activation='sigmoid'）逐式相同：
    #   out = (x.float()·rsqrt(mean(x²)+eps)·w.float()·sigmoid(g.float())).to(x.dtype)
    # （FusedRMSNormGated 是 vLLM CustomOp，实例化要 vllm config 上下文，这里直接展开）
    def gated_rmsnorm(x, g):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * o_norm_w.float() * torch.sigmoid(g.float())).to(x.dtype)

    n_fla = gated_rmsnorm(o_fla, g2.view(a.T, H, K)).view(a.T, H * K)  # bf16
    y_fla = lin("o_proj", n_fla)
    save_f32(os.path.join(a.out_dir, "n_fla.npy"), n_fla)
    save_f32(os.path.join(a.out_dir, "y_fla.npy"), y_fla)

    # --- 纯 torch f32 顺序递归（三方核对）
    o_t = torch_recurrent(q3, k3, v3, g3, raw_beta, a_log, dt_bias, lb, scale)  # f32
    g2f = g2.float().view(a.T, H, K)
    n_t = (o_t * torch.rsqrt(o_t.pow(2).mean(-1, keepdim=True) + eps) * o_norm_w.float() * torch.sigmoid(g2f)).to(bf)
    y_t = lin("o_proj", n_t.view(a.T, H * K))
    save_f32(os.path.join(a.out_dir, "y_torch.npy"), y_t)

    def stat(nm, p, r):
        p, r = p.float().flatten(), r.float().flatten()
        cos = F.cosine_similarity(p, r, dim=0).item()
        rel = ((p - r).abs().max() / r.abs().max()).item()
        print(f"  {nm:<22} rel {rel:.3e} cos {cos:.7f}")
        return rel, cos

    print("参考自检：")
    rel_o, cos_o = stat("fla o vs torch o", o_fla, o_t)
    stat("fla y vs torch y", y_fla, y_t)
    print(f"  y_fla |max| {y_fla.float().abs().max().item():.4f}  mean|y| {y_fla.float().abs().mean().item():.4f}")
    meta = dict(layer=a.layer, T=a.T, seed=a.seed, H=H, K=K, W=W, hidden=hidden, eps=eps, lower_bound=lb,
                files=files, torch=torch.__version__, fla_vs_torch_o=dict(rel=rel_o, cos=cos_o))
    json.dump(meta, open(os.path.join(a.out_dir, "meta.json"), "w"), indent=1)
    print("saved", a.out_dir)


if __name__ == "__main__":
    main()
