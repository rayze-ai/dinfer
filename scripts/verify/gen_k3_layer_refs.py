#!/usr/bin/env python3
"""Kimi-K3 真权重**整层**参考（里程碑 C-2，docs/loops/k3-moe.md「C-2」）：第 1 层 = KDA + latent MoE + AttnRes。

在 vLLM 容器里跑（torch + vLLM 自带 fla kda；**无 fla 包、无 compressed-tensors 解包**，MoE 用纯 torch 按 HF 代码算，
MXFP4 解包按 scripts/verify/k3_mxfp4_realcheck.py 的口径）：
  python3 -I gen_k3_layer_refs.py <模型目录> <输出目录> [--layer 1] [--T 64] [--seed 0]

严格按 HF modeling_kimi_linear.py::KimiDecoderLayer._forward_attn_residual：
  x  = prefix_sum（上一层输出），B = [b0]（块残差：第 0 层是写块层，b0 = 第 0 层的输入；这里 b0 随机，seed+1）
  h  = _apply_attn_res([b0, x], self_attention_res_proj/norm)
  a  = kda(input_layernorm(h))                       （复用 gen_k3_kda_layer_refs.py 的 fla 路径）
  p  = x + a
  m  = _apply_attn_res([b0, p], mlp_res_proj/norm)
  y  = KimiSparseMoeBlock(post_attention_layernorm(m))：
       gate：f32 logits → sigmoid → +e_score_correction_bias → top-16 → w = scores/Σ × 1.0
       z = routed_expert_down_proj(mn)；每专家 w2(situ(cat[w1 z, w3 z]))（MXFP4 解包→bf16 linear）；
       Σ_k w_k·y_k（f32）→ bf16 → routed_expert_norm → routed_expert_up_proj；+ shared_experts(mn)
  out = p + y
输出 .npy：x_bf16 / b0_bf16（<u2 位存）、h / a / p / m / mn / z / ysum / yrouted / y_moe / out（f32）、
topk_ids（int32 [T,16]）、topk_w（f32）、logits（f32 [T,E]）、meta.json。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_k3_kda_layer_refs import conv1d_causal, save_bf16, save_f32  # noqa: E402

PREFIX = "language_model.model."
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def dequant_mxfp4(packed_u8, scale_u8, group=32):
    """[N, K/2] u8 + [N, K/group] u8 → f32 [N, K]。与 mxfp4.cu / k3_mxfp4_realcheck.py 逐位同口径。"""
    p = packed_u8.to(torch.int32)
    lo, hi = p & 0x0F, (p >> 4) & 0x0F
    nib = torch.stack([lo, hi], dim=-1).reshape(p.shape[0], -1)  # 低半字节是偶数下标
    val = E2M1.to(p.device)[nib & 7]
    val = torch.where((nib & 8) != 0, -val, val)
    s = (scale_u8.to(torch.int32) << 23).view(torch.float32)  # e8m0：__uint_as_float(s << 23)
    return val * s.repeat_interleave(group, dim=1)


class Shard:
    """按需读张量（safetensors mmap，只读含该层的分片）。"""

    def __init__(self, model_dir, layer, dev):
        from safetensors import safe_open

        idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        pre = f"{PREFIX}layers.{layer}."
        self.map = {k[len(pre):]: v for k, v in idx.items() if k.startswith(pre)}
        self.files = sorted(set(self.map.values()))
        self.sf = {f: safe_open(os.path.join(model_dir, f), framework="pt", device="cpu") for f in self.files}
        self.pre, self.dev = pre, dev

    def get(self, rel):
        return self.sf[self.map[rel]].get_tensor(self.pre + rel).to(self.dev)

    def names(self):
        return self.map.keys()


def rmsnorm(x, w, eps):
    xf = x.float()
    return w.to(x.dtype) * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def apply_attn_res(prefix_sum, blocks, proj_w, norm_w, eps):
    """HF _apply_attn_res：v = cat(blocks, prefix)，k = v·rsqrt(mean v²+eps)，s = Σ k·(norm.w·proj.w)，softmax，Σ p·v。"""
    v = torch.stack(blocks + [prefix_sum], dim=1)  # [T, R, H]
    vf = v.float()
    k = vf * torch.rsqrt(vf.pow(2).mean(-1, keepdim=True) + eps)
    sw = norm_w.float() * proj_w.squeeze(0).float()
    scores = (k * sw).sum(-1)  # [T, R]
    probs = scores.softmax(-1).unsqueeze(1)
    return torch.matmul(probs, vf).squeeze(1).to(v.dtype), scores


def situ(x, beta, linear_beta):
    d = x.shape[-1] // 2
    g, u = x[..., :d].float(), x[..., d:].float()
    a = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * torch.tanh(u / linear_beta)
    return (a * u).to(x.dtype)


def kda(sh, h, cfg, T, dev):
    """gen_k3_kda_layer_refs.py 的 fla 路径（已在 C-1 对拍过）。"""
    la = cfg["linear_attn_config"]
    H, K = la["num_heads"], la["head_dim"]
    eps, lb = cfg["rms_norm_eps"], la["gate_lower_bound"]
    bf = torch.bfloat16
    w = lambda n: sh.get(f"self_attn.{n}.weight").to(bf)
    lin = lambda n, t: F.linear(t, w(n))
    conv = lambda n, t: conv1d_causal(t, sh.get(f"self_attn.{n}_conv1d.weight").to(bf))
    q, k, v = conv("q", lin("q_proj", h)), conv("k", lin("k_proj", h)), conv("v", lin("v_proj", h))
    raw_g = lin("f_b_proj", lin("f_a_proj", h))
    raw_beta = lin("b_proj", h).float()
    g2 = lin("g_proj", h)
    a_log = sh.get("self_attn.A_log").float()[:H].contiguous()
    dt_bias = sh.get("self_attn.dt_bias").float()
    o_norm_w = sh.get("self_attn.o_norm.weight").to(bf)
    sys.path.insert(0, "/")
    from vllm.third_party.flash_linear_attention.ops.kda import chunk_kda_with_fused_gate

    q3, k3, v3, g3 = (t.view(T, H, K) for t in (q, k, v, raw_g))
    o, _ = chunk_kda_with_fused_gate(
        q=q3.unsqueeze(0), k=k3.unsqueeze(0), v=v3.unsqueeze(0), raw_g=g3.unsqueeze(0),
        beta=torch.sigmoid(raw_beta).unsqueeze(0), A_log=a_log, g_bias=dt_bias, scale=K ** -0.5,
        initial_state=None, output_final_state=True, lower_bound=lb, use_qk_l2norm_in_kernel=True, cu_seqlens=None,
    )
    o = o[0]
    of = o.float()
    n = (of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps) * o_norm_w.float() * torch.sigmoid(g2.view(T, H, K).float())).to(bf)
    return lin("o_proj", n.view(T, H * K))


def moe(sh, mn, cfg, dev, out_dir):
    bf = torch.bfloat16
    T = mn.shape[0]
    E, topk = cfg["num_experts"], cfg["num_experts_per_token"]
    beta, lbeta = cfg.get("activation_situ_beta") or 1.0, cfg.get("activation_situ_linear_beta")
    eps = cfg["rms_norm_eps"]
    assert cfg["moe_router_activation_func"] == "sigmoid" and cfg["num_expert_group"] == 1 and cfg["topk_method"] == "noaux_tc"
    # gate（HF KimiMoEGate）：f32 linear
    gw = sh.get("block_sparse_moe.gate.weight")
    bias = sh.get("block_sparse_moe.gate.e_score_correction_bias").float()
    logits = F.linear(mn.float(), gw.float())
    scores = logits.sigmoid()
    choice = scores + bias.unsqueeze(0)
    _, topk_idx = torch.topk(choice, k=topk, dim=-1, sorted=False)
    topk_w = scores.gather(1, topk_idx)
    if topk > 1 and cfg["moe_renormalize"]:
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)
    topk_w = topk_w * cfg["routed_scaling_factor"]
    # 可判定性：第 16 与第 17 大 choice 的差（硬判决只在差 > 阈值的行断言）
    srt = choice.sort(dim=-1, descending=True).values
    margin = srt[:, topk - 1] - srt[:, topk]
    save_f32(os.path.join(out_dir, "logits.npy"), logits)
    np.save(os.path.join(out_dir, "topk_ids.npy"), topk_idx.cpu().numpy().astype(np.int32))
    save_f32(os.path.join(out_dir, "topk_w.npy"), topk_w)
    save_f32(os.path.join(out_dir, "topk_margin.npy"), margin)

    # latent：down_proj → 专家 → Σ → norm → up_proj
    z = F.linear(mn, sh.get("block_sparse_moe.routed_expert_down_proj.weight").to(bf))
    save_f32(os.path.join(out_dir, "z.npy"), z)
    ids = topk_idx.cpu().numpy()
    used = sorted(set(ids.flatten().tolist()))
    out_pairs = torch.zeros(T, topk, z.shape[1], dtype=bf, device=dev)
    t0 = time.time()
    for e in used:
        p = f"block_sparse_moe.experts.{e}."
        w1 = dequant_mxfp4(sh.get(p + "w1.weight_packed"), sh.get(p + "w1.weight_scale")).to(bf)
        w3 = dequant_mxfp4(sh.get(p + "w3.weight_packed"), sh.get(p + "w3.weight_scale")).to(bf)
        w2 = dequant_mxfp4(sh.get(p + "w2.weight_packed"), sh.get(p + "w2.weight_scale")).to(bf)
        rows, ks = np.nonzero(ids == e)
        xe = z[torch.as_tensor(rows, device=dev)]
        gate_up = torch.cat([F.linear(xe, w1), F.linear(xe, w3)], dim=-1)
        ye = F.linear(situ(gate_up, beta, lbeta), w2)
        out_pairs[torch.as_tensor(rows, device=dev), torch.as_tensor(ks, device=dev)] = ye
    print(f"  {len(used)} 个专家用到（{T}×{topk} pairs），解包+算 {time.time()-t0:.1f}s", flush=True)
    ysum = (out_pairs.float() * topk_w.unsqueeze(-1)).sum(dim=1).to(bf)  # HF：.type(topk_weight.dtype).mul_().sum().type(bf16)
    save_f32(os.path.join(out_dir, "ysum.npy"), ysum)
    y = ysum
    if cfg.get("latent_moe_use_norm"):
        y = rmsnorm(y, sh.get("block_sparse_moe.routed_expert_norm.weight"), eps)
    y = F.linear(y, sh.get("block_sparse_moe.routed_expert_up_proj.weight").to(bf))
    save_f32(os.path.join(out_dir, "yrouted.npy"), y)
    # shared experts：KimiMLP（cat[gate, up] → situ → down）
    sp = "block_sparse_moe.shared_experts."
    gu = torch.cat([F.linear(mn, sh.get(sp + "gate_proj.weight").to(bf)), F.linear(mn, sh.get(sp + "up_proj.weight").to(bf))], dim=-1)
    ysh = F.linear(situ(gu, beta, lbeta), sh.get(sp + "down_proj.weight").to(bf))
    y_moe = y + ysh
    save_f32(os.path.join(out_dir, "y_moe.npy"), y_moe)
    return y_moe, len(used)


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
    bf = torch.bfloat16
    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))["text_config"]
    hidden, eps, bs = cfg["hidden_size"], cfg["rms_norm_eps"], cfg["attn_res_block_size"]
    assert (a.layer + 1) in cfg["linear_attn_config"]["kda_layers"], "不是 KDA 层"
    assert a.layer >= cfg["first_k_dense_replace"], "不是 MoE 层"
    assert a.layer % bs != 0, "本脚本按非写块层写（第 1 层）"

    t0 = time.time()
    sh = Shard(a.model_dir, a.layer, dev)
    print(f"分片 {sh.files}，{len(list(sh.names()))} 张量，{time.time()-t0:.1f}s", flush=True)

    g = torch.Generator(device="cpu").manual_seed(a.seed)
    x = (torch.randn(a.T, hidden, generator=g) * 0.5).to(bf).to(dev)
    g1 = torch.Generator(device="cpu").manual_seed(a.seed + 1)
    b0 = (torch.randn(a.T, hidden, generator=g1) * 0.5).to(bf).to(dev)  # 第 0 层写入的块残差（HF 里 = embedding 输出）
    save_bf16(os.path.join(a.out_dir, "x_bf16.npy"), x)
    save_bf16(os.path.join(a.out_dir, "b0_bf16.npy"), b0)

    h, s_a = apply_attn_res(x, [b0], sh.get("self_attention_res_proj.weight"), sh.get("self_attention_res_norm.weight"), eps)
    save_f32(os.path.join(a.out_dir, "h.npy"), h)
    hn = rmsnorm(h, sh.get("input_layernorm.weight"), eps)
    att = kda(sh, hn, cfg, a.T, dev)
    save_f32(os.path.join(a.out_dir, "a.npy"), att)
    p = x + att
    save_f32(os.path.join(a.out_dir, "p.npy"), p)
    m, s_m = apply_attn_res(p, [b0], sh.get("mlp_res_proj.weight"), sh.get("mlp_res_norm.weight"), eps)
    save_f32(os.path.join(a.out_dir, "m.npy"), m)
    mn = rmsnorm(m, sh.get("post_attention_layernorm.weight"), eps)
    save_f32(os.path.join(a.out_dir, "mn.npy"), mn)
    y_moe, used = moe(sh, mn, cfg, dev, a.out_dir)
    out = p + y_moe
    save_f32(os.path.join(a.out_dir, "out.npy"), out)
    pa = s_a.softmax(-1)
    pm = s_m.softmax(-1)
    print(f"  attn 门 prob(prefix) mean {pa[:,1].mean():.3f}  mlp 门 prob(prefix) mean {pm[:,1].mean():.3f}")
    print(f"  |a| max {att.float().abs().max():.4f}  |y_moe| max {y_moe.float().abs().max():.4f}  |out| max {out.float().abs().max():.4f}")
    meta = dict(layer=a.layer, T=a.T, seed=a.seed, hidden=hidden, eps=eps, files=sh.files, experts_used=used,
                torch=torch.__version__, attn_gate_prob_prefix_mean=float(pa[:, 1].mean()), mlp_gate_prob_prefix_mean=float(pm[:, 1].mean()))
    json.dump(meta, open(os.path.join(a.out_dir, "meta.json"), "w"), indent=1)
    print("saved", a.out_dir)


if __name__ == "__main__":
    main()
