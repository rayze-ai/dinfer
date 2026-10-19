#!/usr/bin/env python3
"""Kimi-K3 真权重**前 N 层主干**参考（里程碑 C-3，docs/loops/k3-model.md）：embed → N 层 → output_attn_res → norm → lm_head。

在 vLLM 容器里跑（torch + vLLM 自带 fla kda；**无 fla 包**，所以不能实例化 HF 类；全部按 HF modeling_kimi_linear.py 用纯 torch
逐行写，KDA 复用 gen_k3_layer_refs.kda（C-1 已对拍），MoE 复用 gen_k3_layer_refs.moe（C-2 已对拍），MLA 本文件实现）：
  python3 -I gen_k3_model_refs.py <模型目录> <输出目录> [--layers 5] [--text "..."] [--T 32]

严格按 HF KimiLinearModel.forward + KimiDecoderLayer._forward_attn_residual：
  x0 = embed(ids)；B = []
  每层：h = B 非空 ? res(B∪{x}) : x；写块层（l%12==0）B ← B∪{x}，p = attn(norm(h))（否则 p = x + attn）；
        m = res(B∪{p})；out = p + mlp(norm(m))
  final = norm(res_out(B∪{x_N}))；logits = lm_head(final)
MLA（KimiMLAAttention，mla_use_nope=True，无 RoPE；mla_use_output_gate=True）：
  q = q_b(rmsnorm(q_a(x))).view(T,H,192)；[kv_c | k_pe] = kv_a_with_mqa(x)；kv = kv_b(rmsnorm(kv_c)).view(T,H,256)
  k = [kv.nope | k_pe 广播]，v = kv.v；softmax(q·kᵀ·192^-0.5，causal)·v（f32）→ *sigmoid(g_proj(x)) → o_proj
输出 .npy：ids（int32 [T]）、embed、in{l} / layer{l}（f32 [T,hidden]，第 l 层输入 / 输出 prefix_sum）、res_out、final、logits_last（f32 [vocab]）、
每 MLA 层 attn{l}（attention 子块输出，o_proj 之后）、每 MoE 层 moe{l}/{topk_ids,topk_margin,...}（路由，隔离口径只在路由一致行判）、meta.json。
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
from gen_k3_kda_layer_refs import save_f32  # noqa: E402
from gen_k3_layer_refs import Shard, apply_attn_res, kda, moe, rmsnorm, situ  # noqa: E402

PREFIX = "language_model.model."


class TopShard:
    """顶层张量（embed / norm / output_attn_res / lm_head），只 mmap 含它们的分片。"""

    def __init__(self, model_dir, dev):
        from safetensors import safe_open

        idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        self.map = {k: v for k, v in idx.items() if k.startswith("language_model.") and ".layers." not in k}
        self.files = sorted(set(self.map.values()))
        self.sf = {f: safe_open(os.path.join(model_dir, f), framework="pt", device="cpu") for f in self.files}
        self.dev = dev

    def get(self, name):
        return self.sf[self.map[name]].get_tensor(name).to(self.dev)


def mla(sh, h, cfg, out_dir, layer):
    bf = torch.bfloat16
    T = h.shape[0]
    H = cfg["num_attention_heads"]
    nope, rope, v_dim = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"]
    q_lora, kv_lora = cfg["q_lora_rank"], cfg["kv_lora_rank"]
    assert cfg["mla_use_nope"] and cfg["mla_use_output_gate"]
    w = lambda n: sh.get(f"self_attn.{n}.weight").to(bf)
    lin = lambda n, t: F.linear(t, w(n))
    a_eps = 1e-6  # HF KimiMLAAttention: KimiRMSNorm(q_lora_rank) / KimiRMSNorm(kv_lora_rank) 没传 eps → 默认 1e-6（vLLM 用 rms_norm_eps）
    q = lin("q_b_proj", rmsnorm(lin("q_a_proj", h), sh.get("self_attn.q_a_layernorm.weight"), a_eps)).view(T, H, nope + rope)
    ckv = lin("kv_a_proj_with_mqa", h)
    kv_c, k_pe = ckv[:, :kv_lora], ckv[:, kv_lora:]
    kv = lin("kv_b_proj", rmsnorm(kv_c, sh.get("self_attn.kv_a_layernorm.weight"), a_eps)).view(T, H, nope + v_dim)
    k_nope, v = kv[..., :nope], kv[..., nope:]
    k = torch.cat([k_nope, k_pe.unsqueeze(1).expand(T, H, rope)], dim=-1)  # 无 RoPE
    # f32 causal attention（flash_attention_2 的 bf16 内部累加是 f32；这里全 f32 更接近"真值"）
    qf, kf, vf = q.float().transpose(0, 1), k.float().transpose(0, 1), v.float().transpose(0, 1)  # [H, T, d]
    s = torch.matmul(qf, kf.transpose(1, 2)) * (nope + rope) ** -0.5
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), 1)
    s = s.masked_fill(mask, float("-inf"))
    o = torch.matmul(s.softmax(-1), vf).transpose(0, 1).reshape(T, H * v_dim).to(bf)
    g = lin("g_proj", h).sigmoid()
    o = o * g
    out = lin("o_proj", o)
    save_f32(os.path.join(out_dir, f"attn{layer}.npy"), out)
    return out


def dense_mlp(sh, x, cfg):
    bf = torch.bfloat16
    beta, lbeta = cfg.get("activation_situ_beta") or 1.0, cfg.get("activation_situ_linear_beta")
    gu = torch.cat([F.linear(x, sh.get("mlp.gate_proj.weight").to(bf)), F.linear(x, sh.get("mlp.up_proj.weight").to(bf))], dim=-1)
    return F.linear(situ(gu, beta, lbeta), sh.get("mlp.down_proj.weight").to(bf))


def encode(model_dir, text, T):
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        ids = tok.encode(text)
        if len(ids) > T:
            ids = ids[:T]
        return ids, tok.decode(ids), "hf-tokenizer"
    except Exception as e:  # noqa: BLE001
        print(f"  tokenizer 不可用（{type(e).__name__}: {str(e)[:120]}），退回固定伪随机 id", flush=True)
        g = torch.Generator().manual_seed(0)
        return torch.randint(1000, 100000, (T,), generator=g).tolist(), None, "random"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog. 今天天气很好，我们去公园散步吧。Rust rewrites of vLLM should be faster because")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device("cuda")
    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))["text_config"]
    hidden, eps, bs = cfg["hidden_size"], cfg["rms_norm_eps"], cfg["attn_res_block_size"]
    kda_layers = set(cfg["linear_attn_config"]["kda_layers"])

    ids, text_back, how = encode(a.model_dir, a.text, a.T)
    T = len(ids)
    print(f"  tokens {T}（{how}）：{ids[:12]}…", flush=True)
    np.save(os.path.join(a.out_dir, "ids.npy"), np.asarray(ids, dtype=np.int32))
    ids_t = torch.tensor(ids, device=dev)

    t0 = time.time()
    top = TopShard(a.model_dir, dev)
    x = top.get(PREFIX + "embed_tokens.weight")[ids_t]  # bf16 [T, hidden]
    print(f"  顶层分片 {top.files}，embed {time.time()-t0:.1f}s", flush=True)
    save_f32(os.path.join(a.out_dir, "embed.npy"), x)

    blocks = []
    kinds = []
    experts_used = {}
    for l in range(a.layers):
        t0 = time.time()
        sh = Shard(a.model_dir, l, dev)
        is_kda = (l + 1) in kda_layers
        is_moe = l >= cfg["first_k_dense_replace"] and l % cfg.get("moe_layer_freq", 1) == 0
        kinds.append(("kda" if is_kda else "mla") + "+" + ("moe" if is_moe else "dense"))
        writer = l % bs == 0
        save_f32(os.path.join(a.out_dir, f"in{l}.npy"), x)  # 本层输入（隔离口径：Rust 每层喂 HF 的输入）
        h = apply_attn_res(x, blocks, sh.get("self_attention_res_proj.weight"), sh.get("self_attention_res_norm.weight"), eps)[0] if blocks else x
        if writer:
            blocks = blocks + [x]
        hn = rmsnorm(h, sh.get("input_layernorm.weight"), eps)
        att = kda(sh, hn, cfg, T, dev) if is_kda else mla(sh, hn, cfg, a.out_dir, l)
        p = att if writer else x + att
        m = apply_attn_res(p, blocks, sh.get("mlp_res_proj.weight"), sh.get("mlp_res_norm.weight"), eps)[0]
        mn = rmsnorm(m, sh.get("post_attention_layernorm.weight"), eps)
        if is_moe:
            layer_dir = os.path.join(a.out_dir, f"moe{l}")
            os.makedirs(layer_dir, exist_ok=True)
            y, used = moe(sh, mn, cfg, dev, layer_dir)
            experts_used[l] = used
        else:
            y = dense_mlp(sh, mn, cfg)
        x = p + y
        save_f32(os.path.join(a.out_dir, f"layer{l}.npy"), x)
        print(f"  layer {l} [{kinds[-1]}] {time.time()-t0:.1f}s  |x| max {x.float().abs().max():.3f} 末 token 范数 {x[-1].float().norm():.3f}", flush=True)
        del sh
        torch.cuda.empty_cache()

    res_out = apply_attn_res(x, blocks, top.get(PREFIX + "output_attn_res_proj.weight"), top.get(PREFIX + "output_attn_res_norm.weight"), eps)[0]
    save_f32(os.path.join(a.out_dir, "res_out.npy"), res_out)
    final = rmsnorm(res_out, top.get(PREFIX + "norm.weight"), eps)
    save_f32(os.path.join(a.out_dir, "final.npy"), final)
    logits = F.linear(final[-1:], top.get("language_model.lm_head.weight").to(torch.bfloat16)).float()[0]
    save_f32(os.path.join(a.out_dir, "logits_last.npy"), logits)
    top5 = torch.topk(logits, 5)
    print(f"  末 token top-5 id {top5.indices.tolist()} logit {[round(v, 3) for v in top5.values.tolist()]}", flush=True)
    meta = dict(layers=a.layers, T=T, ids=ids, text=a.text, text_back=text_back, tokenizer=how, kinds=kinds, n_blocks=len(blocks),
                experts_used=experts_used, hidden=hidden, torch=torch.__version__, top5=top5.indices.tolist())
    json.dump(meta, open(os.path.join(a.out_dir, "meta.json"), "w"), indent=1, ensure_ascii=False)
    print("saved", a.out_dir)


if __name__ == "__main__":
    main()
