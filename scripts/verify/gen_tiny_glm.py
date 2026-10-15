#!/usr/bin/env python3
"""生成 P3 对拍用的 tiny GLM 结构模型（HF DeepseekV3 实现，MLA + dense MLP，与 GLM-5.3 同构）。

输出到 <out_dir>/：
  config.json              HF 配置（字段名与 GLM-5.3 config.json 一致的子集）
  model.safetensors        bf16 随机权重（HF 命名：model.layers.N.self_attn.q_a_proj.weight ...）
  prompt.npy               int32 [T] prompt token ids
  expect_prefill_logits.npy  float32 [T, V]  prefill 全序列 logits（bf16 模型算、转 fp32 存）
  expect_greedy.npy        int32 [32] 从 prompt 出发 greedy 生成的 32 个 token

用法：python3 scripts/verify/gen_tiny_glm.py <out_dir> [--layers 2] [--first-k-dense 1] [--fp8] [--dsa]

--dsa：改用 transformers ≥ 5.15 的 `GlmMoeDsaForCausalLM`（带 DSA 索引器与稀疏 attention；需用
  /data/ref/venv-tf519/bin/python 运行），额外输出：
  indexer_topk_L<i>.npy      int32 [T, topk] 第 i 层（full 层）prefill 时每个 query 的 top-k 索引（HF 原始输出，未排序）
  indexer_score_L<i>.npy     float32 [T, T] 第 i 层 prefill 的 index_scores（causal 外为 -inf），用于平局容忍判定
  prompt_long.npy / expect_prefill_logits_long.npy / expect_greedy_long.npy / expect_step_logits_long.npy
                             长 prompt（--long N，默认 256）版本，确保 top-k < 可见长度时稀疏真正生效
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?", default="test/tiny_glm")
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--gen", type=int, default=32)
ap.add_argument("--first-k-dense", type=int, default=None, help="默认 = layers（全 dense）；设 1 → 第 1 层起 MoE")
ap.add_argument("--fp8", action="store_true",
                help="把投影权重按 GLM-5.3 格式做 128×128 块 fp8(e4m3) 量化存盘（weight + weight_scale_inv），期望值用解量化后的权重算")
ap.add_argument("--dsa", action="store_true", help="用 GlmMoeDsaForCausalLM（DSA 索引器 + 稀疏 attention）")
ap.add_argument("--index-topk", type=int, default=8)
ap.add_argument("--long", type=int, default=256, help="--dsa 时额外生成的长 prompt 长度")
ap.add_argument("--vocab", type=int, default=1000, help="词表大小；server e2e 用 154880 以兼容真 GLM-5.3 tokenizer")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(args.seed)

common = dict(
    vocab_size=args.vocab,
    hidden_size=256,
    intermediate_size=512,
    moe_intermediate_size=128,
    num_hidden_layers=args.layers,
    num_attention_heads=4,
    num_key_value_heads=4,
    n_routed_experts=4,
    num_experts_per_tok=2,
    n_shared_experts=1,
    n_group=1,
    topk_group=1,
    norm_topk_prob=True,
    routed_scaling_factor=2.5,
    first_k_dense_replace=args.layers if args.first_k_dense is None else args.first_k_dense,
    q_lora_rank=64,
    kv_lora_rank=32,
    qk_nope_head_dim=32,
    qk_rope_head_dim=16,
    v_head_dim=32,
    rms_norm_eps=1e-5,
    max_position_embeddings=4096,
    attention_bias=False,
    tie_word_embeddings=False,
)
# DSA 索引器参数（--dsa 时生效；否则只写进 config.json 占位）：2 头 × 32、top-k 默认 8（远小于序列长，稀疏真正生效）、
# indexer_types 由 freq=2 / offset=1 推出：3 层 → full, shared, full（覆盖 "shared 复用上一 full" 路径）
IDX = dict(index_n_heads=4, index_head_dim=32, index_topk=args.index_topk, index_topk_freq=2, index_skip_topk_offset=1)
if args.dsa:
    from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
    cfg = GlmMoeDsaConfig(**common, **IDX, rope_parameters={"rope_type": "default", "rope_theta": 8_000_000.0})
    model = GlmMoeDsaForCausalLM(cfg).to(torch.bfloat16).eval()
    cfg.rope_interleave = True
    cfg.rope_theta = 8_000_000.0
else:
    cfg = DeepseekV3Config(
        **common,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        rope_interleave=True,
        rope_theta=8_000_000.0,
        rope_scaling=None,
        torch_dtype="bfloat16",
    )
    model = DeepseekV3ForCausalLM(cfg).to(torch.bfloat16).eval()
# 让权重幅度合理，避免 logits 退化成常数
with torch.no_grad():
    for n, p in model.named_parameters():
        if p.dim() >= 2:
            p.mul_(0.5 if "embed" in n or "lm_head" in n else 1.0)
        # DSA 索引器：HF 默认初始化下 q·k 太小、relu 后大量恰为 0 → top-k 在 HF 内部就是平局乱选，无法对拍。
        # 放大 wq_b / wk 使分数有分辨力（真模型训练后分数自然有分辨力）；weights_proj 保持。
        if args.dsa and ".indexer.wq_b" in n or args.dsa and ".indexer.wk." in n:
            p.mul_(6.0)
        # MoE 路由：随机 gate 下 sigmoid 分数挤在 [0.3, 0.8]，第 2/3 名专家分差 ≤ 0.01 的行约 9%（4 专家 top-2），
        # 两边 bf16 下会选不同专家 → 该行 logits 不可比，且该行作为后续层索引器的 key 也不可比（shared 层复用 L0 top-k，
        # 几乎每个 decode 步都会选中几个平局行）。放大 gate 让分差拉开（×8：margin ≤ 0.02 的行约 7%）。
        if args.dsa and n.endswith(".mlp.gate.weight"):
            p.mul_(8.0)

# e_score_correction_bias 随机化（HF 默认全 0 会掩盖 bias 逻辑错误）
with torch.no_grad():
    for n, b in model.named_buffers():
        if n.endswith("e_score_correction_bias"):
            b.copy_(torch.randn_like(b) * 0.1)
state = {k: v.contiguous() for k, v in model.state_dict().items()}

# transformers ≥ 5 把专家权重存成融合 3D 张量 `experts.gate_up_proj [E, 2I, H]` / `experts.down_proj [E, H, I]`；
# 真 GLM-5.3 safetensors 与 Rust 加载器用逐专家 `experts.{e}.{gate,up,down}_proj.weight`。这里拆回逐专家布局。
for name in [k for k in state if k.endswith(".experts.gate_up_proj")]:
    base = name[: -len(".gate_up_proj")]
    gu = state.pop(name)                       # [E, 2I, H]
    dn = state.pop(base + ".down_proj")        # [E, H, I]
    E, I2, _ = gu.shape
    for e in range(E):
        g, u = gu[e].chunk(2, dim=0)           # HF: linear(x, gate_up[e]).chunk(2) → gate, up
        state[f"{base}.{e}.gate_proj.weight"] = g.contiguous()
        state[f"{base}.{e}.up_proj.weight"] = u.contiguous()
        state[f"{base}.{e}.down_proj.weight"] = dn[e].contiguous()

# ---- fp8 块量化（可选）：与 GLM-5.3 真权重同格式 ----
# 量化白名单 = 所有 Linear 投影（attention 的 5 个 + MLP/专家的 3 个）；embed / lm_head / norm / 路由 gate 不量化。
FP8_BLOCK = (128, 128)
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448
QUANT_SUFFIXES = ("q_a_proj.weight", "kv_a_proj_with_mqa.weight", "q_b_proj.weight", "kv_b_proj.weight",
                  "o_proj.weight", "gate_proj.weight", "up_proj.weight", "down_proj.weight")


def quant_block_fp8(w: torch.Tensor):
    """w [N,K] → (w8 [N,K] e4m3, scale_inv [⌈N/bn⌉,⌈K/bk⌉] f32, w_deq bf16)。最后一块可不满。"""
    n, k = w.shape
    bn, bk = FP8_BLOCK
    sn, sk = -(-n // bn), -(-k // bk)
    w32 = w.float()
    pad = torch.zeros(sn * bn, sk * bk)
    pad[:n, :k] = w32
    blocks = pad.view(sn, bn, sk, bk).permute(0, 2, 1, 3)  # [sn, sk, bn, bk]
    amax = blocks.abs().amax(dim=(-1, -2)).clamp(min=1e-12)  # [sn, sk]
    scale_inv = (amax / FP8_MAX).float()  # 乘法因子（vLLM 命名 weight_scale_inv）
    q = (blocks / scale_inv[..., None, None]).to(torch.float8_e4m3fn)
    deq = q.float() * scale_inv[..., None, None]
    q = q.permute(0, 2, 1, 3).reshape(sn * bn, sk * bk)[:n, :k].contiguous()
    deq = deq.permute(0, 2, 1, 3).reshape(sn * bn, sk * bk)[:n, :k].contiguous()
    return q, scale_inv.contiguous(), deq.to(torch.bfloat16)


if args.fp8:
    n_q = 0
    with torch.no_grad():
        params = dict(model.named_parameters())
        for name in list(state):
            if name.endswith(QUANT_SUFFIXES) and state[name].dim() == 2:
                q, s, deq = quant_block_fp8(state[name])
                state[name] = q
                state[name[: -len(".weight")] + ".weight_scale_inv"] = s
                params[name].copy_(deq)  # HF 侧用解量化后的权重算期望
                n_q += 1
    print(f"fp8: quantized {n_q} projection weights (block {FP8_BLOCK})")

save_file(state, out / "model.safetensors", metadata={"format": "pt"})

# 导出 config.json：只保留 Rust 侧要读的字段，名字与 GLM-5.3 config.json 一致
keep = [
    "vocab_size", "hidden_size", "intermediate_size", "moe_intermediate_size", "num_hidden_layers",
    "num_attention_heads", "n_routed_experts", "num_experts_per_tok", "n_shared_experts",
    "first_k_dense_replace", "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
    "v_head_dim", "rope_interleave", "rope_theta", "rms_norm_eps", "max_position_embeddings",
]
c = {k: getattr(cfg, k) for k in keep}
c.update({
    "architectures": ["GlmMoeDsaForCausalLM"],
    "model_type": "glm_moe_dsa",
    "hidden_act": "silu",
    "eos_token_id": [999],
    "pad_token_id": 0,
    "moe_layer_freq": 1,
    "routed_scaling_factor": cfg.routed_scaling_factor,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "num_nextn_predict_layers": 0,
    "rope_parameters": {"rope_theta": cfg.rope_theta, "rope_type": "default"},
    # 索引器字段（--dsa 时真用；否则占位，Rust 侧不读 indexer 权重）
    **IDX, "index_topk_pattern": None, "indexer_rope_interleave": True,
    "indexer_types": list(cfg.indexer_types) if args.dsa else None,
    "dsa": bool(args.dsa),
})
if args.fp8:
    c["quantization_config"] = {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                                "weight_block_size": list(FP8_BLOCK)}
(out / "config.json").write_text(json.dumps(c, indent=2))

prompt = torch.tensor([[5, 17, 42, 99, 123, 7, 8, 500, 321, 64, 2, 77, 900, 13]], dtype=torch.long)
with torch.no_grad():
    logits = model(prompt).logits[0]  # [T, V] bf16
    np.save(out / "prompt.npy", prompt[0].to(torch.int32).numpy())
    np.save(out / "expect_prefill_logits.npy", logits.float().numpy())
    gen = model.generate(prompt, max_new_tokens=args.gen, do_sample=False, min_new_tokens=args.gen,
                         pad_token_id=0, eos_token_id=None)
    np.save(out / "expect_greedy.npy", gen[0, prompt.shape[1]:].to(torch.int32).numpy())
    # 每一步 decode 的完整 logits（用 full-sequence 前向取最后一行，避免 KV cache 实现差异）：
    # step i 的输入是 prompt + gen[:i]，取最后一个位置的 logits。
    step_logits = []
    for i in range(args.gen):
        seq = gen[:, : prompt.shape[1] + i]
        step_logits.append(model(seq).logits[0, -1].float())
    np.save(out / "expect_step_logits.npy", torch.stack(step_logits).numpy())  # [gen, V]

    # 第二条 prompt（长度 11，不对齐 block=16）：runner 多请求并发对拍用。
    prompt2 = torch.tensor([[300, 1, 44, 888, 9, 620, 71, 15, 402, 33, 256]], dtype=torch.long)
    gen2 = model.generate(prompt2, max_new_tokens=args.gen, do_sample=False, min_new_tokens=args.gen,
                          pad_token_id=0, eos_token_id=None)
    np.save(out / "prompt2.npy", prompt2[0].to(torch.int32).numpy())
    np.save(out / "expect_greedy2.npy", gen2[0, prompt2.shape[1]:].to(torch.int32).numpy())
    step_logits2 = []
    for i in range(args.gen):
        seq = gen2[:, : prompt2.shape[1] + i]
        step_logits2.append(model(seq).logits[0, -1].float())
    np.save(out / "expect_step_logits2.npy", torch.stack(step_logits2).numpy())


def dump_router_margin(tag: str, p: torch.Tensor):
    """每行 MoE 路由 "第 2 名 vs 第 3 名专家" 的分差（取所有 MoE 层的最小值）→ router_margin{tag}.npy [T] f32。
    bf16 下该差 < ~0.01 的行，两边可能选不同专家，之后的 logits 不可比（与索引器平局同理）。"""
    cap = {}
    hooks = [
        L.mlp.gate.register_forward_hook(lambda mod, a, o, li=li: cap.__setitem__(li, (o[0].float().clone(), mod.e_score_correction_bias.float().clone())))
        for li, L in enumerate(model.model.layers) if hasattr(L.mlp, "gate")
    ]
    with torch.no_grad():
        model(p)
    for h in hooks:
        h.remove()
    margin = None
    for li, (logits, bias) in cap.items():
        sc = torch.sigmoid(logits) + bias
        srt, _ = sc.sort(-1, descending=True)
        m = (srt[:, cfg.num_experts_per_tok - 1] - srt[:, cfg.num_experts_per_tok]).numpy()
        margin = m if margin is None else np.minimum(margin, m)
    np.save(out / f"router_margin{tag}.npy", margin.astype(np.float32))
    return float(np.median(margin)), int((margin < 0.01).sum())


def dump_hidden(tag: str, p: torch.Tensor):
    """每个 decoder 层的输出 hidden（残差流，[T, hidden] f32）→ hidden_L<i><tag>.npy。
    对拍用：两边逐层比 hidden，找出 **实际漂移** 的行（MoE 路由平局翻转导致），只排除这些行而不是所有 near-tie 行。"""
    caps, hooks = {}, []
    for li, L in enumerate(model.model.layers):
        hooks.append(L.register_forward_hook(
            lambda mod, a, o, li=li: caps.__setitem__(li, (o[0] if isinstance(o, tuple) else o)[0].float().cpu().numpy())))
    with torch.no_grad():
        model(p)
    for h in hooks:
        h.remove()
    for li, v in caps.items():
        np.save(out / f"hidden_L{li}{tag}.npy", v)
    return sorted(caps)


def dump_dsa(tag: str, p: torch.Tensor):
    """抓每个 full 层索引器的 top-k（HF 原始输出）与 index_scores（重算，口径同 GlmMoeDsaIndexer.forward）。"""
    caps = {}
    hooks = []
    for li, L in enumerate(model.model.layers):
        idx = L.self_attn.indexer
        if idx is None:
            continue

        def mk(li, idx):
            def pre(mod, args, kwargs):
                hidden_states, q_resid, (cos, sin), attention_mask = args[0], args[1], args[2], args[3]
                B, S, _ = hidden_states.shape
                from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import apply_rotary_pos_emb_interleave
                q = idx.wq_b(q_resid).view(B, S, idx.n_heads, idx.head_dim)
                q_rot, q_pass = torch.split(q, [idx.qk_rope_head_dim, idx.head_dim - idx.qk_rope_head_dim], dim=-1)
                k = idx.k_norm(idx.wk(hidden_states)).unsqueeze(2)
                k_rot, k_pass = torch.split(k, [idx.qk_rope_head_dim, idx.head_dim - idx.qk_rope_head_dim], dim=-1)
                q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
                q = torch.cat([q_rot, q_pass], dim=-1)
                k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)
                scores = torch.relu(torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1)) * idx.softmax_scale)
                wts = idx.weights_proj(hidden_states.to(idx.weights_proj.weight.dtype)).float() * (idx.n_heads ** -0.5)
                sc = torch.matmul(wts.unsqueeze(-2), scores).squeeze(-2)  # [B, S, T]
                mask = attention_mask
                sc = sc.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else sc + mask
                caps[f"score_L{li}"] = sc[0].float().cpu().numpy()

            def post(mod, args, out):
                caps[f"topk_L{li}"] = out[0].to(torch.int32).cpu().numpy()
            return pre, post

        pre, post = mk(li, idx)
        hooks.append(idx.register_forward_pre_hook(pre, with_kwargs=True))
        hooks.append(idx.register_forward_hook(post))
    with torch.no_grad():
        model(p)
    for h in hooks:
        h.remove()
    for k, v in caps.items():
        kind, L = k.split("_")
        np.save(out / f"indexer_{kind}_{L}{tag}.npy", v)
    return sorted(caps)


if args.dsa:
    print("indexer_types =", cfg.indexer_types, "dumped:", dump_dsa("", prompt))
    print("hidden layers (short):", dump_hidden("", prompt))
    print("router margin (short):", dump_router_margin("", prompt), " (short2):", dump_router_margin("2", prompt2))
    # 长 prompt：确定性伪随机 token（种子固定），长度 args.long；期望同短 prompt 三件套
    g = torch.Generator().manual_seed(12345)
    prompt_long = torch.randint(1, 1000, (1, args.long), generator=g, dtype=torch.long)
    with torch.no_grad():
        np.save(out / "prompt_long.npy", prompt_long[0].to(torch.int32).numpy())
        np.save(out / "expect_prefill_logits_long.npy", model(prompt_long).logits[0].float().numpy())
        gen_l = model.generate(prompt_long, max_new_tokens=args.gen, do_sample=False, min_new_tokens=args.gen,
                               pad_token_id=0, eos_token_id=None)
        np.save(out / "expect_greedy_long.npy", gen_l[0, args.long:].to(torch.int32).numpy())
        step_l = []
        for i in range(args.gen):
            step_l.append(model(gen_l[:, : args.long + i]).logits[0, -1].float())
        np.save(out / "expect_step_logits_long.npy", torch.stack(step_l).numpy())
    print("dumped long:", dump_dsa("_long", prompt_long))
    print("hidden layers (long):", dump_hidden("_long", prompt_long))
    print("router margin (long):", dump_router_margin("_long", prompt_long))
    # decode 步的路由 margin：输入 prompt_long + gen[:i] 的最后一行
    rm_steps = []
    for i in range(args.gen):
        dump_router_margin("_tmp", gen_l[:, : args.long + i])
        rm_steps.append(np.load(out / "router_margin_tmp.npy")[-1])
    (out / "router_margin_tmp.npy").unlink()
    np.save(out / "router_margin_long_steps.npy", np.array(rm_steps, dtype=np.float32))
    # 每个 decode 步（输入 prompt_long + gen[:i]，全序列前向）最后一行的 top-k / index_scores，供 Rust decode 路径逐步对拍
    steps_tk, steps_sc = {}, {}
    Lmax = args.long + args.gen
    for i in range(args.gen):
        seq = gen_l[:, : args.long + i]
        names = dump_dsa(f"_step{i}", seq)
        for n in names:
            kind, L = n.split("_")
            arr = np.load(out / f"indexer_{kind}_{L}_step{i}.npy")
            (out / f"indexer_{kind}_{L}_step{i}.npy").unlink()
            row = arr[-1]
            if kind == "score":
                pad = np.full(Lmax, -np.inf, dtype=np.float32)
                pad[: row.shape[0]] = row
                row = pad
            (steps_sc if kind == "score" else steps_tk).setdefault(L, []).append(row)
    for L in steps_tk:
        np.save(out / f"indexer_topk_{L}_steps.npy", np.stack(steps_tk[L]).astype(np.int32))
        np.save(out / f"indexer_score_{L}_steps.npy", np.stack(steps_sc[L]).astype(np.float32))
    print("dumped decode steps:", sorted(steps_tk), f"[{args.gen}, K] / [{args.gen}, {Lmax}]")
    # 每个 decode 步每层输出 hidden 的最后一行 → hidden_L<i>_long_steps.npy [gen, hidden]（decode 路径的漂移检测）
    # 同时记录每步前向里 **前缀行** 相对 prefill-only 前向的 hidden 变化（HF 对整条序列重算，MoE 专家分发在 bf16 下
    # 不是 batch 稳定的，个别路由平局行会翻转 → 该行的 KV 在 HF 里与我们 cache 里的不同，选中它的步不可比）
    # → hidden_L<i>_long_prefix_drift_steps.npy [gen, T] f32 = 每行 max abs diff
    steps_h, prefix_drift = {}, {}
    base_h = {li: np.load(out / f"hidden_L{li}_long.npy") for li in range(args.layers)}
    for i in range(args.gen):
        names = dump_hidden(f"_tmp", gen_l[:, : args.long + i])
        for li in names:
            arr = np.load(out / f"hidden_L{li}_tmp.npy")
            (out / f"hidden_L{li}_tmp.npy").unlink()
            steps_h.setdefault(li, []).append(arr[-1])
            prefix_drift.setdefault(li, []).append(np.abs(arr[: args.long] - base_h[li]).max(axis=1))
    for li, rows in steps_h.items():
        np.save(out / f"hidden_L{li}_long_steps.npy", np.stack(rows).astype(np.float32))
        np.save(out / f"hidden_L{li}_long_prefix_drift_steps.npy", np.stack(prefix_drift[li]).astype(np.float32))
    n_drift = int(sum((np.stack(prefix_drift[li]) > 0.02).any(axis=0).sum() for li in prefix_drift))
    print("dumped decode hidden:", sorted(steps_h), f"[{args.gen}, hidden]; prefix rows that drift in some step (any layer): {n_drift}")

print(f"wrote tiny GLM ({args.layers} layers{', DSA' if args.dsa else ''}) to {out}; greedy[:8] = {gen[0, prompt.shape[1]:prompt.shape[1]+8].tolist()}")
