#!/usr/bin/env python3
"""生成 P3 对拍用的 tiny GLM 结构模型（HF DeepseekV3 实现，MLA + dense MLP，与 GLM-5.3 同构）。

输出到 <out_dir>/：
  config.json              HF 配置（字段名与 GLM-5.3 config.json 一致的子集）
  model.safetensors        bf16 随机权重（HF 命名：model.layers.N.self_attn.q_a_proj.weight ...）
  prompt.npy               int32 [T] prompt token ids
  expect_prefill_logits.npy  float32 [T, V]  prefill 全序列 logits（bf16 模型算、转 fp32 存）
  expect_greedy.npy        int32 [32] 从 prompt 出发 greedy 生成的 32 个 token

用法：python3 scripts/verify/gen_tiny_glm.py <out_dir> [--layers 2]
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
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(args.seed)

cfg = DeepseekV3Config(
    vocab_size=1000,
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
    scoring_func="sigmoid",
    topk_method="noaux_tc",
    first_k_dense_replace=args.layers if args.first_k_dense is None else args.first_k_dense,
    q_lora_rank=64,
    kv_lora_rank=32,
    qk_nope_head_dim=32,
    qk_rope_head_dim=16,
    v_head_dim=32,
    rope_interleave=True,
    rope_theta=8_000_000.0,
    rope_scaling=None,
    rms_norm_eps=1e-5,
    max_position_embeddings=4096,
    attention_bias=False,
    tie_word_embeddings=False,
    torch_dtype="bfloat16",
)
model = DeepseekV3ForCausalLM(cfg).to(torch.bfloat16).eval()
# 让权重幅度合理，避免 logits 退化成常数
with torch.no_grad():
    for n, p in model.named_parameters():
        if p.dim() >= 2:
            p.mul_(0.5 if "embed" in n or "lm_head" in n else 1.0)

# e_score_correction_bias 随机化（HF 默认全 0 会掩盖 bias 逻辑错误）
with torch.no_grad():
    for n, b in model.named_buffers():
        if n.endswith("e_score_correction_bias"):
            b.copy_(torch.randn_like(b) * 0.1)
state = {k: v.contiguous() for k, v in model.state_dict().items()}

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
    # 索引器字段给占位值（P3 不用）
    "index_n_heads": 2, "index_head_dim": 16, "index_topk": 64, "index_topk_freq": 1,
    "index_skip_topk_offset": 0, "index_topk_pattern": None, "indexer_rope_interleave": True,
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

print(f"wrote tiny GLM ({args.layers} layers) to {out}; greedy[:8] = {gen[0, prompt.shape[1]:prompt.shape[1]+8].tolist()}")
