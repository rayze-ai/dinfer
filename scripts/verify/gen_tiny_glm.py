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

print(f"wrote tiny GLM ({args.layers} layers) to {out}; greedy[:8] = {gen[0, prompt.shape[1]:prompt.shape[1]+8].tolist()}")
