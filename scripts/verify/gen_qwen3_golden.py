#!/usr/bin/env python3
"""Qwen3-0.6B 真权重 golden（同机 GPU bf16）：prefill logits、greedy 生成、逐步 logits、chat 模板 / 编码 golden。

用法：python3 -I scripts/verify/gen_qwen3_golden.py <out_dir> [--model /data/ref/models/qwen3-0.6b] [--gen 32]
产出：
  prompt_ids.npy / expect_prefill_logits.npy [T, V] / expect_greedy.npy [gen] / expect_step_logits.npy [gen, V]
  prompt2_ids.npy / expect_greedy2.npy（第二条 prompt，runner 并发对拍用）
  chat_golden.json（cases：text + ids；encode / incremental 样本）
HF 侧用 sdpa attention、bf16；对拍口径见 docs/loops/q1-qwen3.md。
"""
import argparse
import json
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--model", default="/data/ref/models/qwen3-0.6b")
ap.add_argument("--gen", type=int, default=32)
args = ap.parse_args()
from pathlib import Path

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
gen_cfg = json.load(open(f"{args.model}/generation_config.json"))


def run(messages, name, **kw):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    with torch.no_grad():
        logits = model(ids).logits[0].float().cpu().numpy()
        g = model.generate(ids, max_new_tokens=args.gen, do_sample=False, min_new_tokens=args.gen,
                           pad_token_id=gen_cfg["pad_token_id"], eos_token_id=None)
        gen = g[0, ids.shape[1]:]
        steps = []
        for i in range(args.gen):
            steps.append(model(g[:, : ids.shape[1] + i]).logits[0, -1].float().cpu())
    np.save(out / f"{name}_ids.npy", ids[0].cpu().numpy().astype(np.int32))
    np.save(out / f"expect_prefill_logits{'' if name == 'prompt' else '2'}.npy", logits)
    np.save(out / f"expect_greedy{'' if name == 'prompt' else '2'}.npy", gen.cpu().numpy().astype(np.int32))
    np.save(out / f"expect_step_logits{'' if name == 'prompt' else '2'}.npy", torch.stack(steps).numpy())
    print(f"{name}: {ids.shape[1]} prompt tokens, greedy: {tok.decode(gen)[:80]!r}", file=sys.stderr)
    return text


t1 = run([{"role": "user", "content": "用一句话介绍一下你自己。"}], "prompt")
t2 = run([{"role": "system", "content": "You are a terse assistant."},
          {"role": "user", "content": "1+1=?"}], "prompt2", enable_thinking=False)

# chat 模板 golden
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
cases = {
    "user": dict(messages=[{"role": "user", "content": "hi"}]),
    "no_think": dict(messages=[{"role": "user", "content": "hi"}], enable_thinking=False),
    "system_user": dict(messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "a"}]),
    "multi_think_in_content": dict(messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "a"},
                                             {"role": "assistant", "content": "<think>\nt\n</think>\n\nA"},
                                             {"role": "user", "content": "b"}]),
    "multi_reasoning_field": dict(messages=[{"role": "user", "content": "a"},
                                            {"role": "assistant", "content": "A", "reasoning_content": "t"},
                                            {"role": "user", "content": "b"}]),
    "tools": dict(messages=[{"role": "user", "content": "hi"}], tools=tools),
    "tools_roundtrip": dict(messages=[
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                                                              "function": {"name": "get_weather", "arguments": {"city": "Beijing"}}}]},
        {"role": "tool", "content": "{\"temp\": 21}"},
        {"role": "user", "content": "thanks"}], tools=tools),
    "content_parts": dict(messages=[{"role": "user", "content": [{"type": "text", "text": "p1 "}, {"type": "text", "text": "p2"}]}]),
    "no_gen_prompt": dict(messages=[{"role": "user", "content": "hi"}], add_generation_prompt=False),
}
golden = {"specials": {s: tok.convert_tokens_to_ids(s) for s in
                       ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>", "</tool_call>"]},
          "eos_token_ids": gen_cfg["eos_token_id"], "cases": {}}
for name, kw in cases.items():
    agp = kw.pop("add_generation_prompt", True)
    extra = {k: v for k, v in kw.items() if k not in ("messages", "tools")}
    text = tok.apply_chat_template(kw["messages"], tools=kw.get("tools"), tokenize=False, add_generation_prompt=agp, **extra)
    golden["cases"][name] = {"input": {**kw, "add_generation_prompt": agp}, "text": text,
                             "ids": tok(text, add_special_tokens=False).input_ids}
samples = ["Hello, world!", "你好，世界！", "  spaces\ttabs\n", "emoji 🚀 ünï", "<think>\nx\n</think>", "def f(x):\n    return x"]
golden["encode"] = [{"text": s, "ids": tok(s, add_special_tokens=False).input_ids} for s in samples]
ids = tok("The quick brown 狐狸 jumps 🐶 over ünïcode", add_special_tokens=False).input_ids
golden["incremental"] = {"ids": ids, "prefix_texts": [tok.decode(ids[: i + 1]) for i in range(len(ids))]}
json.dump(golden, open(out / "chat_golden.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"wrote {out}: {len(golden['cases'])} chat cases", file=sys.stderr)
