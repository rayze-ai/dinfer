#!/usr/bin/env python3
"""用 transformers 的 apply_chat_template + tokenizer 生成 GLM-5.3 chat 模板 / 编解码 golden，供 Rust 对拍。

用法：python3 -I scripts/verify/gen_chat_golden.py <out.json> [--tokenizer /data/ref/models/glm53]
产出：
  specials        关键特殊 token → id
  eos_token_ids   generation_config 的 eos 集合
  cases{name}     {input:{messages,tools,reasoning_effort,clear_thinking,add_generation_prompt}, text, ids}
  encode[]        {text, ids}（add_special_tokens=False）
  incremental     {ids, prefix_texts}：逐前缀 decode，验证增量 detokenize
"""
import argparse
import json
import sys

from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--tokenizer", default="/data/ref/models/glm53")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
tmpl = open(f"{args.tokenizer}/chat_template.jinja", encoding="utf-8").read()
gen_cfg = json.load(open(f"{args.tokenizer}/generation_config.json"))

special_names = ["[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>", "<|endoftext|>",
                 "<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"]
specials = {t: tok.convert_tokens_to_ids(t) for t in special_names}

tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                         "required": ["city"]}}}]
cases = {
    "single_user": dict(messages=[{"role": "user", "content": "你好，介绍一下你自己。"}]),
    "system_user": dict(messages=[{"role": "system", "content": "You are a terse assistant."},
                                  {"role": "user", "content": "What is 2+2?"}]),
    "effort_low": dict(messages=[{"role": "user", "content": "hi"}], reasoning_effort="low"),
    "effort_high": dict(messages=[{"role": "user", "content": "hi"}], reasoning_effort="high"),
    "effort_invalid_falls_back_max": dict(messages=[{"role": "user", "content": "hi"}], reasoning_effort="medium"),
    "multiturn_reasoning": dict(messages=[{"role": "user", "content": "a"},
                                          {"role": "assistant", "content": "A1", "reasoning_content": "thinking about a"},
                                          {"role": "user", "content": "b"}]),
    "multiturn_clear_thinking": dict(messages=[{"role": "user", "content": "a"},
                                               {"role": "assistant", "content": "A1", "reasoning_content": "thinking about a"},
                                               {"role": "user", "content": "b"}], clear_thinking=True),
    "multiturn_think_in_content": dict(messages=[{"role": "user", "content": "a"},
                                                 {"role": "assistant", "content": "<think>inner</think>A1"},
                                                 {"role": "user", "content": "b"}]),
    "content_parts": dict(messages=[{"role": "user", "content": [{"type": "text", "text": "part one "},
                                                                  {"type": "text", "text": "part two"},
                                                                  {"type": "image_url", "image_url": {"url": "x"}}]}]),
    "tools_user": dict(messages=[{"role": "user", "content": "Weather in Beijing for 3 days?"}], tools=tools),
    "tools_roundtrip": dict(messages=[
        {"role": "user", "content": "Weather in Beijing?"},
        {"role": "assistant", "content": "", "reasoning_content": "need tool",
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "get_weather", "arguments": {"city": "Beijing", "days": 3}}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 21}"},
        {"role": "user", "content": "thanks"}], tools=tools),
    "tools_two_results_sorted": dict(messages=[
        {"role": "user", "content": "Weather in A and B?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "A"}}},
                        {"id": "c2", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "B"}}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "B:10"},
        {"role": "tool", "tool_call_id": "c1", "content": "A:20"}], tools=tools),
    "tools_results_no_ids": dict(messages=[
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "A"}}}]},
        {"role": "tool", "content": "A:20"}], tools=tools),
    "tools_string_arguments": dict(messages=[
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "get_weather", "arguments": {"city": "A", "days": "3"}}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"}], tools=tools),
    "no_gen_prompt": dict(messages=[{"role": "user", "content": "hi"}], add_generation_prompt=False),
}

out = {"specials": specials, "eos_token_ids": gen_cfg["eos_token_id"], "cases": {}}
for name, kw in cases.items():
    agp = kw.pop("add_generation_prompt", True)
    extra = {k: v for k, v in kw.items() if k not in ("messages", "tools")}
    text = tok.apply_chat_template(kw["messages"], tools=kw.get("tools"), chat_template=tmpl, tokenize=False,
                                   add_generation_prompt=agp, **extra)
    ids = tok.encode(text, add_special_tokens=False)
    out["cases"][name] = {"input": {**kw, "add_generation_prompt": agp}, "text": text, "ids": ids}

samples = ["Hello, world!", "你好，世界！", "  leading spaces and\ttabs\n", "emoji 🚀 and ünïcödé",
           "<think>typed by user</think>", "def f(x):\n    return x*2", "[gMASK]<sop><|user|>raw specials"]
out["encode"] = [{"text": s, "ids": tok.encode(s, add_special_tokens=False)} for s in samples]
ids = tok.encode("The quick brown fox 跳过 lazy dog 🐶. 多字节：汉字、emoji 🚀🚀, done", add_special_tokens=False)
out["incremental"] = {"ids": ids, "prefix_texts": [tok.decode(ids[: i + 1]) for i in range(len(ids))]}

json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"wrote {args.out}: {len(out['cases'])} chat cases, {len(out['encode'])} encode samples, "
      f"{len(ids)} incremental ids", file=sys.stderr)
