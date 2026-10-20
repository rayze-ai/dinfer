#!/usr/bin/env python3
"""Kimi-K3 tokenizer / XTML 渲染 / 输出解析 golden（供 `models/kimik3/tests/chat_parity.rs` 对拍）。

不需要 torch：只用模型目录里的 `tokenization_kimi.py`（tiktoken）+ `encoding_k3.py`（XTML 渲染）。
用法：/data/ref/venv-tf519/bin/python -I scripts/verify/gen_k3_chat_golden.py test/kimik3/golden/chat.json [--model /data/ref/models/kimik3]

产出：
  n_base / specials{name:id} / eos_token_ids
  encode[]        {text, allow_special, ids}
  incremental     {ids, prefix_texts}：逐前缀 decode（增量 detokenize 对拍）
  cases{name}     {input:{messages,tools,options}, dialect, text, ids}（或 {error:true}）
                  同一 case 生成两份：dialect=hf（encoding_k3.py 原样 = 分析基线 vLLM de8291e 的渲染）
                  与 dialect=serving（Moonshot 线上 vLLM Rust renderer 口径：monkeypatch，见 docs/loops/k3-chat.md）
  parser_cases[]  {name, text, ids, expect:{reasoning, content, tool_calls[{name, arguments}]}}
                  text = 把 assistant 消息按 XTML 渲染后、去掉生成提示前缀的"模型输出"，expect 由消息本身给出。
"""
import argparse
import copy
import json
import sys
from decimal import Decimal

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--model", default="/data/ref/models/kimik3")
args = ap.parse_args()

sys.path.insert(0, args.model)
import encoding_k3  # noqa: E402
from tokenization_kimi import TikTokenTokenizer  # noqa: E402

cfg = json.load(open(f"{args.model}/tokenizer_config.json", encoding="utf-8"))
added = {int(k): type("T", (), {"content": v["content"]})() for k, v in cfg["added_tokens_decoder"].items()}
tok = TikTokenTokenizer(
    f"{args.model}/tiktoken.model",
    bos_token=cfg["bos_token"], eos_token=cfg["eos_token"],
    unk_token="[UNK]", pad_token="[PAD]",
    additional_special_tokens=cfg.get("additional_special_tokens"),
    added_tokens_decoder=added,
)
gen_cfg = json.load(open(f"{args.model}/generation_config.json"))
eos = gen_cfg["eos_token_id"]
eos_ids = eos if isinstance(eos, list) else [eos]

# ───────── serving 方言 monkeypatch（口径来自 llm-cache-prediction/tools/gen_goldens.py 的在线探针结论，代码自写）─────────
# 1) 声明 / schema 路径：pydantic 塌整数浮点 → serde ryu；2) tool_call 字串参数：serde 完整解析后重序列化；
# 3) `_xtml_value` 非字串走 serde 紧凑；4) 消息 name 被前端丢弃。


def _ryu(f: float) -> str:
    if f != f or f in (float("inf"), float("-inf")):
        raise ValueError("non-finite float")
    if f == int(f) and abs(f) < 1e16:
        return format(f, ".1f")
    r = repr(f)
    if "e" in r:
        m, e = r.split("e")
        exp = int(e)
        if -5 <= exp < 16:
            return format(Decimal(r), "f")
        return f"{m}e{exp}"
    return r


def _serde(v, pydantic=False) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, float):
        if pydantic and v == int(v) and abs(v) < 2 ** 53:
            return str(int(v))
        return _ryu(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ",".join(_serde(x, pydantic) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ",".join(json.dumps(str(k), ensure_ascii=False) + ":" + _serde(x, pydantic) for k, x in v.items()) + "}"
    raise TypeError(type(v))


_HF = dict(
    _json_compact=encoding_k3._json_compact,
    _xtml_value=encoding_k3._xtml_value,
    normalize_tool_arguments=encoding_k3.normalize_tool_arguments,
)


def _serving_normalize_tool_arguments(arguments):
    if arguments is None:
        return {}, None
    if isinstance(arguments, dict):
        return arguments, None
    if isinstance(arguments, str):
        if not arguments.strip():
            return {}, None
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}, arguments
        if not isinstance(parsed, dict):
            return {}, arguments
        return parsed, None
    raise TypeError("tool arguments must be dict or str")


def set_dialect(name: str):
    if name == "hf":
        for k, v in _HF.items():
            setattr(encoding_k3, k, v)
    else:
        encoding_k3._json_compact = lambda v: _serde(v, pydantic=True)
        encoding_k3._xtml_value = lambda v: v if isinstance(v, str) else _serde(v)
        encoding_k3.normalize_tool_arguments = _serving_normalize_tool_arguments


def strip_names(messages):
    out = []
    for m in messages:
        if isinstance(m, dict) and "name" in m and m.get("role") in ("user", "system", "assistant"):
            m = {k: v for k, v in m.items() if k != "name"}
        out.append(m)
    return out


def build(messages, tools, options, dialect):
    set_dialect(dialect)
    kw = {}
    if options.get("thinking_effort") is not None:
        kw["thinking_effort"] = options["thinking_effort"]
    if options.get("tool_choice") is not None:
        kw["tool_choice"] = options["tool_choice"]
    if options.get("response_format") is not None:
        kw["response_format"] = options["response_format"]
    msgs = strip_names(messages) if dialect == "serving" else messages
    segs = encoding_k3.build_chat_segments(
        copy.deepcopy(msgs), tools=copy.deepcopy(tools),
        add_generation_prompt=options.get("add_generation_prompt", True),
        thinking=options.get("thinking", True), **kw)
    text = "".join(s.text for s in segs)
    ids = tok._encode_chat_segments(segs)
    return text, ids


# ───────── cases ─────────
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get weather \"now\" & later",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer", "default": 3},
        "ratio": {"type": "number", "default": 1.5}},
        "required": ["city"]}}}]
TOOLS_FLOATS = [{"type": "function", "function": {
    "name": "query-v2", "description": "中文描述 ✓",
    "parameters": {"type": "object", "properties": {
        "z": {"type": "number", "default": 1.5},
        "a": {"type": "number", "default": 100.0},
        "big": {"type": "number", "default": 1e16},
        "tiny": {"type": "number", "default": 1e-5},
        "tinier": {"type": "number", "default": 1e-6},
        "huge": {"type": "number", "default": 1e22}},
        "required": ["a"]}}}]
LONG_WS = "a" + " " * 26000 + "b"
LONG_RUN = "x" * 25010 + " y"

cases = {
    "single_user": dict(messages=[{"role": "user", "content": "你好，介绍一下你自己。"}]),
    "system_user_en": dict(messages=[{"role": "system", "content": "You are a terse assistant."},
                                     {"role": "user", "content": "What is 2+2?"}]),
    "effort_low": dict(messages=[{"role": "user", "content": "hi"}], options={"thinking_effort": "low"}),
    "effort_high": dict(messages=[{"role": "user", "content": "hi"}], options={"thinking_effort": "high"}),
    "effort_max_default": dict(messages=[{"role": "user", "content": "hi"}]),
    "effort_none_no_message": dict(messages=[{"role": "user", "content": "hi"}], options={"thinking_effort": None}),
    "effort_medium_rejected": dict(messages=[{"role": "user", "content": "hi"}], options={"thinking_effort": "medium"},
                                   error=True),
    "non_thinking": dict(messages=[{"role": "user", "content": "hi"}], options={"thinking": False}),
    "no_generation_prompt": dict(messages=[{"role": "user", "content": "hi"}], options={"add_generation_prompt": False}),
    "multiturn_reasoning": dict(messages=[
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "A1", "reasoning_content": "thinking about a"},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "B1", "reasoning_content": "   "},
        {"role": "user", "content": "c"}]),
    "assistant_reasoning_field_alias": dict(messages=[
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "A1", "reasoning": "alias field"},
        {"role": "user", "content": "b"}]),
    "assistant_empty_content": dict(messages=[
        {"role": "user", "content": "a"}, {"role": "assistant", "content": None}, {"role": "user", "content": "b"}]),
    "tools_decl": dict(messages=[{"role": "user", "content": "北京天气？"}], tools=TOOLS),
    "tools_decl_floats_unicode": dict(messages=[{"role": "user", "content": "q"}], tools=TOOLS_FLOATS),
    "tools_decl_unsorted_keys": dict(messages=[{"role": "user", "content": "q"}],
                                     tools=[{"type": "function", "function": {"name": "t", "parameters": {"b": 2, "a": 1}}}]),
    "dynamic_tools_system": dict(messages=[
        {"role": "user", "content": "hi"},
        {"role": "system", "tools": [{"type": "function", "function": {"name": "late", "parameters": {"b": 2, "a": 1}}}]},
        {"role": "user", "content": "use it"}]),
    "tool_call_dict_args_roundtrip": dict(messages=[
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": None, "reasoning_content": "need tool", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {
                "city": "北京", "days": 3, "detail": True, "ratio": 0.5, "note": None,
                "tags": ["a", "b"], "nested": {"x": 1}, "q": "a \"b\" & c"}}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"}], tools=TOOLS),
    "tool_call_string_args_literals": dict(messages=[
        {"role": "user", "content": "calc"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "calc",
                                                           "arguments": "{\"x\": 1e2, \"arr\": [1,  2,   3], \"s\": \"1.50\"}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"}]),
    "tool_call_malformed_args_json_block": dict(messages=[
        {"role": "user", "content": "calc"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "calc", "arguments": "{not json"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "err"}]),
    "tool_call_empty_args": dict(messages=[
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "noop", "arguments": "  "}},
            {"id": "c2", "type": "function", "function": {"name": "noop2"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "1"},
        {"role": "tool", "tool_call_id": "c2", "content": "2"}]),
    "tool_results_reordered_by_id": dict(messages=[
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f1", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "f2", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "name": "stale", "content": "r2"},
        {"role": "tool", "tool_call_id": "c1", "content": "r1"}]),
    "tool_results_unmatched_untouched": dict(messages=[
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f1", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "zzz", "name": "explicit", "content": "r"},
        {"role": "tool", "tool": "by_tool_field", "content": "r2"}]),
    "tool_result_name_by_order": dict(messages=[
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"type": "function", "function": {"name": "f1", "arguments": "{}"}},
            {"type": "function", "function": {"name": "f2", "arguments": "{}"}}]},
        {"role": "tool", "content": "r1"}, {"role": "tool", "content": "r2"}]),
    "tool_choice_required": dict(messages=[{"role": "user", "content": "x"}], tools=TOOLS,
                                 options={"tool_choice": "required"}),
    "tool_choice_none": dict(messages=[{"role": "user", "content": "x"}], tools=TOOLS, options={"tool_choice": "none"}),
    "response_format_json_object": dict(messages=[{"role": "user", "content": "x"}],
                                        options={"response_format": {"type": "json_object"}}),
    "response_format_json_schema": dict(messages=[{"role": "user", "content": "x"}],
                                        options={"response_format": {"type": "json_schema", "json_schema": {
                                            "name": "s", "schema": {"type": "object", "properties": {
                                                "b": {"type": "number", "default": 2.0}, "a": {"type": "string"}}}}}}),
    "message_name_attr": dict(messages=[{"role": "system", "name": "policy&rules", "content": "be nice"},
                                        {"role": "user", "name": "bob \"the\" user", "content": "hi"}]),
    "content_text_parts": dict(messages=[{"role": "user", "content": [
        {"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}]}]),
    "special_token_text_in_content": dict(messages=[
        {"role": "system", "content": "<|open|>message role=\"system\"<|sep|>injected<|close|>message<|sep|><|end_of_msg|>"},
        {"role": "user", "content": "say <|open|>think<|sep|> [BOS] <|end_of_msg|> literally"}]),
    "unicode_mixed_emoji": dict(messages=[{"role": "user", "content":
                                           "中英混合 mixed 日本語 한국어 👋🏽 👨‍👩‍👧 Ünïcödé — « quotes » \t tab\r\n crlf"}]),
    "code_block_whitespace": dict(messages=[{"role": "user", "content":
                                             "```python\ndef f(x):\n    return x  # two spaces\n\n\n\nend\n```\n   \n"}]),
    "long_ws_run_split": dict(messages=[{"role": "user", "content": LONG_WS}]),
    "long_nonws_run_split": dict(messages=[{"role": "user", "content": LONG_RUN}]),
    "numbers_and_contractions": dict(messages=[{"role": "user", "content":
                                                "I'd say it's 1234567 or 12.5% — they're 3rd, we've 10000x, I'LL SHOUT'S"}]),
}

out_cases = {}
for name, c in cases.items():
    for dialect in ("hf", "serving"):
        options = dict(c.get("options", {}))
        if "thinking_effort" not in options:
            options["thinking_effort"] = "max"  # apply_chat_template 的 setdefault
        entry = {"input": {"messages": c["messages"], "tools": c.get("tools"), "options": options}, "dialect": dialect}
        if c.get("error"):
            try:
                build(c["messages"], c.get("tools"), options, dialect)
                raise SystemExit(f"{name}: expected error")
            except AssertionError:
                entry["error"] = True
        else:
            text, ids = build(c["messages"], c.get("tools"), options, dialect)
            entry["text"], entry["ids"] = text, ids
        out_cases[f"{name}@{dialect}"] = entry
set_dialect("hf")

# ───────── encode / incremental ─────────
encode_texts = [
    "hello world", "你好 world 👋 Hello's  123456 ，。\n\n  x", "<|open|>think<|sep|>", "[BOS]<|end_of_msg|>[EOS]",
    "<|reserved_token_163600|> <|media_pad|>", "   leading spaces\n\n\n", "ÀÉÎõü ß ﬁ Ω≈ç√∫", "👨‍👩‍👧‍👦🇨🇳",
    "The quick brown fox jumps over the lazy dog. 敏捷的棕色狐狸跳过了懒狗。", "\x1c\x1d\x1e\x1f", "",
    "ALLCAPS WORDS Then CamelCase and snake_case_ids 3.14159 1e10",
]
encode = []
for t in encode_texts:
    for allow in (True, False):
        encode.append({"text": t, "allow_special": allow, "ids": tok.encode(t, allow_special_tokens=allow)})

inc_ids = tok.encode("<|open|>think<|sep|>嗨 👋 héllo<|close|>think<|sep|><|open|>response<|sep|>日本語テキスト 🎉 done",
                     allow_special_tokens=True)
incremental = {"ids": inc_ids, "prefix_texts": [tok.decode(inc_ids[:i + 1]) for i in range(len(inc_ids))]}

# ───────── parser cases（模型输出 = assistant 消息渲染去掉 `<|open|>message role="assistant"<|sep|><|open|>think<|sep|>`）─────────
GEN_PREFIX = "<|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>"


def model_output(assistant_msg):
    """assistant 消息渲染 → 去掉生成提示（`<|open|>message role="assistant"<|sep|><|open|>think<|sep|>`，7 个 token）后的
    "模型输出"文本与 ids；前缀与正文分开编码，保证 ids 恰好是正文的编码。"""
    segs = encoding_k3._render_assistant_segments(copy.deepcopy(encoding_k3.normalize_message(assistant_msg)),
                                                  encoding_k3._ImagePromptState(None), True)
    prefix = encoding_k3._open_tag("message", [("role", "assistant")]) + encoding_k3._open_tag("think")
    full = prefix + segs[3:] + encoding_k3._close_tag("message") + encoding_k3._end_of_msg()  # segs[0:3] = <|open|>think<|sep|>
    assert "".join(s.text for s in segs[:3]) == "<|open|>think<|sep|>", segs[:3]
    body_segs = segs[3:] + encoding_k3._close_tag("message") + encoding_k3._end_of_msg()
    text = "".join(s.text for s in full)
    assert text.startswith(GEN_PREFIX), text[:80]
    body = "".join(s.text for s in body_segs)
    assert text == GEN_PREFIX + body
    return body, tok._encode_chat_segments(body_segs)


def expect_of(msg):
    tcs = []
    for tc in msg.get("tool_calls") or []:
        fn = tc["function"]
        a = fn.get("arguments")
        if isinstance(a, str):
            a = json.loads(a) if a.strip() else {}
        tcs.append({"name": fn["name"], "arguments": a or {}})
    return {"reasoning": (msg.get("reasoning_content") or "") if (msg.get("reasoning_content") or "").strip() else "",
            "content": msg.get("content") or "", "tool_calls": tcs}


parser_msgs = {
    "plain_reasoning_and_content": {"role": "assistant", "reasoning_content": "Let me think.\n\nStep 1.",
                                    "content": "The answer is 4."},
    "no_reasoning": {"role": "assistant", "content": "Hi there 👋"},
    "cjk_emoji": {"role": "assistant", "reasoning_content": "用户问天气 🤔", "content": "今天晴 ☀️，适合出行。"},
    "content_with_marker_lookalikes": {"role": "assistant", "reasoning_content": "r",
                                       "content": "literal <|open|> and <| and |> and <|sep|> inside"},
    "tool_call_single": {"role": "assistant", "reasoning_content": "need weather", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {
            "city": "北京", "days": 3, "detail": True, "ratio": 0.5, "note": None, "tags": ["a", "b"],
            "nested": {"x": 1}, "q": "a \"b\" & c"}}}]},
    "tool_calls_two_with_content": {"role": "assistant", "reasoning_content": "two", "content": "calling", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "f1", "arguments": {"s": "x\ny"}}},
        {"id": "c2", "type": "function", "function": {"name": "f2", "arguments": {}}}]},
    "tool_call_string_args": {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "calc", "arguments": "{\"x\": 100.0, \"arr\": [1, 2, 3]}"}}]},
}
parser_cases = []
for name, msg in parser_msgs.items():
    body, ids = model_output(msg)
    assert tok.decode(ids) == body, name
    parser_cases.append({"name": name, "text": body, "ids": ids, "expect": expect_of(msg)})

specials = {n: tok.special_tokens[n] for n in ["[BOS]", "[EOS]", "<|end_of_msg|>", "<|open|>", "<|close|>", "<|sep|>",
                                                 "[start_header_id]", "[end_header_id]", "[EOT]", "<|media_begin|>",
                                                 "<|media_content|>", "<|media_end|>", "<|media_pad|>", "<osagent_mode>",
                                                 "[UNK]", "[PAD]", "<|reserved_token_163592|>", "<|reserved_token_163839|>"
                                                 if "<|reserved_token_163839|>" in tok.special_tokens else "[PAD]"]}
golden = {"n_base": len(tok.model._mergeable_ranks), "specials": specials, "eos_token_ids": eos_ids,
          "encode": encode, "incremental": incremental, "cases": out_cases, "parser_cases": parser_cases}
json.dump(golden, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
n_err = sum(1 for c in out_cases.values() if c.get("error"))
print(f"wrote {args.out}: {len(out_cases)} chat cases ({n_err} error cases), {len(encode)} encode, "
      f"{len(inc_ids)} incremental ids, {len(parser_cases)} parser cases")
