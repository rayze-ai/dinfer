#!/usr/bin/env python3
"""用官方 openai Python SDK 对 dinfer serve 做服务面冒烟（docs/loops/api-complete.md）。

用法：python3 -I scripts/verify/sdk_smoke.py http://127.0.0.1:PORT
每项打印 `PASS <name>` / `FAIL <name>: <原因>`，末尾汇总；任一 FAIL 则 exit 1。
"""
import sys
import traceback

import openai
from openai import OpenAI

MAX_TOKENS = 24
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
MESSAGES = [{"role": "user", "content": "用一句话介绍北京。"}]
PROMPT = "The capital of France is"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某城市当前天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名"}},
                "required": ["city"],
            },
        },
    }
]

results: list[tuple[str, str | None]] = []


def check(name):
    """装饰器：运行一项检查，记录 PASS / FAIL（异常即 FAIL）。"""

    def deco(fn):
        try:
            fn()
            results.append((name, None))
            print(f"PASS {name}", flush=True)
        except Exception as e:  # noqa: BLE001
            reason = f"{type(e).__name__}: {e}".splitlines()[0][:300]
            results.append((name, reason))
            print(f"FAIL {name}: {reason}", flush=True)
            if "-v" in sys.argv:
                traceback.print_exc()
        return fn

    return deco


def main(base_url: str) -> int:
    client = OpenAI(base_url=base_url, api_key="x", max_retries=0, timeout=120)
    state: dict = {}

    @check("models.list")
    def _():
        page = client.models.list()
        ids = [m.id for m in page.data]
        assert ids, "models.list 返回空"
        m = page.data[0]
        assert m.object == "model" and m.owned_by and m.created > 0, f"模型卡字段不全: {m}"
        state["model"] = ids[0]

    model = state.get("model", "unknown")

    @check("models.retrieve")
    def _():
        m = client.models.retrieve(model)
        assert m.id == model, f"id 不符: {m.id} != {model}"

    @check("chat.non_stream")
    def _():
        r = client.chat.completions.create(
            model=model, messages=MESSAGES, temperature=0, max_tokens=MAX_TOKENS, extra_body=NO_THINK
        )
        assert r.object == "chat.completion" and len(r.choices) == 1
        c = r.choices[0]
        assert c.message.role == "assistant"
        assert c.message.content, "content 为空"
        assert c.finish_reason in ("stop", "length"), c.finish_reason
        assert r.usage is not None and r.usage.completion_tokens >= 1
        state["chat_text"] = c.message.content

    @check("chat.stream_equals_non_stream")
    def _():
        parts = []
        finish = None
        for chunk in client.chat.completions.create(
            model=model, messages=MESSAGES, temperature=0, max_tokens=MAX_TOKENS, extra_body=NO_THINK, stream=True
        ):
            assert chunk.object == "chat.completion.chunk"
            for ch in chunk.choices:
                if ch.delta.content:
                    parts.append(ch.delta.content)
                if ch.finish_reason:
                    finish = ch.finish_reason
        text = "".join(parts)
        assert finish in ("stop", "length"), f"流式无 finish_reason: {finish}"
        assert text == state["chat_text"], f"流式 != 非流式:\n  stream={text!r}\n  plain ={state['chat_text']!r}"

    @check("chat.tools")
    def _():
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "北京现在天气怎么样？请调用工具查询。"}],
            tools=TOOLS,
            temperature=0,
            max_tokens=96,
            extra_body=NO_THINK,
        )
        msg = r.choices[0].message
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            assert tc.type == "function" and tc.function.name, f"tool_call 形状不对: {tc}"
            assert tc.id, "tool_call 无 id"
            assert r.choices[0].finish_reason == "tool_calls", r.choices[0].finish_reason
        else:
            assert msg.content, "既无 tool_calls 也无 content"

    @check("completions.non_stream_echo_logprobs")
    def _():
        r = client.completions.create(
            model=model, prompt=PROMPT, temperature=0, max_tokens=MAX_TOKENS, echo=True, logprobs=2
        )
        assert r.object == "text_completion" and len(r.choices) == 1
        c = r.choices[0]
        assert c.text.startswith(PROMPT), f"echo 前缀不是 prompt: {c.text!r}"
        assert len(c.text) > len(PROMPT), "echo 后无生成文本"
        lp = c.logprobs
        assert lp is not None and lp.top_logprobs, "logprobs 为空"
        assert len(lp.tokens) == len(lp.token_logprobs) == len(lp.top_logprobs) == len(lp.text_offset), "四列长度不一"
        for i, d in enumerate(lp.top_logprobs):
            assert 1 <= len(d) <= 3, f"top_logprobs[{i}] 有 {len(d)} 个键: {d}"
            assert lp.tokens[i] in d, f"chosen token {lp.tokens[i]!r} 不在 top_logprobs[{i}]"
        assert lp.text_offset[0] >= len(PROMPT), f"echo 时首 offset 应 ≥ prompt 长度: {lp.text_offset[0]}"
        state["cmpl_text"] = c.text

    @check("completions.stream_equals_non_stream")
    def _():
        parts = []
        n_lp = 0
        for chunk in client.completions.create(
            model=model, prompt=PROMPT, temperature=0, max_tokens=MAX_TOKENS, echo=True, logprobs=2, stream=True
        ):
            assert chunk.object == "text_completion"
            for ch in chunk.choices:
                parts.append(ch.text)
                if ch.logprobs and ch.logprobs.top_logprobs:
                    for d in ch.logprobs.top_logprobs:
                        assert 1 <= len(d) <= 3, f"流式 top_logprobs 有 {len(d)} 个键"
                        n_lp += 1
        text = "".join(parts)
        assert n_lp >= 1, "流式无 logprobs"
        assert text == state["cmpl_text"], f"流式 != 非流式:\n  stream={text!r}\n  plain ={state['cmpl_text']!r}"

    @check("completions.n2")
    def _():
        r = client.completions.create(model=model, prompt=PROMPT, temperature=0, max_tokens=8, n=2)
        assert len(r.choices) == 2, f"n=2 返回 {len(r.choices)} 个 choice"
        assert sorted(c.index for c in r.choices) == [0, 1]
        assert all(c.text for c in r.choices), "有 choice 文本为空"

    @check("chat.n2")
    def _():
        r = client.chat.completions.create(
            model=model, messages=MESSAGES, temperature=0, max_tokens=8, n=2, extra_body=NO_THINK
        )
        assert len(r.choices) == 2, f"n=2 返回 {len(r.choices)} 个 choice"
        assert sorted(c.index for c in r.choices) == [0, 1]

    @check("completions.stop")
    def _():
        r = client.completions.create(
            model=model, prompt="List three colors, one per sentence:", temperature=0, max_tokens=48, stop=["."]
        )
        t = r.choices[0].text
        assert "." not in t, f"stop 后文本仍含 '.': {t!r}"
        assert r.choices[0].finish_reason in ("stop", "length"), r.choices[0].finish_reason

    @check("chat.seed_reproducible")
    def _():
        outs = []
        for _ in range(2):
            r = client.chat.completions.create(
                model=model, messages=MESSAGES, temperature=0.7, seed=42, max_tokens=MAX_TOKENS, extra_body=NO_THINK
            )
            outs.append(r.choices[0].message.content)
        assert outs[0] == outs[1], f"seed=42 两次不同:\n  a={outs[0]!r}\n  b={outs[1]!r}"

    @check("error.unknown_model_404")
    def _():
        try:
            client.chat.completions.create(
                model="no-such-model", messages=MESSAGES, max_tokens=4, extra_body=NO_THINK
            )
        except openai.NotFoundError as e:
            assert e.status_code == 404
            assert "does not exist" in str(e), str(e)
            return
        raise AssertionError("未抛 NotFoundError")

    @check("error.suffix_400")
    def _():
        try:
            client.completions.create(model=model, prompt=PROMPT, suffix=" tail", max_tokens=4)
        except openai.BadRequestError as e:
            assert e.status_code == 400
            assert "suffix" in str(e), str(e)
            return
        raise AssertionError("未抛 BadRequestError")

    @check("error.embeddings_501")
    def _():
        try:
            client.embeddings.create(model=model, input="hello")
        except openai.APIStatusError as e:
            assert e.status_code == 501, f"status {e.status_code}"
            body = e.body if isinstance(e.body, dict) else {}
            assert body.get("type") == "not_implemented" or "not_implemented" in str(e), str(e)
            return
        raise AssertionError("未抛 APIStatusError")

    fails = [(n, r) for n, r in results if r]
    print(f"\nsdk_smoke: {len(results) - len(fails)}/{len(results)} passed" + (f", {len(fails)} FAILED" if fails else ""))
    for n, r in fails:
        print(f"  FAIL {n}: {r}")
    return 1 if fails else 0


if __name__ == "__main__":
    if len(sys.argv) < 2 or not sys.argv[1].startswith("http"):
        print("usage: sdk_smoke.py http://host:port [-v]", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1].rstrip("/") + "/v1"))
