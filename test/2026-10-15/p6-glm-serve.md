# P6 GLM-5.3 tokenizer / chat 模板 / 解析器 / server / MTP（2026-10-15）

## 结论

四步全部通过（`scripts/verify/p6.sh` 全量结果见 `p6.log`）：

| 步 | 测试 | 结果 |
|---|---|---|
| a | `chat_parity`（真 GLM-5.3 tokenizer.json + chat_template.jinja vs HF transformers 5.19） | 15/15 chat case 渲染文本逐字符一致、token id 逐个一致；7 条 encode/decode 往返一致；增量 decode 前缀一致；特殊 token / eos 一致 |
| b | `parser::` 8 单测 | 纯 think / think+content / 1 & 2 tool_call / 无参 / JSON & 字符串值 / `<` 入值 / 未闭合丢弃 / 标签切在每个字符边界 → 全部与整体 feed 一致 |
| c | `serve_e2e`（起真进程：154880 词表 tiny DSA + 真 tokenizer） | `/v1/models`；非流式 `usage.prompt_tokens == HF apply_chat_template 的 token 数`，`prompt_token_ids` 逐个相等；流式 27 chunk，拼接 == 非流式，`[DONE]`，usage chunk；两请求并发；空 messages → 400 |
| d | `mtp_parity` 3 测试 | draft logits vs HF MTP 参考：63/64 行 worst max abs 0.0088，argmax 0 错；oracle draft 接受 16/16、17 步出 32 token、输出 == plain；真 MTP draft：spec 输出 == plain greedy 逐 token（接受 0/31，tiny 随机权重 HF 自己的 MTP 与主干一致率也是 0.000） |

关键实现点：
- minijinja 跑 HF 模板需要：`trim_blocks/lstrip_blocks`（HF 默认）、`.0.attr` → `[0].attr` 改写、Python 方法 shim（strip/split/items…）、Python 风格 `tojson`（`", "` 分隔、非 ASCII 原样）、`preserve_order`（Python dict 序）。
- 投机解码 runner 协议：`BatchPlan.{spec_tokens, logits_row_start, next_known}`，`ModelExecutor::forward_sample → (generated, drafts)`，runner 回退被拒 draft 的 `num_computed`；paged KV 按位置重写，无需显式回滚。
- MTP 层 KV 与主干 "右移一位"：第 pos 行的 K 来自 token[pos+1]；中间 chunk 用 `next_known` 推进 MTP KV。

## 日志节选
```
# chat_parity
# parser
# serve_e2e
e2e: models ok: {"id":"glm-5.3","object":"model","owned_by":"dinfer","max_model_len":512}
e2e: non-stream: {"id":"chatcmpl-6ad0b20e0001","object":"chat.completion","created":1792061966,"model":"glm-5.3","choices":[{"index":0,"message":{"role":"assistant","content":"","reasoning_content":"�
e2e: stream chunks 27, finish Some("length"), usage Some(Object {"prompt_tokens": Number(17), "completion_tokens": Number(24), "total_tokens": Number(41)})
e2e: concurrent ok
e2e: PASS
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.54s
# mtp_parity
test mtp_draft_logits_match_hf ... mtp: draft logits vs HF on 63/64 rows (0 drifted skipped): worst max abs 0.0088, argmax mismatches 0
mtp: plain vs spec first diff at None; acceptance 0/31
mtp: plain greedy matches HF for the first 7/16 tokens
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.10s
```
