# API 补全（对齐 vLLM OpenAI 服务面）

> 用户 2026-10-18：性能线暂停（无 NVLink 机器，T3 只留 TODO），集中做功能。

## 目标

server 现有 `/health /metrics /v1/models /v1/chat/completions`。补齐 vLLM `api_server.py` 里 decoder-only 模型用得到的端点：

| 端点 | 行为 | 状态 |
|---|---|---|
| `POST /v1/completions` | 文本补全：`prompt` 为字符串 / 字符串数组 / token id 数组 / 其数组；`echo`、`logprobs`（整数 = top_logprobs）、`n`、`stream`、`stop`、`max_tokens` 等与 chat 同一套采样；流式 SSE `text_completion` chunk；`usage` | 待做 |
| `POST /tokenize` | `{prompt}` 或 `{messages, add_generation_prompt, tools, chat_template_kwargs}` → `{count, max_model_len, tokens, token_strs?}` | 待做 |
| `POST /detokenize` | `{tokens}` → `{prompt}` | 待做 |
| `GET /version` | `{version}`（Cargo pkg version） | 待做 |
| `GET /ping`、`POST /ping` | 200 空 body（SageMaker 探活，vLLM 两个方法都有） | 待做 |
| `GET /v1/models/{id}` | 单模型卡；未知 id 404 | 待做 |
| `POST /v1/embeddings` `/pooling` `/score` `/rerank` `/v1/rerank` `/v2/rerank` `/classify` `/v1/audio/*` | 501 `{"error":{"message":"not supported by decoder-only model / not implemented","type":"not_implemented"}}`，代码处 `TODO`（embedding 模型不在本项目范围） | 待做 |
| `/v1/chat/completions` | 已有；补 `model` 不匹配时 404（vLLM 行为），`/v1/models` 卡加 `created / root / permission` 字段 | 待做 |

第 2 批（本轮不做，下一 loop）：chat 请求字段缺口 `response_format`（json_object / json_schema → 受限解码）、`tool_choice`（none / auto / required / 指定函数）、`parallel_tool_calls`、`logit_bias`、`prompt_logprobs`、`skip_special_tokens` / `spaces_between_special_tokens`、`truncate_prompt_tokens`、`continue_final_message` / `add_generation_prompt`、`user`。受限解码是其中最大的一块，单开 `docs/loops/guided-decoding.md`。

不做：`/v1/responses`（vLLM 的有状态会话，范围外，留 TODO）、`/invocations`、`/v1/load_lora_adapter`。

## 环境

本机 3090 单卡，`QWEN3_DIR=/data/ref/models/qwen3-0.6b`；e2e 用 `bins/dinfer/tests/serve_e2e_qwen3.rs` 的 `spawn_server_with`。

## 观察方法

- 新 e2e：`e2e_qwen3_openai_surface`：每个端点一条正例 + 一条错例（400 / 404 / 501），`/v1/completions` 流式 vs 非流式同 prompt greedy 逐字相同，`echo=true` 前缀 = prompt，`tokenize(detokenize(x)) == x`，`tokenize` 的 `count` == 对应 chat 请求 `usage.prompt_tokens`。
- 用官方 `openai` Python SDK（本机 pip 有）跑 `client.completions.create(stream=True/False)` 各一次不报错（可选，脚本 `scripts/verify/api_surface.sh`）。

## 停止条件

`scripts/verify/api_surface.sh` exit 0（= 上面 e2e 过 + curl 每个端点状态码符合表）。

## 迭代上限

6 轮。

## 记录

| 轮 | 改动 | 结果 |
|---|---|---|

## 状态

- 规格写好，开工。
