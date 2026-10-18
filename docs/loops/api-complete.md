# API 补全（对齐 vLLM OpenAI 服务面）

> 用户 2026-10-18：性能线暂停（无 NVLink 机器，T3 只留 TODO），集中做功能。

## 目标

server 现有 `/health /metrics /v1/models /v1/chat/completions`。补齐 vLLM `api_server.py` 里 decoder-only 模型用得到的端点：

| 端点 | 行为 | 状态 |
|---|---|---|
| `POST /v1/completions` | 文本补全：`prompt` 为字符串 / 字符串数组 / token id 数组 / 其数组；`echo`、`logprobs`（整数 = top_logprobs）、`n`、`stream`、`stop`、`max_tokens` 等与 chat 同一套采样；流式 SSE `text_completion` chunk；`usage` | 已做 |
| `POST /tokenize` | `{prompt}` 或 `{messages, add_generation_prompt, tools, chat_template_kwargs}` → `{count, max_model_len, tokens, token_strs?}` | 已做 |
| `POST /detokenize` | `{tokens}` → `{prompt}` | 已做 |
| `GET /version` | `{version}`（Cargo pkg version） | 已做 |
| `GET /ping`、`POST /ping` | 200 空 body（SageMaker 探活，vLLM 两个方法都有） | 已做 |
| `GET /v1/models/{id}` | 单模型卡；未知 id 404 | 已做 |
| `POST /v1/embeddings` `/pooling` `/score` `/rerank` `/v1/rerank` `/v2/rerank` `/classify` `/v1/audio/*` | 501 `{"error":{"message":"not supported by decoder-only model / not implemented","type":"not_implemented"}}`，代码处 `TODO`（embedding 模型不在本项目范围） | 已做 |
| `/v1/chat/completions` | 已有；补 `model` 不匹配时 404（vLLM 行为），`/v1/models` 卡加 `created / root / permission` 字段 | 已做 |

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
| 1 | `ChatBackend` 加 `encode / decode / token_strs`（qwen3 / glm53 / dp 三处实现）；`openai.rs` 抽 `SamplingFields`（chat 与 completions flatten 共用）、`ErrorBody` 加 `param: null`、新增 Completion* / Tokenize* / Detokenize* 类型；`chat.rs` 的 `err / sampling_from / submit_choices（多 prompt）/ consume（raw）/ lp_entry / async_stream_lite` 改 `pub(crate)` 复用，加 `check_model` 404；新 `completions.rs`（四种 prompt、echo、logprobs 四列、流式 / 非流式、usage、默认 max_tokens 16、suffix 400）；新 `misc.rs`（`/v1/models[/{id}]` 卡加 created/root/parent/permission、`/tokenize`、`/detokenize`、`/version`、`/ping`、501 统一 handler）；e2e `e2e_qwen3_openai_surface`；`scripts/verify/api_surface.sh` | 本机 3090：`cargo build --features qwen3` / `--features glm53` 过；`cargo test --workspace --features qwen3` 全过；e2e 1 passed（2.25 s）；`api_surface.sh` exit 0 |

## 状态

- 第 1 轮通过（2026-10-18）：表内端点全部实现，`scripts/verify/api_surface.sh` exit 0。
- 留下的 TODO(api)：completions 多 prompt 共用一份 `SamplingParams`（max_tokens 余量按最长 prompt 算，vLLM 逐 prompt）；echo 时不含 prompt 的 logprobs（vLLM 需 `prompt_logprobs`）；`/tokenize` 的 `add_generation_prompt=false` 400（渲染器固定 true）、`add_special_tokens` 忽略；pooling / 音频 / responses 501。
- 未做：`openai` Python SDK 冒烟（可选项）。
