# loop 规格：P6 GLM-5.3 tokenizer / chat 模板 / 流式解析 / server / MTP

- 状态：进行中
- 创建：2026-10-15
- 验证脚本：`scripts/verify/p6.sh`
- 迭代记录：`test/YYYY-MM-DD/p6-glm-serve.md`

## 目标

用 tiny DSA GLM（结构同 GLM-5.3，随机权重）+ **真** GLM-5.3 tokenizer / chat 模板，跑通 OpenAI 兼容 `/v1/chat/completions`（流式 + 非流式，含 `reasoning_content`、`tool_calls`、`reasoning_effort`），
并实现 MTP（nextn=1）投机解码且可开关。真权重验收（输出合理文本）留到有 ≥ 8×H200 的机器。

分四步，每步有独立停止条件，`p6.sh` 顺序跑：

| 步 | 内容 | 产出 |
|---|---|---|
| a | tokenizer（`tokenizers` 加载 `tokenizer.json`，eos 集合，增量 detokenize）+ chat 模板（`minijinja` 跑 `chat_template.jinja`，`reasoning_effort` / `clear_thinking` / tools / tool 结果排序） | `glm53/src/{tokenizer,chat}.rs`，`tests/chat_parity.rs` |
| b | 流式输出解析：`<think>…</think>` → `reasoning_content`，`<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>…</tool_call>` → `tool_calls`（增量、跨 token 边界） | `glm53/src/parser.rs` + 单测 |
| c | engine 循环（Scheduler + Runner + 请求队列 + 增量 detokenize + stop）、server：`/v1/chat/completions`（stream / non-stream，SSE，`usage`）、`/v1/models`、`/health`；bin `serve --model <dir>` | `core/src/engine.rs`，`server/src/{openai,chat,stream}.rs`，`tests/serve_e2e.rs` |
| d | MTP：`model.layers.78`（enorm / hnorm / eh_proj / 一个 DSA decoder 层 / shared_head.norm，lm_head 共享）；draft 1 token → verify → 接受/拒绝；`SpecConfig` 开关 | `glm53/src/mtp.rs`，`core/src/runner.rs` 接 `spec_ids`，`tests/mtp_parity.rs` |

## 环境

- 启动：
  ```bash
  export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
  # fixture（已有）：test/tiny_glm_dsa（gen_tiny_glm.py --dsa）；真 tokenizer / 模板：/data/ref/models/glm53/{tokenizer.json,chat_template.jinja,tokenizer_config.json,generation_config.json}
  # HF golden（a/b 用）：/data/ref/venv-tf519/bin/python 生成 test/tiny_glm_dsa/chat_golden.json（见 scripts/verify/gen_chat_golden.py）
  cargo run -p dinfer --features glm53 -- serve --model test/tiny_glm_dsa --tokenizer /data/ref/models/glm53 --port 8011
  ```
- 依赖：本机 3090；`/data/ref/venv-tf519`（transformers 5.19，`apply_chat_template` 作 golden）；`openai` Python SDK（e2e 用，装进同一 venv）。
- 重置：`pkill -f 'dinfer.*serve'`；fixture 重生成同 p5。

## 观察方法

| 看什么 | 在哪里 | 命令 |
|---|---|---|
| chat 模板渲染 vs HF | `tests/chat_parity.rs` | `TINY_GLM_DIR=... GLM_TOKENIZER_DIR=/data/ref/models/glm53 cargo test -p dinfer-glm53 --features ref-tests --test chat_parity -- --nocapture` |
| 解析器 | 单测 | `cargo test -p dinfer-glm53 parser` |
| server | 日志 + curl | `curl -N localhost:8011/v1/chat/completions -d '{...,"stream":true}'` |
| MTP | `tests/mtp_parity.rs` | 接受率、与非 MTP greedy 输出逐 token 一致 |

## 停止条件（必须可机器判定）

1. a：`chat_golden.json` 全部 case 渲染文本 **逐字符一致**、token id **逐个一致**；encode golden 一致；增量 decode 前缀文本一致。
2. b：解析器单测：纯文本、只 think、think+content、1 个 / 2 个 tool_call、arg_value 为 JSON 对象 / 字符串、标签被切在任意 token 边界（枚举所有切分点）→ 结果一致。
3. c：`tests/serve_e2e.rs`（起 server 进程 + reqwest 或 openai SDK 脚本）：非流式返回 `choices[0].message.{reasoning_content,content}` 且 `usage.prompt_tokens == HF 模板 token 数`；流式 SSE 拼接结果 == 非流式结果（greedy）；`finish_reason` 为 stop / length；`/v1/models` 列出模型。tiny 模型输出是随机 token，不验语义，但 greedy 输出必须等于 `runner_parity` 路径的 token（同 prompt）。
4. d：MTP 开启时 greedy 输出 token 序列与关闭时 **完全一致**（tiny fixture 的 MTP 层随机权重：接受率低但正确性可判）；`gen_tiny_glm.py --dsa --mtp` 导出 HF 侧 MTP 层 logits，draft logits 对拍 max abs < 0.05。
5. `scripts/verify/p6.sh` exit 0（含 p5 回归）。

## 上限

每步 8 轮；总计 2 个工作日。超出就停下，把卡点写进本文件。

## 已知可能出错点（按概率排）

1. minijinja 对该模板的方言支持：`namespace()`、`{%- break -%}`（需 `loop_controls` feature）、`tojson(ensure_ascii=False)`（minijinja `tojson` 无 kwargs，需自定义 filter）、`.split()/.strip()/.items()`、`is mapping/iterable/string`、`| capitalize`。
2. `tokenizers` 对 `[gMASK]<sop>` 等 special token 的 `add_special_tokens=False` 编码：HF 把它们当 added token 直接切；Rust 侧要一致。
3. 增量 detokenize 的 UTF-8 边界（emoji / 中文被切成多个 byte-level token）→ 用 `tokenizers::DecodeStream`。
4. 解析器：`<arg_value>` 内容可能含 `<`；工具参数 JSON 还是原字符串由 tools schema 的类型决定（HF 模板：`v | tojson if v is not string else v`）。
5. MTP：`positions == 0` 的 embedding 置零；eh_proj 输入是 `[enorm(embed) | hnorm(prev_hidden)]`；DSA 索引器在 MTP 层按 `index_share_for_mtp_iteration=True` 复用主干最后一个 full 层的 top-k（vLLM `set_skip_topk`）。
6. SSE 首包 `role: assistant`，`reasoning_content` 与 `content` 字段的 delta 切换点；`stream_options.include_usage`。

## 记录

| 轮 | 现象 | 根因 | 处理 |
|---|---|---|---|
