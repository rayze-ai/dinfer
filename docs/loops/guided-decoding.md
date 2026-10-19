# 受限解码 + chat 字段补全（A1 / A2 / A3）

> 用户 2026-10-19："先完成 A"。A 类不依赖 sm90：grammar → token mask 在 CPU（vLLM 也是 xgrammar / llguidance 在 CPU 算 bitmask），GPU 只多一次 logits mask。

## 目标

1. `response_format`：`{"type":"json_object"}`（任意合法 JSON）与 `{"type":"json_schema","json_schema":{...}}`（按 schema）；`text` = 不约束。
2. `tool_choice`：`none`（不渲染 tools）/ `auto`（现状）/ `required`（必须产生 ≥ 1 个 tool_call，用 tools 的 schema 拼 anyOf 约束）/ `{"type":"function","function":{"name":X}}`（只允许该函数）。`parallel_tool_calls=false` 时只允许一个。
3. 实现：`llguidance` crate（vLLM 的 `guidance` 后端同源，crates.io 1.9.x，`toktrie_hf_tokenizers` 从 tokenizer.json 建 trie）。每请求一个 `Matcher`，每步：`compute_mask()` → `Vec<u32>` bitmask → 上传 → 采样前 `logits_mask_f32` kernel（置 -inf）→ 采样后 `consume_token()`。走 `SampleBatch` 加一列 `grammar_mask: Option<...>`，无约束请求零开销。CUDA graph：mask 缓冲固定 `max_num_seqs × ceil(vocab/32)` u32，图内只读；有约束请求时每步 H2D 一次。
4. 顺手补字段（A2）：`logit_bias`（同一个 mask kernel 加偏置列）、`prompt_logprobs`、`skip_special_tokens` / `spaces_between_special_tokens`、`truncate_prompt_tokens`、`continue_final_message` / `add_generation_prompt`（`/tokenize` 也放开）、`user`（透传忽略）。
5. completions TODO(api)（A3）：多 prompt 逐 prompt 算 max_tokens 余量；`echo + logprobs` 带 prompt 段 logprobs（依赖 `prompt_logprobs`）。

## 环境

本机 3090，`QWEN3_DIR=/data/ref/models/qwen3-0.6b`。e2e 文件 `bins/dinfer/tests/serve_e2e_qwen3.rs`，一次只跑一个 e2e。

## 观察方法

- e2e `e2e_qwen3_guided`：json_object 输出 `serde_json::from_str` 成功 ×5 prompt；json_schema 输出通过 `jsonschema` crate 校验；`tool_choice=required` 返回 ≥1 tool_call 且 arguments 合法 JSON；指定函数名时 name 一致；`none` 时无 tool_call；`logit_bias` 把某 token 置 -100 后输出不含该 token；`prompt_logprobs` 长度 = prompt_tokens；`truncate_prompt_tokens=8` 时 usage.prompt_tokens == 8。
- 性能：无约束请求 TPOT 不变（bench_compare c8 与 bench23 持平 ±2%）；有约束请求每步 mask 开销 < 0.3 ms（打印 step breakdown）。
- 旧 e2e 全部回归（`e2e_qwen3_request_features` / `real_model_chat` / `openai_surface`）。

## 停止条件

`scripts/verify/guided.sh` exit 0（= 上面 e2e + 回归）。

## 迭代上限

8 轮。

## 记录

| 轮 | 改动 | 结果 |
|---|---|---|
| 1 | kernel `logits_mask_f32`（mask_row/-1 + bitmask 行 + bias CSR，无约束行立即返回；`sampling.cu` / `dinfer_kernels.h` / `ffi.rs` / `ops.rs`）；core：`GrammarState` trait + `GrammarHandle`（Arc<Mutex>，`push` 时按输出序列追赶 consume → fill_mask，错误退化为无约束）、`SamplingParams` 加 `grammar / logit_bias / prompt_logprobs / spaces_between_special_tokens`、`SampleBatch` 加 mask / bias CSR / prompt_logprobs 列、`TokenLogprobs.prompt_pos`；executor：StepBuffers 加 mask / bias / plp 缓冲（按 vocab 分配），`upload_sampling` 只在 `any_mask` 时上传、`mask_dirty` 清回，graph 路径 `logits_mask_f32` 常驻（无约束行零工作），eager 路径 `any_mask` 才发；`prompt_logprobs` 复用 `model.logits_for_rows`（hidden 行再过 norm+lm_head）；runner `async_eligible` 加 `!any_mask` | core 单测 4/4 过；qwen3 crate `cargo check` 过 |
| 2 | 新 crate `crates/guided`（`llguidance = 1.9.1` + `toktrie_hf_tokenizers`；`GrammarFactory` 启动建 TokEnv（n_vocab = 模型 vocab）、`GrammarSpec::{JsonSchema,JsonObject,Lark}`、`tool_call_lark`：`<tool_call>` / `</tool_call>` 是 Qwen3 special token → lark 裸 `<name>` 语法而非字面字节，`%json {schema}` 内嵌 arguments）；server：`ChatBackend::{grammar_factory,tool_call_tags}`、`RenderRequest::{add_generation_prompt,continue_final_message}`、`OutputStream::set_skip_special_tokens`；chat：`response_format` / `tool_choice`（none 不渲染 tools）/ `parallel_tool_calls` → `build_grammar`（带约束时默认 `enable_thinking=false`），`logit_bias`（map→CSR）、`prompt_logprobs`（响应 `prompt_logprobs: [null, {tok:{logprob,rank,decoded_token}}…]`）、`skip_special_tokens` / `spaces_between_special_tokens`、`truncate_prompt_tokens`（左截断）、`continue_final_message`（dinfer_text 渲染后截到最后 assistant content）、`add_generation_prompt`（`/tokenize` 放开）、`user` 忽略；**tool_calls → finish_reason "tool_calls"**（流式 / 非流式，coordinator 追加）；completions：逐 prompt `SamplingParams`（room 各算）、echo+logprobs 带 prompt 段（第 0 项 null，`token_logprobs` / `top_logprobs` 列改 Option）、`truncate_prompt_tokens`；e2e `e2e_qwen3_guided`（jsonschema 0.59 校验）+ `scripts/verify/guided.sh`；`sdk_smoke.py` 只接受 "tool_calls" | guided 单测 2/2（含真 tokenizer 文法走查：json_object 首 token 只能 `{`、完整 JSON 后 eos 允许、tool lark 走通、非法 token 退化）；`cargo build --features qwen3` 过；e2e 见下 |
| 3 | 修 `prompt_logprobs` 丢失（`sched/output.rs` 把 logprobs 与 token 1:1 对齐，多出的条目被丢 → 改成挂在该步生成 token 条目的 `TokenLogprobs.prompt` 下，无生成 logprobs 时放 `PLACEHOLDER` 占位，server 展平）；`jsonschema` 0.59 要新 serde_json → 改 `=0.30.0`；`guided.sh` 导出绝对 `QWEN3_GOLDEN`（real_model_chat 的 cwd 问题，与 ngram.sh 同）；与 n-gram 子代理同步：`SampleBatch` 现在按 **logits 行**（带 draft 1+K 行）→ `upload_plan` 的 prompt_logprobs 用 `logits_row_start[i]`；**带文法的请求不提案 draft**（runner 一行门控：draft 行的 mask 需文法逐 token 前进 / 回退，未做）→ mask 列按行索引天然正确（文法请求恒 1 行）；e2e 加 §9：`--speculative ngram` 下 json_object ×2 + tool_choice=required 仍合法 | `e2e_qwen3_guided` PASS（12.7 s，含 ngram 段）；`scripts/verify/guided.sh`：单测 55+2+3 过、guided / request_features PASS，real_model_chat / openai_surface 修 GOLDEN 后单跑 PASS；`cargo test --workspace --features qwen3` 全过；`--features glm53` 编译过 |

## 状态

- **第 3 轮通过（2026-10-19）**：A1 受限解码（json_object / json_schema / tool_choice none|auto|required|指定函数 / parallel_tool_calls）、A2 字段（logit_bias / prompt_logprobs / skip_special_tokens / spaces_between_special_tokens / truncate_prompt_tokens / continue_final_message / add_generation_prompt / user）、A3（completions 逐 prompt room、echo+logprobs prompt 段）全部实现；`scripts/verify/guided.sh` 各步 PASS（整脚本一次跑完的最终记录见 progress）。chat 有 tool_calls 时 `finish_reason = "tool_calls"`（流式 / 非流式），`sdk_smoke.py` 已改为只接受它。
- 性能口径：无约束请求零额外 H2D（`mask_dirty` 机制）、graph 内多一个 `logits_mask_f32` launch（无约束行立即返回，~µs 级）；**未在 4090 跑 bench_compare**（本机只验功能）→ 下一轮性能改动时顺带记一行 CSV 确认 TPOT 持平。
- 第 4 轮（主会话，2026-10-19）：按用户检查把 Qwen 特定内容移出 `crates/`：server 的 `enable_thinking` 硬编码 → `ChatBackend::grammar_render_extra()` 钩子（qwen3 实现）；`crates/guided` 里的 Qwen3 文法走查单测 → `models/qwen3/tests/guided_parity.rs`。guided.sh 复跑 PASS。
- 留下的 TODO(api)：
  - 思考段不约束（vLLM reasoning-aware structured output）：现在带约束时服务端默认 `enable_thinking=false`；要支持思考 + 约束需 parser 状态回传引擎。
  - 受限解码 + 投机解码：文法请求不提案 draft（runner 门控）；要联动需按 vLLM accept/rollback 在 draft 行逐 token 前进文法。
  - `prompt_logprobs` 只在整段 prompt 落在一个 chunk（≤ max_num_batched_tokens）且无 prefix cache 命中时完整（中间 chunk 行 runner 不回传 logprobs；命中前缀无 hidden），缺的位置为 null；流式 chat 不返回 prompt_logprobs（vLLM 亦然）。
  - `spaces_between_special_tokens` 只接受字段（HF `tokenizers` decoder 没有对应开关）；`add_special_tokens` 仍忽略。
  - logprobs 口径：vLLM 的 logprobs 在 bitmask 之后、logit_bias / penalties 之前取；这里 logprobs 用原始 logits（mask 前）。
  - GLM 线：`tool_call_tags` / `grammar_factory` 未接（返回 None → response_format / 约束 tool_choice 400），只做了 render 字段同步编译。
