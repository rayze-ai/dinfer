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

## 状态

- 规格写好，开工。
