//! OpenAI 兼容的请求 / 响应类型（`/v1/chat/completions` 子集：messages、tools、stream、stream_options、
//! max_tokens / max_completion_tokens、temperature、top_p、stop、reasoning_effort、以及 GLM 的 `clear_thinking`）。
//! 响应里 `reasoning_content` 与 `tool_calls` 按 vLLM 的字段形状输出。

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    /// 原样保留（模板直接吃 OpenAI 结构，含 content 为数组、tool_calls、tool_call_id、reasoning_content）。
    pub messages: Vec<serde_json::Value>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    /// 与 `/v1/completions` 共用的采样字段（flatten：JSON 里仍是顶层字段）。
    #[serde(flatten)]
    pub sampling: SamplingFields,
    /// OpenAI `logprobs`（bool）+ `top_logprobs`（0..=20）。
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<u32>,
    /// 1..=8：n 个独立采样（引擎侧 n 个请求，seed 依次 +1）。
    #[serde(default)]
    pub n: Option<u32>,
    /// `low` / `high` / `max`（GLM 模板只认 low/high，其余按 max）。
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// GLM 扩展：多轮时丢弃早轮思考内容。
    #[serde(default)]
    pub clear_thinking: Option<bool>,
    /// vLLM 风格透传给模板的额外变量。
    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
    /// 返回 token id 列表（调试）。
    #[serde(default)]
    pub return_token_ids: Option<bool>,
    /// OpenAI `response_format`：`{"type":"text"}` / `{"type":"json_object"}` / `{"type":"json_schema","json_schema":{"schema":{...}}}`。
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    /// OpenAI `tool_choice`：`"none"` / `"auto"` / `"required"` / `{"type":"function","function":{"name":X}}`。
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// OpenAI `parallel_tool_calls`（默认 true）；只在受限 `tool_choice`（required / 指定函数）下生效。
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    /// vLLM `continue_final_message` / `add_generation_prompt`（渲染层）。
    #[serde(default)]
    pub continue_final_message: Option<bool>,
    #[serde(default)]
    pub add_generation_prompt: Option<bool>,
    /// PD 分离（design §10.3，字段名对齐 vLLM disagg proxy）：proxy 内部使用，普通客户端不传。
    /// `{"do_remote_decode": true}` → 本实例是 P：只出首 token、导出 KV；响应头 `x-pd-request-id` 回传 P 侧 id。
    /// `{"do_remote_prefill": true, "remote_request_id": N}` → 本实例是 D：按 N 配对已到的 KV 接力。
    #[serde(default)]
    pub kv_transfer_params: Option<KvTransferParams>,
}

/// chat 与 completions 共用的采样字段（JSON 顶层，`#[serde(flatten)]` 进各请求）。
#[derive(Debug, Clone, Deserialize, Default)]
pub struct SamplingFields {
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// vLLM 扩展：按 token id 停（字符串 `stop` 见 Q2-b）。
    #[serde(default)]
    pub stop_token_ids: Option<Vec<u32>>,
    #[serde(default)]
    pub ignore_eos: Option<bool>,
    #[serde(default)]
    pub min_tokens: Option<u32>,
    /// OpenAI `stop`：字符串或字符串数组（≤ 16 个）。
    #[serde(default)]
    pub stop: Option<serde_json::Value>,
    /// vLLM 扩展：命中的 stop 字符串是否留在输出里（默认 false）。
    #[serde(default)]
    pub include_stop_str_in_output: Option<bool>,
    /// OpenAI `logit_bias`：`{"<token_id>": bias}`（-100..100，≤ 300 项）。
    #[serde(default)]
    pub logit_bias: Option<serde_json::Map<String, serde_json::Value>>,
    /// vLLM `prompt_logprobs`：Some(k) → 响应带 `prompt_logprobs`（prompt 每个位置的 logprob + top-k）。
    #[serde(default)]
    pub prompt_logprobs: Option<u32>,
    #[serde(default)]
    pub skip_special_tokens: Option<bool>,
    #[serde(default)]
    pub spaces_between_special_tokens: Option<bool>,
    /// vLLM `truncate_prompt_tokens`：只保留 prompt 最后 N 个 token（左截断）。
    #[serde(default)]
    pub truncate_prompt_tokens: Option<u32>,
    /// OpenAI `user`：接受并忽略。
    #[serde(default)]
    pub user: Option<String>,
}

/// OpenAI `response_format`。
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub kind: String,
    /// `json_schema` 时：`{"name":..., "schema": {...}, "strict": ...}`（OpenAI）或直接给 schema（vLLM 也接受）。
    #[serde(default)]
    pub json_schema: Option<serde_json::Value>,
}

/// vLLM `prompt_logprobs` 响应形状：`[null, {"<tok>": {"logprob","rank","decoded_token"}}, ...]`，长度 = prompt_tokens。
pub type PromptLogprobs = Vec<Option<serde_json::Map<String, serde_json::Value>>>;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct ChatMessage {
    pub role: &'static str,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatLogprobs>,
}

/// OpenAI `choices[].logprobs`：`{"content": [ {token, logprob, bytes, top_logprobs: [...]}, ... ]}`。
#[derive(Debug, Clone, Serialize, Default)]
pub struct ChatLogprobs {
    pub content: Vec<ChatLogprobContent>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatLogprobContent {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<ChatLogprob>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_token_ids: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_ids: Option<Vec<u32>>,
    /// vLLM 扩展（请求给了 `prompt_logprobs` 时）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_logprobs: Option<PromptLogprobs>,
}

// ---- streaming ----

#[derive(Debug, Clone, Serialize, Default)]
pub struct DeltaToolCallFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DeltaToolCall {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<&'static str>,
    pub function: DeltaToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<DeltaToolCall>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatLogprobs>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    /// vLLM 固定输出 `param: null`。
    pub param: Option<String>,
    pub code: u16,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorBody,
}

pub fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// `chatcmpl-<hex>`，进程内单调递增 + 时间戳，不依赖随机数 crate。
pub fn new_id(prefix: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(1);
    let n = N.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{:x}{:04x}", now_secs(), n & 0xffff)
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct KvTransferParams {
    #[serde(default)]
    pub do_remote_decode: bool,
    #[serde(default)]
    pub do_remote_prefill: bool,
    #[serde(default)]
    pub remote_request_id: Option<u64>,
    /// D 侧：P 已出的首 token id（proxy 从 P 的 `token_ids` 取）。KV 没到而本地重算时，把它拼进 prompt 续写，
    /// 使回退与接力两种路径的输出都不含首 token（proxy 已把首 token 发给客户端）。
    #[serde(default)]
    pub first_token: Option<u32>,
}

// ---- /v1/completions ----

/// vLLM `CompletionRequest` 子集。`prompt`：字符串 / 字符串数组 / token id 数组 / 其数组（见 completions.rs）。
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: serde_json::Value,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// vLLM 默认 16。
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// 整数 = 每个 token 附带的 top_logprobs 个数（0 也返回 chosen token 的 logprob）。
    #[serde(default)]
    pub logprobs: Option<u32>,
    #[serde(default)]
    pub echo: Option<bool>,
    /// 不支持：给了就 400。
    #[serde(default)]
    pub suffix: Option<String>,
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(flatten)]
    pub sampling: SamplingFields,
}

/// vLLM `CompletionLogProbs`：按 token 的四列（`top_logprobs[i]` 为 `{token: logprob}` 映射）。
#[derive(Debug, Clone, Serialize, Default)]
pub struct CompletionLogProbs {
    pub text_offset: Vec<usize>,
    /// echo 时 prompt 第 0 个 token 没有 logprob → null（vLLM 同形）。
    pub token_logprobs: Vec<Option<f32>>,
    pub tokens: Vec<String>,
    pub top_logprobs: Vec<Option<serde_json::Map<String, serde_json::Value>>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<CompletionLogProbs>,
    pub finish_reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_logprobs: Option<PromptLogprobs>,
}

/// 非流式与流式同形（`object = "text_completion"`）；流式 chunk 的 `usage` 只在最后一个 chunk 出现。
#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

// ---- /tokenize /detokenize ----

/// vLLM `TokenizeCompletionRequest | TokenizeChatRequest`：`prompt` 与 `messages` 二选一。
#[derive(Debug, Clone, Deserialize)]
pub struct TokenizeRequest {
    pub model: Option<String>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub messages: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub add_generation_prompt: Option<bool>,
    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default)]
    pub return_token_strs: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenizeResponse {
    pub count: usize,
    pub max_model_len: u32,
    pub tokens: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_strs: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DetokenizeRequest {
    pub model: Option<String>,
    pub tokens: Vec<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DetokenizeResponse {
    pub prompt: String,
}
