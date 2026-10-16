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
}

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
