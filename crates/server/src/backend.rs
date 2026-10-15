//! server 与模型 crate 之间的边界：模型 crate 实现 `ChatBackend`（渲染 prompt、编码、起请求、增量 detokenize +
//! 输出解析），server 只管 HTTP / SSE / OpenAI 字段形状。保持 server 对具体模型零依赖（构建 = core + server + 一个模型 crate）。

use anyhow::Result;
use dinfer_core::engine::EngineEvent;
use dinfer_core::request::FinishReason;
use dinfer_core::sample::SamplingParams;

/// 一次 chat 请求渲染 + 编码的结果。
pub struct Prompt {
    pub text: String,
    pub ids: Vec<u32>,
}

/// 解析后的输出增量（与 glm53 `parser::Delta` 同形，但 server 不依赖模型 crate 的类型）。
#[derive(Debug, Clone, Default)]
pub struct OutDelta {
    pub reasoning: String,
    pub content: String,
    /// (index, name 首次出现时 Some, arguments JSON 文本 Some)
    pub tool_calls: Vec<(usize, Option<String>, Option<String>)>,
}

impl OutDelta {
    pub fn is_empty(&self) -> bool {
        self.reasoning.is_empty() && self.content.is_empty() && self.tool_calls.is_empty()
    }
}

/// 每个请求一个：增量 detokenize + 格式解析的状态机。
pub trait OutputStream: Send {
    /// 喂一批 token，返回可下发的增量。
    fn push(&mut self, ids: &[u32]) -> Result<OutDelta>;
    /// 流结束：冲掉保留的尾巴。
    fn finish(&mut self) -> Result<OutDelta>;
}

pub struct RenderRequest<'a> {
    pub messages: &'a [serde_json::Value],
    pub tools: Option<&'a [serde_json::Value]>,
    pub reasoning_effort: Option<&'a str>,
    pub clear_thinking: bool,
    pub extra: Option<&'a serde_json::Map<String, serde_json::Value>>,
}

pub trait ChatBackend: Send + Sync + 'static {
    fn model_name(&self) -> &str;
    fn max_model_len(&self) -> u32;
    /// 默认采样（来自 generation_config.json），请求未指定的字段用它。
    fn default_sampling(&self) -> SamplingParams;
    fn render(&self, req: &RenderRequest<'_>) -> Result<Prompt>;
    /// 提交到引擎；返回事件接收端。
    fn submit(
        &self,
        prompt_ids: Vec<u32>,
        sampling: SamplingParams,
    ) -> tokio::sync::mpsc::UnboundedReceiver<EngineEvent>;
    fn new_output_stream(&self) -> Box<dyn OutputStream>;
}

pub fn finish_reason_str(f: FinishReason) -> &'static str {
    match f {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::Abort => "abort",
    }
}
