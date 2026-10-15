//! GLM-5.3 chat 模板 = 通用 `dinfer_text::ChatTemplate` + GLM 的模板变量（`reasoning_effort` / `clear_thinking`）。
//! 对齐口径与方言补丁见 `dinfer_text::chat`。

use anyhow::Result;
pub use dinfer_text::ChatTemplate;

/// GLM 模板参数（映射到 `dinfer_text::RenderParams::extra`）。
#[derive(Debug, Clone, Default)]
pub struct RenderParams<'a> {
    pub messages: &'a [serde_json::Value],
    pub tools: Option<&'a [serde_json::Value]>,
    /// `low` / `high`；其他值（含 None）模板按 `max` 处理。
    pub reasoning_effort: Option<&'a str>,
    /// true：只保留最后一个 user 之后的 assistant 思考内容。
    pub clear_thinking: bool,
    pub add_generation_prompt: bool,
}

pub fn render(t: &ChatTemplate, p: &RenderParams<'_>) -> Result<String> {
    let mut extra = serde_json::Map::new();
    extra.insert("clear_thinking".into(), serde_json::Value::Bool(p.clear_thinking));
    extra.insert(
        "reasoning_effort".into(),
        p.reasoning_effort
            .map(|s| serde_json::Value::String(s.to_string()))
            .unwrap_or(serde_json::Value::Null),
    );
    t.render(&dinfer_text::RenderParams {
        messages: p.messages,
        tools: p.tools,
        add_generation_prompt: p.add_generation_prompt,
        extra: &extra,
    })
}
