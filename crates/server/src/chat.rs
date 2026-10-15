//! `/v1/chat/completions`：请求 → 后端渲染/编码 → 引擎事件流 → 增量 detokenize + 解析 → 非流式聚合或 SSE。
//! 字段形状对齐 vLLM：首个 chunk `delta.role = "assistant"`；思考段走 `reasoning_content`，回复走 `content`；
//! tool_call 在 `</tool_call>` 闭合时一次性下发 `{index, id, type, function:{name, arguments}}`；
//! 结束 chunk `finish_reason`；`stream_options.include_usage` 时再发一个只有 `usage` 的 chunk；最后 `data: [DONE]`。

use crate::backend::{finish_reason_str, ChatBackend, OutDelta, RenderRequest};
use crate::openai::*;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use dinfer_core::engine::EngineEvent;
use dinfer_core::request::FinishReason;
use dinfer_core::sample::SamplingParams;
use std::convert::Infallible;
use std::sync::Arc;

pub type Backend = Arc<dyn ChatBackend>;

fn err(status: StatusCode, msg: impl Into<String>) -> Response {
    let kind = if status.is_client_error() { "invalid_request_error" } else { "server_error" };
    (status, Json(ErrorResponse { error: ErrorBody { message: msg.into(), kind, code: status.as_u16() } }))
        .into_response()
}

fn sampling_from(
    req: &ChatCompletionRequest,
    defaults: SamplingParams,
    max_model_len: u32,
    prompt_len: u32,
) -> Result<SamplingParams, String> {
    let mut sp = defaults;
    if let Some(t) = req.temperature {
        sp.temperature = t;
    }
    if let Some(p) = req.top_p {
        sp.top_p = p;
    }
    if let Some(k) = req.top_k {
        sp.top_k = k;
    }
    if req.seed.is_some() {
        sp.seed = req.seed;
    }
    let room = max_model_len.saturating_sub(prompt_len);
    if room == 0 {
        return Err(format!("prompt has {prompt_len} tokens, max_model_len is {max_model_len}"));
    }
    sp.max_tokens = req.max_completion_tokens.or(req.max_tokens).unwrap_or(room).min(room);
    if matches!(req.n, Some(n) if n != 1) {
        return Err("n != 1 is not supported".into());
    }
    Ok(sp)
}

pub async fn chat_completions(State(be): State<Backend>, Json(req): Json<ChatCompletionRequest>) -> Response {
    if req.messages.is_empty() {
        return err(StatusCode::BAD_REQUEST, "messages must not be empty");
    }
    let rr = RenderRequest {
        messages: &req.messages,
        tools: req.tools.as_deref(),
        reasoning_effort: req.reasoning_effort.as_deref(),
        clear_thinking: req.clear_thinking.unwrap_or(false),
        extra: req.chat_template_kwargs.as_ref(),
    };
    let prompt = match be.render(&rr) {
        Ok(p) => p,
        Err(e) => return err(StatusCode::BAD_REQUEST, format!("chat template: {e:#}")),
    };
    let prompt_len = prompt.ids.len() as u32;
    let sp = match sampling_from(&req, be.default_sampling(), be.max_model_len(), prompt_len) {
        Ok(s) => s,
        Err(m) => return err(StatusCode::BAD_REQUEST, m),
    };
    let model = req.model.clone().unwrap_or_else(|| be.model_name().to_string());
    let id = new_id("chatcmpl");
    let created = now_secs();
    let rx = be.submit(prompt.ids.clone(), sp);
    let out = be.new_output_stream();
    let return_ids = req.return_token_ids.unwrap_or(false);

    if req.stream {
        let include_usage = req.stream_options.as_ref().is_some_and(|o| o.include_usage);
        let stream = stream_chunks(rx, out, id, created, model, prompt_len, include_usage);
        return Sse::new(stream).keep_alive(KeepAlive::default()).into_response();
    }

    // 非流式：聚合
    let mut agg = Aggregate::default();
    let mut rx = rx;
    let mut out = out;
    let mut finish: Option<FinishReason> = None;
    let mut all_ids = Vec::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            EngineEvent::Tokens(t) => {
                all_ids.extend_from_slice(&t);
                match out.push(&t) {
                    Ok(d) => agg.push(&d),
                    Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, format!("detokenize: {e:#}")),
                }
            }
            EngineEvent::Finished(f) => {
                finish = Some(f);
                break;
            }
            EngineEvent::Error(m) => return err(StatusCode::INTERNAL_SERVER_ERROR, m),
        }
    }
    if let Ok(d) = out.finish() {
        agg.push(&d);
    }
    let completion = all_ids.len() as u32;
    let resp = ChatCompletionResponse {
        id,
        object: "chat.completion",
        created,
        model,
        choices: vec![Choice {
            index: 0,
            message: agg.into_message(),
            finish_reason: Some(finish.map(finish_reason_str).unwrap_or("stop")),
        }],
        usage: Usage {
            prompt_tokens: prompt_len,
            completion_tokens: completion,
            total_tokens: prompt_len + completion,
        },
        prompt_token_ids: return_ids.then_some(prompt.ids),
        token_ids: return_ids.then_some(all_ids),
    };
    Json(resp).into_response()
}

#[derive(Default)]
struct Aggregate {
    reasoning: String,
    content: String,
    tool_calls: Vec<ToolCall>,
}

impl Aggregate {
    fn push(&mut self, d: &OutDelta) {
        self.reasoning.push_str(&d.reasoning);
        self.content.push_str(&d.content);
        for (_, name, args) in &d.tool_calls {
            self.tool_calls.push(ToolCall {
                id: new_id("call"),
                kind: "function",
                function: FunctionCall {
                    name: name.clone().unwrap_or_default(),
                    arguments: args.clone().unwrap_or_default(),
                },
            });
        }
    }
    fn into_message(self) -> ChatMessage {
        ChatMessage {
            role: "assistant",
            // OpenAI：有 tool_calls 且无文本时 content 为 null；否则给字符串（可为空）
            content: if self.content.is_empty() && !self.tool_calls.is_empty() {
                None
            } else {
                Some(self.content)
            },
            reasoning_content: if self.reasoning.is_empty() { None } else { Some(self.reasoning) },
            tool_calls: if self.tool_calls.is_empty() { None } else { Some(self.tool_calls) },
        }
    }
}

fn chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: Delta,
    finish: Option<&'static str>,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice { index: 0, delta, finish_reason: finish }],
        usage: None,
    }
}

fn delta_from(d: &OutDelta) -> Delta {
    Delta {
        role: None,
        content: if d.content.is_empty() { None } else { Some(d.content.clone()) },
        reasoning_content: if d.reasoning.is_empty() { None } else { Some(d.reasoning.clone()) },
        tool_calls: if d.tool_calls.is_empty() {
            None
        } else {
            Some(
                d.tool_calls
                    .iter()
                    .map(|(i, name, args)| DeltaToolCall {
                        index: *i,
                        id: name.as_ref().map(|_| new_id("call")),
                        kind: name.as_ref().map(|_| "function"),
                        function: DeltaToolCallFunction { name: name.clone(), arguments: args.clone() },
                    })
                    .collect(),
            )
        },
    }
}

fn stream_chunks(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<EngineEvent>,
    mut out: Box<dyn crate::backend::OutputStream>,
    id: String,
    created: u64,
    model: String,
    prompt_len: u32,
    include_usage: bool,
) -> impl futures_core::Stream<Item = Result<Event, Infallible>> + Send {
    async_stream_lite(move |tx| async move {
        let send = |tx: &tokio::sync::mpsc::UnboundedSender<Event>, c: &ChatCompletionChunk| {
            let _ = tx.send(Event::default().data(serde_json::to_string(c).unwrap()));
        };
        // 首 chunk：role
        let first = chunk(
            &id,
            created,
            &model,
            Delta { role: Some("assistant"), content: Some(String::new()), ..Default::default() },
            None,
        );
        send(&tx, &first);
        let mut completion = 0u32;
        let mut finish: Option<FinishReason> = None;
        while let Some(ev) = rx.recv().await {
            match ev {
                EngineEvent::Tokens(t) => {
                    completion += t.len() as u32;
                    if let Ok(d) = out.push(&t) {
                        if !d.is_empty() {
                            send(&tx, &chunk(&id, created, &model, delta_from(&d), None));
                        }
                    }
                }
                EngineEvent::Finished(f) => {
                    finish = Some(f);
                    break;
                }
                EngineEvent::Error(m) => {
                    let _ = tx.send(Event::default().event("error").data(m));
                    break;
                }
            }
        }
        if let Ok(d) = out.finish() {
            if !d.is_empty() {
                send(&tx, &chunk(&id, created, &model, delta_from(&d), None));
            }
        }
        send(
            &tx,
            &chunk(
                &id,
                created,
                &model,
                Delta::default(),
                Some(finish.map(finish_reason_str).unwrap_or("stop")),
            ),
        );
        if include_usage {
            let mut c = chunk(&id, created, &model, Delta::default(), None);
            c.choices.clear();
            c.usage = Some(Usage {
                prompt_tokens: prompt_len,
                completion_tokens: completion,
                total_tokens: prompt_len + completion,
            });
            send(&tx, &c);
        }
        let _ = tx.send(Event::default().data("[DONE]"));
    })
}

/// 用一个 unbounded channel + spawn 把 async 生产者变成 Stream（避免引入 async-stream 宏 crate）。
fn async_stream_lite<F, Fut>(f: F) -> impl futures_core::Stream<Item = Result<Event, Infallible>> + Send
where
    F: FnOnce(tokio::sync::mpsc::UnboundedSender<Event>) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    tokio::spawn(f(tx));
    tokio_stream::wrappers::UnboundedReceiverStream::new(rx).map_ok()
}

trait MapOk: Sized {
    fn map_ok(self) -> MapOkStream<Self>;
}
impl<S> MapOk for S
where
    S: futures_core::Stream<Item = Event>,
{
    fn map_ok(self) -> MapOkStream<Self> {
        MapOkStream(self)
    }
}
pub struct MapOkStream<S>(S);
impl<S: futures_core::Stream<Item = Event> + Unpin> futures_core::Stream for MapOkStream<S> {
    type Item = Result<Event, Infallible>;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::new(&mut self.0).poll_next(cx).map(|o| o.map(Ok))
    }
}
