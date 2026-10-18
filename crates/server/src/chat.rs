//! `/v1/chat/completions`：请求 → 后端渲染/编码 → 引擎事件流 → 增量 detokenize + 解析 → 非流式聚合或 SSE。
//! 字段形状对齐 vLLM：首个 chunk `delta.role = "assistant"`；思考段走 `reasoning_content`，回复走 `content`；
//! tool_call 在 `</tool_call>` 闭合时一次性下发 `{index, id, type, function:{name, arguments}}`；
//! 结束 chunk `finish_reason`；`stream_options.include_usage` 时再发一个只有 `usage` 的 chunk；最后 `data: [DONE]`。
//!
//! Q2-b：`n` 个 choice = 引擎侧 n 个独立请求（seed 依次 +1），一个 handler 并发驱动、按 `index` 交错下发；
//! `stop` 字符串在 detokenize 之后、格式解析之前用 `StopChecker` 判定（只扫增量），命中即 `abort` 释放 KV；
//! `logprobs` 由 executor 一次算好（`EngineEvent::Logprobs`），这里只做 token→文本映射；客户端断开 → 全部 abort。

use crate::backend::{finish_reason_str, ChatBackend, OutDelta, OutputStream, RenderRequest};
use crate::openai::*;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use dinfer_core::engine::{EngineEvent, PdSubmit};
use dinfer_core::request::RequestId;
use dinfer_core::sample::SamplingParams;
use dinfer_core::sampler::TokenLogprobs;
use dinfer_text::StopChecker;
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
    if let Some(x) = req.min_p {
        sp.min_p = x;
    }
    if let Some(x) = req.repetition_penalty {
        sp.repetition_penalty = x;
    }
    if let Some(x) = req.frequency_penalty {
        sp.frequency_penalty = x;
    }
    if let Some(x) = req.presence_penalty {
        sp.presence_penalty = x;
    }
    if let Some(x) = &req.stop_token_ids {
        sp.stop_token_ids = x.clone();
    }
    if let Some(x) = req.ignore_eos {
        sp.ignore_eos = x;
    }
    if let Some(x) = req.min_tokens {
        sp.min_tokens = x;
    }
    if let Some(x) = req.include_stop_str_in_output {
        sp.include_stop_str_in_output = x;
    }
    sp.stop = match &req.stop {
        None | Some(serde_json::Value::Null) => Vec::new(),
        Some(serde_json::Value::String(s)) => vec![s.clone()],
        Some(serde_json::Value::Array(a)) => {
            let mut v = Vec::with_capacity(a.len());
            for x in a {
                match x.as_str() {
                    Some(s) => v.push(s.to_string()),
                    None => return Err("stop must be a string or an array of strings".into()),
                }
            }
            v
        }
        Some(_) => return Err("stop must be a string or an array of strings".into()),
    };
    if req.logprobs == Some(true) {
        sp.logprobs = Some(req.top_logprobs.unwrap_or(0));
    } else if req.top_logprobs.is_some() {
        return Err("top_logprobs requires logprobs=true".into());
    }
    let room = max_model_len.saturating_sub(prompt_len);
    if room == 0 {
        return Err(format!("prompt has {prompt_len} tokens, max_model_len is {max_model_len}"));
    }
    sp.max_tokens = req.max_completion_tokens.or(req.max_tokens).unwrap_or(room).min(room);
    sp.validate()?;
    Ok(sp)
}

const MAX_N: u32 = 8;

/// 一个 choice 的流式状态。
struct ChoiceState {
    rid: RequestId,
    out: Box<dyn OutputStream>,
    stop: StopChecker,
    /// 本 choice 已收到但尚未随 delta 下发的 logprobs（流式）/ 累计（非流式由 Aggregate 收）。
    pending_lp: Vec<ChatLogprobContent>,
    completion: u32,
    finish: Option<&'static str>,
    /// 产出 token id（`return_token_ids` 用；只在非流式收集）。
    ids: Vec<u32>,
}

/// 多个 choice 的事件汇到一个 channel：(choice index, event)。
type Merged = tokio::sync::mpsc::UnboundedReceiver<(u32, EngineEvent)>;

/// handler / SSE 任务结束（含客户端断开）时取消所有未完成的引擎请求。
struct AbortGuard {
    be: Backend,
    live: Vec<Option<RequestId>>,
}
impl AbortGuard {
    fn done(&mut self, i: u32) {
        self.live[i as usize] = None;
    }
}
impl Drop for AbortGuard {
    fn drop(&mut self) {
        for rid in self.live.iter().flatten() {
            self.be.abort(*rid);
        }
    }
}

fn submit_choices(
    be: &Backend,
    prompt_ids: &[u32],
    sp: &SamplingParams,
    n: u32,
    pd: Option<PdSubmit>,
) -> (Merged, Vec<ChoiceState>, AbortGuard) {
    let (mtx, merged) = tokio::sync::mpsc::unbounded_channel::<(u32, EngineEvent)>();
    let mut states = Vec::with_capacity(n as usize);
    let mut live = Vec::with_capacity(n as usize);
    for i in 0..n {
        let mut spi = sp.clone();
        if let Some(seed) = sp.seed {
            spi.seed = Some(seed.wrapping_add(i as u64)); // n 个 choice 不同种子；无 seed 时引擎各自随机
        }
        let (rid, mut rx) = be.submit(prompt_ids.to_vec(), spi, pd.clone());
        let mtx = mtx.clone();
        tokio::spawn(async move {
            while let Some(ev) = rx.recv().await {
                if mtx.send((i, ev)).is_err() {
                    break;
                }
            }
        });
        states.push(ChoiceState {
            rid,
            out: be.new_output_stream(),
            stop: StopChecker::new(&sp.stop, sp.include_stop_str_in_output),
            pending_lp: Vec::new(),
            completion: 0,
            finish: None,
            ids: Vec::new(),
        });
        live.push(Some(rid));
    }
    (merged, states, AbortGuard { be: be.clone(), live })
}

fn lp_entry(out: &dyn OutputStream, lp: &TokenLogprobs) -> ChatLogprobContent {
    let text = |id: u32| out.token_text(id).unwrap_or_else(|| format!("token_id:{id}"));
    let t = text(lp.token_id);
    ChatLogprobContent {
        bytes: Some(t.as_bytes().to_vec()),
        token: t,
        logprob: lp.logprob,
        top_logprobs: lp
            .top
            .iter()
            .map(|(id, l)| {
                let t = text(*id);
                ChatLogprob { bytes: Some(t.as_bytes().to_vec()), token: t, logprob: *l }
            })
            .collect(),
    }
}

/// 驱动所有 choice 到结束。`emit(i, delta, logprobs)` 下发增量（返回 false = 客户端已断开，停止）；
/// `finish(i, reason)` 在 choice 结束时调一次。返回 Err(msg) = 引擎错误。
async fn consume(
    merged: &mut Merged,
    states: &mut [ChoiceState],
    guard: &mut AbortGuard,
    mut emit: impl FnMut(u32, &OutDelta, Vec<ChatLogprobContent>) -> bool,
    mut finish: impl FnMut(u32, &'static str) -> bool,
) -> Result<(), String> {
    let mut remaining = states.len();
    while remaining > 0 {
        let Some((i, ev)) = merged.recv().await else { break };
        let st = &mut states[i as usize];
        if st.finish.is_some() {
            continue; // stop 命中 abort 后引擎回的 Finished(Abort)
        }
        let mut end: Option<&'static str> = None;
        match ev {
            EngineEvent::Tokens(t) => {
                st.completion += t.len() as u32;
                st.ids.extend_from_slice(&t);
                let text = st.out.push_text(&t).map_err(|e| format!("detokenize: {e:#}"))?;
                let (text, hit) = st.stop.push(&text);
                let d = st.out.parse(&text);
                if !d.is_empty() && !emit(i, &d, std::mem::take(&mut st.pending_lp)) {
                    return Ok(());
                }
                if hit.is_some() {
                    guard.be.abort(st.rid);
                    end = Some("stop");
                }
            }
            EngineEvent::Logprobs(lps) => {
                st.pending_lp.extend(lps.iter().map(|lp| lp_entry(st.out.as_ref(), lp)));
            }
            EngineEvent::Finished(f) => end = Some(finish_reason_str(f)),
            EngineEvent::Error(m) => return Err(m),
        }
        if let Some(reason) = end {
            let tail = st.stop.finish();
            let mut d = st.out.parse(&tail);
            if let Ok(fd) = st.out.finish() {
                d.reasoning.push_str(&fd.reasoning);
                d.content.push_str(&fd.content);
                d.tool_calls.extend(fd.tool_calls);
            }
            let lps = std::mem::take(&mut st.pending_lp);
            if (!d.is_empty() || !lps.is_empty()) && !emit(i, &d, lps) {
                return Ok(());
            }
            st.finish = Some(reason);
            guard.done(i);
            remaining -= 1;
            if !finish(i, reason) {
                return Ok(());
            }
        }
    }
    Ok(())
}

pub async fn chat_completions(State(be): State<Backend>, Json(req): Json<ChatCompletionRequest>) -> Response {
    if req.messages.is_empty() {
        return err(StatusCode::BAD_REQUEST, "messages must not be empty");
    }
    let n = req.n.unwrap_or(1);
    if !(1..=MAX_N).contains(&n) {
        return err(StatusCode::BAD_REQUEST, format!("n must be in 1..={MAX_N}, got {n}"));
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
    let return_ids = req.return_token_ids.unwrap_or(false);
    let want_lp = sp.logprobs.is_some();
    // PD 分离：P 侧强制 max_tokens=1（只出首 token）且 n=1；D 侧按 remote id 配对。
    let pd = match req.kv_transfer_params.as_ref() {
        Some(k) if k.do_remote_decode => Some(PdSubmit::Export),
        Some(k) if k.do_remote_prefill => match k.remote_request_id {
            Some(r) => Some(PdSubmit::Import { remote_id: r }),
            None => return err(StatusCode::BAD_REQUEST, "kv_transfer_params.remote_request_id required for do_remote_prefill"),
        },
        _ => None,
    };
    let mut sp = sp;
    // D 侧：在 HTTP 线程里等 KV 到达（最多 pd_wait），没到就按普通请求提交（本地重算），并告诉 proxy 首 token 要自己出。
    let mut pd = pd;
    let mut pd_imported = None;
    if let Some(PdSubmit::Import { remote_id }) = pd {
        let be2 = be.clone();
        let ok = tokio::task::spawn_blocking(move || be2.pd_wait_for(remote_id, std::time::Duration::from_secs(10))).await.unwrap_or(false);
        pd_imported = Some(ok);
        if !ok {
            tracing::warn!(remote_id, "pd: KV did not arrive in time, serving as local prefill");
            if let Some(m) = be.metrics() {
                m.pd_fallback_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            pd = None;
        }
    }
    if matches!(pd, Some(PdSubmit::Export)) {
        if n != 1 {
            return err(StatusCode::BAD_REQUEST, "PD prefill side requires n=1");
        }
        sp.max_tokens = 1;
    }
    let (mut merged, mut states, mut guard) = submit_choices(&be, &prompt.ids, &sp, n, pd);
    // P 侧把引擎 request id 回给 proxy（响应头），proxy 再发 D
    let pd_rid: Option<RequestId> = states.first().map(|s| s.rid);

    if req.stream {
        let include_usage = req.stream_options.as_ref().is_some_and(|o| o.include_usage);
        let stream = async_stream_lite(move |tx| async move {
            let send = |c: &ChatCompletionChunk| {
                tx.send(Event::default().data(serde_json::to_string(c).unwrap())).is_ok()
            };
            for i in 0..n {
                // 首 chunk：role
                let d = Delta { role: Some("assistant"), content: Some(String::new()), ..Default::default() };
                if !send(&chunk(&id, created, &model, i, d, None, None)) {
                    return;
                }
            }
            let r = consume(
                &mut merged,
                &mut states,
                &mut guard,
                |i, d, lps| {
                    let lp = want_lp.then_some(ChatLogprobs { content: lps });
                    send(&chunk(&id, created, &model, i, delta_from(d), None, lp))
                },
                |i, reason| send(&chunk(&id, created, &model, i, Delta::default(), Some(reason), None)),
            )
            .await;
            if let Err(m) = r {
                let _ = tx.send(Event::default().event("error").data(m));
                return;
            }
            if include_usage {
                let completion: u32 = states.iter().map(|s| s.completion).sum();
                let mut c = chunk(&id, created, &model, 0, Delta::default(), None, None);
                c.choices.clear();
                c.usage = Some(Usage {
                    prompt_tokens: prompt_len,
                    completion_tokens: completion,
                    total_tokens: prompt_len + completion,
                });
                let _ = send(&c);
            }
            let _ = tx.send(Event::default().data("[DONE]"));
            // guard 在这里 drop：未完成的 choice（客户端断开）全部 abort
        });
        let mut resp = Sse::new(stream).keep_alive(KeepAlive::default()).into_response();
        pd_header(&mut resp, pd_rid, pd_imported);
        return resp;
    }

    // 非流式：聚合
    let mut aggs: Vec<Aggregate> = (0..n).map(|_| Aggregate::default()).collect();
    let r = consume(
        &mut merged,
        &mut states,
        &mut guard,
        |i, d, lps| {
            let a = &mut aggs[i as usize];
            a.push(d);
            a.logprobs.extend(lps);
            true
        },
        |_, _| true,
    )
    .await;
    if let Err(m) = r {
        return err(StatusCode::INTERNAL_SERVER_ERROR, m);
    }
    let completion: u32 = states.iter().map(|s| s.completion).sum();
    let choices = aggs
        .into_iter()
        .zip(states.iter())
        .enumerate()
        .map(|(i, (a, st))| {
            let lp = want_lp.then(|| ChatLogprobs { content: a.logprobs.clone() });
            Choice {
                index: i as u32,
                message: a.into_message(),
                finish_reason: Some(st.finish.unwrap_or("stop")),
                logprobs: lp,
            }
        })
        .collect();
    // 单 choice 时沿用 Q1 的 token_ids 字段；多 choice 没有单一序列
    let token_ids = (return_ids && n == 1).then(|| std::mem::take(&mut states[0].ids));
    let resp = ChatCompletionResponse {
        id,
        object: "chat.completion",
        created,
        model,
        choices,
        usage: Usage {
            prompt_tokens: prompt_len,
            completion_tokens: completion,
            total_tokens: prompt_len + completion,
        },
        prompt_token_ids: return_ids.then_some(prompt.ids),
        token_ids,
    };
    let mut r = Json(resp).into_response();
    pd_header(&mut r, pd_rid, pd_imported);
    r
}

/// P 侧：把引擎 request id 放进 `x-pd-request-id`，proxy 读它转给 D。
fn pd_header(resp: &mut Response, rid: Option<RequestId>, imported: Option<bool>) {
    if let Some(r) = rid {
        if let Ok(v) = axum::http::HeaderValue::from_str(&r.to_string()) {
            resp.headers_mut().insert("x-pd-request-id", v);
        }
    }
    // D 侧：1 = 接力了 P 的 KV（首 token 已由 P 出，本响应不含它）；0 = 本地重算（本响应含首 token）
    if let Some(ok) = imported {
        resp.headers_mut().insert("x-pd-imported", axum::http::HeaderValue::from_static(if ok { "1" } else { "0" }));
    }
}

#[derive(Default)]
struct Aggregate {
    reasoning: String,
    content: String,
    tool_calls: Vec<ToolCall>,
    logprobs: Vec<ChatLogprobContent>,
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
    index: u32,
    delta: Delta,
    finish: Option<&'static str>,
    logprobs: Option<ChatLogprobs>,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice { index, delta, finish_reason: finish, logprobs }],
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
