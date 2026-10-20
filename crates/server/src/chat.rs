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

/// vLLM 形状的错误：`{"error":{"message","type","param":null,"code"}}`；404 的 type 是 `NotFoundError`。
pub(crate) fn err(status: StatusCode, msg: impl Into<String>) -> Response {
    let kind = match status {
        StatusCode::NOT_FOUND => "NotFoundError",
        s if s.is_client_error() => "invalid_request_error",
        _ => "server_error",
    };
    (
        status,
        Json(ErrorResponse {
            error: ErrorBody { message: msg.into(), kind, param: None, code: status.as_u16() },
        }),
    )
        .into_response()
}

/// 请求给了 `model` 且与服务的模型名不同 → Some(404 响应)（vLLM 行为）。
pub(crate) fn check_model(be: &dyn ChatBackend, model: Option<&str>) -> Option<Response> {
    match model {
        Some(m) if m != be.model_name() => {
            Some(err(StatusCode::NOT_FOUND, format!("The model `{m}` does not exist.")))
        }
        _ => None,
    }
}

/// 请求采样字段 → `SamplingParams`。`max_tokens`：None = 填满剩余上下文；`logprobs`：Some(k) = 每 token 带 k 个 top。
pub(crate) fn sampling_from(
    req: &SamplingFields,
    defaults: SamplingParams,
    max_model_len: u32,
    prompt_len: u32,
    max_tokens: Option<u32>,
    logprobs: Option<u32>,
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
    sp.logprobs = logprobs;
    sp.prompt_logprobs = req.prompt_logprobs;
    if let Some(x) = req.skip_special_tokens {
        sp.skip_special_tokens = x;
    }
    if let Some(x) = req.spaces_between_special_tokens {
        sp.spaces_between_special_tokens = x;
    }
    if let Some(lb) = &req.logit_bias {
        let mut v = Vec::with_capacity(lb.len());
        for (k, b) in lb {
            let id: u32 = k.parse().map_err(|_| format!("logit_bias key {k:?} is not a token id"))?;
            let b = b.as_f64().ok_or_else(|| format!("logit_bias[{k}] must be a number"))? as f32;
            v.push((id, b));
        }
        sp.logit_bias = v;
    }
    let room = max_model_len.saturating_sub(prompt_len);
    if room == 0 {
        return Err(format!("prompt has {prompt_len} tokens, max_model_len is {max_model_len}"));
    }
    sp.max_tokens = max_tokens.unwrap_or(room).min(room);
    sp.validate()?;
    Ok(sp)
}

/// vLLM `truncate_prompt_tokens`：左截断到最后 N 个 token（N=0 非法）。
pub(crate) fn truncate_prompt(ids: &mut Vec<u32>, n: Option<u32>) -> Result<(), String> {
    if let Some(n) = n {
        if n == 0 {
            return Err("truncate_prompt_tokens must be >= 1".into());
        }
        let n = n as usize;
        if ids.len() > n {
            ids.drain(..ids.len() - n);
        }
    }
    Ok(())
}

/// `tool_choice` 解析结果。
#[derive(Debug, PartialEq)]
pub(crate) enum ToolChoice {
    None,
    Auto,
    Required,
    Named(String),
}

pub(crate) fn parse_tool_choice(v: Option<&serde_json::Value>, has_tools: bool) -> Result<ToolChoice, String> {
    Ok(match v {
        None | Some(serde_json::Value::Null) => {
            if has_tools {
                ToolChoice::Auto
            } else {
                ToolChoice::None
            }
        }
        Some(serde_json::Value::String(s)) => match s.as_str() {
            "none" => ToolChoice::None,
            "auto" => ToolChoice::Auto,
            "required" => ToolChoice::Required,
            other => return Err(format!("tool_choice must be none|auto|required or an object, got {other:?}")),
        },
        Some(serde_json::Value::Object(o)) => {
            if o.get("type").and_then(|t| t.as_str()) != Some("function") {
                return Err("tool_choice object must have type=function".into());
            }
            match o.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()) {
                Some(n) => ToolChoice::Named(n.to_string()),
                None => return Err("tool_choice.function.name is required".into()),
            }
        }
        Some(_) => return Err("tool_choice must be a string or an object".into()),
    })
}

/// 从 OpenAI `tools` 抽 (name, parameters)。
pub(crate) fn tool_fns(tools: &[serde_json::Value]) -> Vec<dinfer_guided::ToolFn<'_>> {
    tools
        .iter()
        .filter(|t| t.get("type").and_then(|x| x.as_str()).unwrap_or("function") == "function")
        .filter_map(|t| {
            let f = t.get("function")?;
            Some(dinfer_guided::ToolFn { name: f.get("name")?.as_str()?, parameters: f.get("parameters") })
        })
        .collect()
}

/// 把 `response_format` / `tool_choice` 编成文法；None = 不约束。错误文本直接 400。
pub(crate) fn build_grammar(
    be: &dyn ChatBackend,
    response_format: Option<&crate::openai::ResponseFormat>,
    tool_choice: &ToolChoice,
    tools: Option<&[serde_json::Value]>,
    parallel: bool,
) -> Result<Option<dinfer_core::sample::GrammarHandle>, String> {
    use dinfer_guided::GrammarSpec;
    let mut spec: Option<GrammarSpec> = None;
    if let Some(rf) = response_format {
        spec = match rf.kind.as_str() {
            "text" => None,
            "json_object" => Some(GrammarSpec::JsonObject),
            "json_schema" => {
                let js = rf.json_schema.as_ref().ok_or("response_format.json_schema is required for type=json_schema")?;
                // OpenAI：{"name","schema",...}；vLLM 也接受直接给 schema
                let schema = js.get("schema").cloned().unwrap_or_else(|| js.clone());
                if !schema.is_object() {
                    return Err("response_format.json_schema.schema must be an object".into());
                }
                Some(GrammarSpec::JsonSchema(schema))
            }
            other => return Err(format!("response_format.type must be text|json_object|json_schema, got {other:?}")),
        };
    }
    match tool_choice {
        ToolChoice::Required | ToolChoice::Named(_) => {
            if spec.is_some() {
                return Err("response_format and a constrained tool_choice cannot be combined".into());
            }
            let tools = tools.filter(|t| !t.is_empty()).ok_or("tool_choice requires non-empty `tools`")?;
            let tags = be.tool_call_tags().ok_or("constrained tool_choice is not supported by this model backend")?;
            let only = match tool_choice {
                ToolChoice::Named(n) => Some(n.as_str()),
                _ => None,
            };
            let lark = dinfer_guided::tool_call_lark(&tool_fns(tools), tags.open, tags.close, parallel, only)
                .map_err(|e| format!("tool_choice: {e:#}"))?;
            spec = Some(GrammarSpec::Lark(lark));
        }
        _ => {}
    }
    let Some(spec) = spec else { return Ok(None) };
    let f = be.grammar_factory().ok_or("structured output (response_format / tool_choice) is not supported by this backend")?;
    f.compile(&spec).map(Some).map_err(|e| format!("{e:#}"))
}

pub(crate) const MAX_N: u32 = 8;

/// 一个 choice 的流式状态。
pub(crate) struct ChoiceState {
    rid: RequestId,
    out: Box<dyn OutputStream>,
    stop: StopChecker,
    /// 本 choice 已收到但尚未随 delta 下发的 logprobs（流式）/ 累计（非流式由 Aggregate 收）。
    pending_lp: Vec<ChatLogprobContent>,
    pub(crate) completion: u32,
    pub(crate) finish: Option<&'static str>,
    /// 产出 token id（`return_token_ids` 用；只在非流式收集）。
    ids: Vec<u32>,
    /// `prompt_logprobs`：(prompt 位置, 原始 logprobs)，consume 从 Logprobs 事件里按 `prompt_pos` 分流出来。
    pub(crate) prompt_lp: Vec<(u32, TokenLogprobs)>,
    /// 该 choice 已产出 tool_call → finish_reason 报 "tool_calls"（OpenAI / vLLM 口径）。
    saw_tool_call: bool,
    /// 引擎报的 prefix cache 命中 token 数（`EngineEvent::PrefixCached`）；None = 没报。
    pub(crate) cached_tokens: Option<u32>,
}

/// 多个 choice 汇成一个 `usage.prompt_tokens_details.cached_tokens`：同一 prompt 的 n 个 choice 取最大（vLLM 取首个 choice）；
/// 多 prompt（completions）由调用方按 prompt 分组后相加。全部没报 → None。
pub(crate) fn cached_tokens_of(states: &[ChoiceState]) -> Option<u32> {
    states.iter().filter_map(|s| s.cached_tokens).max()
}

/// 多个 choice 的事件汇到一个 channel：(choice index, event)。
pub(crate) type Merged = tokio::sync::mpsc::UnboundedReceiver<(u32, EngineEvent)>;

/// handler / SSE 任务结束（含客户端断开）时取消所有未完成的引擎请求。
pub(crate) struct AbortGuard {
    be: Backend,
    live: Vec<Option<RequestId>>,
}
impl ChoiceState {
    pub(crate) fn prompt_logprobs_json(&self, prompt_len: usize) -> crate::openai::PromptLogprobs {
        prompt_logprobs_json(self.out.as_ref(), prompt_len, &self.prompt_lp)
    }
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

/// 每个 prompt 起 `n` 个引擎请求；choice index = prompt_idx × n + i（vLLM completions 口径；chat 只有一个 prompt）。
pub(crate) fn submit_choices(
    be: &Backend,
    prompts: &[&[u32]],
    sps: &[SamplingParams],
    n: u32,
    pd: Option<PdSubmit>,
) -> (Merged, Vec<ChoiceState>, AbortGuard) {
    let (mtx, merged) = tokio::sync::mpsc::unbounded_channel::<(u32, EngineEvent)>();
    let total = prompts.len() * n as usize;
    let mut states = Vec::with_capacity(total);
    let mut live = Vec::with_capacity(total);
    assert!(sps.len() == 1 || sps.len() == prompts.len(), "submit_choices: sampling per prompt");
    for (pi, prompt_ids) in prompts.iter().enumerate() {
        let sp = &sps[if sps.len() == 1 { 0 } else { pi }];
        for ci in 0..n {
            let i = pi as u32 * n + ci;
            let mut spi = sp.clone();
            if let Some(seed) = sp.seed {
                spi.seed = Some(seed.wrapping_add(ci as u64)); // n 个 choice 不同种子；无 seed 时引擎各自随机
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
                prompt_lp: Vec::new(),
                saw_tool_call: false,
                cached_tokens: None,
            });
            if !sp.skip_special_tokens {
                states.last_mut().unwrap().out.set_skip_special_tokens(false);
            }
            live.push(Some(rid));
        }
    }
    (merged, states, AbortGuard { be: be.clone(), live })
}

pub(crate) fn lp_entry(out: &dyn OutputStream, lp: &TokenLogprobs) -> ChatLogprobContent {
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

/// vLLM `prompt_logprobs` 形状：长度 = prompt_len，第 0 项 null，其余 `{"<tok_id>": {logprob, rank, decoded_token}}`
///（含 chosen 与 top-k）。缺的位置（chunked prefill 中间 chunk / prefix cache 命中）为 null。
pub(crate) fn prompt_logprobs_json(
    out: &dyn OutputStream,
    prompt_len: usize,
    raw: &[(u32, TokenLogprobs)],
) -> crate::openai::PromptLogprobs {
    let mut v: crate::openai::PromptLogprobs = vec![None; prompt_len];
    for (p, lp) in raw {
        let Some(slot) = v.get_mut(*p as usize) else { continue };
        let mut m = serde_json::Map::new();
        let text = |id: u32| out.token_text(id).unwrap_or_else(|| format!("token_id:{id}"));
        let ent = |lpv: f32, rank: u32, id: u32| {
            serde_json::json!({"logprob": lpv, "rank": rank, "decoded_token": text(id)})
        };
        for (r, (id, lpv)) in lp.top.iter().enumerate() {
            m.insert(id.to_string(), ent(*lpv, r as u32 + 1, *id));
        }
        m.entry(lp.token_id.to_string()).or_insert_with(|| ent(lp.logprob, lp.rank + 1, lp.token_id));
        *slot = Some(m);
    }
    v
}

/// 驱动所有 choice 到结束。`emit(i, delta, logprobs)` 下发增量（返回 false = 客户端已断开，停止）；
/// `finish(i, reason)` 在 choice 结束时调一次。返回 Err(msg) = 引擎错误。
/// `raw = true`（completions）：不过思考 / tool_call 解析器，detokenize 的原始文本直接进 `delta.content`。
pub(crate) async fn consume(
    merged: &mut Merged,
    states: &mut [ChoiceState],
    guard: &mut AbortGuard,
    raw: bool,
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
                let d =
                    if raw { OutDelta { content: text, ..Default::default() } } else { st.out.parse(&text) };
                st.saw_tool_call |= !d.tool_calls.is_empty();
                if !d.is_empty() && !emit(i, &d, std::mem::take(&mut st.pending_lp)) {
                    return Ok(());
                }
                if hit.is_some() {
                    guard.be.abort(st.rid);
                    end = Some("stop");
                }
            }
            EngineEvent::Logprobs(lps) => {
                for mut lp in lps {
                    for p in std::mem::take(&mut lp.prompt) {
                        if let Some(pos) = p.prompt_pos {
                            st.prompt_lp.push((pos, p));
                        }
                    }
                    if !lp.is_placeholder() {
                        st.pending_lp.push(lp_entry(st.out.as_ref(), &lp));
                    }
                }
            }
            EngineEvent::PrefixCached(n) => st.cached_tokens = Some(n),
            EngineEvent::Finished(f) => end = Some(finish_reason_str(f)),
            EngineEvent::Error(m) => return Err(m),
        }
        if let Some(reason) = end {
            let tail = st.stop.finish();
            let mut d =
                if raw { OutDelta { content: tail, ..Default::default() } } else { st.out.parse(&tail) };
            if !raw {
                if let Ok(fd) = st.out.finish() {
                    d.reasoning.push_str(&fd.reasoning);
                    d.content.push_str(&fd.content);
                    d.tool_calls.extend(fd.tool_calls);
                }
            }
            st.saw_tool_call |= !d.tool_calls.is_empty();
            let lps = std::mem::take(&mut st.pending_lp);
            if (!d.is_empty() || !lps.is_empty()) && !emit(i, &d, lps) {
                return Ok(());
            }
            // OpenAI / vLLM：有 tool_calls 且不是被截断时 finish_reason = "tool_calls"
            let reason = if st.saw_tool_call && reason == "stop" { "tool_calls" } else { reason };
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
    if let Some(r) = check_model(be.as_ref(), req.model.as_deref()) {
        return r;
    }
    if req.messages.is_empty() {
        return err(StatusCode::BAD_REQUEST, "messages must not be empty");
    }
    let n = req.n.unwrap_or(1);
    if !(1..=MAX_N).contains(&n) {
        return err(StatusCode::BAD_REQUEST, format!("n must be in 1..={MAX_N}, got {n}"));
    }
    // tool_choice / response_format → 文法（编译失败 400）；tool_choice=none 时渲染不带 tools
    let tool_choice = match parse_tool_choice(req.tool_choice.as_ref(), req.tools.as_ref().is_some_and(|t| !t.is_empty())) {
        Ok(t) => t,
        Err(m) => return err(StatusCode::BAD_REQUEST, m),
    };
    let tools_for_render = if tool_choice == ToolChoice::None { None } else { req.tools.as_deref() };
    let grammar = match build_grammar(
        be.as_ref(),
        req.response_format.as_ref(),
        &tool_choice,
        req.tools.as_deref(),
        req.parallel_tool_calls.unwrap_or(true),
    ) {
        Ok(g) => g,
        Err(m) => return err(StatusCode::BAD_REQUEST, m),
    };
    let continue_final = req.continue_final_message.unwrap_or(false);
    let add_gen = req.add_generation_prompt.unwrap_or(!continue_final);
    if continue_final && add_gen {
        return err(StatusCode::BAD_REQUEST, "continue_final_message and add_generation_prompt cannot both be true");
    }
    // 文法从第一个生成 token 起生效：带约束时按后端给的模板变量关思考段（模型相关，server 不认识具体变量名）。
    // TODO(api)：vLLM 的 reasoning-aware 结构化输出（思考段不约束）需要 parser 状态回传引擎。
    let mut extra = req.chat_template_kwargs.clone();
    if grammar.is_some() {
        let add = be.grammar_render_extra();
        if !add.is_empty() {
            let e = extra.get_or_insert_with(Default::default);
            for (k, v) in add {
                e.entry(k).or_insert(v);
            }
        }
    }
    let rr = RenderRequest {
        messages: &req.messages,
        tools: tools_for_render,
        reasoning_effort: req.reasoning_effort.as_deref(),
        clear_thinking: req.clear_thinking.unwrap_or(false),
        extra: extra.as_ref(),
        add_generation_prompt: add_gen,
        continue_final_message: continue_final,
    };
    let mut prompt = match be.render(&rr) {
        Ok(p) => p,
        Err(e) => return err(StatusCode::BAD_REQUEST, format!("chat template: {e:#}")),
    };
    if let Err(m) = truncate_prompt(&mut prompt.ids, req.sampling.truncate_prompt_tokens) {
        return err(StatusCode::BAD_REQUEST, m);
    }
    let prompt_len = prompt.ids.len() as u32;
    // OpenAI chat：`logprobs: bool` + `top_logprobs: 0..=20`
    let logprobs = if req.logprobs == Some(true) {
        Some(req.top_logprobs.unwrap_or(0))
    } else if req.top_logprobs.is_some() {
        return err(StatusCode::BAD_REQUEST, "top_logprobs requires logprobs=true");
    } else {
        None
    };
    let max_tokens = req.max_completion_tokens.or(req.max_tokens);
    let sp = match sampling_from(
        &req.sampling,
        be.default_sampling(),
        be.max_model_len(),
        prompt_len,
        max_tokens,
        logprobs,
    ) {
        Ok(mut s) => {
            s.grammar = grammar;
            s
        }
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
            None => {
                return err(
                    StatusCode::BAD_REQUEST,
                    "kv_transfer_params.remote_request_id required for do_remote_prefill",
                )
            }
        },
        _ => None,
    };
    let mut sp = sp;
    // D 侧：在 HTTP 线程里等 KV 到达（最多 pd_wait），没到就按普通请求提交（本地重算），并告诉 proxy 首 token 要自己出。
    let mut pd = pd;
    let mut pd_imported = None;
    if let Some(PdSubmit::Import { remote_id }) = pd {
        let be2 = be.clone();
        let ok = tokio::task::spawn_blocking(move || {
            be2.pd_wait_for(remote_id, std::time::Duration::from_secs(10))
        })
        .await
        .unwrap_or(false);
        pd_imported = Some(ok);
        if !ok {
            tracing::warn!(remote_id, "pd: KV did not arrive in time, serving as local prefill");
            if let Some(m) = be.metrics() {
                m.pd_fallback_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            pd = None;
            // proxy 已把 P 的首 token 发给客户端：本地重算从 prompt+首 token 续写，输出口径与接力路径一致（不含首 token）
            if let Some(ft) = req.kv_transfer_params.as_ref().and_then(|k| k.first_token) {
                prompt.ids.push(ft);
                sp.max_tokens = sp.max_tokens.saturating_sub(1).max(1);
            }
        }
    }
    if matches!(pd, Some(PdSubmit::Export)) {
        if n != 1 {
            return err(StatusCode::BAD_REQUEST, "PD prefill side requires n=1");
        }
        sp.max_tokens = 1;
    }
    let (mut merged, mut states, mut guard) = submit_choices(&be, &[&prompt.ids], std::slice::from_ref(&sp), n, pd);
    // P 侧把引擎 request id 回给 proxy（响应头），proxy 再发 D
    let pd_rid: Option<RequestId> = states.first().map(|s| s.rid);

    if req.stream {
        let include_usage = req.stream_options.as_ref().is_some_and(|o| o.include_usage);
        let stream = async_stream_lite(move |tx| async move {
            let send = |c: &ChatCompletionChunk| {
                tx.send(Event::default().data(serde_json::to_string(c).unwrap())).is_ok()
            };
            // 首 chunk（role + 空 content）**随首个 token 一起发**，不在请求接受时发：客户端 / 压测工具（vLLM `bench serve`）
            // 把首个 SSE chunk 当首 token 计 TTFT，提前发会把 TTFT 测成 5 ms 的伪像（2026-10-20 引擎层面对比发现，
            // design §15.4）。vLLM 的 serving_chat 同样是首个输出时才发 role chunk。
            let mut role_sent = vec![false; n as usize];
            let r = consume(
                &mut merged,
                &mut states,
                &mut guard,
                false,
                |i, d, lps| {
                    if !role_sent[i as usize] {
                        role_sent[i as usize] = true;
                        let d0 = Delta { role: Some("assistant"), content: Some(String::new()), ..Default::default() };
                        if !send(&chunk(&id, created, &model, i, d0, None, None)) {
                            return false;
                        }
                    }
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
                c.usage = Some(Usage::new(prompt_len, completion, cached_tokens_of(&states)));
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
        false,
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
    // prompt_logprobs：n 个 choice 同一 prompt，取第一个 choice 的（vLLM 同样只有一份）。流式不返回（vLLM 亦然）。
    let prompt_logprobs = sp.prompt_logprobs.map(|_| states[0].prompt_logprobs_json(prompt_len as usize));
    let resp = ChatCompletionResponse {
        id,
        object: "chat.completion",
        created,
        model,
        choices,
        usage: Usage::new(prompt_len, completion, cached_tokens_of(&states)),
        prompt_token_ids: return_ids.then_some(prompt.ids),
        token_ids,
        prompt_logprobs,
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
        resp.headers_mut()
            .insert("x-pd-imported", axum::http::HeaderValue::from_static(if ok { "1" } else { "0" }));
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
pub(crate) fn async_stream_lite<F, Fut>(
    f: F,
) -> impl futures_core::Stream<Item = Result<Event, Infallible>> + Send
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
