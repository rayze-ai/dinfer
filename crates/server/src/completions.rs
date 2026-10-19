//! `/v1/completions`：文本补全（vLLM `serving_completion.py` 口径）。
//! `prompt` 四种形态：字符串 / 字符串数组 / token id 数组 / token id 数组的数组；多 prompt × `n` 时
//! choice index = prompt_idx × n + i。输出**不过**思考 / tool_call 解析器（`consume(raw = true)`），detokenize 原文直出。
//! `echo` 把 prompt 文本拼在该 choice 首个增量之前（流式与非流式同形）；`logprobs: k` = 每 token 带 k 个 top
//! （k = 0 仍返回 chosen token 的 logprob），响应形状 `{tokens, token_logprobs, top_logprobs, text_offset}`。
//! 流式 chunk `object = "text_completion"`，`stream_options.include_usage` 时最后补一个只有 `usage` 的 chunk。

use crate::backend::ChatBackend;
use crate::chat::{async_stream_lite, check_model, consume, err, sampling_from, submit_choices, truncate_prompt, MAX_N};
use crate::openai::*;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use std::sync::Arc;

type Backend = Arc<dyn ChatBackend>;

/// `prompt` 字段 → 每个 prompt 的 token ids（字符串走 `encode`）。
fn parse_prompts(be: &dyn ChatBackend, v: &serde_json::Value) -> Result<Vec<Vec<u32>>, String> {
    let ids_of = |a: &[serde_json::Value]| -> Result<Vec<u32>, String> {
        a.iter()
            .map(|x| {
                x.as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| "prompt token ids must be non-negative integers".to_string())
            })
            .collect()
    };
    let enc = |s: &str| be.encode(s).map_err(|e| format!("tokenize: {e:#}"));
    match v {
        serde_json::Value::String(s) => Ok(vec![enc(s)?]),
        serde_json::Value::Array(a) if a.is_empty() => Err("prompt must not be empty".into()),
        serde_json::Value::Array(a) => match &a[0] {
            serde_json::Value::String(_) => a
                .iter()
                .map(|x| {
                    x.as_str().ok_or_else(|| "prompt array must be all strings".to_string()).and_then(enc)
                })
                .collect(),
            serde_json::Value::Number(_) => Ok(vec![ids_of(a)?]),
            serde_json::Value::Array(_) => a
                .iter()
                .map(|x| {
                    x.as_array()
                        .ok_or_else(|| "prompt must be an array of token id arrays".to_string())
                        .and_then(|x| ids_of(x))
                })
                .collect(),
            _ => Err(
                "prompt must be a string, array of strings, array of token ids, or array of token id arrays"
                    .into(),
            ),
        },
        _ => {
            Err("prompt must be a string, array of strings, array of token ids, or array of token id arrays"
                .into())
        }
    }
}

/// chat 形状的 logprobs → vLLM completions 四列；`offset` 为该 choice 的文本游标（echo 时从 prompt 长度起）。
fn to_completion_lp(lps: &[ChatLogprobContent], offset: &mut usize) -> CompletionLogProbs {
    let mut out = CompletionLogProbs::default();
    for lp in lps {
        out.text_offset.push(*offset);
        *offset += lp.token.len();
        out.token_logprobs.push(Some(lp.logprob));
        out.tokens.push(lp.token.clone());
        let mut top = serde_json::Map::new();
        for t in &lp.top_logprobs {
            top.insert(t.token.clone(), serde_json::json!(t.logprob));
        }
        // vLLM：chosen token 不在 top-k 时也放进去（所以长度可能 = k + 1）
        top.entry(lp.token.clone()).or_insert(serde_json::json!(lp.logprob));
        out.top_logprobs.push(Some(top));
    }
    out
}

/// echo + logprobs 的 prompt 段（vLLM：第 0 个 token `null`，其余来自 prompt_logprobs；缺的位置也 null）。
/// `offset` 走完 prompt 文本。
fn prompt_segment_lp(
    be: &dyn ChatBackend,
    prompt: &[u32],
    plp: &crate::openai::PromptLogprobs,
    offset: &mut usize,
) -> CompletionLogProbs {
    let mut out = CompletionLogProbs::default();
    for (i, &id) in prompt.iter().enumerate() {
        let tok = be.decode(&[id]).unwrap_or_else(|_| format!("token_id:{id}"));
        out.text_offset.push(*offset);
        *offset += tok.len();
        match plp.get(i).and_then(|x| x.as_ref()) {
            Some(m) => {
                let mine = m.get(&id.to_string()).and_then(|e| e["logprob"].as_f64()).map(|x| x as f32);
                out.token_logprobs.push(mine);
                let mut top = serde_json::Map::new();
                for (k, e) in m {
                    let t = k.parse::<u32>().ok().and_then(|t| be.decode(&[t]).ok()).unwrap_or_else(|| k.clone());
                    top.insert(t, e["logprob"].clone());
                }
                out.top_logprobs.push(Some(top));
            }
            None => {
                out.token_logprobs.push(None);
                out.top_logprobs.push(None);
            }
        }
        out.tokens.push(tok);
    }
    out
}

fn merge_lp(dst: &mut CompletionLogProbs, src: CompletionLogProbs) {
    dst.text_offset.extend(src.text_offset);
    dst.token_logprobs.extend(src.token_logprobs);
    dst.tokens.extend(src.tokens);
    dst.top_logprobs.extend(src.top_logprobs);
}

pub async fn completions(State(be): State<Backend>, Json(req): Json<CompletionRequest>) -> Response {
    if let Some(r) = check_model(be.as_ref(), req.model.as_deref()) {
        return r;
    }
    if req.suffix.is_some() {
        return err(StatusCode::BAD_REQUEST, "suffix is not supported");
    }
    let n = req.n.unwrap_or(1);
    if !(1..=MAX_N).contains(&n) {
        return err(StatusCode::BAD_REQUEST, format!("n must be in 1..={MAX_N}, got {n}"));
    }
    let mut prompts = match parse_prompts(be.as_ref(), &req.prompt) {
        Ok(p) => p,
        Err(m) => return err(StatusCode::BAD_REQUEST, m),
    };
    if prompts.iter().any(|p| p.is_empty()) {
        return err(StatusCode::BAD_REQUEST, "prompt must not be empty");
    }
    for p in &mut prompts {
        if let Err(m) = truncate_prompt(p, req.sampling.truncate_prompt_tokens) {
            return err(StatusCode::BAD_REQUEST, m);
        }
    }
    let echo = req.echo.unwrap_or(false);
    // 逐 prompt 一份 SamplingParams：max_tokens 的上下文余量按各自 prompt 算（vLLM 口径）。
    // echo + logprobs：vLLM 会把 prompt 段的 logprobs 列进来 → 用 prompt_logprobs（请求没给就按 logprobs 的 k）。
    let max_tokens = Some(req.max_tokens.unwrap_or(16));
    let mut sps = Vec::with_capacity(prompts.len());
    for p in &prompts {
        match sampling_from(&req.sampling, be.default_sampling(), be.max_model_len(), p.len() as u32, max_tokens, req.logprobs) {
            Ok(mut s) => {
                if echo && s.logprobs.is_some() && s.prompt_logprobs.is_none() {
                    s.prompt_logprobs = s.logprobs;
                }
                sps.push(s)
            }
            Err(m) => return err(StatusCode::BAD_REQUEST, m),
        }
    }
    let sp = sps[0].clone();
    // echo：每个 choice 前缀 = 其 prompt 的 decode（字符串 prompt 时等于原文）
    let echo_text: Vec<String> = if echo {
        let mut v = Vec::with_capacity(prompts.len());
        for p in &prompts {
            match be.decode(p) {
                Ok(t) => v.push(t),
                Err(e) => return err(StatusCode::BAD_REQUEST, format!("detokenize prompt: {e:#}")),
            }
        }
        v
    } else {
        Vec::new()
    };
    let model = req.model.clone().unwrap_or_else(|| be.model_name().to_string());
    let id = new_id("cmpl");
    let created = now_secs();
    let want_lp = sp.logprobs.is_some();
    let prompt_tokens: u32 = prompts.iter().map(|p| p.len() as u32).sum();
    let prompt_refs: Vec<&[u32]> = prompts.iter().map(|p| p.as_slice()).collect();
    let (mut merged, mut states, mut guard) = submit_choices(&be, &prompt_refs, &sps, n, None);
    let total = states.len();
    // 每个 choice 的 echo 前缀（未下发时 Some）与 logprobs 文本游标
    // Mutex（流式 future 需 Send）：emit / finish 两个闭包都要 take 它
    let echo_pending: std::sync::Mutex<Vec<Option<String>>> =
        std::sync::Mutex::new((0..total).map(|i| echo.then(|| echo_text[i / n as usize].clone())).collect());
    let mut offsets: Vec<usize> =
        (0..total).map(|i| if echo { echo_text[i / n as usize].len() } else { 0 }).collect();

    if req.stream {
        let include_usage = req.stream_options.as_ref().is_some_and(|o| o.include_usage);
        let stream = async_stream_lite(move |tx| async move {
            let send = |c: &CompletionResponse| {
                tx.send(Event::default().data(serde_json::to_string(c).unwrap())).is_ok()
            };
            let mk = |i: u32, text: String, lp: Option<CompletionLogProbs>, finish: Option<&'static str>| {
                CompletionResponse {
                    id: id.clone(),
                    object: "text_completion",
                    created,
                    model: model.clone(),
                    choices: vec![CompletionChoice { index: i, text, logprobs: lp, finish_reason: finish, prompt_logprobs: None }],
                    usage: None,
                }
            };
            let r = consume(
                &mut merged,
                &mut states,
                &mut guard,
                true,
                |i, d, lps| {
                    let mut text = echo_pending.lock().unwrap()[i as usize].take().unwrap_or_default();
                    text.push_str(&d.content);
                    let lp = want_lp.then(|| to_completion_lp(&lps, &mut offsets[i as usize]));
                    send(&mk(i, text, lp, None))
                },
                |i, reason| {
                    // 首个增量为空就结束的 choice：echo 前缀随结束 chunk 下发
                    let text = echo_pending.lock().unwrap()[i as usize].take().unwrap_or_default();
                    send(&mk(i, text, None, Some(reason)))
                },
            )
            .await;
            if let Err(m) = r {
                let _ = tx.send(Event::default().event("error").data(m));
                return;
            }
            if include_usage {
                let completion: u32 = states.iter().map(|s| s.completion).sum();
                let mut c = mk(0, String::new(), None, None);
                c.choices.clear();
                c.usage = Some(Usage {
                    prompt_tokens,
                    completion_tokens: completion,
                    total_tokens: prompt_tokens + completion,
                });
                let _ = send(&c);
            }
            let _ = tx.send(Event::default().data("[DONE]"));
        });
        return Sse::new(stream).keep_alive(KeepAlive::default()).into_response();
    }

    // 非流式：聚合
    let mut texts: Vec<String> =
        echo_pending.into_inner().unwrap().into_iter().map(|e| e.unwrap_or_default()).collect();
    let mut lps: Vec<CompletionLogProbs> = (0..total).map(|_| CompletionLogProbs::default()).collect();
    let r = consume(
        &mut merged,
        &mut states,
        &mut guard,
        true,
        |i, d, l| {
            texts[i as usize].push_str(&d.content);
            if want_lp {
                let part = to_completion_lp(&l, &mut offsets[i as usize]);
                merge_lp(&mut lps[i as usize], part);
            }
            true
        },
        |_, _| true,
    )
    .await;
    if let Err(m) = r {
        return err(StatusCode::INTERNAL_SERVER_ERROR, m);
    }
    let completion: u32 = states.iter().map(|s| s.completion).sum();
    let choices = texts
        .into_iter()
        .zip(lps)
        .zip(states.iter())
        .enumerate()
        .map(|(i, ((text, lp), st))| {
            let pi = i / n as usize;
            let plp = sps[pi].prompt_logprobs.map(|_| st.prompt_logprobs_json(prompts[pi].len()));
            let lp = want_lp.then(|| match (&plp, echo) {
                (Some(p), true) => {
                    let mut off = 0usize;
                    let mut seg = prompt_segment_lp(be.as_ref(), &prompts[pi], p, &mut off);
                    merge_lp(&mut seg, lp);
                    seg
                }
                _ => lp,
            });
            CompletionChoice {
                index: i as u32,
                text,
                logprobs: lp,
                finish_reason: Some(st.finish.unwrap_or("stop")),
                // vLLM：请求显式给了 prompt_logprobs 才单列（echo 隐式开的只进 logprobs 四列）
                prompt_logprobs: req.sampling.prompt_logprobs.and(plp),
            }
        })
        .collect();
    Json(CompletionResponse {
        id,
        object: "text_completion",
        created,
        model,
        choices,
        usage: Some(Usage {
            prompt_tokens,
            completion_tokens: completion,
            total_tokens: prompt_tokens + completion,
        }),
    })
    .into_response()
}
