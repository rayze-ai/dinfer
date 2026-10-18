//! PD 分离 proxy（design §10.3）。对标 vLLM `examples/disaggregated/disaggregated_serving/disagg_proxy_*.py`：
//! 1. 原请求 + `kv_transfer_params.do_remote_decode` → **prefill 实例 P**（P 侧 chat.rs 强制 max_tokens=1）：
//!    拿首 token 和响应头 `x-pd-request-id`（P 的引擎 request id；P 在 finish 后把 KV 推给 D）。
//! 2. 原请求 + `kv_transfer_params.{do_remote_prefill, remote_request_id}` → **decode 实例 D**：D 配对 KV 后接力，
//!    输出不含首 token（P 已出）。
//! 3. 回客户端：流式 = **P 的首 chunk 立刻发**（不等 D 的响应头，D 要等 KV 到达）+ D 的全部 chunk 原样透传；
//!    非流式 = 拼 P 的首 token 文本 + D 的文本。P 的首 token id 经 `kv_transfer_params.first_token` 给 D：
//!    D 等 KV 超时本地重算时从 prompt+首 token 续写，两条路径输出都不含首 token（v2 (c)，loop q4-pd 第 14 轮）。
//! 回退：P 不可达 / 无 `x-pd-request-id` → 直接把原请求发 D 单机跑（只丢 TTFT 优势，不丢请求）。
//! 无鉴权；只应绑定内网。
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio_stream::StreamExt;

#[derive(Clone)]
pub struct ProxyState {
    pub http: reqwest::Client,
    pub prefill: String,
    pub decode: String,
}

pub async fn serve(addr: &str, prefill: String, decode: String) -> anyhow::Result<()> {
    let st = Arc::new(ProxyState {
        http: reqwest::Client::builder().pool_max_idle_per_host(64).build()?,
        prefill: prefill.trim_end_matches('/').to_string(),
        decode: decode.trim_end_matches('/').to_string(),
    });
    let app = Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok", "role": "pd-proxy"})) }))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .with_state(st);
    let l = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(l, app).await?;
    Ok(())
}

async fn models(State(st): State<Arc<ProxyState>>) -> Response {
    match st.http.get(format!("{}/v1/models", st.decode)).send().await {
        Ok(r) => passthrough(r).await,
        Err(e) => err(StatusCode::BAD_GATEWAY, format!("decode unreachable: {e}")),
    }
}

fn err(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(json!({"error": {"message": msg.into(), "type": "pd_proxy"}}))).into_response()
}

async fn passthrough(r: reqwest::Response) -> Response {
    let status = StatusCode::from_u16(r.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let ct = r.headers().get(reqwest::header::CONTENT_TYPE).cloned();
    let body = Body::from_stream(r.bytes_stream());
    let mut resp = Response::new(body);
    *resp.status_mut() = status;
    if let Some(ct) = ct {
        if let Ok(v) = HeaderValue::from_bytes(ct.as_bytes()) {
            resp.headers_mut().insert(axum::http::header::CONTENT_TYPE, v);
        }
    }
    resp
}

async fn chat(State(st): State<Arc<ProxyState>>, Json(mut req): Json<Value>) -> Response {
    let Some(obj) = req.as_object_mut() else { return err(StatusCode::BAD_REQUEST, "body must be an object") };
    let stream = obj.get("stream").and_then(Value::as_bool).unwrap_or(false);
    // ---- 1. prefill
    let mut preq = obj.clone();
    preq.insert("kv_transfer_params".into(), json!({"do_remote_decode": true}));
    preq.insert("stream".into(), Value::Bool(false)); // P 只出 1 token，非流式更省
    preq.insert("return_token_ids".into(), Value::Bool(true)); // 首 token id 要转给 D（回退时续写用）
    preq.remove("stream_options");
    let p = st.http.post(format!("{}/v1/chat/completions", st.prefill)).json(&Value::Object(preq)).send().await;
    let (remote_id, first) = match p {
        Ok(r) if r.status().is_success() => {
            let rid = r.headers().get("x-pd-request-id").and_then(|v| v.to_str().ok()).and_then(|s| s.parse::<u64>().ok());
            match (rid, r.json::<Value>().await) {
                (Some(rid), Ok(v)) => (Some(rid), Some(v)),
                (rid, res) => {
                    tracing::warn!(?rid, ok = res.is_ok(), "pd proxy: prefill reply unusable, falling back to decode-only");
                    (None, None)
                }
            }
        }
        Ok(r) => {
            tracing::warn!(status = %r.status(), "pd proxy: prefill failed, falling back to decode-only");
            (None, None)
        }
        Err(e) => {
            tracing::warn!(error = %e, "pd proxy: prefill unreachable, falling back to decode-only");
            (None, None)
        }
    };
    // P 的首 token 就是 EOS / stop：不用 D，直接把 P 的结果回客户端
    if let Some(f) = first.as_ref().filter(|f| f["choices"][0]["finish_reason"].as_str() == Some("stop")) {
        if !stream {
            return Json(f.clone()).into_response();
        }
        let body = format!("{}{}data: [DONE]\n\n", first_as_stream_chunk(f), finish_stream_chunk(f, "stop"));
        return sse_response(Body::from(body));
    }
    // ---- 2. decode（有 remote_id 则接力，否则单机）
    let first_tok = first.as_ref().and_then(|f| f["token_ids"][0].as_u64());
    match (remote_id, first_tok) {
        (Some(rid), Some(ft)) => {
            obj.insert("kv_transfer_params".into(), json!({"do_remote_prefill": true, "remote_request_id": rid, "first_token": ft}));
        }
        _ => {
            obj.remove("kv_transfer_params");
        }
    }
    let first = first.filter(|_| remote_id.is_some() && first_tok.is_some());
    let dreq = st.http.post(format!("{}/v1/chat/completions", st.decode)).json(&Value::Object(obj.clone())).send();
    // ---- 3a. 流式且有首 token：先把 P 的首 chunk 发给客户端，不等 D 的响应头（D 要等 KV 到达）；D 的 SSE 到了接着透传
    if let (true, Some(first)) = (stream, first.as_ref()) {
        let first_chunk = first_as_stream_chunk(first);
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<axum::body::Bytes, std::io::Error>>(32);
        tokio::spawn(async move {
            if tx.send(Ok(axum::body::Bytes::from(first_chunk))).await.is_err() {
                return;
            }
            let d = match dreq.await {
                Ok(d) => d,
                Err(e) => {
                    let _ = tx.send(Ok(sse_error(format!("decode unreachable: {e}")))).await;
                    return;
                }
            };
            if !d.status().is_success() {
                let status = d.status();
                let body = d.text().await.unwrap_or_default();
                let _ = tx.send(Ok(sse_error(format!("decode returned {status}: {body}")))).await;
                return;
            }
            if d.headers().get("x-pd-imported").and_then(|v| v.to_str().ok()) == Some("0") {
                tracing::warn!("pd proxy: decode did not import KV (timeout?), it recomputed locally from prompt+first token");
            }
            let mut tail = d.bytes_stream();
            while let Some(b) = tail.next().await {
                if tx.send(b.map_err(std::io::Error::other)).await.is_err() {
                    return;
                }
            }
        });
        return sse_response(Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx)));
    }
    // ---- 3b. 非流式 / 无首 token：等 D 的响应
    let d = match dreq.await {
        Ok(r) => r,
        Err(e) => return err(StatusCode::BAD_GATEWAY, format!("decode unreachable: {e}")),
    };
    if !d.status().is_success() {
        return passthrough(d).await;
    }
    if d.headers().get("x-pd-imported").and_then(|v| v.to_str().ok()) == Some("0") {
        tracing::warn!("pd proxy: decode did not import KV (timeout?), it recomputed locally from prompt+first token");
    }
    let Some(first) = first else { return passthrough(d).await };
    let mut dv: Value = match d.json().await {
        Ok(v) => v,
        Err(e) => return err(StatusCode::BAD_GATEWAY, format!("decode body: {e}")),
    };
    merge_nonstream(&first, &mut dv);
    Json(dv).into_response()
}

fn sse_response(body: Body) -> Response {
    let mut resp = Response::new(body);
    resp.headers_mut().insert(axum::http::header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    resp.headers_mut().insert(axum::http::header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    resp
}

/// 响应头已发出后 D 侧出错：按 server 同款 `event: error` 事件告知客户端并结束流（无 `[DONE]`）。
fn sse_error(msg: String) -> axum::body::Bytes {
    axum::body::Bytes::from(format!("event: error\ndata: {}\n\n", json!({"error": {"message": msg, "type": "pd_proxy"}})))
}

/// P 的非流式首 token → 一条 SSE chunk（role + content，无 finish_reason）。
fn first_as_stream_chunk(p: &Value) -> String {
    let content = p["choices"][0]["message"]["content"].as_str().unwrap_or("");
    let reasoning = p["choices"][0]["message"].get("reasoning_content").cloned();
    let mut delta = json!({"role": "assistant", "content": content});
    if let Some(r) = reasoning {
        delta["reasoning_content"] = r;
    }
    let chunk = json!({
        "id": p["id"], "object": "chat.completion.chunk", "created": p["created"], "model": p["model"],
        "choices": [{"index": 0, "delta": delta, "finish_reason": Value::Null}],
    });
    format!("data: {chunk}\n\n")
}

/// 只有 P 参与（首 token 即 EOS）时的收尾 chunk。
fn finish_stream_chunk(p: &Value, reason: &str) -> String {
    let chunk = json!({
        "id": p["id"], "object": "chat.completion.chunk", "created": p["created"], "model": p["model"],
        "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
    });
    format!("data: {chunk}\n\n")
}

/// 非流式：D 的文本前面拼上 P 的首 token 文本；usage.completion_tokens +1。
fn merge_nonstream(p: &Value, d: &mut Value) {
    let pc = p["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();
    if let Some(dc) = d["choices"][0]["message"].get_mut("content") {
        let s = format!("{pc}{}", dc.as_str().unwrap_or(""));
        *dc = Value::String(s);
    }
    for k in ["completion_tokens", "total_tokens"] {
        if let Some(n) = d["usage"].get(k).and_then(Value::as_u64) {
            d["usage"][k] = json!(n + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p_reply(content: &str, finish: &str, tok: u32) -> Value {
        json!({"id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "m",
               "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
               "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}, "token_ids": [tok]})
    }

    #[test]
    fn first_chunk_is_sse_with_role_and_content_and_no_finish() {
        let s = first_as_stream_chunk(&p_reply("Hi", "length", 42));
        assert!(s.starts_with("data: ") && s.ends_with("\n\n"), "{s}");
        let v: Value = serde_json::from_str(s.trim_start_matches("data: ").trim()).unwrap();
        assert_eq!(v["object"], "chat.completion.chunk");
        assert_eq!(v["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(v["choices"][0]["delta"]["content"], "Hi");
        assert!(v["choices"][0]["finish_reason"].is_null());
    }

    #[test]
    fn finish_chunk_and_sse_error_shapes() {
        let f = finish_stream_chunk(&p_reply("", "stop", 2), "stop");
        let v: Value = serde_json::from_str(f.trim_start_matches("data: ").trim()).unwrap();
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        let e = String::from_utf8(sse_error("boom".into()).to_vec()).unwrap();
        assert!(e.starts_with("event: error\ndata: "), "{e}");
        let ev: Value = serde_json::from_str(e.trim_start_matches("event: error\ndata: ").trim()).unwrap();
        assert_eq!(ev["error"]["message"], "boom");
    }

    #[test]
    fn merge_nonstream_prepends_first_token_and_bumps_usage() {
        let p = p_reply("Red", "length", 7);
        let mut d = json!({"choices": [{"index": 0, "message": {"role": "assistant", "content": ", green, blue"}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}});
        merge_nonstream(&p, &mut d);
        assert_eq!(d["choices"][0]["message"]["content"], "Red, green, blue");
        assert_eq!(d["usage"]["completion_tokens"], 6);
        assert_eq!(d["usage"]["total_tokens"], 9);
    }
}
