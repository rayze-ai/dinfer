//! PD 分离 proxy（design §10.3）。对标 vLLM `examples/disaggregated/disaggregated_serving/disagg_proxy_*.py`：
//! 1. 原请求 + `kv_transfer_params.do_remote_decode` → **prefill 实例 P**（P 侧 chat.rs 强制 max_tokens=1）：
//!    拿首 token 和响应头 `x-pd-request-id`（P 的引擎 request id；P 在 finish 后把 KV 推给 D）。
//! 2. 原请求 + `kv_transfer_params.{do_remote_prefill, remote_request_id}` → **decode 实例 D**：D 配对 KV 后接力，
//!    输出不含首 token（P 已出）。
//! 3. 回客户端：流式 = P 的 chunk（去掉 finish）+ D 的全部 chunk 原样透传；非流式 = 拼 P 的首 token 文本 + D 的文本。
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
    // ---- 2. decode（有 remote_id 则接力，否则单机）
    if let Some(rid) = remote_id {
        obj.insert("kv_transfer_params".into(), json!({"do_remote_prefill": true, "remote_request_id": rid}));
    } else {
        obj.remove("kv_transfer_params");
    }
    let d = match st.http.post(format!("{}/v1/chat/completions", st.decode)).json(&Value::Object(obj.clone())).send().await {
        Ok(r) => r,
        Err(e) => return err(StatusCode::BAD_GATEWAY, format!("decode unreachable: {e}")),
    };
    if !d.status().is_success() {
        return passthrough(d).await;
    }
    let imported = d.headers().get("x-pd-imported").and_then(|v| v.to_str().ok()) == Some("1");
    let had_first = first.is_some();
    let Some(first) = first.filter(|_| imported) else {
        if had_first {
            tracing::warn!("pd proxy: decode did not import KV (timeout?), serving decode-only output");
        }
        return passthrough(d).await;
    };
    // ---- 3. 合并
    if stream {
        // 首 chunk：P 的首 token 以 SSE delta 形式先发，然后 D 的 SSE 原样透传
        let first_chunk = first_as_stream_chunk(&first);
        let head = tokio_stream::once(Ok::<_, std::io::Error>(axum::body::Bytes::from(first_chunk)));
        let tail = d.bytes_stream().map(|r| r.map_err(std::io::Error::other));
        let body = Body::from_stream(head.chain(tail));
        let mut resp = Response::new(body);
        resp.headers_mut().insert(axum::http::header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        resp.headers_mut().insert(axum::http::header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
        resp
    } else {
        let mut dv: Value = match d.json().await {
            Ok(v) => v,
            Err(e) => return err(StatusCode::BAD_GATEWAY, format!("decode body: {e}")),
        };
        merge_nonstream(&first, &mut dv);
        Json(dv).into_response()
    }
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
