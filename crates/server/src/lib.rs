//! dinfer-server：axum 路由。`/health`、`/metrics`（Prometheus）、`/v1/models[/{id}]`、`/v1/chat/completions`、
//! `/v1/completions`（均流式 + 非流式）、`/tokenize`、`/detokenize`、`/version`、`/ping`；pooling / 音频 / responses 系列 501。
//! 模型通过 `backend::ChatBackend` 接入（见 glm53 `service.rs`）；server 本身不依赖任何模型 crate。
//!
//! 安全：无鉴权。默认只绑 127.0.0.1；对外暴露必须放在网关后面。

#![forbid(unsafe_code)]

pub mod backend;
pub mod chat;
pub mod completions;
pub mod dp;
pub mod misc;
pub mod openai;
pub mod proxy;

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use backend::ChatBackend;
use serde::Serialize;
use std::sync::Arc;

#[derive(Serialize)]
struct Health {
    status: &'static str,
}

/// 绑定地址并一直服务到进程退出。
pub async fn serve(addr: &str, backend: Arc<dyn ChatBackend>) -> anyhow::Result<()> {
    let app = router(backend);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "listening");
    // SSE 的首个 role chunk 与首个内容 chunk 都是几十字节的小写入：默认 Nagle + 对端延迟 ACK 会把第二个小段攒
    // ~40 ms，直接体现为 TTFT（bench 实测 43 ms，engine 内只有 5 ms）。uvicorn 默认 TCP_NODELAY，这里对齐。
    let listener = axum::serve::ListenerExt::tap_io(listener, |io: &mut tokio::net::TcpStream| {
        let _ = io.set_nodelay(true);
    });
    axum::serve(listener, app).await?;
    Ok(())
}

/// Prometheus 指标名前缀（与 vLLM 的 `vllm:` 同位置）。
pub const METRICS_PREFIX: &str = "dinfer:";

pub fn router(backend: Arc<dyn ChatBackend>) -> Router {
    let mut r = Router::new()
        .route("/health", get(|| async { Json(Health { status: "ok" }) }))
        .route("/metrics", get(metrics))
        .route("/version", get(misc::version))
        .route("/ping", get(misc::ping).post(misc::ping))
        .route("/v1/models", get(misc::models))
        .route("/v1/models/{id}", get(misc::model_by_id))
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/v1/completions", post(completions::completions))
        .route("/tokenize", post(misc::tokenize))
        .route("/detokenize", post(misc::detokenize));
    // vLLM 有、本项目范围外的端点：统一 501（见 misc::not_implemented 的 TODO）
    for path in [
        "/v1/embeddings",
        "/pooling",
        "/score",
        "/v1/score",
        "/rerank",
        "/v1/rerank",
        "/v2/rerank",
        "/classify",
        "/v1/audio/transcriptions",
        "/v1/audio/translations",
        "/v1/responses",
    ] {
        r = r.route(path, post(misc::not_implemented));
    }
    r.with_state(backend)
}

async fn metrics(State(be): State<Arc<dyn ChatBackend>>) -> axum::response::Response {
    use axum::response::IntoResponse;
    let body = match be.metrics() {
        Some(m) => m.render(METRICS_PREFIX),
        None => String::new(),
    };
    ([(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")], body).into_response()
}
