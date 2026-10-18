//! dinfer-server：axum 路由。`/health`、`/metrics`（Prometheus）、`/v1/models`、`/v1/chat/completions`（流式 + 非流式）。
//! 模型通过 `backend::ChatBackend` 接入（见 glm53 `service.rs`）；server 本身不依赖任何模型 crate。
//!
//! 安全：无鉴权。默认只绑 127.0.0.1；对外暴露必须放在网关后面。

#![forbid(unsafe_code)]

pub mod backend;
pub mod chat;
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

#[derive(Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelCard>,
}

#[derive(Serialize)]
struct ModelCard {
    id: String,
    object: &'static str,
    owned_by: &'static str,
    max_model_len: u32,
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
    Router::new()
        .route("/health", get(|| async { Json(Health { status: "ok" }) }))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat::chat_completions))
        .with_state(backend)
}

async fn metrics(State(be): State<Arc<dyn ChatBackend>>) -> axum::response::Response {
    use axum::response::IntoResponse;
    let body = match be.metrics() {
        Some(m) => m.render(METRICS_PREFIX),
        None => String::new(),
    };
    ([(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")], body).into_response()
}

async fn models(State(be): State<Arc<dyn ChatBackend>>) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: vec![ModelCard {
            id: be.model_name().to_string(),
            object: "model",
            owned_by: "dinfer",
            max_model_len: be.max_model_len(),
        }],
    })
}
