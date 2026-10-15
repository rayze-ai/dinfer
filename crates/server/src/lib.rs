//! dinfer-server：axum 路由。`/health`、`/v1/models`、`/v1/chat/completions`（流式 + 非流式）。
//! 模型通过 `backend::ChatBackend` 接入（见 glm53 `service.rs`）；server 本身不依赖任何模型 crate。
//!
//! 安全：无鉴权。默认只绑 127.0.0.1；对外暴露必须放在网关后面。

#![forbid(unsafe_code)]

pub mod backend;
pub mod chat;
pub mod openai;

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
    axum::serve(listener, app).await?;
    Ok(())
}

pub fn router(backend: Arc<dyn ChatBackend>) -> Router {
    Router::new()
        .route("/health", get(|| async { Json(Health { status: "ok" }) }))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat::chat_completions))
        .with_state(backend)
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
