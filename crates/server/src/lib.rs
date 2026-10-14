//! dinfer-server：axum 路由。P0 只有 /health 与 /v1/models 占位；P6 补全。

#![forbid(unsafe_code)]

use axum::{routing::get, Json, Router};
use serde::Serialize;

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
}

/// 绑定地址并一直服务到进程退出。
pub async fn serve(addr: &str, served_model_name: String) -> anyhow::Result<()> {
    let app = router(served_model_name);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

pub fn router(served_model_name: String) -> Router {
    Router::new().route("/health", get(|| async { Json(Health { status: "ok" }) })).route(
        "/v1/models",
        get(move || {
            let id = served_model_name.clone();
            async move {
                Json(ModelList {
                    object: "list",
                    data: vec![ModelCard { id, object: "model", owned_by: "dinfer" }],
                })
            }
        }),
    )
}
