//! 杂项端点（vLLM `api_server.py` 里 decoder-only 模型用得到的非生成接口）：
//! `/v1/models`、`/v1/models/{id}`、`/tokenize`、`/detokenize`、`/version`、`/ping`，
//! 以及 pooling / 音频 / responses 系列的统一 501。

use crate::backend::{ChatBackend, RenderRequest};
use crate::chat::{check_model, err};
use crate::openai::*;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use std::sync::{Arc, OnceLock};

type Backend = Arc<dyn ChatBackend>;

/// 进程启动时间（unix 秒）：模型卡的 `created`。
fn started_at() -> u64 {
    static T: OnceLock<u64> = OnceLock::new();
    *T.get_or_init(now_secs)
}

#[derive(Serialize)]
pub struct ModelList {
    object: &'static str,
    data: Vec<ModelCard>,
}

/// vLLM `ModelCard`（`permission` 恒为空：无 LoRA / 多模型）。
#[derive(Serialize)]
pub struct ModelCard {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
    root: String,
    parent: Option<String>,
    max_model_len: u32,
    permission: Vec<serde_json::Value>,
}

fn card(be: &dyn ChatBackend) -> ModelCard {
    ModelCard {
        id: be.model_name().to_string(),
        object: "model",
        created: started_at(),
        owned_by: "dinfer",
        root: be.model_name().to_string(),
        parent: None,
        max_model_len: be.max_model_len(),
        permission: Vec::new(),
    }
}

pub async fn models(State(be): State<Backend>) -> Json<ModelList> {
    Json(ModelList { object: "list", data: vec![card(be.as_ref())] })
}

pub async fn model_by_id(State(be): State<Backend>, Path(id): Path<String>) -> Response {
    if let Some(r) = check_model(be.as_ref(), Some(&id)) {
        return r;
    }
    Json(card(be.as_ref())).into_response()
}

#[derive(Serialize)]
pub struct Version {
    version: &'static str,
}

pub async fn version() -> Json<Version> {
    Json(Version { version: env!("CARGO_PKG_VERSION") })
}

/// SageMaker 探活：200 空 body（GET / POST 同一个 handler）。
pub async fn ping() -> StatusCode {
    StatusCode::OK
}

/// `{prompt}` → 纯文本编码；`{messages, ...}` → chat 模板渲染后编码（与 `/v1/chat/completions` 的 `usage.prompt_tokens` 同口径）。
pub async fn tokenize(State(be): State<Backend>, Json(req): Json<TokenizeRequest>) -> Response {
    if let Some(r) = check_model(be.as_ref(), req.model.as_deref()) {
        return r;
    }
    let ids = match (&req.prompt, &req.messages) {
        (Some(_), Some(_)) => {
            return err(StatusCode::BAD_REQUEST, "prompt and messages are mutually exclusive")
        }
        (None, None) => return err(StatusCode::BAD_REQUEST, "either prompt or messages is required"),
        (Some(p), None) => match be.encode(p) {
            Ok(ids) => ids,
            Err(e) => return err(StatusCode::BAD_REQUEST, format!("tokenize: {e:#}")),
        },
        (None, Some(msgs)) => {
            if msgs.is_empty() {
                return err(StatusCode::BAD_REQUEST, "messages must not be empty");
            }
            // TODO(api): 渲染器固定 add_generation_prompt=true；=false 需要模板层支持
            if req.add_generation_prompt == Some(false) {
                return err(StatusCode::BAD_REQUEST, "add_generation_prompt=false is not supported");
            }
            let rr = RenderRequest {
                messages: msgs,
                tools: req.tools.as_deref(),
                reasoning_effort: None,
                clear_thinking: false,
                extra: req.chat_template_kwargs.as_ref(),
            };
            match be.render(&rr) {
                Ok(p) => p.ids,
                Err(e) => return err(StatusCode::BAD_REQUEST, format!("chat template: {e:#}")),
            }
        }
    };
    let token_strs = if req.return_token_strs.unwrap_or(false) {
        match be.token_strs(&ids) {
            Some(s) => Some(s),
            None => {
                return err(StatusCode::BAD_REQUEST, "return_token_strs is not supported by this backend")
            }
        }
    } else {
        None
    };
    Json(TokenizeResponse { count: ids.len(), max_model_len: be.max_model_len(), tokens: ids, token_strs })
        .into_response()
}

pub async fn detokenize(State(be): State<Backend>, Json(req): Json<DetokenizeRequest>) -> Response {
    if let Some(r) = check_model(be.as_ref(), req.model.as_deref()) {
        return r;
    }
    match be.decode(&req.tokens) {
        Ok(prompt) => Json(DetokenizeResponse { prompt }).into_response(),
        Err(e) => err(StatusCode::BAD_REQUEST, format!("detokenize: {e:#}")),
    }
}

/// pooling / 音频 / responses 系列：统一 501 `{"error":{"message","type":"not_implemented","code":501}}`。
// TODO(api): embedding / pooling 模型（/v1/embeddings /pooling /score /rerank /classify）与音频转写不在项目范围；
// TODO(api): /v1/responses 是 vLLM 的有状态会话接口，范围外。
pub async fn not_implemented(uri: axum::http::Uri) -> Response {
    let body = ErrorResponse {
        error: ErrorBody {
            message: format!("{} is not implemented: not supported by a decoder-only text model", uri.path()),
            kind: "not_implemented",
            param: None,
            code: 501,
        },
    };
    (StatusCode::NOT_IMPLEMENTED, Json(body)).into_response()
}
