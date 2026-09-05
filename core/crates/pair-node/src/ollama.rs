//! Ollama-compatible lane on `:11434`.
//!
//! PAIR references (`docs/pair-contract.md` §1.5, §3.2, §3.3):
//! - `probeOllama` (`services/nvpair-manual-nodes/manager.go:448-471`) demands a
//!   bare **200** on `GET /` — the body is closed unread (`:458`) — and only
//!   then fetches `/api/tags`. A non-200 marks the whole lane down.
//! - `fetchOllamaModels` (`:473-497`) reads `models[].name`; the proxy's
//!   `/api/tags` fan-out prefers `models[].model`, falls back to `name`, and
//!   rejects a record with neither (`services/ollama-proxy/proxy.go:1062-1074`).
//!   So every entry carries both.
//! - Everything except the model-list routes is forwarded verbatim by
//!   `ollama-proxy` through a stock `httputil.ReverseProxy`
//!   (`services/ollama-proxy/proxy.go:1153-1169`); NDJSON is never parsed, and a
//!   response with no `Content-Length` flushes per write.
//! - `ollamaModelKey` (`services/ollama-proxy/proxy.go:967-974`) appends
//!   `:latest` when the last path segment carries no `:` or `@`, so `llama` and
//!   `llama:latest` are the same model on this lane (still case-sensitive).
//! - `shouldRetry` (`services/ollama-proxy/proxy.go:1202-1214`): 404 on a POST
//!   to an inference path and 503 make PAIR fail over; 400 is returned as-is.

use crate::stream::chunked_body;
use crate::{access_log, not_found, AppState, LoggedModel};
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use futures::stream::StreamExt;
use pair_engine::{
    ChatMessage, ChatRole, EngineError, FinishReason, GenerationParams, ModelInfo, TokenEvent,
};
use pair_protocol::ollama::{
    ChatRequest, ChatResponse, ErrorResponse, GenerateRequest, GenerateResponse, Message, ModelDetails,
    Options, PsModel, PsResponse, ShowRequest, ShowResponse, TagModel, TagsResponse, Timings,
    VersionResponse,
};
use std::time::Instant;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(root))
        .route("/api/tags", get(tags))
        .route("/api/version", get(version))
        .route("/api/chat", post(chat))
        .route("/api/generate", post(generate))
        .route("/api/show", post(show))
        .route("/api/ps", get(ps))
        .with_state(state)
        .fallback(not_found)
        .layer(axum::middleware::from_fn(access_log))
}

// ------------------------------------------------------------------- helpers

fn error(status: StatusCode, message: impl Into<String>) -> Response {
    (status, axum::Json(ErrorResponse { error: message.into() })).into_response()
}

fn model_not_found(model: &str) -> Response {
    error(StatusCode::NOT_FOUND, format!("model '{model}' not found"))
}

fn engine_error(err: EngineError) -> Response {
    match err {
        EngineError::ModelNotFound(m) => model_not_found(&m),
        EngineError::Busy => error(StatusCode::SERVICE_UNAVAILABLE, "engine is busy"),
        EngineError::ContextExceeded { prompt_tokens, context_length } => error(
            StatusCode::BAD_REQUEST,
            format!("context length exceeded: prompt {prompt_tokens} tokens > context {context_length}"),
        ),
        other => error(StatusCode::INTERNAL_SERVER_ERROR, other.to_string()),
    }
}

/// Port of PAIR's `ollamaModelKey` (`services/ollama-proxy/proxy.go:967-974`):
/// trim, and append `:latest` when the last path segment has no `:` or `@`.
/// No lowercasing — `Llama` and `llama` are different models.
pub fn model_key(model: &str) -> String {
    let model = model.trim();
    let last = match model.rfind('/') {
        Some(i) => &model[i + 1..],
        None => model,
    };
    if !last.is_empty() && !last.contains(':') && !last.contains('@') {
        format!("{model}:latest")
    } else {
        model.to_string()
    }
}

/// Resolve a request's model against the catalogue with `:latest` normalisation.
async fn resolve(state: &AppState, requested: &str) -> Option<ModelInfo> {
    if let Some(info) = state.engine.model(requested).await {
        return Some(info);
    }
    let key = model_key(requested);
    state.engine.list_models().await.into_iter().find(|m| model_key(&m.name) == key)
}

fn details_of(info: &ModelInfo) -> ModelDetails {
    ModelDetails {
        parent_model: String::new(),
        format: "gguf".to_string(),
        family: info.family.clone(),
        families: Some(vec![info.family.clone()]),
        parameter_size: info.parameter_size.clone(),
        quantization_level: info.quantization.clone(),
    }
}

/// `/api/tags` and `/api/ps` advertise `sha256:<64 hex>`. An engine that already
/// supplies that form is passed through; anything else is hashed into shape so
/// the value is at least stable per model.
fn digest_of(info: &ModelInfo) -> String {
    let raw = info.digest.trim();
    let hex = raw.strip_prefix("sha256:").unwrap_or(raw);
    if hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return format!("sha256:{}", hex.to_ascii_lowercase());
    }
    // Deterministic filler: repeat a FNV-1a digest of the name to 64 hex chars.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in info.name.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let mut hex = String::with_capacity(64);
    for i in 0..4u64 {
        hex.push_str(&format!("{:016x}", h.wrapping_add(i.wrapping_mul(0x9e37_79b9_7f4a_7c15))));
    }
    format!("sha256:{hex}")
}

fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Nanos, true)
}

fn done_reason(reason: &FinishReason) -> &'static str {
    match reason {
        FinishReason::Length => "length",
        FinishReason::Stop | FinishReason::Cancelled => "stop",
    }
}

fn params_from(options: &Option<Options>) -> GenerationParams {
    let Some(o) = options else { return GenerationParams::default() };
    GenerationParams {
        // Ollama uses -1 / -2 for "unlimited" / "fill context".
        max_tokens: o.num_predict.filter(|n| *n > 0).map(|n| n as u32),
        temperature: o.temperature,
        top_p: o.top_p,
        stop: o.stop.clone().unwrap_or_default(),
        seed: o.seed,
    }
}

fn role_of(role: &str) -> ChatRole {
    match role {
        "system" => ChatRole::System,
        "assistant" => ChatRole::Assistant,
        "tool" => ChatRole::Tool,
        _ => ChatRole::User,
    }
}

fn assistant(content: &str) -> Message {
    Message { role: "assistant".to_string(), content: content.to_string(), images: None }
}

fn ns(ms: u64) -> u64 {
    ms.saturating_mul(1_000_000)
}

fn ndjson_line(value: &impl serde::Serialize) -> Bytes {
    let mut line = serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string());
    line.push('\n');
    Bytes::from(line)
}

/// What the two NDJSON lanes differ in: how a token and the final object are
/// rendered (`message` vs `response`).
#[derive(Clone, Copy)]
enum Shape {
    Chat,
    Generate,
}

impl Shape {
    fn token(self, model: &str, token: &str) -> serde_json::Value {
        let created_at = now_rfc3339();
        match self {
            Shape::Chat => serde_json::to_value(ChatResponse {
                model: model.to_string(),
                created_at,
                message: assistant(token),
                done: false,
                done_reason: None,
                timings: None,
            }),
            Shape::Generate => serde_json::to_value(GenerateResponse {
                model: model.to_string(),
                created_at,
                response: token.to_string(),
                done: false,
                done_reason: None,
                timings: None,
            }),
        }
        .unwrap_or_default()
    }

    fn final_object(self, model: &str, content: &str, reason: &str, timings: Timings) -> serde_json::Value {
        let created_at = now_rfc3339();
        match self {
            Shape::Chat => serde_json::to_value(ChatResponse {
                model: model.to_string(),
                created_at,
                message: assistant(content),
                done: true,
                done_reason: Some(reason.to_string()),
                timings: Some(timings),
            }),
            Shape::Generate => serde_json::to_value(GenerateResponse {
                model: model.to_string(),
                created_at,
                response: content.to_string(),
                done: true,
                done_reason: Some(reason.to_string()),
                timings: Some(timings),
            }),
        }
        .unwrap_or_default()
    }
}

// ------------------------------------------------------------------ handlers

/// PAIR's liveness check. Real Ollama answers `200 "Ollama is running"`;
/// `manager.go:458` never reads the body, but the string keeps us honest for
/// any other Ollama client.
async fn root() -> Response {
    ([(header::CONTENT_TYPE, HeaderValue::from_static("text/plain; charset=utf-8"))], "Ollama is running")
        .into_response()
}

async fn tags(State(state): State<AppState>) -> Response {
    let models = state
        .engine
        .list_models()
        .await
        .into_iter()
        .map(|m| TagModel {
            name: m.name.clone(),
            model: m.name.clone(),
            modified_at: m.modified_at.clone(),
            size: m.size_bytes,
            digest: digest_of(&m),
            details: details_of(&m),
        })
        .collect();
    axum::Json(TagsResponse { models }).into_response()
}

async fn version(State(state): State<AppState>) -> Response {
    axum::Json(VersionResponse { version: state.config.ollama_version.clone() }).into_response()
}

async fn ps(State(state): State<AppState>) -> Response {
    let status = state.engine.status();
    let mut models = Vec::new();
    if let Some(name) = status.loaded_model {
        if let Some(info) = state.engine.model(&name).await {
            models.push(PsModel {
                name: info.name.clone(),
                model: info.name.clone(),
                size: info.size_bytes,
                digest: digest_of(&info),
                details: details_of(&info),
                // Ollama's default keep-alive is 5 minutes.
                expires_at: (chrono::Utc::now() + chrono::Duration::minutes(5))
                    .to_rfc3339_opts(chrono::SecondsFormat::Nanos, true),
                size_vram: status.loaded_bytes,
            });
        }
    }
    axum::Json(PsResponse { models }).into_response()
}

async fn show(State(state): State<AppState>, body: Bytes) -> Response {
    let req: ShowRequest = match serde_json::from_slice(&body) {
        Ok(req) => req,
        Err(e) => return error(StatusCode::BAD_REQUEST, format!("invalid request body: {e}")),
    };
    // Ollama accepts both `model` and the legacy `name`.
    let requested = req.model.or(req.name).unwrap_or_default();
    let Some(info) = resolve(&state, &requested).await else {
        return model_not_found(&requested);
    };

    let mut model_info = std::collections::BTreeMap::new();
    model_info.insert("general.architecture".to_string(), info.family.clone().into());
    model_info.insert("general.basename".to_string(), info.name.clone().into());
    model_info.insert("general.parameter_size".to_string(), info.parameter_size.clone().into());
    model_info.insert("general.quantization_level".to_string(), info.quantization.clone().into());
    model_info.insert(format!("{}.context_length", info.family), info.context_length.into());

    axum::Json(ShowResponse {
        modelfile: String::new(),
        parameters: String::new(),
        template: String::new(),
        details: details_of(&info),
        model_info,
        capabilities: Some(vec!["completion".to_string()]),
    })
    .into_response()
}

async fn chat(State(state): State<AppState>, body: Bytes) -> Response {
    let req: ChatRequest = match serde_json::from_slice(&body) {
        Ok(req) => req,
        Err(e) => return error(StatusCode::BAD_REQUEST, format!("invalid request body: {e}")),
    };
    let messages = req
        .messages
        .iter()
        .map(|m| ChatMessage { role: role_of(&m.role), content: m.content.clone() })
        .collect();
    let mut response =
        run(state, Shape::Chat, &req.model, messages, params_from(&req.options), req.stream).await;
    response.extensions_mut().insert(LoggedModel(req.model));
    response
}

async fn generate(State(state): State<AppState>, body: Bytes) -> Response {
    let req: GenerateRequest = match serde_json::from_slice(&body) {
        Ok(req) => req,
        Err(e) => return error(StatusCode::BAD_REQUEST, format!("invalid request body: {e}")),
    };
    let mut messages = Vec::with_capacity(2);
    if let Some(system) = req.system.filter(|s| !s.is_empty()) {
        messages.push(ChatMessage { role: ChatRole::System, content: system });
    }
    messages.push(ChatMessage { role: ChatRole::User, content: req.prompt.clone() });

    let mut response =
        run(state, Shape::Generate, &req.model, messages, params_from(&req.options), req.stream).await;
    response.extensions_mut().insert(LoggedModel(req.model));
    response
}

async fn run(
    state: AppState,
    shape: Shape,
    requested: &str,
    messages: Vec<ChatMessage>,
    params: GenerationParams,
    stream: bool,
) -> Response {
    let Some(info) = resolve(&state, requested).await else {
        return model_not_found(requested);
    };
    if let pair_telemetry::Admission::Refuse(reason) = state.telemetry.admission() {
        return error(StatusCode::SERVICE_UNAVAILABLE, reason);
    }

    // The engine is addressed by its catalogue name; the response echoes the
    // name the caller used, exactly as Ollama does.
    let engine_request = pair_engine::ChatRequest { model: info.name.clone(), messages, params };
    let token_stream = match state.engine.chat(engine_request).await {
        Ok(s) => s,
        Err(e) => return engine_error(e),
    };

    if stream {
        streaming_response(token_stream, shape, requested.to_string())
    } else {
        collected_response(token_stream, shape, requested.to_string()).await
    }
}

async fn collected_response(mut stream: pair_engine::TokenStream, shape: Shape, model: String) -> Response {
    let start = Instant::now();
    let mut content = String::new();
    let mut timings = Timings::default();
    let mut reason = "stop";
    while let Some(event) = stream.next().await {
        match event {
            Ok(TokenEvent::Start { prompt_tokens }) => timings.prompt_eval_count = prompt_tokens,
            Ok(TokenEvent::Token(t)) => content.push_str(&t),
            Ok(TokenEvent::Done {
                finish_reason,
                prompt_tokens,
                completion_tokens,
                load_ms,
                prompt_ms,
                eval_ms,
            }) => {
                reason = done_reason(&finish_reason);
                timings = Timings {
                    total_duration: 0,
                    load_duration: ns(load_ms),
                    prompt_eval_count: prompt_tokens,
                    prompt_eval_duration: ns(prompt_ms),
                    eval_count: completion_tokens,
                    eval_duration: ns(eval_ms),
                };
            }
            Err(e) => return engine_error(e),
        }
    }
    timings.total_duration = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
    axum::Json(shape.final_object(&model, &content, reason, timings)).into_response()
}

/// NDJSON, one object per line, terminated by an object with `done: true` and
/// the timing block. The engine stream is owned by the response body so a
/// vanished client cancels the generation.
fn streaming_response(stream: pair_engine::TokenStream, shape: Shape, model: String) -> Response {
    struct St {
        stream: pair_engine::TokenStream,
        shape: Shape,
        model: String,
        start: Instant,
        finished: bool,
    }

    let state = St { stream, shape, model, start: Instant::now(), finished: false };

    let body = futures::stream::unfold(state, |mut s| async move {
        if s.finished {
            return None;
        }
        loop {
            match s.stream.next().await {
                Some(Ok(TokenEvent::Start { .. })) => continue,
                Some(Ok(TokenEvent::Token(token))) => {
                    let line = ndjson_line(&s.shape.token(&s.model, &token));
                    return Some((Ok(line), s));
                }
                Some(Ok(TokenEvent::Done {
                    finish_reason,
                    prompt_tokens,
                    completion_tokens,
                    load_ms,
                    prompt_ms,
                    eval_ms,
                })) => {
                    let timings = Timings {
                        total_duration: s.start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                        load_duration: ns(load_ms),
                        prompt_eval_count: prompt_tokens,
                        prompt_eval_duration: ns(prompt_ms),
                        eval_count: completion_tokens,
                        eval_duration: ns(eval_ms),
                    };
                    let value = s.shape.final_object(&s.model, "", done_reason(&finish_reason), timings);
                    s.finished = true;
                    return Some((Ok(ndjson_line(&value)), s));
                }
                Some(Err(e)) => {
                    // Headers are already on the wire; Ollama surfaces a
                    // mid-stream failure as a line carrying `error`.
                    tracing::warn!(error = %e, "engine failed mid-stream");
                    s.finished = true;
                    let line = ndjson_line(&serde_json::json!({"error": e.to_string()}));
                    return Some((Ok(line), s));
                }
                None => return None,
            }
        }
    });

    let mut response = Response::new(chunked_body(body));
    response.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("application/x-ndjson"));
    response
        .headers_mut()
        .insert(header::HeaderName::from_static("x-accel-buffering"), HeaderValue::from_static("no"));
    response
}
