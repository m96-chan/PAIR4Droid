//! OpenAI-compatible lane on `:1234` — the "LM Studio" lane in PAIR's language.
//!
//! PAIR references (`docs/pair-contract.md` §1.5, §3.1, §3.3):
//! - `probeLMStudio` (`services/nvpair-manual-nodes/manager.go:409-446`) uses a
//!   single `GET /v1/models` as both liveness check and inventory, reading only
//!   non-empty `data[].id`.
//! - `lmstudio-proxy` forwards `POST /v1/chat/completions` through a stock
//!   `httputil.ReverseProxy` (`services/lmstudio-proxy/proxy.go:1132-1250`): it
//!   never parses SSE, so the only hard requirements are a prompt status line
//!   (`ResponseHeaderTimeout` 120 s) and a body that flushes per write — which
//!   `text/event-stream` guarantees.
//! - `shouldRetry` (`services/lmstudio-proxy/proxy.go:1007-1019`): 404 on a POST
//!   to an inference path makes PAIR fail over to the next owner, 503 likewise;
//!   400/422 are returned to the caller as-is. Hence unknown model → 404,
//!   admission refused → 503, malformed JSON → 400.
//! - Model matching on this lane is exact and case-sensitive
//!   (`services/lmstudio-proxy/proxy.go:1528-1538`) — no `:latest` normalisation.

use crate::stream::chunked_body;
use crate::{access_log, not_found, AppState, LoggedModel};
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::Router;
use futures::stream::StreamExt;
use pair_engine::{
    ChatMessage, ChatRequest, ChatRole, EngineError, FinishReason, GenerationParams, TokenEvent,
};
use pair_protocol::openai::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Choice, ChunkChoice, Content,
    ContentPart, Delta, ErrorBody, ErrorResponse, Model, ModelList, Role, Usage,
};

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        // Phase 1 does not implement these; PAIR tracks them as inference paths
        // (`services/lmstudio-proxy/proxy.go:151-160`) so a 404 makes it fail
        // over to a node that does.
        .route("/v1/completions", any(not_found))
        .route("/v1/embeddings", any(not_found))
        .with_state(state)
        .fallback(not_found)
        .layer(axum::middleware::from_fn(access_log))
}

// ------------------------------------------------------------------- helpers

const OWNED_BY: &str = "pair4droid";

fn error(status: StatusCode, message: impl Into<String>, code: Option<&str>) -> Response {
    let body = ErrorResponse {
        error: ErrorBody {
            message: message.into(),
            kind: "invalid_request_error".to_string(),
            code: code.map(str::to_string),
        },
    };
    (status, axum::Json(body)).into_response()
}

fn server_error(message: impl Into<String>) -> Response {
    let body = ErrorResponse {
        error: ErrorBody { message: message.into(), kind: "server_error".to_string(), code: None },
    };
    (StatusCode::INTERNAL_SERVER_ERROR, axum::Json(body)).into_response()
}

/// `EngineError` → HTTP, matching `docs/architecture.md` "Failure semantics".
fn engine_error(err: EngineError) -> Response {
    match err {
        EngineError::ModelNotFound(m) => model_not_found(&m),
        EngineError::Busy => error(StatusCode::SERVICE_UNAVAILABLE, "engine is busy", Some("busy")),
        EngineError::ContextExceeded { prompt_tokens, context_length } => error(
            StatusCode::BAD_REQUEST,
            format!("context length exceeded: prompt {prompt_tokens} tokens > context {context_length}"),
            Some("context_length_exceeded"),
        ),
        other => server_error(other.to_string()),
    }
}

fn model_not_found(model: &str) -> Response {
    error(StatusCode::NOT_FOUND, format!("model '{model}' not found"), Some("model_not_found"))
}

fn finish_reason(reason: &FinishReason) -> &'static str {
    match reason {
        FinishReason::Length => "length",
        // A cancelled generation only reaches here when the engine reports it
        // without the client having gone away; "stop" is the honest OpenAI word.
        FinishReason::Stop | FinishReason::Cancelled => "stop",
    }
}

fn now_secs() -> i64 {
    chrono::Utc::now().timestamp()
}

/// `created` is the unix time of the model file, falling back to now.
fn created_from(modified_at: &str) -> i64 {
    chrono::DateTime::parse_from_rfc3339(modified_at).map(|d| d.timestamp()).unwrap_or_else(|_| now_secs())
}

fn text_of(content: &Option<Content>) -> String {
    match content {
        None => String::new(),
        Some(Content::Text(t)) => t.clone(),
        Some(Content::Parts(parts)) => parts
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text { text } => Some(text.as_str()),
                ContentPart::Unsupported => None,
            })
            .collect::<String>(),
    }
}

fn role_of(role: &Role) -> ChatRole {
    match role {
        Role::System => ChatRole::System,
        Role::User => ChatRole::User,
        Role::Assistant => ChatRole::Assistant,
        Role::Tool => ChatRole::Tool,
    }
}

fn to_engine_request(req: &ChatCompletionRequest) -> ChatRequest {
    ChatRequest {
        model: req.model.clone(),
        messages: req
            .messages
            .iter()
            .map(|m| ChatMessage { role: role_of(&m.role), content: text_of(&m.content) })
            .collect(),
        params: GenerationParams {
            // `max_completion_tokens` supersedes the deprecated `max_tokens`.
            max_tokens: req.max_completion_tokens.or(req.max_tokens),
            temperature: req.temperature,
            top_p: req.top_p,
            stop: req.stop.clone().unwrap_or_default(),
            seed: req.seed,
        },
    }
}

fn chunk(id: &str, model: &str, created: i64, choices: Vec<ChunkChoice>) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices,
        usage: None,
    }
}

fn sse_data(value: &impl serde::Serialize) -> Bytes {
    let mut line = String::from("data: ");
    line.push_str(&serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string()));
    line.push_str("\n\n");
    Bytes::from(line)
}

// ------------------------------------------------------------------ handlers

async fn list_models(State(state): State<AppState>) -> Response {
    let models = state.engine.list_models().await;
    let list = ModelList {
        object: "list".to_string(),
        data: models
            .into_iter()
            .map(|m| Model {
                id: m.name,
                object: "model".to_string(),
                created: created_from(&m.modified_at),
                owned_by: OWNED_BY.to_string(),
            })
            .collect(),
    };
    axum::Json(list).into_response()
}

async fn chat_completions(State(state): State<AppState>, body: Bytes) -> Response {
    // Parsed by hand rather than through `Json` so *any* malformed body is a 400
    // (axum's own rejection would answer 422 for a well-formed-but-wrong body,
    // and PAIR returns 422 to the caller instead of failing over).
    let req: ChatCompletionRequest = match serde_json::from_slice(&body) {
        Ok(req) => req,
        Err(e) => return error(StatusCode::BAD_REQUEST, format!("invalid request body: {e}"), None),
    };

    let mut response = chat_completions_inner(state, &req).await;
    response.extensions_mut().insert(LoggedModel(req.model.clone()));
    response
}

async fn chat_completions_inner(state: AppState, req: &ChatCompletionRequest) -> Response {
    if state.engine.model(&req.model).await.is_none() {
        return model_not_found(&req.model);
    }
    if let pair_telemetry::Admission::Refuse(reason) = state.telemetry.admission() {
        return error(StatusCode::SERVICE_UNAVAILABLE, reason, Some("node_unavailable"));
    }

    let stream = match state.engine.chat(to_engine_request(req)).await {
        Ok(stream) => stream,
        Err(e) => return engine_error(e),
    };

    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = now_secs();
    if req.stream {
        streaming_response(stream, id, req.model.clone(), created, include_usage(req))
    } else {
        collected_response(stream, id, req.model.clone(), created).await
    }
}

fn include_usage(req: &ChatCompletionRequest) -> bool {
    req.stream_options.as_ref().is_some_and(|o| o.include_usage)
}

async fn collected_response(
    mut stream: pair_engine::TokenStream,
    id: String,
    model: String,
    created: i64,
) -> Response {
    let mut content = String::new();
    let mut usage = Usage::default();
    let mut reason = "stop";
    while let Some(event) = stream.next().await {
        match event {
            Ok(TokenEvent::Start { prompt_tokens }) => usage.prompt_tokens = prompt_tokens,
            Ok(TokenEvent::Token(t)) => content.push_str(&t),
            Ok(TokenEvent::Done { finish_reason, prompt_tokens, completion_tokens, .. }) => {
                reason = self::finish_reason(&finish_reason);
                usage.prompt_tokens = prompt_tokens;
                usage.completion_tokens = completion_tokens;
            }
            Err(e) => return engine_error(e),
        }
    }
    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;

    let body = ChatCompletionResponse {
        id,
        object: "chat.completion".to_string(),
        created,
        model,
        choices: vec![Choice {
            index: 0,
            message: pair_protocol::openai::ChatMessage {
                role: Role::Assistant,
                content: Some(Content::Text(content)),
                name: None,
            },
            finish_reason: Some(reason.to_string()),
        }],
        usage,
    };
    axum::Json(body).into_response()
}

/// SSE per the OpenAI streaming shape: a first chunk carrying the assistant
/// role, one chunk per token, a chunk with an empty delta and the finish
/// reason, an optional usage-only chunk, then `data: [DONE]`.
///
/// The engine stream is owned by the response body, so when PAIR (or its
/// client) goes away axum drops the body and the generation is cancelled —
/// the battery-saving invariant in CLAUDE.md.
fn streaming_response(
    stream: pair_engine::TokenStream,
    id: String,
    model: String,
    created: i64,
    include_usage: bool,
) -> Response {
    enum Phase {
        Streaming,
        Usage(Usage),
        Done,
        Finished,
    }

    struct State {
        stream: pair_engine::TokenStream,
        id: String,
        model: String,
        created: i64,
        include_usage: bool,
        prompt_tokens: u32,
        phase: Phase,
        role_sent: bool,
    }

    let state = State {
        stream,
        id,
        model,
        created,
        include_usage,
        prompt_tokens: 0,
        phase: Phase::Streaming,
        role_sent: false,
    };

    let body = futures::stream::unfold(state, |mut s| async move {
        loop {
            match s.phase {
                Phase::Finished => return None,
                Phase::Done => {
                    s.phase = Phase::Finished;
                    return Some((Ok(Bytes::from_static(b"data: [DONE]\n\n")), s));
                }
                Phase::Usage(ref usage) => {
                    let mut c = chunk(&s.id, &s.model, s.created, Vec::new());
                    c.usage = Some(*usage);
                    s.phase = Phase::Done;
                    return Some((Ok(sse_data(&c)), s));
                }
                Phase::Streaming => {}
            }

            if !s.role_sent {
                s.role_sent = true;
                let c = chunk(
                    &s.id,
                    &s.model,
                    s.created,
                    vec![ChunkChoice {
                        index: 0,
                        delta: Delta { role: Some(Role::Assistant), content: Some(String::new()) },
                        finish_reason: None,
                    }],
                );
                return Some((Ok(sse_data(&c)), s));
            }

            match s.stream.next().await {
                Some(Ok(TokenEvent::Start { prompt_tokens })) => {
                    s.prompt_tokens = prompt_tokens;
                }
                Some(Ok(TokenEvent::Token(token))) => {
                    let c = chunk(
                        &s.id,
                        &s.model,
                        s.created,
                        vec![ChunkChoice {
                            index: 0,
                            delta: Delta { role: None, content: Some(token) },
                            finish_reason: None,
                        }],
                    );
                    return Some((Ok(sse_data(&c)), s));
                }
                Some(Ok(TokenEvent::Done {
                    finish_reason: reason,
                    prompt_tokens,
                    completion_tokens,
                    ..
                })) => {
                    let usage = Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    };
                    s.phase = if s.include_usage { Phase::Usage(usage) } else { Phase::Done };
                    let c = chunk(
                        &s.id,
                        &s.model,
                        s.created,
                        vec![ChunkChoice {
                            index: 0,
                            delta: Delta::default(),
                            finish_reason: Some(finish_reason(&reason).to_string()),
                        }],
                    );
                    return Some((Ok(sse_data(&c)), s));
                }
                Some(Err(e)) => {
                    // Status and headers are already on the wire, so the only
                    // way to surface this is inside the stream.
                    tracing::warn!(error = %e, "engine failed mid-stream");
                    let payload = serde_json::json!({
                        "error": {"message": e.to_string(), "type": "server_error"}
                    });
                    s.phase = Phase::Done;
                    return Some((Ok(sse_data(&payload)), s));
                }
                None => {
                    s.phase = Phase::Done;
                }
            }
        }
    });

    let mut response = Response::new(chunked_body(body));
    let headers = response.headers_mut();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    headers.insert(header::CONNECTION, HeaderValue::from_static("keep-alive"));
    headers.insert(header::HeaderName::from_static("x-accel-buffering"), HeaderValue::from_static("no"));
    response
}
