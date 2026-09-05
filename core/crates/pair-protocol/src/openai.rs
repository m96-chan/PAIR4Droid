//! OpenAI-compatible lane on port [`crate::ports::OPENAI`] — what PAIR's
//! `lmstudio-proxy` forwards to, and what `nvpair-manual-nodes` probes for the
//! node's model list.
//!
//! Surface (see `docs/pair-contract.md` §1.5 and §3.1):
//! - `GET  /v1/models`           → [`ModelList`]. A `200` is the whole liveness
//!   check; PAIR reads `data[].id` and nothing else
//!   (`services/nvpair-manual-nodes/manager.go:431-442`).
//! - `POST /v1/chat/completions` → [`ChatCompletionRequest`] → either
//!   [`ChatCompletionResponse`] or an SSE stream of [`ChatCompletionChunk`]
//!   (see [`sse`]).
//!
//! Leniency is a requirement, not a nicety: `lmstudio-proxy` replays the client's
//! body verbatim and only *reads* the top-level `"model"` string out of it
//! (`services/lmstudio-proxy/proxy.go:199-215`). Whatever the client sent —
//! `tools`, `response_format`, `logprobs` — arrives here untouched, and refusing
//! it would fail a request PAIR considers well-formed.

use serde::{Deserialize, Serialize};

use crate::serde_util::null_to_default;

// ---------------------------------------------------------------------------
// GET /v1/models
// ---------------------------------------------------------------------------

/// `{"object":"list","data":[…]}` — the shape both PAIR's prober and its
/// `/v1/models` fan-out expect (`services/ollama-proxy/proxy.go:1109-1120`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelList {
    /// Always `"list"`.
    pub object: String,
    /// A Go peer may send `null` here
    /// (`services/lmstudio-proxy/failover_test.go:387`).
    #[serde(default, deserialize_with = "null_to_default")]
    pub data: Vec<Model>,
}

impl Default for ModelList {
    fn default() -> Self {
        Self { object: "list".into(), data: Vec::new() }
    }
}

impl ModelList {
    pub fn new(data: Vec<Model>) -> Self {
        Self { object: "list".into(), data }
    }

    /// Build the list from the engine's catalogue. The ids are PAIR's exact-match
    /// routing key (`docs/pair-contract.md` §3.3), so they must be unique in the
    /// fleet.
    pub fn from_ids<I, S>(ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::new(ids.into_iter().map(Model::new).collect())
    }
}

/// One entry of [`ModelList::data`]. PAIR keeps only non-empty `id`s
/// (`services/nvpair-manual-nodes/manager.go:437-442`); the rest is for the
/// benefit of ordinary OpenAI clients.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    /// Always `"model"`. Defaulted on decode: PAIR's fakes often omit it
    /// (`services/lmstudio-proxy/failover_test.go:382`).
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created: i64,
    #[serde(default)]
    pub owned_by: String,
}

impl Model {
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into(), object: "model".into(), created: 0, owned_by: OWNED_BY.into() }
    }

    pub fn created_at(mut self, created: i64) -> Self {
        self.created = created;
        self
    }
}

/// `owned_by` value this node advertises. PAIR never reads it.
pub const OWNED_BY: &str = "pair4droid";

// ---------------------------------------------------------------------------
// Chat messages
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// `content` is either a plain string or an array of typed parts. Both forms are
/// legal OpenAI and both arrive here, so the enum is `untagged`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl Content {
    /// The plain text an engine can consume: the string itself, or every `text`
    /// part concatenated. Unsupported parts contribute nothing.
    pub fn text(&self) -> String {
        match self {
            Content::Text(s) => s.clone(),
            Content::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    ContentPart::Unsupported => None,
                })
                .collect(),
        }
    }
}

impl From<&str> for Content {
    fn from(s: &str) -> Self {
        Content::Text(s.to_string())
    }
}

impl From<String> for Content {
    fn from(s: String) -> Self {
        Content::Text(s)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    /// Any other part kind (`image_url`, `input_audio`, …). Accepted so the
    /// request parses, then ignored — there is no vision in Phase 1.
    #[serde(other)]
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    /// `null` is legal on an assistant message that only carried tool calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Content>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    pub fn new(role: Role, content: impl Into<Content>) -> Self {
        Self { role, content: Some(content.into()), name: None }
    }

    pub fn assistant(content: impl Into<Content>) -> Self {
        Self::new(Role::Assistant, content)
    }

    /// Flattened text of this message; `""` when the content is absent.
    pub fn text(&self) -> String {
        self.content.as_ref().map(Content::text).unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions — request
// ---------------------------------------------------------------------------

/// Unknown fields are ignored by design; see the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    #[serde(default, deserialize_with = "null_to_default")]
    pub messages: Vec<ChatMessage>,
    /// OpenAI's default is non-streaming (unlike Ollama's, see
    /// [`crate::ollama::ChatRequest::stream`]).
    #[serde(default)]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
}

impl ChatCompletionRequest {
    /// Text of the last `user` message — what [`crate::ollama`]'s and the mock
    /// engine's `echo:` behaviour keys on.
    pub fn last_user_text(&self) -> Option<String> {
        self.messages.iter().rev().find(|m| m.role == Role::User).map(ChatMessage::text)
    }

    /// The generation budget, preferring the newer field name.
    pub fn token_budget(&self) -> Option<u32> {
        self.max_completion_tokens.or(self.max_tokens)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions — non-streaming response
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
}

impl Usage {
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    #[serde(default)]
    pub id: String,
    /// Always `"chat.completion"`.
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created: i64,
    #[serde(default)]
    pub model: String,
    #[serde(default, deserialize_with = "null_to_default")]
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub usage: Usage,
}

impl ChatCompletionResponse {
    /// A single-choice assistant answer, the only shape this node produces.
    pub fn assistant(
        id: impl Into<String>,
        model: impl Into<String>,
        created: i64,
        text: impl Into<Content>,
        finish_reason: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            object: OBJECT_CHAT_COMPLETION.into(),
            created,
            model: model.into(),
            choices: vec![Choice {
                index: 0,
                message: ChatMessage::assistant(text),
                finish_reason: Some(finish_reason.into()),
            }],
            usage: Usage::default(),
        }
    }

    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = usage;
        self
    }
}

impl Default for ChatCompletionResponse {
    fn default() -> Self {
        Self {
            id: String::new(),
            object: OBJECT_CHAT_COMPLETION.into(),
            created: 0,
            model: String::new(),
            choices: Vec::new(),
            usage: Usage::default(),
        }
    }
}

pub const OBJECT_CHAT_COMPLETION: &str = "chat.completion";
pub const OBJECT_CHAT_COMPLETION_CHUNK: &str = "chat.completion.chunk";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Choice {
    #[serde(default)]
    pub index: u32,
    pub message: ChatMessage,
    /// `"stop"` | `"length"`; `null` while a choice is still open.
    #[serde(default)]
    pub finish_reason: Option<String>,
}

// ---------------------------------------------------------------------------
// SSE chunks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    #[serde(default)]
    pub id: String,
    /// Always `"chat.completion.chunk"`.
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created: i64,
    #[serde(default)]
    pub model: String,
    #[serde(default, deserialize_with = "null_to_default")]
    pub choices: Vec<ChunkChoice>,
    /// Only ever present on the final chunk, and only when the client asked
    /// (`stream_options.include_usage`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

impl ChatCompletionChunk {
    fn with_choice(
        id: impl Into<String>,
        model: impl Into<String>,
        created: i64,
        delta: Delta,
        finish_reason: Option<String>,
    ) -> Self {
        Self {
            id: id.into(),
            object: OBJECT_CHAT_COMPLETION_CHUNK.into(),
            created,
            model: model.into(),
            choices: vec![ChunkChoice { index: 0, delta, finish_reason }],
            usage: None,
        }
    }

    /// The opening chunk: role only, no content.
    pub fn first(id: impl Into<String>, model: impl Into<String>, created: i64) -> Self {
        Self::with_choice(id, model, created, Delta { role: Some(Role::Assistant), content: None }, None)
    }

    /// One generated token (or word, or fragment).
    pub fn token(
        id: impl Into<String>,
        model: impl Into<String>,
        created: i64,
        text: impl Into<String>,
    ) -> Self {
        Self::with_choice(id, model, created, Delta { role: None, content: Some(text.into()) }, None)
    }

    /// The terminating chunk: empty delta plus a `finish_reason`.
    pub fn finish(
        id: impl Into<String>,
        model: impl Into<String>,
        created: i64,
        finish_reason: impl Into<String>,
    ) -> Self {
        Self::with_choice(id, model, created, Delta::default(), Some(finish_reason.into()))
    }

    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Text carried by the first choice, if any.
    pub fn text(&self) -> Option<&str> {
        self.choices.first()?.delta.content.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkChoice {
    #[serde(default)]
    pub index: u32,
    #[serde(default)]
    pub delta: Delta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Delta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// OpenAI-style envelope: `{"error":{"message":…,"type":…,"code":…}}`.
///
/// PAIR's own proxies answer with a *bare string* error instead
/// (`{"error":"model not found"}`, `services/lmstudio-proxy/proxy.go:985`,
/// `:1242`) — a different shape, and not one a node has to produce. PAIR reads
/// only the status code from us: 404 makes it fail over to the next owner
/// (`docs/pair-contract.md` §3.1 "Failover statuses").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorBody,
}

impl ErrorResponse {
    pub fn new(message: impl Into<String>, kind: impl Into<String>, code: Option<&str>) -> Self {
        Self {
            error: ErrorBody { message: message.into(), kind: kind.into(), code: code.map(str::to_string) },
        }
    }

    /// The 404 body for a model this node does not host. PAIR fails over on 404.
    pub fn model_not_found(model: &str) -> Self {
        Self::new(format!("model '{model}' not found"), "invalid_request_error", Some("model_not_found"))
    }

    /// The 503 body when admission is refused (thermal / battery / queue).
    pub fn overloaded(message: impl Into<String>) -> Self {
        Self::new(message, "server_error", Some("overloaded"))
    }

    /// The 400 body for a request we could not parse.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(message, "invalid_request_error", None)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorBody {
    pub message: String,
    #[serde(rename = "type", default)]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

// ---------------------------------------------------------------------------
// SSE framing
// ---------------------------------------------------------------------------

/// Server-Sent Events framing for the streaming half of `/v1/chat/completions`.
///
/// PAIR does **not** parse this: `lmstudio-proxy` is a stock
/// `httputil.ReverseProxy` and relays bytes, never looking for `data:` or
/// `[DONE]` (`docs/pair-contract.md` §3.1). The framing therefore has to be
/// right for the *client* behind PAIR, and PAIR's only requirement is that we
/// flush — which it gets for free from a chunked response.
pub mod sse {
    use super::ChatCompletionChunk;

    /// The terminating frame of an OpenAI stream, including its blank line.
    pub const DONE: &str = "data: [DONE]\n\n";

    /// One SSE frame: `data: <compact json>\n\n`.
    ///
    /// The JSON is guaranteed single-line (serde_json emits no raw newlines),
    /// so the frame is always exactly one `data:` line plus the blank line that
    /// dispatches the event.
    pub fn encode_chunk(chunk: &ChatCompletionChunk) -> String {
        let body = serde_json::to_string(chunk).expect("ChatCompletionChunk is always serialisable");
        format!("data: {body}\n\n")
    }

    /// What one SSE `data:` line carried.
    #[derive(Debug, Clone, PartialEq)]
    pub enum SseEvent {
        Chunk(Box<ChatCompletionChunk>),
        Done,
    }

    /// Parse a single line of an SSE body.
    ///
    /// Returns `None` for anything that is not a decodable `data:` payload —
    /// blank separator lines, comments (`: keep-alive`), `event:` / `id:` /
    /// `retry:` fields, and payloads that are not a chunk. Accepts both
    /// `data: x` and `data:x`, and tolerates a trailing `\r\n` or `\n`.
    pub fn decode_line(line: &str) -> Option<SseEvent> {
        let line = line.trim_end_matches('\n').trim_end_matches('\r');
        let payload = line.strip_prefix("data:")?.trim_start_matches(' ');
        if payload == "[DONE]" {
            return Some(SseEvent::Done);
        }
        serde_json::from_str::<ChatCompletionChunk>(payload).ok().map(|c| SseEvent::Chunk(Box::new(c)))
    }
}
