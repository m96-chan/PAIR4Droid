//! Ollama-compatible lane on port [`crate::ports::OLLAMA`] — what PAIR's
//! `ollama-proxy` forwards to, and the lane `nvpair-manual-nodes` uses for its
//! liveness check.
//!
//! Surface (`docs/pair-contract.md` §1.5, §3.2):
//! - `GET  /`             → **200** with any body. This is PAIR's liveness gate:
//!   a non-200 marks the node down and `/api/tags` is not even attempted
//!   (`services/nvpair-manual-nodes/manager.go:449-471`). Real Ollama answers
//!   `"Ollama is running"`; PAIR's own fake writes an empty 200
//!   (`services/tests/broker_management_test.go:43`).
//! - `GET  /api/tags`     → [`TagsResponse`]. PAIR reads `models[].name` only
//!   (`services/nvpair-manual-nodes/manager.go:473-497`).
//! - `GET  /api/version`  → [`VersionResponse`]
//! - `POST /api/chat`     → [`ChatRequest`] → NDJSON of [`ChatResponse`]
//! - `POST /api/generate` → [`GenerateRequest`] → NDJSON of [`GenerateResponse`]
//! - `POST /api/show`     → [`ShowRequest`] → [`ShowResponse`]
//! - `GET  /api/ps`       → [`PsResponse`]
//!
//! Streaming is `application/x-ndjson`: one JSON object per line (see [`ndjson`]),
//! the last one carrying `"done": true` plus the flattened [`Timings`].
//! `ollama-proxy` does not parse those lines either — it relays them and relies
//! on Go's `ReverseProxy` flushing per write because the body is chunked
//! (`docs/pair-contract.md` §3.2).

use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize};

use crate::serde_util::{default_true, null_to_default};

// ---------------------------------------------------------------------------
// GET /api/tags
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ModelDetails {
    #[serde(default)]
    pub parent_model: String,
    /// `"gguf"`.
    #[serde(default)]
    pub format: String,
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub families: Option<Vec<String>>,
    /// e.g. `"1.5B"`.
    #[serde(default)]
    pub parameter_size: String,
    /// e.g. `"Q4_K_M"`.
    #[serde(default)]
    pub quantization_level: String,
}

/// One row of `GET /api/tags`.
///
/// Only `name` is load-bearing for PAIR: it is the routing key, matched by exact
/// string equality after Ollama's implicit `:latest` normalisation
/// (`docs/pair-contract.md` §3.3). Every other field is defaulted on decode
/// because PAIR's fakes send `{"name":"llama3.2:latest"}` and nothing else
/// (`services/tests/broker_management_test.go:46`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TagModel {
    pub name: String,
    #[serde(default)]
    pub model: String,
    /// RFC3339.
    #[serde(default)]
    pub modified_at: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub digest: String,
    #[serde(default)]
    pub details: ModelDetails,
}

impl TagModel {
    /// A minimal row: `name` == `model`, which is what Ollama reports for a
    /// locally imported model.
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        Self { model: name.clone(), name, ..Default::default() }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TagsResponse {
    /// A Go peer may send `null` (`services/ollama-proxy/failover_test.go:387`).
    #[serde(default, deserialize_with = "null_to_default")]
    pub models: Vec<TagModel>,
}

impl TagsResponse {
    pub fn new(models: Vec<TagModel>) -> Self {
        Self { models }
    }

    pub fn from_names<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::new(names.into_iter().map(TagModel::new).collect())
    }
}

// ---------------------------------------------------------------------------
// GET /api/version
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct VersionResponse {
    pub version: String,
}

impl VersionResponse {
    pub fn new(version: impl Into<String>) -> Self {
        Self { version: version.into() }
    }
}

// ---------------------------------------------------------------------------
// Chat / generate requests
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Message {
    /// `"system"` | `"user"` | `"assistant"` | `"tool"`. Kept as a string
    /// because Ollama accepts any role and we must not reject an unknown one.
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub content: String,
    /// Base64 images; accepted and ignored (no vision in Phase 1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
}

impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self { role: role.into(), content: content.into(), images: None }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

/// Subset of Ollama `options` we honour; every other key is ignored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Options {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    #[serde(default, deserialize_with = "null_to_default")]
    pub messages: Vec<Message>,
    /// **Ollama streams unless told not to** — the opposite of the OpenAI lane's
    /// default. An absent `stream` therefore means `true`.
    #[serde(default = "default_true")]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<Options>,
    /// Free-form: a duration string (`"5m"`), a nanosecond count, or `0`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<serde_json::Value>,
}

impl ChatRequest {
    /// Content of the last `user` message.
    pub fn last_user_text(&self) -> Option<&str> {
        self.messages.iter().rev().find(|m| m.role == "user").map(|m| m.content.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub model: String,
    #[serde(default)]
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Same default as [`ChatRequest::stream`].
    #[serde(default = "default_true")]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<Options>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Chat / generate responses
// ---------------------------------------------------------------------------

/// Fields Ollama emits on the final (`done: true`) object, flattened into it.
/// All durations are nanoseconds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Timings {
    #[serde(default)]
    pub total_duration: u64,
    #[serde(default)]
    pub load_duration: u64,
    #[serde(default)]
    pub prompt_eval_count: u32,
    #[serde(default)]
    pub prompt_eval_duration: u64,
    #[serde(default)]
    pub eval_count: u32,
    #[serde(default)]
    pub eval_duration: u64,
}

/// Keys that make a flattened object a [`Timings`] block rather than just
/// unrelated unknown fields.
const TIMING_KEYS: [&str; 6] = [
    "total_duration",
    "load_duration",
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
];

/// `#[serde(flatten)] Option<T>` cannot express "absent" on its own: serde hands
/// the flattened map to `Option::deserialize`, which always takes the `Some`
/// branch and then fails if `T` needs fields the map lacks. So we look at the
/// leftover keys ourselves and only build a [`Timings`] when at least one
/// timing key is actually on the wire — which is exactly Ollama's rule (they
/// appear on the final object only).
fn deserialize_timings<'de, D>(de: D) -> Result<Option<Timings>, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error;

    let rest = serde_json::Map::<String, serde_json::Value>::deserialize(de)?;
    if !TIMING_KEYS.iter().any(|k| rest.contains_key(*k)) {
        return Ok(None);
    }
    serde_json::from_value(serde_json::Value::Object(rest)).map(Some).map_err(D::Error::custom)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ChatResponse {
    #[serde(default)]
    pub model: String,
    /// RFC3339, nanosecond precision — see [`crate::now_rfc3339`].
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub message: Message,
    #[serde(default)]
    pub done: bool,
    /// `"stop"` | `"load"` | `"unload"` | `"length"`; final object only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    #[serde(
        flatten,
        default,
        deserialize_with = "deserialize_timings",
        skip_serializing_if = "Option::is_none"
    )]
    pub timings: Option<Timings>,
}

impl ChatResponse {
    /// One streamed token: `done:false`, no `done_reason`, no timings.
    pub fn token(model: impl Into<String>, created_at: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            created_at: created_at.into(),
            message: Message::assistant(text),
            done: false,
            done_reason: None,
            timings: None,
        }
    }

    /// The terminating object: empty message, `done:true`, `done_reason`, timings.
    pub fn final_(
        model: impl Into<String>,
        created_at: impl Into<String>,
        done_reason: impl Into<String>,
        timings: Timings,
    ) -> Self {
        Self {
            model: model.into(),
            created_at: created_at.into(),
            message: Message::assistant(""),
            done: true,
            done_reason: Some(done_reason.into()),
            timings: Some(timings),
        }
    }

    /// The whole answer in one object, for `"stream": false`.
    pub fn complete(
        model: impl Into<String>,
        created_at: impl Into<String>,
        text: impl Into<String>,
        done_reason: impl Into<String>,
        timings: Timings,
    ) -> Self {
        Self {
            model: model.into(),
            created_at: created_at.into(),
            message: Message::assistant(text),
            done: true,
            done_reason: Some(done_reason.into()),
            timings: Some(timings),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct GenerateResponse {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub response: String,
    #[serde(default)]
    pub done: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    #[serde(
        flatten,
        default,
        deserialize_with = "deserialize_timings",
        skip_serializing_if = "Option::is_none"
    )]
    pub timings: Option<Timings>,
}

impl GenerateResponse {
    pub fn token(model: impl Into<String>, created_at: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            created_at: created_at.into(),
            response: text.into(),
            done: false,
            done_reason: None,
            timings: None,
        }
    }

    pub fn final_(
        model: impl Into<String>,
        created_at: impl Into<String>,
        done_reason: impl Into<String>,
        timings: Timings,
    ) -> Self {
        Self {
            model: model.into(),
            created_at: created_at.into(),
            response: String::new(),
            done: true,
            done_reason: Some(done_reason.into()),
            timings: Some(timings),
        }
    }

    pub fn complete(
        model: impl Into<String>,
        created_at: impl Into<String>,
        text: impl Into<String>,
        done_reason: impl Into<String>,
        timings: Timings,
    ) -> Self {
        Self {
            model: model.into(),
            created_at: created_at.into(),
            response: text.into(),
            done: true,
            done_reason: Some(done_reason.into()),
            timings: Some(timings),
        }
    }
}

// ---------------------------------------------------------------------------
// POST /api/show
// ---------------------------------------------------------------------------

/// Ollama accepts both the current `model` and the legacy `name`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ShowRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ShowRequest {
    /// The model the caller meant: `model` wins, `name` is the legacy fallback.
    pub fn resolved(&self) -> Option<&str> {
        self.model.as_deref().or(self.name.as_deref())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ShowResponse {
    #[serde(default)]
    pub modelfile: String,
    #[serde(default)]
    pub parameters: String,
    #[serde(default)]
    pub template: String,
    #[serde(default)]
    pub details: ModelDetails,
    #[serde(default)]
    pub model_info: BTreeMap<String, serde_json::Value>,
    /// `["completion"]` for a plain text model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<Vec<String>>,
}

// ---------------------------------------------------------------------------
// GET /api/ps
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct PsModel {
    pub name: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub digest: String,
    #[serde(default)]
    pub details: ModelDetails,
    /// RFC3339 — when the model is unloaded if untouched.
    #[serde(default)]
    pub expires_at: String,
    #[serde(default)]
    pub size_vram: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct PsResponse {
    #[serde(default, deserialize_with = "null_to_default")]
    pub models: Vec<PsModel>,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Ollama's error shape: a bare string. PAIR's own proxies use the same shape
/// on *both* lanes (`services/ollama-proxy/failover_test.go:222`, `:267`, `:333`;
/// `services/ollama-proxy/proxy.go:1106`, `:1177`).
///
/// PAIR reads only the status code from a node: 404 → fail over to the next
/// owner, 503 → retry elsewhere (`docs/pair-contract.md` §3.1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ErrorResponse {
    pub error: String,
}

impl ErrorResponse {
    pub fn new(error: impl Into<String>) -> Self {
        Self { error: error.into() }
    }

    /// The 404 body for a model this node does not host.
    pub fn model_not_found(model: &str) -> Self {
        Self::new(format!("model '{model}' not found"))
    }
}

// ---------------------------------------------------------------------------
// NDJSON framing
// ---------------------------------------------------------------------------

/// `application/x-ndjson` framing for the streaming halves of `/api/chat` and
/// `/api/generate`: one compact JSON object per line, terminated by `\n`.
pub mod ndjson {
    use serde::Serialize;

    /// Content type to set on a streaming Ollama response.
    pub const CONTENT_TYPE: &str = "application/x-ndjson";

    /// `<compact json>\n`. `serde_json` never emits a raw newline inside the
    /// object, so the result is always exactly one line.
    pub fn encode_line<T: Serialize>(value: &T) -> String {
        let mut s = serde_json::to_string(value).expect("value is serialisable");
        s.push('\n');
        s
    }
}
