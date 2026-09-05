//! Ollama-compatible lane on `:11434` (what PAIR's `ollama-proxy` forwards).
//!
//! TODO(ticket: protocol/ollama): implement + tests.
//! Required surface (see docs/pair-contract.md §3):
//! - `GET  /`             → `200 "Ollama is running"` (liveness; PAIR requires 200)
//! - `GET  /api/tags`     → [`TagsResponse`] (`models[].name` is all PAIR reads for the list)
//! - `GET  /api/version`  → [`VersionResponse`]
//! - `POST /api/chat`     → [`ChatRequest`] → NDJSON stream of [`ChatResponse`] (or single object when `stream:false`)
//! - `POST /api/generate` → [`GenerateRequest`] → NDJSON stream of [`GenerateResponse`]
//! - `POST /api/show`     → [`ShowRequest`] → [`ShowResponse`]
//! - `GET  /api/ps`       → [`PsResponse`]
//!
//! Streaming in Ollama is `application/x-ndjson`, one JSON object per line, the
//! final object has `"done": true` plus timing fields.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ModelDetails {
    #[serde(default)]
    pub parent_model: String,
    #[serde(default)]
    pub format: String, // "gguf"
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub families: Option<Vec<String>>,
    #[serde(default)]
    pub parameter_size: String, // e.g. "1.5B"
    #[serde(default)]
    pub quantization_level: String, // e.g. "Q4_K_M"
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TagModel {
    pub name: String,
    pub model: String,
    pub modified_at: String, // RFC3339
    pub size: u64,
    pub digest: String,
    pub details: ModelDetails,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TagsResponse {
    pub models: Vec<TagModel>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VersionResponse {
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: String, // "system" | "user" | "assistant" | "tool"
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
}

/// Subset of Ollama `options` we honour; everything else is ignored.
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
    #[serde(default)]
    pub messages: Vec<Message>,
    /// Ollama defaults to streaming when absent.
    #[serde(default = "default_true")]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<Options>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: Message,
    pub done: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    #[serde(flatten, default, skip_serializing_if = "Option::is_none")]
    pub timings: Option<Timings>,
}

/// Fields Ollama emits on the final (`done: true`) object. Durations in ns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Timings {
    pub total_duration: u64,
    pub load_duration: u64,
    pub prompt_eval_count: u32,
    pub prompt_eval_duration: u64,
    pub eval_count: u32,
    pub eval_duration: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub model: String,
    #[serde(default)]
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(default = "default_true")]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<Options>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    #[serde(flatten, default, skip_serializing_if = "Option::is_none")]
    pub timings: Option<Timings>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShowRequest {
    /// Ollama accepts both `model` and legacy `name`.
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ShowResponse {
    #[serde(default)]
    pub modelfile: String,
    #[serde(default)]
    pub parameters: String,
    #[serde(default)]
    pub template: String,
    pub details: ModelDetails,
    #[serde(default)]
    pub model_info: BTreeMap<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<Vec<String>>, // ["completion"]
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PsModel {
    pub name: String,
    pub model: String,
    pub size: u64,
    pub digest: String,
    pub details: ModelDetails,
    pub expires_at: String,
    pub size_vram: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct PsResponse {
    pub models: Vec<PsModel>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
}

fn default_true() -> bool {
    true
}
