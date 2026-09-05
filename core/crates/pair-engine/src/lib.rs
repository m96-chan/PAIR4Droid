//! Inference engine abstraction.
//!
//! `pair-node` only ever talks to [`Engine`]; concrete backends are:
//! - [`mock::MockEngine`] – deterministic, dependency-free, used by every test.
//! - `llama::LlamaEngine` – llama.cpp via `llama-cpp-2` (cargo feature `llama`).
//!
//! Design contract (do not change signatures without updating pair-node):
//! - One engine instance owns a *catalogue* of models (GGUF files) and at most
//!   one loaded model at a time (phones are RAM-limited). Loading is lazy on
//!   first request; `unload()` frees it.
//! - Generation is a stream of [`TokenEvent`]s; dropping the stream cancels.
//! - Concurrency: `Engine` decides. `MockEngine` is unlimited; `LlamaEngine`
//!   serialises requests (queue) and reports `pending()` for telemetry.
//! - No HTTP types here. Prompt formatting (chat template) is the engine's job.

#[cfg(feature = "llama")]
pub mod llama;
pub mod mock;

use futures::stream::BoxStream;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    /// Public model name as advertised to PAIR (exact-match routing key),
    /// e.g. `"qwen2.5-1.5b-instruct-q4_k_m"`. Should be unique across the fleet.
    pub name: String,
    /// Absolute path of the GGUF file (empty for mock models).
    pub path: String,
    pub size_bytes: u64,
    /// e.g. "qwen2", "llama"
    pub family: String,
    /// e.g. "1.5B"
    pub parameter_size: String,
    /// e.g. "Q4_K_M"
    pub quantization: String,
    /// Context length the model will be loaded with.
    pub context_length: u32,
    /// Stable digest for `/api/tags` (sha256 of file, or of name for mock).
    pub digest: String,
    /// RFC3339 timestamp of the file.
    pub modified_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerationParams {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop: Vec<String>,
    pub seed: Option<u64>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self { max_tokens: None, temperature: None, top_p: None, stop: Vec::new(), seed: None }
    }
}

/// A chat request (messages → assistant reply). `/api/generate` and
/// `/v1/completions` are mapped onto this by the caller (single user message,
/// optional system, `raw` = no template).
#[derive(Debug, Clone, PartialEq)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub params: GenerationParams,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenEvent {
    /// Emitted once before the first token, after the prompt is processed.
    Start { prompt_tokens: u32 },
    /// A piece of decoded text (may be a partial UTF-8 grapheme joined by the engine; always valid UTF-8).
    Token(String),
    /// Terminal event; always emitted exactly once unless the stream errors.
    Done {
        finish_reason: FinishReason,
        prompt_tokens: u32,
        completion_tokens: u32,
        load_ms: u64,
        prompt_ms: u64,
        eval_ms: u64,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("model failed to load: {0}")]
    LoadFailed(String),
    #[error("engine is busy")]
    Busy,
    #[error("context length exceeded: prompt {prompt_tokens} > ctx {context_length}")]
    ContextExceeded { prompt_tokens: u32, context_length: u32 },
    #[error("generation failed: {0}")]
    Generation(String),
    #[error("cancelled")]
    Cancelled,
}

pub type TokenStream = BoxStream<'static, Result<TokenEvent, EngineError>>;

/// Snapshot for telemetry / `/api/ps`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EngineStatus {
    pub loaded_model: Option<String>,
    pub loaded_bytes: u64,
    /// Requests currently generating.
    pub active: u32,
    /// Requests waiting for a slot.
    pub queued: u32,
}

#[async_trait::async_trait]
pub trait Engine: Send + Sync + 'static {
    /// Models available on this node (the catalogue), regardless of load state.
    async fn list_models(&self) -> Vec<ModelInfo>;
    async fn model(&self, name: &str) -> Option<ModelInfo>;
    /// Start generating. Returns an error immediately for unknown model / load failure;
    /// otherwise a stream that yields `Start`, `Token*`, `Done`.
    async fn chat(&self, req: ChatRequest) -> Result<TokenStream, EngineError>;
    /// Tokenise for accounting (used for `prompt_eval_count` when no generation happens).
    async fn count_tokens(&self, model: &str, text: &str) -> Result<u32, EngineError>;
    async fn unload(&self);
    fn status(&self) -> EngineStatus;
}

pub type SharedEngine = Arc<dyn Engine>;
