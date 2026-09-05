//! Deterministic mock engine for tests and for running the node without a model.
//!
//! TODO(ticket: engine/mock): implement.
//! Behaviour contract:
//! - `MockEngine::with_models(&["a","b"])` advertises those names; `chat` on an
//!   unknown name → `EngineError::ModelNotFound`.
//! - Reply text is deterministic: echo of the last user message prefixed with
//!   `"echo: "`, split into whitespace tokens, one `Token` per word (with a
//!   leading space between words). `max_tokens` truncates → `FinishReason::Length`.
//! - Optional per-token delay (`with_token_delay`) so streaming tests can observe
//!   chunk boundaries; cancellation on drop must stop the producer task.
//! - `status()` reflects active count while a stream is alive.

#[allow(unused_imports)]
use crate::*;

pub struct MockEngine {
    // TODO
}

impl MockEngine {
    pub fn with_models(names: &[&str]) -> Self {
        let _ = names;
        todo!("ticket engine/mock")
    }
}
