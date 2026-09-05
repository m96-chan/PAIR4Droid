//! Deterministic mock engine for tests and for running the node without a model.
//!
//! Behaviour contract (ticket #6):
//! - [`MockEngine::with_models`] advertises those names; `chat` on an unknown name
//!   → [`EngineError::ModelNotFound`].
//! - Reply text is deterministic: `"echo: " + <last user message>` (or `"echo:"`
//!   when there is no user message), split on whitespace, one [`TokenEvent::Token`]
//!   per word; every token after the first carries a single leading space so that
//!   concatenating the tokens reproduces the reply verbatim.
//! - `max_tokens` truncates → [`FinishReason::Length`]; a `stop` string ends the
//!   generation early → [`FinishReason::Stop`].
//! - [`MockEngine::with_token_delay`] inserts a sleep between tokens so streaming
//!   tests can observe chunk boundaries.
//! - The producer runs in a spawned tokio task feeding an mpsc channel; dropping
//!   the stream drops the receiver, the next send fails and the task stops.
//! - `status().active` counts live streams; `loaded_model` is the last model used.

use crate::*;

use futures::StreamExt;
use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

/// Context length advertised for every mock model.
pub const MOCK_CONTEXT_LENGTH: u32 = 4096;
/// Prefix of every mock reply.
pub const MOCK_REPLY_PREFIX: &str = "echo:";
/// Model advertised by [`MockEngine::default`].
pub const MOCK_DEFAULT_MODEL: &str = "mock";

#[derive(Debug, Default)]
struct State {
    loaded_model: Option<String>,
    active: u32,
}

#[derive(Debug)]
struct Inner {
    models: Vec<ModelInfo>,
    state: Mutex<State>,
}

impl Inner {
    fn find(&self, name: &str) -> Option<&ModelInfo> {
        self.models.iter().find(|m| m.name == name)
    }
}

/// Decrements `active` however the producer task ends (normal finish, early
/// `break` on a closed channel, or a panic).
struct ActiveGuard(Arc<Inner>);

impl ActiveGuard {
    fn acquire(inner: Arc<Inner>) -> Self {
        inner.state.lock().active += 1;
        Self(inner)
    }
}

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        let mut st = self.0.state.lock();
        st.active = st.active.saturating_sub(1);
    }
}

/// Deterministic, dependency-free [`Engine`] used by every test and by
/// `pair4droid serve --mock`.
#[derive(Debug, Clone)]
pub struct MockEngine {
    inner: Arc<Inner>,
    token_delay: Duration,
}

impl Default for MockEngine {
    fn default() -> Self {
        Self::with_models(&[MOCK_DEFAULT_MODEL])
    }
}

impl MockEngine {
    /// Advertise `names` as the catalogue. Digests are the sha256 of the name so
    /// they are stable across runs; `modified_at` is the construction time.
    pub fn with_models(names: &[&str]) -> Self {
        let modified_at = chrono::Utc::now().to_rfc3339();
        let models = names
            .iter()
            .map(|name| ModelInfo {
                name: (*name).to_string(),
                path: String::new(),
                size_bytes: 0,
                family: "mock".to_string(),
                parameter_size: "0B".to_string(),
                quantization: "none".to_string(),
                context_length: MOCK_CONTEXT_LENGTH,
                digest: sha256_hex(name.as_bytes()),
                modified_at: modified_at.clone(),
            })
            .collect();
        Self {
            inner: Arc::new(Inner { models, state: Mutex::new(State::default()) }),
            token_delay: Duration::ZERO,
        }
    }

    /// Sleep this long between tokens so streaming tests can observe chunk
    /// boundaries (and so cancellation has something to cancel).
    #[must_use]
    pub fn with_token_delay(mut self, delay: Duration) -> Self {
        self.token_delay = delay;
        self
    }

    /// The reply this engine would produce for `messages` (without tokenisation).
    pub fn reply_text(messages: &[ChatMessage]) -> String {
        match last_user_message(messages) {
            Some(content) if !content.trim().is_empty() => {
                format!("{MOCK_REPLY_PREFIX} {}", content.trim())
            }
            _ => MOCK_REPLY_PREFIX.to_string(),
        }
    }
}

/// `"echo: a b"` → `["echo:", " a", " b"]`, so `tokens.concat() == text`.
fn split_tokens(text: &str) -> Vec<String> {
    text.split_whitespace()
        .enumerate()
        .map(|(i, w)| if i == 0 { w.to_string() } else { format!(" {w}") })
        .collect()
}

fn last_user_message(messages: &[ChatMessage]) -> Option<&str> {
    messages.iter().rev().find(|m| m.role == ChatRole::User).map(|m| m.content.as_str())
}

/// The mock tokeniser: whitespace-separated words.
pub fn word_count(text: &str) -> u32 {
    text.split_whitespace().count() as u32
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}

#[async_trait::async_trait]
impl Engine for MockEngine {
    async fn list_models(&self) -> Vec<ModelInfo> {
        self.inner.models.clone()
    }

    async fn model(&self, name: &str) -> Option<ModelInfo> {
        self.inner.find(name).cloned()
    }

    async fn chat(&self, req: ChatRequest) -> Result<TokenStream, EngineError> {
        if self.inner.find(&req.model).is_none() {
            return Err(EngineError::ModelNotFound(req.model));
        }
        let started = Instant::now();
        self.inner.state.lock().loaded_model = Some(req.model.clone());

        let prompt_tokens: u32 = req.messages.iter().map(|m| word_count(&m.content)).sum();
        let all_tokens = split_tokens(&MockEngine::reply_text(&req.messages));
        let params = req.params;
        let delay = self.token_delay;
        let inner = Arc::clone(&self.inner);

        // Bounded so a slow consumer back-pressures the producer, and so a
        // dropped receiver is noticed on the very next send.
        let (tx, rx) = mpsc::channel::<Result<TokenEvent, EngineError>>(1);

        tokio::spawn(async move {
            {
                let _guard = ActiveGuard::acquire(inner);
                produce(&tx, started, prompt_tokens, all_tokens, params, delay).await;
            }
            // `active` is already back to 0 when the consumer observes the end
            // of the stream: the receiver only sees the close on this drop.
            drop(tx);
        });

        Ok(ReceiverStream::new(rx).boxed())
    }

    async fn count_tokens(&self, model: &str, text: &str) -> Result<u32, EngineError> {
        if self.inner.find(model).is_none() {
            return Err(EngineError::ModelNotFound(model.to_string()));
        }
        Ok(word_count(text))
    }

    async fn unload(&self) {
        self.inner.state.lock().loaded_model = None;
    }

    fn status(&self) -> EngineStatus {
        let st = self.inner.state.lock();
        EngineStatus { loaded_model: st.loaded_model.clone(), loaded_bytes: 0, active: st.active, queued: 0 }
    }
}

/// Emit `Start` → `Token*` → `Done`. Returns early (without `Done`) as soon as
/// the receiver is gone, which is how a dropped stream cancels the generation.
async fn produce(
    tx: &mpsc::Sender<Result<TokenEvent, EngineError>>,
    started: Instant,
    prompt_tokens: u32,
    all_tokens: Vec<String>,
    params: GenerationParams,
    delay: Duration,
) {
    if tx.send(Ok(TokenEvent::Start { prompt_tokens })).await.is_err() {
        return;
    }
    let prompt_ms = started.elapsed().as_millis() as u64;
    let eval_start = Instant::now();

    let limit = params.max_tokens.unwrap_or(u32::MAX) as usize;
    let stops: Vec<&String> = params.stop.iter().filter(|s| !s.is_empty()).collect();

    let mut emitted = 0u32;
    let mut text = String::new();
    let mut finish_reason = FinishReason::Stop;

    for piece in &all_tokens {
        if emitted as usize >= limit {
            finish_reason = FinishReason::Length;
            break;
        }
        // A stop sequence ends the generation *before* the offending piece.
        let candidate = format!("{text}{piece}");
        if stops.iter().any(|s| candidate.contains(s.as_str())) {
            break;
        }
        if emitted > 0 && !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        if tx.send(Ok(TokenEvent::Token(piece.clone()))).await.is_err() {
            return; // stream dropped → cancelled
        }
        text = candidate;
        emitted += 1;
    }

    let _ = tx
        .send(Ok(TokenEvent::Done {
            finish_reason,
            prompt_tokens,
            completion_tokens: emitted,
            load_ms: 0,
            prompt_ms,
            eval_ms: eval_start.elapsed().as_millis() as u64,
        }))
        .await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_concatenate_back_to_the_text() {
        let t = split_tokens("echo: a b");
        assert_eq!(t, ["echo:", " a", " b"]);
        assert_eq!(t.concat(), "echo: a b");
        assert!(split_tokens("   ").is_empty());
    }

    #[test]
    fn reply_uses_the_last_user_message() {
        let msgs = vec![
            ChatMessage { role: ChatRole::User, content: "one".into() },
            ChatMessage { role: ChatRole::Assistant, content: "two".into() },
            ChatMessage { role: ChatRole::User, content: "three".into() },
        ];
        assert_eq!(MockEngine::reply_text(&msgs), "echo: three");
        assert_eq!(MockEngine::reply_text(&[]), "echo:");
        let only_system = vec![ChatMessage { role: ChatRole::System, content: "s".into() }];
        assert_eq!(MockEngine::reply_text(&only_system), "echo:");
    }

    #[test]
    fn word_count_counts_whitespace_separated_words() {
        assert_eq!(word_count("a  b\nc\td"), 4);
        assert_eq!(word_count(""), 0);
    }

    #[test]
    fn digest_is_sha256_hex() {
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }
}
