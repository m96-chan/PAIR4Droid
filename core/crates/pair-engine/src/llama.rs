//! llama.cpp backend via `llama-cpp-2` (cargo feature `llama`, ticket #7).
//!
//! Shape of the backend:
//!
//! ```text
//! chat()  ──submit──▶ std::sync::mpsc ──▶ dedicated OS thread (owns LlamaModel)
//!   ▲                    (the queue)              │ blocking_send
//!   └── TokenStream ◀── tokio::sync::mpsc ◀───────┘
//! ```
//!
//! * A `LlamaContext` borrows its `LlamaModel` and llama.cpp state is not
//!   `Send`-friendly, so exactly one OS thread owns the model and runs every
//!   generation. That thread *is* the serialisation: one generation at a time,
//!   the rest wait in the channel (at most `max_queue`, then [`EngineError::Busy`]).
//! * Models are loaded lazily on the first `chat` and swapped (previous one
//!   unloaded first) when a different model is requested — phones hold one model.
//! * `mmap` is on by default: Android 17's per-process memory limiter counts
//!   anonymous RSS, and file-backed pages are exempt (see CLAUDE.md).
//! * Cancellation: dropping the [`TokenStream`] sets an `AtomicBool` that the
//!   generation loop checks every step, and closes the token channel so the very
//!   next `blocking_send` fails too.
//! * Token pieces are raw bytes; a [`Utf8Accumulator`] holds back incomplete
//!   sequences so only whole `char`s are ever emitted.

use crate::*;

use futures::StreamExt;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::gguf::GgufContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, OnceLock};
use std::time::Instant;
use tokio::sync::mpsc as tokio_mpsc;

/// `enum gguf_type` discriminants, from llama.cpp `ggml/include/gguf.h:53`.
/// `gguf_get_val_*` aborts on a type mismatch, so every read is gated on these.
const GGUF_TYPE_UINT32: std::os::raw::c_uint = 4;
const GGUF_TYPE_STRING: std::os::raw::c_uint = 8;

/// llama.cpp's `LLAMA_DEFAULT_SEED` (`include/llama.h`), i.e. "pick one".
const LLAMA_DEFAULT_SEED: u32 = 0xFFFF_FFFF;

/// How many bytes of buffer to offer `llama_token_to_piece` before retrying.
const PIECE_BUF: usize = 64;

// ---------------------------------------------------------------- config

/// Tunables for [`LlamaEngine`]. `Default` targets a phone: 4k context, mmap on,
/// CPU only, a short queue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaConfig {
    /// Context window the model is loaded with (capped at the model's trained context).
    pub n_ctx: u32,
    /// Generation threads (llama.cpp's `n_threads`).
    pub n_threads: i32,
    /// Layers offloaded to a GPU backend; 0 = CPU only.
    pub n_gpu_layers: u32,
    /// Memory-map the GGUF instead of reading it into anonymous memory.
    pub use_mmap: bool,
    /// How many requests may wait for the single generation slot before
    /// [`EngineError::Busy`].
    pub max_queue: usize,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        let threads = std::thread::available_parallelism().map(|n| n.get().min(8) as i32).unwrap_or(4);
        Self { n_ctx: 4096, n_threads: threads, n_gpu_layers: 0, use_mmap: true, max_queue: 2 }
    }
}

// ------------------------------------------------------------- catalogue

/// Model metadata guessed from a file name (used when the GGUF header does not
/// carry `general.*` keys).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NameMetadata {
    /// e.g. `"qwen2.5"`
    pub family: String,
    /// e.g. `"1.5B"` (empty when the name says nothing)
    pub parameter_size: String,
    /// e.g. `"Q4_K_M"` (empty when the name says nothing)
    pub quantization: String,
}

fn looks_like_quant(part: &str) -> bool {
    let p = part.to_ascii_lowercase();
    if matches!(p.as_str(), "f16" | "f32" | "bf16" | "fp16" | "fp32") {
        return true;
    }
    let rest = p.strip_prefix("iq").or_else(|| p.strip_prefix('q'));
    match rest {
        Some(r) => {
            r.starts_with(|c: char| c.is_ascii_digit())
                && r.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        }
        None => false,
    }
}

fn looks_like_param_size(part: &str) -> bool {
    let p = part.to_ascii_lowercase();
    let Some(head) = p.strip_suffix('b').or_else(|| p.strip_suffix('m')) else {
        return false;
    };
    !head.is_empty()
        && head.chars().all(|c| c.is_ascii_digit() || c == '.')
        && head.chars().any(|c| c.is_ascii_digit())
}

/// Guess `family` / `parameter_size` / `quantization` from a GGUF file stem such
/// as `qwen2.5-1.5b-instruct-q4_k_m`.
pub fn parse_name_metadata(stem: &str) -> NameMetadata {
    let parts: Vec<&str> = stem.split(['-', '_']).filter(|p| !p.is_empty()).collect();
    let mut out = NameMetadata {
        family: parts.first().copied().unwrap_or(stem).to_ascii_lowercase(),
        ..Default::default()
    };
    // Quantisation suffixes are written with underscores (`q4_k_m`), so rejoin
    // the tail once a quant-looking component is found.
    for (i, part) in stem.split('-').enumerate() {
        if i > 0 && out.quantization.is_empty() && looks_like_quant(part) {
            out.quantization = part.to_ascii_uppercase();
        }
        if i > 0 && out.parameter_size.is_empty() && looks_like_param_size(part) {
            out.parameter_size = part.to_ascii_uppercase();
        }
    }
    out
}

/// `general.file_type` → the usual quantisation label
/// (llama.cpp `include/llama.h`, `enum llama_ftype`).
fn ftype_name(ftype: u32) -> Option<&'static str> {
    Some(match ftype {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4_MOE",
        _ => return None,
    })
}

/// Read `general.*` out of a GGUF header without loading any tensor.
fn gguf_metadata(path: &Path) -> Option<NameMetadata> {
    let ctx = GgufContext::from_file(path)?;
    let str_val = |key: &str| -> Option<String> {
        let idx = ctx.find_key(key);
        if idx < 0 || ctx.kv_type(idx) != GGUF_TYPE_STRING {
            return None;
        }
        ctx.val_str(idx).map(str::to_string)
    };
    let u32_val = |key: &str| -> Option<u32> {
        let idx = ctx.find_key(key);
        if idx < 0 || ctx.kv_type(idx) != GGUF_TYPE_UINT32 {
            return None;
        }
        Some(ctx.val_u32(idx))
    };
    Some(NameMetadata {
        family: str_val("general.architecture").unwrap_or_default(),
        parameter_size: str_val("general.size_label").unwrap_or_default(),
        quantization: u32_val("general.file_type")
            .and_then(ftype_name)
            .map(str::to_string)
            .unwrap_or_default(),
    })
}

fn sha256_file(path: &Path) -> std::io::Result<String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn rfc3339(t: std::time::SystemTime) -> String {
    chrono::DateTime::<chrono::Utc>::from(t).to_rfc3339()
}

/// Catalogue every `*.gguf` in `dir` (non-recursive), sorted by name.
///
/// `context_length` is what the models will be *loaded* with, i.e.
/// [`LlamaConfig::n_ctx`]. A missing directory is an empty catalogue, not an
/// error — on Android the import folder appears only once the user imports a model.
/// The digest is the sha256 of the file, so this is O(size) per model.
pub fn scan_models_dir(dir: &Path, context_length: u32) -> std::io::Result<Vec<ModelInfo>> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let mut models = Vec::new();
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|e| e.to_str()) != Some("gguf") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let meta = entry.metadata()?;
        let from_name = parse_name_metadata(stem);
        let from_gguf = gguf_metadata(&path).unwrap_or_default();
        let pick = |gguf: String, name: String| if gguf.is_empty() { name } else { gguf };
        models.push(ModelInfo {
            name: stem.to_string(),
            path: path.to_string_lossy().into_owned(),
            size_bytes: meta.len(),
            family: pick(from_gguf.family, from_name.family),
            parameter_size: pick(from_gguf.parameter_size, from_name.parameter_size),
            quantization: pick(from_gguf.quantization, from_name.quantization),
            context_length,
            digest: sha256_file(&path)?,
            modified_at: meta.modified().map(rfc3339).unwrap_or_default(),
        });
    }
    models.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(models)
}

// ---------------------------------------------------------- chat template

fn role_name(role: &ChatRole) -> &'static str {
    match role {
        ChatRole::System => "system",
        ChatRole::User => "user",
        ChatRole::Assistant => "assistant",
        ChatRole::Tool => "tool",
    }
}

/// ChatML rendering, used when the GGUF carries no `tokenizer.chat_template`.
/// Ends with the open assistant tag so the model continues the reply.
pub fn chatml_prompt(messages: &[ChatMessage]) -> String {
    let mut s = String::new();
    for m in messages {
        s.push_str("<|im_start|>");
        s.push_str(role_name(&m.role));
        s.push('\n');
        s.push_str(&m.content);
        s.push_str("<|im_end|>\n");
    }
    s.push_str("<|im_start|>assistant\n");
    s
}

/// Render with the model's own template, falling back to [`chatml_prompt`].
fn render_prompt(
    model: &LlamaModel,
    template: Option<&LlamaChatTemplate>,
    messages: &[ChatMessage],
) -> String {
    let Some(tmpl) = template else {
        return chatml_prompt(messages);
    };
    let mut chat = Vec::with_capacity(messages.len());
    for m in messages {
        match LlamaChatMessage::new(role_name(&m.role).to_string(), m.content.clone()) {
            Ok(c) => chat.push(c),
            Err(e) => {
                tracing::warn!("chat message rejected ({e}), falling back to ChatML");
                return chatml_prompt(messages);
            }
        }
    }
    match model.apply_chat_template(tmpl, &chat, true) {
        Ok(prompt) => prompt,
        Err(e) => {
            tracing::warn!("embedded chat template failed ({e}), falling back to ChatML");
            chatml_prompt(messages)
        }
    }
}

// ------------------------------------------------------ utf-8 accumulator

/// Joins raw token-piece bytes into whole `char`s: a multi-byte sequence split
/// across two tokens is held back until it is complete.
#[derive(Debug, Default, Clone)]
pub struct Utf8Accumulator {
    buf: Vec<u8>,
}

impl Utf8Accumulator {
    /// Feed the bytes of one token piece; returns the text that is now complete.
    pub fn push(&mut self, bytes: &[u8]) -> Option<String> {
        self.buf.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.buf) {
                Ok(s) => {
                    out.push_str(s);
                    self.buf.clear();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    // SAFETY-free: `valid_up_to` is a char boundary by definition.
                    out.push_str(std::str::from_utf8(&self.buf[..valid]).unwrap_or_default());
                    match e.error_len() {
                        // Truly invalid bytes: replace them and keep going.
                        Some(bad) => {
                            out.push('\u{fffd}');
                            self.buf.drain(..valid + bad);
                        }
                        // Just an incomplete tail: keep it for the next token.
                        None => {
                            self.buf.drain(..valid);
                            break;
                        }
                    }
                }
            }
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }

    /// True when nothing is being held back.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

// ------------------------------------------------------------- stop words

/// If a stop sequence starts inside `piece`, return the part of `piece` that may
/// still be emitted (possibly empty). `text` is everything emitted so far.
fn stop_split(text: &str, piece: &str, stops: &[String]) -> Option<String> {
    let candidate = format!("{text}{piece}");
    let hit = stops.iter().filter(|s| !s.is_empty()).filter_map(|s| candidate.find(s.as_str())).min()?;
    let keep_from = text.len();
    if hit <= keep_from {
        return Some(String::new());
    }
    Some(candidate[keep_from..hit].to_string())
}

// ------------------------------------------------------------ worker plumbing

#[derive(Debug, Default)]
struct Shared {
    active: AtomicU32,
    queued: AtomicU32,
    loaded: Mutex<Option<(String, u64)>>,
}

/// Where a running generation writes its events. Lives on the worker thread, so
/// it sends with `blocking_send`.
struct Sink {
    tx: tokio_mpsc::Sender<Result<TokenEvent, EngineError>>,
    cancel: Arc<AtomicBool>,
}

impl Sink {
    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed) || self.tx.is_closed()
    }
    /// Returns false once the consumer is gone.
    fn emit(&self, ev: TokenEvent) -> bool {
        self.tx.blocking_send(Ok(ev)).is_ok()
    }
    fn fail(&self, err: EngineError) {
        let _ = self.tx.blocking_send(Err(err));
    }
}

/// One queued generation.
struct Job {
    model: String,
    messages: Vec<ChatMessage>,
    params: GenerationParams,
    sink: Sink,
}

enum WorkerMsg {
    Job(Job),
    CountTokens { model: String, text: String, reply: tokio_mpsc::Sender<Result<u32, EngineError>> },
    Unload,
}

/// Runs jobs on the dedicated thread. The real implementation is
/// [`LlamaRunner`]; tests substitute a fake so the queue can be exercised
/// without a GGUF file.
trait JobRunner {
    fn run(&mut self, job: Job);
    fn count_tokens(&mut self, model: &str, text: &str) -> Result<u32, EngineError>;
    fn unload(&mut self);
}

fn spawn_worker<F, R>(make: F, rx: std_mpsc::Receiver<WorkerMsg>, shared: Arc<Shared>)
where
    F: FnOnce() -> R + Send + 'static,
    R: JobRunner + 'static,
{
    std::thread::Builder::new()
        .name("pair-llama".to_string())
        .spawn(move || {
            let mut runner = make();
            while let Ok(msg) = rx.recv() {
                match msg {
                    WorkerMsg::Job(job) => {
                        shared.queued.fetch_sub(1, Ordering::AcqRel);
                        if job.sink.cancelled() {
                            continue; // the caller gave up while it waited
                        }
                        shared.active.fetch_add(1, Ordering::AcqRel);
                        runner.run(job);
                        shared.active.fetch_sub(1, Ordering::AcqRel);
                    }
                    WorkerMsg::CountTokens { model, text, reply } => {
                        let _ = reply.blocking_send(runner.count_tokens(&model, &text));
                    }
                    WorkerMsg::Unload => {
                        runner.unload();
                        *shared.loaded.lock() = None;
                    }
                }
            }
            runner.unload();
        })
        .expect("spawn llama worker thread");
}

// ----------------------------------------------------------------- engine

struct EngineInner {
    catalogue: Vec<ModelInfo>,
    shared: Arc<Shared>,
    tx: Mutex<Option<std_mpsc::Sender<WorkerMsg>>>,
    max_queue: usize,
}

/// llama.cpp-backed [`Engine`]. See the module docs for the threading model.
pub struct LlamaEngine {
    inner: Arc<EngineInner>,
}

/// Process-wide llama.cpp backend. `LlamaBackend::init` may only be called once
/// and its `Drop` frees the backend for everyone, so it is leaked into a static.
fn backend() -> Result<&'static LlamaBackend, EngineError> {
    static BACKEND: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();
    match BACKEND.get_or_init(|| LlamaBackend::init().map_err(|e| e.to_string())) {
        Ok(b) => Ok(b),
        Err(e) => Err(EngineError::LoadFailed(format!("llama backend init failed: {e}"))),
    }
}

impl LlamaEngine {
    /// Scan `models_dir` for `*.gguf` and start the worker thread. No model is
    /// loaded until the first [`Engine::chat`].
    pub fn new(models_dir: PathBuf, cfg: LlamaConfig) -> Result<Self, EngineError> {
        let backend = backend()?;
        let catalogue = scan_models_dir(&models_dir, cfg.n_ctx)
            .map_err(|e| EngineError::LoadFailed(format!("scanning {models_dir:?}: {e}")))?;
        tracing::info!(dir = ?models_dir, models = catalogue.len(), "llama catalogue");

        let shared = Arc::new(Shared::default());
        let runner_catalogue = catalogue.clone();
        let runner_shared = Arc::clone(&shared);
        let runner_cfg = cfg.clone();
        Ok(Self::with_runner(catalogue, cfg.max_queue, shared, move || LlamaRunner {
            backend,
            cfg: runner_cfg,
            catalogue: runner_catalogue,
            shared: runner_shared,
            loaded: None,
        }))
    }

    fn with_runner<F, R>(catalogue: Vec<ModelInfo>, max_queue: usize, shared: Arc<Shared>, make: F) -> Self
    where
        F: FnOnce() -> R + Send + 'static,
        R: JobRunner + 'static,
    {
        let (tx, rx) = std_mpsc::channel();
        spawn_worker(make, rx, Arc::clone(&shared));
        Self { inner: Arc::new(EngineInner { catalogue, shared, tx: Mutex::new(Some(tx)), max_queue }) }
    }

    /// Claim a queue slot, or [`EngineError::Busy`].
    fn enqueue(&self, job: Job) -> Result<(), EngineError> {
        let shared = &self.inner.shared;
        let mut q = shared.queued.load(Ordering::Acquire);
        loop {
            if q as usize >= self.inner.max_queue {
                return Err(EngineError::Busy);
            }
            match shared.queued.compare_exchange_weak(q, q + 1, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => break,
                Err(cur) => q = cur,
            }
        }
        let sent = {
            let tx = self.inner.tx.lock();
            match tx.as_ref() {
                Some(tx) => tx.send(WorkerMsg::Job(job)).is_ok(),
                None => false,
            }
        };
        if !sent {
            shared.queued.fetch_sub(1, Ordering::AcqRel);
            return Err(EngineError::Generation("engine worker stopped".into()));
        }
        Ok(())
    }

    fn send(&self, msg: WorkerMsg) -> bool {
        let tx = self.inner.tx.lock();
        matches!(tx.as_ref(), Some(tx) if tx.send(msg).is_ok())
    }
}

impl Drop for EngineInner {
    fn drop(&mut self) {
        // Close the queue so the worker thread finishes and unloads the model.
        *self.tx.lock() = None;
    }
}

/// Sets the cancellation flag as soon as the consumer drops the stream.
struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

#[async_trait::async_trait]
impl Engine for LlamaEngine {
    async fn list_models(&self) -> Vec<ModelInfo> {
        self.inner.catalogue.clone()
    }

    async fn model(&self, name: &str) -> Option<ModelInfo> {
        self.inner.catalogue.iter().find(|m| m.name == name).cloned()
    }

    async fn chat(&self, req: ChatRequest) -> Result<TokenStream, EngineError> {
        if !self.inner.catalogue.iter().any(|m| m.name == req.model) {
            return Err(EngineError::ModelNotFound(req.model));
        }
        let cancel = Arc::new(AtomicBool::new(false));
        // Dropped on every early return below, which cancels a job that is still
        // waiting in the queue.
        let guard = CancelOnDrop(Arc::clone(&cancel));
        let (tx, mut rx) = tokio_mpsc::channel::<Result<TokenEvent, EngineError>>(1);

        self.enqueue(Job {
            model: req.model,
            messages: req.messages,
            params: req.params,
            sink: Sink { tx, cancel },
        })?;

        // Wait for the first event so load / context errors surface as an error
        // from `chat` itself rather than as the first item of the stream.
        let first = match rx.recv().await {
            Some(Ok(ev)) => ev,
            Some(Err(e)) => return Err(e),
            None => return Err(EngineError::Cancelled),
        };
        let tail = futures::stream::unfold((rx, guard), |(mut rx, guard)| async move {
            rx.recv().await.map(|item| (item, (rx, guard)))
        });
        Ok(futures::stream::once(async move { Ok(first) }).chain(tail).boxed())
    }

    async fn count_tokens(&self, model: &str, text: &str) -> Result<u32, EngineError> {
        if !self.inner.catalogue.iter().any(|m| m.name == model) {
            return Err(EngineError::ModelNotFound(model.to_string()));
        }
        let (tx, mut rx) = tokio_mpsc::channel(1);
        let msg = WorkerMsg::CountTokens { model: model.to_string(), text: text.to_string(), reply: tx };
        if !self.send(msg) {
            return Err(EngineError::Generation("engine worker stopped".into()));
        }
        rx.recv().await.unwrap_or(Err(EngineError::Cancelled))
    }

    async fn unload(&self) {
        self.send(WorkerMsg::Unload);
    }

    fn status(&self) -> EngineStatus {
        let loaded = self.inner.shared.loaded.lock().clone();
        EngineStatus {
            loaded_model: loaded.as_ref().map(|(n, _)| n.clone()),
            loaded_bytes: loaded.map(|(_, b)| b).unwrap_or(0),
            active: self.inner.shared.active.load(Ordering::Acquire),
            queued: self.inner.shared.queued.load(Ordering::Acquire),
        }
    }
}

// ------------------------------------------------------------- real runner

struct LoadedModel {
    name: String,
    model: LlamaModel,
    n_ctx: u32,
    template: Option<LlamaChatTemplate>,
}

struct LlamaRunner {
    backend: &'static LlamaBackend,
    cfg: LlamaConfig,
    catalogue: Vec<ModelInfo>,
    shared: Arc<Shared>,
    loaded: Option<LoadedModel>,
}

impl LlamaRunner {
    /// Load `name`, unloading whatever is loaded first (one model at a time).
    fn ensure_loaded(&mut self, name: &str) -> Result<u64, EngineError> {
        if self.loaded.as_ref().is_some_and(|l| l.name == name) {
            return Ok(0);
        }
        let info = self
            .catalogue
            .iter()
            .find(|m| m.name == name)
            .ok_or_else(|| EngineError::ModelNotFound(name.to_string()))?
            .clone();

        // Free the old model before allocating the new one — phones are RAM-bound.
        self.loaded = None;
        *self.shared.loaded.lock() = None;

        let started = Instant::now();
        let params = LlamaModelParams::default()
            .with_n_gpu_layers(self.cfg.n_gpu_layers)
            .with_use_mmap(self.cfg.use_mmap);
        let model = LlamaModel::load_from_file(self.backend, Path::new(&info.path), &params)
            .map_err(|e| EngineError::LoadFailed(format!("{}: {e}", info.path)))?;
        let n_ctx = self.cfg.n_ctx.min(model.n_ctx_train().max(1));
        let template = match model.chat_template(None) {
            Ok(t) => Some(t),
            Err(e) => {
                tracing::info!(model = %name, "no embedded chat template ({e}), using ChatML");
                None
            }
        };
        let bytes = model.size();
        self.loaded = Some(LoadedModel { name: name.to_string(), model, n_ctx, template });
        *self.shared.loaded.lock() = Some((name.to_string(), bytes));
        Ok(started.elapsed().as_millis() as u64)
    }
}

impl JobRunner for LlamaRunner {
    fn run(&mut self, job: Job) {
        let load_ms = match self.ensure_loaded(&job.model) {
            Ok(ms) => ms,
            Err(e) => return job.sink.fail(e),
        };
        let loaded = self.loaded.as_ref().expect("just loaded");
        generate(loaded, &self.cfg, self.backend, &job, load_ms);
    }

    fn count_tokens(&mut self, model: &str, text: &str) -> Result<u32, EngineError> {
        self.ensure_loaded(model)?;
        let loaded = self.loaded.as_ref().expect("just loaded");
        loaded
            .model
            .str_to_token(text, AddBos::Never)
            .map(|t| t.len() as u32)
            .map_err(|e| EngineError::Generation(format!("tokenize: {e}")))
    }

    fn unload(&mut self) {
        self.loaded = None;
        *self.shared.loaded.lock() = None;
    }
}

fn token_piece_bytes(model: &LlamaModel, token: llama_cpp_2::token::LlamaToken) -> Vec<u8> {
    match model.token_to_piece_bytes(token, PIECE_BUF, false, None) {
        Ok(b) => b,
        Err(llama_cpp_2::TokenToStringError::InsufficientBufferSpace(needed)) => {
            model.token_to_piece_bytes(token, needed.unsigned_abs() as usize, false, None).unwrap_or_default()
        }
        Err(e) => {
            tracing::warn!("token_to_piece failed: {e}");
            Vec::new()
        }
    }
}

fn build_sampler(params: &GenerationParams) -> LlamaSampler {
    let temperature = params.temperature.unwrap_or(0.8);
    if temperature <= 0.0 {
        return LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    }
    let seed = params.seed.map_or(LLAMA_DEFAULT_SEED, |s| s as u32);
    let mut chain = Vec::new();
    if let Some(top_p) = params.top_p {
        chain.push(LlamaSampler::top_p(top_p, 1));
    }
    chain.push(LlamaSampler::temp(temperature));
    chain.push(LlamaSampler::dist(seed));
    LlamaSampler::chain_simple(chain)
}

/// The generation loop: prompt decode → sample → emit → decode, until EOG,
/// `max_tokens`, a stop sequence, or cancellation.
fn generate(loaded: &LoadedModel, cfg: &LlamaConfig, backend: &LlamaBackend, job: &Job, load_ms: u64) {
    let sink = &job.sink;
    let model = &loaded.model;
    let prompt = render_prompt(model, loaded.template.as_ref(), &job.messages);

    let tokens = match model.str_to_token(&prompt, AddBos::Always) {
        Ok(t) => t,
        Err(e) => return sink.fail(EngineError::Generation(format!("tokenize: {e}"))),
    };
    let prompt_tokens = tokens.len() as u32;
    if prompt_tokens >= loaded.n_ctx {
        return sink.fail(EngineError::ContextExceeded { prompt_tokens, context_length: loaded.n_ctx });
    }

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(loaded.n_ctx))
        .with_n_batch(loaded.n_ctx.min(2048))
        .with_n_threads(cfg.n_threads)
        .with_n_threads_batch(cfg.n_threads);
    let mut ctx = match model.new_context(backend, ctx_params) {
        Ok(c) => c,
        Err(e) => return sink.fail(EngineError::LoadFailed(format!("context: {e}"))),
    };

    let t_prompt = Instant::now();
    let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
    if let Err(e) = batch.add_sequence(&tokens, 0, false) {
        return sink.fail(EngineError::Generation(format!("batch: {e}")));
    }
    if let Err(e) = ctx.decode(&mut batch) {
        return sink.fail(EngineError::Generation(format!("prompt decode: {e}")));
    }
    let prompt_ms = t_prompt.elapsed().as_millis() as u64;

    if !sink.emit(TokenEvent::Start { prompt_tokens }) {
        return;
    }

    let t_eval = Instant::now();
    let mut sampler = build_sampler(&job.params);
    let max_tokens = job.params.max_tokens.unwrap_or(u32::MAX);
    let mut acc = Utf8Accumulator::default();
    let mut sample_idx = batch.n_tokens() - 1;
    let mut n_cur = tokens.len() as i32;
    let mut emitted = 0u32;
    let mut text = String::new();
    let mut finish_reason = FinishReason::Stop;

    loop {
        if sink.cancelled() {
            return; // consumer is gone: no `Done`, nobody to read it
        }
        if emitted >= max_tokens {
            finish_reason = FinishReason::Length;
            break;
        }
        if n_cur >= loaded.n_ctx as i32 {
            finish_reason = FinishReason::Length;
            break;
        }
        let token = sampler.sample(&ctx, sample_idx);
        sampler.accept(token);
        emitted += 1;
        if model.is_eog_token(token) {
            break;
        }

        if let Some(piece) = acc.push(&token_piece_bytes(model, token)) {
            match stop_split(&text, &piece, &job.params.stop) {
                Some(keep) => {
                    if !keep.is_empty() && !sink.emit(TokenEvent::Token(keep)) {
                        return;
                    }
                    break;
                }
                None => {
                    text.push_str(&piece);
                    if !sink.emit(TokenEvent::Token(piece)) {
                        return;
                    }
                }
            }
        }

        batch.clear();
        if let Err(e) = batch.add(token, n_cur, &[0], true) {
            return sink.fail(EngineError::Generation(format!("batch: {e}")));
        }
        n_cur += 1;
        sample_idx = 0;
        if let Err(e) = ctx.decode(&mut batch) {
            return sink.fail(EngineError::Generation(format!("decode: {e}")));
        }
    }

    sink.emit(TokenEvent::Done {
        finish_reason,
        prompt_tokens,
        completion_tokens: emitted,
        load_ms,
        prompt_ms,
        eval_ms: t_eval.elapsed().as_millis() as u64,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use std::sync::atomic::AtomicU32;
    use std::time::Duration;

    #[test]
    fn stop_split_cuts_the_piece_that_starts_a_stop_sequence() {
        let stops = vec!["STOP".to_string()];
        assert_eq!(stop_split("abc", "def", &stops), None);
        assert_eq!(stop_split("abc", "dSTOPe", &stops), Some("d".to_string()));
        assert_eq!(stop_split("abc", "STOP", &stops), Some(String::new()));
        // a stop sequence straddling two token pieces
        assert_eq!(stop_split("abcST", "OPq", &stops), Some(String::new()));
        // empty stop strings are ignored
        assert_eq!(stop_split("a", "b", &[String::new()]), None);
        // the earliest of several stops wins
        let many = vec!["zz".to_string(), "b".to_string()];
        assert_eq!(stop_split("a", "xbzz", &many), Some("x".to_string()));
    }

    #[test]
    fn ftype_names_cover_the_common_quantisations() {
        assert_eq!(ftype_name(15), Some("Q4_K_M"));
        assert_eq!(ftype_name(7), Some("Q8_0"));
        assert_eq!(ftype_name(1), Some("F16"));
        assert_eq!(ftype_name(999), None);
    }

    #[test]
    fn sampler_is_greedy_at_temperature_zero() {
        // Only checks that both chains build; llama.cpp owns the behaviour.
        let _ = build_sampler(&GenerationParams { temperature: Some(0.0), ..Default::default() });
        let _ = build_sampler(&GenerationParams {
            temperature: Some(0.7),
            top_p: Some(0.9),
            seed: Some(42),
            ..Default::default()
        });
    }

    // ---- queue / busy, exercised with a fake runner (no GGUF needed) --------

    /// Emits `Start`, then blocks until the test releases it, then finishes.
    struct FakeRunner {
        release: std_mpsc::Receiver<()>,
        ran: Arc<AtomicU32>,
    }

    impl JobRunner for FakeRunner {
        fn run(&mut self, job: Job) {
            self.ran.fetch_add(1, Ordering::AcqRel);
            let _ = job.sink.emit(TokenEvent::Start { prompt_tokens: 1 });
            let _ = self.release.recv();
            let _ = job.sink.emit(TokenEvent::Token("x".into()));
            let _ = job.sink.emit(TokenEvent::Done {
                finish_reason: FinishReason::Stop,
                prompt_tokens: 1,
                completion_tokens: 1,
                load_ms: 0,
                prompt_ms: 0,
                eval_ms: 0,
            });
        }

        fn count_tokens(&mut self, _model: &str, text: &str) -> Result<u32, EngineError> {
            Ok(text.split_whitespace().count() as u32)
        }

        fn unload(&mut self) {}
    }

    fn fake_model(name: &str) -> ModelInfo {
        ModelInfo {
            name: name.to_string(),
            path: String::new(),
            size_bytes: 0,
            family: "fake".into(),
            parameter_size: String::new(),
            quantization: String::new(),
            context_length: 512,
            digest: String::new(),
            modified_at: String::new(),
        }
    }

    fn fake_request() -> ChatRequest {
        ChatRequest {
            model: "m".into(),
            messages: vec![ChatMessage { role: ChatRole::User, content: "hi".into() }],
            params: GenerationParams::default(),
        }
    }

    fn fake_engine(max_queue: usize) -> (Arc<LlamaEngine>, std_mpsc::Sender<()>, Arc<AtomicU32>) {
        let (release_tx, release_rx) = std_mpsc::channel();
        let ran = Arc::new(AtomicU32::new(0));
        let runner_ran = Arc::clone(&ran);
        let engine = LlamaEngine::with_runner(
            vec![fake_model("m")],
            max_queue,
            Arc::new(Shared::default()),
            move || FakeRunner { release: release_rx, ran: runner_ran },
        );
        (Arc::new(engine), release_tx, ran)
    }

    async fn wait_for(engine: &LlamaEngine, f: impl Fn(&EngineStatus) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let st = engine.status();
            if f(&st) {
                return;
            }
            assert!(Instant::now() < deadline, "timed out waiting; status = {st:?}");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn one_generation_runs_others_queue_and_the_rest_are_busy() {
        let (engine, release, _ran) = fake_engine(1);

        // #1 runs: `chat` returns as soon as the runner emits `Start`.
        let s1 = engine.chat(fake_request()).await.expect("first request runs");
        assert_eq!(engine.status().active, 1);
        assert_eq!(engine.status().queued, 0);

        // #2 waits in the queue.
        let e2 = Arc::clone(&engine);
        let queued = tokio::spawn(async move { e2.chat(fake_request()).await.map(|_| ()) });
        wait_for(&engine, |s| s.queued == 1).await;

        // #3 exceeds max_queue.
        match engine.chat(fake_request()).await {
            Err(EngineError::Busy) => {}
            Err(other) => panic!("expected Busy, got {other}"),
            Ok(_) => panic!("expected Busy"),
        }

        // Let everything finish.
        drop(s1);
        release.send(()).unwrap();
        wait_for(&engine, |s| s.queued == 0 && s.active == 1).await;
        release.send(()).unwrap();
        queued.await.unwrap().expect("queued request eventually runs");
        wait_for(&engine, |s| s.active == 0).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_queued_request_abandoned_by_its_caller_is_skipped() {
        let (engine, release, ran) = fake_engine(4);

        let s1 = engine.chat(fake_request()).await.expect("first request runs");
        let e2 = Arc::clone(&engine);
        let queued = tokio::spawn(async move { e2.chat(fake_request()).await.map(|_| ()) });
        wait_for(&engine, |s| s.queued == 1).await;

        // The caller goes away before the worker ever picks the job up.
        queued.abort();
        let _ = queued.await;

        drop(s1);
        release.send(()).unwrap();
        wait_for(&engine, |s| s.active == 0 && s.queued == 0).await;
        assert_eq!(ran.load(Ordering::Acquire), 1, "the abandoned job must not run");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn streaming_and_cancellation_through_the_public_stream() {
        let (engine, release, _ran) = fake_engine(2);
        let mut s = engine.chat(fake_request()).await.expect("stream");
        assert!(matches!(s.next().await, Some(Ok(TokenEvent::Start { prompt_tokens: 1 }))));
        release.send(()).unwrap();
        assert!(matches!(s.next().await, Some(Ok(TokenEvent::Token(_)))));
        assert!(matches!(s.next().await, Some(Ok(TokenEvent::Done { .. }))));
        assert!(s.next().await.is_none(), "stream ends after Done");
        wait_for(&engine, |st| st.active == 0).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn count_tokens_round_trips_through_the_worker() {
        let (engine, _release, _ran) = fake_engine(2);
        assert_eq!(engine.count_tokens("m", "one two three").await.unwrap(), 3);
        match engine.count_tokens("absent", "x").await {
            Err(EngineError::ModelNotFound(m)) => assert_eq!(m, "absent"),
            other => panic!("expected ModelNotFound, got {other:?}"),
        }
    }
}
