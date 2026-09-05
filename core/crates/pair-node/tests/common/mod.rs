//! Fakes shared by pair-node's integration tests.
//!
//! `pair-node` depends only on the `pair_engine::Engine` and
//! `pair_telemetry::TelemetrySource` traits (CLAUDE.md, "Design invariants"), so
//! the tests inject their own implementations rather than the real crates —
//! those are built concurrently and must never be edited from here.
#![allow(dead_code)]

use futures::stream::StreamExt;
use pair_engine::{
    ChatRequest, Engine, EngineError, EngineStatus, FinishReason, ModelInfo, SharedEngine, TokenEvent,
    TokenStream,
};
use pair_node::{BoundPorts, Node, NodeConfig, NodeHandle};
use pair_telemetry::{Admission, InferenceLoad, TelemetrySource};
use parking_lot::Mutex;
use std::net::{IpAddr, Ipv4Addr};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

// ---------------------------------------------------------------- FakeEngine

/// Errors `FakeEngine` can be scripted to return from `chat()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FakeError {
    Busy,
    ContextExceeded,
    Generation,
    LoadFailed,
}

impl FakeError {
    fn build(self) -> EngineError {
        match self {
            FakeError::Busy => EngineError::Busy,
            FakeError::ContextExceeded => {
                EngineError::ContextExceeded { prompt_tokens: 9_000, context_length: 4_096 }
            }
            FakeError::Generation => EngineError::Generation("fake boom".into()),
            FakeError::LoadFailed => EngineError::LoadFailed("fake load".into()),
        }
    }
}

/// Deterministic `Engine` with configurable catalogue, scripted token events,
/// an optional per-token delay and a live "generations in flight" counter that
/// drops back to zero as soon as the returned stream is dropped.
pub struct FakeEngine {
    models: Vec<ModelInfo>,
    tokens: Vec<String>,
    delay: Option<Duration>,
    finish: FinishReason,
    error: Option<FakeError>,
    /// Emitted mid-stream (after this many tokens) instead of `Done`.
    error_after_tokens: Option<usize>,
    active: Arc<AtomicU32>,
    loaded: Mutex<Option<String>>,
    /// Last `ChatRequest` handed to `chat()`, for parameter-mapping assertions.
    pub last_request: Arc<Mutex<Option<ChatRequest>>>,
}

impl FakeEngine {
    pub fn new(models: Vec<ModelInfo>) -> Self {
        Self {
            models,
            tokens: vec!["Hello".into(), " world".into()],
            delay: None,
            finish: FinishReason::Stop,
            error: None,
            error_after_tokens: None,
            active: Arc::new(AtomicU32::new(0)),
            loaded: Mutex::new(None),
            last_request: Arc::new(Mutex::new(None)),
        }
    }

    pub fn with_tokens<I: IntoIterator<Item = &'static str>>(mut self, tokens: I) -> Self {
        self.tokens = tokens.into_iter().map(str::to_string).collect();
        self
    }

    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = Some(delay);
        self
    }

    pub fn with_finish(mut self, finish: FinishReason) -> Self {
        self.finish = finish;
        self
    }

    pub fn with_error(mut self, error: FakeError) -> Self {
        self.error = Some(error);
        self
    }

    pub fn with_error_after_tokens(mut self, n: usize) -> Self {
        self.error_after_tokens = Some(n);
        self
    }

    pub fn with_loaded(self, model: &str) -> Self {
        *self.loaded.lock() = Some(model.to_string());
        self
    }

    /// Handle to the in-flight counter; survives `Arc`-ing the engine.
    pub fn active_handle(&self) -> Arc<AtomicU32> {
        Arc::clone(&self.active)
    }

    pub fn shared(self) -> SharedEngine {
        Arc::new(self)
    }
}

/// Decrements the in-flight counter when the token stream is dropped — the
/// property the "client disconnect cancels generation" tests assert on.
struct ActiveGuard(Arc<AtomicU32>);

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

struct StreamState {
    idx: usize,
    tokens: Vec<String>,
    delay: Option<Duration>,
    finish: FinishReason,
    error_after_tokens: Option<usize>,
    _guard: ActiveGuard,
}

#[async_trait::async_trait]
impl Engine for FakeEngine {
    async fn list_models(&self) -> Vec<ModelInfo> {
        self.models.clone()
    }

    async fn model(&self, name: &str) -> Option<ModelInfo> {
        self.models.iter().find(|m| m.name == name).cloned()
    }

    async fn chat(&self, req: ChatRequest) -> Result<TokenStream, EngineError> {
        *self.last_request.lock() = Some(req.clone());
        if self.models.iter().all(|m| m.name != req.model) {
            return Err(EngineError::ModelNotFound(req.model));
        }
        if let Some(err) = self.error {
            return Err(err.build());
        }
        self.active.fetch_add(1, Ordering::SeqCst);
        let mut tokens = self.tokens.clone();
        if let Some(max) = req.params.max_tokens {
            tokens.truncate(max as usize);
        }
        let state = StreamState {
            idx: 0,
            tokens,
            delay: self.delay,
            finish: self.finish.clone(),
            error_after_tokens: self.error_after_tokens,
            _guard: ActiveGuard(Arc::clone(&self.active)),
        };
        let stream = futures::stream::unfold(state, |mut s| async move {
            let n = s.tokens.len();
            let item = if s.idx == 0 {
                Ok(TokenEvent::Start { prompt_tokens: 7 })
            } else if s.idx <= n {
                if let Some(stop_at) = s.error_after_tokens {
                    if s.idx - 1 == stop_at {
                        return Some((Err(EngineError::Generation("fake mid-stream".into())), s));
                    }
                }
                if let Some(d) = s.delay {
                    tokio::time::sleep(d).await;
                }
                Ok(TokenEvent::Token(s.tokens[s.idx - 1].clone()))
            } else if s.idx == n + 1 {
                Ok(TokenEvent::Done {
                    finish_reason: s.finish.clone(),
                    prompt_tokens: 7,
                    completion_tokens: n as u32,
                    load_ms: 12,
                    prompt_ms: 34,
                    eval_ms: 56,
                })
            } else {
                return None;
            };
            s.idx += 1;
            Some((item, s))
        });
        Ok(stream.boxed())
    }

    async fn count_tokens(&self, _model: &str, text: &str) -> Result<u32, EngineError> {
        Ok(text.split_whitespace().count() as u32)
    }

    async fn unload(&self) {
        *self.loaded.lock() = None;
    }

    fn status(&self) -> EngineStatus {
        let loaded = self.loaded.lock().clone();
        EngineStatus {
            loaded_bytes: loaded
                .as_ref()
                .and_then(|n| self.models.iter().find(|m| &m.name == n))
                .map(|m| m.size_bytes)
                .unwrap_or(0),
            loaded_model: loaded,
            active: self.active.load(Ordering::SeqCst),
            queued: 0,
        }
    }
}

/// Catalogue entry helper.
pub fn model(name: &str) -> ModelInfo {
    ModelInfo {
        name: name.to_string(),
        path: format!("/models/{name}.gguf"),
        size_bytes: 1_073_741_824,
        family: "qwen2".to_string(),
        parameter_size: "1.5B".to_string(),
        quantization: "Q4_K_M".to_string(),
        context_length: 4096,
        digest: "sha256:9f2c0a1b3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8".to_string(),
        modified_at: "2026-02-03T04:05:06Z".to_string(),
    }
}

pub fn two_models() -> Vec<ModelInfo> {
    vec![model("qwen2.5-1.5b-instruct-q4_k_m"), model("gemma-2b-it-q4_k_m")]
}

// ------------------------------------------------------------- FakeTelemetry

/// `TelemetrySource` returning a fixed `NodeInfoResponse` and a settable
/// [`Admission`], counting `tick()`s and recording the pushed [`InferenceLoad`].
pub struct FakeTelemetry {
    info: Mutex<pair_protocol::node_info::NodeInfoResponse>,
    admission: Mutex<Admission>,
    ticks: AtomicU32,
    load: Mutex<InferenceLoad>,
    interval: Duration,
}

impl Default for FakeTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

impl FakeTelemetry {
    pub fn new() -> Self {
        use pair_protocol::node_info::{CpuInfo, GpuInfo, MemoryInfo, NodeInfoResponse};
        Self {
            info: Mutex::new(NodeInfoResponse {
                gpus: vec![GpuInfo {
                    name: "Adreno 750 (llama.cpp)".into(),
                    vram_bytes: 6_442_450_944,
                    vram_used_bytes: 1_073_741_824,
                    utilization_percent: 42,
                }],
                cpu: Some(CpuInfo { name: "Snapdragon 8 Gen 3".into(), cores: 8, utilization_percent: 17 }),
                memory: Some(MemoryInfo { total_bytes: 17_179_869_184, used_bytes: 8_589_934_592 }),
                telemetry_valid: true,
                ms_since: 250,
                host_uuid: "8f14e45f-ea3c-4f1e-9b0a-1d2c3b4a5f60".into(),
                cluster_uuid: None,
            }),
            admission: Mutex::new(Admission::Accept),
            ticks: AtomicU32::new(0),
            load: Mutex::new(InferenceLoad::default()),
            interval: Duration::from_millis(20),
        }
    }

    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.interval = interval;
        self
    }

    pub fn set_admission(&self, admission: Admission) {
        *self.admission.lock() = admission;
    }

    pub fn ticks(&self) -> u32 {
        self.ticks.load(Ordering::SeqCst)
    }

    pub fn load(&self) -> InferenceLoad {
        self.load.lock().clone()
    }

    pub fn expected(&self) -> pair_protocol::node_info::NodeInfoResponse {
        self.info.lock().clone()
    }

    /// Mutate the reported node-info so tests can prove the handler does not cache.
    pub fn set_ms_since(&self, ms: i64) {
        self.info.lock().ms_since = ms;
    }
}

impl TelemetrySource for FakeTelemetry {
    fn node_info(&self) -> pair_protocol::node_info::NodeInfoResponse {
        self.info.lock().clone()
    }
    fn admission(&self) -> Admission {
        self.admission.lock().clone()
    }
    fn set_inference_load(&self, load: InferenceLoad) {
        *self.load.lock() = load;
    }
    fn tick(&self) {
        self.ticks.fetch_add(1, Ordering::SeqCst);
    }
    fn sample_interval(&self) -> Duration {
        self.interval
    }
}

// ------------------------------------------------------------------- harness

pub const LOCALHOST: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);

/// Boot a node on loopback with ephemeral ports (CLAUDE.md: "tests bind port 0").
pub async fn start_node(engine: SharedEngine, telemetry: Arc<dyn TelemetrySource>) -> NodeHandle {
    let config = NodeConfig {
        bind: LOCALHOST,
        openai_port: 0,
        ollama_port: 0,
        node_info_port: 0,
        ..NodeConfig::default()
    };
    Node::start(config, engine, telemetry).await.expect("node starts")
}

pub async fn start_with(engine: FakeEngine) -> (NodeHandle, Arc<FakeTelemetry>) {
    let telemetry = Arc::new(FakeTelemetry::new());
    let handle = start_node(engine.shared(), telemetry.clone()).await;
    (handle, telemetry)
}

pub fn url(ports_port: u16, path: &str) -> String {
    format!("http://{LOCALHOST}:{ports_port}{path}")
}

pub fn openai_url(ports: BoundPorts, path: &str) -> String {
    url(ports.openai, path)
}

pub fn ollama_url(ports: BoundPorts, path: &str) -> String {
    url(ports.ollama, path)
}

pub fn node_info_url(ports: BoundPorts, path: &str) -> String {
    url(ports.node_info, path)
}

pub fn client() -> reqwest::Client {
    reqwest::Client::builder().timeout(Duration::from_secs(10)).build().expect("client")
}

/// Poll `cond` until it holds or `within` elapses.
pub async fn wait_until<F: FnMut() -> bool>(within: Duration, mut cond: F) -> bool {
    let deadline = std::time::Instant::now() + within;
    while std::time::Instant::now() < deadline {
        if cond() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    cond()
}
