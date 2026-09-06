//! UniFFI surface for the Android app (ADR-0002).
//!
//! Kotlin sees `object PairNode` (the hand-written wrapper in
//! `android/app/src/main/java/uniffi/pair_ffi/PairNode.kt`) delegating to the
//! top-level functions exported here:
//!
//! | Kotlin | Rust |
//! | --- | --- |
//! | `PairNode.start(config)` | [`pair_node_start`] |
//! | `PairNode.stop()` | [`pair_node_stop`] |
//! | `PairNode.status()` | [`pair_node_status`] |
//! | `PairNode.pushSignals(signals)` | [`pair_node_push_signals`] |
//! | `PairNode.setModelsDir(path)` | [`pair_node_set_models_dir`] |
//! | `PairNode.listModels()` | [`pair_node_list_models`] |
//! | `PairNode.setEventListener(events)` | [`pair_node_set_event_listener`] |
//!
//! Design contract
//! - Everything is process-global: the app has exactly one node. State lives in a
//!   `OnceLock<Mutex<State>>` holding a dedicated 2-worker tokio runtime, the
//!   [`pair_node::NodeHandle`], the engine and the telemetry.
//! - Every call blocks (Kotlin wraps them in coroutines on `Dispatchers.IO`).
//! - Logging: a `tracing` layer installed once forwards INFO+ events to
//!   [`NodeEvents::on_log`], and pair-node's access-log events (which carry
//!   `path`/`model`/`status`/`ms`) additionally to [`NodeEvents::on_request`].
//! - [`NodeEvents::on_state_changed`] fires on start, on stop, and from a 2 s poll
//!   task whenever `loaded_model` / `active` / `queued` change.
//! - `pair_node_status` and `pair_node_list_models` work while stopped: the model
//!   list then comes from scanning the models directory for `*.gguf`, so the
//!   Models screen has something to show before the first start.

uniffi::setup_scaffolding!("pair_ffi");

mod logging;
mod scan;

use pair_engine::mock::MockEngine;
use pair_engine::SharedEngine;
use pair_node::{Node, NodeHandle};
use pair_telemetry::{ProcfsSampler, Telemetry, TelemetryConfig, TelemetrySource};
use parking_lot::{Mutex, RwLock};
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio::sync::watch;

/// How often the poll task compares the engine's status against the last one it
/// pushed to Kotlin.
const POLL_INTERVAL: Duration = Duration::from_secs(2);

// ------------------------------------------------------------------ types

/// What Kotlin passes to `PairNode.start`. `bind` is an IP literal
/// (`"0.0.0.0"`); a port of `0` means "let the OS pick" (tests only — PAIR's
/// ports are compile-time constants, see CLAUDE.md).
#[derive(Debug, Clone, PartialEq, Eq, uniffi::Record)]
pub struct NodeConfig {
    pub bind: String,
    pub openai_port: u16,
    pub ollama_port: u16,
    pub node_info_port: u16,
    /// Stable per-device UUID; PAIR dedupes manual nodes against mDNS peers by it.
    pub host_uuid: String,
    /// Reported as `GPUs[0].name`, e.g. the SoC model.
    pub accelerator_name: String,
    /// Reported as `GPUs[0].vram_bytes`: RAM this node dedicates to models.
    pub model_budget_bytes: u64,
    /// Non-empty → run the deterministic mock engine with these model names
    /// instead of llama.cpp.
    pub mock_models: Vec<String>,
}

/// The ports the node actually bound (exact, even when the config asked for 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Record)]
pub struct BoundPorts {
    pub openai: u16,
    pub ollama: u16,
    pub node_info: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, uniffi::Record)]
pub struct NodeStatus {
    pub running: bool,
    /// `None` while stopped.
    pub ports: Option<BoundPorts>,
    pub loaded_model: Option<String>,
    pub active: u32,
    pub queued: u32,
    /// Why the last start failed, if it did. Cleared by a successful start.
    pub last_error: Option<String>,
}

/// Android thermal levels (`PowerManager.THERMAL_STATUS_*`), pushed in from Kotlin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, uniffi::Enum)]
pub enum ThermalStatus {
    #[default]
    None,
    Light,
    Moderate,
    Severe,
    Critical,
    Emergency,
    Shutdown,
}

/// Battery/thermal/screen signals. Rust never polls Android APIs (CLAUDE.md).
#[derive(Debug, Clone, PartialEq, Eq, Default, uniffi::Record)]
pub struct ExternalSignals {
    pub battery_percent: Option<u8>,
    pub charging: Option<bool>,
    pub thermal: ThermalStatus,
    pub screen_on: Option<bool>,
}

/// One model in the node's catalogue. `quant` is `pair_engine::ModelInfo::quantization`.
#[derive(Debug, Clone, PartialEq, Eq, uniffi::Record)]
pub struct ModelInfo {
    pub name: String,
    pub path: String,
    pub size_bytes: u64,
    pub family: String,
    pub parameter_size: String,
    pub quant: String,
    /// `0` when the node is stopped and the entry came from scanning the
    /// models directory (the context length is only known once loaded).
    pub context_length: u32,
}

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum PairError {
    #[error("the node is already running")]
    AlreadyRunning,
    #[error("the node is not running")]
    NotRunning,
    #[error("bind failed: {msg}")]
    Bind { msg: String },
    #[error("engine error: {msg}")]
    Engine { msg: String },
    #[error("io error: {msg}")]
    Io { msg: String },
}

/// Implemented by Kotlin (`NodeRepository.events`). Every method is called from a
/// Rust thread, so implementations must be cheap and non-blocking.
#[uniffi::export(callback_interface)]
pub trait NodeEvents: Send + Sync {
    fn on_log(&self, level: String, msg: String);
    fn on_request(&self, lane: String, model: String, status: i32, ms: i64);
    fn on_state_changed(&self, status: NodeStatus);
}

// ------------------------------------------------------------- conversions

impl From<ThermalStatus> for pair_telemetry::ThermalStatus {
    fn from(value: ThermalStatus) -> Self {
        match value {
            ThermalStatus::None => Self::None,
            ThermalStatus::Light => Self::Light,
            ThermalStatus::Moderate => Self::Moderate,
            ThermalStatus::Severe => Self::Severe,
            ThermalStatus::Critical => Self::Critical,
            ThermalStatus::Emergency => Self::Emergency,
            ThermalStatus::Shutdown => Self::Shutdown,
        }
    }
}

impl From<ExternalSignals> for pair_telemetry::ExternalSignals {
    fn from(value: ExternalSignals) -> Self {
        Self {
            battery_percent: value.battery_percent,
            charging: value.charging,
            thermal: value.thermal.into(),
            screen_on: value.screen_on,
        }
    }
}

impl From<pair_node::BoundPorts> for BoundPorts {
    fn from(value: pair_node::BoundPorts) -> Self {
        Self { openai: value.openai, ollama: value.ollama, node_info: value.node_info }
    }
}

impl From<pair_engine::ModelInfo> for ModelInfo {
    fn from(value: pair_engine::ModelInfo) -> Self {
        Self {
            name: value.name,
            path: value.path,
            size_bytes: value.size_bytes,
            family: value.family,
            parameter_size: value.parameter_size,
            quant: value.quantization,
            context_length: value.context_length,
        }
    }
}

// ------------------------------------------------------------------ state

struct Running {
    handle: NodeHandle,
    ports: BoundPorts,
    engine: SharedEngine,
    telemetry: Arc<Telemetry>,
    /// Sending `true` stops the poll task.
    poll_stop: watch::Sender<bool>,
}

struct State {
    runtime: tokio::runtime::Runtime,
    running: Option<Running>,
    models_dir: Option<PathBuf>,
    /// Config of the last successful `pair_node_start`, so `set_models_dir`
    /// can restart the node in place with the same lanes/uuid (ticket #24).
    config: Option<NodeConfig>,
    /// Last signals pushed from Kotlin; re-applied to a freshly started node so
    /// signals that arrived while stopped are not lost.
    signals: ExternalSignals,
    last_error: Option<String>,
}

fn state() -> &'static Mutex<State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE.get_or_init(|| {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("pair-node")
            .enable_all()
            .build()
            .expect("pair-ffi: failed to build the tokio runtime");
        Mutex::new(State {
            runtime,
            running: None,
            models_dir: None,
            config: None,
            signals: ExternalSignals::default(),
            last_error: None,
        })
    })
}

fn snapshot(state: &State) -> NodeStatus {
    match &state.running {
        Some(running) => {
            let engine = running.engine.status();
            NodeStatus {
                running: true,
                ports: Some(running.ports),
                loaded_model: engine.loaded_model,
                active: engine.active,
                queued: engine.queued,
                last_error: state.last_error.clone(),
            }
        }
        None => NodeStatus {
            running: false,
            ports: None,
            loaded_model: None,
            active: 0,
            queued: 0,
            last_error: state.last_error.clone(),
        },
    }
}

// -------------------------------------------------------------- listener

fn listener_slot() -> &'static RwLock<Option<Arc<dyn NodeEvents>>> {
    static LISTENER: OnceLock<RwLock<Option<Arc<dyn NodeEvents>>>> = OnceLock::new();
    LISTENER.get_or_init(|| RwLock::new(None))
}

pub(crate) fn listener() -> Option<Arc<dyn NodeEvents>> {
    listener_slot().read().clone()
}

fn emit_state(status: &NodeStatus) {
    if let Some(events) = listener() {
        events.on_state_changed(status.clone());
    }
}

// ---------------------------------------------------------------- engine

fn build_engine(config: &NodeConfig, models_dir: Option<&Path>) -> Result<SharedEngine, PairError> {
    if !config.mock_models.is_empty() {
        let names: Vec<&str> = config.mock_models.iter().map(String::as_str).collect();
        return Ok(Arc::new(MockEngine::with_models(&names)));
    }
    #[cfg(feature = "llama")]
    {
        let dir =
            models_dir.ok_or_else(|| PairError::Engine { msg: "no models directory set".to_string() })?;
        let engine = pair_engine::llama::LlamaEngine::new(
            dir.to_path_buf(),
            pair_engine::llama::LlamaConfig::default(),
        )
        .map_err(|e| PairError::Engine { msg: e.to_string() })?;
        Ok(Arc::new(engine))
    }
    #[cfg(not(feature = "llama"))]
    {
        let _ = models_dir;
        Err(PairError::Engine { msg: "built without llama feature and no mock models".to_string() })
    }
}

/// Pushes a state change whenever the engine's status moves. `NodeStatus` is
/// otherwise only observed when Kotlin calls `status()`.
///
/// `initial` is sampled synchronously by the caller *before* this task is
/// spawned. Sampling it here would race with requests that arrive before the
/// task first runs (a slow runner already serves a chat request by then), and
/// the resulting `loaded_model` change would never be reported.
async fn poll_loop(
    engine: SharedEngine,
    ports: BoundPorts,
    initial: pair_engine::EngineStatus,
    mut stop: watch::Receiver<bool>,
) {
    let mut ticker = tokio::time::interval(POLL_INTERVAL);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    ticker.tick().await; // the first tick completes immediately

    let mut previous = (initial.loaded_model, initial.active, initial.queued);
    loop {
        tokio::select! {
            _ = ticker.tick() => {}
            _ = stop.changed() => break,
        }
        let engine_status = engine.status();
        let current = (engine_status.loaded_model, engine_status.active, engine_status.queued);
        if current != previous {
            previous = current.clone();
            emit_state(&NodeStatus {
                running: true,
                ports: Some(ports),
                loaded_model: current.0,
                active: current.1,
                queued: current.2,
                last_error: None,
            });
        }
    }
}

// -------------------------------------------------------------------- API

/// Bind the three lanes and start serving. Returns the status with the ports the
/// OS actually gave us. Fails with [`PairError::AlreadyRunning`] if a node is up.
#[uniffi::export]
pub fn pair_node_start(config: NodeConfig) -> Result<NodeStatus, PairError> {
    logging::install();
    let mut state = state().lock();
    if state.running.is_some() {
        return Err(PairError::AlreadyRunning);
    }

    let status = start_locked(&mut state, config);
    match status {
        Ok(status) => {
            drop(state);
            emit_state(&status);
            Ok(status)
        }
        Err(err) => {
            state.last_error = Some(err.to_string());
            let status = snapshot(&state);
            drop(state);
            emit_state(&status);
            Err(err)
        }
    }
}

fn start_locked(state: &mut State, config: NodeConfig) -> Result<NodeStatus, PairError> {
    let bind: IpAddr = config
        .bind
        .parse()
        .map_err(|e| PairError::Bind { msg: format!("invalid bind address {:?}: {e}", config.bind) })?;

    let engine = build_engine(&config, state.models_dir.as_deref())?;

    let telemetry = Arc::new(Telemetry::new(
        TelemetryConfig::default_for(
            config.host_uuid.clone(),
            config.accelerator_name.clone(),
            config.model_budget_bytes,
        ),
        Box::new(ProcfsSampler::default()),
    ));
    telemetry.set_external(state.signals.clone().into());

    let node_config = pair_node::NodeConfig {
        bind,
        openai_port: config.openai_port,
        ollama_port: config.ollama_port,
        node_info_port: config.node_info_port,
        ..Default::default()
    };

    let handle = state
        .runtime
        .block_on(Node::start(
            node_config,
            Arc::clone(&engine),
            Arc::clone(&telemetry) as Arc<dyn TelemetrySource>,
        ))
        .map_err(|e| PairError::Bind { msg: format!("{e:#}") })?;

    let ports = BoundPorts::from(handle.ports());
    let (poll_stop, poll_rx) = watch::channel(false);
    let initial = engine.status();
    state.runtime.spawn(poll_loop(Arc::clone(&engine), ports, initial, poll_rx));

    state.last_error = None;
    state.config = Some(config);
    state.running = Some(Running { handle, ports, engine, telemetry, poll_stop });
    Ok(snapshot(state))
}

/// Shut the running node down (no-op when stopped). Shared by `pair_node_stop`
/// and the in-place restart in `pair_node_set_models_dir`.
fn stop_locked(state: &mut State) -> bool {
    let Some(running) = state.running.take() else {
        return false;
    };
    let _ = running.poll_stop.send(true);
    state.runtime.block_on(running.handle.shutdown());
    true
}

/// Stop the three lanes and free the ports. [`PairError::NotRunning`] if stopped.
#[uniffi::export]
pub fn pair_node_stop() -> Result<(), PairError> {
    let mut state = state().lock();
    if !stop_locked(&mut state) {
        return Err(PairError::NotRunning);
    }
    let status = snapshot(&state);
    drop(state);
    emit_state(&status);
    Ok(())
}

/// The current status. Valid whether or not the node is running.
#[uniffi::export]
pub fn pair_node_status() -> NodeStatus {
    snapshot(&state().lock())
}

/// Push Android's battery/thermal/screen signals into telemetry. Remembered
/// while stopped and re-applied on the next start.
#[uniffi::export]
pub fn pair_node_push_signals(signals: ExternalSignals) {
    let mut state = state().lock();
    state.signals = signals.clone();
    if let Some(running) = &state.running {
        running.telemetry.set_external(signals.into());
    }
}

/// Where `*.gguf` files live. While stopped it is remembered for the next
/// start. While running, the node is restarted in place with the config from
/// the last `pair_node_start` so the engine re-reads the directory: Kotlin calls
/// this after every import/rename/delete and PAIR must see the new catalogue
/// (ticket #24). The listener observes `running:false` then `running:true`;
/// a failed restart leaves the node stopped with `last_error` set.
#[uniffi::export]
pub fn pair_node_set_models_dir(path: String) {
    let mut state = state().lock();
    state.models_dir = Some(PathBuf::from(path));
    if state.running.is_none() {
        return;
    }
    let Some(config) = state.config.clone() else {
        return;
    };

    stop_locked(&mut state);
    let stopped = snapshot(&state);
    let restarted = match start_locked(&mut state, config) {
        Ok(status) => status,
        Err(err) => {
            tracing::error!(error = %err, "pair-ffi: restart after models-dir change failed");
            state.last_error = Some(err.to_string());
            snapshot(&state)
        }
    };
    // Same discipline as `pair_node_start`: never call the listener with the
    // state lock held (Kotlin's callback may call back into `status()`).
    drop(state);
    emit_state(&stopped);
    emit_state(&restarted);
}

/// The running engine's catalogue, or — while stopped — a scan of the models
/// directory so the Models screen works before the first start.
#[uniffi::export]
pub fn pair_node_list_models() -> Vec<ModelInfo> {
    let state = state().lock();
    if let Some(running) = &state.running {
        let engine = Arc::clone(&running.engine);
        let models = state.runtime.block_on(async move { engine.list_models().await });
        return models.into_iter().map(ModelInfo::from).collect();
    }
    match &state.models_dir {
        Some(dir) => scan::scan_gguf(dir),
        None => Vec::new(),
    }
}

/// Register Kotlin's event sink. Replaces any previous one; also installs the
/// `tracing` forwarder the first time it is called.
#[uniffi::export]
pub fn pair_node_set_event_listener(events: Box<dyn NodeEvents>) {
    logging::install();
    *listener_slot().write() = Some(Arc::from(events));
}
