//! Device telemetry → PAIR `node-info`.
//!
//! TODO(ticket: telemetry/core): implement + tests.
//!
//! Design contract
//! - [`Sampler`] reads raw counters. `ProcfsSampler` (Linux/Android: `/proc/stat`,
//!   `/proc/meminfo`) is the default; tests inject a `FakeSampler`.
//! - [`ExternalSignals`] is what the Android layer pushes in (battery %, charging,
//!   thermal status, screen state). It is *not* sampled by Rust.
//! - [`InferenceLoad`] is what `pair-node` pushes in from `Engine::status()`.
//! - [`Telemetry`] combines them into a `NodeInfoResponse`:
//!   * `cpu.utilization_percent`: delta of /proc/stat busy/total between samples.
//!   * `memory.*`: MemTotal / (MemTotal - MemAvailable).
//!   * `GPUs[0]`: the inference accelerator entry. `name` from config
//!     (e.g. "Adreno 750 (llama.cpp)"), `utilization_percent` = EWMA of
//!     "engine busy" (1 while a generation is active, else 0) sampled every tick,
//!     `vram_bytes` = device RAM budget for models, `vram_used_bytes` = loaded model bytes.
//!     This is deliberate: it makes PAIR's GPUPressure (40/70/85 % bands) track
//!     how busy the phone actually is.
//!   * `telemetryValid`: true once ≥2 CPU samples exist; false if the sampler failed.
//!   * `msSince`: ms since last successful sample.
//!   * `hostUuid`: from [`HostIdentity`] (persisted by the caller).
//! - Thermal/battery policy hooks: [`Telemetry::admission`] returns whether new
//!   requests should be accepted (e.g. refuse when thermal ≥ SEVERE or battery
//!   < min% and discharging). pair-node consults it before generation.

use pair_protocol::node_info::NodeInfoResponse;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RawCpuSample {
    /// Jiffies spent busy (user+nice+system+irq+softirq+steal).
    pub busy: u64,
    /// Jiffies total (busy + idle + iowait).
    pub total: u64,
    pub cores: u32,
    pub model_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RawMemSample {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

pub trait Sampler: Send + Sync + 'static {
    fn cpu(&self) -> std::io::Result<RawCpuSample>;
    fn mem(&self) -> std::io::Result<RawMemSample>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
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

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ExternalSignals {
    pub battery_percent: Option<u8>,
    pub charging: Option<bool>,
    pub thermal: ThermalStatus,
    pub screen_on: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct InferenceLoad {
    pub active: u32,
    pub queued: u32,
    pub loaded_bytes: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TelemetryConfig {
    pub host_uuid: String,
    pub accelerator_name: String,
    /// Bytes of RAM this node is willing to dedicate to models (reported as vram_bytes).
    pub model_budget_bytes: u64,
    pub sample_interval: Duration,
    /// EWMA smoothing factor for accelerator utilisation, 0 < alpha ≤ 1.
    pub ewma_alpha: f64,
    pub min_battery_percent_on_battery: u8,
    pub max_thermal: ThermalStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Admission {
    Accept,
    /// Reason string is surfaced in the HTTP 503 body.
    Refuse(String),
}

/// What `pair-node` needs from telemetry. `Telemetry` implements it; node tests use a fake.
pub trait TelemetrySource: Send + Sync + 'static {
    fn node_info(&self) -> NodeInfoResponse;
    fn admission(&self) -> Admission;
    fn set_inference_load(&self, load: InferenceLoad);
    /// Sample now. pair-node calls this on a `TelemetryConfig::sample_interval` timer.
    fn tick(&self);
    fn sample_interval(&self) -> Duration;
}

pub struct Telemetry {
    // TODO(ticket telemetry/core)
}

impl TelemetrySource for Telemetry {
    fn node_info(&self) -> NodeInfoResponse {
        Telemetry::node_info(self)
    }
    fn admission(&self) -> Admission {
        Telemetry::admission(self)
    }
    fn set_inference_load(&self, load: InferenceLoad) {
        Telemetry::set_inference_load(self, load)
    }
    fn tick(&self) {
        Telemetry::tick(self)
    }
    fn sample_interval(&self) -> Duration {
        todo!("ticket telemetry/core")
    }
}

impl Telemetry {
    pub fn new(config: TelemetryConfig, sampler: Box<dyn Sampler>) -> Self {
        let _ = (config, sampler);
        todo!("ticket telemetry/core")
    }
    /// Take one sample now (called by a tokio interval in pair-node, and directly by tests).
    pub fn tick(&self) {
        todo!()
    }
    pub fn set_external(&self, signals: ExternalSignals) {
        let _ = signals;
        todo!()
    }
    pub fn set_inference_load(&self, load: InferenceLoad) {
        let _ = load;
        todo!()
    }
    pub fn node_info(&self) -> NodeInfoResponse {
        todo!()
    }
    pub fn admission(&self) -> Admission {
        todo!()
    }
}

/// Linux/Android `/proc` based sampler.
pub struct ProcfsSampler;

impl Sampler for ProcfsSampler {
    fn cpu(&self) -> std::io::Result<RawCpuSample> {
        todo!("ticket telemetry/core")
    }
    fn mem(&self) -> std::io::Result<RawMemSample> {
        todo!("ticket telemetry/core")
    }
}
