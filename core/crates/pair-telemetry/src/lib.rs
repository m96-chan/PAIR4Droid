//! Device telemetry → PAIR `node-info`.
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
//!     how busy the phone actually is. See `docs/adr/0003-accelerator-utilization-is-inference-busy.md`.
//!   * `telemetryValid`: true once ≥2 CPU samples exist; false if the sampler failed.
//!   * `msSince`: ms since last successful sample.
//!   * `hostUuid`: from [`TelemetryConfig::host_uuid`] (persisted by the caller).
//! - Thermal/battery policy hooks: [`Telemetry::admission`] returns whether new
//!   requests should be accepted (e.g. refuse when thermal ≥ SEVERE or battery
//!   < min% and discharging). pair-node consults it before generation.

pub mod procfs;

use pair_protocol::node_info::{CpuInfo, GpuInfo, MemoryInfo, NodeInfoResponse};
use parking_lot::Mutex;
use std::time::{Duration, Instant};

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

/// Lets a caller keep a handle to a shared sampler (e.g. a test's fake, or a
/// sampler shared across more than one `Telemetry`) while also handing a
/// `Box<dyn Sampler>` to [`Telemetry::new`].
impl<T: Sampler + ?Sized> Sampler for std::sync::Arc<T> {
    fn cpu(&self) -> std::io::Result<RawCpuSample> {
        (**self).cpu()
    }
    fn mem(&self) -> std::io::Result<RawMemSample> {
        (**self).mem()
    }
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

impl TelemetryConfig {
    /// Sensible defaults for a real device: sample every 2s, moderate EWMA
    /// smoothing, refuse admission under 20% battery while discharging or at
    /// `Severe` thermal or worse.
    pub fn default_for(host_uuid: String, accelerator_name: String, model_budget_bytes: u64) -> Self {
        Self {
            host_uuid,
            accelerator_name,
            model_budget_bytes,
            sample_interval: Duration::from_secs(2),
            ewma_alpha: 0.3,
            min_battery_percent_on_battery: 20,
            max_thermal: ThermalStatus::Severe,
        }
    }
}

impl Default for TelemetryConfig {
    /// A random `host_uuid` (v4) plus [`Self::default_for`]'s defaults. Real
    /// callers should persist and reuse a stable `host_uuid` instead of
    /// relying on this — PAIR dedupes manual nodes against mDNS peers by it.
    fn default() -> Self {
        Self::default_for(uuid::Uuid::new_v4().to_string(), String::new(), 0)
    }
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

#[derive(Debug, Default)]
struct State {
    last_cpu: Option<RawCpuSample>,
    cpu_utilization_percent: u32,
    cpu_cores: u32,
    cpu_model_name: String,
    mem_total_bytes: u64,
    mem_used_bytes: u64,
    telemetry_valid: bool,
    last_sample_at: Option<Instant>,
    /// Count of sampler errors (cpu or mem) seen so far. Not exposed via the
    /// public API (nothing in the design contract asks for it); logged via
    /// `tracing::warn!` on each error and kept here for future diagnostics.
    error_count: u64,
    external: ExternalSignals,
    inference_load: InferenceLoad,
    gpu_ewma: f64,
    gpu_utilization_percent: u32,
}

pub struct Telemetry {
    config: TelemetryConfig,
    sampler: Box<dyn Sampler>,
    state: Mutex<State>,
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
        self.config.sample_interval
    }
}

impl Telemetry {
    /// # Panics
    /// Panics if `config.ewma_alpha` is not in `(0, 1]` — an alpha of 0 would
    /// never move off the initial EWMA value, and negative/`>1`/NaN values
    /// make the smoothing formula meaningless. This is a configuration bug,
    /// not a runtime condition, so it panics unconditionally (not just in
    /// debug builds) rather than silently clamping into a value nobody chose.
    pub fn new(config: TelemetryConfig, sampler: Box<dyn Sampler>) -> Self {
        assert!(
            config.ewma_alpha > 0.0 && config.ewma_alpha <= 1.0,
            "pair-telemetry: TelemetryConfig::ewma_alpha must be in (0, 1], got {}",
            config.ewma_alpha
        );
        let mut config = config;
        if config.host_uuid.is_empty() {
            // PAIR ignores node-info with an empty hostUuid (falls back to
            // neutral pressure, docs/pair-contract.md §2), so node_info()
            // must never emit one even if misconfigured.
            tracing::warn!("pair-telemetry: TelemetryConfig::host_uuid was empty; generating a random one");
            config.host_uuid = uuid::Uuid::new_v4().to_string();
        }
        Self { config, sampler, state: Mutex::new(State::default()) }
    }

    /// Take one sample now (called by a tokio interval in pair-node, and directly by tests).
    pub fn tick(&self) {
        let mut state = self.state.lock();

        // The accelerator EWMA reflects inference activity pushed in via
        // `set_inference_load`, not the CPU/mem sampler, so it advances every
        // tick regardless of whether `/proc` sampling below succeeds.
        let busy_sample = if state.inference_load.active > 0 { 100.0 } else { 0.0 };
        let alpha = self.config.ewma_alpha;
        state.gpu_ewma = alpha * busy_sample + (1.0 - alpha) * state.gpu_ewma;
        state.gpu_utilization_percent = state.gpu_ewma.round().clamp(0.0, 100.0) as u32;

        let had_prev_cpu = state.last_cpu.is_some();

        let cpu_ok = match self.sampler.cpu() {
            Ok(raw) => {
                state.cpu_cores = raw.cores;
                if !raw.model_name.is_empty() {
                    state.cpu_model_name = raw.model_name.clone();
                }
                if let Some(prev) = &state.last_cpu {
                    let delta_busy = raw.busy.saturating_sub(prev.busy);
                    let delta_total = raw.total.saturating_sub(prev.total);
                    if let Some(pct) = delta_busy.saturating_mul(100).checked_div(delta_total) {
                        state.cpu_utilization_percent = pct.min(100) as u32;
                    }
                }
                state.last_cpu = Some(raw);
                true
            }
            Err(err) => {
                tracing::warn!(error = %err, "pair-telemetry: cpu sample failed, keeping last good value");
                state.error_count += 1;
                false
            }
        };

        let mem_ok = match self.sampler.mem() {
            Ok(raw) => {
                state.mem_total_bytes = raw.total_bytes;
                state.mem_used_bytes = raw.total_bytes.saturating_sub(raw.available_bytes);
                true
            }
            Err(err) => {
                tracing::warn!(error = %err, "pair-telemetry: mem sample failed, keeping last good value");
                state.error_count += 1;
                false
            }
        };

        let sampled_ok = cpu_ok && mem_ok;
        if sampled_ok {
            state.last_sample_at = Some(Instant::now());
        }
        // Valid only once this tick sampled cleanly *and* a previous CPU
        // sample exists to diff against (ADR-0003 / design contract).
        state.telemetry_valid = sampled_ok && had_prev_cpu;
    }

    pub fn set_external(&self, signals: ExternalSignals) {
        self.state.lock().external = signals;
    }

    pub fn set_inference_load(&self, load: InferenceLoad) {
        self.state.lock().inference_load = load;
    }

    pub fn node_info(&self) -> NodeInfoResponse {
        let state = self.state.lock();

        let cpu = CpuInfo {
            name: state.cpu_model_name.clone(),
            cores: state.cpu_cores,
            utilization_percent: state.cpu_utilization_percent,
        };
        let memory = MemoryInfo { total_bytes: state.mem_total_bytes, used_bytes: state.mem_used_bytes };
        let gpu = GpuInfo {
            name: self.config.accelerator_name.clone(),
            vram_bytes: self.config.model_budget_bytes,
            vram_used_bytes: state.inference_load.loaded_bytes,
            utilization_percent: state.gpu_utilization_percent,
        };
        let ms_since = state.last_sample_at.map(|at| at.elapsed().as_millis() as i64).unwrap_or(0);

        NodeInfoResponse {
            gpus: vec![gpu],
            cpu: Some(cpu),
            memory: Some(memory),
            telemetry_valid: state.telemetry_valid,
            ms_since,
            host_uuid: self.config.host_uuid.clone(),
            cluster_uuid: None,
        }
    }

    pub fn admission(&self) -> Admission {
        let state = self.state.lock();
        let external = &state.external;

        if external.thermal >= self.config.max_thermal {
            return Admission::Refuse(format!(
                "device thermal status is {:?}, at or above the configured maximum {:?}",
                external.thermal, self.config.max_thermal
            ));
        }

        if external.charging == Some(false) {
            if let Some(battery_percent) = external.battery_percent {
                if battery_percent < self.config.min_battery_percent_on_battery {
                    return Admission::Refuse(format!(
                        "battery at {}% while discharging, below the configured minimum {}%",
                        battery_percent, self.config.min_battery_percent_on_battery
                    ));
                }
            }
        }

        Admission::Accept
    }
}

/// Reproduces PAIR's GPU-pressure banding thresholds so a node can predict
/// (e.g. for logging/UI) which band its reported `utilization_percent` will
/// land in once a fresh EWMA is seeded on PAIR's side. Matches `pressureBand`
/// in the PAIR reference checkout,
/// `services/nvpair-job-scheduler/telemetry.go:111-122`:
/// `<40 -> 0, <70 -> 1, <85 -> 2, else -> 3`.
///
/// This is the *seeding* function PAIR uses when a node has no prior EWMA
/// (or stale/invalid telemetry) — see [`pair_pressure_band_with_previous`]
/// for the steady-state banding, which additionally applies hysteresis.
/// Our node only ever reports a raw `utilization_percent` to PAIR, never a
/// band; both functions here exist for our own logging/tests/UI.
pub fn pair_pressure_band(utilization_percent: u32) -> u8 {
    match utilization_percent {
        0..=39 => 0,
        40..=69 => 1,
        70..=84 => 2,
        _ => 3,
    }
}

/// Reproduces PAIR's steady-state pressure banding *with hysteresis*: rising
/// thresholds 40/70/85 %, falling thresholds 35/65/80 %, so a node hovering
/// right at a boundary does not flap between two bands. Matches
/// `pressureWithHysteresis` in the PAIR reference checkout,
/// `services/nvpair-job-scheduler/telemetry.go:124-138`.
///
/// `previous_band` is the band this node was in before this sample (from a
/// prior call to this function, or from [`pair_pressure_band`] when seeding
/// the very first one). A `previous_band` outside `0..=3` is treated as
/// unknown and falls back to [`pair_pressure_band`], exactly as PAIR's Go
/// does for its `previous < 0 || previous > 3` guard.
pub fn pair_pressure_band_with_previous(utilization_percent: u32, previous_band: u8) -> u8 {
    if previous_band > 3 {
        return pair_pressure_band(utilization_percent);
    }

    const UP: [u32; 3] = [40, 70, 85];
    const DOWN: [u32; 4] = [0, 35, 65, 80];

    let mut pressure = previous_band;
    while pressure < 3 && utilization_percent >= UP[pressure as usize] {
        pressure += 1;
    }
    while pressure > 0 && utilization_percent < DOWN[pressure as usize] {
        pressure -= 1;
    }
    pressure
}

/// Linux/Android `/proc` based sampler.
pub struct ProcfsSampler;

impl Sampler for ProcfsSampler {
    fn cpu(&self) -> std::io::Result<RawCpuSample> {
        let stat = std::fs::read_to_string("/proc/stat")?;
        let mut sample = procfs::parse_stat(&stat)?;
        // Best-effort: a missing/unreadable /proc/cpuinfo must never fail the
        // whole sample, only leave the model name empty.
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            sample.model_name = procfs::parse_cpuinfo_model_name(&cpuinfo);
        }
        Ok(sample)
    }

    fn mem(&self) -> std::io::Result<RawMemSample> {
        let meminfo = std::fs::read_to_string("/proc/meminfo")?;
        procfs::parse_meminfo(&meminfo)
    }
}
