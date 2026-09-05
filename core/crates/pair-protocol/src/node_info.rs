//! `GET /v1/node-info` on port [`crate::ports::NODE_INFO`].
//!
//! Mirrors `NodeInfoResponse` in `services/nvpair-node-info/main.go:74-102`
//! (the emitter) and the subset `services/nvpair-manual-nodes/manager.go:69-81`
//! parses (the consumer). Field-by-field notes live in `docs/pair-contract.md`
//! §2.1; the citations below point at the Go struct tags themselves, because
//! those tags — not the docs — are the contract.
//!
//! Two rules that are easy to get wrong:
//!
//! - `GPUs`, `telemetryValid` and `msSince` carry **no** `omitempty`
//!   (`services/nvpair-node-info/main.go:75`, `:78`, `:79`), so they are always
//!   on the wire. Everything else is dropped when zero/empty, because PAIR
//!   deliberately uses "absent" to mean "this host could not read the value"
//!   (`services/nvpair-node-info/main.go:51-58`).
//! - `clusterUuid` is a `*string` with `omitempty`, giving **three** wire states:
//!   absent = unknown, `""` = belongs to no cluster, a value = that principal
//!   (`services/nvpair-node-info/main.go:94-101`). Collapsing absent into `""`
//!   makes a peer clear a correct annotation elsewhere in the fleet.

use serde::{Deserialize, Serialize};

use crate::serde_util::{is_zero_u32, is_zero_u64, null_to_default};

/// One accelerator entry — `GPUInfo`, `services/nvpair-node-info/main.go:30-49`.
///
/// PAIR takes the **maximum** `utilization_percent` across the array as the
/// node's GPU pressure input (`docs/pair-contract.md` §2.4), so an Android node
/// reports its inference accelerator (GPU/NPU/CPU fallback) here; see
/// `docs/adr/0003-accelerator-utilization-is-inference-busy.md`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct GpuInfo {
    /// `services/nvpair-node-info/main.go:31` — no `omitempty`, always emitted.
    pub name: String,
    /// Total VRAM in bytes; on a unified-memory host, total system RAM
    /// (`services/nvpair-node-info/README.md:86`).
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub vram_bytes: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub vram_used_bytes: u64,
    /// 0–100. The only field in the whole payload that feeds scheduling
    /// (`services/nvpair-job-scheduler/telemetry.go`, `docs/pair-contract.md` §2.4).
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub utilization_percent: u32,
}

impl GpuInfo {
    /// The single accelerator entry an Android node advertises.
    pub fn accelerator(name: impl Into<String>, utilization_percent: u32) -> Self {
        Self { name: name.into(), utilization_percent, ..Default::default() }
    }
}

/// `CPUInfo`, `services/nvpair-node-info/main.go:59-63`. Every field is
/// `omitempty`; the whole object is omitted when the host cannot introspect it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct CpuInfo {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub cores: u32,
    /// Display-only in PAIR's UI; it does not feed scheduling.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub utilization_percent: u32,
}

/// `MemoryInfo`, `services/nvpair-node-info/main.go:69-72`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct MemoryInfo {
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub total_bytes: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub used_bytes: u64,
}

/// Body of `GET /v1/node-info` — `services/nvpair-node-info/main.go:74-102`.
///
/// Field order here matches the Go declaration order, so `serde_json` reproduces
/// PAIR's `json.Marshal` output byte for byte (pinned by
/// `tests/node_info.rs::full_body_round_trips_byte_for_byte`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct NodeInfoResponse {
    /// JSON key is literally `GPUs` — the only PascalCase key in the payload
    /// (`services/nvpair-node-info/main.go:75`). Always emitted; a Go peer may
    /// send `null` here, which decodes as empty.
    #[serde(rename = "GPUs", default, deserialize_with = "null_to_default")]
    pub gpus: Vec<GpuInfo>,
    /// Omitted whole when the host could not read its CPU
    /// (`services/nvpair-node-info/main.go:199-203`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu: Option<CpuInfo>,
    /// Omitted whole when total physical memory is unknown
    /// (`services/nvpair-node-info/main.go:204-209`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory: Option<MemoryInfo>,
    /// `false` means "`GPUs[].utilization_percent` is not trustworthy"; PAIR
    /// then scores the node at the neutral pressure band instead of band 0
    /// (`docs/pair-contract.md` §2.6). Always on the wire
    /// (`services/nvpair-node-info/main.go:78`).
    #[serde(rename = "telemetryValid", default)]
    pub telemetry_valid: bool,
    /// Age in milliseconds of the sample `telemetry_valid` describes; 0 when
    /// invalid (`services/nvpair-node-info/main.go:214-223`). The broker adds
    /// its own elapsed time and drops telemetry older than 10 s
    /// (`services/nvpair-ui-broker/telemetry.go:141-149`), so keep it small.
    /// Always on the wire (`services/nvpair-node-info/main.go:79`).
    #[serde(rename = "msSince", default)]
    pub ms_since: i64,
    /// Stable per-install identity. PAIR rekeys a manual node to this as soon
    /// as it appears and dedupes it against an mDNS record of the same machine
    /// (`services/nvpair-manual-nodes/manager.go:75-80`,
    /// `services/nvpair-ui-broker/manualnodes.go:71-97`). The scheduler drops
    /// telemetry with an empty `hostUuid`
    /// (`services/nvpair-job-scheduler/telemetry.go:42-44`).
    #[serde(rename = "hostUuid", default, skip_serializing_if = "String::is_empty")]
    pub host_uuid: String,
    /// Three states, see the module docs. `None` (absent) is the honest answer
    /// for a node that is not cluster-aware, which is Phase 1.
    #[serde(rename = "clusterUuid", default, skip_serializing_if = "Option::is_none")]
    pub cluster_uuid: Option<String>,
}

impl NodeInfoResponse {
    /// The minimum body that makes a node visible to PAIR's scheduler:
    /// one accelerator, valid telemetry, and a stable identity
    /// (`docs/pair-contract.md` §2.7).
    pub fn scheduling_visible(
        host_uuid: impl Into<String>,
        accelerator: impl Into<String>,
        utilization_percent: u32,
        ms_since: i64,
    ) -> Self {
        Self {
            gpus: vec![GpuInfo::accelerator(accelerator, utilization_percent)],
            telemetry_valid: true,
            ms_since,
            host_uuid: host_uuid.into(),
            ..Default::default()
        }
    }

    /// The value PAIR bands into `GPUPressure`: the maximum utilisation across
    /// `GPUs`, or 0 when the array is empty (`docs/pair-contract.md` §2.4).
    pub fn max_utilization_percent(&self) -> u32 {
        self.gpus.iter().map(|g| g.utilization_percent).max().unwrap_or(0)
    }
}
