//! `GET /v1/node-info` — mirrors `NodeInfoResponse` in
//! `services/nvpair-manual-nodes/manager.go` and the emitter in
//! `services/nvpair-node-info/main.go`.
//!
//! TODO(ticket: protocol/node-info): implement + tests. Field list below is the
//! design contract; keep JSON names exactly as PAIR's struct tags.

use serde::{Deserialize, Serialize};

/// One accelerator entry. PAIR takes the max `utilization_percent` across GPUs
/// as the node's GPU pressure input, so an Android node may report its
/// inference accelerator (GPU/NPU/CPU-fallback) here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct GpuInfo {
    pub name: String,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub vram_bytes: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub vram_used_bytes: u64,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub utilization_percent: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct CpuInfo {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub cores: u32,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub utilization_percent: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct MemoryInfo {
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub total_bytes: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub used_bytes: u64,
}

/// Body of `GET /v1/node-info`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct NodeInfoResponse {
    /// JSON key is literally `GPUs` (capitalised) in PAIR.
    #[serde(rename = "GPUs", default)]
    pub gpus: Vec<GpuInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu: Option<CpuInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory: Option<MemoryInfo>,
    /// False means "GPUs[] utilisation is not trustworthy"; PAIR then treats
    /// the node as telemetry-less (neutral pressure).
    #[serde(rename = "telemetryValid", default)]
    pub telemetry_valid: bool,
    /// Milliseconds since the last telemetry sample.
    #[serde(rename = "msSince", default)]
    pub ms_since: i64,
    /// Stable per-install identity; PAIR dedupes a manual node against mDNS by it.
    #[serde(rename = "hostUuid", default, skip_serializing_if = "String::is_empty")]
    pub host_uuid: String,
    /// Present only when clustered (Phase 2). `null`/absent when not.
    #[serde(rename = "clusterUuid", default, skip_serializing_if = "Option::is_none")]
    pub cluster_uuid: Option<String>,
}

fn is_zero_u64(v: &u64) -> bool {
    *v == 0
}
fn is_zero_u32(v: &u32) -> bool {
    *v == 0
}
