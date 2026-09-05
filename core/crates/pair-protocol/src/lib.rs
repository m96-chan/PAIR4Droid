//! Wire types that NVIDIA Personal AI Router (PAIR) reads from a *manual node*.
//!
//! Source of truth: `docs/pair-contract.md` (every field cites the PAIR Go source).
//! This crate is pure data: serde structs + (de)serialisation tests against the
//! exact JSON bodies PAIR's own tests/fakes use. No I/O, no async.
//!
//! Module map
//! - [`node_info`]  – `GET :14318/v1/node-info` (`NodeInfoResponse`, `GpuInfo`, `CpuInfo`, `MemoryInfo`)
//! - [`openai`]     – `:1234` lane (`/v1/models`, `/v1/chat/completions` request/response/SSE chunk)
//! - [`ollama`]     – `:11434` lane (`/api/tags`, `/api/chat`, `/api/generate`, `/api/show`, `/api/version`, `/api/ps`)
//!
//! Rules
//! - Field names/JSON casing MUST match PAIR byte-for-byte (`GPUs`, `telemetryValid`, `msSince`, `hostUuid`, `utilization_percent`, ...).
//! - Unknown incoming fields are ignored (`#[serde(default)]`), never rejected.
//! - Optional outgoing fields use `skip_serializing_if` exactly where PAIR uses `omitempty`.

pub mod node_info;
pub mod ollama;
pub mod openai;

/// Ports PAIR probes on a manual node. Compile-time constants in PAIR
/// (`services/nvpair-manual-nodes/manager.go`), so they are constants here too.
pub mod ports {
    /// Ollama-compatible lane (`GET /` must be 200, `GET /api/tags`).
    pub const OLLAMA: u16 = 11434;
    /// OpenAI-compatible lane (LM Studio default; `GET /v1/models`).
    pub const OPENAI: u16 = 1234;
    /// `GET /v1/node-info` (plain HTTP).
    pub const NODE_INFO: u16 = 14318;
}
