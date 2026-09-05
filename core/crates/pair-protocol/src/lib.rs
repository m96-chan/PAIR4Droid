//! Wire types that NVIDIA Personal AI Router (PAIR) reads from a *manual node*.
//!
//! Source of truth: `docs/pair-contract.md` (every field cites the PAIR Go
//! source). This crate is pure data: serde structs plus (de)serialisation tests
//! against the exact JSON bodies PAIR's own tests/fakes use. No I/O, no async.
//!
//! Module map
//! - [`node_info`] – `GET :14318/v1/node-info` ([`node_info::NodeInfoResponse`], [`node_info::GpuInfo`], [`node_info::CpuInfo`], [`node_info::MemoryInfo`])
//! - [`openai`]    – `:1234` lane (`/v1/models`, `/v1/chat/completions` request/response/SSE chunk, [`openai::sse`])
//! - [`ollama`]    – `:11434` lane (`/api/tags`, `/api/chat`, `/api/generate`, `/api/show`, `/api/version`, `/api/ps`, [`ollama::ndjson`])
//!
//! Rules
//! - Field names/JSON casing MUST match PAIR byte-for-byte (`GPUs`,
//!   `telemetryValid`, `msSince`, `hostUuid`, `utilization_percent`, …).
//! - Unknown incoming fields are ignored, never rejected — PAIR forwards a
//!   client's body verbatim, so anything an OpenAI/Ollama client can send will
//!   arrive here.
//! - Outgoing fields use `skip_serializing_if` exactly where PAIR uses
//!   `omitempty`, because PAIR uses "absent" to mean "unknown".
//! - Go marshals a nil slice as `null`; every list we decode tolerates it.

pub mod node_info;
pub mod ollama;
pub mod openai;

mod serde_util;

#[doc(inline)]
pub use ollama::ndjson;
#[doc(inline)]
pub use openai::sse;

/// Ports PAIR probes on a manual node. Compile-time constants in PAIR
/// (`services/nvpair-manual-nodes/manager.go:254`, `:400-404`, `:264`), not
/// configurable per node — a node that does not bind these is never reached.
pub mod ports {
    /// Ollama-compatible lane (`GET /` must be 200, `GET /api/tags`).
    pub const OLLAMA: u16 = 11434;
    /// OpenAI-compatible lane (LM Studio's default; `GET /v1/models`).
    pub const OPENAI: u16 = 1234;
    /// `GET /v1/node-info` (plain HTTP).
    pub const NODE_INFO: u16 = 14318;
}

/// `created_at` for the Ollama lane: RFC3339 UTC with nanosecond precision,
/// e.g. `2026-09-05T12:34:56.123456789Z` — the format real Ollama emits.
pub fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Nanos, true)
}

/// `created` for the OpenAI lane: seconds since the Unix epoch.
pub fn now_unix() -> i64 {
    chrono::Utc::now().timestamp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ports_are_pairs_compile_time_constants() {
        assert_eq!((ports::OLLAMA, ports::OPENAI, ports::NODE_INFO), (11434, 1234, 14318));
    }

    #[test]
    fn now_rfc3339_is_nanosecond_utc() {
        let s = now_rfc3339();
        assert!(s.ends_with('Z'), "{s}");
        // yyyy-mm-ddThh:mm:ss.nnnnnnnnnZ
        assert_eq!(s.len(), 30, "{s}");
        assert!(chrono::DateTime::parse_from_rfc3339(&s).is_ok(), "{s}");
    }

    #[test]
    fn now_unix_is_seconds() {
        assert!(now_unix() > 1_700_000_000);
    }
}
