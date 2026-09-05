//! The node: three axum servers sharing one `Engine` and one `TelemetrySource`.
//!
//! TODO(ticket: node/*): implement + tests (see docs/pair-contract.md).
//!
//! Design contract
//! - [`NodeConfig`] chooses bind address (default `0.0.0.0`) and the three ports
//!   (defaults = `pair_protocol::ports::*`; tests pass `0` for ephemeral ports).
//! - [`Node::start`] binds all listeners *before* returning (so `NodeHandle::ports()`
//!   is exact) and spawns the servers + the telemetry tick loop on the current
//!   tokio runtime. `NodeHandle::shutdown()` stops everything gracefully.
//! - Lanes (one module each):
//!   * [`node_info`]  `GET /v1/node-info` → `telemetry.node_info()`.
//!   * [`openai`]     `GET /v1/models`, `POST /v1/chat/completions` (SSE when `stream:true`,
//!                    `Content-Type: text/event-stream`, `data: {chunk}\n\n`, terminated by `data: [DONE]\n\n`).
//!   * [`ollama`]     `GET /` → `200 text/plain "Ollama is running"`, `/api/tags`, `/api/version`,
//!                    `/api/chat`, `/api/generate` (NDJSON, `application/x-ndjson`), `/api/show`, `/api/ps`.
//! - Errors: unknown model → 404 with the lane's error envelope (PAIR fails over
//!   to another owner on 404); admission refused → 503; malformed JSON → 400.
//! - Every request is logged with `tracing` at info (method, path, model, status, ms).
//! - Security: bind is LAN-wide by design (PAIR probes from another host); no auth
//!   (PAIR sends none). Paths outside the contract → 404.

pub mod node_info;
pub mod ollama;
pub mod openai;

use pair_engine::SharedEngine;
use pair_telemetry::TelemetrySource;
use std::net::{IpAddr, Ipv4Addr};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeConfig {
    pub bind: IpAddr,
    pub openai_port: u16,
    pub ollama_port: u16,
    pub node_info_port: u16,
    /// Reported by `GET /api/version` (Ollama lane). Keep it a valid semver.
    pub ollama_version: String,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            bind: IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            openai_port: pair_protocol::ports::OPENAI,
            ollama_port: pair_protocol::ports::OLLAMA,
            node_info_port: pair_protocol::ports::NODE_INFO,
            ollama_version: "0.11.0".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundPorts {
    pub openai: u16,
    pub ollama: u16,
    pub node_info: u16,
}

/// Shared state handed to every handler.
#[derive(Clone)]
pub struct AppState {
    pub engine: SharedEngine,
    pub telemetry: Arc<dyn TelemetrySource>,
    pub config: Arc<NodeConfig>,
}

pub struct Node;

pub struct NodeHandle {
    // TODO(ticket node/server)
}

impl Node {
    pub async fn start(
        config: NodeConfig,
        engine: SharedEngine,
        telemetry: Arc<dyn TelemetrySource>,
    ) -> anyhow::Result<NodeHandle> {
        let _ = (config, engine, telemetry);
        todo!("ticket node/server")
    }
}

impl NodeHandle {
    pub fn ports(&self) -> BoundPorts {
        todo!()
    }
    pub async fn shutdown(self) {
        todo!()
    }
}
