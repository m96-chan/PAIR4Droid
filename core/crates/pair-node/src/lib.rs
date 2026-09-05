//! The node: three axum servers sharing one `Engine` and one `TelemetrySource`.
//!
//! Design contract
//! - [`NodeConfig`] chooses bind address (default `0.0.0.0`) and the three ports
//!   (defaults = `pair_protocol::ports::*`; tests pass `0` for ephemeral ports).
//! - [`Node::start`] binds all listeners *before* returning (so `NodeHandle::ports()`
//!   is exact) and spawns the servers + the telemetry tick loop on the current
//!   tokio runtime. `NodeHandle::shutdown()` stops everything gracefully.
//! - Lanes (one module each):
//!   * [`node_info`]  `GET /v1/node-info` → `telemetry.node_info()`.
//!   * [`openai`]     `GET /v1/models`, `POST /v1/chat/completions` (SSE when
//!     `stream:true`, `Content-Type: text/event-stream`, `data: {chunk}\n\n`, terminated
//!     by `data: [DONE]\n\n`).
//!   * [`ollama`]     `GET /` → `200 text/plain "Ollama is running"`, `/api/tags`,
//!     `/api/version`, `/api/chat`, `/api/generate` (NDJSON, `application/x-ndjson`),
//!     `/api/show`, `/api/ps`.
//! - [`probe`] replays PAIR's own manual-node probe against a node (used by the CLI
//!   and by the conformance tests).
//! - Errors: unknown model → 404 with the lane's error envelope (PAIR fails over
//!   to another owner on 404); admission refused → 503; malformed JSON → 400.
//! - Every request is logged with `tracing` at info (method, path, model, status, ms).
//! - Security: bind is LAN-wide by design (PAIR probes from another host); no auth
//!   (PAIR sends none). Paths outside the contract → 404.
//!
//! PAIR references (checkout at `/home/user/nvidia/personal-ai-router`, see
//! `docs/pair-contract.md`): the probe sequence is
//! `services/nvpair-manual-nodes/manager.go:250-281` and the ports are compile-time
//! constants there (`:400-404`, `:254`, `:264`).

pub mod node_info;
pub mod ollama;
pub mod openai;
pub mod probe;

mod stream;

use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use axum::Router;
use pair_engine::SharedEngine;
use pair_telemetry::{InferenceLoad, TelemetrySource};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio::sync::watch;
use tokio::task::JoinHandle;

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

impl BoundPorts {
    /// The ports PAIR probes by default (`docs/pair-contract.md` §1.6).
    pub const DEFAULT: BoundPorts = BoundPorts {
        openai: pair_protocol::ports::OPENAI,
        ollama: pair_protocol::ports::OLLAMA,
        node_info: pair_protocol::ports::NODE_INFO,
    };
}

impl Default for BoundPorts {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Shared state handed to every handler.
#[derive(Clone)]
pub struct AppState {
    pub engine: SharedEngine,
    pub telemetry: Arc<dyn TelemetrySource>,
    pub config: Arc<NodeConfig>,
}

/// Response extension a lane handler sets so the access log can name the model
/// of an inference request without re-parsing the body.
#[derive(Debug, Clone)]
pub struct LoggedModel(pub String);

pub struct Node;

pub struct NodeHandle {
    ports: BoundPorts,
    shutdown: watch::Sender<bool>,
    tasks: Vec<JoinHandle<()>>,
}

impl Node {
    /// Bind the three listeners, then spawn the three servers and the telemetry
    /// tick loop. Binding happens before this returns so [`NodeHandle::ports`]
    /// reports the ports the OS actually gave us (tests bind port 0).
    pub async fn start(
        config: NodeConfig,
        engine: SharedEngine,
        telemetry: Arc<dyn TelemetrySource>,
    ) -> anyhow::Result<NodeHandle> {
        let openai_listener = bind(config.bind, config.openai_port, "openai").await?;
        let ollama_listener = bind(config.bind, config.ollama_port, "ollama").await?;
        let node_info_listener = bind(config.bind, config.node_info_port, "node-info").await?;

        let ports = BoundPorts {
            openai: openai_listener.local_addr()?.port(),
            ollama: ollama_listener.local_addr()?.port(),
            node_info: node_info_listener.local_addr()?.port(),
        };

        let state = AppState {
            engine: Arc::clone(&engine),
            telemetry: Arc::clone(&telemetry),
            config: Arc::new(config),
        };

        let (shutdown, _) = watch::channel(false);
        let tasks = vec![
            serve(openai_listener, openai::router(state.clone()), shutdown.subscribe(), "openai"),
            serve(ollama_listener, ollama::router(state.clone()), shutdown.subscribe(), "ollama"),
            serve(node_info_listener, node_info::router(state.clone()), shutdown.subscribe(), "node-info"),
            spawn_telemetry_loop(engine, telemetry, shutdown.subscribe()),
        ];

        tracing::info!(
            openai = ports.openai,
            ollama = ports.ollama,
            node_info = ports.node_info,
            "pair node listening"
        );

        Ok(NodeHandle { ports, shutdown, tasks })
    }
}

impl NodeHandle {
    pub fn ports(&self) -> BoundPorts {
        self.ports
    }

    /// Signal graceful shutdown to every server and the telemetry loop, then
    /// await all of them.
    pub async fn shutdown(self) {
        let _ = self.shutdown.send(true);
        for task in self.tasks {
            let _ = task.await;
        }
    }
}

async fn bind(addr: IpAddr, port: u16, lane: &str) -> anyhow::Result<TcpListener> {
    let sock = SocketAddr::new(addr, port);
    TcpListener::bind(sock).await.map_err(|e| anyhow::anyhow!("failed to bind {lane} lane on {sock}: {e}"))
}

fn serve(
    listener: TcpListener,
    router: Router,
    mut shutdown: watch::Receiver<bool>,
    lane: &'static str,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let result = axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = shutdown.changed().await;
            })
            .await;
        if let Err(e) = result {
            tracing::error!(lane, error = %e, "lane server stopped with an error");
        }
    })
}

/// Pushes `Engine::status()` into telemetry and samples on the source's own
/// cadence. PAIR only folds a node's telemetry into scheduling while
/// `telemetryValid` is true and `msSince <= 10000`
/// (`services/nvpair-job-scheduler/telemetry.go:45-46`), so this loop must keep
/// running for the whole lifetime of the node.
fn spawn_telemetry_loop(
    engine: SharedEngine,
    telemetry: Arc<dyn TelemetrySource>,
    mut shutdown: watch::Receiver<bool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let period = telemetry.sample_interval().max(Duration::from_millis(1));
        let mut ticker = tokio::time::interval(period);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                _ = ticker.tick() => {
                    let status = engine.status();
                    telemetry.set_inference_load(InferenceLoad {
                        active: status.active,
                        queued: status.queued,
                        loaded_bytes: status.loaded_bytes,
                    });
                    telemetry.tick();
                }
                _ = shutdown.changed() => break,
            }
        }
    })
}

/// Access log: method, path, status, elapsed ms, plus the model when an
/// inference handler recorded one via [`LoggedModel`].
pub(crate) async fn access_log(req: Request, next: Next) -> Response {
    let method = req.method().clone();
    let path = req.uri().path().to_owned();
    let start = Instant::now();
    let response = next.run(req).await;
    let ms = start.elapsed().as_millis();
    let status = response.status().as_u16();
    match response.extensions().get::<LoggedModel>() {
        Some(LoggedModel(model)) => {
            tracing::info!(%method, path, model = %model, status, ms, "request");
        }
        None => tracing::info!(%method, path, status, ms, "request"),
    }
    response
}

/// 404 for anything outside the contract. PAIR treats a 404 on an inference
/// path as "this owner does not have the model" and fails over
/// (`services/lmstudio-proxy/proxy.go:1015-1016`).
pub(crate) async fn not_found() -> Response {
    (axum::http::StatusCode::NOT_FOUND, "not found").into_response_with_nosniff()
}

/// Small helper so every lane answers 404s the same way.
trait IntoResponseWithNosniff {
    fn into_response_with_nosniff(self) -> Response;
}

impl<T: axum::response::IntoResponse> IntoResponseWithNosniff for T {
    fn into_response_with_nosniff(self) -> Response {
        let mut response = axum::response::IntoResponse::into_response(self);
        response.headers_mut().insert(
            axum::http::header::X_CONTENT_TYPE_OPTIONS,
            axum::http::HeaderValue::from_static("nosniff"),
        );
        response
    }
}
