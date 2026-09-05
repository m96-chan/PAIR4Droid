//! PAIR's manual-node probe, replayed.
//!
//! This is a faithful Rust port of the four requests
//! `services/nvpair-manual-nodes/manager.go:250-281` issues against a manual
//! node every `probeInterval = 10s` with `probeTimeout = 3s` (`:28-44`). It
//! exists so `pair-cli probe` and the conformance tests can answer "would PAIR
//! see this node?" without running PAIR.
//!
//! The success criteria are PAIR's, not ours:
//!
//! | Probe | Criterion | Source |
//! | --- | --- | --- |
//! | `GET :11434/` | HTTP 200 exactly; body closed unread | `manager.go:448-471` |
//! | `GET :11434/api/tags` | 200 + JSON; failure yields no models but does **not** flip `ollama_up` | `manager.go:473-497` |
//! | `GET :1234/v1/models` | 200 exactly; a 200 whose body fails to parse still counts as **up** | `manager.go:409-446` |
//! | `GET :14318/v1/node-info` | 200 **and** decodes into `NodeInfoResponse` | `manager.go:499-529` |
//!
//! PAIR disables HTTP keep-alives on the probe transport (`manager.go:196-204`),
//! so every probe is a fresh connection; we mirror that.

use pair_protocol::node_info::NodeInfoResponse;
use std::net::{IpAddr, SocketAddr};
use std::time::{Duration, Instant};

use crate::BoundPorts;

/// PAIR's `probeTimeout` (`services/nvpair-manual-nodes/manager.go:30`).
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(3);

/// Wall-clock cost of each of the three lane probes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ProbeDurations {
    pub ollama: Duration,
    pub lmstudio: Duration,
    pub node_info: Duration,
}

/// The Rust shape of PAIR's `ManualNodeStatus` fields that a probe fills in
/// (`services/nvpair-manual-nodes/manager.go:105-131`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ProbeReport {
    pub ollama_up: bool,
    pub ollama_models: Vec<String>,
    pub lmstudio_up: bool,
    pub lmstudio_models: Vec<String>,
    pub node_info_up: bool,
    pub node_info: Option<NodeInfoResponse>,
    pub durations: ProbeDurations,
}

impl ProbeReport {
    /// PAIR's own reachability predicate (`manager.go:305`): any lane answering
    /// keeps the node alive; three consecutive fully-unreachable probes only
    /// raise an error, they never evict (`manager.go:32-43`).
    pub fn reachable(&self) -> bool {
        self.ollama_up || self.lmstudio_up || self.node_info_up
    }
}

fn base(addr: IpAddr, port: u16) -> String {
    // `SocketAddr`'s Display brackets IPv6 exactly like Go's `net.JoinHostPort`.
    format!("http://{}", SocketAddr::new(addr, port))
}

fn probe_client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(timeout)
        // `noKeepAliveTransport` (`manager.go:196-204`).
        .pool_max_idle_per_host(0)
        .build()
        .unwrap_or_default()
}

/// Run PAIR's probe sequence against a node and report what PAIR would see.
pub async fn probe(addr: IpAddr, ports: BoundPorts, timeout: Duration) -> ProbeReport {
    let client = probe_client(timeout);
    let mut report = ProbeReport::default();

    let start = Instant::now();
    let (up, models) = probe_ollama(&client, addr, ports.ollama).await;
    report.durations.ollama = start.elapsed();
    report.ollama_up = up;
    report.ollama_models = models;

    let start = Instant::now();
    let (up, models) = probe_lmstudio(&client, addr, ports.openai).await;
    report.durations.lmstudio = start.elapsed();
    report.lmstudio_up = up;
    report.lmstudio_models = models;

    let start = Instant::now();
    let (up, info) = probe_node_info(&client, addr, ports.node_info).await;
    report.durations.node_info = start.elapsed();
    report.node_info_up = up;
    report.node_info = info;

    report
}

/// `probeOllama` (`services/nvpair-manual-nodes/manager.go:448-471`): `GET /`
/// must be exactly 200; only then is `/api/tags` fetched.
pub async fn probe_ollama(client: &reqwest::Client, addr: IpAddr, port: u16) -> (bool, Vec<String>) {
    let url = format!("{}/", base(addr, port));
    let Ok(resp) = client.get(&url).send().await else {
        tracing::debug!(%url, "probe ollama failed");
        return (false, Vec::new());
    };
    if resp.status() != reqwest::StatusCode::OK {
        tracing::debug!(%url, status = resp.status().as_u16(), "probe ollama non-OK");
        return (false, Vec::new());
    }
    // PAIR closes the body without reading it (`manager.go:458`).
    drop(resp);
    (true, fetch_ollama_models(client, addr, port).await)
}

/// `fetchOllamaModels` (`manager.go:473-497`): reads `models[].name` only, and a
/// failure here does **not** mark the Ollama lane down.
async fn fetch_ollama_models(client: &reqwest::Client, addr: IpAddr, port: u16) -> Vec<String> {
    #[derive(serde::Deserialize)]
    struct Tags {
        #[serde(default)]
        models: Vec<Named>,
    }
    #[derive(serde::Deserialize)]
    struct Named {
        #[serde(default)]
        name: String,
    }

    let url = format!("{}/api/tags", base(addr, port));
    let Ok(resp) = client.get(&url).send().await else { return Vec::new() };
    if resp.status() != reqwest::StatusCode::OK {
        return Vec::new();
    }
    match resp.json::<Tags>().await {
        Ok(tags) => tags.models.into_iter().map(|m| m.name).collect(),
        Err(_) => Vec::new(),
    }
}

/// `probeLMStudio` (`services/nvpair-manual-nodes/manager.go:409-446`): one
/// `GET /v1/models` is both liveness and inventory. A 200 whose body fails to
/// decode still reports **up** with no models (`:431-436`); only non-empty
/// `data[].id` values are kept (`:437-442`).
pub async fn probe_lmstudio(client: &reqwest::Client, addr: IpAddr, port: u16) -> (bool, Vec<String>) {
    #[derive(serde::Deserialize)]
    struct Models {
        #[serde(default)]
        data: Vec<Identified>,
    }
    #[derive(serde::Deserialize)]
    struct Identified {
        #[serde(default)]
        id: String,
    }

    let url = format!("{}/v1/models", base(addr, port));
    let Ok(resp) = client.get(&url).send().await else {
        tracing::debug!(%url, "probe lmstudio failed");
        return (false, Vec::new());
    };
    if resp.status() != reqwest::StatusCode::OK {
        tracing::debug!(%url, status = resp.status().as_u16(), "probe lmstudio non-OK");
        return (false, Vec::new());
    }
    match resp.json::<Models>().await {
        Ok(models) => (true, models.data.into_iter().map(|m| m.id).filter(|id| !id.is_empty()).collect()),
        Err(_) => {
            tracing::debug!(%url, "probe lmstudio up (models parse failed)");
            (true, Vec::new())
        }
    }
}

/// `probeNodeInfo` (`services/nvpair-manual-nodes/manager.go:499-529`): a
/// transport error, a non-200 **or** a decode failure all yield "down" and a
/// zero-valued `NodeInfoResponse`.
pub async fn probe_node_info(
    client: &reqwest::Client,
    addr: IpAddr,
    port: u16,
) -> (bool, Option<NodeInfoResponse>) {
    let url = format!("{}/v1/node-info", base(addr, port));
    let Ok(resp) = client.get(&url).send().await else {
        tracing::debug!(%url, "probe node-info failed");
        return (false, None);
    };
    if resp.status() != reqwest::StatusCode::OK {
        tracing::debug!(%url, status = resp.status().as_u16(), "probe node-info non-OK");
        return (false, None);
    }
    match resp.json::<NodeInfoResponse>().await {
        Ok(info) => (true, Some(info)),
        Err(e) => {
            tracing::debug!(%url, error = %e, "probe node-info decode failed");
            (false, None)
        }
    }
}
