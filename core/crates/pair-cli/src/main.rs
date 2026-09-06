//! `pair4droid` — desktop runner for developing against a real PAIR without a
//! phone (ticket #13). Two subcommands:
//!
//! ```text
//! pair4droid serve [--bind 0.0.0.0] [--openai-port 1234] [--ollama-port 11434]
//!                  [--node-info-port 14318] [--mock model1,model2]
//!                  [--models-dir DIR] [--accelerator-name "CPU (llama.cpp)"]
//!                  [--host-uuid <uuid>] [--model-budget-bytes N]
//!
//! pair4droid probe <addr> [--openai-port 1234] [--ollama-port 11434]
//!                  [--node-info-port 14318] [--timeout-secs 3] [--json]
//! ```
//!
//! `serve` runs the three PAIR lanes (`pair-node::Node`) against either
//! [`pair_engine::mock::MockEngine`] (`--mock`, or the default when neither
//! `--mock` nor `--models-dir` is given) or [`pair_engine::llama::LlamaEngine`]
//! (`--models-dir`, requires this binary to be built with `--features llama`).
//! It prints a machine-readable `ports openai=.. ollama=.. node_info=..` line
//! (used by `tests/cli.rs` and by anyone scripting against ephemeral ports),
//! one human-readable line per lane, and PAIR's "add manual node" hint, then
//! waits for Ctrl-C to shut down gracefully.
//!
//! `probe` replays PAIR's own manual-node probe sequence
//! (`pair_node::probe::probe`) against any node and prints a verdict; it exits
//! 0 when at least one inference lane (Ollama or OpenAI/LM Studio) is up with
//! at least one model, and 2 otherwise, so it can be used as a health check.

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use pair_engine::mock::MockEngine;
use pair_engine::SharedEngine;
use pair_node::probe::{probe, ProbeReport};
use pair_node::{BoundPorts, Node, NodeConfig};
use pair_protocol::node_info::NodeInfoResponse;
use pair_telemetry::{ProcfsSampler, Telemetry, TelemetryConfig, TelemetrySource};
use std::io::Write;
use std::net::{IpAddr, Ipv4Addr};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

/// Default accelerator name reported in node-info when `--accelerator-name`
/// is not given (ticket #13 acceptance criteria).
const DEFAULT_ACCELERATOR_NAME: &str = "CPU (llama.cpp)";
/// Fallback model RAM budget when `/proc/meminfo` cannot be read.
const FALLBACK_MODEL_BUDGET_BYTES: u64 = 4 * 1024 * 1024 * 1024;

#[derive(Parser)]
#[command(name = "pair4droid", about = "Run a PAIR4Droid node, or probe one, from the desktop.")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run a PAIR manual node (openai + ollama + node-info lanes).
    Serve(ServeArgs),
    /// Replay PAIR's manual-node probe against a node and print the verdict.
    Probe(ProbeArgs),
}

#[derive(Args)]
struct ServeArgs {
    /// Address to bind all three lanes on.
    #[arg(long, default_value = "0.0.0.0")]
    bind: IpAddr,
    #[arg(long, default_value_t = pair_protocol::ports::OPENAI)]
    openai_port: u16,
    #[arg(long, default_value_t = pair_protocol::ports::OLLAMA)]
    ollama_port: u16,
    #[arg(long, default_value_t = pair_protocol::ports::NODE_INFO)]
    node_info_port: u16,
    /// Comma-separated model names to advertise via `MockEngine`. Default
    /// (`demo`) applies only when neither this nor `--models-dir` is given.
    #[arg(long, value_name = "model1,model2")]
    mock: Option<String>,
    /// Directory of GGUF models for `LlamaEngine`. Requires this binary to be
    /// built with `--features llama`.
    #[arg(long)]
    models_dir: Option<PathBuf>,
    #[arg(long, default_value = DEFAULT_ACCELERATOR_NAME)]
    accelerator_name: String,
    /// Stable node identity. Default: read (or create) `~/.pair4droid/host-uuid`.
    #[arg(long)]
    host_uuid: Option<String>,
    /// Bytes of RAM to advertise as available for models. Default: 50% of
    /// `/proc/meminfo` MemTotal, or 4 GiB if that can't be read.
    #[arg(long)]
    model_budget_bytes: Option<u64>,
}

#[derive(Args)]
struct ProbeArgs {
    /// Address of the node to probe.
    addr: IpAddr,
    #[arg(long, default_value_t = pair_protocol::ports::OPENAI)]
    openai_port: u16,
    #[arg(long, default_value_t = pair_protocol::ports::OLLAMA)]
    ollama_port: u16,
    #[arg(long, default_value_t = pair_protocol::ports::NODE_INFO)]
    node_info_port: u16,
    #[arg(long, default_value_t = 3)]
    timeout_secs: u64,
    /// Print the full `ProbeReport` as JSON instead of the verdict table.
    #[arg(long)]
    json: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    match cli.command {
        Command::Serve(args) => run_serve(args).await?,
        Command::Probe(args) => {
            let code = run_probe(args).await?;
            std::process::exit(code);
        }
    }
    Ok(())
}

fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}

// --------------------------------------------------------------------- serve

async fn run_serve(args: ServeArgs) -> Result<()> {
    let engine = build_engine(args.mock, args.models_dir)?;

    let host_uuid = match args.host_uuid {
        Some(uuid) => uuid,
        None => load_or_create_host_uuid()?,
    };
    let model_budget_bytes = args.model_budget_bytes.unwrap_or_else(default_model_budget_bytes);

    let telemetry_config = TelemetryConfig::default_for(host_uuid, args.accelerator_name, model_budget_bytes);
    let telemetry: Arc<dyn TelemetrySource> =
        Arc::new(Telemetry::new(telemetry_config, Box::new(ProcfsSampler::default())));

    let node_config = NodeConfig {
        bind: args.bind,
        openai_port: args.openai_port,
        ollama_port: args.ollama_port,
        node_info_port: args.node_info_port,
        ..NodeConfig::default()
    };

    let handle = Node::start(node_config, engine, telemetry).await.context("failed to start pair node")?;
    let ports = handle.ports();

    // Machine-readable, printed before anything else so a caller (or
    // `tests/cli.rs`) can learn the actual bound ports without racing the
    // human-readable lines below.
    println!("ports openai={} ollama={} node_info={}", ports.openai, ports.ollama, ports.node_info);

    let display_ip = if args.bind.is_unspecified() { detect_lan_ip() } else { args.bind };
    println!("openai      http://{display_ip}:{}/v1/models", ports.openai);
    println!("ollama      http://{display_ip}:{}/", ports.ollama);
    println!("node-info   http://{display_ip}:{}/v1/node-info", ports.node_info);
    println!("PAIR → Nodes → Add manual node → {display_ip}");
    let _ = std::io::stdout().flush();

    tokio::signal::ctrl_c().await.context("failed to listen for ctrl-c")?;
    tracing::info!("received ctrl-c, shutting down");
    handle.shutdown().await;
    Ok(())
}

/// `--mock` wins if both are given; with neither, default to a single `demo`
/// mock model (ticket #13 acceptance criteria).
fn build_engine(mock: Option<String>, models_dir: Option<PathBuf>) -> Result<SharedEngine> {
    if let Some(spec) = mock {
        return Ok(Arc::new(mock_engine_from_arg(&spec)));
    }
    if let Some(dir) = models_dir {
        return build_llama_engine(dir);
    }
    Ok(Arc::new(MockEngine::with_models(&["demo"])))
}

fn mock_engine_from_arg(spec: &str) -> MockEngine {
    let names: Vec<&str> = spec.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    if names.is_empty() {
        MockEngine::with_models(&["demo"])
    } else {
        MockEngine::with_models(&names)
    }
}

#[cfg(feature = "llama")]
fn build_llama_engine(dir: PathBuf) -> Result<SharedEngine> {
    let engine = pair_engine::llama::LlamaEngine::new(dir, pair_engine::llama::LlamaConfig::default())
        .map_err(|e| anyhow::anyhow!("failed to initialise llama engine: {e}"))?;
    Ok(Arc::new(engine))
}

#[cfg(not(feature = "llama"))]
fn build_llama_engine(dir: PathBuf) -> Result<SharedEngine> {
    anyhow::bail!(
        "--models-dir {} was given, but this pair4droid binary was built without the `llama` \
         cargo feature; rebuild with `cargo build --features llama` to use LlamaEngine, or drop \
         --models-dir to use --mock instead",
        dir.display()
    )
}

/// Reads `~/.pair4droid/host-uuid`, creating a fresh v4 UUID and the file (and
/// its parent directory) on first run. `HOME` unset (no home directory to
/// persist into) falls back to a fresh UUID for this run only.
fn load_or_create_host_uuid() -> Result<String> {
    let Some(home) = std::env::var_os("HOME") else {
        tracing::warn!("HOME is not set; using a fresh host-uuid for this run only");
        return Ok(uuid::Uuid::new_v4().to_string());
    };
    let path = PathBuf::from(home).join(".pair4droid").join("host-uuid");

    if let Ok(existing) = std::fs::read_to_string(&path) {
        let trimmed = existing.trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_string());
        }
    }

    let new_uuid = uuid::Uuid::new_v4().to_string();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::write(&path, &new_uuid).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(new_uuid)
}

/// 50% of `/proc/meminfo`'s `MemTotal`, or [`FALLBACK_MODEL_BUDGET_BYTES`] if
/// that can't be read or parsed.
fn default_model_budget_bytes() -> u64 {
    let Ok(content) = std::fs::read_to_string("/proc/meminfo") else {
        return FALLBACK_MODEL_BUDGET_BYTES;
    };
    for line in content.lines() {
        let Some(rest) = line.strip_prefix("MemTotal:") else { continue };
        let Some(kb) = rest.split_whitespace().next().and_then(|s| s.parse::<u64>().ok()) else {
            break;
        };
        if kb > 0 {
            return (kb * 1024) / 2;
        }
    }
    FALLBACK_MODEL_BUDGET_BYTES
}

/// The LAN-facing IPv4 address, guessed via the classic UDP-connect trick
/// (connecting a UDP socket never sends a packet, but makes the OS pick the
/// outbound interface, which `local_addr` then reports). Falls back to
/// loopback when there's no route (offline, sandboxed, IPv6-only, ...).
fn detect_lan_ip() -> IpAddr {
    (|| -> std::io::Result<IpAddr> {
        let socket = std::net::UdpSocket::bind("0.0.0.0:0")?;
        socket.connect("10.255.255.255:1")?;
        Ok(socket.local_addr()?.ip())
    })()
    .unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST))
}

// --------------------------------------------------------------------- probe

async fn run_probe(args: ProbeArgs) -> Result<i32> {
    let ports =
        BoundPorts { openai: args.openai_port, ollama: args.ollama_port, node_info: args.node_info_port };
    let timeout = Duration::from_secs(args.timeout_secs);
    let report = probe(args.addr, ports, timeout).await;

    // At least one inference lane up with at least one model (ticket #13
    // acceptance criteria); node-info alone doesn't count, PAIR can't route
    // inference to a node with no model list.
    let healthy = (report.ollama_up && !report.ollama_models.is_empty())
        || (report.lmstudio_up && !report.lmstudio_models.is_empty());

    if args.json {
        let json = ProbeReportJson::from(&report);
        println!("{}", serde_json::to_string_pretty(&json).context("failed to serialise probe report")?);
    } else {
        print_probe_table(&report);
    }
    let _ = std::io::stdout().flush();

    Ok(if healthy { 0 } else { 2 })
}

fn print_probe_table(r: &ProbeReport) {
    println!("ollama_up            {}", r.ollama_up);
    println!("ollama_models        {}", r.ollama_models.join(", "));
    println!("lmstudio_up          {}", r.lmstudio_up);
    println!("lmstudio_models      {}", r.lmstudio_models.join(", "));
    println!("node_info_up         {}", r.node_info_up);
    match &r.node_info {
        Some(info) => {
            let util = info.gpus.first().map(|g| g.utilization_percent);
            println!("hostUuid             {}", info.host_uuid);
            println!("telemetryValid       {}", info.telemetry_valid);
            match util {
                Some(u) => println!("GPUs[0].utilization_percent {u}"),
                None => println!("GPUs[0].utilization_percent (none)"),
            }
        }
        None => println!("node_info            (unreachable)"),
    }
}

/// `ProbeReport` isn't `Serialize` (it lives in `pair-node`, which this ticket
/// must not edit), so this mirrors its fields for `--json` output.
#[derive(serde::Serialize)]
struct ProbeReportJson {
    ollama_up: bool,
    ollama_models: Vec<String>,
    lmstudio_up: bool,
    lmstudio_models: Vec<String>,
    node_info_up: bool,
    node_info: Option<NodeInfoResponse>,
    durations_ms: ProbeDurationsJson,
}

#[derive(serde::Serialize)]
struct ProbeDurationsJson {
    ollama: u128,
    lmstudio: u128,
    node_info: u128,
}

impl From<&ProbeReport> for ProbeReportJson {
    fn from(r: &ProbeReport) -> Self {
        Self {
            ollama_up: r.ollama_up,
            ollama_models: r.ollama_models.clone(),
            lmstudio_up: r.lmstudio_up,
            lmstudio_models: r.lmstudio_models.clone(),
            node_info_up: r.node_info_up,
            node_info: r.node_info.clone(),
            durations_ms: ProbeDurationsJson {
                ollama: r.durations.ollama.as_millis(),
                lmstudio: r.durations.lmstudio.as_millis(),
                node_info: r.durations.node_info.as_millis(),
            },
        }
    }
}
