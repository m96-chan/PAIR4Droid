//! Ticket #13 — end-to-end: spawn the real `pair4droid` binary and drive it
//! through `serve` + `probe` exactly as PAIR would see it from outside.

use std::io::{BufRead, BufReader, Read};
use std::net::TcpListener;
use std::process::{Child, Command, Stdio};

struct Ports {
    openai: u16,
    ollama: u16,
    node_info: u16,
}

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_pair4droid")
}

/// Spawn `serve --mock alpha,beta` on ephemeral ports and read back the
/// `ports openai=.. ollama=.. node_info=..` line the CLI must print on
/// startup, before any lane is guaranteed reachable.
fn spawn_serve() -> (Child, Ports) {
    let mut child = Command::new(bin())
        .args([
            "serve",
            "--mock",
            "alpha,beta",
            "--bind",
            "127.0.0.1",
            "--openai-port",
            "0",
            "--ollama-port",
            "0",
            "--node-info-port",
            "0",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn pair4droid serve");

    let stdout = child.stdout.take().expect("piped stdout");
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    let ports = loop {
        line.clear();
        let n = reader.read_line(&mut line).expect("read child stdout");
        assert!(n > 0, "child exited before printing the `ports ...` line");
        let Some(rest) = line.trim().strip_prefix("ports ") else { continue };

        let mut openai = None;
        let mut ollama = None;
        let mut node_info = None;
        for kv in rest.split_whitespace() {
            let (k, v) = kv.split_once('=').unwrap_or_else(|| panic!("malformed ports line: {line:?}"));
            let v: u16 = v.parse().unwrap_or_else(|_| panic!("bad port in ports line: {line:?}"));
            match k {
                "openai" => openai = Some(v),
                "ollama" => ollama = Some(v),
                "node_info" => node_info = Some(v),
                _ => {}
            }
        }
        break Ports {
            openai: openai.unwrap_or_else(|| panic!("no openai= in ports line: {line:?}")),
            ollama: ollama.unwrap_or_else(|| panic!("no ollama= in ports line: {line:?}")),
            node_info: node_info.unwrap_or_else(|| panic!("no node_info= in ports line: {line:?}")),
        };
    };

    // Keep draining the child's stdout in the background: it's a line-buffered
    // pipe, and letting it fill up would make the child block on further
    // `println!`s (including its shutdown log line) for the rest of the test.
    std::thread::spawn(move || {
        let mut sink = Vec::new();
        let _ = reader.read_to_end(&mut sink);
    });

    (child, ports)
}

#[test]
fn serve_then_probe_sees_both_lanes_up_with_the_advertised_models() {
    let (mut child, ports) = spawn_serve();

    let output = Command::new(bin())
        .args([
            "probe",
            "127.0.0.1",
            "--openai-port",
            &ports.openai.to_string(),
            "--ollama-port",
            &ports.ollama.to_string(),
            "--node-info-port",
            &ports.node_info.to_string(),
            "--json",
        ])
        .output()
        .expect("run pair4droid probe");

    let _ = child.kill();
    let _ = child.wait();

    assert!(
        output.status.success(),
        "probe should exit 0 when both inference lanes are up with models\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("probe --json output must be valid JSON");

    assert_eq!(report["ollama_up"], serde_json::json!(true));
    assert_eq!(report["lmstudio_up"], serde_json::json!(true));
    assert_eq!(report["node_info_up"], serde_json::json!(true));
    assert_eq!(report["ollama_models"], serde_json::json!(["alpha", "beta"]));
    assert_eq!(report["lmstudio_models"], serde_json::json!(["alpha", "beta"]));
}

#[test]
fn probe_against_closed_ports_exits_2() {
    // Bind then immediately drop three listeners so the OS is free to hand
    // those ports right back out, but nothing is listening on them anymore.
    let free_port = || {
        let l = TcpListener::bind(("127.0.0.1", 0)).expect("bind ephemeral port");
        l.local_addr().unwrap().port()
    };
    let openai = free_port();
    let ollama = free_port();
    let node_info = free_port();

    let output = Command::new(bin())
        .args([
            "probe",
            "127.0.0.1",
            "--openai-port",
            &openai.to_string(),
            "--ollama-port",
            &ollama.to_string(),
            "--node-info-port",
            &node_info.to_string(),
            "--timeout-secs",
            "1",
        ])
        .output()
        .expect("run pair4droid probe");

    assert_eq!(
        output.status.code(),
        Some(2),
        "probe of closed ports must exit 2; stdout: {}",
        String::from_utf8_lossy(&output.stdout)
    );
}
