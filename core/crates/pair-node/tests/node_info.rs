//! Ticket #9 — server skeleton + `GET /v1/node-info`.
//!
//! The node-info assertions replay PAIR's `probeNodeInfo`
//! (`services/nvpair-manual-nodes/manager.go:493-527`): GET the URL, require
//! 200, decode the body into `NodeInfoResponse`.

mod common;

use common::*;
use pair_telemetry::TelemetrySource;
use std::sync::Arc;
use std::time::Duration;

#[tokio::test]
async fn binds_three_ephemeral_ports_before_start_returns() {
    let (handle, _t) = start_with(FakeEngine::new(two_models())).await;
    let ports = handle.ports();
    assert_ne!(ports.openai, 0, "openai port must be resolved");
    assert_ne!(ports.ollama, 0, "ollama port must be resolved");
    assert_ne!(ports.node_info, 0, "node-info port must be resolved");
    assert_ne!(ports.openai, ports.ollama);
    assert_ne!(ports.ollama, ports.node_info);
    handle.shutdown().await;
}

/// Replays `probeNodeInfo` (manager.go:493): GET → 200 → decode.
#[tokio::test]
async fn probe_node_info_returns_decodable_node_info() {
    let telemetry = Arc::new(FakeTelemetry::new());
    let handle = start_node(FakeEngine::new(two_models()).shared(), telemetry.clone()).await;
    let ports = handle.ports();

    let resp = client().get(node_info_url(ports, "/v1/node-info")).send().await.expect("node-info reachable");
    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers().get(reqwest::header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).map(|v| v
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_string()),
        Some("application/json".to_string())
    );

    let raw: serde_json::Value = resp.json().await.expect("json body");
    // PAIR reads these exact keys (manager.go NodeInfoResponse struct tags).
    assert!(raw.get("GPUs").is_some(), "PAIR reads `GPUs`, got {raw}");
    assert_eq!(raw["telemetryValid"], serde_json::json!(true));
    assert_eq!(raw["msSince"], serde_json::json!(250));
    assert_eq!(raw["hostUuid"], serde_json::json!("8f14e45f-ea3c-4f1e-9b0a-1d2c3b4a5f60"));

    let decoded: pair_protocol::node_info::NodeInfoResponse =
        serde_json::from_value(raw).expect("decodes as NodeInfoResponse");
    assert_eq!(decoded, telemetry.expected());

    handle.shutdown().await;
}

#[tokio::test]
async fn unknown_paths_are_404_on_every_lane() {
    let (handle, _t) = start_with(FakeEngine::new(two_models())).await;
    let ports = handle.ports();
    let c = client();
    for url in [
        node_info_url(ports, "/nope"),
        node_info_url(ports, "/v1/models"),
        openai_url(ports, "/nope"),
        ollama_url(ports, "/nope"),
    ] {
        let resp = c.get(&url).send().await.expect("reachable");
        assert_eq!(resp.status(), 404, "{url} should 404");
    }
    handle.shutdown().await;
}

#[tokio::test]
async fn telemetry_tick_loop_samples_and_receives_inference_load() {
    let telemetry = Arc::new(FakeTelemetry::new().with_interval(Duration::from_millis(10)));
    let handle = start_node(
        FakeEngine::new(two_models()).with_loaded("qwen2.5-1.5b-instruct-q4_k_m").shared(),
        telemetry.clone(),
    )
    .await;

    let t = telemetry.clone();
    assert!(
        wait_until(Duration::from_secs(2), move || t.ticks() >= 2).await,
        "telemetry.tick() must be driven on a sample_interval timer"
    );
    let t = telemetry.clone();
    assert!(
        wait_until(Duration::from_secs(2), move || t.load().loaded_bytes > 0).await,
        "engine.status() must be pushed into telemetry.set_inference_load()"
    );

    handle.shutdown().await;
}

#[tokio::test]
async fn shutdown_releases_the_listeners() {
    let (handle, _t) = start_with(FakeEngine::new(two_models())).await;
    let ports = handle.ports();
    handle.shutdown().await;

    let err =
        client().get(node_info_url(ports, "/v1/node-info")).timeout(Duration::from_secs(2)).send().await;
    assert!(err.is_err(), "node-info listener must be closed after shutdown");
}

#[tokio::test]
async fn telemetry_source_is_shared_not_copied() {
    // Sanity: the trait object we hand to the node is the same one the test holds.
    let telemetry = Arc::new(FakeTelemetry::new());
    let handle = start_node(FakeEngine::new(vec![]).shared(), telemetry.clone()).await;
    telemetry.set_admission(pair_telemetry::Admission::Refuse("test".into()));
    assert!(matches!(TelemetrySource::admission(&*telemetry), pair_telemetry::Admission::Refuse(_)));
    handle.shutdown().await;
}

/// `docs/pair-contract.md` §2.7(b): PAIR only folds telemetry into scheduling
/// while `msSince` is fresh, so the handler must re-read the source per request.
#[tokio::test]
async fn node_info_is_not_cached_between_requests() {
    let telemetry = Arc::new(FakeTelemetry::new());
    let handle = start_node(FakeEngine::new(vec![]).shared(), telemetry.clone()).await;
    let ports = handle.ports();
    let c = client();

    let first: serde_json::Value =
        c.get(node_info_url(ports, "/v1/node-info")).send().await.unwrap().json().await.unwrap();
    assert_eq!(first["msSince"], serde_json::json!(250));

    telemetry.set_ms_since(31);
    let second: serde_json::Value =
        c.get(node_info_url(ports, "/v1/node-info")).send().await.unwrap().json().await.unwrap();
    assert_eq!(second["msSince"], serde_json::json!(31));

    handle.shutdown().await;
}
