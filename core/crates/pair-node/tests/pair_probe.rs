//! Ticket #12 — conformance: our `probe` module replays PAIR's manual-node
//! probe (`services/nvpair-manual-nodes/manager.go:250-281`) against a running
//! `Node`, and the lanes obey PAIR's 404-failover semantics.

mod common;

use common::*;
use pair_node::probe::{probe, ProbeReport, DEFAULT_TIMEOUT};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;

const M: &str = "qwen2.5-1.5b-instruct-q4_k_m";

async fn report(handle: &pair_node::NodeHandle) -> ProbeReport {
    probe(LOCALHOST, handle.ports(), DEFAULT_TIMEOUT).await
}

#[tokio::test]
async fn probe_sees_all_three_lanes_up_with_both_model_lists() {
    let telemetry = Arc::new(FakeTelemetry::new());
    let handle = start_node(FakeEngine::new(two_models()).shared(), telemetry.clone()).await;

    let r = report(&handle).await;

    assert!(r.ollama_up, "GET :ollama/ must be 200 (manager.go:456-464)");
    assert!(r.lmstudio_up, "GET :openai/v1/models must be 200 (manager.go:419-427)");
    assert!(r.node_info_up, "GET :node_info/v1/node-info must be 200 and decode");

    let expected = vec![M.to_string(), "gemma-2b-it-q4_k_m".to_string()];
    assert_eq!(r.ollama_models, expected, "from /api/tags models[].name");
    assert_eq!(r.lmstudio_models, expected, "from /v1/models data[].id");
    assert_eq!(r.node_info.as_ref().unwrap(), &telemetry.expected());

    assert!(r.durations.ollama > Duration::ZERO);
    assert!(r.durations.lmstudio > Duration::ZERO);
    assert!(r.durations.node_info > Duration::ZERO);

    // `reachable = ollama_up || lmstudio_up || node_info_up` (manager.go:305).
    assert!(r.reachable());

    handle.shutdown().await;
}

#[tokio::test]
async fn probe_reports_an_empty_catalogue_without_marking_the_lane_down() {
    let handle = start_node(FakeEngine::new(vec![]).shared(), Arc::new(FakeTelemetry::new())).await;
    let r = report(&handle).await;
    assert!(r.ollama_up && r.lmstudio_up && r.node_info_up);
    assert!(r.ollama_models.is_empty());
    assert!(r.lmstudio_models.is_empty());
    handle.shutdown().await;
}

#[tokio::test]
async fn probe_of_a_dead_node_reports_everything_down() {
    let handle = start_node(FakeEngine::new(vec![]).shared(), Arc::new(FakeTelemetry::new())).await;
    let ports = handle.ports();
    handle.shutdown().await;

    let r = probe(LOCALHOST, ports, Duration::from_millis(500)).await;
    assert!(!r.ollama_up);
    assert!(!r.lmstudio_up);
    assert!(!r.node_info_up);
    assert!(r.node_info.is_none());
    assert!(!r.reachable());
}

/// PAIR's own default: `probeTimeout = 3 * time.Second`
/// (`services/nvpair-manual-nodes/manager.go:30`).
#[test]
fn default_timeout_matches_pairs_probe_timeout() {
    assert_eq!(DEFAULT_TIMEOUT, Duration::from_secs(3));
}

/// Mirrors the owner-404 fake in `services/tests/model_routing_interop_test.go:23-39`:
/// an owner that does not have the model answers 404, which is what makes
/// `shouldRetry` fail over to the next owner
/// (`services/lmstudio-proxy/proxy.go:1015-1016`, `services/ollama-proxy/proxy.go:1210-1211`).
#[tokio::test]
async fn unknown_model_is_404_on_both_inference_lanes() {
    let handle = start_node(FakeEngine::new(two_models()).shared(), Arc::new(FakeTelemetry::new())).await;
    let ports = handle.ports();
    let c = client();

    // Request bodies copied from model_routing_interop_test.go:117-119.
    let openai = c
        .post(openai_url(ports, "/v1/chat/completions"))
        .json(&json!({"model": "strict-routing-lmstudio", "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(openai.status(), 404);
    assert_eq!(openai.json::<serde_json::Value>().await.unwrap()["error"]["code"], json!("model_not_found"));

    let ollama = c
        .post(ollama_url(ports, "/api/chat"))
        .json(&json!({"model": "strict-routing-ollama", "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(ollama.status(), 404);
    assert_eq!(
        ollama.json::<serde_json::Value>().await.unwrap(),
        json!({"error": "model 'strict-routing-ollama' not found"})
    );

    // An advertised model must still be served, so failover lands somewhere.
    let ok = c
        .post(ollama_url(ports, "/api/chat"))
        .json(&json!({"model": M, "messages": [], "stream": false}))
        .send()
        .await
        .unwrap();
    assert_eq!(ok.status(), 200);

    handle.shutdown().await;
}

/// `GET /api/version` returning 404 is passed straight through by the proxy
/// (`services/ollama-proxy/failover_test.go:358-363`) — a non-inference route
/// must therefore not be 404 on a healthy node.
#[tokio::test]
async fn non_inference_ollama_routes_are_served() {
    let handle = start_node(FakeEngine::new(two_models()).shared(), Arc::new(FakeTelemetry::new())).await;
    let ports = handle.ports();
    let c = client();
    for path in ["/", "/api/tags", "/api/version", "/api/ps"] {
        assert_eq!(c.get(ollama_url(ports, path)).send().await.unwrap().status(), 200, "{path}");
    }
    handle.shutdown().await;
}
