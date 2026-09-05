//! Ticket #11 — Ollama lane on the `:11434` port.

mod common;

use common::*;
use pair_engine::FinishReason;
use pair_telemetry::Admission;
use serde_json::json;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

const M: &str = "qwen2.5-1.5b-instruct-q4_k_m";

fn ct(resp: &reqwest::Response) -> String {
    resp.headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string()
}

fn ndjson(text: &str) -> Vec<serde_json::Value> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("each line is JSON"))
        .collect()
}

/// `probeOllama` (`services/nvpair-manual-nodes/manager.go:448-471`) requires a
/// bare 200 on `GET /` before it even tries `/api/tags`.
#[tokio::test]
async fn root_is_200_ollama_is_running() {
    let (handle, _t) = start_with(FakeEngine::new(two_models())).await;
    let ports = handle.ports();
    let resp = client().get(ollama_url(ports, "/")).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(ct(&resp), "text/plain; charset=utf-8");
    assert_eq!(resp.text().await.unwrap(), "Ollama is running");
    handle.shutdown().await;
}

/// `fetchOllamaModels` (`manager.go:473-497`) reads `models[].name`; the
/// proxy's `/api/tags` fan-out prefers `models[].model` and rejects records with
/// no identity (`services/ollama-proxy/proxy.go:1062-1074`).
#[tokio::test]
async fn api_tags_carries_both_name_and_model() {
    let (handle, _t) = start_with(FakeEngine::new(two_models())).await;
    let ports = handle.ports();
    let resp = client().get(ollama_url(ports, "/api/tags")).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    assert!(ct(&resp).starts_with("application/json"));
    let body: serde_json::Value = resp.json().await.unwrap();

    let models = body["models"].as_array().unwrap();
    assert_eq!(models.len(), 2);
    assert_eq!(models[0]["name"], json!(M));
    assert_eq!(models[0]["model"], json!(M));
    assert_eq!(models[0]["modified_at"], json!("2026-02-03T04:05:06Z"));
    assert_eq!(models[0]["size"], json!(1_073_741_824u64));
    let digest = models[0]["digest"].as_str().unwrap();
    assert!(digest.starts_with("sha256:"), "digest was {digest}");
    assert_eq!(digest.len(), "sha256:".len() + 64);
    assert_eq!(models[0]["details"]["family"], json!("qwen2"));
    assert_eq!(models[0]["details"]["format"], json!("gguf"));
    assert_eq!(models[0]["details"]["parameter_size"], json!("1.5B"));
    assert_eq!(models[0]["details"]["quantization_level"], json!("Q4_K_M"));

    let typed: pair_protocol::ollama::TagsResponse = serde_json::from_value(body).unwrap();
    assert_eq!(typed.models[1].name, "gemma-2b-it-q4_k_m");
    handle.shutdown().await;
}

#[tokio::test]
async fn api_version_reports_the_configured_version() {
    let (handle, _t) = start_with(FakeEngine::new(two_models())).await;
    let ports = handle.ports();
    let body: serde_json::Value =
        client().get(ollama_url(ports, "/api/version")).send().await.unwrap().json().await.unwrap();
    assert_eq!(body, json!({"version": pair_node::NodeConfig::default().ollama_version}));
    handle.shutdown().await;
}

#[tokio::test]
async fn api_chat_streams_ndjson_by_default() {
    let (handle, _t) = start_with(FakeEngine::new(two_models()).with_tokens(["Hel", "lo"])).await;
    let ports = handle.ports();

    // `stream` absent → Ollama streams.
    let resp = client()
        .post(ollama_url(ports, "/api/chat"))
        .json(&json!({"model": M, "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(ct(&resp), "application/x-ndjson");
    assert!(
        resp.headers().get(reqwest::header::CONTENT_LENGTH).is_none(),
        "chunked, so Go's ReverseProxy flushes per write (pair-contract §3.2)"
    );
    let lines = ndjson(&resp.text().await.unwrap());
    assert_eq!(lines.len(), 3, "2 token lines + final done line: {lines:?}");

    assert_eq!(lines[0]["model"], json!(M));
    assert_eq!(lines[0]["message"], json!({"role": "assistant", "content": "Hel"}));
    assert_eq!(lines[0]["done"], json!(false));
    assert!(lines[0].get("done_reason").is_none());
    assert!(lines[0].get("total_duration").is_none());
    assert!(!lines[0]["created_at"].as_str().unwrap().is_empty());

    assert_eq!(lines[1]["message"]["content"], json!("lo"));

    let last = &lines[2];
    assert_eq!(last["done"], json!(true));
    assert_eq!(last["done_reason"], json!("stop"));
    assert_eq!(last["message"], json!({"role": "assistant", "content": ""}));
    assert!(last["total_duration"].as_u64().unwrap() > 0);
    assert_eq!(last["load_duration"], json!(12_000_000u64));
    assert_eq!(last["prompt_eval_count"], json!(7));
    assert_eq!(last["prompt_eval_duration"], json!(34_000_000u64));
    assert_eq!(last["eval_count"], json!(2));
    assert_eq!(last["eval_duration"], json!(56_000_000u64));

    let typed: pair_protocol::ollama::ChatResponse = serde_json::from_value(last.clone()).unwrap();
    assert!(typed.timings.is_some());

    handle.shutdown().await;
}

#[tokio::test]
async fn api_chat_stream_false_returns_one_object() {
    let (handle, _t) = start_with(FakeEngine::new(two_models()).with_tokens(["Hel", "lo"])).await;
    let ports = handle.ports();
    let resp = client()
        .post(ollama_url(ports, "/api/chat"))
        .json(&json!({"model": M, "messages": [], "stream": false}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert!(ct(&resp).starts_with("application/json"));
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["message"], json!({"role": "assistant", "content": "Hello"}));
    assert_eq!(body["done"], json!(true));
    assert_eq!(body["done_reason"], json!("stop"));
    assert_eq!(body["eval_count"], json!(2));
    handle.shutdown().await;
}

#[tokio::test]
async fn api_chat_done_reason_length() {
    let (handle, _t) =
        start_with(FakeEngine::new(two_models()).with_tokens(["a"]).with_finish(FinishReason::Length)).await;
    let ports = handle.ports();
    let body: serde_json::Value = client()
        .post(ollama_url(ports, "/api/chat"))
        .json(&json!({"model": M, "messages": [], "stream": false}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["done_reason"], json!("length"));
    handle.shutdown().await;
}

#[tokio::test]
async fn api_chat_honours_options() {
    let engine = FakeEngine::new(two_models());
    let seen = engine.last_request.clone();
    let handle = start_node(engine.shared(), Arc::new(FakeTelemetry::new())).await;
    let ports = handle.ports();

    client()
        .post(ollama_url(ports, "/api/chat"))
        .json(&json!({
            "model": M,
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            "stream": false,
            "options": {"num_predict": 12, "temperature": 0.5, "top_p": 0.8,
                        "seed": 7, "stop": ["<eot>"], "num_ctx": 2048},
            "keep_alive": "5m"
        }))
        .send()
        .await
        .unwrap();

    let req = seen.lock().clone().unwrap();
    assert_eq!(req.params.max_tokens, Some(12));
    assert_eq!(req.params.temperature, Some(0.5));
    assert_eq!(req.params.top_p, Some(0.8));
    assert_eq!(req.params.seed, Some(7));
    assert_eq!(req.params.stop, vec!["<eot>".to_string()]);
    assert_eq!(req.messages.len(), 2);
    assert_eq!(req.messages[0].role, pair_engine::ChatRole::System);
    handle.shutdown().await;
}

/// Ollama's `num_predict: -1` means "unlimited"; it must not become a token cap.
#[tokio::test]
async fn negative_num_predict_means_unlimited() {
    let engine = FakeEngine::new(two_models());
    let seen = engine.last_request.clone();
    let handle = start_node(engine.shared(), Arc::new(FakeTelemetry::new())).await;
    let ports = handle.ports();
    client()
        .post(ollama_url(ports, "/api/chat"))
        .json(&json!({"model": M, "messages": [], "stream": false,
                      "options": {"num_predict": -1}}))
        .send()
        .await
        .unwrap();
    assert_eq!(seen.lock().clone().unwrap().params.max_tokens, None);
    handle.shutdown().await;
}

#[tokio::test]
async fn api_generate_uses_the_response_field() {
    let engine = FakeEngine::new(two_models()).with_tokens(["Hel", "lo"]);
    let seen = engine.last_request.clone();
    let handle = start_node(engine.shared(), Arc::new(FakeTelemetry::new())).await;
    let ports = handle.ports();

    let text = client()
        .post(ollama_url(ports, "/api/generate"))
        .json(&json!({"model": M, "prompt": "hi", "system": "be terse"}))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let lines = ndjson(&text);
    assert_eq!(lines.len(), 3);
    assert_eq!(lines[0]["response"], json!("Hel"));
    assert_eq!(lines[0]["done"], json!(false));
    assert!(lines[0].get("message").is_none(), "generate uses `response`, not `message`");
    assert_eq!(lines[2]["done"], json!(true));
    assert_eq!(lines[2]["done_reason"], json!("stop"));
    assert_eq!(lines[2]["response"], json!(""));
    assert_eq!(lines[2]["eval_count"], json!(2));

    let typed: pair_protocol::ollama::GenerateResponse = serde_json::from_value(lines[2].clone()).unwrap();
    assert!(typed.timings.is_some());

    let req = seen.lock().clone().unwrap();
    assert_eq!(req.messages.len(), 2);
    assert_eq!(req.messages[0].role, pair_engine::ChatRole::System);
    assert_eq!(req.messages[0].content, "be terse");
    assert_eq!(req.messages[1].role, pair_engine::ChatRole::User);
    assert_eq!(req.messages[1].content, "hi");

    handle.shutdown().await;
}

#[tokio::test]
async fn api_generate_stream_false() {
    let (handle, _t) = start_with(FakeEngine::new(two_models()).with_tokens(["Hel", "lo"])).await;
    let ports = handle.ports();
    // Body copied from PAIR's secure-inference fixture
    // (`services/tests/secure_inference_test.go:279`).
    let body: serde_json::Value = client()
        .post(ollama_url(ports, "/api/generate"))
        .json(&json!({"model": M, "prompt": "hi", "stream": false}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["model"], json!(M));
    assert_eq!(body["response"], json!("Hello"));
    assert_eq!(body["done"], json!(true));
    handle.shutdown().await;
}

#[tokio::test]
async fn api_show_accepts_model_or_name() {
    let (handle, _t) = start_with(FakeEngine::new(two_models())).await;
    let ports = handle.ports();
    let c = client();
    for payload in [json!({"model": M}), json!({"name": M})] {
        let resp = c.post(ollama_url(ports, "/api/show")).json(&payload).send().await.unwrap();
        assert_eq!(resp.status(), 200, "payload {payload}");
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["details"]["family"], json!("qwen2"));
        assert_eq!(body["details"]["format"], json!("gguf"));
        assert_eq!(body["capabilities"], json!(["completion"]));
        assert_eq!(body["model_info"]["general.architecture"], json!("qwen2"));
        assert_eq!(body["model_info"]["qwen2.context_length"], json!(4096));
        let _typed: pair_protocol::ollama::ShowResponse = serde_json::from_value(body).unwrap();
    }
    let resp = c.post(ollama_url(ports, "/api/show")).json(&json!({"model": "x"})).send().await.unwrap();
    assert_eq!(resp.status(), 404);
    handle.shutdown().await;
}

#[tokio::test]
async fn api_ps_reports_the_loaded_model_only() {
    let (handle, _t) = start_with(FakeEngine::new(two_models())).await;
    let ports = handle.ports();
    let body: serde_json::Value =
        client().get(ollama_url(ports, "/api/ps")).send().await.unwrap().json().await.unwrap();
    assert_eq!(body, json!({"models": []}), "nothing loaded → empty list");
    handle.shutdown().await;

    let (handle, _t) = start_with(FakeEngine::new(two_models()).with_loaded(M)).await;
    let ports = handle.ports();
    let body: serde_json::Value =
        client().get(ollama_url(ports, "/api/ps")).send().await.unwrap().json().await.unwrap();
    let models = body["models"].as_array().unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0]["name"], json!(M));
    assert_eq!(models[0]["model"], json!(M));
    assert_eq!(models[0]["size_vram"], json!(1_073_741_824u64));
    assert!(!models[0]["expires_at"].as_str().unwrap().is_empty());
    let _typed: pair_protocol::ollama::PsResponse = serde_json::from_value(body).unwrap();
    handle.shutdown().await;
}

/// PAIR's Ollama lane normalises a tagless reference to `<name>:latest`
/// (`services/ollama-proxy/proxy.go:967-974`), so both spellings must resolve
/// to the same catalogue entry — including for `/api/show`.
#[tokio::test]
async fn latest_tag_is_normalised_on_the_ollama_lane() {
    let (handle, _t) = start_with(FakeEngine::new(two_models())).await;
    let ports = handle.ports();
    let c = client();
    let tagged = format!("{M}:latest");

    for name in [M.to_string(), tagged.clone()] {
        let resp = c
            .post(ollama_url(ports, "/api/chat"))
            .json(&json!({"model": name, "messages": [], "stream": false}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "model {name} must resolve");
        let resp = c
            .post(ollama_url(ports, "/api/generate"))
            .json(&json!({"model": name, "prompt": "hi", "stream": false}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "generate {name} must resolve");
        let resp = c.post(ollama_url(ports, "/api/show")).json(&json!({"model": name})).send().await.unwrap();
        assert_eq!(resp.status(), 200, "show {name} must resolve");
    }

    // The OpenAI lane stays exact (`services/lmstudio-proxy/proxy.go:1528-1538`).
    let resp = c
        .post(openai_url(ports, "/v1/chat/completions"))
        .json(&json!({"model": tagged, "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404, "OpenAI lane must not normalise :latest");

    handle.shutdown().await;
}

#[tokio::test]
async fn unknown_model_is_404_with_the_ollama_envelope() {
    let (handle, _t) = start_with(FakeEngine::new(two_models())).await;
    let ports = handle.ports();
    let c = client();
    for path in ["/api/chat", "/api/generate"] {
        let resp = c
            .post(ollama_url(ports, path))
            .json(&json!({"model": "x", "messages": [], "prompt": "hi"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404, "{path}");
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body, json!({"error": "model 'x' not found"}));
    }
    handle.shutdown().await;
}

#[tokio::test]
async fn admission_refused_and_bad_json_and_engine_errors() {
    let (handle, telemetry) = start_with(FakeEngine::new(two_models())).await;
    let ports = handle.ports();
    let c = client();

    telemetry.set_admission(Admission::Refuse("battery too low".into()));
    let resp = c
        .post(ollama_url(ports, "/api/chat"))
        .json(&json!({"model": M, "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 503);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body, json!({"error": "battery too low"}));
    telemetry.set_admission(Admission::Accept);

    let resp = c
        .post(ollama_url(ports, "/api/chat"))
        .header("content-type", "application/json")
        .body("{\"model\":")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    assert!(resp.json::<serde_json::Value>().await.unwrap()["error"].is_string());
    handle.shutdown().await;

    for (err, status) in
        [(FakeError::Busy, 503), (FakeError::ContextExceeded, 400), (FakeError::Generation, 500)]
    {
        let (handle, _t) = start_with(FakeEngine::new(two_models()).with_error(err)).await;
        let ports = handle.ports();
        let resp = client()
            .post(ollama_url(ports, "/api/chat"))
            .json(&json!({"model": M, "messages": []}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), status, "{err:?}");
        handle.shutdown().await;
    }
}

#[tokio::test]
async fn ollama_client_disconnect_cancels_the_engine_stream() {
    let engine = FakeEngine::new(two_models())
        .with_tokens(["a", "b", "c", "d", "e", "f", "g", "h"])
        .with_delay(Duration::from_millis(80));
    let active = engine.active_handle();
    let handle = start_node(engine.shared(), Arc::new(FakeTelemetry::new())).await;
    let ports = handle.ports();

    let mut resp = client()
        .post(ollama_url(ports, "/api/chat"))
        .json(&json!({"model": M, "messages": [], "stream": true}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let chunk = resp.chunk().await.unwrap().expect("first NDJSON line");
    assert!(chunk.ends_with(b"\n"));
    assert_eq!(active.load(Ordering::SeqCst), 1);

    drop(resp);
    let a = active.clone();
    assert!(
        wait_until(Duration::from_secs(1), move || a.load(Ordering::SeqCst) == 0).await,
        "engine stream must be dropped when the client vanishes"
    );
    handle.shutdown().await;
}

/// Same flush requirement as the OpenAI lane: NDJSON with no `Content-Length`
/// is flushed per write by Go's `ReverseProxy` (`docs/pair-contract.md` §3.2).
#[tokio::test]
async fn ndjson_lines_are_flushed_as_they_are_produced() {
    let engine = FakeEngine::new(two_models())
        .with_tokens(["a", "b", "c", "d", "e", "f"])
        .with_delay(Duration::from_millis(100));
    let handle = start_node(engine.shared(), Arc::new(FakeTelemetry::new())).await;
    let ports = handle.ports();

    let started = std::time::Instant::now();
    let mut resp = client()
        .post(ollama_url(ports, "/api/chat"))
        .json(&json!({"model": M, "messages": []}))
        .send()
        .await
        .unwrap();
    let first = resp.chunk().await.unwrap().unwrap();
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_millis(350),
        "first line took {elapsed:?}; the generation needs ~600ms, so the body is buffered"
    );
    let line: serde_json::Value =
        serde_json::from_slice(first.strip_suffix(b"\n").unwrap_or(&first)).unwrap();
    assert_eq!(line["message"]["content"], json!("a"));

    handle.shutdown().await;
}
