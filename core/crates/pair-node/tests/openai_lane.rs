//! Ticket #10 — OpenAI lane on the `:1234` port (what PAIR's `lmstudio-proxy`
//! forwards to and what `probeLMStudio` reads).

mod common;

use common::*;
use pair_engine::FinishReason;
use pair_telemetry::Admission;
use serde_json::json;
use std::sync::atomic::Ordering;
use std::time::Duration;

const M: &str = "qwen2.5-1.5b-instruct-q4_k_m";

fn ct(resp: &reqwest::Response) -> String {
    resp.headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string()
}

/// Replays `probeLMStudio` (`services/nvpair-manual-nodes/manager.go:409-446`):
/// one `GET /v1/models` is both the liveness check and the inventory; PAIR keeps
/// non-empty `data[].id` values.
#[tokio::test]
async fn v1_models_matches_what_probe_lmstudio_reads() {
    let (handle, _t) = start_with(FakeEngine::new(two_models())).await;
    let ports = handle.ports();

    let resp = client().get(openai_url(ports, "/v1/models")).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    assert!(ct(&resp).starts_with("application/json"));
    let body: serde_json::Value = resp.json().await.unwrap();

    assert_eq!(body["object"], json!("list"));
    let data = body["data"].as_array().expect("data[]");
    assert_eq!(data.len(), 2);
    let ids: Vec<&str> = data.iter().map(|m| m["id"].as_str().unwrap()).collect();
    assert_eq!(ids, vec![M, "gemma-2b-it-q4_k_m"]);
    for m in data {
        assert_eq!(m["object"], json!("model"));
        assert_eq!(m["owned_by"], json!("pair4droid"));
    }
    let expected_created = chrono::DateTime::parse_from_rfc3339("2026-02-03T04:05:06Z").unwrap().timestamp();
    assert_eq!(data[0]["created"], json!(expected_created));

    // And it decodes into the wire type.
    let typed: pair_protocol::openai::ModelList = serde_json::from_value(body).unwrap();
    assert_eq!(typed.data[0].id, M);

    handle.shutdown().await;
}

#[tokio::test]
async fn non_streaming_chat_completion() {
    let (handle, _t) = start_with(FakeEngine::new(two_models()).with_tokens(["Hel", "lo ", "world"])).await;
    let ports = handle.ports();

    let resp = client()
        .post(openai_url(ports, "/v1/chat/completions"))
        .json(&json!({"model": M, "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert!(ct(&resp).starts_with("application/json"));
    let body: serde_json::Value = resp.json().await.unwrap();

    let id = body["id"].as_str().unwrap();
    assert!(id.starts_with("chatcmpl-"), "id was {id}");
    assert_eq!(id.len(), "chatcmpl-".len() + 32, "uuid simple form, got {id}");
    assert_eq!(body["object"], json!("chat.completion"));
    assert_eq!(body["model"], json!(M));
    assert_eq!(body["choices"][0]["index"], json!(0));
    assert_eq!(body["choices"][0]["message"]["role"], json!("assistant"));
    assert_eq!(body["choices"][0]["message"]["content"], json!("Hello world"));
    assert_eq!(body["choices"][0]["finish_reason"], json!("stop"));
    assert_eq!(body["usage"]["prompt_tokens"], json!(7));
    assert_eq!(body["usage"]["completion_tokens"], json!(3));
    assert_eq!(body["usage"]["total_tokens"], json!(10));

    let _typed: pair_protocol::openai::ChatCompletionResponse =
        serde_json::from_value(body).expect("decodes as ChatCompletionResponse");

    handle.shutdown().await;
}

#[tokio::test]
async fn finish_reason_length_is_reported() {
    let (handle, _t) =
        start_with(FakeEngine::new(two_models()).with_tokens(["a"]).with_finish(FinishReason::Length)).await;
    let ports = handle.ports();
    let body: serde_json::Value = client()
        .post(openai_url(ports, "/v1/chat/completions"))
        .json(&json!({"model": M, "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["choices"][0]["finish_reason"], json!("length"));
    handle.shutdown().await;
}

fn sse_events(body: &str) -> Vec<String> {
    body.split("\n\n")
        .filter(|b| !b.trim().is_empty())
        .map(|b| b.strip_prefix("data: ").unwrap_or(b).trim().to_string())
        .collect()
}

#[tokio::test]
async fn streaming_chat_completion_is_sse() {
    let (handle, _t) = start_with(FakeEngine::new(two_models()).with_tokens(["Hel", "lo"])).await;
    let ports = handle.ports();

    let resp = client()
        .post(openai_url(ports, "/v1/chat/completions"))
        .json(&json!({"model": M, "messages": [{"role": "user", "content": "hi"}], "stream": true}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert!(ct(&resp).starts_with("text/event-stream"), "content-type was {}", ct(&resp));
    assert_eq!(resp.headers().get(reqwest::header::CACHE_CONTROL).unwrap().to_str().unwrap(), "no-cache");
    assert!(
        resp.headers().get(reqwest::header::CONTENT_LENGTH).is_none(),
        "a Content-Length defeats Go ReverseProxy's per-write flush (pair-contract §3.1)"
    );
    let text = resp.text().await.unwrap();
    let events = sse_events(&text);

    assert_eq!(*events.last().unwrap(), "[DONE]", "stream must end with data: [DONE]");
    let chunks: Vec<serde_json::Value> =
        events[..events.len() - 1].iter().map(|e| serde_json::from_str(e).expect("chunk is JSON")).collect();
    assert_eq!(chunks.len(), 4, "role chunk + 2 tokens + finish chunk, got {chunks:?}");

    assert_eq!(chunks[0]["object"], json!("chat.completion.chunk"));
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], json!("assistant"));
    assert_eq!(chunks[0]["choices"][0]["delta"]["content"], json!(""));
    assert_eq!(chunks[0]["choices"][0]["finish_reason"], json!(null));

    assert_eq!(chunks[1]["choices"][0]["delta"]["content"], json!("Hel"));
    assert!(chunks[1]["choices"][0]["delta"].get("role").is_none());
    assert_eq!(chunks[2]["choices"][0]["delta"]["content"], json!("lo"));

    assert_eq!(chunks[3]["choices"][0]["finish_reason"], json!("stop"));
    assert_eq!(chunks[3]["choices"][0]["delta"], json!({}));

    // One id for the whole stream.
    let id = chunks[0]["id"].as_str().unwrap();
    assert!(id.starts_with("chatcmpl-"));
    for c in &chunks {
        assert_eq!(c["id"], json!(id));
        assert_eq!(c["model"], json!(M));
        assert!(c.get("usage").is_none(), "usage only with include_usage");
    }

    for c in &chunks {
        let _typed: pair_protocol::openai::ChatCompletionChunk =
            serde_json::from_value(c.clone()).expect("decodes as ChatCompletionChunk");
    }

    handle.shutdown().await;
}

#[tokio::test]
async fn streaming_include_usage_appends_a_usage_only_chunk() {
    let (handle, _t) = start_with(FakeEngine::new(two_models()).with_tokens(["a", "b"])).await;
    let ports = handle.ports();

    let text = client()
        .post(openai_url(ports, "/v1/chat/completions"))
        .json(&json!({
            "model": M,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "stream_options": {"include_usage": true}
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    let events = sse_events(&text);
    assert_eq!(*events.last().unwrap(), "[DONE]");
    let usage_chunk: serde_json::Value = serde_json::from_str(&events[events.len() - 2]).unwrap();
    assert_eq!(usage_chunk["choices"], json!([]));
    assert_eq!(usage_chunk["usage"]["prompt_tokens"], json!(7));
    assert_eq!(usage_chunk["usage"]["completion_tokens"], json!(2));
    assert_eq!(usage_chunk["usage"]["total_tokens"], json!(9));

    handle.shutdown().await;
}

/// PAIR fails over to the next owner on a 404 for a POST to an inference path
/// (`services/lmstudio-proxy/proxy.go:1015-1016`).
#[tokio::test]
async fn unknown_model_is_404_with_the_openai_envelope() {
    let (handle, _t) = start_with(FakeEngine::new(two_models())).await;
    let ports = handle.ports();
    let resp = client()
        .post(openai_url(ports, "/v1/chat/completions"))
        .json(&json!({"model": "x", "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        body,
        json!({"error": {
            "message": "model 'x' not found",
            "type": "invalid_request_error",
            "code": "model_not_found"
        }})
    );
    handle.shutdown().await;
}

#[tokio::test]
async fn admission_refused_is_503_with_the_reason() {
    let (handle, telemetry) = start_with(FakeEngine::new(two_models())).await;
    let ports = handle.ports();
    telemetry.set_admission(Admission::Refuse("device is too hot".into()));

    let resp = client()
        .post(openai_url(ports, "/v1/chat/completions"))
        .json(&json!({"model": M, "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 503);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["message"], json!("device is too hot"));
    handle.shutdown().await;
}

#[tokio::test]
async fn malformed_json_is_400() {
    let (handle, _t) = start_with(FakeEngine::new(two_models())).await;
    let ports = handle.ports();
    let resp = client()
        .post(openai_url(ports, "/v1/chat/completions"))
        .header("content-type", "application/json")
        .body("{\"model\": ")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["type"], json!("invalid_request_error"));
    handle.shutdown().await;
}

#[tokio::test]
async fn engine_errors_map_to_status_codes() {
    for (err, status) in [
        (FakeError::Busy, 503),
        (FakeError::ContextExceeded, 400),
        (FakeError::Generation, 500),
        (FakeError::LoadFailed, 500),
    ] {
        let (handle, _t) = start_with(FakeEngine::new(two_models()).with_error(err)).await;
        let ports = handle.ports();
        let resp = client()
            .post(openai_url(ports, "/v1/chat/completions"))
            .json(&json!({"model": M, "messages": []}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), status, "{err:?} should map to {status}");
        handle.shutdown().await;
    }
}

#[tokio::test]
async fn request_parameters_are_mapped_onto_the_engine_request() {
    let engine = FakeEngine::new(two_models());
    let seen = engine.last_request.clone();
    let telemetry = std::sync::Arc::new(FakeTelemetry::new());
    let handle = start_node(engine.shared(), telemetry).await;
    let ports = handle.ports();

    client()
        .post(openai_url(ports, "/v1/chat/completions"))
        .json(&json!({
            "model": M,
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": [
                    {"type": "text", "text": "part one "},
                    {"type": "image_url", "image_url": {"url": "http://x"}},
                    {"type": "text", "text": "part two"}
                ]}
            ],
            "max_completion_tokens": 5,
            "temperature": 0.25,
            "top_p": 0.9,
            "stop": ["</s>"],
            "seed": 42,
            "tools": [],
            "response_format": {"type": "text"}
        }))
        .send()
        .await
        .unwrap();

    let req = seen.lock().clone().expect("engine received a request");
    assert_eq!(req.model, M);
    assert_eq!(req.messages.len(), 2);
    assert_eq!(req.messages[0].role, pair_engine::ChatRole::System);
    assert_eq!(req.messages[0].content, "be terse");
    assert_eq!(req.messages[1].role, pair_engine::ChatRole::User);
    assert_eq!(req.messages[1].content, "part one part two");
    assert_eq!(req.params.max_tokens, Some(5));
    assert_eq!(req.params.temperature, Some(0.25));
    assert_eq!(req.params.top_p, Some(0.9));
    assert_eq!(req.params.stop, vec!["</s>".to_string()]);
    assert_eq!(req.params.seed, Some(42));

    handle.shutdown().await;
}

#[tokio::test]
async fn max_tokens_is_accepted_as_a_fallback_for_max_completion_tokens() {
    let engine = FakeEngine::new(two_models());
    let seen = engine.last_request.clone();
    let handle = start_node(engine.shared(), std::sync::Arc::new(FakeTelemetry::new())).await;
    let ports = handle.ports();
    client()
        .post(openai_url(ports, "/v1/chat/completions"))
        .json(&json!({"model": M, "messages": [], "max_tokens": 3}))
        .send()
        .await
        .unwrap();
    assert_eq!(seen.lock().clone().unwrap().params.max_tokens, Some(3));
    handle.shutdown().await;
}

/// CLAUDE.md design invariant: "dropping the HTTP response must cancel the
/// engine stream" — this is what saves the phone's battery.
#[tokio::test]
async fn client_disconnect_cancels_the_engine_stream() {
    let engine = FakeEngine::new(two_models())
        .with_tokens(["a", "b", "c", "d", "e", "f", "g", "h"])
        .with_delay(Duration::from_millis(80));
    let active = engine.active_handle();
    let handle = start_node(engine.shared(), std::sync::Arc::new(FakeTelemetry::new())).await;
    let ports = handle.ports();

    let mut resp = client()
        .post(openai_url(ports, "/v1/chat/completions"))
        .json(&json!({"model": M, "messages": [], "stream": true}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let chunk = resp.chunk().await.unwrap().expect("first SSE chunk");
    assert!(!chunk.is_empty());
    assert_eq!(active.load(Ordering::SeqCst), 1, "a generation is in flight");

    drop(resp);

    let a = active.clone();
    assert!(
        wait_until(Duration::from_secs(1), move || a.load(Ordering::SeqCst) == 0).await,
        "engine stream must be dropped within ~1s of the client vanishing"
    );

    handle.shutdown().await;
}

#[tokio::test]
async fn phase1_unimplemented_openai_routes_are_404() {
    let (handle, _t) = start_with(FakeEngine::new(two_models())).await;
    let ports = handle.ports();
    let c = client();
    for path in ["/v1/completions", "/v1/embeddings"] {
        let resp = c.post(openai_url(ports, path)).json(&json!({"model": M})).send().await.unwrap();
        assert_eq!(resp.status(), 404, "{path}");
    }
    handle.shutdown().await;
}

/// PAIR's proxy relies on Go's `ReverseProxy` flush heuristic, which streams
/// per write for `text/event-stream` (`docs/pair-contract.md` §3.1) — so our
/// chunks must leave the server as they are produced, not at the end.
#[tokio::test]
async fn sse_chunks_are_flushed_as_they_are_produced() {
    let engine = FakeEngine::new(two_models())
        .with_tokens(["a", "b", "c", "d", "e", "f"])
        .with_delay(Duration::from_millis(100));
    let handle = start_node(engine.shared(), std::sync::Arc::new(FakeTelemetry::new())).await;
    let ports = handle.ports();

    let started = std::time::Instant::now();
    let mut resp = client()
        .post(openai_url(ports, "/v1/chat/completions"))
        .json(&json!({"model": M, "messages": [], "stream": true}))
        .send()
        .await
        .unwrap();
    let first = resp.chunk().await.unwrap().unwrap();
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_millis(300),
        "first chunk took {elapsed:?}; the whole generation needs ~600ms, so the body is buffered"
    );
    assert!(String::from_utf8_lossy(&first).starts_with("data: "));

    handle.shutdown().await;
}
