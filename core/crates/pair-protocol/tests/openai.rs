//! Ticket #4 — OpenAI-compatible lane on `:1234` (`lmstudio-proxy` talks to it).
//!
//! Fixtures, all copied verbatim from PAIR:
//!
//! | fixture | PAIR source |
//! | --- | --- |
//! | `models_broker_management.json` | `services/tests/broker_management_test.go:66` |
//! | `models_fanout_a.json` | `services/lmstudio-proxy/failover_test.go:382` |
//! | `models_fanout_b.json` | `services/lmstudio-proxy/failover_test.go:384` |
//! | `models_empty.json` | `services/lmstudio-proxy/failover_test.go:455`, `:469`, `:476` |
//! | `models_null_data.json` | `services/lmstudio-proxy/failover_test.go:387` |
//! | `models_ids_only.json` | `services/nvpair-engine-manager/models_test.go:31` |
//! | `fakeengine_models.json` | `services/nvpair-engine-manager/testdata/fakeengine/main.go:247-253` |
//! | `fakeengine_chat_completion.json` | `services/nvpair-engine-manager/testdata/fakeengine/main.go:288-293` |
//! | `chat_completion_routing_interop.json` | `services/tests/model_routing_interop_test.go:32` |
//! | `chunk_partial_tokens.json` | `services/lmstudio-proxy/zombie_test.go:76` |
//! | `chunk_first.json` | `services/lmstudio-proxy/zombie_test.go:124` |
//! | `chunk_empty_choices.json` | `services/lmstudio-proxy/activity_test.go:23`, `services/tests/scheduler_interop_test.go:568` |
//! | `request_model_only.json` | `services/lmstudio-proxy/failover_test.go:147` |
//! | `request_messages_empty.json` | `services/tests/workload_identity_interop_test.go:107` |
//! | `request_strict_routing.json` | `services/tests/model_routing_interop_test.go:119` |
//! | `error_plain_model_not_found.json` | `services/tests/model_routing_interop_test.go:35` |
//! | `error_plain_no_owner.json` | `services/lmstudio-proxy/proxy.go:985` |
//! | `error_plain_upstream.json` | `services/lmstudio-proxy/proxy.go:1242` |
//!
//! The `fakeengine_*` fixtures are the fake engine's own `json.Marshal` output
//! replayed for a request naming `qwen2.5-7b-instruct` (the model id PAIR's
//! `services/tests/broker_management_test.go:66` fake advertises) — the fake
//! echoes whichever model the caller asked for.
//!
//! PAIR has **no** SSE fixture anywhere in the tree — `lmstudio-proxy` never
//! parses `data:` lines (`docs/pair-contract.md` §3.1), it relays bytes. The SSE
//! framing tests below are therefore against the OpenAI wire format itself.

mod common;

use common::*;
use pair_protocol::openai::{
    sse::{self, SseEvent},
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Content, ContentPart, Delta,
    ErrorResponse, Model, ModelList, Role, Usage,
};
use serde_json::json;

// ---------------------------------------------------------------------------
// GET /v1/models
// ---------------------------------------------------------------------------

#[test]
fn model_list_fixtures_decode() {
    let m: ModelList = decode("openai/models_broker_management.json");
    assert_eq!(m.object, "list");
    assert_eq!(m.data.len(), 1);
    assert_eq!(m.data[0].id, "qwen2.5-7b-instruct");
    assert_eq!(m.data[0].object, "model");

    // `owned_by` present, `object` absent.
    let a: ModelList = decode("openai/models_fanout_a.json");
    let ids: Vec<&str> = a.data.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(ids, ["a", "shared"]);
    assert_eq!(a.data[1].owned_by, "first");

    let b: ModelList = decode("openai/models_fanout_b.json");
    assert_eq!(b.data[1].id, "c");

    // Only `id` — `services/nvpair-engine-manager/models_test.go:31`.
    let ids_only: ModelList = decode("openai/models_ids_only.json");
    assert_eq!(ids_only.data.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(), ["phi-3", "gemma-2b"]);

    let empty: ModelList = decode("openai/models_empty.json");
    assert!(empty.data.is_empty());
}

/// `{"object":"list","data":null}` is what a malformed fan-out member returns
/// (`services/lmstudio-proxy/failover_test.go:387`). Must decode, not error.
#[test]
fn null_data_decodes_as_empty_list() {
    let m: ModelList = decode("openai/models_null_data.json");
    assert!(m.data.is_empty());
}

#[test]
fn model_list_round_trips() {
    // `object` + `data[].id` + `data[].object` present; we additionally always
    // emit `created` and `owned_by`, which real LM Studio also sends.
    assert_roundtrip_superset::<ModelList>("openai/models_broker_management.json", &["created", "owned_by"]);
    assert_roundtrip_superset::<ModelList>("openai/fakeengine_models.json", &["created", "owned_by"]);
    assert_roundtrip_superset::<ModelList>("openai/models_fanout_a.json", &["created", "object"]);
    assert_roundtrip_exact::<ModelList>("openai/models_empty.json");
}

/// PAIR reads exactly `data[].id` and nothing else
/// (`services/nvpair-manual-nodes/manager.go:437-442`).
#[test]
fn model_list_we_emit_is_shaped_for_pair() {
    let list = ModelList::from_ids(["qwen2.5-7b-instruct", "llama3.2:1b"]);
    let v = serde_json::to_value(&list).unwrap();
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"][0]["id"], "qwen2.5-7b-instruct");
    assert_eq!(v["data"][0]["object"], "model");
    assert_eq!(v["data"][1]["id"], "llama3.2:1b");
    assert_eq!(keys_of(&Model::new("x")), ["created", "id", "object", "owned_by"]);
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions — request
// ---------------------------------------------------------------------------

#[test]
fn request_fixtures_decode() {
    let r: ChatCompletionRequest = decode("openai/request_model_only.json");
    assert_eq!(r.model, "llama");
    assert!(r.messages.is_empty());
    assert!(!r.stream, "OpenAI defaults stream to false");

    let r: ChatCompletionRequest = decode("openai/request_messages_empty.json");
    assert_eq!(r.model, "crossengine-model");
    assert!(r.messages.is_empty());

    let r: ChatCompletionRequest = decode("openai/request_strict_routing.json");
    assert_eq!(r.model, "strict-routing-lmstudio");
}

#[test]
fn content_is_a_string_or_an_array_of_parts() {
    let body = json!({
        "model": "m",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": [
                {"type": "text", "text": "hello "},
                {"type": "text", "text": "world"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
            ]},
            {"role": "assistant", "content": null}
        ]
    });
    let r: ChatCompletionRequest = serde_json::from_value(body).expect("mixed content forms");
    assert_eq!(r.messages[0].role, Role::System);
    assert_eq!(r.messages[0].content, Some(Content::Text("be brief".into())));

    let Some(Content::Parts(parts)) = &r.messages[1].content else {
        panic!("expected parts, got {:?}", r.messages[1].content);
    };
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0], ContentPart::Text { text: "hello ".into() });
    assert_eq!(parts[2], ContentPart::Unsupported, "no vision in phase 1");

    // Both forms flatten to the text the engine gets.
    assert_eq!(r.messages[0].text(), "be brief");
    assert_eq!(r.messages[1].text(), "hello world");
    assert_eq!(r.messages[2].text(), "");
    assert_eq!(r.last_user_text().as_deref(), Some("hello world"));
}

/// PAIR forwards whatever the client sent, so our parser must never 400 on a
/// field it has not heard of (`docs/pair-contract.md` §3.1: the body is not
/// rewritten, only the top-level `model` is read).
#[test]
fn unknown_request_fields_are_ignored() {
    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi", "extra": 1}],
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "tool_choice": "auto",
        "response_format": {"type": "json_object"},
        "n": 1,
        "logprobs": true,
        "top_logprobs": 5,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "logit_bias": {"1": 2},
        "user": "someone",
        "parallel_tool_calls": false
    });
    let r: ChatCompletionRequest = serde_json::from_value(body).expect("unknown fields ignored");
    assert_eq!(r.model, "m");
    assert_eq!(r.messages.len(), 1);
}

#[test]
fn sampling_options_decode() {
    let body = json!({
        "model": "m", "messages": [], "stream": true,
        "max_tokens": 128, "max_completion_tokens": 256,
        "temperature": 0.2, "top_p": 0.9, "seed": 7,
        "stop": ["\n\n"], "stream_options": {"include_usage": true}
    });
    let r: ChatCompletionRequest = serde_json::from_value(body).unwrap();
    assert!(r.stream);
    assert_eq!(r.max_tokens, Some(128));
    assert_eq!(r.max_completion_tokens, Some(256));
    assert_eq!(r.temperature, Some(0.2));
    assert_eq!(r.top_p, Some(0.9));
    assert_eq!(r.seed, Some(7));
    assert_eq!(r.stop.as_deref(), Some(["\n\n".to_string()].as_slice()));
    assert!(r.stream_options.unwrap().include_usage);
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions — non-streaming response
// ---------------------------------------------------------------------------

#[test]
fn chat_completion_response_fixture_decodes() {
    let r: ChatCompletionResponse = decode("openai/fakeengine_chat_completion.json");
    assert_eq!(r.object, "chat.completion");
    assert_eq!(r.model, "qwen2.5-7b-instruct");
    assert_eq!(r.choices.len(), 1);
    assert_eq!(r.choices[0].index, 0);
    assert_eq!(r.choices[0].message.role, Role::Assistant);
    assert_eq!(r.choices[0].message.text(), "ok");
    assert_eq!(r.choices[0].finish_reason.as_deref(), Some("stop"));
}

/// `{"done":true,"choices":[]}` — the routing-interop upstream
/// (`services/tests/model_routing_interop_test.go:32`) answers both lanes with
/// one body, so `done` is an unknown field on the OpenAI side.
#[test]
fn routing_interop_body_decodes_on_the_openai_lane() {
    let r: ChatCompletionResponse = decode("openai/chat_completion_routing_interop.json");
    assert!(r.choices.is_empty());
}

#[test]
fn we_emit_a_complete_chat_completion() {
    let r = ChatCompletionResponse::assistant("chatcmpl-1", "m", 1_700_000_000, "hello", "stop")
        .with_usage(Usage { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 });
    assert_eq!(
        serde_json::to_value(&r).unwrap(),
        json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1_700_000_000i64,
            "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
        })
    );
}

// ---------------------------------------------------------------------------
// Streaming chunks
// ---------------------------------------------------------------------------

#[test]
fn chunk_fixtures_decode() {
    let c: ChatCompletionChunk = decode("openai/chunk_partial_tokens.json");
    assert_eq!(c.choices[0].delta.content.as_deref(), Some("partial tokens"));
    assert_eq!(c.choices[0].delta.role, None);
    assert_eq!(c.choices[0].finish_reason, None);

    let c: ChatCompletionChunk = decode("openai/chunk_first.json");
    assert_eq!(c.choices[0].delta.content.as_deref(), Some("first chunk"));

    let c: ChatCompletionChunk = decode("openai/chunk_empty_choices.json");
    assert!(c.choices.is_empty());
}

#[test]
fn chunk_constructors_produce_the_openai_stream_shape() {
    let (id, model, created) = ("chatcmpl-1", "m", 1_700_000_000i64);

    let first = ChatCompletionChunk::first(id, model, created);
    assert_eq!(
        serde_json::to_value(&first).unwrap(),
        json!({
            "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
        })
    );

    let tok = ChatCompletionChunk::token(id, model, created, "hi");
    assert_eq!(tok.choices[0].delta.content.as_deref(), Some("hi"));
    assert_eq!(tok.object, "chat.completion.chunk");

    let last = ChatCompletionChunk::finish(id, model, created, "stop");
    assert_eq!(last.choices[0].finish_reason.as_deref(), Some("stop"));
    assert_eq!(last.choices[0].delta, Delta::default());
    assert_eq!(
        serde_json::to_value(&last).unwrap()["choices"][0]["delta"],
        json!({}),
        "an empty delta must be `{{}}`, not `{{\"role\":null,\"content\":null}}`"
    );
    assert!(serde_json::to_value(&last).unwrap().get("usage").is_none());
}

// ---------------------------------------------------------------------------
// SSE framing
// ---------------------------------------------------------------------------

#[test]
fn sse_frames_are_exactly_data_json_blankline() {
    let c = ChatCompletionChunk::token("chatcmpl-1", "m", 1, "hi");
    let frame = sse::encode_chunk(&c);
    let body = serde_json::to_string(&c).unwrap();
    assert_eq!(frame, format!("data: {body}\n\n"));
    assert!(frame.ends_with("\n\n"));
    assert!(!body.contains('\n'), "the JSON payload must be single-line");
    assert_eq!(sse::DONE, "data: [DONE]\n\n");
}

#[test]
fn sse_decode_line_round_trips_a_chunk() {
    let c = ChatCompletionChunk::token("chatcmpl-1", "m", 1, "hi");
    let frame = sse::encode_chunk(&c);
    let line = frame.trim_end_matches('\n');
    match sse::decode_line(line) {
        Some(SseEvent::Chunk(got)) => assert_eq!(*got, c),
        other => panic!("decode_line -> {other:?}"),
    }
    assert!(matches!(sse::decode_line(sse::DONE), Some(SseEvent::Done)));
    assert!(matches!(sse::decode_line("data: [DONE]"), Some(SseEvent::Done)));
    // Tolerate the no-space form some servers emit.
    assert!(matches!(sse::decode_line("data:[DONE]"), Some(SseEvent::Done)));
}

#[test]
fn sse_decode_line_ignores_non_data_lines() {
    for line in ["", "\n", ": keep-alive", "event: message", "id: 7", "retry: 100", "garbage"] {
        assert!(sse::decode_line(line).is_none(), "line {line:?} should be ignored");
    }
    assert!(sse::decode_line("data: {not json}").is_none());
}

#[test]
fn a_whole_sse_stream_decodes() {
    let (id, model, created) = ("chatcmpl-1", "m", 1i64);
    let mut stream = String::new();
    stream.push_str(&sse::encode_chunk(&ChatCompletionChunk::first(id, model, created)));
    for word in ["echo:", " hello"] {
        stream.push_str(&sse::encode_chunk(&ChatCompletionChunk::token(id, model, created, word)));
    }
    stream.push_str(&sse::encode_chunk(&ChatCompletionChunk::finish(id, model, created, "stop")));
    stream.push_str(sse::DONE);

    let events: Vec<SseEvent> = stream.lines().filter_map(sse::decode_line).collect();
    assert_eq!(events.len(), 5);
    assert!(matches!(events[4], SseEvent::Done));

    let text: String = events
        .iter()
        .filter_map(|e| match e {
            SseEvent::Chunk(c) => c.choices.first()?.delta.content.clone(),
            SseEvent::Done => None,
        })
        .collect();
    assert_eq!(text, "echo: hello");
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Our own contract (`docs/architecture.md` "Failure semantics"): the OpenAI
/// lane answers an unknown model with the OpenAI error envelope.
#[test]
fn openai_error_envelope_round_trips() {
    let body = r#"{"error":{"message":"model 'x' not found","type":"invalid_request_error","code":"model_not_found"}}"#;
    let e: ErrorResponse = serde_json::from_str(body).unwrap();
    assert_eq!(e.error.code.as_deref(), Some("model_not_found"));
    assert_eq!(e.error.kind, "invalid_request_error");
    assert_eq!(serde_json::to_string(&e).unwrap(), body);

    let e = ErrorResponse::model_not_found("x");
    let v = serde_json::to_value(&e).unwrap();
    assert_eq!(v["error"]["code"], "model_not_found");
    assert!(v["error"]["message"].as_str().unwrap().contains('x'));
}

/// PAIR's *proxies* answer with a bare string error on both lanes
/// (`services/lmstudio-proxy/failover_test.go:222`, `services/lmstudio-proxy/proxy.go:985`).
/// That shape is distinct from the OpenAI envelope; pin the difference so
/// nobody "fixes" one into the other.
#[test]
fn pair_proxy_errors_are_a_different_shape() {
    for f in [
        "openai/error_plain_model_not_found.json",
        "openai/error_plain_no_owner.json",
        "openai/error_plain_upstream.json",
    ] {
        assert!(
            serde_json::from_str::<ErrorResponse>(&raw(f)).is_err(),
            "{f} must not decode as the OpenAI error envelope"
        );
        let e: pair_protocol::ollama::ErrorResponse = decode(f);
        assert!(!e.error.is_empty());
    }
}
