//! Ticket #5 — Ollama-compatible lane on `:11434` (`ollama-proxy` talks to it).
//!
//! Fixtures, all copied verbatim from PAIR:
//!
//! | fixture | PAIR source |
//! | --- | --- |
//! | `tags_broker_management.json` | `services/tests/broker_management_test.go:46` |
//! | `tags_secure_inference.json` | `services/tests/secure_inference_test.go:221` |
//! | `tags_fanout_a.json` | `services/ollama-proxy/failover_test.go:382` |
//! | `tags_fanout_b.json` | `services/ollama-proxy/failover_test.go:384` |
//! | `tags_empty.json` | `services/ollama-proxy/failover_test.go:498`, `:512`, `:519` |
//! | `tags_null.json` | `services/ollama-proxy/failover_test.go:387` |
//! | `tags_names_only.json` | `services/nvpair-engine-manager/models_test.go:24` |
//! | `generate_secure_inference.json` | `services/tests/secure_inference_test.go:225` |
//! | `generate_chunk_partial.json` | `services/ollama-proxy/zombie_test.go:72` |
//! | `generate_chunk_first.json` | `services/ollama-proxy/zombie_test.go:121` |
//! | `generate_chunk_x.json` | `services/ollama-proxy/activity_test.go:117` |
//! | `fakeengine_generate.json` | `services/nvpair-engine-manager/testdata/fakeengine/main.go:233` |
//! | `fakeengine_generate_load.json` | `services/nvpair-engine-manager/testdata/fakeengine/main.go:216-221` |
//! | `fakeengine_ps.json` | `services/nvpair-engine-manager/testdata/fakeengine/main.go:238-245` |
//! | `chat_response_alias.json` | `services/tests/ollama_host_alias_test.go:34` |
//! | `done_true.json` | `services/ollama-proxy/failover_test.go:138`, `:166`, `:193`, `:276`, `:338` |
//! | `request_model_only.json` | `services/ollama-proxy/failover_test.go:352` |
//! | `request_stream_true.json` | `services/ollama-proxy/activity_test.go:131` |
//! | `request_generate_no_stream.json` | `services/tests/secure_inference_test.go:279` |
//! | `request_chat_messages_empty.json` | `services/tests/ollama_host_alias_test.go:64` |
//! | `request_run_model.json` | `services/nvpair-engine-manager/modelops.go:53` (`services/nvpair-engine-manager/model_test.go:63`) |
//! | `error_bad_request.json` | `services/ollama-proxy/failover_test.go:222` |
//! | `error_loading_model.json` | `services/ollama-proxy/failover_test.go:267` |
//! | `error_model_not_found.json` | `services/ollama-proxy/failover_test.go:333` |
//! | `error_inventory_unavailable.json` | `services/ollama-proxy/proxy.go:1106` |
//! | `version_test.json` | `services/nvpair-engine-manager/executor_test.go:374` |
//!
//! The `fakeengine_*` fixtures are the fake engine's own `json.Marshal` output
//! replayed for a request naming `qwen2.5-7b-instruct` (the model id PAIR's
//! `services/tests/broker_management_test.go:66` fake advertises) — the fake
//! echoes whichever model the caller asked for.
//!
//! PAIR's own tests carry no `/api/show` body and no timing-bearing final
//! object; those shapes come from Ollama's API and are asserted inline.

mod common;

use common::*;
use pair_protocol::ollama::{
    ndjson, ChatRequest, ChatResponse, ErrorResponse, GenerateRequest, GenerateResponse, Message,
    ModelDetails, PsResponse, ShowRequest, ShowResponse, TagsResponse, Timings, VersionResponse,
};
use serde_json::json;

// ---------------------------------------------------------------------------
// GET /api/tags
// ---------------------------------------------------------------------------

#[test]
fn tags_fixtures_decode() {
    // The only field PAIR reads is `models[].name`
    // (`services/nvpair-manual-nodes/manager.go:473-497`).
    let t: TagsResponse = decode("ollama/tags_broker_management.json");
    assert_eq!(t.models.len(), 1);
    assert_eq!(t.models[0].name, "llama3.2:latest");
    assert_eq!(t.models[0].model, "", "the fixture omits `model`");

    let t: TagsResponse = decode("ollama/tags_secure_inference.json");
    assert_eq!((t.models[0].name.as_str(), t.models[0].model.as_str()), ("m:latest", "m:latest"));

    let t: TagsResponse = decode("ollama/tags_fanout_a.json");
    assert_eq!(t.models.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(), ["a", "shared"]);
    assert_eq!(t.models[0].digest, "a-only");

    let t: TagsResponse = decode("ollama/tags_fanout_b.json");
    assert_eq!(t.models[0].name, "shared:latest");

    let t: TagsResponse = decode("ollama/tags_names_only.json");
    assert_eq!(t.models.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(), ["llama3:8b", "qwen:0.5b"]);

    let t: TagsResponse = decode("ollama/tags_empty.json");
    assert!(t.models.is_empty());
}

/// `{"models":null}` is the malformed fan-out member
/// (`services/ollama-proxy/failover_test.go:387`).
#[test]
fn null_models_decodes_as_empty_list() {
    let t: TagsResponse = decode("ollama/tags_null.json");
    assert!(t.models.is_empty());
}

#[test]
fn tags_round_trip() {
    // We always emit the full Ollama record, so a fixture that carries only a
    // subset may gain the remaining always-present keys.
    const MAY_ADD: &[&str] = &["model", "modified_at", "size", "digest", "details"];
    for f in [
        "ollama/tags_broker_management.json",
        "ollama/tags_secure_inference.json",
        "ollama/tags_fanout_a.json",
        "ollama/tags_fanout_b.json",
        "ollama/tags_names_only.json",
    ] {
        assert_roundtrip_superset::<TagsResponse>(f, MAY_ADD);
    }
    assert_roundtrip_exact::<TagsResponse>("ollama/tags_empty.json");
}

#[test]
fn a_tag_we_emit_carries_everything_ollama_sends() {
    let t = TagsResponse {
        models: vec![pair_protocol::ollama::TagModel {
            name: "qwen2.5-1.5b-instruct-q4_k_m".into(),
            model: "qwen2.5-1.5b-instruct-q4_k_m".into(),
            modified_at: "2026-09-05T00:00:00Z".into(),
            size: 1_117_320_512,
            digest: "0".repeat(64),
            details: ModelDetails {
                format: "gguf".into(),
                family: "qwen2".into(),
                families: Some(vec!["qwen2".into()]),
                parameter_size: "1.5B".into(),
                quantization_level: "Q4_K_M".into(),
                ..Default::default()
            },
        }],
    };
    let v = serde_json::to_value(&t).unwrap();
    assert_eq!(keys_of(&t.models[0]), ["details", "digest", "model", "modified_at", "name", "size"]);
    assert_eq!(v["models"][0]["details"]["quantization_level"], "Q4_K_M");
    assert_eq!(v["models"][0]["details"]["families"], json!(["qwen2"]));
}

// ---------------------------------------------------------------------------
// GET /api/version
// ---------------------------------------------------------------------------

#[test]
fn version_round_trips() {
    let v: VersionResponse = decode("ollama/version_test.json");
    assert_eq!(v.version, "test");
    assert_roundtrip_exact::<VersionResponse>("ollama/version_test.json");
}

// ---------------------------------------------------------------------------
// POST /api/chat — request
// ---------------------------------------------------------------------------

/// Ollama streams unless the client says otherwise
/// (`docs/pair-contract.md` §3.2; Ollama's own default).
#[test]
fn chat_stream_defaults_to_true() {
    let r: ChatRequest = decode("ollama/request_model_only.json");
    assert_eq!(r.model, "llama");
    assert!(r.stream, "absent `stream` must mean true");
    assert!(r.messages.is_empty());

    let r: ChatRequest = decode("ollama/request_stream_true.json");
    assert!(r.stream);

    let r: ChatRequest = decode("ollama/request_chat_messages_empty.json");
    assert_eq!(r.model, "alias-e2e-model");
    assert!(r.stream);

    let r: ChatRequest = serde_json::from_str(r#"{"model":"m","stream":false}"#).unwrap();
    assert!(!r.stream);
}

#[test]
fn generate_stream_defaults_to_true_and_false_is_honoured() {
    let r: GenerateRequest = decode("ollama/request_generate_no_stream.json");
    assert_eq!((r.model.as_str(), r.prompt.as_str()), ("m:latest", "hi"));
    assert!(!r.stream);

    let r: GenerateRequest = decode("ollama/request_run_model.json");
    assert_eq!(r.model, "demo-model:1b");
    assert_eq!(r.prompt, "");
    assert!(!r.stream);

    let r: GenerateRequest = serde_json::from_str(r#"{"model":"m","prompt":"p"}"#).unwrap();
    assert!(r.stream);
}

#[test]
fn chat_request_options_and_unknown_fields() {
    let body = json!({
        "model": "m",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi", "images": ["AA=="]}
        ],
        "stream": false,
        "keep_alive": "5m",
        "format": "json",
        "tools": [],
        "think": false,
        "options": {"num_predict": 64, "temperature": 0.1, "top_p": 0.8, "seed": 3,
                    "stop": ["</s>"], "num_ctx": 4096, "repeat_penalty": 1.1}
    });
    let r: ChatRequest = serde_json::from_value(body).expect("unknown fields ignored");
    assert_eq!(r.messages.len(), 2);
    assert_eq!(r.messages[1].images.as_deref(), Some(["AA==".to_string()].as_slice()));
    assert_eq!(r.keep_alive, Some(json!("5m")));
    let o = r.options.expect("options");
    assert_eq!((o.num_predict, o.seed, o.num_ctx), (Some(64), Some(3), Some(4096)));
    assert_eq!(o.stop.as_deref(), Some(["</s>".to_string()].as_slice()));
}

// ---------------------------------------------------------------------------
// POST /api/chat, /api/generate — responses
// ---------------------------------------------------------------------------

#[test]
fn chat_response_fixture_decodes() {
    let r: ChatResponse = decode("ollama/chat_response_alias.json");
    assert_eq!(r.message.role, "assistant");
    assert_eq!(r.message.content, "routed-through-alias");
    assert!(r.done);
    assert_eq!(r.timings, None, "no timing keys on the wire -> no Timings");
}

#[test]
fn generate_response_fixtures_decode() {
    let r: GenerateResponse = decode("ollama/generate_secure_inference.json");
    assert_eq!((r.model.as_str(), r.response.as_str()), ("m:latest", "hello from the backend"));
    assert!(r.done);
    assert_eq!(r.timings, None);

    let r: GenerateResponse = decode("ollama/generate_chunk_partial.json");
    assert!(!r.done);
    assert_eq!(r.response, "partial tokens before the client vanished");

    let r: GenerateResponse = decode("ollama/generate_chunk_first.json");
    assert_eq!((r.model.as_str(), r.response.as_str(), r.done), ("", "first chunk", false));

    let r: GenerateResponse = decode("ollama/generate_chunk_x.json");
    assert_eq!(r.response, "x");
    assert!(!r.done);

    let r: GenerateResponse = decode("ollama/fakeengine_generate.json");
    assert_eq!(r.response, "ack:Say OK.");

    let r: GenerateResponse = decode("ollama/fakeengine_generate_load.json");
    assert_eq!(r.done_reason.as_deref(), Some("load"));
    assert_eq!(r.response, "");
}

/// `{"done":true}` is the whole body every `ollama-proxy` failover fake returns.
#[test]
fn bare_done_decodes_on_both_response_types() {
    let c: ChatResponse = decode("ollama/done_true.json");
    assert!(c.done);
    assert_eq!(c.message, Message::default());
    let g: GenerateResponse = decode("ollama/done_true.json");
    assert!(g.done);
}

/// The final NDJSON object flattens the timing block into the top level.
#[test]
fn final_object_flattens_timings() {
    let body = json!({
        "model": "m", "created_at": "2026-09-05T00:00:00.123456789Z",
        "message": {"role": "assistant", "content": ""},
        "done": true, "done_reason": "stop",
        "total_duration": 4_883_583_458u64, "load_duration": 1_334_875u64,
        "prompt_eval_count": 26, "prompt_eval_duration": 342_546_000u64,
        "eval_count": 282, "eval_duration": 4_535_599_000u64
    });
    let r: ChatResponse = serde_json::from_value(body.clone()).expect("flattened timings");
    let t = r.timings.clone().expect("timings");
    assert_eq!(t.total_duration, 4_883_583_458);
    assert_eq!(t.prompt_eval_count, 26);
    assert_eq!(t.eval_count, 282);
    assert_eq!(t.eval_duration, 4_535_599_000);
    assert_eq!(serde_json::to_value(&r).unwrap(), body, "timings must re-flatten");
}

#[test]
fn non_final_object_omits_timings_entirely() {
    let r = ChatResponse::token("m", "2026-09-05T00:00:00Z", "hi");
    let v = serde_json::to_value(&r).unwrap();
    assert_eq!(
        v,
        json!({
            "model": "m", "created_at": "2026-09-05T00:00:00Z",
            "message": {"role": "assistant", "content": "hi"},
            "done": false
        })
    );
    for k in ["total_duration", "eval_count", "done_reason"] {
        assert!(v.get(k).is_none(), "{k} must be absent on a token chunk: {v}");
    }
}

#[test]
fn final_constructors_carry_done_reason_and_timings() {
    let t = Timings { eval_count: 3, eval_duration: 1_000_000, ..Default::default() };
    let r = ChatResponse::final_("m", "2026-09-05T00:00:00Z", "stop", t.clone());
    let v = serde_json::to_value(&r).unwrap();
    assert_eq!(v["done"], json!(true));
    assert_eq!(v["done_reason"], json!("stop"));
    assert_eq!(v["eval_count"], json!(3));
    assert_eq!(v["message"], json!({"role": "assistant", "content": ""}));

    let g = GenerateResponse::final_("m", "2026-09-05T00:00:00Z", "stop", t);
    let v = serde_json::to_value(&g).unwrap();
    assert_eq!((v["done"].clone(), v["response"].clone()), (json!(true), json!("")));
    assert_eq!(v["eval_duration"], json!(1_000_000));
}

// ---------------------------------------------------------------------------
// GET /api/ps, POST /api/show
// ---------------------------------------------------------------------------

#[test]
fn ps_fixture_decodes() {
    let p: PsResponse = decode("ollama/fakeengine_ps.json");
    assert_eq!(p.models.len(), 1);
    assert_eq!(p.models[0].name, "qwen2.5-7b-instruct");
    assert_eq!(p.models[0].size_vram, 1_712_345_088);
    assert_eq!(p.models[0].expires_at, "2026-01-01T00:00:00Z");
    assert_roundtrip_superset::<PsResponse>("ollama/fakeengine_ps.json", &["size", "digest", "details"]);
}

#[test]
fn show_request_accepts_model_or_legacy_name() {
    let r: ShowRequest = serde_json::from_str(r#"{"model":"m"}"#).unwrap();
    assert_eq!(r.resolved(), Some("m"));
    let r: ShowRequest = serde_json::from_str(r#"{"name":"legacy"}"#).unwrap();
    assert_eq!(r.resolved(), Some("legacy"));
    let r: ShowRequest = serde_json::from_str(r#"{"model":"m","name":"legacy","verbose":true}"#).unwrap();
    assert_eq!(r.resolved(), Some("m"), "`model` wins");
    let r: ShowRequest = serde_json::from_str("{}").unwrap();
    assert_eq!(r.resolved(), None);
}

#[test]
fn show_response_shape() {
    let mut s = ShowResponse {
        details: ModelDetails {
            format: "gguf".into(),
            family: "qwen2".into(),
            parameter_size: "1.5B".into(),
            quantization_level: "Q4_K_M".into(),
            ..Default::default()
        },
        capabilities: Some(vec!["completion".into()]),
        ..Default::default()
    };
    s.model_info.insert("general.architecture".into(), json!("qwen2"));
    let v = serde_json::to_value(&s).unwrap();
    assert_eq!(v["details"]["family"], "qwen2");
    assert_eq!(v["model_info"]["general.architecture"], "qwen2");
    assert_eq!(v["capabilities"], json!(["completion"]));

    let back: ShowResponse = serde_json::from_value(v).unwrap();
    assert_eq!(back, s);
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[test]
fn error_fixtures_round_trip() {
    for (f, want) in [
        ("ollama/error_bad_request.json", "bad request"),
        ("ollama/error_loading_model.json", "loading model"),
        ("ollama/error_model_not_found.json", "model not found"),
        ("ollama/error_inventory_unavailable.json", "model inventory unavailable"),
    ] {
        let e: ErrorResponse = decode(f);
        assert_eq!(e.error, want);
        assert_roundtrip_exact::<ErrorResponse>(f);
    }
    // The 404 body our node sends (`docs/architecture.md` "Failure semantics").
    let e = ErrorResponse::model_not_found("qwen");
    assert_eq!(serde_json::to_string(&e).unwrap(), r#"{"error":"model 'qwen' not found"}"#);
}

// ---------------------------------------------------------------------------
// NDJSON framing
// ---------------------------------------------------------------------------

#[test]
fn ndjson_lines_are_exactly_json_newline() {
    let r = ChatResponse::token("m", "2026-09-05T00:00:00Z", "hi");
    let line = ndjson::encode_line(&r);
    assert_eq!(line, format!("{}\n", serde_json::to_string(&r).unwrap()));
    assert!(line.ends_with('\n'));
    assert_eq!(line.matches('\n').count(), 1, "one object per line");
    assert_eq!(ndjson::encode_line(&json!({"done": true})), "{\"done\":true}\n");
}

#[test]
fn an_ndjson_stream_decodes_back() {
    let created = "2026-09-05T00:00:00Z";
    let mut body = String::new();
    for word in ["echo:", " hello"] {
        body.push_str(&ndjson::encode_line(&ChatResponse::token("m", created, word)));
    }
    body.push_str(&ndjson::encode_line(&ChatResponse::final_(
        "m",
        created,
        "stop",
        Timings { eval_count: 2, ..Default::default() },
    )));

    let objects: Vec<ChatResponse> = body.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(objects.len(), 3);
    assert!(!objects[0].done && !objects[1].done && objects[2].done);
    assert_eq!(objects.iter().map(|o| o.message.content.as_str()).collect::<String>(), "echo: hello");
    assert_eq!(objects[2].timings.as_ref().unwrap().eval_count, 2);
}

// ---------------------------------------------------------------------------
// Leniency
// ---------------------------------------------------------------------------

#[test]
fn unknown_fields_are_ignored_everywhere() {
    let _: TagsResponse =
        serde_json::from_str(r#"{"models":[{"name":"a","expires_at":"x"}],"extra":1}"#).unwrap();
    let _: ChatResponse = serde_json::from_str(
        r#"{"model":"m","created_at":"t","message":{"role":"assistant","content":"c","thinking":"x"},"done":true,"context":[1,2]}"#,
    )
    .unwrap();
    let _: PsResponse = serde_json::from_str(r#"{"models":[{"name":"a","context_length":4096}]}"#).unwrap();
    let v: VersionResponse = serde_json::from_str(r#"{"version":"0.1.0","build":"x"}"#).unwrap();
    assert_eq!(v.version, "0.1.0");
}
