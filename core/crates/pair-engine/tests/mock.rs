//! Behaviour tests for [`pair_engine::mock::MockEngine`] (ticket #6).

use futures::StreamExt;
use pair_engine::mock::MockEngine;
use pair_engine::*;
use std::time::Duration;

fn user(content: &str) -> ChatMessage {
    ChatMessage { role: ChatRole::User, content: content.to_string() }
}

fn req(model: &str, messages: Vec<ChatMessage>, params: GenerationParams) -> ChatRequest {
    ChatRequest { model: model.to_string(), messages, params }
}

/// Drain a stream into (start_prompt_tokens, tokens, done).
async fn drain(mut s: TokenStream) -> (u32, Vec<String>, TokenEvent) {
    let mut prompt_tokens = None;
    let mut tokens = Vec::new();
    let mut done = None;
    while let Some(ev) = s.next().await {
        match ev.expect("mock never errors mid-stream") {
            TokenEvent::Start { prompt_tokens: p } => {
                assert!(prompt_tokens.is_none(), "Start emitted twice");
                assert!(tokens.is_empty(), "Token before Start");
                prompt_tokens = Some(p);
            }
            TokenEvent::Token(t) => {
                assert!(prompt_tokens.is_some(), "Token before Start");
                assert!(done.is_none(), "Token after Done");
                tokens.push(t);
            }
            ev @ TokenEvent::Done { .. } => {
                assert!(done.is_none(), "Done emitted twice");
                done = Some(ev);
            }
        }
    }
    (prompt_tokens.expect("no Start"), tokens, done.expect("no Done"))
}

#[tokio::test]
async fn catalogue_advertises_requested_models() {
    let e = MockEngine::with_models(&["alpha", "beta"]);
    let models = e.list_models().await;
    assert_eq!(models.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(), ["alpha", "beta"]);

    let alpha = e.model("alpha").await.expect("alpha in catalogue");
    assert_eq!(alpha.path, "");
    assert_eq!(alpha.family, "mock");
    assert_eq!(alpha.parameter_size, "0B");
    assert_eq!(alpha.quantization, "none");
    assert_eq!(alpha.context_length, 4096);
    assert_eq!(alpha.size_bytes, 0);
    assert_eq!(alpha.digest.len(), 64, "digest is sha256 hex");

    assert!(e.model("gamma").await.is_none());
}

#[tokio::test]
async fn digest_is_sha256_of_name() {
    let e = MockEngine::with_models(&["abc"]);
    let m = e.model("abc").await.unwrap();
    // sha256("abc")
    assert_eq!(m.digest, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    assert!(m.modified_at.contains('T'), "modified_at is RFC3339: {}", m.modified_at);
}

#[tokio::test]
async fn unknown_model_is_model_not_found() {
    let e = MockEngine::with_models(&["alpha"]);
    match e.chat(req("nope", vec![user("hi")], Default::default())).await {
        Err(EngineError::ModelNotFound(m)) => assert_eq!(m, "nope"),
        Err(other) => panic!("wrong error: {other}"),
        Ok(_) => panic!("unknown model must not start a stream"),
    }
}

#[tokio::test]
async fn echoes_last_user_message_one_token_per_word() {
    let e = MockEngine::with_models(&["alpha"]);
    let messages = vec![
        ChatMessage { role: ChatRole::System, content: "sys".into() },
        user("first question"),
        ChatMessage { role: ChatRole::Assistant, content: "answer".into() },
        user("hello wide world"),
    ];
    let s = e.chat(req("alpha", messages, Default::default())).await.unwrap();
    let (prompt_tokens, tokens, done) = drain(s).await;

    assert_eq!(tokens, ["echo:", " hello", " wide", " world"]);
    assert_eq!(tokens.concat(), "echo: hello wide world");
    // prompt_tokens = whitespace word count over all message contents
    assert_eq!(prompt_tokens, 1 + 2 + 1 + 3);
    match done {
        TokenEvent::Done { finish_reason, prompt_tokens: p, completion_tokens, .. } => {
            assert_eq!(finish_reason, FinishReason::Stop);
            assert_eq!(p, 7);
            assert_eq!(completion_tokens, 4);
        }
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn no_user_message_echoes_bare_prefix() {
    let e = MockEngine::with_models(&["alpha"]);
    let messages = vec![ChatMessage { role: ChatRole::System, content: "only system".into() }];
    let s = e.chat(req("alpha", messages, Default::default())).await.unwrap();
    let (prompt_tokens, tokens, _) = drain(s).await;
    assert_eq!(tokens, ["echo:"]);
    assert_eq!(prompt_tokens, 2);
}

#[tokio::test]
async fn max_tokens_truncates_with_length_finish_reason() {
    let e = MockEngine::with_models(&["alpha"]);
    let params = GenerationParams { max_tokens: Some(2), ..Default::default() };
    let s = e.chat(req("alpha", vec![user("a b c d")], params)).await.unwrap();
    let (_, tokens, done) = drain(s).await;
    assert_eq!(tokens, ["echo:", " a"]);
    match done {
        TokenEvent::Done { finish_reason, completion_tokens, .. } => {
            assert_eq!(finish_reason, FinishReason::Length);
            assert_eq!(completion_tokens, 2);
        }
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn max_tokens_at_exact_length_still_stops() {
    let e = MockEngine::with_models(&["alpha"]);
    let params = GenerationParams { max_tokens: Some(3), ..Default::default() };
    let s = e.chat(req("alpha", vec![user("a b")], params)).await.unwrap();
    let (_, tokens, done) = drain(s).await;
    assert_eq!(tokens, ["echo:", " a", " b"]);
    assert!(matches!(done, TokenEvent::Done { finish_reason: FinishReason::Stop, .. }));
}

#[tokio::test]
async fn stop_string_ends_generation_early() {
    let e = MockEngine::with_models(&["alpha"]);
    let params = GenerationParams { stop: vec!["stopme".into()], ..Default::default() };
    let s = e.chat(req("alpha", vec![user("keep going stopme and more")], params)).await.unwrap();
    let (_, tokens, done) = drain(s).await;
    assert_eq!(tokens, ["echo:", " keep", " going"]);
    match done {
        TokenEvent::Done { finish_reason, completion_tokens, .. } => {
            assert_eq!(finish_reason, FinishReason::Stop);
            assert_eq!(completion_tokens, 3);
        }
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn count_tokens_is_whitespace_word_count() {
    let e = MockEngine::with_models(&["alpha"]);
    assert_eq!(e.count_tokens("alpha", "one two  three\nfour").await.unwrap(), 4);
    assert_eq!(e.count_tokens("alpha", "   ").await.unwrap(), 0);
    let err = e.count_tokens("nope", "x").await.unwrap_err();
    assert!(matches!(err, EngineError::ModelNotFound(_)));
}

#[tokio::test]
async fn loaded_model_tracks_last_used_and_unload_clears_it() {
    let e = MockEngine::with_models(&["alpha", "beta"]);
    assert_eq!(e.status().loaded_model, None);

    let s = e.chat(req("beta", vec![user("hi")], Default::default())).await.unwrap();
    drain(s).await;
    assert_eq!(e.status().loaded_model.as_deref(), Some("beta"));
    assert_eq!(e.status().active, 0);

    e.unload().await;
    assert_eq!(e.status().loaded_model, None);
}

/// Consume whatever is left of a partially-read stream.
async fn drain_rest(mut s: TokenStream) {
    while let Some(ev) = s.next().await {
        ev.expect("mock never errors mid-stream");
    }
}

#[tokio::test]
async fn active_is_one_while_a_stream_is_alive() {
    let e = MockEngine::with_models(&["alpha"]).with_token_delay(Duration::from_millis(20));
    let mut s = e.chat(req("alpha", vec![user("a b c d e")], Default::default())).await.unwrap();
    // Start event
    assert!(matches!(s.next().await, Some(Ok(TokenEvent::Start { .. }))));
    assert!(matches!(s.next().await, Some(Ok(TokenEvent::Token(_)))));
    assert_eq!(e.status().active, 1, "active while streaming");
    drain_rest(s).await;
    assert_eq!(e.status().active, 0, "active back to 0 after Done");
}

#[tokio::test]
async fn dropping_the_stream_stops_the_producer() {
    let e = MockEngine::with_models(&["alpha"]).with_token_delay(Duration::from_millis(50));
    let mut s = e.chat(req("alpha", vec![user("a b c d e f g h")], Default::default())).await.unwrap();
    assert!(matches!(s.next().await, Some(Ok(TokenEvent::Start { .. }))));
    assert!(matches!(s.next().await, Some(Ok(TokenEvent::Token(_)))));
    assert_eq!(e.status().active, 1);
    drop(s);

    // Producer must notice the closed channel and unregister quickly.
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while e.status().active != 0 {
        assert!(std::time::Instant::now() < deadline, "producer did not stop after drop");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(e.status().active, 0);
}

#[tokio::test]
async fn concurrent_streams_are_allowed_and_counted() {
    let e = MockEngine::with_models(&["alpha"]).with_token_delay(Duration::from_millis(20));
    let mut a = e.chat(req("alpha", vec![user("a b c d")], Default::default())).await.unwrap();
    let mut b = e.chat(req("alpha", vec![user("e f g h")], Default::default())).await.unwrap();
    a.next().await;
    b.next().await;
    a.next().await;
    b.next().await;
    assert_eq!(e.status().active, 2);
    assert_eq!(e.status().queued, 0);
    drain_rest(a).await;
    drain_rest(b).await;
    assert_eq!(e.status().active, 0);
}

#[tokio::test]
async fn token_delay_produces_observable_chunk_boundaries() {
    let e = MockEngine::with_models(&["alpha"]).with_token_delay(Duration::from_millis(30));
    let start = std::time::Instant::now();
    let s = e.chat(req("alpha", vec![user("a b c")], Default::default())).await.unwrap();
    let (_, tokens, _) = drain(s).await;
    assert_eq!(tokens.len(), 4);
    assert!(start.elapsed() >= Duration::from_millis(60), "delay was not applied: {:?}", start.elapsed());
}

#[tokio::test]
async fn engine_is_usable_as_a_trait_object() {
    let e: SharedEngine = std::sync::Arc::new(MockEngine::with_models(&["alpha"]));
    assert_eq!(e.list_models().await.len(), 1);
}

#[tokio::test]
async fn default_engine_has_a_demo_model() {
    let e = MockEngine::default();
    assert!(!e.list_models().await.is_empty());
}
