//! Tests for the llama.cpp backend (ticket #7).
//!
//! Everything here runs without a GGUF file except the `#[ignore]`d tests at the
//! bottom, which need `PAIR4DROID_TEST_GGUF=/path/to/model.gguf` and are run with
//! `cargo test -p pair-engine --features llama -- --ignored`.
#![cfg(feature = "llama")]

use futures::StreamExt;
use pair_engine::llama::{
    chatml_prompt, parse_name_metadata, scan_models_dir, LlamaConfig, LlamaEngine, Utf8Accumulator,
};
use pair_engine::*;
use std::path::PathBuf;

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("pair-engine-llama-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

// ---------------------------------------------------------------- catalogue

#[tokio::test]
async fn empty_directory_yields_an_empty_catalogue() {
    let dir = tmpdir("empty");
    let e = LlamaEngine::new(dir.clone(), LlamaConfig::default()).expect("engine");
    assert!(e.list_models().await.is_empty());
    assert_eq!(e.status(), EngineStatus::default());
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn missing_directory_yields_an_empty_catalogue() {
    let dir = std::env::temp_dir().join("pair-engine-llama-does-not-exist-xyz");
    let e = LlamaEngine::new(dir, LlamaConfig::default()).expect("engine");
    assert!(e.list_models().await.is_empty());
}

#[tokio::test]
async fn scan_picks_up_gguf_files_only_and_derives_metadata() {
    let dir = tmpdir("scan");
    std::fs::write(dir.join("qwen2.5-1.5b-instruct-q4_k_m.gguf"), b"not really gguf").unwrap();
    std::fs::write(dir.join("notes.txt"), b"ignore me").unwrap();

    let models = scan_models_dir(&dir, 2048).expect("scan");
    assert_eq!(models.len(), 1, "only *.gguf files are catalogued");
    let m = &models[0];
    assert_eq!(m.name, "qwen2.5-1.5b-instruct-q4_k_m");
    assert_eq!(m.path, dir.join("qwen2.5-1.5b-instruct-q4_k_m.gguf").to_string_lossy());
    assert_eq!(m.size_bytes, "not really gguf".len() as u64);
    assert_eq!(m.context_length, 2048);
    // sha256("not really gguf")
    assert_eq!(m.digest.len(), 64);
    assert!(m.modified_at.contains('T'), "RFC3339 mtime: {}", m.modified_at);
    // no readable GGUF metadata → filename fallback
    assert_eq!(m.family, "qwen2.5");
    assert_eq!(m.parameter_size, "1.5B");
    assert_eq!(m.quantization, "Q4_K_M");
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn unknown_model_is_model_not_found() {
    let dir = tmpdir("unknown");
    let e = LlamaEngine::new(dir.clone(), LlamaConfig::default()).unwrap();
    let req = ChatRequest {
        model: "nope".into(),
        messages: vec![ChatMessage { role: ChatRole::User, content: "hi".into() }],
        params: GenerationParams::default(),
    };
    match e.chat(req).await {
        Err(EngineError::ModelNotFound(m)) => assert_eq!(m, "nope"),
        Err(other) => panic!("wrong error: {other}"),
        Ok(_) => panic!("unknown model must not start a stream"),
    }
    match e.count_tokens("nope", "hi").await {
        Err(EngineError::ModelNotFound(_)) => {}
        other => panic!("expected ModelNotFound, got {:?}", other.map(|n| n.to_string())),
    }
    std::fs::remove_dir_all(&dir).ok();
}

// ------------------------------------------------------- filename metadata

#[test]
fn filename_metadata_parsing() {
    let m = parse_name_metadata("qwen2.5-1.5b-instruct-q4_k_m");
    assert_eq!(
        (m.family.as_str(), m.parameter_size.as_str(), m.quantization.as_str()),
        ("qwen2.5", "1.5B", "Q4_K_M")
    );

    let m = parse_name_metadata("Llama-3.2-3B-Instruct-Q8_0");
    assert_eq!(
        (m.family.as_str(), m.parameter_size.as_str(), m.quantization.as_str()),
        ("llama", "3B", "Q8_0")
    );

    let m = parse_name_metadata("phi-3-mini-4k-instruct-f16");
    assert_eq!((m.family.as_str(), m.parameter_size.as_str(), m.quantization.as_str()), ("phi", "", "F16"));

    // Nothing recognisable: family is the whole stem, the rest empty.
    let m = parse_name_metadata("tinyllama");
    assert_eq!(
        (m.family.as_str(), m.parameter_size.as_str(), m.quantization.as_str()),
        ("tinyllama", "", "")
    );

    let m = parse_name_metadata("gemma-2-2b-it-IQ4_XS");
    assert_eq!((m.parameter_size.as_str(), m.quantization.as_str()), ("2B", "IQ4_XS"));

    let m = parse_name_metadata("mixtral-8x7b-instruct-q5_k_s");
    assert_eq!(m.quantization, "Q5_K_S");
}

// ------------------------------------------------------------ chat template

#[test]
fn chatml_fallback_renders_the_documented_shape() {
    let msgs = vec![
        ChatMessage { role: ChatRole::System, content: "be nice".into() },
        ChatMessage { role: ChatRole::User, content: "hi".into() },
        ChatMessage { role: ChatRole::Assistant, content: "hello".into() },
        ChatMessage { role: ChatRole::User, content: "bye".into() },
    ];
    assert_eq!(
        chatml_prompt(&msgs),
        "<|im_start|>system\nbe nice<|im_end|>\n\
         <|im_start|>user\nhi<|im_end|>\n\
         <|im_start|>assistant\nhello<|im_end|>\n\
         <|im_start|>user\nbye<|im_end|>\n\
         <|im_start|>assistant\n"
    );
    assert_eq!(chatml_prompt(&[]), "<|im_start|>assistant\n");

    let tool = vec![ChatMessage { role: ChatRole::Tool, content: "42".into() }];
    assert_eq!(chatml_prompt(&tool), "<|im_start|>tool\n42<|im_end|>\n<|im_start|>assistant\n");
}

// -------------------------------------------------------- UTF-8 accumulator

#[test]
fn utf8_accumulator_emits_only_complete_characters() {
    let mut acc = Utf8Accumulator::default();
    let bytes = "日本".as_bytes().to_vec();
    assert_eq!(bytes.len(), 6);

    assert_eq!(acc.push(&bytes[0..1]), None);
    assert_eq!(acc.push(&bytes[1..2]), None);
    assert_eq!(acc.push(&bytes[2..3]), Some("日".to_string()));
    assert_eq!(acc.push(&bytes[3..5]), None);
    assert_eq!(acc.push(&bytes[5..6]), Some("本".to_string()));
    assert!(acc.is_empty());
}

#[test]
fn utf8_accumulator_splits_a_mixed_chunk() {
    let mut acc = Utf8Accumulator::default();
    let mut bytes = b"ok ".to_vec();
    bytes.extend_from_slice(&"日".as_bytes()[..2]);
    assert_eq!(acc.push(&bytes), Some("ok ".to_string()));
    assert!(!acc.is_empty());
    assert_eq!(acc.push(&"日".as_bytes()[2..]), Some("日".to_string()));
}

#[test]
fn utf8_accumulator_replaces_invalid_bytes() {
    let mut acc = Utf8Accumulator::default();
    assert_eq!(acc.push(&[b'a', 0xFF, b'b']), Some("a\u{fffd}b".to_string()));
    assert!(acc.is_empty());
    assert_eq!(acc.push(b""), None);
}

// ------------------------------------------------------------------ config

#[test]
fn config_defaults_are_phone_friendly() {
    let c = LlamaConfig::default();
    assert!(c.use_mmap, "mmap keeps model pages out of the anonymous RSS budget");
    assert!(c.n_ctx >= 512);
    assert!(c.max_queue >= 1);
}

// ------------------------------------------------- real model (opt-in only)

fn test_gguf() -> Option<PathBuf> {
    std::env::var_os("PAIR4DROID_TEST_GGUF").map(PathBuf::from).filter(|p| p.exists())
}

fn engine_for_test_gguf() -> Option<(LlamaEngine, String)> {
    let gguf = test_gguf()?;
    let dir = gguf.parent().unwrap().to_path_buf();
    let name = gguf.file_stem().unwrap().to_string_lossy().to_string();
    let cfg = LlamaConfig { n_ctx: 1024, ..LlamaConfig::default() };
    Some((LlamaEngine::new(dir, cfg).unwrap(), name))
}

#[tokio::test]
#[ignore = "needs PAIR4DROID_TEST_GGUF"]
async fn real_model_streams_start_tokens_done() {
    let Some((e, name)) = engine_for_test_gguf() else { return };
    assert!(e.model(&name).await.is_some(), "catalogue must contain {name}");

    let req = ChatRequest {
        model: name.clone(),
        messages: vec![ChatMessage { role: ChatRole::User, content: "Say hi.".into() }],
        params: GenerationParams { max_tokens: Some(8), temperature: Some(0.0), ..Default::default() },
    };
    let mut s = e.chat(req).await.expect("stream");
    let mut seen_start = false;
    let mut tokens = 0;
    let mut done = false;
    while let Some(ev) = s.next().await {
        match ev.expect("no stream error") {
            TokenEvent::Start { .. } => {
                assert!(!seen_start && tokens == 0);
                seen_start = true;
            }
            TokenEvent::Token(t) => {
                assert!(seen_start && !done);
                assert!(!t.is_empty());
                tokens += 1;
            }
            TokenEvent::Done { completion_tokens, .. } => {
                assert!(!done);
                assert!(completion_tokens <= 8, "max_tokens honoured");
                done = true;
            }
        }
    }
    assert!(seen_start && done);
    assert_eq!(e.status().loaded_model.as_deref(), Some(name.as_str()));
    assert!(e.status().loaded_bytes > 0);

    e.unload().await;
    // unload is processed on the worker thread
    for _ in 0..200 {
        if e.status().loaded_model.is_none() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(e.status().loaded_model, None);
}

#[tokio::test]
#[ignore = "needs PAIR4DROID_TEST_GGUF"]
async fn real_model_counts_tokens() {
    let Some((e, name)) = engine_for_test_gguf() else { return };
    let n = e.count_tokens(&name, "Hello, world!").await.expect("count");
    assert!(n > 0);
}

#[tokio::test]
#[ignore = "needs PAIR4DROID_TEST_GGUF"]
async fn real_model_rejects_a_prompt_longer_than_the_context() {
    let Some(gguf) = test_gguf() else { return };
    let dir = gguf.parent().unwrap().to_path_buf();
    let name = gguf.file_stem().unwrap().to_string_lossy().to_string();
    let e = LlamaEngine::new(dir, LlamaConfig { n_ctx: 32, ..LlamaConfig::default() }).unwrap();
    let long = "word ".repeat(400);
    let req = ChatRequest {
        model: name,
        messages: vec![ChatMessage { role: ChatRole::User, content: long }],
        params: GenerationParams::default(),
    };
    match e.chat(req).await {
        Err(EngineError::ContextExceeded { prompt_tokens, context_length }) => {
            assert!(prompt_tokens > context_length);
            assert_eq!(context_length, 32);
        }
        Err(other) => panic!("wrong error: {other}"),
        Ok(_) => panic!("expected ContextExceeded"),
    }
}

#[tokio::test]
#[ignore = "needs PAIR4DROID_TEST_GGUF"]
async fn real_model_stops_on_a_stop_string() {
    let Some((e, name)) = engine_for_test_gguf() else { return };
    let req = ChatRequest {
        model: name,
        messages: vec![ChatMessage { role: ChatRole::User, content: "Count: 1 2 3 4 5".into() }],
        params: GenerationParams {
            max_tokens: Some(64),
            temperature: Some(0.0),
            stop: vec!["3".into()],
            ..Default::default()
        },
    };
    let mut s = e.chat(req).await.expect("stream");
    let mut text = String::new();
    while let Some(ev) = s.next().await {
        if let TokenEvent::Token(t) = ev.unwrap() {
            text.push_str(&t);
        }
    }
    assert!(!text.contains('3'), "stop string must not be emitted: {text:?}");
}

#[tokio::test]
#[ignore = "needs PAIR4DROID_TEST_GGUF"]
async fn real_model_cancels_when_the_stream_is_dropped() {
    let Some((e, name)) = engine_for_test_gguf() else { return };
    let req = ChatRequest {
        model: name,
        messages: vec![ChatMessage { role: ChatRole::User, content: "Write a long story.".into() }],
        params: GenerationParams { max_tokens: Some(4096), ..Default::default() },
    };
    let mut s = e.chat(req).await.expect("stream");
    s.next().await;
    s.next().await;
    drop(s);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while e.status().active != 0 {
        assert!(std::time::Instant::now() < deadline, "generation did not stop after drop");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}
