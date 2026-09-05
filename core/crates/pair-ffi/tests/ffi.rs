//! Host tests for the UniFFI surface (the lib target is `pair4droid_ffi`).
//!
//! The FFI is deliberately a process-global singleton (Kotlin sees `object PairNode`),
//! so every test here takes [`GUARD`] and leaves the node stopped.

use pair4droid_ffi::*;
use parking_lot::Mutex;
use std::net::{IpAddr, Ipv4Addr};
use std::sync::Arc;
use std::time::Duration;

/// Serialises the tests: they all drive the same global node.
static GUARD: Mutex<()> = Mutex::new(());

/// Stops the node when a test ends — including by panic — so one failing test
/// cannot leave the singleton running and cascade `AlreadyRunning` into the rest.
struct StopOnDrop;

impl Drop for StopOnDrop {
    fn drop(&mut self) {
        let _ = pair_node_stop();
    }
}

fn guard() -> (parking_lot::MutexGuard<'static, ()>, StopOnDrop) {
    (GUARD.lock(), StopOnDrop)
}

/// Polls `pred` every 50 ms for up to `timeout`; returns whether it became true.
fn wait_until(timeout: Duration, mut pred: impl FnMut() -> bool) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if pred() {
            return true;
        }
        if std::time::Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

const MODEL: &str = "mock-a";

fn config() -> NodeConfig {
    NodeConfig {
        bind: "127.0.0.1".to_string(),
        // Port 0 everywhere: the OS picks, `NodeStatus::ports` reports what it picked.
        openai_port: 0,
        ollama_port: 0,
        node_info_port: 0,
        host_uuid: "11111111-2222-3333-4444-555555555555".to_string(),
        accelerator_name: "test-accel".to_string(),
        model_budget_bytes: 4 * 1024 * 1024 * 1024,
        mock_models: vec![MODEL.to_string(), "mock-b".to_string()],
    }
}

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap()
}

/// Collects everything Kotlin would see on its `NodeEvents` implementation.
#[derive(Default)]
struct Recorder {
    logs: Mutex<Vec<(String, String)>>,
    requests: Mutex<Vec<(String, String, i32, i64)>>,
    states: Mutex<Vec<NodeStatus>>,
}

impl NodeEvents for Recorder {
    fn on_log(&self, level: String, msg: String) {
        self.logs.lock().push((level, msg));
    }
    fn on_request(&self, lane: String, model: String, status: i32, ms: i64) {
        self.requests.lock().push((lane, model, status, ms));
    }
    fn on_state_changed(&self, status: NodeStatus) {
        self.states.lock().push(status);
    }
}

/// `Box<dyn NodeEvents>` is what the callback interface hands us, so the test
/// double is registered through a clonable `Arc` we keep a handle on.
struct Forward(Arc<Recorder>);

impl NodeEvents for Forward {
    fn on_log(&self, level: String, msg: String) {
        self.0.on_log(level, msg)
    }
    fn on_request(&self, lane: String, model: String, status: i32, ms: i64) {
        self.0.on_request(lane, model, status, ms)
    }
    fn on_state_changed(&self, status: NodeStatus) {
        self.0.on_state_changed(status)
    }
}

fn record() -> Arc<Recorder> {
    let recorder = Arc::new(Recorder::default());
    pair_node_set_event_listener(Box::new(Forward(Arc::clone(&recorder))));
    recorder
}

fn get(port: u16, path: &str) -> (u16, String) {
    rt().block_on(async move {
        let url = format!("http://127.0.0.1:{port}{path}");
        let response = reqwest::get(&url).await.expect("request failed");
        (response.status().as_u16(), response.text().await.unwrap_or_default())
    })
}

fn post_chat(port: u16, model: &str) -> u16 {
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
    });
    rt().block_on(async move {
        let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
        reqwest::Client::new().post(&url).json(&body).send().await.expect("request failed").status().as_u16()
    })
}

#[test]
fn start_serves_the_three_lanes_and_stop_frees_the_ports() {
    let _guard = guard();
    let events = record();

    let status = pair_node_start(config()).expect("start");
    assert!(status.running);
    let ports = status.ports.expect("ports are known once start returned");
    assert!(ports.openai != 0 && ports.ollama != 0 && ports.node_info != 0);
    assert_eq!(status.active, 0);
    assert_eq!(status.queued, 0);
    assert!(status.last_error.is_none());

    // The Models screen and PAIR's `/v1/models` probe see the mock catalogue.
    let (code, body) = get(ports.openai, "/v1/models");
    assert_eq!(code, 200);
    assert!(body.contains(MODEL), "models body: {body}");

    // node-info must carry the host_uuid we configured, or PAIR ignores the node.
    let (code, body) = get(ports.node_info, "/v1/node-info");
    assert_eq!(code, 200);
    let node_info: serde_json::Value = serde_json::from_str(&body).expect("node-info json");
    assert_eq!(node_info["hostUuid"], config().host_uuid);

    // PAIR's Ollama liveness probe.
    assert_eq!(get(ports.ollama, "/").0, 200);

    // At least one `onStateChanged` reaches Kotlin because of the start itself.
    assert!(!events.states.lock().is_empty(), "expected an onStateChanged callback after start");
    assert!(events.states.lock().last().unwrap().running);

    // Requests are forwarded to `onRequest` (lane/model/status/ms).
    assert!(
        events.requests.lock().iter().any(|(lane, _, status, _)| lane == "openai" && *status == 200),
        "expected an openai onRequest callback, got {:?}",
        events.requests.lock()
    );
    assert!(!events.logs.lock().is_empty(), "expected onLog callbacks");

    pair_node_stop().expect("stop");
    let status = pair_node_status();
    assert!(!status.running);
    assert!(status.ports.is_none());

    // The ports are free again: something else can bind them.
    let listener = std::net::TcpListener::bind((IpAddr::V4(Ipv4Addr::LOCALHOST), ports.openai));
    assert!(listener.is_ok(), "openai port still bound after stop: {:?}", listener.err());
}

#[test]
fn double_start_is_already_running() {
    let _guard = guard();
    pair_node_start(config()).expect("start");
    let err = pair_node_start(config()).expect_err("second start must fail");
    assert!(matches!(err, PairError::AlreadyRunning), "got {err:?}");
    pair_node_stop().expect("stop");
    assert!(matches!(pair_node_stop(), Err(PairError::NotRunning)));
}

#[test]
fn thermal_shutdown_refuses_inference_with_503() {
    let _guard = guard();
    let status = pair_node_start(config()).expect("start");
    let ports = status.ports.unwrap();

    assert_eq!(post_chat(ports.openai, MODEL), 200);

    pair_node_push_signals(ExternalSignals {
        battery_percent: Some(90),
        charging: Some(true),
        thermal: ThermalStatus::Shutdown,
        screen_on: Some(false),
    });
    assert_eq!(post_chat(ports.openai, MODEL), 503);

    // Unknown model still wins: PAIR fails over on 404, not on 503.
    assert_eq!(post_chat(ports.openai, "nope"), 404);

    pair_node_push_signals(ExternalSignals {
        battery_percent: Some(90),
        charging: Some(true),
        thermal: ThermalStatus::None,
        screen_on: Some(false),
    });
    assert_eq!(post_chat(ports.openai, MODEL), 200);

    pair_node_stop().expect("stop");
}

#[test]
fn status_and_list_models_work_before_start() {
    let _guard = guard();
    let status = pair_node_status();
    assert!(!status.running);
    assert!(status.ports.is_none());
    assert!(status.loaded_model.is_none());

    let dir = std::env::temp_dir().join(format!("pair-ffi-models-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("qwen2.5-1.5b-instruct-q4_k_m.gguf"), vec![7u8; 1024]).unwrap();
    std::fs::write(dir.join("notes.txt"), b"not a model").unwrap();

    pair_node_set_models_dir(dir.to_string_lossy().to_string());
    let models = pair_node_list_models();
    assert_eq!(models.len(), 1, "only .gguf files are models: {models:?}");
    let model = &models[0];
    assert_eq!(model.name, "qwen2.5-1.5b-instruct-q4_k_m");
    assert_eq!(model.size_bytes, 1024);
    assert_eq!(model.quant, "Q4_K_M");
    assert_eq!(model.parameter_size, "1.5B");
    assert!(model.path.ends_with("qwen2.5-1.5b-instruct-q4_k_m.gguf"));

    // Once running, the catalogue is the engine's.
    pair_node_start(config()).expect("start");
    let names: Vec<String> = pair_node_list_models().into_iter().map(|m| m.name).collect();
    assert_eq!(names, vec![MODEL.to_string(), "mock-b".to_string()]);
    pair_node_stop().expect("stop");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn start_without_mock_models_and_without_llama_reports_an_engine_error() {
    let _guard = guard();
    let dir = std::env::temp_dir().join(format!("pair-ffi-empty-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    pair_node_set_models_dir(dir.to_string_lossy().to_string());

    let mut cfg = config();
    cfg.mock_models = vec![];
    let result = pair_node_start(cfg);
    #[cfg(not(feature = "llama"))]
    {
        let err = result.expect_err("no engine is available without mock models");
        assert!(matches!(err, PairError::Engine { .. }), "got {err:?}");
        // A failed start leaves the node stopped and records the reason.
        let status = pair_node_status();
        assert!(!status.running);
        assert!(status.last_error.is_some());
    }
    #[cfg(feature = "llama")]
    {
        if result.is_ok() {
            pair_node_stop().expect("stop");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn poll_pushes_state_changes_to_the_listener() {
    let _guard = guard();
    let events = record();
    let status = pair_node_start(config()).expect("start");
    let ports = status.ports.unwrap();
    let before = events.states.lock().len();

    assert_eq!(post_chat(ports.openai, MODEL), 200);
    // The poll task runs every 2 s; the first tick after the request observes the
    // engine's new `loaded_model` and pushes a state change. CI runners are slow,
    // so wait generously instead of sleeping a fixed 2.5 s.
    let pushed = wait_until(Duration::from_secs(10), || events.states.lock().len() > before);
    let states = events.states.lock().clone();
    assert!(pushed, "expected a polled onStateChanged, got {states:?}");
    assert!(
        states.iter().any(|s| s.loaded_model.as_deref() == Some(MODEL)),
        "expected the loaded model to reach Kotlin, got {states:?}"
    );

    pair_node_stop().expect("stop");
}
