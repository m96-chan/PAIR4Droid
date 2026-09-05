//! UniFFI surface for the Android app. TODO(ticket: ffi): implement.
//!
//! Design contract (Kotlin sees exactly this):
//! - `object PairNode` with `start(config: NodeConfig): NodeStatus`, `stop()`,
//!   `status(): NodeStatus`, `pushSignals(signals: ExternalSignals)`,
//!   `setModelsDir(path: String)`, `listModels(): List<ModelInfo>`.
//! - `NodeConfig { bind, openaiPort, ollamaPort, nodeInfoPort, hostUuid, acceleratorName, modelBudgetBytes, mockModels: List<String> }`
//! - `NodeStatus { running, ports, loadedModel, active, queued, lastError }`
//! - Callback interface `NodeEvents { onLog(level, msg), onRequest(lane, model, status, ms), onStateChanged(status) }`
//! - Owns a dedicated multi-thread tokio runtime; all calls are blocking from Kotlin's side
//!   (Kotlin wraps them in coroutines).
uniffi::setup_scaffolding!();
