//! `GET :14318/v1/node-info` — the telemetry lane PAIR probes.
//!
//! PAIR: `probeNodeInfo` (`services/nvpair-manual-nodes/manager.go:493-527`)
//! requires HTTP 200 and a body that decodes into its `NodeInfoResponse`
//! (`:69-81`); a transport error, a non-200 or a decode failure all mark the
//! lane down. The reference emitter sets `Content-Type: application/json`
//! (`services/nvpair-node-info/main.go:260-272`).
//!
//! The handler never caches: PAIR only folds telemetry into scheduling while
//! `telemetryValid` is true and `msSince <= 10000`
//! (`services/nvpair-job-scheduler/telemetry.go:45-46`), so every response must
//! carry the freshest sample the [`TelemetrySource`](pair_telemetry::TelemetrySource)
//! has.

use crate::{access_log, not_found, AppState};
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/node-info", get(node_info))
        .with_state(state)
        .fallback(not_found)
        .layer(axum::middleware::from_fn(access_log))
}

async fn node_info(State(state): State<AppState>) -> Response {
    axum::Json(state.telemetry.node_info()).into_response()
}
