//! Forwards `tracing` events to Kotlin's `NodeEvents`.
//!
//! pair-node logs every request through its access-log middleware with the
//! structured fields `method`, `path`, `status`, `ms` and (for inference paths)
//! `model`, under the message `"request"` (`pair-node/src/lib.rs`, `access_log`).
//! Those become `onRequest(lane, model, status, ms)`; everything at INFO or above
//! also becomes `onLog(level, msg)`.
//!
//! The lane is derived from the path because the middleware does not know which
//! listener accepted the connection: `/v1/node-info` → `node-info`, other `/v1/*`
//! → `openai`, the rest → `ollama`.

use std::cell::Cell;
use std::fmt;
use std::sync::Once;
use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::util::SubscriberInitExt;

thread_local! {
    /// Guards against a listener that itself logs (which would re-enter this
    /// layer from inside `on_log` forever).
    static IN_CALLBACK: Cell<bool> = const { Cell::new(false) };
}

/// Install the forwarder as the process' global `tracing` subscriber. Idempotent;
/// silently does nothing if something else already installed one.
pub(crate) fn install() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let _ = tracing_subscriber::registry().with(Forwarder.with_filter(LevelFilter::INFO)).try_init();
    });
}

struct Forwarder;

impl<S: Subscriber> Layer<S> for Forwarder {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        if IN_CALLBACK.get() {
            return;
        }
        let Some(events) = crate::listener() else {
            return;
        };

        let mut fields = Fields::default();
        event.record(&mut fields);

        let _guard = ReentryGuard::enter();
        events.on_log(event.metadata().level().as_str().to_string(), fields.render());
        if fields.message.as_deref() == Some("request") {
            if let (Some(status), Some(ms)) = (fields.get_i64("status"), fields.get_i64("ms")) {
                events.on_request(
                    fields.lane(),
                    fields.get("model").unwrap_or_default().to_string(),
                    status as i32,
                    ms,
                );
            }
        }
    }
}

/// Clears [`IN_CALLBACK`] however the callback ends, panic included.
struct ReentryGuard;

impl ReentryGuard {
    fn enter() -> Self {
        IN_CALLBACK.set(true);
        Self
    }
}

impl Drop for ReentryGuard {
    fn drop(&mut self) {
        IN_CALLBACK.set(false);
    }
}

/// Every field of an event, rendered to a string (the values pair-node emits are
/// a mix of `u16`, `u128`, `String` and `Display`, so strings are the only shape
/// they all share).
#[derive(Default)]
struct Fields {
    message: Option<String>,
    rest: Vec<(&'static str, String)>,
}

impl Fields {
    fn get(&self, name: &str) -> Option<&str> {
        self.rest.iter().find(|(k, _)| *k == name).map(|(_, v)| v.as_str())
    }

    fn get_i64(&self, name: &str) -> Option<i64> {
        self.get(name)?.parse().ok()
    }

    /// An explicit `lane` field wins; otherwise the path says which lane it was.
    fn lane(&self) -> String {
        if let Some(lane) = self.get("lane") {
            return lane.to_string();
        }
        match self.get("path").unwrap_or_default() {
            "/v1/node-info" => "node-info",
            path if path.starts_with("/v1/") => "openai",
            _ => "ollama",
        }
        .to_string()
    }

    fn render(&self) -> String {
        let mut out = self.message.clone().unwrap_or_default();
        for (key, value) in &self.rest {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(key);
            out.push('=');
            out.push_str(value);
        }
        out
    }

    fn push(&mut self, field: &Field, value: String) {
        if field.name() == "message" {
            self.message = Some(value);
        } else {
            self.rest.push((field.name(), value));
        }
    }
}

impl Visit for Fields {
    fn record_str(&mut self, field: &Field, value: &str) {
        self.push(field, value.to_string());
    }
    fn record_i64(&mut self, field: &Field, value: i64) {
        self.push(field, value.to_string());
    }
    fn record_u64(&mut self, field: &Field, value: u64) {
        self.push(field, value.to_string());
    }
    fn record_i128(&mut self, field: &Field, value: i128) {
        self.push(field, value.to_string());
    }
    fn record_u128(&mut self, field: &Field, value: u128) {
        self.push(field, value.to_string());
    }
    fn record_bool(&mut self, field: &Field, value: bool) {
        self.push(field, value.to_string());
    }
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.push(field, format!("{value:?}"));
    }
}
