# Architecture

## What PAIR sees

```
PAIR host (nvpair-manual-nodes, every 10 s, 3 s timeout, 3 misses → evicted)
   │  GET http://<phone>:11434/            → 200                      ┐ Ollama lane up?
   │  GET http://<phone>:11434/api/tags    → {"models":[{"name":..}]} ┘
   │  GET http://<phone>:1234/v1/models    → {"data":[{"id":..}]}       OpenAI lane up? + model list
   │  GET http://<phone>:14318/v1/node-info→ {"GPUs":[..],"cpu":..,"memory":..,"telemetryValid":..,"hostUuid":..}
   ▼
inference: lmstudio-proxy → POST :1234/v1/chat/completions (SSE)   |  ollama-proxy → POST :11434/api/chat (NDJSON)
```

Manual nodes are registered by the user in the PAIR UI with the phone's LAN IP. Nothing is
discovered automatically in Phase 1 (see ADR-0001).

## Crates

```
                 ┌──────────────┐   HTTP (axum)    ┌─────────────┐
 PAIR ─────────▶ │  pair-node   │ ───────────────▶ │ pair-engine │ Engine trait
                 │ :1234 :11434 │                  │  Mock / llama.cpp
                 │ :14318       │ ◀── node_info ── ├─────────────┤
                 └──────┬───────┘                  │pair-telemetry│ /proc + signals + EWMA
                        │ uses types               └──────▲──────┘
                 ┌──────▼───────┐                         │ pushSignals (battery, thermal)
                 │pair-protocol │                  ┌──────┴──────┐
                 │ serde only   │                  │  pair-ffi   │ UniFFI → Kotlin
                 └──────────────┘                  └──────▲──────┘
                                                          │
                                                  android/ Foreground Service + Compose UI
```

- `pair-protocol` — pure data. Fixtures are copied from PAIR's own tests so a serialisation
  change that would break PAIR breaks our tests first.
- `pair-engine` — `Engine` trait: catalogue of GGUF models, lazy single-model load, token stream
  with `Start/Token/Done`, cancellation on drop, `status()` for telemetry. `MockEngine` is
  deterministic (`echo: <last user msg>`, one token per word). `LlamaEngine` (feature `llama`)
  wraps llama.cpp via `llama-cpp-2`, mmap on, one request at a time, queue depth reported.
- `pair-telemetry` — `Sampler` (procfs or fake) → CPU %, memory. `ExternalSignals` from Kotlin
  (battery, charging, thermal, screen). `InferenceLoad` from the node. Produces `NodeInfoResponse`
  and an `Admission` decision (refuse when too hot / battery low & discharging).
- `pair-node` — three axum routers on three listeners sharing `AppState { engine, telemetry, config }`.
  `probe` module replays PAIR's probe sequence and is reused by the CLI and by conformance tests.
- `pair-cli` — desktop runner for developing against a real PAIR without a phone.
- `pair-ffi` — UniFFI proc-macro API: `start/stop/status/pushSignals/listModels` + event callbacks;
  owns a tokio runtime.

## Telemetry → scheduling (ADR-0003)

PAIR's scheduler ranks owners of a model by `Pending` then `GPUPressure`, where pressure is the
EWMA of the node's max `GPUs[].utilization_percent` banded at 40/70/85 %. A phone has no GPU
counter PAIR can read, so we *define* the accelerator's utilisation as the EWMA of
"a generation is in flight". Idle phone → 0 % → pressure 0; saturated phone → 100 % → pressure 3.
`telemetryValid` is true once two CPU samples exist.

Verified against PAIR source (docs/pair-contract.md §2): a manual node's node-info **does** feed
scheduling. The broker reduces `GPUs[]` to `max(utilization_percent)`
(`services/nvpair-ui-broker/manualnodes.go:49-62`) and the scheduler applies it when `hostUuid`
is non-empty, `telemetryValid` is true and `msSince` ≤ 10 000 ms
(`services/nvpair-job-scheduler/telemetry.go`). PAIR then runs its own EWMA (α = 0.35) and bands
at 40/70/85 % with hysteresis at 35/65/80 %. Stale or invalid telemetry falls back to the neutral
band 1. Hence: keep the sample interval ≪ 10 s (default 2 s) and never emit an empty `hostUuid`.

## Model-name matching

PAIR's Capability Gate matches the request's `model` against the names a node advertised.
OpenAI lane: exact, case-sensitive. Ollama lane: PAIR appends `:latest` to a name whose last
segment has no tag before comparing, so `phone-qwen` and `phone-qwen:latest` are the same model;
the node must accept both spellings on `/api/chat`, `/api/generate`, `/api/show`.

## Failure semantics

| Situation                    | OpenAI lane                         | Ollama lane                       | Why |
|-----------------------------|-------------------------------------|-----------------------------------|-----|
| unknown model               | 404 `{"error":{...,"code":"model_not_found"}}` | 404 `{"error":"model 'x' not found"}` | PAIR fails over to the next owner on 404 for POSTs to inference paths; 408/429/5xx also fail over; 400/401/403/422 are returned as-is |
| admission refused (thermal/battery) | 503                          | 503                               | temporary; PAIR retries elsewhere |
| malformed JSON              | 400                                 | 400                               | |
| engine busy beyond queue    | 503                                 | 503                               | |
| client disconnect           | stream dropped → engine cancelled   | same                              | saves battery |

## Android lifecycle

`NodeService` (foreground, `specialUse`/`dataSync`) starts the Rust node via `pair-ffi`, holds a
Wi-Fi low-latency lock and a partial wake lock, and pushes battery/thermal into Rust every 30 s or on
change. The UI shows LAN IP + ports so the user can type them into PAIR's "add manual node".
Model files are imported through the Storage Access Framework into app-private storage and loaded
with mmap (Android 17 memory limiter counts anonymous RSS; file-backed pages are exempt).

## Phase 2 sketch

mDNS `_nvpair-node._tcp` advert (TXT `v=1;uuid=;ip=;ni=;ol=;lm=;cl=;em=`), `:14321` split
listener (first byte `0x16` → TLS else plain pairing), EAP-NOOB pairing with 6-digit PIN,
Ed25519 self-signed leaf pinned by DER equality, mTLS ingress on `ol`/`lm`, `em /v1/models`.
Tracked in Epic #2.
