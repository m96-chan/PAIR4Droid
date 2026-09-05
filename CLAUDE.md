# PAIR4Droid — agent guide

PAIR4Droid turns an Android device into a node of **NVIDIA Personal AI Router (PAIR)**
so PAIR can route inference to the phone and see its resources. Phase 1 targets PAIR's
*manual node* contract (plain HTTP on fixed ports); Phase 2 targets the full mDNS/mTLS peer.

Read in this order: this file → `docs/architecture.md` → `docs/pair-contract.md` (the
PAIR wire contract, cited to PAIR's Go source) → the ticket you are working on.

## Layout

```
core/                      Rust workspace (all logic lives here)
  crates/pair-protocol/    serde wire types for node-info / OpenAI lane / Ollama lane   (no I/O)
  crates/pair-engine/      Engine trait, MockEngine, llama.cpp backend (feature `llama`)
  crates/pair-telemetry/   /proc sampling, EWMA accelerator load, admission policy → node-info
  crates/pair-node/        axum servers :1234 :11434 :14318 + `probe` (PAIR's probe replayed)
  crates/pair-cli/         `pair4droid serve|probe` desktop runner (dev + e2e against real PAIR)
  crates/pair-ffi/         UniFFI bindings → Kotlin
android/                   Kotlin + Compose app: Foreground Service, model import, calls pair-ffi
docs/                      architecture.md, pair-contract.md, adr/
.github/workflows/ci.yml   Rust checks + Android build (this dev container has no Android SDK)
```

Reference checkout of PAIR (read-only, not committed): `/home/user/nvidia/personal-ai-router`.
Cite it as `services/<path>:<line>` in docs and code comments.

## Commands

```bash
cd core
cargo test --workspace                       # host tests; must stay green
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
cargo run -p pair-cli -- serve --mock demo   # run a mock node on 1234/11434/14318
cargo run -p pair-cli -- probe 127.0.0.1     # replay PAIR's manual-node probe
cargo test -p pair-engine --features llama   # llama.cpp backend (needs cmake; GGUF via PAIR4DROID_TEST_GGUF)
cd ../android && ./gradlew assembleDebug     # needs Android SDK + NDK + cargo-ndk (CI does this)
```

When several agents build concurrently, set `CARGO_TARGET_DIR=core/target-<yourname>`
so cargo's build lock does not serialise everyone.

## Process rules (TiDD + TDD — non-negotiable)

1. **Ticket first.** Every change maps to a GitHub Issue in `m96-chan/PAIR4Droid`. No ticket → open one
   (title `<area>: <what>`, labels `phase-N`, `area:<crate>`, `tdd`) before coding.
2. **Test first.** Write the failing test, run it, watch it fail, then implement, then refactor.
   Wire-format tests use fixtures copied *verbatim* from PAIR's own tests/fakes (`core/crates/pair-protocol/tests/fixtures/`).
3. **Commit per ticket**, message `<area>: <imperative summary>\n\nRefs #N` (or `Closes #N` when the
   acceptance list is fully ticked). Never mix tickets in one commit.
4. **Green before push**: `cargo fmt --all --check && cargo clippy ... -D warnings && cargo test --workspace`.
5. Close a ticket only when every acceptance checkbox is true; tick them in the issue as you go.

## Design invariants (do not break without an ADR in `docs/adr/`)

- JSON names must match PAIR byte-for-byte: `GPUs`, `telemetryValid`, `msSince`, `hostUuid`,
  `utilization_percent`, `data[].id`, `models[].name`. PAIR ignores everything else.
- Ports 1234 / 11434 / 14318 are PAIR compile-time constants; defaults never change, tests bind port 0.
- `GET :11434/` must answer 200 or PAIR marks the Ollama lane down.
- Unknown model → **404** (PAIR fails over to the next owner on 404). Overload/thermal → 503.
- Model names are PAIR's exact-match routing key. Advertise names unique in the fleet.
- `pair-node` depends only on traits (`pair_engine::Engine`, `pair_telemetry::TelemetrySource`); tests inject fakes.
- Streaming: OpenAI lane = SSE `data: {chunk}\n\n` … `data: [DONE]\n\n`; Ollama lane = NDJSON with final `done:true`.
  Flush every chunk; dropping the HTTP response must cancel the engine stream.
- Telemetry: `GPUs[0].utilization_percent` = EWMA of "inference busy" (ADR-0003). It is how PAIR's
  GPUPressure bands (40/70/85 %) learn the phone is busy.
- All logic in Rust. Kotlin only does Android lifecycle, permissions, UI, and pushes signals into Rust.
- No auth on the lanes (PAIR sends none); bind LAN-wide. Never add a path outside the contract without a ticket.
- Do not edit `/home/user/nvidia/personal-ai-router`.

## Android specifics to remember

- Foreground Service counts as *not visible* for Android 17's per-process memory limiter (≈5 GiB on a 16 GB
  device). Models are loaded with `mmap` so file-backed pages stay out of the anonymous-RSS budget.
- Hold `WifiLock(FULL_LOW_LATENCY)` + partial `WakeLock` only while the node runs.
- The device is the *server*; `ACCESS_LOCAL_NETWORK` (client-side permission) is not needed.
- Thermal / battery come from Kotlin via `pushSignals`; Rust never polls Android APIs.

## Collaboration

Design decisions come from the design session (Fable). A separate PAIR analysis session
(NVIDIA/Personal-AI-Router) is the reference for protocol questions; its findings land in
`docs/pair-contract.md`. When PAIR's code and the doc disagree, PAIR's code wins — fix the doc.
