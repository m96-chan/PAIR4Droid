# PAIR4Droid

Turn an Android phone into a node of **NVIDIA Personal AI Router (PAIR)**: PAIR routes
inference to the phone's local LLM and sees the phone's CPU / memory / accelerator load.

- **Phase 1 (current):** PAIR *manual node*. The app serves PAIR's fixed-port contract —
  `:1234` OpenAI-compatible, `:11434` Ollama-compatible, `:14318` node-info — with zero
  changes to PAIR. Add the phone's LAN IP in PAIR → Nodes → *Add manual node*.
- **Phase 2:** full mDNS / mTLS peer (auto-discovery, cluster pairing). See Epic #2.

## Repository

| Path | What |
|------|------|
| `core/` | Rust workspace: protocol types, inference engine (mock + llama.cpp), telemetry, HTTP node, CLI, UniFFI |
| `android/` | Kotlin + Compose app: Foreground Service, model import, calls the Rust core |
| `docs/` | `architecture.md`, `pair-contract.md` (PAIR wire contract cited to its Go source), ADRs |
| `CLAUDE.md` | Working rules for agents and contributors (TiDD + TDD) |

## Quick start (desktop, no phone)

```bash
cd core
cargo run -p pair-cli -- serve --mock phone-demo      # mock node on 1234 / 11434 / 14318
cargo run -p pair-cli -- probe 127.0.0.1              # replay PAIR's probe: all three lanes should be "up"
```

Then in PAIR add `127.0.0.1` (or this machine's LAN IP) as a manual node and send a chat to
model `phone-demo`.

## Development

```bash
cd core && cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings
```

Every change is tied to a GitHub issue (`Refs #N`) and starts with a failing test. Details in
`CLAUDE.md`.

## Status

Scratch development in progress; see the issue tracker for the Phase 1 tickets.
