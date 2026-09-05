# ADR-0002: All logic in a Rust core; Kotlin only for Android lifecycle; UniFFI bridge

**Status:** accepted · 2026-09-05

## Decision
Protocol, HTTP servers, inference, telemetry and policy live in Rust crates that build and test on
the host. The Android app links `libpair4droid_ffi.so` through UniFFI (proc-macro mode) and only
handles Service lifecycle, permissions, locks, UI, model import, and pushing battery/thermal signals.

## Why
- TDD on the host: every PAIR-facing behaviour is testable with `cargo test` without a device or SDK.
- One implementation runs on the phone (`pair-ffi`) and on a desktop (`pair-cli`) for e2e against a real PAIR.
- llama.cpp bindings, axum, tokio and rustls are mature on `aarch64-linux-android`.

## Consequences
- The Android build needs NDK + `cargo-ndk`; CI does it (this dev container has no Android SDK).
- The Kotlin↔Rust API is small and versioned by the UniFFI contract in `pair-ffi/src/lib.rs`.
