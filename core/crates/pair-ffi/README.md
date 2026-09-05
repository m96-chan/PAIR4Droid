# pair-ffi

The UniFFI (0.32, proc-macro mode) surface the Android app calls. Namespace `pair_ffi`
→ Kotlin package `uniffi.pair_ffi`; the cdylib is `libpair4droid_ffi.so`.

## What Kotlin sees

| Kotlin | Rust |
| --- | --- |
| `PairNode.start(config): NodeStatus` | `pair_node_start` |
| `PairNode.stop()` | `pair_node_stop` |
| `PairNode.status(): NodeStatus` | `pair_node_status` |
| `PairNode.pushSignals(signals)` | `pair_node_push_signals` |
| `PairNode.setModelsDir(path)` | `pair_node_set_models_dir` |
| `PairNode.listModels(): List<ModelInfo>` | `pair_node_list_models` |
| `PairNode.setEventListener(events)` | `pair_node_set_event_listener` |

UniFFI cannot generate a Kotlin `object` with static methods, so it generates the
top-level functions `pairNodeStart(config)`, `pairNodeStop()`, … and the hand-written
`android/app/src/main/java/uniffi/pair_ffi/PairNode.kt` (same package) declares the
`object PairNode` facade that delegates to them. That file is the only hand-written
Kotlin in `uniffi.pair_ffi`; keep it in step with the exports here.

## Generating the bindings

`android/app/build.gradle.kts` does this as part of `preBuild`. By hand, from `core/`:

```bash
cargo build -p pair-ffi                                  # or cargo ndk … for the phone
cargo run -p pair-ffi --features bindgen --bin uniffi-bindgen -- \
    generate --library target/debug/libpair4droid_ffi.so \
    --language kotlin --out-dir /tmp/pair-ffi-bindings
```

`--features bindgen` is required: the `uniffi-bindgen` binary pulls in clap +
`uniffi_bindgen`, which have no business in the cdylib shipped on the phone, so it is
gated by `required-features = ["bindgen"]`.

`generated-preview/pair_ffi.kt` is that command's output, committed verbatim so the
design session can diff the generated Kotlin against the app without a Rust toolchain.
Regenerate it whenever this crate's public surface changes; it is *not* on the app's
source path (Gradle uses its own freshly generated copy under `build/generated/uniffi`).

## Notes

- All state is process-global (`OnceLock<Mutex<State>>`): one node per app, one dedicated
  2-worker tokio runtime, all calls blocking — Kotlin wraps them in coroutines.
- `pair_node_status` / `pair_node_list_models` work while stopped; the model list then
  comes from scanning the models dir for `*.gguf` so the Models screen has content
  before the first start.
- Without the `llama` feature, a start with an empty `mock_models` fails with
  `PairError::Engine { msg: "built without llama feature and no mock models" }`.
