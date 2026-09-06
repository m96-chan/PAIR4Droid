# PAIR4Droid — Android app

Kotlin + Jetpack Compose app that turns the phone into a PAIR manual node. All protocol,
HTTP, inference and telemetry logic lives in `core/` (Rust); this module is lifecycle,
permissions, UI, and model file management only — see `../CLAUDE.md` and
`../docs/architecture.md`.

## Prerequisites

- JDK 17
- Android SDK: `platforms;android-36`, `build-tools;36.0.0`
- Android NDK `27.2.12479018` (`sdkmanager "ndk;27.2.12479018"`), with `ANDROID_NDK_HOME`
  pointing at it (NDK 29 works too)
- `cmake` ≥ 3.14 on the PATH — llama.cpp is built into the app's native library
- Rust with the Android target: `rustup target add aarch64-linux-android`
  (add `x86_64-linux-android` too if you want to run on the emulator)
- `cargo install cargo-ndk`

CI (`.github/workflows/ci.yml`) installs exactly this set before running `./gradlew
--no-daemon assembleDebug`, so that workflow is the source of truth if this drifts.

## Building

```bash
cd android
./gradlew assembleDebug
```

`assembleDebug` depends on two Exec tasks defined in `app/build.gradle.kts` that run before
Kotlin compilation:

1. **`cargoBuild`** — cross-compiles `pair-ffi` with `--features llama` to `arm64-v8a`
   (+ `x86_64` for debug builds) via `cargo ndk --platform 29`, writing `.so`s straight
   into `app/src/main/jniLibs/`. The platform must match `minSdk`: llama.cpp's mmap
   loader needs `posix_madvise`, which bionic only exposes from API 23, and cargo-ndk
   would otherwise default to 21. libc++ is linked statically on Android
   (`core/crates/pair-engine/Cargo.toml`), so the APK carries a single native library
   and no `libc++_shared.so`. `./gradlew assembleDebug -Ppair4droid.mockOnly=true` skips
   the llama.cpp build for a quick UI-only iteration.
2. **`generateUniffiBindings`** — runs pair-ffi's `uniffi-bindgen` binary against the built
   `libpair4droid_ffi.so` to generate the Kotlin bindings under
   `app/build/generated/uniffi/`, which is added as a Kotlin source directory.

Both read `CARGO_TARGET_DIR` if you've set it (see the root `CLAUDE.md` note about running
several agents against `core/` concurrently).

## Running

Install the debug APK on a device on the same LAN/Wi-Fi as your PAIR host, launch it, and
flip the switch on the Node tab. With no `.gguf` files imported yet, debug builds fall back
to a `phone-demo` mock model (`BuildConfig.MOCK_MODELS`) so the node still comes up; once
a `.gguf` is imported the node serves it with llama.cpp (CPU, mmap, one request at a time).

## Adding the node in PAIR

1. Open the app, flip the node on.
2. Note the IPv4 address shown on the Node tab (read from the device's Wi-Fi link — the
   phone must be on the same LAN as the PAIR host).
3. In PAIR: **Nodes → Add manual node** → enter that IP. PAIR probes
   `:1234`/`:11434`/`:14318` itself (docs/architecture.md); no further configuration needed.

## Managing models

Models tab → **Import .gguf** (via the system file picker) copies the file into app-private
storage (`filesDir/models/`). Renaming matters: the filename is PAIR's exact-match routing
key across your fleet, so keep it unique — the app suggests a `phone-<model>` pattern.
Rename/delete/import all trigger an automatic model-directory rescan on the running node.

## Known assumptions to confirm once pair-ffi is implemented

`core/crates/pair-ffi/src/lib.rs` is currently a doc-comment-only stub; this app was written
against that contract, filling a few gaps with the most conventional UniFFI choice. Flagged
here for the design session:

- **Kotlin bindings package name**: assumed `uniffi.pair_ffi` (UniFFI's default namespace
  derived from the crate name `pair-ffi`).
- **`NodeEvents` registration**: the doc comment describes `NodeEvents` as a callback
  interface but never names how Kotlin registers an implementation with `PairNode`. This
  code assumes `PairNode.setEventListener(events)` (called once in
  `repo/NodeRepository.kt`); if pair-ffi instead threads it through `start(config, events)`,
  move the call there.
- **`ExternalSignals.thermal` type**: assumed a UniFFI enum `ThermalStatus` with variants
  `NONE/LIGHT/MODERATE/SEVERE/CRITICAL/EMERGENCY/SHUTDOWN`, mirroring
  `PowerManager.THERMAL_STATUS_*` (`util/Thermal.kt`).
- **`ModelInfo` fields**: assumed `name: String`, `sizeBytes: <numeric>`, `quant: String`.
  The Models screen matches these against files in `filesDir/models/` best-effort by name
  and falls back to the file's on-disk size if there's no match, so a mismatch here degrades
  gracefully rather than crashing.
- **Numeric field widths** (ports, `active`/`queued`, `sizeBytes`, `modelBudgetBytes`): the
  Kotlin code converts with `.toInt()`/`.toLong()`/`.toUShort()`/`.toULong()` at every
  boundary rather than assuming a specific signed/unsigned width, since UniFFI's Kotlin
  codegen for `u16`/`u32`/`u64` isn't pinned down until pair-ffi's Rust types are written.
- `NodeStatus.ports` is deliberately never read — the app already knows the three ports from
  `NodeConfig` it passed in, so it sidesteps guessing that field's shape.
