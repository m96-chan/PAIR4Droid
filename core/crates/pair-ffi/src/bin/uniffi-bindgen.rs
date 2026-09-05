//! The bindings generator this crate ships (ADR-0002); `android/app/build.gradle.kts`
//! runs it on the freshly built cdylib. Behind the `bindgen` feature so the cdylib
//! we ship on the phone does not link clap/uniffi_bindgen — see the crate README.

fn main() {
    uniffi::uniffi_bindgen_main()
}
