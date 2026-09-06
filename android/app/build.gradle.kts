import java.io.File

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

// The Rust workspace this app links against (see docs/architecture.md and
// docs/adr/0002-rust-core-uniffi.md). Never edit anything under `core/` from here.
val coreDir: File = rootProject.file("../core")

// `cargo-ndk` writes its .so outputs under this dir; respects CARGO_TARGET_DIR so a
// developer building several agents in parallel (see root CLAUDE.md) still finds the
// right library.
val cargoTargetDir: File =
    System.getenv("CARGO_TARGET_DIR")?.let { file(it) } ?: coreDir.resolve("target")

val uniffiLibraryFile: File = cargoTargetDir.resolve("aarch64-linux-android/release/libpair4droid_ffi.so")

// Must match `android.defaultConfig.minSdk` below. Passed to cargo-ndk as `--platform`
// so both the Rust target and llama.cpp's CMake build compile against this API level:
// cargo-ndk's default (21) predates bionic's `posix_madvise` (API 23), which
// llama.cpp's mmap loader needs (ticket #20).
val minSdkVersion = 29

// `cargoBuild` cross-compiles the pair-ffi cdylib for the ABIs this variant needs.
// Kept as a single (non-variant-aware) task for simplicity: it always builds arm64-v8a,
// and additionally builds x86_64 (for the emulator) unless the requested tasks look
// release-only. CI only ever runs `assembleDebug`, so both ABIs are built there.
//
// `--features llama` compiles llama.cpp (via `llama-cpp-2`) into the cdylib, so the APK
// runs real GGUF models; it needs cmake plus the NDK (`ANDROID_NDK_HOME`). libc++ is
// linked statically on Android (see core/crates/pair-engine/Cargo.toml), so no
// `libc++_shared.so` has to be packaged. Set `-Ppair4droid.mockOnly=true` to skip the
// llama.cpp build for a quick UI-only iteration (the node then only serves the
// `phone-demo` mock model of debug builds).
val cargoBuild = tasks.register<Exec>("cargoBuild") {
    group = "pair4droid"
    description = "Cross-compiles the pair-ffi cdylib for Android via cargo-ndk."
    workingDir = coreDir

    val releaseOnly = gradle.startParameter.taskNames.any { it.contains("Release", ignoreCase = true) } &&
        gradle.startParameter.taskNames.none { it.contains("Debug", ignoreCase = true) }
    val abis = if (releaseOnly) listOf("arm64-v8a") else listOf("arm64-v8a", "x86_64")
    val mockOnly = (project.findProperty("pair4droid.mockOnly") as String?)?.toBoolean() == true

    val cargoArgs = mutableListOf("ndk", "--platform", minSdkVersion.toString())
    abis.forEach { cargoArgs += listOf("-t", it) }
    cargoArgs += listOf("-o", file("src/main/jniLibs").path, "build", "--release", "-p", "pair-ffi")
    if (!mockOnly) cargoArgs += listOf("--features", "llama")
    commandLine(listOf("cargo") + cargoArgs)
}

// `generateUniffiBindings` runs the `uniffi-bindgen` binary pair-ffi ships (ADR-0002) to
// turn the built cdylib into Kotlin bindings under build/generated/uniffi. That binary
// does not exist yet in this checkout (core/crates/pair-ffi is a stub) — see android/README.md
// for the manual fallback until it lands.
val generateUniffiBindings = tasks.register<Exec>("generateUniffiBindings") {
    group = "pair4droid"
    description = "Generates Kotlin bindings for pair-ffi from the built cdylib."
    workingDir = coreDir
    dependsOn(cargoBuild)

    val outDir = layout.buildDirectory.dir("generated/uniffi").get().asFile
    doFirst { outDir.mkdirs() }

    commandLine(
        "cargo", "run", "-p", "pair-ffi", "--features", "bindgen", "--bin", "uniffi-bindgen", "--",
        "generate", "--library", uniffiLibraryFile.path,
        "--language", "kotlin", "--out-dir", outDir.path,
    )
}

tasks.named("preBuild") { dependsOn(generateUniffiBindings) }

android {
    namespace = "com.pair4droid"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.pair4droid"
        minSdk = minSdkVersion
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"

        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        debug {
            ndk {
                abiFilters += listOf("x86_64")
            }
            // Phase 1 demo model advertised only when `models/` in app storage is empty
            // (see NodeRepository.start / ticket #15).
            buildConfigField("String", "MOCK_MODELS", "\"phone-demo\"")
        }
        release {
            isMinifyEnabled = false
            buildConfigField("String", "MOCK_MODELS", "\"\"")
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
        // Material3 still marks TopAppBar & co. experimental; opt in project-wide.
        freeCompilerArgs += listOf("-opt-in=androidx.compose.material3.ExperimentalMaterial3Api")
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    packaging {
        jniLibs {
            useLegacyPackaging = false
        }
    }

    sourceSets {
        getByName("main") {
            kotlin.srcDir(layout.buildDirectory.dir("generated/uniffi"))
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.service)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.activity.compose)

    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.material.icons.extended)
    debugImplementation(libs.androidx.compose.ui.tooling)
    implementation(libs.androidx.compose.ui.tooling.preview)

    implementation(libs.kotlinx.coroutines.android)

    // UniFFI's generated Kotlin bindings talk to the cdylib through JNA.
    implementation("net.java.dev.jna:jna:${libs.versions.jna.get()}@aar")
}
