//! Integration tests for `Telemetry` / `TelemetrySource` / `pair_pressure_band`.
//!
//! Ticket: telemetry/core (GitHub issue #8).

use pair_telemetry::{
    pair_pressure_band, pair_pressure_band_with_previous, Admission, ExternalSignals, InferenceLoad,
    RawCpuSample, RawMemSample, Sampler, Telemetry, TelemetryConfig, TelemetrySource, ThermalStatus,
};
use std::collections::VecDeque;
use std::io;
use std::sync::Arc;
use std::time::Duration;

/// A `Sampler` whose cpu()/mem() results are queued by the test and popped one
/// per call. `Arc<FakeSampler>` is `Clone`, so the test keeps one handle to
/// push more results while another handle (also an `Arc`) is boxed into the
/// `Telemetry` under test.
#[derive(Default)]
struct FakeSampler {
    cpu: parking_lot::Mutex<VecDeque<io::Result<RawCpuSample>>>,
    mem: parking_lot::Mutex<VecDeque<io::Result<RawMemSample>>>,
}

impl FakeSampler {
    fn push_cpu_ok(&self, busy: u64, total: u64) {
        self.cpu.lock().push_back(Ok(RawCpuSample { busy, total, cores: 8, model_name: String::new() }));
    }

    fn push_cpu_err(&self) {
        self.cpu.lock().push_back(Err(io::Error::other("fake cpu read failure")));
    }

    fn push_mem_ok(&self, total_bytes: u64, available_bytes: u64) {
        self.mem.lock().push_back(Ok(RawMemSample { total_bytes, available_bytes }));
    }
}

impl Sampler for FakeSampler {
    fn cpu(&self) -> io::Result<RawCpuSample> {
        self.cpu.lock().pop_front().unwrap_or(Ok(RawCpuSample::default()))
    }
    fn mem(&self) -> io::Result<RawMemSample> {
        self.mem.lock().pop_front().unwrap_or(Ok(RawMemSample { total_bytes: 1, available_bytes: 1 }))
    }
}

fn config(alpha: f64) -> TelemetryConfig {
    TelemetryConfig {
        host_uuid: "host-1234".to_string(),
        accelerator_name: "Adreno 750 (llama.cpp)".to_string(),
        model_budget_bytes: 4 * 1024 * 1024 * 1024,
        sample_interval: Duration::from_secs(2),
        ewma_alpha: alpha,
        min_battery_percent_on_battery: 20,
        max_thermal: ThermalStatus::Severe,
    }
}

fn always_idle_sampler() -> Arc<FakeSampler> {
    let sampler = Arc::new(FakeSampler::default());
    for _ in 0..64 {
        sampler.push_cpu_ok(0, 100);
        sampler.push_mem_ok(1024, 512);
    }
    sampler
}

// ---------------------------------------------------------------------------
// cpu.utilization_percent / telemetryValid
// ---------------------------------------------------------------------------

#[test]
fn first_tick_has_no_percent_and_is_invalid() {
    let sampler = Arc::new(FakeSampler::default());
    sampler.push_cpu_ok(1000, 2000);
    sampler.push_mem_ok(16_000_000, 8_000_000);
    let telemetry = Telemetry::new(config(0.3), Box::new(sampler));

    telemetry.tick();
    let info = telemetry.node_info();

    assert!(!info.telemetry_valid, "first sample has no previous delta, must be invalid");
    assert_eq!(info.cpu.as_ref().unwrap().utilization_percent, 0);
}

#[test]
fn second_tick_computes_delta_percent_and_becomes_valid() {
    let sampler = Arc::new(FakeSampler::default());
    // busy delta 300 over total delta 1000 -> 30%
    sampler.push_cpu_ok(1_000, 5_000);
    sampler.push_cpu_ok(1_300, 6_000);
    sampler.push_mem_ok(16_000_000, 8_000_000);
    sampler.push_mem_ok(16_000_000, 6_000_000);
    let telemetry = Telemetry::new(config(0.3), Box::new(sampler));

    telemetry.tick();
    telemetry.tick();
    let info = telemetry.node_info();

    assert!(info.telemetry_valid);
    assert_eq!(info.cpu.as_ref().unwrap().utilization_percent, 30);
    assert_eq!(info.cpu.as_ref().unwrap().cores, 8);
    let mem = info.memory.as_ref().unwrap();
    assert_eq!(mem.total_bytes, 16_000_000);
    assert_eq!(mem.used_bytes, 16_000_000 - 6_000_000);
}

#[test]
fn sampler_error_marks_invalid_but_keeps_last_good_values() {
    let sampler = Arc::new(FakeSampler::default());
    sampler.push_cpu_ok(1_000, 5_000);
    sampler.push_cpu_ok(1_300, 6_000); // 30%
    sampler.push_cpu_err();
    sampler.push_mem_ok(16_000_000, 8_000_000);
    sampler.push_mem_ok(16_000_000, 6_000_000);
    sampler.push_mem_ok(16_000_000, 6_000_000);
    let telemetry = Telemetry::new(config(0.3), Box::new(sampler));

    telemetry.tick(); // first sample, no delta
    telemetry.tick(); // valid, 30%
    telemetry.tick(); // cpu sampler errors this round

    let info = telemetry.node_info();
    assert!(!info.telemetry_valid, "an errored sample must flip telemetryValid false");
    assert_eq!(
        info.cpu.as_ref().unwrap().utilization_percent,
        30,
        "last good utilization must be retained across a sampler error"
    );
}

#[test]
fn cpu_model_name_from_first_reported_sample_is_retained() {
    let sampler = Arc::new(FakeSampler::default());
    sampler.cpu.lock().push_back(Ok(RawCpuSample {
        busy: 100,
        total: 500,
        cores: 8,
        model_name: "Qualcomm Technologies, Inc SM8550".to_string(),
    }));
    sampler.cpu.lock().push_back(Ok(RawCpuSample {
        busy: 200,
        total: 700,
        cores: 8,
        model_name: String::new(),
    }));
    sampler.push_mem_ok(1, 1);
    sampler.push_mem_ok(1, 1);
    let telemetry = Telemetry::new(config(0.3), Box::new(sampler));

    telemetry.tick();
    telemetry.tick();

    assert_eq!(
        telemetry.node_info().cpu.unwrap().name,
        "Qualcomm Technologies, Inc SM8550",
        "a later empty model_name sample must not blank out a previously known name"
    );
}

// ---------------------------------------------------------------------------
// GPUs[0]: name / vram / EWMA-derived utilization
// ---------------------------------------------------------------------------

#[test]
fn gpu_fields_come_from_config_and_inference_load() {
    let sampler = always_idle_sampler();
    let telemetry = Telemetry::new(config(0.3), Box::new(sampler));
    telemetry.set_inference_load(InferenceLoad { active: 1, queued: 2, loaded_bytes: 777 });

    telemetry.tick();
    let info = telemetry.node_info();

    assert_eq!(info.gpus.len(), 1);
    let gpu = &info.gpus[0];
    assert_eq!(gpu.name, "Adreno 750 (llama.cpp)");
    assert_eq!(gpu.vram_bytes, 4 * 1024 * 1024 * 1024);
    assert_eq!(gpu.vram_used_bytes, 777);
}

#[test]
fn gpu_utilization_is_ewma_of_inference_busy_alpha_half_idle_busy_idle() {
    let sampler = always_idle_sampler();
    let telemetry = Telemetry::new(config(0.5), Box::new(sampler));

    // tick 1: idle -> ewma 0
    telemetry.tick();
    assert_eq!(telemetry.node_info().gpus[0].utilization_percent, 0);
    assert_eq!(pair_pressure_band(telemetry.node_info().gpus[0].utilization_percent), 0);

    telemetry.set_inference_load(InferenceLoad { active: 1, queued: 0, loaded_bytes: 0 });
    // tick 2: busy -> ewma 50
    telemetry.tick();
    assert_eq!(telemetry.node_info().gpus[0].utilization_percent, 50);
    // tick 3: busy -> ewma 75
    telemetry.tick();
    assert_eq!(telemetry.node_info().gpus[0].utilization_percent, 75);
    // tick 4: busy -> ewma 87.5 -> rounds to 88 -> band 3
    telemetry.tick();
    let u4 = telemetry.node_info().gpus[0].utilization_percent;
    assert_eq!(u4, 88);
    assert_eq!(pair_pressure_band(u4), 3, "saturated node must reach the top pressure band");

    telemetry.set_inference_load(InferenceLoad::default());
    // tick 5: idle -> ewma 43.75 -> 44 -> band 1
    telemetry.tick();
    let u5 = telemetry.node_info().gpus[0].utilization_percent;
    assert_eq!(u5, 44);
    assert_eq!(pair_pressure_band(u5), 1);
    // tick 6: idle -> ewma 21.875 -> 22 -> band 0
    telemetry.tick();
    let u6 = telemetry.node_info().gpus[0].utilization_percent;
    assert_eq!(u6, 22);
    assert_eq!(
        pair_pressure_band(u6),
        0,
        "node must fall back to band 0 once it has been idle for a couple of ticks"
    );
}

#[test]
fn gpu_utilization_with_small_alpha_lingers_in_bands_1_and_2() {
    let sampler = always_idle_sampler();
    let telemetry = Telemetry::new(config(0.1), Box::new(sampler));
    telemetry.set_inference_load(InferenceLoad { active: 1, queued: 0, loaded_bytes: 0 });

    let mut last_u = 0u32;
    for i in 1..=20 {
        telemetry.tick();
        last_u = telemetry.node_info().gpus[0].utilization_percent;
        if i == 4 {
            assert_eq!(pair_pressure_band(last_u), 0, "tick {i}: still ramping up in band 0");
        }
        if i == 8 {
            assert_eq!(last_u, 57);
            assert_eq!(pair_pressure_band(last_u), 1, "tick {i}: small alpha lingers in band 1");
        }
        if i == 12 {
            assert_eq!(last_u, 72);
            assert_eq!(
                pair_pressure_band(last_u),
                2,
                "tick {i}: now band 2, well after alpha=0.5 would be band 3"
            );
        }
        if i == 17 {
            assert_eq!(pair_pressure_band(last_u), 2, "tick {i}: still band 2 just before crossing 85");
        }
    }
    // after 20 always-busy ticks a small-alpha EWMA has finally climbed into band 3.
    assert_eq!(pair_pressure_band(last_u), 3);
}

// ---------------------------------------------------------------------------
// pair_pressure_band edges
// ---------------------------------------------------------------------------

#[test]
fn pressure_band_edges_match_pair_go_source() {
    // services/nvpair-job-scheduler/telemetry.go:111-122 (pressureBand)
    assert_eq!(pair_pressure_band(0), 0);
    assert_eq!(pair_pressure_band(39), 0);
    assert_eq!(pair_pressure_band(40), 1);
    assert_eq!(pair_pressure_band(69), 1);
    assert_eq!(pair_pressure_band(70), 2);
    assert_eq!(pair_pressure_band(84), 2);
    assert_eq!(pair_pressure_band(85), 3);
    assert_eq!(pair_pressure_band(100), 3);
}

#[test]
fn pressure_band_with_hysteresis_matches_pair_go_source() {
    // services/nvpair-job-scheduler/telemetry.go:124-138 (pressureWithHysteresis)

    // Rising thresholds (40/70/85): exactly at the threshold, band rises.
    assert_eq!(pair_pressure_band_with_previous(39, 0), 0);
    assert_eq!(pair_pressure_band_with_previous(40, 0), 1);
    assert_eq!(pair_pressure_band_with_previous(69, 1), 1);
    assert_eq!(pair_pressure_band_with_previous(70, 1), 2);
    assert_eq!(pair_pressure_band_with_previous(84, 2), 2);
    assert_eq!(pair_pressure_band_with_previous(85, 2), 3);

    // Falling thresholds (35/65/80): exactly at the threshold, band holds;
    // strictly below it, band falls.
    assert_eq!(pair_pressure_band_with_previous(35, 1), 1, "35 is not < down[1]=35, must hold");
    assert_eq!(pair_pressure_band_with_previous(34, 1), 0, "34 < down[1]=35, must fall");
    assert_eq!(pair_pressure_band_with_previous(65, 2), 2, "65 is not < down[2]=65, must hold");
    assert_eq!(pair_pressure_band_with_previous(64, 2), 1, "64 < down[2]=65, must fall");
    assert_eq!(pair_pressure_band_with_previous(80, 3), 3, "80 is not < down[3]=80, must hold");
    assert_eq!(pair_pressure_band_with_previous(79, 3), 2, "79 < down[3]=80, must fall");

    // Hysteresis band: a value that dropped below the *rising* threshold but
    // not below the *falling* one keeps the previous band (no flapping).
    assert_eq!(pair_pressure_band_with_previous(38, 1), 1);
    assert_eq!(pair_pressure_band_with_previous(68, 2), 2);
    assert_eq!(pair_pressure_band_with_previous(82, 3), 3);

    // A big jump in one sample can cross multiple bands at once, same as the
    // Go `for` loop.
    assert_eq!(pair_pressure_band_with_previous(95, 0), 3);
    assert_eq!(pair_pressure_band_with_previous(0, 3), 0);

    // previous_band out of 0..=3 is "unknown": fall back to the plain seed.
    assert_eq!(pair_pressure_band_with_previous(50, 4), pair_pressure_band(50));
    assert_eq!(pair_pressure_band_with_previous(50, 255), 1);
}

// ---------------------------------------------------------------------------
// admission()
// ---------------------------------------------------------------------------

#[test]
fn admission_accepts_by_default() {
    let telemetry = Telemetry::new(config(0.3), Box::new(FakeSampler::default()));
    assert_eq!(telemetry.admission(), Admission::Accept);
}

#[test]
fn admission_refuses_when_thermal_at_or_above_max() {
    let telemetry = Telemetry::new(config(0.3), Box::new(FakeSampler::default()));
    telemetry.set_external(ExternalSignals { thermal: ThermalStatus::Severe, ..Default::default() });
    match telemetry.admission() {
        Admission::Refuse(reason) => assert!(!reason.is_empty()),
        Admission::Accept => panic!("expected refusal at max thermal"),
    }

    telemetry.set_external(ExternalSignals { thermal: ThermalStatus::Critical, ..Default::default() });
    assert!(matches!(telemetry.admission(), Admission::Refuse(_)));
}

#[test]
fn admission_accepts_below_max_thermal() {
    let telemetry = Telemetry::new(config(0.3), Box::new(FakeSampler::default()));
    telemetry.set_external(ExternalSignals { thermal: ThermalStatus::Moderate, ..Default::default() });
    assert_eq!(telemetry.admission(), Admission::Accept);
}

#[test]
fn admission_refuses_low_battery_while_discharging() {
    let telemetry = Telemetry::new(config(0.3), Box::new(FakeSampler::default()));
    telemetry.set_external(ExternalSignals {
        battery_percent: Some(10),
        charging: Some(false),
        ..Default::default()
    });
    assert!(matches!(telemetry.admission(), Admission::Refuse(_)));
}

#[test]
fn admission_accepts_low_battery_while_charging() {
    let telemetry = Telemetry::new(config(0.3), Box::new(FakeSampler::default()));
    telemetry.set_external(ExternalSignals {
        battery_percent: Some(5),
        charging: Some(true),
        ..Default::default()
    });
    assert_eq!(telemetry.admission(), Admission::Accept);
}

#[test]
fn admission_accepts_battery_above_minimum_while_discharging() {
    let telemetry = Telemetry::new(config(0.3), Box::new(FakeSampler::default()));
    telemetry.set_external(ExternalSignals {
        battery_percent: Some(80),
        charging: Some(false),
        ..Default::default()
    });
    assert_eq!(telemetry.admission(), Admission::Accept);
}

// ---------------------------------------------------------------------------
// hostUuid / clusterUuid / msSince / sample_interval
// ---------------------------------------------------------------------------

#[test]
fn host_uuid_is_from_config_and_cluster_uuid_is_none() {
    let telemetry = Telemetry::new(config(0.3), Box::new(FakeSampler::default()));
    let info = telemetry.node_info();
    assert_eq!(info.host_uuid, "host-1234");
    assert_eq!(info.cluster_uuid, None);
}

#[test]
fn ms_since_grows_after_a_successful_sample() {
    let sampler = always_idle_sampler();
    let telemetry = Telemetry::new(config(0.3), Box::new(sampler));
    telemetry.tick();
    std::thread::sleep(Duration::from_millis(20));
    let ms = telemetry.node_info().ms_since;
    assert!(ms >= 15, "expected msSince to reflect elapsed wall time, got {ms}");
}

#[test]
fn ms_since_is_pinned_to_the_last_successful_sample_not_a_failed_one() {
    let sampler = Arc::new(FakeSampler::default());
    sampler.push_cpu_ok(0, 100);
    sampler.push_mem_ok(1024, 512);
    let telemetry = Telemetry::new(config(0.3), Box::new(sampler.clone()));

    telemetry.tick(); // succeeds, pins last_sample_at
    let ms_right_after_success = telemetry.node_info().ms_since;
    assert!(ms_right_after_success < 5, "got {ms_right_after_success}");

    std::thread::sleep(Duration::from_millis(30));
    sampler.push_cpu_err(); // this tick fails: msSince must NOT reset to ~0
    telemetry.tick();

    let ms_after_failed_tick = telemetry.node_info().ms_since;
    assert!(
        ms_after_failed_tick >= 25,
        "a failed sample must not move msSince back to ~0, got {ms_after_failed_tick}"
    );
}

#[test]
fn host_uuid_is_never_empty_even_if_config_leaves_it_blank() {
    let mut cfg = config(0.3);
    cfg.host_uuid = String::new();
    let telemetry = Telemetry::new(cfg, Box::new(FakeSampler::default()));
    assert!(!telemetry.node_info().host_uuid.is_empty());
}

#[test]
fn sample_interval_matches_config() {
    let cfg = config(0.3);
    let interval = cfg.sample_interval;
    let telemetry = Telemetry::new(cfg, Box::new(FakeSampler::default()));
    assert_eq!(TelemetrySource::sample_interval(&telemetry), interval);
}

// ---------------------------------------------------------------------------
// TelemetryConfig::default_for
// ---------------------------------------------------------------------------

#[test]
fn default_for_has_sensible_defaults() {
    let cfg = TelemetryConfig::default_for("host-9".to_string(), "CPU (llama.cpp)".to_string(), 123);
    assert_eq!(cfg.host_uuid, "host-9");
    assert_eq!(cfg.accelerator_name, "CPU (llama.cpp)");
    assert_eq!(cfg.model_budget_bytes, 123);
    assert_eq!(cfg.sample_interval, Duration::from_secs(2));
    assert!((cfg.ewma_alpha - 0.3).abs() < f64::EPSILON);
    assert_eq!(cfg.min_battery_percent_on_battery, 20);
    assert_eq!(cfg.max_thermal, ThermalStatus::Severe);
}

// ---------------------------------------------------------------------------
// alpha validation
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "ewma_alpha")]
fn new_panics_on_zero_alpha() {
    let _ = Telemetry::new(config(0.0), Box::new(FakeSampler::default()));
}

#[test]
#[should_panic(expected = "ewma_alpha")]
fn new_panics_on_alpha_above_one() {
    let _ = Telemetry::new(config(1.5), Box::new(FakeSampler::default()));
}

#[test]
fn new_accepts_alpha_of_exactly_one() {
    let _ = Telemetry::new(config(1.0), Box::new(FakeSampler::default()));
}

// ---------------------------------------------------------------------------
// thread-safety
// ---------------------------------------------------------------------------

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn telemetry_is_send_and_sync() {
    assert_send_sync::<Telemetry>();
}
