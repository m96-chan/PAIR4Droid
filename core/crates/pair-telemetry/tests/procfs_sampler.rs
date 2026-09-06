//! `ProcfsSampler` on hosts where `/proc/stat` is unreadable (Android SELinux
//! denies it to untrusted apps) must still produce CPU samples — ticket #22.

use pair_telemetry::{ProcfsSampler, Sampler, Telemetry, TelemetryConfig};
use std::path::PathBuf;

fn unreadable() -> PathBuf {
    PathBuf::from("/nonexistent/pair4droid/proc/stat")
}

#[test]
fn falls_back_to_process_cpu_time_when_proc_stat_is_unreadable() {
    let sampler = ProcfsSampler::with_stat_path(unreadable());

    let first = sampler.cpu().expect("fallback sample must succeed");
    assert!(first.cores >= 1, "cores from the online-cpu count, got {}", first.cores);

    // Burn a little CPU so utime advances, then sample again.
    let mut x: u64 = 1;
    for i in 0..20_000_000u64 {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(i);
    }
    std::hint::black_box(x);
    std::thread::sleep(std::time::Duration::from_millis(30));

    let second = sampler.cpu().unwrap();
    assert!(second.total > first.total, "wall-clock total must advance: {first:?} -> {second:?}");
    assert!(second.busy >= first.busy, "process cpu time must not go backwards: {first:?} -> {second:?}");
    assert!(
        second.busy - first.busy <= second.total - first.total,
        "busy delta may not exceed wall-clock × cores: {first:?} -> {second:?}"
    );
}

#[test]
fn telemetry_becomes_valid_with_the_fallback_sampler() {
    let telemetry = Telemetry::new(
        TelemetryConfig::default_for("uuid".into(), "test".into(), 0),
        Box::new(ProcfsSampler::with_stat_path(unreadable())),
    );
    telemetry.tick();
    assert!(!telemetry.node_info().telemetry_valid, "needs two cpu samples");
    std::thread::sleep(std::time::Duration::from_millis(20));
    telemetry.tick();
    let info = telemetry.node_info();
    assert!(info.telemetry_valid, "{info:?}");
    assert!(info.cpu.as_ref().unwrap().cores >= 1);
}
