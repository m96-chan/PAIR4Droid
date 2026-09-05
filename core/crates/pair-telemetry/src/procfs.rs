//! Pure parsers over `/proc/stat`, `/proc/meminfo` and `/proc/cpuinfo` text.
//!
//! Kept free of any filesystem I/O so they can be unit-tested against literal
//! fixture strings (including Android-shaped ones). [`super::ProcfsSampler`]
//! is the thin I/O wrapper that reads the real files and calls these.

use crate::{RawCpuSample, RawMemSample};
use std::io;

/// Parses the aggregate `cpu` line (and counts the per-core `cpuN` lines) of
/// `/proc/stat`.
///
/// Field order per `man proc` / Linux `kernel/sched/cputime.c`:
/// `user nice system idle iowait irq softirq steal guest guest_nice`.
///
/// `busy = user + nice + system + irq + softirq + steal` (excludes `idle`,
/// `iowait`, and the two `guest` fields, which are already included in
/// `user`/`nice` on Linux and would double-count if added again).
/// `total = busy + idle + iowait`.
///
/// `model_name` is always empty here — `/proc/stat` carries no such field,
/// see [`parse_cpuinfo_model_name`].
pub fn parse_stat(s: &str) -> io::Result<RawCpuSample> {
    let mut cores: u32 = 0;
    let mut aggregate: Option<(u64, u64)> = None;

    for line in s.lines() {
        let Some(rest) = line.strip_prefix("cpu") else { continue };
        match rest.chars().next() {
            // "cpu  <fields...>" — the aggregate line.
            Some(c) if c.is_whitespace() => {
                if aggregate.is_none() {
                    aggregate = Some(parse_cpu_fields(rest)?);
                }
            }
            // "cpu0 <fields...>", "cpu1 ...", ... — one per core.
            Some(c) if c.is_ascii_digit() => cores += 1,
            _ => {}
        }
    }

    let (busy, total) = aggregate
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no aggregate 'cpu' line in /proc/stat"))?;

    Ok(RawCpuSample { busy, total, cores, model_name: String::new() })
}

fn parse_cpu_fields(rest: &str) -> io::Result<(u64, u64)> {
    let fields: Vec<u64> = rest
        .split_whitespace()
        .map(|f| {
            f.parse::<u64>().map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, format!("bad /proc/stat field: {f:?}"))
            })
        })
        .collect::<io::Result<_>>()?;

    // user nice system idle iowait irq softirq [steal [guest [guest_nice]]]
    if fields.len() < 7 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "short aggregate 'cpu' line in /proc/stat"));
    }
    let user = fields[0];
    let nice = fields[1];
    let system = fields[2];
    let idle = fields[3];
    let iowait = fields[4];
    let irq = fields[5];
    let softirq = fields[6];
    let steal = fields.get(7).copied().unwrap_or(0);

    let busy = user + nice + system + irq + softirq + steal;
    let total = busy + idle + iowait;
    Ok((busy, total))
}

/// Best-effort CPU model name. Android/ARM kernels expose a `Hardware` line;
/// x86 kernels expose `model name`. Never fails — returns `""` when neither
/// is present, matching the "best effort" acceptance criterion.
pub fn parse_cpuinfo_model_name(s: &str) -> String {
    let mut model_name: Option<&str> = None;
    let mut hardware: Option<&str> = None;

    for line in s.lines() {
        let Some((key, value)) = line.split_once(':') else { continue };
        let key = key.trim();
        let value = value.trim();
        if value.is_empty() {
            continue;
        }
        if model_name.is_none() && key.eq_ignore_ascii_case("model name") {
            model_name = Some(value);
        } else if hardware.is_none() && key.eq_ignore_ascii_case("Hardware") {
            hardware = Some(value);
        }
    }

    model_name.or(hardware).unwrap_or("").to_string()
}

/// Parses `MemTotal`/`MemAvailable` out of `/proc/meminfo` (values are in kB,
/// converted here to bytes).
pub fn parse_meminfo(s: &str) -> io::Result<RawMemSample> {
    let mut total_bytes = None;
    let mut available_bytes = None;

    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total_bytes = Some(parse_kb_value(rest)?);
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            available_bytes = Some(parse_kb_value(rest)?);
        }
    }

    let total_bytes = total_bytes
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing MemTotal in /proc/meminfo"))?;
    let available_bytes = available_bytes
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing MemAvailable in /proc/meminfo"))?;

    Ok(RawMemSample { total_bytes, available_bytes })
}

fn parse_kb_value(rest: &str) -> io::Result<u64> {
    let digits = rest.trim().trim_end_matches("kB").trim();
    let kb: u64 = digits.parse().map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidData, format!("bad /proc/meminfo value: {rest:?}"))
    })?;
    Ok(kb * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DESKTOP_STAT: &str = "\
cpu  9734 0 3225 352699 2197 0 136 4 0 0
cpu0 2525 0 1065 87693 649 0 68 1 0 0
cpu1 2429 0 680 88351 501 0 41 1 0 0
cpu2 2245 0 699 88585 478 0 10 0 0 0
cpu3 2533 0 780 88068 567 0 16 1 0 0
intr 12345 0 0 0
ctxt 6789
btime 1700000000
processes 4242
";

    // A plausible Android /proc/stat for an 8-core big.LITTLE SoC.
    const ANDROID_STAT_8_CORE: &str = "\
cpu  123456 30 45678 9876543 1234 0 567 12 0 0
cpu0 15432 4 5709 1234567 154 0 70 1 0 0
cpu1 15431 4 5709 1234567 154 0 70 1 0 0
cpu2 15431 3 5709 1234567 154 0 70 1 0 0
cpu3 15431 3 5709 1234567 154 0 70 1 0 0
cpu4 15432 4 5710 1234567 154 0 70 2 0 0
cpu5 15432 4 5710 1234567 155 0 70 2 0 0
cpu6 15432 4 5711 1234567 155 0 74 2 0 0
cpu7 15435 4 5711 1234568 154 0 73 2 0 0
intr 555 0 0
ctxt 999
btime 1700000000
processes 555
";

    const ANDROID_CPUINFO: &str = "\
processor\t: 0
BogoMIPS\t: 38.40
Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32
CPU implementer\t: 0x51
CPU architecture: 8
CPU variant\t: 0xd
CPU part\t: 0x804
CPU revision\t: 0

Hardware\t: Qualcomm Technologies, Inc SM8550
Revision\t: 0000
Serial\t: 0000000000000000
";

    const DESKTOP_CPUINFO: &str = "\
processor\t: 0
vendor_id\t: GenuineIntel
cpu family\t: 6
model\t\t: 85
model name\t: Intel(R) Xeon(R) Processor @ 2.80GHz
stepping\t: 7
";

    const DESKTOP_MEMINFO: &str = "\
MemTotal:       16461028 kB
MemFree:        14248808 kB
MemAvailable:   15698984 kB
Buffers:           33832 kB
Cached:          1653092 kB
";

    #[test]
    fn parses_desktop_stat_busy_and_total() {
        let sample = parse_stat(DESKTOP_STAT).unwrap();
        // busy = user+nice+system+irq+softirq+steal = 9734+0+3225+0+136+4 = 13099
        // total = busy + idle + iowait = 13099 + 352699 + 2197 = 367995
        assert_eq!(sample.busy, 13_099);
        assert_eq!(sample.total, 367_995);
        assert_eq!(sample.cores, 4);
        assert!(sample.model_name.is_empty());
    }

    #[test]
    fn parses_android_8_core_stat() {
        let sample = parse_stat(ANDROID_STAT_8_CORE).unwrap();
        // busy = 123456+30+45678+0+567+12 = 169743
        assert_eq!(sample.busy, 169_743);
        assert_eq!(sample.total, 169_743 + 9_876_543 + 1_234);
        assert_eq!(sample.cores, 8);
    }

    #[test]
    fn parse_stat_rejects_missing_aggregate_line() {
        let err = parse_stat("intr 0\nctxt 0\n").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn parse_stat_rejects_short_aggregate_line() {
        let err = parse_stat("cpu  1 2 3\n").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn cpuinfo_model_name_prefers_model_name_field() {
        assert_eq!(parse_cpuinfo_model_name(DESKTOP_CPUINFO), "Intel(R) Xeon(R) Processor @ 2.80GHz");
    }

    #[test]
    fn cpuinfo_model_name_falls_back_to_hardware_field() {
        assert_eq!(parse_cpuinfo_model_name(ANDROID_CPUINFO), "Qualcomm Technologies, Inc SM8550");
    }

    #[test]
    fn cpuinfo_model_name_is_empty_when_neither_field_present() {
        assert_eq!(parse_cpuinfo_model_name("processor: 0\nBogoMIPS: 1.0\n"), "");
    }

    #[test]
    fn parses_meminfo_kb_to_bytes() {
        let sample = parse_meminfo(DESKTOP_MEMINFO).unwrap();
        assert_eq!(sample.total_bytes, 16_461_028 * 1024);
        assert_eq!(sample.available_bytes, 15_698_984 * 1024);
    }

    #[test]
    fn parse_meminfo_rejects_missing_fields() {
        let err = parse_meminfo("MemFree: 100 kB\n").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }
}
