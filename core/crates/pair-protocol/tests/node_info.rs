//! Ticket #3 — `GET :14318/v1/node-info` wire conformance.
//!
//! Every fixture is a JSON body copied verbatim from PAIR:
//!
//! | fixture | PAIR source |
//! | --- | --- |
//! | `gpus_empty.json` | `services/nvpair-manual-nodes/manager_test.go:419`, `services/nvpair-node-info/cluster_mtls_test.go:102`, `services/nvpair-node-scanner/cluster_identity_refresh_test.go:358` |
//! | `telemetry_absent.json` | `services/nvpair-node-info/observed_wiring_test.go:58` and `:149` |
//! | `host_uuid_only.json` | `services/tests/broker_management_test.go:106` (`hostUuid` = the test's `learned-host-uuid`), `services/tests/ghost_node_test.go:64` |
//! | `cluster_uuid_empty.json` | `services/nvpair-node-scanner/cluster_identity_refresh_test.go:118` |
//! | `cluster_uuid_principal.json` | `services/nvpair-node-scanner/cluster_identity_refresh_test.go:163` |
//! | `cluster_uuid_absent.json` | `services/nvpair-node-scanner/cluster_identity_refresh_test.go:189` |
//! | `cluster_uuid_empty_telemetry.json` | `services/nvpair-node-scanner/cluster_identity_refresh_test.go:369` |
//! | `manual_nodes_sample_info.json` | `services/nvpair-manual-nodes/manager_test.go:265-278` marshalled through `:108` |
//! | `readme_canonical.json` | `services/nvpair-node-info/README.md:53-75` |

mod common;

use common::*;
use pair_protocol::node_info::{CpuInfo, GpuInfo, MemoryInfo, NodeInfoResponse};
use serde_json::json;

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

#[test]
fn minimum_viable_body_decodes() {
    // `services/nvpair-manual-nodes/manager_test.go:419`
    let n: NodeInfoResponse = decode("node_info/gpus_empty.json");
    assert!(n.gpus.is_empty());
    assert!(!n.telemetry_valid);
    assert_eq!(n.ms_since, 0);
    assert_eq!(n.host_uuid, "");
    assert_eq!(n.cluster_uuid, None);
}

#[test]
fn host_uuid_is_read() {
    // `services/tests/broker_management_test.go:106`
    let n: NodeInfoResponse = decode("node_info/host_uuid_only.json");
    assert_eq!(n.host_uuid, "learned-host-uuid");
}

/// Mirrors `TestNodeInfoResponseDistinguishesAbsentFromEmpty`
/// (`services/nvpair-node-scanner/cluster_identity_refresh_test.go:356-382`):
/// absent means unknown, present-and-empty means unclustered.
#[test]
fn cluster_uuid_has_three_states() {
    let absent: NodeInfoResponse = decode("node_info/cluster_uuid_absent.json");
    assert_eq!(absent.cluster_uuid, None);
    assert!(!absent.telemetry_valid);
    assert_eq!(absent.ms_since, 0);

    let empty: NodeInfoResponse = decode("node_info/cluster_uuid_empty.json");
    assert_eq!(empty.cluster_uuid.as_deref(), Some(""));

    let principal: NodeInfoResponse = decode("node_info/cluster_uuid_principal.json");
    assert_eq!(principal.cluster_uuid.as_deref(), Some("their-principal"));

    let telemetry: NodeInfoResponse = decode("node_info/cluster_uuid_empty_telemetry.json");
    assert_eq!(telemetry.cluster_uuid.as_deref(), Some(""));
    assert!(telemetry.telemetry_valid);
    assert_eq!(telemetry.ms_since, 137);
}

#[test]
fn full_body_decodes() {
    let n: NodeInfoResponse = decode("node_info/manual_nodes_sample_info.json");
    assert_eq!(n.gpus.len(), 1);
    assert_eq!(n.gpus[0].name, "RTX 6000");
    assert_eq!(n.gpus[0].vram_bytes, 48 << 30);
    assert_eq!(n.gpus[0].vram_used_bytes, 12 << 30);
    assert_eq!(n.gpus[0].utilization_percent, 42);
    let cpu = n.cpu.as_ref().expect("cpu");
    assert_eq!((cpu.name.as_str(), cpu.cores, cpu.utilization_percent), ("Threadripper", 64, 7));
    let mem = n.memory.as_ref().expect("memory");
    assert_eq!((mem.total_bytes, mem.used_bytes), (128 << 30, 32 << 30));
    assert!(n.telemetry_valid);
    assert_eq!(n.ms_since, 137);
}

#[test]
fn readme_canonical_body_decodes() {
    let n: NodeInfoResponse = decode("node_info/readme_canonical.json");
    assert_eq!(n.gpus[0].name, "NVIDIA GeForce RTX 3080");
    assert_eq!(n.gpus[0].vram_bytes, 10_737_418_240);
    assert_eq!(n.host_uuid, "8661676a-0d1c-4bd3-ac5e-4d370e6f1a9c");
    assert_eq!(n.cluster_uuid.as_deref(), Some(""));
    assert_eq!(n.memory.unwrap().used_bytes, 12_884_901_888);
}

// ---------------------------------------------------------------------------
// Round-trip
// ---------------------------------------------------------------------------

/// The strongest proof available: PAIR's own `json.Marshal` output, reproduced
/// byte for byte. Pins key names *and* field order against
/// `services/nvpair-manual-nodes/manager.go:69-81`.
#[test]
fn full_body_round_trips_byte_for_byte() {
    assert_roundtrip_bytes::<NodeInfoResponse>("node_info/manual_nodes_sample_info.json");
}

#[test]
fn readme_canonical_body_round_trips() {
    assert_roundtrip_exact::<NodeInfoResponse>("node_info/readme_canonical.json");
}

/// `GPUs`, `telemetryValid` and `msSince` carry no `omitempty` in
/// `services/nvpair-node-info/main.go:74-79`, so they are the only keys we may
/// add to a fixture that omitted them. Everything else must stay omitted.
#[test]
fn sparse_bodies_round_trip_adding_only_non_omitempty_keys() {
    const ALWAYS: &[&str] = &["GPUs", "telemetryValid", "msSince"];
    for f in [
        "node_info/gpus_empty.json",
        "node_info/telemetry_absent.json",
        "node_info/host_uuid_only.json",
        "node_info/cluster_uuid_absent.json",
        "node_info/cluster_uuid_empty.json",
        "node_info/cluster_uuid_principal.json",
        "node_info/cluster_uuid_empty_telemetry.json",
    ] {
        assert_roundtrip_superset::<NodeInfoResponse>(f, ALWAYS);
    }
}

// ---------------------------------------------------------------------------
// omitempty parity
// ---------------------------------------------------------------------------

/// Go emits `{"GPUs":[],"telemetryValid":false,"msSince":0}` for a zero
/// `NodeInfoResponse` — `cpu`, `memory`, `hostUuid`, `clusterUuid` are all
/// pointer/`omitempty` (`services/nvpair-node-info/main.go:74-101`).
#[test]
fn zero_value_encodes_like_go() {
    let s = serde_json::to_string(&NodeInfoResponse::default()).unwrap();
    assert_eq!(s, r#"{"GPUs":[],"telemetryValid":false,"msSince":0}"#);
}

/// Mirrors `TestBuildResponseOmitsZero`
/// (`services/nvpair-node-info/stats_test.go:400-425`): a GPU with no stats
/// match must drop the dynamic keys, not report literal zeros.
#[test]
fn gpu_dynamic_fields_are_omitted_when_zero() {
    let n = NodeInfoResponse {
        gpus: vec![GpuInfo { name: "GPU 0".into(), vram_bytes: 4 << 30, ..Default::default() }],
        ..Default::default()
    };
    let v = serde_json::to_value(&n).unwrap();
    let gpu = &v["GPUs"][0];
    assert!(gpu.get("name").is_some(), "name has no omitempty: {gpu}");
    assert!(gpu.get("vram_bytes").is_some(), "{gpu}");
    assert!(gpu.get("vram_used_bytes").is_none(), "{gpu}");
    assert!(gpu.get("utilization_percent").is_none(), "{gpu}");
}

/// `GPUs[].name` has **no** `omitempty` (`services/nvpair-node-info/main.go:31`).
#[test]
fn gpu_name_is_always_emitted() {
    let v = serde_json::to_value(GpuInfo::default()).unwrap();
    assert_eq!(v, json!({"name": ""}));
}

/// Mirrors `TestBuildResponseCPUMemoryMatrix`
/// (`services/nvpair-node-info/stats_test.go:427-545`).
#[test]
fn cpu_and_memory_objects_are_omitted_whole() {
    let cpu = CpuInfo { name: "Intel Core i9-13900K".into(), cores: 24, utilization_percent: 42 };
    let mem = MemoryInfo { total_bytes: 32 << 30, used_bytes: 10 << 30 };

    for (cpu_in, mem_in, want_cpu, want_mem) in [
        (Some(cpu.clone()), Some(mem.clone()), true, true),
        (Some(cpu.clone()), None, true, false),
        (None, Some(mem.clone()), false, true),
        (None, None, false, false),
    ] {
        let n = NodeInfoResponse { cpu: cpu_in, memory: mem_in, ..Default::default() };
        let v = serde_json::to_value(&n).unwrap();
        assert_eq!(v.get("cpu").is_some(), want_cpu, "cpu key in {v}");
        assert_eq!(v.get("memory").is_some(), want_mem, "memory key in {v}");
    }
}

#[test]
fn cpu_and_memory_inner_fields_are_omitted_when_zero() {
    let v = serde_json::to_value(NodeInfoResponse {
        cpu: Some(CpuInfo::default()),
        memory: Some(MemoryInfo::default()),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(v["cpu"], json!({}));
    assert_eq!(v["memory"], json!({}));
}

#[test]
fn host_uuid_and_cluster_uuid_are_omitted_when_unset() {
    let v = serde_json::to_value(NodeInfoResponse::default()).unwrap();
    assert!(v.get("hostUuid").is_none(), "{v}");
    assert!(v.get("clusterUuid").is_none(), "{v}");

    // present-and-empty must survive encoding — that is "unclustered", not "unknown".
    let v =
        serde_json::to_value(NodeInfoResponse { cluster_uuid: Some(String::new()), ..Default::default() })
            .unwrap();
    assert_eq!(v["clusterUuid"], json!(""));
}

// ---------------------------------------------------------------------------
// Leniency
// ---------------------------------------------------------------------------

/// PAIR decodes with `encoding/json`, which ignores unknown keys; so must we.
/// `inference_hardware_ids` is a real example — the desktop reads it, the Go
/// service never emits it (`docs/pair-contract.md` §2.2).
#[test]
fn unknown_fields_are_ignored() {
    let body = r#"{"GPUs":[{"name":"Adreno 750","utilization_percent":37,"driver":"x"}],
        "telemetryValid":true,"msSince":80,"hostUuid":"u",
        "inference_hardware_ids":["a"],"cpu":{"name":"c","threads":8},
        "memory":{"total_bytes":1,"swap_bytes":2},"somethingNew":{"deep":[1,2]}}"#;
    let n: NodeInfoResponse = serde_json::from_str(body).expect("unknown fields must be ignored");
    assert_eq!(n.gpus[0].utilization_percent, 37);
    assert_eq!(n.cpu.unwrap().name, "c");
    assert_eq!(n.memory.unwrap().total_bytes, 1);
    assert_eq!(n.host_uuid, "u");
}

/// A nil Go slice marshals to `null`, and `services/nvpair-node-info/main.go:75`
/// has no `omitempty` on `GPUs` — so `"GPUs":null` is on the wire. It must
/// decode as an empty list, not an error.
#[test]
fn null_gpus_decodes_as_empty() {
    let n: NodeInfoResponse = serde_json::from_str(r#"{"GPUs":null}"#).expect("null GPUs");
    assert!(n.gpus.is_empty());
    let n: NodeInfoResponse =
        serde_json::from_str(r#"{"GPUs":[],"cpu":null,"memory":null,"clusterUuid":null}"#)
            .expect("null objects");
    assert!(n.cpu.is_none() && n.memory.is_none() && n.cluster_uuid.is_none());
}

/// An entirely empty object is what a node too old to report anything sends.
#[test]
fn empty_object_decodes_to_default() {
    let n: NodeInfoResponse = serde_json::from_str("{}").expect("{}");
    assert_eq!(n, NodeInfoResponse::default());
}

// ---------------------------------------------------------------------------
// Key names
// ---------------------------------------------------------------------------

/// The design invariant from CLAUDE.md: PascalCase `GPUs`, camelCase
/// `telemetryValid` / `msSince` / `hostUuid` / `clusterUuid`, snake_case
/// everywhere else — in one object.
#[test]
fn key_names_match_pair_byte_for_byte() {
    let n = NodeInfoResponse {
        gpus: vec![GpuInfo { name: "g".into(), vram_bytes: 1, vram_used_bytes: 2, utilization_percent: 3 }],
        cpu: Some(CpuInfo { name: "c".into(), cores: 4, utilization_percent: 5 }),
        memory: Some(MemoryInfo { total_bytes: 6, used_bytes: 7 }),
        telemetry_valid: true,
        ms_since: 8,
        host_uuid: "h".into(),
        cluster_uuid: Some("c".into()),
    };
    assert_eq!(
        serde_json::to_string(&n).unwrap(),
        r#"{"GPUs":[{"name":"g","vram_bytes":1,"vram_used_bytes":2,"utilization_percent":3}],"cpu":{"name":"c","cores":4,"utilization_percent":5},"memory":{"total_bytes":6,"used_bytes":7},"telemetryValid":true,"msSince":8,"hostUuid":"h","clusterUuid":"c"}"#
    );
}
