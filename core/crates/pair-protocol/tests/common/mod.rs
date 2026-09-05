//! Fixture loading + the two round-trip shapes every wire test uses.
//!
//! Every file under `tests/fixtures/` is a JSON body copied **verbatim** out of
//! NVIDIA Personal AI Router's own tests/fakes (or replayed through the exact
//! `json.Marshal` call those fakes make). The provenance of each file is listed
//! in the test module that consumes it, citing `services/<path>:<line>`.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::Debug;
use std::path::PathBuf;

pub fn fixture_path(rel: &str) -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures")).join(rel)
}

/// Raw bytes of a fixture, trailing newline trimmed (files are stored with one).
pub fn raw(rel: &str) -> String {
    let p = fixture_path(rel);
    std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("read fixture {}: {e}", p.display()))
        .trim_end_matches('\n')
        .to_string()
}

pub fn json(rel: &str) -> Value {
    serde_json::from_str(&raw(rel)).unwrap_or_else(|e| panic!("fixture {rel} is not JSON: {e}"))
}

pub fn decode<T: for<'de> Deserialize<'de>>(rel: &str) -> T {
    serde_json::from_str(&raw(rel)).unwrap_or_else(|e| panic!("decode fixture {rel}: {e}"))
}

/// decode → encode must reproduce the fixture *exactly* (as `Value`, so key
/// order does not matter). Use for fixtures that carry every key our encoder
/// emits.
pub fn assert_roundtrip_exact<T>(rel: &str)
where
    T: Serialize + for<'de> Deserialize<'de> + Debug,
{
    let want = json(rel);
    let typed: T = decode(rel);
    let got = serde_json::to_value(&typed).expect("encode");
    assert_eq!(got, want, "round-trip of {rel} diverged\ntyped: {typed:?}");
}

/// decode → encode must preserve every key the fixture carries, and may only
/// *add* keys listed in `may_add` — the fields the Go struct emits
/// unconditionally (no `omitempty`) but the fixture happened to omit.
pub fn assert_roundtrip_superset<T>(rel: &str, may_add: &[&str])
where
    T: Serialize + for<'de> Deserialize<'de> + Debug,
{
    let want = json(rel);
    let typed: T = decode(rel);
    let got = serde_json::to_value(&typed).expect("encode");
    assert_superset(&got, &want, rel);

    let added = added_keys(&got, &want);
    for key in &added {
        assert!(
            may_add.contains(&key.as_str()),
            "{rel}: our encoder added key {key:?} which the fixture omits; \
             allowed additions are {may_add:?} (added: {added:?})"
        );
    }

    // Re-decoding our own output must land on the same value: encoding is stable.
    let again: T = serde_json::from_value(got).expect("re-decode our own output");
    assert_eq!(
        serde_json::to_value(&again).unwrap(),
        serde_json::to_value(&typed).unwrap(),
        "{rel}: encode is not idempotent"
    );
}

/// Every key/element in `expected` must appear in `actual` with an equal value.
pub fn assert_superset(actual: &Value, expected: &Value, ctx: &str) {
    fn walk(actual: &Value, expected: &Value, path: &str, ctx: &str) {
        match (actual, expected) {
            (Value::Object(a), Value::Object(e)) => {
                for (k, ev) in e {
                    let av = a.get(k).unwrap_or_else(|| panic!("{ctx}: missing key {path}/{k}"));
                    walk(av, ev, &format!("{path}/{k}"), ctx);
                }
            }
            (Value::Array(a), Value::Array(e)) => {
                assert_eq!(a.len(), e.len(), "{ctx}: array length at {path}");
                for (i, ev) in e.iter().enumerate() {
                    walk(&a[i], ev, &format!("{path}[{i}]"), ctx);
                }
            }
            (a, e) => assert_eq!(a, e, "{ctx}: value at {path}"),
        }
    }
    walk(actual, expected, "", ctx);
}

/// Top-level (and nested-object) keys present in `actual` but not `expected`.
pub fn added_keys(actual: &Value, expected: &Value) -> Vec<String> {
    let mut out = Vec::new();
    fn walk(actual: &Value, expected: &Value, path: &str, out: &mut Vec<String>) {
        match (actual, expected) {
            (Value::Object(a), Value::Object(e)) => {
                for (k, av) in a {
                    match e.get(k) {
                        None => out.push(k.clone()),
                        Some(ev) => walk(av, ev, &format!("{path}/{k}"), out),
                    }
                }
            }
            (Value::Array(a), Value::Array(e)) if a.len() == e.len() => {
                for (i, av) in a.iter().enumerate() {
                    walk(av, &e[i], &format!("{path}[{i}]"), out);
                }
            }
            _ => {}
        }
    }
    walk(actual, expected, "", &mut out);
    out.sort();
    out.dedup();
    out
}

/// decode → encode must reproduce the fixture **byte for byte**, which also
/// pins our struct field order against Go's declaration order.
pub fn assert_roundtrip_bytes<T>(rel: &str)
where
    T: Serialize + for<'de> Deserialize<'de> + Debug,
{
    let want = raw(rel);
    let typed: T = decode(rel);
    let got = serde_json::to_string(&typed).expect("encode");
    assert_eq!(got, want, "byte round-trip of {rel} diverged");
}

/// Top-level JSON object keys of a serialised value (sorted — `serde_json::Value`
/// is a `BTreeMap`), i.e. the key *set* our encoder emits.
pub fn keys_of<T: Serialize>(v: &T) -> Vec<String> {
    match serde_json::to_value(v).expect("encode") {
        Value::Object(map) => map.keys().cloned().collect(),
        other => panic!("not a JSON object: {other}"),
    }
}
