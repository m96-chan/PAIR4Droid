//! Small serde helpers shared by the three lanes.
//!
//! The recurring problem: Go marshals a `nil` slice or map as `null`, not `[]`.
//! PAIR does this on the wire in several places we have to parse —
//! `{"GPUs":null}` (`services/nvpair-node-info/main.go:75`, no `omitempty`),
//! `{"models":null}` (`services/ollama-proxy/failover_test.go:387`) and
//! `{"object":"list","data":null}` (`services/lmstudio-proxy/failover_test.go:387`).
//! `#[serde(default)]` alone does not cover it: it fires on a *missing* key, not
//! on an explicit `null`.

use serde::{Deserialize, Deserializer};

/// Deserialize `T`, mapping an explicit JSON `null` to `T::default()`.
///
/// Pair with `#[serde(default)]` so a missing key is handled too:
/// `#[serde(default, deserialize_with = "crate::serde_util::null_to_default")]`.
pub(crate) fn null_to_default<'de, D, T>(de: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    Ok(Option::<T>::deserialize(de)?.unwrap_or_default())
}

pub(crate) fn is_zero_u32(v: &u32) -> bool {
    *v == 0
}

pub(crate) fn is_zero_u64(v: &u64) -> bool {
    *v == 0
}

pub(crate) fn default_true() -> bool {
    true
}
