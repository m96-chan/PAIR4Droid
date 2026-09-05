//! Model catalogue for a *stopped* node: a plain directory scan.
//!
//! Once the node runs, `Engine::list_models` is the truth (it reads the GGUF
//! headers). Before that the Models screen still needs names/sizes/quantisation,
//! and the phone must not pay for opening every GGUF just to draw a list — so the
//! metadata is guessed from the file name, with the same heuristics
//! `pair_engine::llama::parse_name_metadata` uses (that function is behind the
//! `llama` feature, which this crate must work without).

use crate::ModelInfo;
use std::path::Path;

/// Every `*.gguf` file directly under `dir`, sorted by name. A missing or
/// unreadable directory is simply an empty catalogue.
pub(crate) fn scan_gguf(dir: &Path) -> Vec<ModelInfo> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };

    let mut models: Vec<ModelInfo> = entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if !path.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")) {
                return None;
            }
            let metadata = entry.metadata().ok()?;
            if !metadata.is_file() {
                return None;
            }
            let stem = path.file_stem()?.to_string_lossy().to_string();
            let (family, parameter_size, quant) = parse_name_metadata(&stem);
            Some(ModelInfo {
                name: stem,
                path: path.to_string_lossy().to_string(),
                size_bytes: metadata.len(),
                family,
                parameter_size,
                quant,
                context_length: 0,
            })
        })
        .collect();

    models.sort_by(|a, b| a.name.cmp(&b.name));
    models
}

/// `(family, parameter_size, quant)` guessed from a stem such as
/// `qwen2.5-1.5b-instruct-q4_k_m`.
fn parse_name_metadata(stem: &str) -> (String, String, String) {
    let family = stem.split(['-', '_']).find(|p| !p.is_empty()).unwrap_or(stem).to_ascii_lowercase();

    let mut parameter_size = String::new();
    let mut quant = String::new();
    // Quantisation suffixes are written with underscores (`q4_k_m`), so split on
    // `-` only and keep the tail intact.
    for (i, part) in stem.split('-').enumerate() {
        if i == 0 {
            continue;
        }
        if quant.is_empty() && looks_like_quant(part) {
            quant = part.to_ascii_uppercase();
        }
        if parameter_size.is_empty() && looks_like_param_size(part) {
            parameter_size = part.to_ascii_uppercase();
        }
    }
    (family, parameter_size, quant)
}

fn looks_like_quant(part: &str) -> bool {
    let part = part.to_ascii_lowercase();
    if matches!(part.as_str(), "f16" | "f32" | "bf16" | "fp16" | "fp32") {
        return true;
    }
    match part.strip_prefix("iq").or_else(|| part.strip_prefix('q')) {
        Some(rest) => {
            rest.starts_with(|c: char| c.is_ascii_digit())
                && rest.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        }
        None => false,
    }
}

fn looks_like_param_size(part: &str) -> bool {
    let part = part.to_ascii_lowercase();
    let Some(head) = part.strip_suffix('b').or_else(|| part.strip_suffix('m')) else {
        return false;
    };
    !head.is_empty()
        && head.chars().all(|c| c.is_ascii_digit() || c == '.')
        && head.chars().any(|c| c.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guesses_metadata_from_the_file_stem() {
        assert_eq!(
            parse_name_metadata("qwen2.5-1.5b-instruct-q4_k_m"),
            ("qwen2.5".to_string(), "1.5B".to_string(), "Q4_K_M".to_string())
        );
        assert_eq!(
            parse_name_metadata("llama-3.2-3b-f16"),
            ("llama".to_string(), "3B".to_string(), "F16".to_string())
        );
        assert_eq!(parse_name_metadata("model").2, "");
    }
}
