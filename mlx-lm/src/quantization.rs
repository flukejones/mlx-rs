//! `quantization_config` parsing for MLX checkpoints.
//!
//! mlx-community + lmstudio-community publish two checkpoint shapes:
//!
//! - **Uniform:** `{group_size, bits, mode}`. Every quantisable
//!   parameter uses the same body settings.
//! - **Per-tensor overrides:** body `{group_size, bits, mode}` plus
//!   additional path-keyed entries like
//!   `"language_model.model.layers.0.mlp.gate": {group_size, bits}`.
//!   The router + shared-expert-gate tensors on Qwen3.6-MoE
//!   checkpoints (both q4 and q8) ship the gates at 8-bit even when
//!   the body is 4-bit.
//!
//! Loaders consult [`QuantizationConfig::for_path`] when building each
//! quantisable slot so overrides land on the right `(group_size, bits)`
//! and the safetensors `.weight`/`.scales`/`.biases` triple binds
//! cleanly into the param walk.

use std::collections::HashMap;

use serde::Deserialize;

const DEFAULT_QUANT_MODE: &str = "affine";

/// Body quantisation + optional per-key overrides.
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
    pub mode: String,
    /// Per-key overrides keyed by the raw safetensors prefix (before
    /// sanitisation), e.g.
    /// `language_model.model.layers.0.mlp.gate`.
    pub overrides: HashMap<String, (i32, i32)>,
}

impl QuantizationConfig {
    /// Body `(group_size, bits)` after override lookup. Caller passes
    /// the **sanitised** param path (e.g. `model.layers.0.mlp.gate`);
    /// `raw_prefix` is the corresponding HF safetensors key prefix.
    /// Returns the body defaults when no override matches.
    pub fn for_path(&self, raw_prefix: &str) -> (i32, i32) {
        self.overrides
            .get(raw_prefix)
            .copied()
            .unwrap_or((self.group_size, self.bits))
    }
}

/// Prefer `quantization`; fall back to legacy `quantization_config`.
pub fn resolve_quantization<'a>(
    primary: &'a Option<QuantizationConfig>,
    legacy: &'a Option<QuantizationConfig>,
) -> Option<&'a QuantizationConfig> {
    primary.as_ref().or(legacy.as_ref())
}

// ─── serde plumbing ───────────────────────────────────────────────

/// Intermediate shape: top-level body knobs flatten into a catch-all
/// `serde_json::Value` map so the per-key override entries fall out
/// of the residual.
#[derive(Deserialize)]
struct Raw {
    group_size: i32,
    bits: i32,
    #[serde(default = "default_quant_mode")]
    mode: String,
    /// Per-tensor overrides — every entry that isn't `group_size`,
    /// `bits`, or `mode` lands here.
    #[serde(flatten)]
    extras: HashMap<String, serde_json::Value>,
}

fn default_quant_mode() -> String {
    DEFAULT_QUANT_MODE.to_owned()
}

#[derive(Deserialize)]
struct OverrideValue {
    group_size: i32,
    bits: i32,
}

impl<'de> Deserialize<'de> for QuantizationConfig {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw = Raw::deserialize(d)?;
        let mut overrides = HashMap::with_capacity(raw.extras.len());
        for (k, v) in raw.extras {
            // Quietly skip unrecognised non-override keys
            // (transformers may add advisory fields here).
            if let Ok(ov) = serde_json::from_value::<OverrideValue>(v) {
                overrides.insert(k, (ov.group_size, ov.bits));
            }
        }
        Ok(Self {
            group_size: raw.group_size,
            bits: raw.bits,
            mode: raw.mode,
            overrides,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    use super::*;

    #[test]
    fn parses_uniform_config() {
        let q: QuantizationConfig = serde_json::from_str(
            r#"{"group_size": 64, "bits": 8, "mode": "affine"}"#,
        )
        .unwrap();
        assert_eq!(q.group_size, 64);
        assert_eq!(q.bits, 8);
        assert_eq!(q.mode, "affine");
        assert!(q.overrides.is_empty());
        assert_eq!(q.for_path("anything"), (64, 8));
    }

    #[test]
    fn parses_per_tensor_overrides() {
        let q: QuantizationConfig = serde_json::from_str(
            r#"{
                "group_size": 64,
                "bits": 4,
                "mode": "affine",
                "language_model.model.layers.0.mlp.gate": {"group_size": 64, "bits": 8},
                "language_model.model.layers.0.mlp.shared_expert_gate": {"group_size": 64, "bits": 8}
            }"#,
        )
        .unwrap();
        assert_eq!(q.bits, 4);
        assert_eq!(q.overrides.len(), 2);
        assert_eq!(
            q.for_path("language_model.model.layers.0.mlp.gate"),
            (64, 8)
        );
        assert_eq!(
            q.for_path("language_model.model.layers.0.mlp.shared_expert_gate"),
            (64, 8)
        );
        // Body fallback for an unmatched path.
        assert_eq!(
            q.for_path("language_model.model.layers.0.self_attn.q_proj"),
            (64, 4)
        );
    }

    #[test]
    fn default_mode_filled() {
        let q: QuantizationConfig =
            serde_json::from_str(r#"{"group_size": 32, "bits": 4}"#).unwrap();
        assert_eq!(q.mode, "affine");
    }
}
