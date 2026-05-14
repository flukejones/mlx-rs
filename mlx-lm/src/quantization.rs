//! Shared `quantization_config` parsing for MLX checkpoints.
//!
//! Mirrors `mlx_lm.models.{model}.{Model,ModelArgs}` in Python: the same
//! checkpoint may carry `quantization` *or* `quantization_config` (older
//! exports do both); both fields have the same shape, only one is honoured.
//! Use [`resolve_quantization`] to pick the effective config from a struct
//! that has both fields.

use serde::Deserialize;

const DEFAULT_QUANT_MODE: &str = "affine";

/// Quantisation settings parsed from `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
    #[serde(default = "default_quant_mode")]
    pub mode: String,
}

fn default_quant_mode() -> String {
    DEFAULT_QUANT_MODE.to_string()
}

/// Return the effective [`QuantizationConfig`], preferring the primary
/// `quantization` slot and falling back to the legacy `quantization_config`
/// slot if only the latter is populated.
pub fn resolve_quantization<'a>(
    primary: &'a Option<QuantizationConfig>,
    legacy: &'a Option<QuantizationConfig>,
) -> Option<&'a QuantizationConfig> {
    primary.as_ref().or(legacy.as_ref())
}
