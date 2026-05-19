//! `quantization_config` parsing for MLX checkpoints.

use serde::Deserialize;

const DEFAULT_QUANT_MODE: &str = "affine";

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
    #[serde(default = "default_quant_mode")]
    pub mode: String,
}

fn default_quant_mode() -> String {
    DEFAULT_QUANT_MODE.to_owned()
}

/// Prefer `quantization`; fall back to legacy `quantization_config`.
pub fn resolve_quantization<'a>(
    primary: &'a Option<QuantizationConfig>,
    legacy: &'a Option<QuantizationConfig>,
) -> Option<&'a QuantizationConfig> {
    primary.as_ref().or(legacy.as_ref())
}
