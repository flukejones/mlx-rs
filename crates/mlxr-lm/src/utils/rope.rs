//! Config-side `serde` helper for HF rope parameters.
//!
//! HF `rope_parameters` blocks mix numeric values (`factor: 32.0`)
//! with string discriminants (`rope_type: "default"`), often in the
//! same map. [`FloatOrString`] is the serde-untagged enum that
//! captures either; the gemma4 config tree reads them through it.

use serde::Deserialize;

/// Owned `f32` or `String` — used as the value type in HF
/// `rope_parameters` maps where one key might be numeric and another
/// a discriminant string.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum FloatOrString {
    Float(f32),
    String(String),
}
