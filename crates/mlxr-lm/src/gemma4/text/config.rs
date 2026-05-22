//! Gemma 4 text-only config (mirrors HF `Gemma4TextConfig`).

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::error::Error;
use crate::quantization::QuantizationConfig;
use crate::utils::rope::FloatOrString;

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4Config {
    #[serde(default = "default_model_type")]
    pub model_type: String,

    #[serde(default = "default_hidden_size")]
    pub hidden_size: i32,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: i32,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: i32,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: i32,
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    #[serde(default = "default_global_head_dim")]
    pub global_head_dim: i32,
    #[serde(default = "default_num_kv_heads")]
    pub num_key_value_heads: i32,
    pub num_global_key_value_heads: Option<i32>,
    #[serde(default)]
    pub num_kv_shared_layers: i32,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: i32,
    #[serde(default = "default_sliding_window_pattern")]
    pub sliding_window_pattern: i32,

    pub rope_parameters: Option<HashMap<String, HashMap<String, FloatOrString>>>,
    #[serde(default)]
    pub rope_traditional: bool,

    #[serde(default)]
    pub attention_k_eq_v: bool,
    #[serde(default = "default_final_logit_softcapping")]
    pub final_logit_softcapping: f32,
    #[serde(default = "default_use_double_wide_mlp")]
    pub use_double_wide_mlp: bool,

    /// MoE expert routing (26B-A4B). When `true`, each `DecoderLayer`
    /// carries a `Router` + `Experts` branch in addition to the dense
    /// MLP. Default `false`.
    #[serde(default)]
    pub enable_moe_block: bool,
    pub num_experts: Option<i32>,
    pub top_k_experts: Option<i32>,
    pub moe_intermediate_size: Option<i32>,

    /// Per-layer input embeddings (E2B/E4B variants). When `> 0`,
    /// each layer takes an extra `[B, L, hidden_size_per_layer_input]`
    /// gate vector projected from its own slice of an N×D_pl-wide
    /// embedding lookup.
    #[serde(default)]
    pub hidden_size_per_layer_input: i32,
    #[serde(default = "default_vocab_size_per_layer_input")]
    pub vocab_size_per_layer_input: i32,

    /// Optional explicit per-layer types list (`"sliding_attention"` or
    /// `"full_attention"`). Derived from `sliding_window_pattern` if
    /// absent.
    pub layer_types: Option<Vec<String>>,

    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
    #[serde(default)]
    pub quantization_config: Option<QuantizationConfig>,
}

fn default_model_type() -> String {
    "gemma4_text".to_owned()
}
const fn default_hidden_size() -> i32 {
    1536
}
const fn default_num_hidden_layers() -> i32 {
    35
}
const fn default_intermediate_size() -> i32 {
    6144
}
const fn default_num_attention_heads() -> i32 {
    8
}
const fn default_head_dim() -> i32 {
    256
}
const fn default_global_head_dim() -> i32 {
    512
}
const fn default_num_kv_heads() -> i32 {
    1
}
const fn default_rms_norm_eps() -> f32 {
    1e-6
}
const fn default_vocab_size() -> i32 {
    262144
}
const fn default_max_position_embeddings() -> i32 {
    131072
}
const fn default_sliding_window() -> i32 {
    512
}
const fn default_sliding_window_pattern() -> i32 {
    5
}
const fn default_final_logit_softcapping() -> f32 {
    30.0
}
const fn default_use_double_wide_mlp() -> bool {
    true
}
const fn default_tie_word_embeddings() -> bool {
    true
}
const fn default_vocab_size_per_layer_input() -> i32 {
    262144
}

/// Outer multimodal-wrapper config schema (`gemma4.ModelArgs` in
/// Python). HF Gemma 4 checkpoints store all text-model architecture
/// fields under `text_config`; the wrapper carries the `quantization`
/// block, and audio/vision configs we don't load.
#[derive(Debug, Clone, Deserialize)]
struct OuterConfig {
    pub text_config: Option<Gemma4Config>,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
    #[serde(default)]
    pub quantization_config: Option<QuantizationConfig>,
}

impl Gemma4Config {
    /// Parse the HF config.json. Accepts both forms:
    ///   - Multimodal wrapper: `{ model_type, text_config: {...}, quantization: {...} }`
    ///   - Flat text-only:    `{ hidden_size, ... }`
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        let raw = fs::read_to_string(path.as_ref())?;
        // Try the outer-wrapper form first; flat-form falls through.
        if let Ok(outer) = serde_json::from_str::<OuterConfig>(&raw) {
            if let Some(mut inner) = outer.text_config {
                // Propagate the outer-level quantization block down so
                // `try_into_quantized` finds it on the inner config.
                if inner.quantization.is_none() {
                    inner.quantization = outer.quantization;
                }
                if inner.quantization_config.is_none() {
                    inner.quantization_config = outer.quantization_config;
                }
                return Ok(inner);
            }
        }
        let cfg = serde_json::from_str::<Self>(&raw)?;
        Ok(cfg)
    }

    /// Pattern-derived layer-type table when `layer_types` is absent.
    /// Mirrors mlxr_lm Python:
    /// `pattern = ["sliding"]*(P-1) + ["full"]`, tiled to N layers.
    pub fn layer_types_resolved(&self) -> Vec<LayerKind> {
        if let Some(explicit) = &self.layer_types {
            return explicit.iter().map(|s| LayerKind::parse(s)).collect();
        }
        let pattern_len = self.sliding_window_pattern as usize;
        (0..self.num_hidden_layers as usize)
            .map(|i| {
                if (i % pattern_len) == pattern_len - 1 {
                    LayerKind::FullAttention
                } else {
                    LayerKind::SlidingAttention
                }
            })
            .collect()
    }

    /// Sliding-window pattern length (distance between consecutive full
    /// attention layers, inclusive of the full layer itself). Derived
    /// from `layer_types` when present — mlx-community checkpoints can
    /// have a stale or null `sliding_window_pattern` while `layer_types`
    /// carries the truth (gemma-4-31b-it-8bit has the field null but
    /// the first full_attention at index 5, implying pattern of 6).
    /// Falls back to the explicit config field, then to the default 5.
    pub fn effective_sliding_window_pattern(&self) -> i32 {
        if let Some(types) = &self.layer_types {
            for (i, ty) in types.iter().enumerate() {
                if ty == "full_attention" {
                    return (i as i32) + 1;
                }
            }
        }
        self.sliding_window_pattern
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerKind {
    SlidingAttention,
    FullAttention,
}

impl LayerKind {
    pub fn parse(s: &str) -> Self {
        match s {
            "full_attention" => Self::FullAttention,
            _ => Self::SlidingAttention,
        }
    }
}
