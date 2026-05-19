//! Configuration types for the Qwen3.5 family of models.
//!
//! Mirrors `mlx_vlm.models.qwen3_5.config` plus the inherited
//! `mlx_vlm.models.qwen3_vl.config.VisionConfig`. Both `TextConfig` and
//! `VisionConfig` are deserialized straight from the `text_config` /
//! `vision_config` sub-objects of the model's `config.json`.

use serde::Deserialize;

/// Sentinel layer in [`TextConfig::layer_types`] that uses Gated DeltaNet (SSM).
pub const LAYER_TYPE_LINEAR: &str = "linear_attention";

/// Sentinel layer in [`TextConfig::layer_types`] that uses regular GQA attention.
pub const LAYER_TYPE_FULL: &str = "full_attention";

/// Default EOS for Qwen chat templates. Used as a fallback when `eos_token_id`
/// is missing from `config.json`.
pub const QWEN_CHAT_EOS_TOKEN_ID: u32 = 248046;

/// Parameters for the Qwen3.5 multimodal RoPE.
///
/// `mrope_section` slices the rotary dimension into three independent axes
/// (t/h/w of the multimodal grid). `partial_rotary_factor` keeps only the
/// first `head_dim * partial_rotary_factor` features rotated and passes the
/// rest through unchanged.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    /// Lengths of the three mrope axes — must sum to `head_dim *
    /// partial_rotary_factor / 2`.
    pub mrope_section: Vec<i32>,
    /// Base used to compute angular frequency. Renamed to `rope_theta` in some
    /// checkpoints; serde reads either via the field name.
    pub rope_theta: f32,
    /// Fraction of `head_dim` that is rotated. 0.25 for Qwen3.5.
    pub partial_rotary_factor: f32,
    /// Type tag — accepted values `"default"` and `"mrope"`. The Qwen3.5
    /// reference `Qwen3_5RotaryEmbedding` doesn't branch on this field, so
    /// other variants (yarn / longrope) are rejected at `Attention` build
    /// time rather than silently parsed.
    #[serde(default = "default_rope_type", rename = "type", alias = "rope_type")]
    pub rope_type: String,
    /// Some configs emit a top-level `mrope_interleaved` flag; we accept it but
    /// the interleaved layout is the only one currently implemented.
    #[serde(default)]
    pub mrope_interleaved: bool,
}

fn default_rope_type() -> String {
    "default".to_owned()
}

/// Text-decoder hyperparameters for Qwen3.5.
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub model_type: String,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub max_position_embeddings: i32,

    /// Per-layer architecture string. Length must equal `num_hidden_layers`.
    /// Each entry is one of [`LAYER_TYPE_LINEAR`] or [`LAYER_TYPE_FULL`].
    pub layer_types: Vec<String>,

    /// Every `full_attention_interval`-th layer is a full-attention layer.
    /// Used when [`Self::layer_types`] is empty.
    #[serde(default = "default_full_attention_interval")]
    pub full_attention_interval: i32,

    /// Number of key heads for the linear-attention (Gated DeltaNet) block.
    pub linear_num_key_heads: i32,
    /// Number of value heads for the linear-attention block.
    pub linear_num_value_heads: i32,
    /// Per-head key dim for the linear-attention block.
    pub linear_key_head_dim: i32,
    /// Per-head value dim for the linear-attention block.
    pub linear_value_head_dim: i32,
    /// Causal Conv1d kernel size used inside the GDN block.
    pub linear_conv_kernel_dim: i32,

    /// Whether the LM head is tied to `embed_tokens`.
    #[serde(default)]
    pub tie_word_embeddings: bool,

    /// Bias on `q_proj`/`k_proj`/`v_proj`/`o_proj`. Always false for Qwen3.5.
    #[serde(default)]
    pub attention_bias: bool,

    /// If true, full-attention layers add a sigmoid gate on the attention
    /// output (`output = o_proj(attn_out * sigmoid(gate))`). Always true for
    /// Qwen3.5.
    #[serde(default = "default_attn_output_gate")]
    pub attn_output_gate: bool,

    /// Rotary embedding parameters.
    pub rope_parameters: RopeParameters,

    /// Optional explicit EOS token. May be a single id or a list of ids.
    #[serde(default)]
    pub eos_token_id: Option<EosTokenId>,
}

fn default_full_attention_interval() -> i32 {
    4
}

fn default_attn_output_gate() -> bool {
    true
}

/// EOS may be a scalar or a list in the config JSON.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EosTokenId {
    /// Single EOS token id.
    Single(u32),
    /// Multiple acceptable EOS token ids.
    Multiple(Vec<u32>),
}

impl EosTokenId {
    /// Returns every EOS id as a flat list, appending [`QWEN_CHAT_EOS_TOKEN_ID`]
    /// if not already present — mirrors `resolve_qwen_eos_token_id` from the
    /// Python implementation.
    pub fn into_vec_with_chat_eos(self) -> Vec<u32> {
        let mut v = match self {
            Self::Single(x) => vec![x],
            Self::Multiple(xs) => xs,
        };
        if !v.contains(&QWEN_CHAT_EOS_TOKEN_ID) {
            v.push(QWEN_CHAT_EOS_TOKEN_ID);
        }
        v
    }
}

/// Vision-tower hyperparameters (shared with Qwen3-VL, with Qwen3.5 defaults).
#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    #[serde(default = "default_vision_model_type")]
    pub model_type: String,
    pub depth: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub out_hidden_size: i32,
    pub num_heads: i32,
    pub patch_size: i32,
    pub in_channels: i32,
    pub spatial_merge_size: i32,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: i32,
    #[serde(default = "default_num_position_embeddings")]
    pub num_position_embeddings: i32,
    /// Layer indices in the ViT where intermediate hidden states are injected
    /// back into the LM residual stream. Empty for Qwen3.5 (deepstack
    /// disabled).
    #[serde(default)]
    pub deepstack_visual_indexes: Vec<i32>,
}

fn default_vision_model_type() -> String {
    "qwen3_5".to_owned()
}

fn default_temporal_patch_size() -> i32 {
    2
}

fn default_num_position_embeddings() -> i32 {
    2304
}

pub use crate::quantization::QuantizationConfig;

/// Top-level model config matching `config.json` shape.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub model_type: String,
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,

    #[serde(default = "default_image_token_id")]
    pub image_token_id: u32,
    #[serde(default = "default_video_token_id")]
    pub video_token_id: u32,
    #[serde(default = "default_vision_start_token_id")]
    pub vision_start_token_id: u32,
    #[serde(default = "default_vision_end_token_id")]
    pub vision_end_token_id: u32,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub eos_token_id: Option<EosTokenId>,

    /// Quantisation settings. Some checkpoints emit a second
    /// `quantization_config` block with identical contents; that field is
    /// captured separately and ignored unless `quantization` is missing.
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,

    #[serde(default)]
    quantization_config: Option<QuantizationConfig>,
}

fn default_image_token_id() -> u32 {
    248056
}
fn default_video_token_id() -> u32 {
    248057
}
fn default_vision_start_token_id() -> u32 {
    248045
}
fn default_vision_end_token_id() -> u32 {
    248046
}

impl ModelConfig {
    /// Load a `ModelConfig` from the standard `config.json` at the root of an
    /// MLX-format checkpoint directory.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, crate::error::Error> {
        let f = std::fs::File::open(path.as_ref())?;
        let mut cfg: Self = serde_json::from_reader(f)?;
        if cfg.quantization.is_none() {
            cfg.quantization = cfg.quantization_config.take();
        }
        Ok(cfg)
    }

    /// Effective quantisation settings, preferring `quantization` and falling
    /// back to `quantization_config` if only the latter was provided.
    pub fn effective_quantization(&self) -> Option<&QuantizationConfig> {
        self.quantization
            .as_ref()
            .or(self.quantization_config.as_ref())
    }

    /// Returns true for the `i`-th decoder layer if it is a linear-attention
    /// (Gated DeltaNet) layer rather than a full-attention layer.
    ///
    /// Prefers `layer_types` when available; falls back to the
    /// `full_attention_interval` heuristic.
    pub fn is_linear_layer(&self, layer_idx: usize) -> bool {
        let lt = &self.text_config.layer_types;
        if !lt.is_empty() {
            return lt
                .get(layer_idx)
                .map(|s| s.as_str() == LAYER_TYPE_LINEAR)
                .unwrap_or(false);
        }
        let interval = self.text_config.full_attention_interval;
        if interval <= 0 {
            return false;
        }
        ((layer_idx as i32 + 1) % interval) != 0
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    #![allow(clippy::print_stdout, reason = "test code")]
    #![allow(clippy::print_stderr, reason = "test code")]
    use super::*;

    const CHANDRA_CONFIG_JSON: &str = r#"
    {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "eos_token_id": 248044,
        "image_token_id": 248056,
        "model_type": "qwen3_5",
        "quantization": {"group_size": 64, "bits": 8, "mode": "affine"},
        "quantization_config": {"group_size": 64, "bits": 8, "mode": "affine"},
        "text_config": {
            "attention_bias": false,
            "attn_output_gate": true,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_size": 2560,
            "intermediate_size": 9216,
            "layer_types": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention"
            ],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "model_type": "qwen3_5_text",
            "num_attention_heads": 16,
            "num_hidden_layers": 32,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default"
            },
            "tie_word_embeddings": true,
            "vocab_size": 248320
        },
        "tie_word_embeddings": true,
        "video_token_id": 248057,
        "vision_config": {
            "deepstack_visual_indexes": [],
            "depth": 24,
            "hidden_size": 1024,
            "in_channels": 3,
            "intermediate_size": 4096,
            "model_type": "qwen3_5",
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": 2560,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2
        },
        "vision_end_token_id": 248054,
        "vision_start_token_id": 248053
    }
    "#;

    #[test]
    fn parses_chandra_config() {
        let cfg: ModelConfig = serde_json::from_str(CHANDRA_CONFIG_JSON).unwrap();
        assert_eq!(cfg.model_type, "qwen3_5");
        assert_eq!(cfg.text_config.hidden_size, 2560);
        assert_eq!(cfg.text_config.head_dim, 256);
        assert_eq!(cfg.text_config.num_attention_heads, 16);
        assert_eq!(cfg.text_config.num_key_value_heads, 4);
        assert_eq!(cfg.text_config.num_hidden_layers, 32);
        assert_eq!(cfg.text_config.layer_types.len(), 32);
        assert!(cfg.text_config.attn_output_gate);
        assert_eq!(
            cfg.text_config.rope_parameters.mrope_section,
            vec![11, 11, 10]
        );
        assert_eq!(cfg.text_config.rope_parameters.partial_rotary_factor, 0.25);
        assert_eq!(cfg.text_config.rope_parameters.rope_theta, 10_000_000.0);
        assert!(cfg.text_config.rope_parameters.mrope_interleaved);

        assert_eq!(cfg.vision_config.depth, 24);
        assert_eq!(cfg.vision_config.hidden_size, 1024);
        assert_eq!(cfg.vision_config.out_hidden_size, 2560);
        assert_eq!(cfg.vision_config.num_heads, 16);
        assert!(cfg.vision_config.deepstack_visual_indexes.is_empty());

        assert_eq!(cfg.image_token_id, 248056);
        assert_eq!(cfg.vision_start_token_id, 248053);
        assert_eq!(cfg.vision_end_token_id, 248054);

        let q = cfg.effective_quantization().unwrap();
        assert_eq!(q.bits, 8);
        assert_eq!(q.group_size, 64);
        assert_eq!(q.mode, "affine");
    }

    #[test]
    fn layer_type_dispatch() {
        let cfg: ModelConfig = serde_json::from_str(CHANDRA_CONFIG_JSON).unwrap();
        // layer_types: [linear, linear, linear, full, linear, linear, linear, full, ...]
        assert!(cfg.is_linear_layer(0));
        assert!(cfg.is_linear_layer(1));
        assert!(cfg.is_linear_layer(2));
        assert!(!cfg.is_linear_layer(3));
        assert!(cfg.is_linear_layer(4));
        assert!(!cfg.is_linear_layer(7));
        assert!(!cfg.is_linear_layer(31));
    }

    #[test]
    #[ignore = "requires local model files at ~/MLXModels/chandra2/chandra-ocr-2-mlx-q8"]
    fn loads_chandra_q8_config_from_disk() {
        let home = std::env::var("HOME").unwrap();
        let path = std::path::PathBuf::from(home)
            .join("MLXModels/chandra2/chandra-ocr-2-mlx-q8/config.json");
        let cfg = ModelConfig::from_file(&path).expect("parse chandra q8 config");
        assert_eq!(cfg.text_config.num_hidden_layers, 32);
        assert_eq!(cfg.text_config.layer_types.len(), 32);
        assert_eq!(cfg.vision_config.depth, 24);
        assert_eq!(cfg.image_token_id, 248056);
        assert!(cfg.effective_quantization().is_some());
    }

    #[test]
    fn eos_resolves_chat_token() {
        let single = EosTokenId::Single(248044);
        let v = single.into_vec_with_chat_eos();
        assert_eq!(v, vec![248044, QWEN_CHAT_EOS_TOKEN_ID]);

        let multi = EosTokenId::Multiple(vec![1, QWEN_CHAT_EOS_TOKEN_ID]);
        let v = multi.into_vec_with_chat_eos();
        assert_eq!(v, vec![1, QWEN_CHAT_EOS_TOKEN_ID]);
    }
}
