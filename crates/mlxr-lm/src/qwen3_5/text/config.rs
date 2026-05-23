//! Configuration types for the Qwen3.5 family of models.
//!
//! Both [`TextConfig`] and [`VisionConfig`] are deserialized straight
//! from the `text_config` / `vision_config` sub-objects of the
//! model's `config.json`.

use serde::Deserialize;

/// Per-layer architecture tag from `config.json::layer_types`. Unknown
/// strings hard-error rather than silently routing as one or the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QwenLayerKind {
    /// Gated DeltaNet (linear-attention SSM).
    LinearAttention,
    /// Regular GQA full attention.
    FullAttention,
}

/// Rotary embedding variant from `rope_parameters.type`. Qwen3.5
/// implements `default` and `mrope`; yarn / longrope are rejected at
/// model build time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QwenRopeType {
    /// Standard RoPE.
    #[default]
    Default,
    /// Multimodal RoPE used by the VL variants.
    Mrope,
}

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
    /// RoPE variant. Qwen3.5 implements `default` and `mrope`;
    /// yarn / longrope deserialize-fail before reaching `Attention::new`.
    #[serde(default, rename = "type", alias = "rope_type")]
    pub rope_type: QwenRopeType,
    /// Some configs emit a top-level `mrope_interleaved` flag; we accept it but
    /// the interleaved layout is the only one currently implemented.
    #[serde(default)]
    pub mrope_interleaved: bool,
}

/// Text-decoder hyperparameters for Qwen3.5.
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: i32,
    /// Dense MLP intermediate size. Absent on MoE variants
    /// (Qwen3.6-35B-A3B etc.) which use [`Self::moe_intermediate_size`]
    /// for the routed experts instead.
    #[serde(default)]
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub max_position_embeddings: i32,

    /// Per-layer architecture tag. Length must equal `num_hidden_layers`.
    pub layer_types: Vec<QwenLayerKind>,

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
    ///
    /// Qwen 3.6 configs also carry `output_gate_type: "swish"` — vestigial;
    /// upstream implementations unconditionally compute
    /// `output * sigmoid(gate)` regardless. Field is silently dropped by
    /// serde here; do not re-parse without confirming the reference path
    /// actually branches on it.
    #[serde(default = "default_attn_output_gate")]
    pub attn_output_gate: bool,

    /// Rotary embedding parameters.
    pub rope_parameters: RopeParameters,

    /// Optional explicit EOS token. May be a single id or a list of ids.
    #[serde(default)]
    pub eos_token_id: Option<EosTokenId>,

    // ── MoE fields (Qwen3.6-35B-A3B; absent on dense checkpoints) ──
    /// Number of routed experts. `0` on dense checkpoints; `is_moe()`
    /// gates on this.
    #[serde(default)]
    pub num_experts: i32,
    /// Top-k routing fan-out per token.
    #[serde(default)]
    pub num_experts_per_tok: i32,
    /// Inner hidden width per routed expert.
    #[serde(default)]
    pub moe_intermediate_size: i32,
    /// Inner hidden width of the always-on dense shared expert.
    #[serde(default)]
    pub shared_expert_intermediate_size: i32,

    // ── MTP (Multi-Token Prediction) fields ──
    /// Number of MTP layers (0 disables).
    #[serde(default)]
    pub mtp_num_hidden_layers: i32,
    /// If false, the MTP head shares `embed_tokens` with the main decoder.
    #[serde(default)]
    pub mtp_use_dedicated_embeddings: bool,
}

impl TextConfig {
    /// True for MoE variants (any non-zero `num_experts`).
    pub fn is_moe(&self) -> bool {
        self.num_experts > 0
    }
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
    /// if not already present.
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

/// Qwen-family envelope of `config.json`. Stored inside
/// [`crate::config::Family::Qwen35`] / [`crate::config::Family::Qwen35Moe`].
/// Quantisation lives on the outer [`crate::config::ModelConfig`].
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub text_config: TextConfig,
    /// Text-only checkpoints (Qwen 3.6 MoE) omit this entirely; the
    /// VLM adapter requires it and errors at its own load_context if
    /// absent.
    #[serde(default)]
    pub vision_config: Option<VisionConfig>,

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
                .map(|k| *k == QwenLayerKind::LinearAttention)
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
    use crate::config::ModelConfig as Config;

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

    fn parse_outer(json: &str) -> Config {
        serde_json::from_str(json).unwrap()
    }

    fn unwrap_qwen(cfg: &Config) -> &ModelConfig {
        cfg.family.as_qwen35().expect("expected qwen3_5 family")
    }

    #[test]
    fn parses_chandra_config() {
        let cfg = parse_outer(CHANDRA_CONFIG_JSON);
        assert_eq!(cfg.family_name(), "qwen3_5");
        let env = unwrap_qwen(&cfg);
        assert_eq!(env.text_config.hidden_size, 2560);
        assert_eq!(env.text_config.head_dim, 256);
        assert_eq!(env.text_config.num_attention_heads, 16);
        assert_eq!(env.text_config.num_key_value_heads, 4);
        assert_eq!(env.text_config.num_hidden_layers, 32);
        assert_eq!(env.text_config.layer_types.len(), 32);
        assert!(env.text_config.attn_output_gate);
        assert_eq!(
            env.text_config.rope_parameters.mrope_section,
            vec![11, 11, 10]
        );
        assert_eq!(env.text_config.rope_parameters.partial_rotary_factor, 0.25);
        assert_eq!(env.text_config.rope_parameters.rope_theta, 10_000_000.0);
        assert!(env.text_config.rope_parameters.mrope_interleaved);

        assert_eq!(env.vision_config.as_ref().unwrap().depth, 24);
        assert_eq!(env.vision_config.as_ref().unwrap().hidden_size, 1024);
        assert_eq!(env.vision_config.as_ref().unwrap().out_hidden_size, 2560);
        assert_eq!(env.vision_config.as_ref().unwrap().num_heads, 16);
        assert!(env
            .vision_config
            .as_ref()
            .unwrap()
            .deepstack_visual_indexes
            .is_empty());

        assert_eq!(env.image_token_id, 248056);
        assert_eq!(env.vision_start_token_id, 248053);
        assert_eq!(env.vision_end_token_id, 248054);

        let q = cfg.quantization().unwrap();
        assert_eq!(q.bits, 8);
        assert_eq!(q.group_size, 64);
        assert_eq!(q.mode, crate::quantization::QuantMode::Affine);
    }

    #[test]
    fn layer_type_dispatch() {
        let cfg = parse_outer(CHANDRA_CONFIG_JSON);
        let env = unwrap_qwen(&cfg);
        // layer_types: [linear, linear, linear, full, linear, linear, linear, full, ...]
        assert!(env.is_linear_layer(0));
        assert!(env.is_linear_layer(1));
        assert!(env.is_linear_layer(2));
        assert!(!env.is_linear_layer(3));
        assert!(env.is_linear_layer(4));
        assert!(!env.is_linear_layer(7));
        assert!(!env.is_linear_layer(31));
    }

    /// Qwen3.6-27B reuses the qwen3_5 schema with extra unknown fields
    /// (`output_gate_type`, `mtp_*`, `mamba_ssm_dtype`). Verifies serde
    /// ignores them and the known dims land on the right slots.
    #[test]
    fn parses_qwen3_6_27b_config() {
        let json = include_str!("../../../tests/fixtures/qwen3_6_27b/config.json");
        let cfg = parse_outer(json);
        let env = unwrap_qwen(&cfg);
        assert_eq!(env.text_config.num_hidden_layers, 64);
        assert_eq!(env.text_config.hidden_size, 5120);
        assert_eq!(env.text_config.intermediate_size, 17408);
        assert_eq!(env.text_config.num_attention_heads, 24);
        assert_eq!(env.text_config.num_key_value_heads, 4);
        assert_eq!(env.text_config.head_dim, 256);
        assert_eq!(env.text_config.linear_num_value_heads, 48);
        assert_eq!(env.text_config.linear_num_key_heads, 16);
        assert_eq!(env.text_config.layer_types.len(), 64);
        assert!(env.text_config.attn_output_gate);
        assert!(!env.text_config.tie_word_embeddings);
        let q = cfg.quantization().unwrap();
        assert_eq!(q.bits, 4);
        assert_eq!(q.group_size, 64);
    }

    /// Qwen3.6-35B-A3B is the MoE sibling of Qwen3.6-27B. Same
    /// `qwen3_5_moe_text` schema with `num_experts`,
    /// `num_experts_per_tok`, `moe_intermediate_size`, and
    /// `shared_expert_intermediate_size`; `intermediate_size` is absent
    /// (no dense MLP). The q4 checkpoint also ships per-tensor
    /// quantisation overrides for `mlp.gate` + `mlp.shared_expert_gate`.
    #[test]
    fn parses_qwen3_6_35b_a3b_config() {
        let json = include_str!("../../../tests/fixtures/qwen3_6_35b_a3b/config.json");
        let cfg = parse_outer(json);
        let env = unwrap_qwen(&cfg);
        assert_eq!(env.text_config.num_hidden_layers, 40);
        assert_eq!(env.text_config.hidden_size, 2048);
        assert_eq!(env.text_config.intermediate_size, 0); // absent, defaults to 0
        assert!(env.text_config.is_moe());
        assert_eq!(env.text_config.num_experts, 256);
        assert_eq!(env.text_config.num_experts_per_tok, 8);
        assert_eq!(env.text_config.moe_intermediate_size, 512);
        assert_eq!(env.text_config.shared_expert_intermediate_size, 512);
        assert_eq!(env.text_config.num_attention_heads, 16);
        assert_eq!(env.text_config.num_key_value_heads, 2);
        assert_eq!(env.text_config.head_dim, 256);
        assert_eq!(env.text_config.linear_num_value_heads, 32);
        assert_eq!(env.text_config.layer_types.len(), 40);
        assert!(!env.text_config.tie_word_embeddings);

        let q = cfg.quantization().unwrap();
        assert_eq!(q.bits, 4);
        assert_eq!(q.group_size, 64);
        // Per-tensor quant overrides: router + shared-expert gate at 8b.
        let (gs, bits) = q.for_path("language_model.model.layers.0.mlp.gate");
        assert_eq!((gs, bits), (64, 8));
        let (gs, bits) = q.for_path("language_model.model.layers.0.mlp.shared_expert_gate");
        assert_eq!((gs, bits), (64, 8));
        // Body fallback for non-override paths.
        let (gs, bits) = q.for_path("language_model.model.layers.0.self_attn.q_proj");
        assert_eq!((gs, bits), (64, 4));
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
