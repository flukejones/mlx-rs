//! Per-layer caches for the Qwen3.5 hybrid decoder.
//!
//! Full-attention layers reuse [`crate::cache::ConcatKeyValueCache`]. Linear-
//! attention (Gated DeltaNet) layers carry their own recurrent state plus the
//! tail of the causal Conv1d input from previous steps, which we model with
//! [`LinearAttnCache`]. [`LayerCache`] is the per-layer slot the model walks
//! through during forward.

use mlx_rs::Array;

use crate::cache::ConcatKeyValueCache;

/// State carried across decode steps inside a Gated DeltaNet block.
///
/// `conv_state` is the last `linear_conv_kernel_dim - 1` rows of the projected
/// `(q, k, v)` features along the sequence axis — i.e. the causal Conv1d
/// history.
///
/// `recurrent_state` is the live `[B, Hv, Dv, Dk]` SSM state.
///
/// Both fields are `None` for the first forward pass; the GDN block treats
/// `None` as a zero-initialised tensor of the right shape.
#[derive(Debug, Clone, Default)]
pub struct LinearAttnCache {
    /// Causal Conv1d history. `[B, conv_kernel_size - 1, conv_dim]`.
    pub conv_state: Option<Array>,
    /// Recurrent SSM state. `[B, Hv, Dv, Dk]`.
    pub recurrent_state: Option<Array>,
    /// Number of tokens absorbed so far. Mirrors the offset on the KV cache.
    pub offset: i32,
}

impl LinearAttnCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Resets the cache as if no tokens had been seen yet.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// One slot in the per-layer cache list. Each Qwen3.5 layer holds exactly one
/// of these and uses it for the full duration of a generation.
#[derive(Debug, Clone)]
pub enum LayerCache {
    /// Cache for a full-attention (GQA) layer.
    FullAttention(ConcatKeyValueCache),
    /// Cache for a linear-attention (Gated DeltaNet) layer.
    LinearAttention(LinearAttnCache),
}

impl LayerCache {
    /// Build the matching cache for a layer based on whether it is a
    /// linear-attention (Gated DeltaNet) layer.
    pub fn for_layer(is_linear: bool) -> Self {
        if is_linear {
            LayerCache::LinearAttention(LinearAttnCache::new())
        } else {
            LayerCache::FullAttention(ConcatKeyValueCache::new())
        }
    }

    /// Get a mutable reference to the full-attention cache, panicking if this
    /// slot is a linear-attention cache instead.
    pub fn as_full_attention_mut(&mut self) -> &mut ConcatKeyValueCache {
        match self {
            LayerCache::FullAttention(c) => c,
            LayerCache::LinearAttention(_) => {
                panic!("LayerCache: expected FullAttention slot, got LinearAttention")
            }
        }
    }

    /// Get a mutable reference to the linear-attention cache, panicking if
    /// this slot is a full-attention cache instead.
    pub fn as_linear_attention_mut(&mut self) -> &mut LinearAttnCache {
        match self {
            LayerCache::LinearAttention(c) => c,
            LayerCache::FullAttention(_) => {
                panic!("LayerCache: expected LinearAttention slot, got FullAttention")
            }
        }
    }

    /// Returns true if this slot is a linear-attention cache.
    pub fn is_linear(&self) -> bool {
        matches!(self, LayerCache::LinearAttention(_))
    }

    /// Returns the running offset for this layer.
    pub fn offset(&self) -> i32 {
        match self {
            LayerCache::FullAttention(c) => {
                use crate::cache::KeyValueCache;
                c.offset()
            }
            LayerCache::LinearAttention(c) => c.offset,
        }
    }
}

/// Build one [`LayerCache`] per layer for a [`crate::models::qwen3_5::ModelConfig`].
pub fn make_caches(config: &crate::models::qwen3_5::ModelConfig) -> Vec<LayerCache> {
    let n = config.text_config.num_hidden_layers as usize;
    (0..n)
        .map(|i| LayerCache::for_layer(config.is_linear_layer(i)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_config(layer_types: Vec<&str>) -> crate::models::qwen3_5::ModelConfig {
        let layers: Vec<String> = layer_types.into_iter().map(String::from).collect();
        let n = layers.len() as i32;
        let json = serde_json::json!({
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": n,
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "rms_norm_eps": 1e-6,
                "vocab_size": 100,
                "max_position_embeddings": 256,
                "layer_types": layers,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 4,
                "linear_conv_kernel_dim": 4,
                "rope_parameters": {
                    "mrope_section": [2, 1, 1],
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 1.0,
                    "type": "default"
                }
            },
            "vision_config": {
                "depth": 2,
                "hidden_size": 16,
                "intermediate_size": 32,
                "out_hidden_size": 32,
                "num_heads": 2,
                "patch_size": 16,
                "in_channels": 3,
                "spatial_merge_size": 2
            }
        });
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn make_caches_matches_layer_types() {
        let cfg = synthetic_config(vec![
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
        ]);
        let caches = make_caches(&cfg);
        assert_eq!(caches.len(), 4);
        assert!(caches[0].is_linear());
        assert!(caches[1].is_linear());
        assert!(!caches[2].is_linear());
        assert!(caches[3].is_linear());
    }

    #[test]
    fn for_layer_dispatches_to_correct_variant() {
        let a = LayerCache::for_layer(true);
        assert!(matches!(a, LayerCache::LinearAttention(_)));
        let b = LayerCache::for_layer(false);
        assert!(matches!(b, LayerCache::FullAttention(_)));
    }

    #[test]
    fn linear_attn_cache_starts_empty() {
        let c = LinearAttnCache::new();
        assert!(c.conv_state.is_none());
        assert!(c.recurrent_state.is_none());
        assert_eq!(c.offset, 0);
    }
}
