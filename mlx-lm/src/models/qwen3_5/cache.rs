//! Per-layer caches for the Qwen3.5 hybrid decoder.
//!
//! Full-attention layers reuse [`crate::cache::ConcatKeyValueCache`]. Linear-
//! attention (Gated DeltaNet) layers carry their own recurrent state plus the
//! tail of the causal Conv1d input from previous steps, which we model with
//! [`LinearAttnCache`]. [`LayerCache`] is the per-layer slot the model walks
//! through during forward.

use mlx_rs::Array;

use crate::cache::turboquant::cache::{TurboQuantConfig, TurboQuantKVCache};
use crate::cache::ConcatKeyValueCache;
use crate::error::Error;

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
///
/// `FullAttentionTQ` is the TurboQuant variant — only applicable to
/// full-attention layers (Gated DeltaNet linear-attention layers carry SSM
/// state, not K/V, and don't benefit from KV quantisation).
#[derive(Debug)]
pub enum LayerCache {
    /// Cache for a full-attention (GQA) layer.
    FullAttention(ConcatKeyValueCache),
    /// TurboQuant cache for a full-attention layer. Behaviourally
    /// interchangeable with [`LayerCache::FullAttention`]; both implement
    /// [`crate::cache::KeyValueCache`].
    FullAttentionTQ(Box<TurboQuantKVCache>),
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
    ///
    /// Both `FullAttention` and `FullAttentionTQ` are full-attention slots
    /// but they hold *different concrete types*, so a single mutable
    /// reference can't be returned uniformly. Callers that need to mix
    /// TQ caches into qwen3.5 must use [`Self::full_attention_kv_mut`]
    /// (returns `&mut dyn KeyValueCache`).
    pub fn as_full_attention_mut(&mut self) -> &mut ConcatKeyValueCache {
        match self {
            LayerCache::FullAttention(c) => c,
            LayerCache::FullAttentionTQ(_) => {
                panic!(
                    "LayerCache: expected non-TQ FullAttention slot, got FullAttentionTQ — \
                     use full_attention_kv_mut() instead"
                )
            }
            LayerCache::LinearAttention(_) => {
                panic!("LayerCache: expected FullAttention slot, got LinearAttention")
            }
        }
    }

    /// Get a mutable trait-object reference to any full-attention cache
    /// (regular or TurboQuant). Linear-attention layers return `None`.
    /// Required for callers that want to mix TQ caches uniformly with the
    /// existing `ConcatKeyValueCache`-typed attention path.
    pub fn full_attention_kv_mut(&mut self) -> Option<&mut dyn crate::cache::KeyValueCache> {
        match self {
            LayerCache::FullAttention(c) => Some(c),
            LayerCache::FullAttentionTQ(c) => Some(c.as_mut()),
            LayerCache::LinearAttention(_) => None,
        }
    }

    /// Get a mutable reference to the linear-attention cache, panicking if
    /// this slot is a full-attention cache instead.
    pub fn as_linear_attention_mut(&mut self) -> &mut LinearAttnCache {
        match self {
            LayerCache::LinearAttention(c) => c,
            LayerCache::FullAttention(_) | LayerCache::FullAttentionTQ(_) => {
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
        use crate::cache::KeyValueCache;
        match self {
            LayerCache::FullAttention(c) => c.offset(),
            LayerCache::FullAttentionTQ(c) => c.offset(),
            LayerCache::LinearAttention(c) => c.offset,
        }
    }
}

/// Build a per-layer cache list with TurboQuant caches at every full-
/// attention layer and standard `LinearAttnCache` at every linear-attention
/// layer.
///
/// Per-layer seed: `base_seed + i` so each full-attention layer gets an
/// independent Π / S.
///
/// Note (Phase 4): qwen3.5's attention forward currently consumes
/// `Option<&mut ConcatKeyValueCache>` — concretely typed — so the TQ
/// caches built here only flow through the model once the attention
/// forward is generic-ised or dispatches via `full_attention_kv_mut`.
/// This factory is the data-plane half of that work; the dispatch
/// follow-up is tracked separately.
pub fn make_caches_with_tq(
    config: &crate::models::qwen3_5::ModelConfig,
    base_seed: u64,
) -> Result<Vec<LayerCache>, Error> {
    let n = config.text_config.num_hidden_layers as usize;
    let head_dim = config.text_config.head_dim;
    (0..n)
        .map(|i| {
            if config.is_linear_layer(i) {
                Ok(LayerCache::LinearAttention(LinearAttnCache::new()))
            } else {
                let cfg = TurboQuantConfig::new(head_dim, base_seed.wrapping_add(i as u64));
                let cache = TurboQuantKVCache::new(cfg)?;
                Ok(LayerCache::FullAttentionTQ(Box::new(cache)))
            }
        })
        .collect()
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
    fn make_caches_with_tq_uses_turboquant_at_full_attn_layers() {
        let cfg = synthetic_config(vec![
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ]);
        let caches = make_caches_with_tq(&cfg, 17).unwrap();
        assert_eq!(caches.len(), 4);
        assert!(matches!(caches[0], LayerCache::LinearAttention(_)));
        assert!(matches!(caches[1], LayerCache::FullAttentionTQ(_)));
        assert!(matches!(caches[2], LayerCache::LinearAttention(_)));
        assert!(matches!(caches[3], LayerCache::FullAttentionTQ(_)));

        // Independent seeds per layer.
        if let (LayerCache::FullAttentionTQ(a), LayerCache::FullAttentionTQ(b)) =
            (&caches[1], &caches[3])
        {
            assert_eq!(a.config().seed, 18);
            assert_eq!(b.config().seed, 20);
        }
    }

    #[test]
    fn full_attention_kv_mut_dispatches_uniformly() {
        let cfg = synthetic_config(vec!["full_attention", "linear_attention"]);
        let mut caches = make_caches_with_tq(&cfg, 1).unwrap();
        // Both full-attention slots (regular + TQ) should expose &mut dyn.
        assert!(caches[0].full_attention_kv_mut().is_some());
        // Linear attention returns None.
        assert!(caches[1].full_attention_kv_mut().is_none());
    }

    #[test]
    fn linear_attn_cache_starts_empty() {
        let c = LinearAttnCache::new();
        assert!(c.conv_state.is_none());
        assert!(c.recurrent_state.is_none());
        assert_eq!(c.offset, 0);
    }
}
