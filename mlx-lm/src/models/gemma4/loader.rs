//! Safetensors weight loader + tokenizer + cache factory for Gemma 4.

use std::path::Path;

use crate::cache::{KVCache, RotatingKVCache};
use crate::error::Error;
use crate::models::gemma4::config::{Gemma4Config, LayerKind};
use crate::models::gemma4::text::Model;
use crate::models::gemma4::weights::load_gemma4_model_sanitized;

/// Loads a Gemma 4 checkpoint. mlx-community checkpoints carry the
/// `language_model.model.layers.X` prefix and MoE expert weights in a
/// fused `experts.gate_up_proj` layout that the generic `load_sharded`
/// cannot interpret — both transforms live in
/// `weights::load_gemma4_model_sanitized`.
pub(crate) fn load_gemma4_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    load_gemma4_model_sanitized(model_dir)
}

/// One cache slot per non-shared layer. Shared-KV layers share the
/// underlying KV state of an earlier same-kind layer at forward time
/// via `Gemma4TextModel::previous_kvs`; the cache vec only owns the
/// upstream slots (`num_hidden_layers - num_kv_shared_layers`).
pub(crate) fn make_gemma4_caches(args: &Gemma4Config) -> Vec<Option<Gemma4LayerCache>> {
    let first_kv_shared = (args.num_hidden_layers - args.num_kv_shared_layers).max(0);
    let layer_types = args.layer_types_resolved();
    (0..args.num_hidden_layers as usize)
        .map(|i| {
            if (i as i32) >= first_kv_shared && args.num_kv_shared_layers > 0 {
                None
            } else {
                Some(match layer_types[i] {
                    // Full attention: steel `causal=true` ≡ standard causal mask.
                    LayerKind::FullAttention => {
                        Gemma4LayerCache::Global(KVCache::new().with_steel_prefill())
                    }
                    // Sliding attention: steel `causal=true` ≠ sliding-causal once
                    // prompt > sliding_window. Keep `fast::SDPA` for correctness.
                    LayerKind::SlidingAttention => {
                        Gemma4LayerCache::Sliding(RotatingKVCache::new(args.sliding_window, 0))
                    }
                })
            }
        })
        .collect()
}

#[derive(Debug, Clone)]
pub enum Gemma4LayerCache {
    Global(KVCache),
    Sliding(RotatingKVCache),
}

impl Default for Gemma4LayerCache {
    fn default() -> Self {
        Self::Global(KVCache::new())
    }
}

impl crate::cache::KeyValueCache for Gemma4LayerCache {
    fn update_and_fetch(
        &mut self,
        keys: mlx_rs::Array,
        values: mlx_rs::Array,
    ) -> Result<(mlx_rs::Array, mlx_rs::Array), mlx_rs::error::Exception> {
        match self {
            Self::Global(c) => c.update_and_fetch(keys, values),
            Self::Sliding(c) => c.update_and_fetch(keys, values),
        }
    }

    fn offset(&self) -> i32 {
        match self {
            Self::Global(c) => c.offset(),
            Self::Sliding(c) => c.offset(),
        }
    }

    fn max_size(&self) -> Option<i32> {
        match self {
            Self::Global(c) => c.max_size(),
            Self::Sliding(c) => c.max_size(),
        }
    }

    fn class_name(&self) -> &'static str {
        match self {
            Self::Global(_) => "Gemma4LayerCache::Global",
            Self::Sliding(_) => "Gemma4LayerCache::Sliding",
        }
    }
}
