//! Gemma 4 per-layer cache factory.
//! Global → shared `FullAttnCache`; sliding → `RotatingKVCache`
//! (quantised rotating not implemented; sliding ignores `opts.kind`).
//! Shared-KV slots are `None`; upstream layers own the state.

use mlxr::Array;

use crate::cache::{CacheOptions, FullAttnCache, KeyValueCache, RotatingKVCache};
use crate::gemma4::text::config::{LayerKind, TextConfig};

/// Build caches with an externally-provided Π. Caller shares the matrix
/// across resets by holding it on the adapter.
pub(crate) fn make_caches_with_rotation(
    args: &TextConfig,
    opts: CacheOptions,
    rotation: Option<&Array>,
) -> Vec<Option<LayerCache>> {
    let first_kv_shared = (args.num_hidden_layers - args.num_kv_shared_layers).max(0);
    let layer_types = args.layer_types_resolved();
    (0..args.num_hidden_layers as usize)
        .map(|i| {
            if (i as i32) >= first_kv_shared && args.num_kv_shared_layers > 0 {
                None
            } else {
                Some(match layer_types[i] {
                    LayerKind::FullAttention => {
                        LayerCache::Global(FullAttnCache::from_options(opts, rotation.cloned()))
                    }
                    LayerKind::SlidingAttention => {
                        LayerCache::Sliding(RotatingKVCache::new(args.sliding_window, 0))
                    }
                })
            }
        })
        .collect()
}


#[derive(Debug, Clone)]
pub enum LayerCache {
    Global(FullAttnCache),
    Sliding(RotatingKVCache),
}

impl Default for LayerCache {
    fn default() -> Self {
        Self::Global(FullAttnCache::default())
    }
}

impl KeyValueCache for LayerCache {
    crate::delegate_kv!(LayerCache { Global, Sliding });
}
