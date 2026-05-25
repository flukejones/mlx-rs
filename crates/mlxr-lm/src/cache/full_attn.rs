//! `FullAttnCache` — shared full-attention KV slot. Standard or quantised.

use mlxr::Array;

use crate::cache::rotation;
use crate::cache::{CacheKind, CacheOptions, KVCache, KeyValueCache, QuantizedKVCache};
use crate::error::Error;

use super::DEFAULT_KV_CACHE_STEP;

#[derive(Debug, Clone)]
pub enum FullAttnCache {
    Standard(KVCache),
    Quantized(QuantizedKVCache),
}

impl FullAttnCache {
    /// Build one cache slot. `rotation` is the pre-built TurboQuant Π
    /// (shape `[head_dim, head_dim]`), shared across layers. Pass
    /// `None` for non-quantised or non-rotated configs.
    pub fn from_options(opts: CacheOptions, rotation: Option<Array>) -> Self {
        match opts.kind {
            CacheKind::Dense => {
                let mut c = KVCache::new();
                if opts.steel_prefill {
                    c = c.with_steel_prefill();
                }
                Self::Standard(c)
            }
            CacheKind::Quantized { group_size, bits } => {
                let mut c = QuantizedKVCache::with_config(DEFAULT_KV_CACHE_STEP, group_size, bits);
                if opts.steel_prefill {
                    c = c.with_steel_prefill();
                }
                if opts.fused_kernel {
                    c = c.with_fused_kernel();
                }
                if let Some(pi) = rotation {
                    c = c.with_rotation_matrix(pi);
                }
                Self::Quantized(c)
            }
        }
    }
}

/// Build the rotation matrix `opts` calls for, if any.
/// Returns `Ok(None)` for non-quantised or no-seed configs.
/// Build once per `make_caches` call; clone the `Array` handle into
/// every layer (shared_ptr — cheap).
pub fn build_rotation(opts: CacheOptions, head_dim: i32) -> Result<Option<Array>, Error> {
    match (opts.kind, opts.turbo_quant_seed) {
        (CacheKind::Quantized { .. }, Some(seed)) => {
            Ok(Some(rotation::generate_rotation_matrix(head_dim, seed)?))
        }
        _ => Ok(None),
    }
}

impl Default for FullAttnCache {
    fn default() -> Self {
        Self::Standard(KVCache::new())
    }
}

impl KeyValueCache for FullAttnCache {
    delegate_kv!(FullAttnCache { Standard, Quantized });
}
