//! `CacheOptions` — KV-cache backing + per-cache toggles.

use crate::cache::KeyValueCache;

/// Backing kind for full-attention layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CacheKind {
    #[default]
    Dense,
    Quantized {
        group_size: i32,
        bits: i32,
    },
}

impl CacheKind {
    pub fn quantized_q8() -> Self {
        Self::Quantized {
            group_size: 64,
            bits: 8,
        }
    }

    pub fn quantized_q4() -> Self {
        Self::Quantized {
            group_size: 64,
            bits: 4,
        }
    }
}

/// Default prefill chunk cap when neither user nor cache imposes one.
/// 2048 fits comfortably in unified RAM for 27–35B models at bf16/q8.
pub const DEFAULT_PREFILL_CHUNK: i32 = 2048;

#[derive(Debug, Clone, Copy)]
pub struct CacheOptions {
    pub kind: CacheKind,
    pub steel_prefill: bool,
    pub fused_kernel: bool,
    /// `Some(seed)` enables TurboQuant random-orthogonal Π rotation
    /// applied to K/V pre-quantize. Ignored when `kind == Dense`.
    pub turbo_quant_seed: Option<u64>,
    /// Max tokens per prefill forward pass. `None` = single-pass
    /// (caller manages memory). Combined with cache `max_size()` via
    /// `min`, so sliding-window caps still apply.
    pub max_prefill_chunk: Option<i32>,
}

impl Default for CacheOptions {
    fn default() -> Self {
        Self {
            kind: CacheKind::Dense,
            steel_prefill: false,
            fused_kernel: false,
            turbo_quant_seed: None,
            max_prefill_chunk: Some(DEFAULT_PREFILL_CHUNK),
        }
    }
}

impl CacheOptions {
    pub fn standard_with_steel_prefill() -> Self {
        Self {
            steel_prefill: true,
            ..Self::default()
        }
    }

    pub fn quantized_q8() -> Self {
        Self {
            kind: CacheKind::quantized_q8(),
            ..Self::default()
        }
    }

    pub fn quantized_q4() -> Self {
        Self {
            kind: CacheKind::quantized_q4(),
            ..Self::default()
        }
    }

    pub fn quantized_q8_with_turbo(seed: u64) -> Self {
        Self {
            turbo_quant_seed: Some(seed),
            ..Self::quantized_q8()
        }
    }

    pub fn quantized_q4_with_turbo(seed: u64) -> Self {
        Self {
            turbo_quant_seed: Some(seed),
            ..Self::quantized_q4()
        }
    }
}

/// Effective prefill chunk size: min of the user cap and any cache's
/// `max_size()`. `None` iff both are `None`.
pub fn effective_prefill_chunk<C: KeyValueCache>(
    caches: &[C],
    user_cap: Option<i32>,
) -> Option<i32> {
    let cache_cap = caches.iter().filter_map(|c| c.max_size()).min();
    match (user_cap, cache_cap) {
        (Some(u), Some(c)) => Some(u.min(c)),
        (Some(u), None) => Some(u),
        (None, Some(c)) => Some(c),
        (None, None) => None,
    }
}

/// Like [`effective_prefill_chunk`] but operates on `[Option<C>]`
/// (gemma4's shared-KV layout where some slots are `None`).
pub fn effective_prefill_chunk_opt<C: KeyValueCache>(
    caches: &[Option<C>],
    user_cap: Option<i32>,
) -> Option<i32> {
    let cache_cap = caches.iter().filter_map(|c| c.as_ref()?.max_size()).min();
    match (user_cap, cache_cap) {
        (Some(u), Some(c)) => Some(u.min(c)),
        (Some(u), None) => Some(u),
        (None, Some(c)) => Some(c),
        (None, None) => None,
    }
}
