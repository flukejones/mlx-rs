//! The [`KeyValueCache`] trait + the blanket `&mut T` forwarding impl.

use std::collections::HashMap;

use mlxr::{
    error::Exception,
    fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask},
    Array,
};

/// Ceiling-divide `s` up to the next multiple of `step`.
///
/// Shared by [`super::kvcache::KVCache`] and
/// [`super::quantized_kvcache::QuantizedKVCache`] for buffer-growth
/// rounding.
#[inline]
pub(super) fn ceil_step(s: i32, step: i32) -> i32 {
    ((s + step - 1) / step) * step
}

/// `debug_assertions` only: panic with a clear message when the mask's
/// key axis does not match the cache-concatenated K seq len.
///
/// Catches the canonical bug of building a `[L, L]` causal mask without
/// `cache.offset()`: turn 1 (offset 0) silently passes, turn 2 fails
/// deep inside SDPA with a cryptic `broadcast_shapes` exception.
#[inline]
pub(super) fn assert_mask_matches_keys(mask: Option<&Array>, k_full: &Array) {
    if !cfg!(debug_assertions) {
        return;
    }
    let Some(mask) = mask else { return };
    let m_shape = mask.shape();
    let k_shape = k_full.shape();
    let m_last = m_shape.last().copied().unwrap_or(0);
    let k_last = k_shape[k_shape.len() - 2];
    debug_assert!(
        m_last == k_last,
        "mask key axis ({m_last}) does not match cache-concatenated K seq len ({k_last}); \
         likely missing cache.offset() in mask construction. \
         mask shape {m_shape:?}, k_full shape {k_shape:?}",
    );
}

// TODO: somehow move quantized methods to a separate trait?
pub trait KeyValueCache {
    fn is_quantized(&self) -> bool {
        false
    }

    /// Returns the group size used for quantization. `None` if not quantized.
    fn group_size(&self) -> Option<i32> {
        None
    }

    /// Returns the number of bits used for quantization. `None` if not quantized.
    fn bits(&self) -> Option<i32> {
        None
    }

    fn offset(&self) -> i32;

    fn max_size(&self) -> Option<i32>;

    fn update_and_fetch(&mut self, keys: Array, values: Array)
        -> Result<(Array, Array), Exception>;

    /// Returns `true` if this cache supports [`trim`](Self::trim) with a
    /// non-zero argument. Default implementations return `false`; pre-
    /// allocated caches override.
    fn is_trimmable(&self) -> bool {
        false
    }

    /// Drop the trailing `n` tokens from the cache. Returns the number of
    /// tokens actually trimmed (may be less than `n` if the cache is
    /// shorter). Default implementation is a no-op.
    fn trim(&mut self, _n: i32) -> i32 {
        0
    }

    /// Stable identifier for this cache kind. Persisted as metadata
    /// and used by [`KeyValueCache::from_state`] to dispatch on load.
    fn class_name(&self) -> &'static str {
        "DefaultCache"
    }

    /// Per-layer arrays to persist. Order is significant — must match the
    /// order [`KeyValueCache::from_state`] consumes. Default returns empty.
    fn state(&self) -> Vec<Array> {
        Vec::new()
    }

    /// String-keyed scalars (offset, step, bits, etc.) to persist alongside
    /// [`state`](Self::state). Default returns empty.
    fn meta_state(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    /// `softmax(scaled_q @ K.T) @ V` over the full cached history.
    /// Default appends K/V then dispatches dense fused SDPA. Quantised
    /// caches override to skip K/V dequant on the hot path.
    fn attention(
        &mut self,
        queries: &Array,
        keys: Array,
        values: Array,
        scale: f32,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let (k_full, v_full) = self.update_and_fetch(keys, values)?;
        assert_mask_matches_keys(mask, &k_full);
        scaled_dot_product_attention(
            queries,
            k_full,
            v_full,
            scale,
            mask.map(ScaledDotProductAttentionMask::Array),
            None,
        )
    }
}

impl<T> KeyValueCache for &'_ mut T
where
    T: KeyValueCache,
{
    fn is_quantized(&self) -> bool {
        T::is_quantized(self)
    }

    fn group_size(&self) -> Option<i32> {
        T::group_size(self)
    }

    fn bits(&self) -> Option<i32> {
        T::bits(self)
    }

    fn offset(&self) -> i32 {
        T::offset(self)
    }

    fn max_size(&self) -> Option<i32> {
        T::max_size(self)
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        T::update_and_fetch(self, keys, values)
    }

    fn is_trimmable(&self) -> bool {
        T::is_trimmable(self)
    }

    fn trim(&mut self, n: i32) -> i32 {
        T::trim(self, n)
    }

    fn class_name(&self) -> &'static str {
        T::class_name(self)
    }

    fn state(&self) -> Vec<Array> {
        T::state(self)
    }

    fn meta_state(&self) -> HashMap<String, String> {
        T::meta_state(self)
    }

    fn attention(
        &mut self,
        queries: &Array,
        keys: Array,
        values: Array,
        scale: f32,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        T::attention(self, queries, keys, values, scale, mask)
    }
}
