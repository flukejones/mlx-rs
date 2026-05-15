use mlx_rs::{
    error::Exception,
    ops::{
        indexing::{Ellipsis, IndexOp, TryIndexMutOp},
        zeros_dtype,
    },
    Array,
};

/// Default step in tokens for [`KVCache`]'s pre-allocated buffer growth.
pub const DEFAULT_KV_CACHE_STEP: i32 = 256;

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
}

/// Legacy alias for [`KVCache`].
///
/// Pre-migration this was a naive concat-every-step cache. The
/// implementation is now the pre-allocated step-based [`KVCache`];
/// the type alias is kept so out-of-tree code that named the old type
/// continues to compile.
pub type ConcatKeyValueCache = KVCache;

/// Pre-allocated KV cache with step-based growth.
///
/// Mirrors Python `mlx_lm.models.cache.KVCache`: the underlying `keys` /
/// `values` buffers are `[B, H, capacity, D]` `Array`s pre-allocated in
/// chunks of `step` rows. On each `update_and_fetch` call we grow the
/// buffers if `offset + S > capacity`, slice-write the new tokens into
/// the `[offset:offset+S]` axis, and return a slice view of the populated
/// `[:offset+S]` rows.
///
/// This replaces the naive per-step `concatenate_axis` allocation in
/// [`ConcatKeyValueCache`] and is the recommended cache for all
/// decoder-only models. The amortised cost of `step`-sized buffer growth
/// is much smaller than per-token concat, especially at S≥256.
#[derive(Debug, Clone)]
pub struct KVCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
    /// Buffer-growth step in tokens.
    step: i32,
}

impl Default for KVCache {
    fn default() -> Self {
        Self::with_step(DEFAULT_KV_CACHE_STEP)
    }
}

impl KVCache {
    /// New empty cache using the default step ([`DEFAULT_KV_CACHE_STEP`]).
    pub fn new() -> Self {
        Self::default()
    }

    /// New empty cache with a custom step. `step` must be positive.
    pub fn with_step(step: i32) -> Self {
        assert!(step > 0, "KVCache step must be positive");
        Self {
            keys: None,
            values: None,
            offset: 0,
            step,
        }
    }

    /// Configured step (buffer-growth chunk size).
    pub fn step(&self) -> i32 {
        self.step
    }

    /// Current allocated capacity along the token axis. `0` if unallocated.
    pub fn capacity(&self) -> i32 {
        match self.keys.as_ref() {
            Some(k) => {
                let shape = k.shape();
                shape[shape.len() - 2]
            }
            None => 0,
        }
    }

    /// Ceiling-divide `s` up to the next multiple of `step`.
    fn ceil_step(s: i32, step: i32) -> i32 {
        ((s + step - 1) / step) * step
    }

    /// Allocate fresh `[B, H, capacity, D]` zero buffers matching `template`.
    fn alloc_like(template: &Array, capacity: i32) -> Result<Array, Exception> {
        let shape = template.shape();
        let mut buf_shape = shape.to_vec();
        let t_axis = buf_shape.len() - 2;
        buf_shape[t_axis] = capacity;
        zeros_dtype(&buf_shape, template.dtype())
    }

    /// Grow the pre-allocated buffers so they have room for `additional`
    /// more tokens beyond the current `offset`. No-op if already large
    /// enough.
    fn grow_to_fit(
        keys: &mut Option<Array>,
        values: &mut Option<Array>,
        offset: i32,
        new_tokens: &Array,
        new_values: &Array,
        step: i32,
    ) -> Result<(), Exception> {
        let required = offset + new_tokens.shape()[new_tokens.shape().len() - 2];
        let target_cap = Self::ceil_step(required, step);

        let current_cap = keys
            .as_ref()
            .map(|k| k.shape()[k.shape().len() - 2])
            .unwrap_or(0);

        if target_cap <= current_cap {
            return Ok(());
        }

        let new_keys = Self::alloc_like(new_tokens, target_cap)?;
        let new_values_buf = Self::alloc_like(new_values, target_cap)?;

        // Copy existing populated rows into the new larger buffer.
        if let (Some(old_k), Some(old_v)) = (keys.take(), values.take()) {
            if offset > 0 {
                let mut grown_k = new_keys;
                let mut grown_v = new_values_buf;
                grown_k.try_index_mut(
                    (Ellipsis, 0..offset, ..),
                    old_k.index((Ellipsis, 0..offset, ..)),
                )?;
                grown_v.try_index_mut(
                    (Ellipsis, 0..offset, ..),
                    old_v.index((Ellipsis, 0..offset, ..)),
                )?;
                *keys = Some(grown_k);
                *values = Some(grown_v);
                return Ok(());
            }
        }

        *keys = Some(new_keys);
        *values = Some(new_values_buf);
        Ok(())
    }
}

impl KeyValueCache for KVCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn is_trimmable(&self) -> bool {
        true
    }

    fn trim(&mut self, n: i32) -> i32 {
        let trimmed = n.min(self.offset).max(0);
        self.offset -= trimmed;
        trimmed
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        let key_shape = keys.shape();
        let t_axis = key_shape.len() - 2;
        let s = key_shape[t_axis];

        Self::grow_to_fit(
            &mut self.keys,
            &mut self.values,
            self.offset,
            &keys,
            &values,
            self.step,
        )?;

        let buf_k = self.keys.as_mut().expect("keys buffer just allocated");
        let buf_v = self.values.as_mut().expect("values buffer just allocated");

        buf_k.try_index_mut((Ellipsis, self.offset..self.offset + s, ..), keys)?;
        buf_v.try_index_mut((Ellipsis, self.offset..self.offset + s, ..), values)?;

        self.offset += s;

        let end = self.offset;
        Ok((
            buf_k.index((Ellipsis, 0..end, ..)),
            buf_v.index((Ellipsis, 0..end, ..)),
        ))
    }
}

/// TODO: A generic KV Cache
pub struct DefaultKeyValueCache {}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::{ops::concatenate_axis, transforms::eval, Dtype};

    /// Make a fresh `[B=1, H=2, S, D=4]` float32 array filled with sequential
    /// values per token-row, distinct from any other call (so we can spot
    /// out-of-order writes).
    fn token_block(s: i32, base: f32) -> Array {
        let b = 1;
        let h = 2;
        let d = 4;
        let mut data = Vec::with_capacity((b * h * s * d) as usize);
        for t in 0..s {
            for hi in 0..h {
                for di in 0..d {
                    data.push(base + (t * 1000 + hi * 100 + di) as f32);
                }
            }
        }
        // Layout above is `[t, h, d]`; reshape to `[B, H, S, D]` by writing
        // axis order then transposing in.
        let raw = Array::from_slice(&data, &[s, h, d]);
        let with_batch = raw.expand_dims(0).unwrap();
        // Swap axes 1 and 2 to get `[B, H, S, D]`.
        with_batch.swap_axes(1, 2).unwrap()
    }

    #[test]
    fn kvcache_first_update_returns_input_rows() {
        let mut cache = KVCache::new();
        let k = token_block(3, 0.0);
        let v = token_block(3, 100.0);
        let (out_k, out_v) = cache.update_and_fetch(k.clone(), v.clone()).unwrap();
        eval([&out_k, &out_v]).unwrap();
        assert_eq!(out_k.shape(), &[1, 2, 3, 4]);
        assert_eq!(out_v.shape(), &[1, 2, 3, 4]);
        assert_eq!(cache.offset(), 3);
        assert_eq!(cache.capacity(), 256);
        // Out should equal input (only S rows are populated, returned view is exactly those).
        let diff = out_k
            .subtract(&k)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap();
        assert!(diff.item::<f32>() < 1e-6);
    }

    #[test]
    fn kvcache_appends_across_updates_in_token_order() {
        let mut cache = KVCache::new();
        let k1 = token_block(2, 0.0);
        let v1 = token_block(2, 1000.0);
        cache.update_and_fetch(k1.clone(), v1.clone()).unwrap();

        let k2 = token_block(3, 2.0);
        let v2 = token_block(3, 1002.0);
        let (out_k, out_v) = cache.update_and_fetch(k2.clone(), v2.clone()).unwrap();
        eval([&out_k, &out_v]).unwrap();

        assert_eq!(out_k.shape(), &[1, 2, 5, 4]);
        assert_eq!(cache.offset(), 5);

        // Reconstruct expected by concatenating in token-order along axis 2.
        let expected_k = concatenate_axis(&[k1, k2], -2).unwrap();
        let expected_v = concatenate_axis(&[v1, v2], -2).unwrap();
        let dk = out_k
            .subtract(&expected_k)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap();
        let dv = out_v
            .subtract(&expected_v)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap();
        assert!(dk.item::<f32>() < 1e-6, "K mismatch: {}", dk.item::<f32>());
        assert!(dv.item::<f32>() < 1e-6, "V mismatch: {}", dv.item::<f32>());
    }

    #[test]
    fn kvcache_grows_buffer_past_initial_step() {
        let mut cache = KVCache::with_step(4);
        // First update: 3 tokens -> capacity should round up to 4.
        cache
            .update_and_fetch(token_block(3, 0.0), token_block(3, 0.0))
            .unwrap();
        assert_eq!(cache.capacity(), 4);
        // Second update: 5 more tokens -> total 8 -> capacity should round up to 8.
        cache
            .update_and_fetch(token_block(5, 100.0), token_block(5, 100.0))
            .unwrap();
        assert_eq!(cache.offset(), 8);
        assert_eq!(cache.capacity(), 8);
    }

    #[test]
    fn kvcache_trim_drops_trailing_tokens() {
        let mut cache = KVCache::new();
        cache
            .update_and_fetch(token_block(10, 0.0), token_block(10, 0.0))
            .unwrap();
        assert!(cache.is_trimmable());
        assert_eq!(cache.trim(3), 3);
        assert_eq!(cache.offset(), 7);
        // Trim more than offset -> only trims what's available.
        assert_eq!(cache.trim(100), 7);
        assert_eq!(cache.offset(), 0);
        // Trim with empty cache -> 0.
        assert_eq!(cache.trim(5), 0);
    }

    #[test]
    fn kvcache_dtype_matches_inputs() {
        let mut cache = KVCache::new();
        let k = token_block(2, 0.0).as_dtype(Dtype::Bfloat16).unwrap();
        let v = token_block(2, 0.0).as_dtype(Dtype::Bfloat16).unwrap();
        let (out_k, out_v) = cache.update_and_fetch(k, v).unwrap();
        assert_eq!(out_k.dtype(), Dtype::Bfloat16);
        assert_eq!(out_v.dtype(), Dtype::Bfloat16);
    }
}
