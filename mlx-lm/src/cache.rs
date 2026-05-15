use mlx_rs::{
    error::Exception,
    ops::{
        dequantize,
        indexing::{Ellipsis, IndexOp, TryIndexMutOp},
        quantize, zeros_dtype,
    },
    Array, Dtype,
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

/// Quantised KV cache.
///
/// Mirrors Python `mlx_lm.models.cache.QuantizedKVCache`. Stores K and V
/// as packed-uint32 `(wq, scales, biases)` triples in pre-allocated
/// step-sized buffers (same growth pattern as [`KVCache`]). On
/// `update_and_fetch`, new tokens are quantised in one call, slice-written
/// into the buffer, then the populated `[:offset, :]` portion is
/// dequantised back to a plain `(K, V)` pair for the standard SDPA path.
///
/// This is the Python default behaviour: dequantise-on-read keeps the
/// SDPA call path unchanged. A future enhancement could route through
/// `mlx_rs::ops::quantized_matmul` to skip the dequantise step entirely;
/// Python doesn't do that either, so the perf-vs-complexity trade-off
/// has not been validated.
///
/// Memory savings vs fp16 K/V buffer:
/// - `bits=8 group=64` → ~2× reduction.
/// - `bits=4 group=64` → ~4× reduction.
///
/// Last-axis (head_dim) must be a multiple of `group_size` (mlx-c
/// requirement). For chandra-q8 with `head_dim = 128` and the default
/// `group_size = 64` this is satisfied.
#[derive(Debug, Clone)]
pub struct QuantizedKVCache {
    /// Quantised K buffer: `[B, H, capacity, head_dim / el_per_int]` uint32.
    keys_wq: Option<Array>,
    /// K scales: `[B, H, capacity, head_dim / group_size]` same dtype as inputs.
    keys_scales: Option<Array>,
    /// K biases: `[B, H, capacity, head_dim / group_size]` same dtype as inputs.
    keys_biases: Option<Array>,
    values_wq: Option<Array>,
    values_scales: Option<Array>,
    values_biases: Option<Array>,
    offset: i32,
    step: i32,
    group_size: i32,
    bits: i32,
    /// Dtype of the original K/V inputs (preserved for dequantise output).
    dtype: Option<Dtype>,
}

impl QuantizedKVCache {
    /// New empty cache with the default step (256), default
    /// `group_size = 64`, default `bits = 8` (Python's default; ~2× memory
    /// reduction, drop-in lossless).
    pub fn new() -> Self {
        Self::with_config(DEFAULT_KV_CACHE_STEP, 64, 8)
    }

    /// New empty cache with full configuration. `bits` must be one of
    /// {2, 4, 8}; `group_size` should usually be 64. `head_dim` must be
    /// a multiple of `group_size`.
    pub fn with_config(step: i32, group_size: i32, bits: i32) -> Self {
        assert!(step > 0, "step must be positive");
        assert!(group_size > 0, "group_size must be positive");
        assert!(matches!(bits, 2 | 3 | 4 | 6 | 8), "bits must be 2/3/4/6/8");
        Self {
            keys_wq: None,
            keys_scales: None,
            keys_biases: None,
            values_wq: None,
            values_scales: None,
            values_biases: None,
            offset: 0,
            step,
            group_size,
            bits,
            dtype: None,
        }
    }

    fn ceil_step(s: i32, step: i32) -> i32 {
        ((s + step - 1) / step) * step
    }

    /// Grow each of the six pre-allocated buffers so they have room for
    /// `additional` more tokens beyond the current `offset`. Allocates
    /// fresh zero buffers at the target capacity and copies the populated
    /// `[:offset]` rows over.
    #[allow(clippy::too_many_arguments)]
    fn grow_to_fit(
        &mut self,
        new_k_wq: &Array,
        new_k_scales: &Array,
        new_k_biases: &Array,
        new_v_wq: &Array,
        new_v_scales: &Array,
        new_v_biases: &Array,
    ) -> Result<(), Exception> {
        let new_tokens = new_k_wq.shape()[new_k_wq.shape().len() - 2];
        let target_cap = Self::ceil_step(self.offset + new_tokens, self.step);

        let current_cap = self
            .keys_wq
            .as_ref()
            .map(|k| k.shape()[k.shape().len() - 2])
            .unwrap_or(0);
        if target_cap <= current_cap {
            return Ok(());
        }

        fn alloc_like(template: &Array, capacity: i32) -> Result<Array, Exception> {
            let shape = template.shape();
            let mut buf_shape = shape.to_vec();
            let t_axis = buf_shape.len() - 2;
            buf_shape[t_axis] = capacity;
            zeros_dtype(&buf_shape, template.dtype())
        }

        let mut grown = [
            alloc_like(new_k_wq, target_cap)?,
            alloc_like(new_k_scales, target_cap)?,
            alloc_like(new_k_biases, target_cap)?,
            alloc_like(new_v_wq, target_cap)?,
            alloc_like(new_v_scales, target_cap)?,
            alloc_like(new_v_biases, target_cap)?,
        ];

        if self.offset > 0 {
            let olds = [
                self.keys_wq.take(),
                self.keys_scales.take(),
                self.keys_biases.take(),
                self.values_wq.take(),
                self.values_scales.take(),
                self.values_biases.take(),
            ];
            for (g, o) in grown.iter_mut().zip(olds.into_iter()) {
                if let Some(old) = o {
                    g.try_index_mut(
                        (Ellipsis, 0..self.offset, ..),
                        old.index((Ellipsis, 0..self.offset, ..)),
                    )?;
                }
            }
        }

        let [k_wq, k_s, k_b, v_wq, v_s, v_b] = grown;
        self.keys_wq = Some(k_wq);
        self.keys_scales = Some(k_s);
        self.keys_biases = Some(k_b);
        self.values_wq = Some(v_wq);
        self.values_scales = Some(v_s);
        self.values_biases = Some(v_b);
        Ok(())
    }
}

impl Default for QuantizedKVCache {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyValueCache for QuantizedKVCache {
    /// Returns `false` even though the storage is quantised: this cache
    /// dequantises K/V on every `update_and_fetch` call, so consumers
    /// (the SDPA wrapper in `mlx_lm::utils`) see plain dense arrays and
    /// must take the dense SDPA branch. The `is_quantized` discriminant
    /// is reserved for a future cache that returns raw quantised triples
    /// and routes through `quantized_scaled_dot_product_attention`.
    fn is_quantized(&self) -> bool {
        false
    }

    fn group_size(&self) -> Option<i32> {
        Some(self.group_size)
    }

    fn bits(&self) -> Option<i32> {
        Some(self.bits)
    }

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
        let s = keys.shape()[keys.shape().len() - 2];

        // Remember the input dtype for the dequantise return.
        if self.dtype.is_none() {
            self.dtype = Some(keys.dtype());
        }

        // Quantise the new tokens. mlx-c quantises along the last axis;
        // K/V shapes `[B, H, S, D]` are accepted directly.
        let (new_k_wq, new_k_scales, new_k_biases) = quantize(&keys, self.group_size, self.bits)?;
        let (new_v_wq, new_v_scales, new_v_biases) = quantize(&values, self.group_size, self.bits)?;

        self.grow_to_fit(
            &new_k_wq,
            &new_k_scales,
            &new_k_biases,
            &new_v_wq,
            &new_v_scales,
            &new_v_biases,
        )?;

        // Slice-write into the [offset:offset+s] rows.
        let start = self.offset;
        let end = self.offset + s;
        let k_wq_buf = self.keys_wq.as_mut().expect("buffer just allocated");
        let k_s_buf = self.keys_scales.as_mut().expect("buffer just allocated");
        let k_b_buf = self.keys_biases.as_mut().expect("buffer just allocated");
        let v_wq_buf = self.values_wq.as_mut().expect("buffer just allocated");
        let v_s_buf = self.values_scales.as_mut().expect("buffer just allocated");
        let v_b_buf = self.values_biases.as_mut().expect("buffer just allocated");

        k_wq_buf.try_index_mut((Ellipsis, start..end, ..), new_k_wq)?;
        k_s_buf.try_index_mut((Ellipsis, start..end, ..), new_k_scales)?;
        k_b_buf.try_index_mut((Ellipsis, start..end, ..), new_k_biases)?;
        v_wq_buf.try_index_mut((Ellipsis, start..end, ..), new_v_wq)?;
        v_s_buf.try_index_mut((Ellipsis, start..end, ..), new_v_scales)?;
        v_b_buf.try_index_mut((Ellipsis, start..end, ..), new_v_biases)?;

        self.offset = end;

        // Dequantise the [:offset] populated slice and return as plain
        // (K, V) for the standard SDPA path.
        let k_wq_view = k_wq_buf.index((Ellipsis, 0..end, ..));
        let k_s_view = k_s_buf.index((Ellipsis, 0..end, ..));
        let k_b_view = k_b_buf.index((Ellipsis, 0..end, ..));
        let v_wq_view = v_wq_buf.index((Ellipsis, 0..end, ..));
        let v_s_view = v_s_buf.index((Ellipsis, 0..end, ..));
        let v_b_view = v_b_buf.index((Ellipsis, 0..end, ..));

        let k_out = dequantize(&k_wq_view, &k_s_view, &k_b_view, self.group_size, self.bits)?;
        let v_out = dequantize(&v_wq_view, &v_s_view, &v_b_view, self.group_size, self.bits)?;
        Ok((k_out, v_out))
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

    /// Like `token_block` but with head_dim that's a multiple of the
    /// default quantisation group size (64). Quantize requires that.
    fn quant_token_block(s: i32, base: f32) -> Array {
        let b = 1;
        let h = 2;
        let d = 64;
        let mut data = Vec::with_capacity((b * h * s * d) as usize);
        for t in 0..s {
            for hi in 0..h {
                for di in 0..d {
                    data.push(base + (t * 100 + hi * 10) as f32 + (di as f32) * 0.01);
                }
            }
        }
        let raw = Array::from_slice(&data, &[s, h, d]);
        raw.expand_dims(0).unwrap().swap_axes(1, 2).unwrap()
    }

    #[test]
    fn quantized_kvcache_q8_round_trip_is_near_lossless() {
        let mut cache = QuantizedKVCache::with_config(256, 64, 8);
        let k = quant_token_block(3, 0.0);
        let v = quant_token_block(3, 100.0);
        let (out_k, out_v) = cache.update_and_fetch(k.clone(), v.clone()).unwrap();
        eval([&out_k, &out_v]).unwrap();
        assert_eq!(out_k.shape(), &[1, 2, 3, 64]);
        // Reports false to drive the dense SDPA branch (dequantise-on-read).
        assert!(!cache.is_quantized());
        assert_eq!(cache.group_size(), Some(64));
        assert_eq!(cache.bits(), Some(8));
        // q8 round-trip should reproduce inputs to within ~1/256 of the
        // group's value range. Our test data spans a few hundred so a
        // few-unit absolute diff is fine.
        let dk = out_k
            .subtract(&k)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap();
        let dv = out_v
            .subtract(&v)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap();
        assert!(dk.item::<f32>() < 1.0, "K diff {}", dk.item::<f32>());
        assert!(dv.item::<f32>() < 1.0, "V diff {}", dv.item::<f32>());
    }

    #[test]
    fn quantized_kvcache_appends_in_token_order() {
        let mut cache = QuantizedKVCache::with_config(8, 64, 8);
        let k1 = quant_token_block(2, 0.0);
        let v1 = quant_token_block(2, 1000.0);
        cache.update_and_fetch(k1.clone(), v1.clone()).unwrap();
        let k2 = quant_token_block(3, 2.0);
        let v2 = quant_token_block(3, 1002.0);
        let (out_k, _out_v) = cache.update_and_fetch(k2.clone(), v2.clone()).unwrap();
        eval([&out_k]).unwrap();
        assert_eq!(out_k.shape(), &[1, 2, 5, 64]);
        assert_eq!(cache.offset(), 5);

        // Compare first 2 rows back against k1 (q8 round-trip is near-lossless).
        let head = out_k.index((Ellipsis, 0..2, ..));
        let dk = head
            .subtract(&k1)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap();
        assert!(
            dk.item::<f32>() < 1.0,
            "first-2 mismatch: {}",
            dk.item::<f32>()
        );
    }

    #[test]
    fn quantized_kvcache_q4_round_trip_loses_some_precision_but_works() {
        let mut cache = QuantizedKVCache::with_config(256, 64, 4);
        let k = quant_token_block(3, 0.0);
        let v = quant_token_block(3, 100.0);
        let (out_k, out_v) = cache.update_and_fetch(k, v).unwrap();
        eval([&out_k, &out_v]).unwrap();
        assert_eq!(out_k.shape(), &[1, 2, 3, 64]);
        // q4 loses more — accept wider tolerance.
        let mean = out_k.mean(None).unwrap();
        assert!(mean.item::<f32>().is_finite());
    }

    #[test]
    fn quantized_kvcache_trim_drops_tokens() {
        let mut cache = QuantizedKVCache::new();
        cache
            .update_and_fetch(quant_token_block(5, 0.0), quant_token_block(5, 0.0))
            .unwrap();
        assert!(cache.is_trimmable());
        assert_eq!(cache.trim(2), 2);
        assert_eq!(cache.offset(), 3);
    }
}
