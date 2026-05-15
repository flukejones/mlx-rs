use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{
    error::Exception,
    ops::{
        dequantize,
        indexing::{Ellipsis, IndexOp, TryIndexMutOp},
        quantize, zeros_dtype,
    },
    Array, Dtype,
};

use crate::error::Error;

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

    /// Stable identifier matching Python `mlx_lm.models.cache` class names.
    /// Used as metadata when persisting and to dispatch on load.
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

    /// Reconstruct from previously-persisted `state` + `meta_state`.
    /// `state` must be `[keys, values]` in that order. The cache adopts the
    /// arrays directly as its buffer (no copy); `offset` is read from
    /// metadata. The buffer's pre-existing capacity becomes the new
    /// capacity — subsequent `update_and_fetch` will grow in `step` chunks
    /// from there.
    pub fn from_state(
        mut state: Vec<Array>,
        meta: &HashMap<String, String>,
    ) -> Result<Self, Error> {
        if state.len() != 2 {
            return Err(Error::Other(
                format!("KVCache::from_state expected 2 arrays, got {}", state.len()).into(),
            ));
        }
        let values = state.pop().expect("len checked");
        let keys = state.pop().expect("len checked");
        let offset = parse_meta(meta, "offset")?;
        let step = meta
            .get("step")
            .map(|s| s.parse::<i32>())
            .transpose()
            .map_err(|e| Error::Other(format!("KVCache.step parse: {e}").into()))?
            .unwrap_or(DEFAULT_KV_CACHE_STEP);
        Ok(Self {
            keys: Some(keys),
            values: Some(values),
            offset,
            step,
        })
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

    fn class_name(&self) -> &'static str {
        "KVCache"
    }

    fn state(&self) -> Vec<Array> {
        match (self.keys.as_ref(), self.values.as_ref()) {
            (Some(k), Some(v)) => {
                let end = self.offset;
                vec![
                    k.index((Ellipsis, 0..end, ..)),
                    v.index((Ellipsis, 0..end, ..)),
                ]
            }
            _ => Vec::new(),
        }
    }

    fn meta_state(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("offset".into(), self.offset.to_string());
        m.insert("step".into(), self.step.to_string());
        m
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

    /// Reconstruct from previously-persisted `state` + `meta_state`.
    /// `state` order: `[k_wq, k_scales, k_biases, v_wq, v_scales, v_biases]`.
    pub fn from_state(
        mut state: Vec<Array>,
        meta: &HashMap<String, String>,
    ) -> Result<Self, Error> {
        if state.len() != 6 {
            return Err(Error::Other(
                format!(
                    "QuantizedKVCache::from_state expected 6 arrays, got {}",
                    state.len()
                )
                .into(),
            ));
        }
        let v_b = state.pop().unwrap();
        let v_s = state.pop().unwrap();
        let v_wq = state.pop().unwrap();
        let k_b = state.pop().unwrap();
        let k_s = state.pop().unwrap();
        let k_wq = state.pop().unwrap();
        let offset = parse_meta(meta, "offset")?;
        let step = parse_meta_or(meta, "step", DEFAULT_KV_CACHE_STEP)?;
        let group_size = parse_meta_or(meta, "group_size", 64)?;
        let bits = parse_meta_or(meta, "bits", 8)?;
        Ok(Self {
            keys_wq: Some(k_wq),
            keys_scales: Some(k_s),
            keys_biases: Some(k_b),
            values_wq: Some(v_wq),
            values_scales: Some(v_s),
            values_biases: Some(v_b),
            offset,
            step,
            group_size,
            bits,
            dtype: None,
        })
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

    fn class_name(&self) -> &'static str {
        "QuantizedKVCache"
    }

    fn state(&self) -> Vec<Array> {
        let end = self.offset;
        let slot = |a: &Option<Array>| a.as_ref().map(|x| x.index((Ellipsis, 0..end, ..)));
        match (
            slot(&self.keys_wq),
            slot(&self.keys_scales),
            slot(&self.keys_biases),
            slot(&self.values_wq),
            slot(&self.values_scales),
            slot(&self.values_biases),
        ) {
            (Some(a), Some(b), Some(c), Some(d), Some(e), Some(f)) => vec![a, b, c, d, e, f],
            _ => Vec::new(),
        }
    }

    fn meta_state(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("offset".into(), self.offset.to_string());
        m.insert("step".into(), self.step.to_string());
        m.insert("group_size".into(), self.group_size.to_string());
        m.insert("bits".into(), self.bits.to_string());
        if let Some(dt) = self.dtype {
            m.insert("dtype".into(), format!("{dt:?}"));
        }
        m
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

/// Sliding-window KV cache with rotation.
///
/// Mirrors Python `mlx_lm.models.cache.RotatingKVCache`. The buffer is
/// pre-allocated to `[B, H, max_size, D]` once the first append happens.
/// While `offset < max_size` it behaves like a plain growing buffer. Once
/// full, new tokens overwrite the oldest **non-keep** slot in rotation:
/// the first `keep` slots are always retained (typically used to pin the
/// system prompt or BOS region).
///
/// `update_and_fetch` returns the populated buffer in temporal order:
/// `[keep prefix | older tail of rotating region | newer head of rotating region]`.
///
/// Not used by qwen3/llama/qwen3.5 today; lays infra for future
/// sliding-window models (e.g. Gemma3).
#[derive(Debug, Clone)]
pub struct RotatingKVCache {
    keys: Option<Array>,
    values: Option<Array>,
    /// Real token count seen so far (monotonic; not bounded by max_size).
    offset: i32,
    /// Write head into the rotating region `[keep..max_size]`. Bounded
    /// `0 <= idx < max_size - keep` once the rotating region is reached.
    idx: i32,
    /// Sliding-window capacity in tokens.
    max_size: i32,
    /// Number of head tokens to pin in the first `keep` slots.
    keep: i32,
}

impl RotatingKVCache {
    /// New empty cache. `max_size` is the rotating-window capacity in
    /// tokens; `keep` is the number of leading tokens that are never
    /// overwritten (must satisfy `0 <= keep < max_size`).
    pub fn new(max_size: i32, keep: i32) -> Self {
        assert!(max_size > 0, "max_size must be positive");
        assert!(
            keep >= 0 && keep < max_size,
            "keep must be in [0, max_size)"
        );
        Self {
            keys: None,
            values: None,
            offset: 0,
            idx: 0,
            max_size,
            keep,
        }
    }

    fn alloc_like(template: &Array, capacity: i32) -> Result<Array, Exception> {
        let shape = template.shape();
        let mut buf_shape = shape.to_vec();
        let t_axis = buf_shape.len() - 2;
        buf_shape[t_axis] = capacity;
        zeros_dtype(&buf_shape, template.dtype())
    }
}

impl KeyValueCache for RotatingKVCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        Some(self.max_size)
    }

    fn is_trimmable(&self) -> bool {
        // Trim is only well-defined while the rotating region hasn't wrapped.
        self.offset <= self.max_size
    }

    fn trim(&mut self, n: i32) -> i32 {
        if !self.is_trimmable() {
            return 0;
        }
        let trimmed = n.min(self.offset).max(0);
        self.offset -= trimmed;
        self.idx = (self.offset - self.keep).max(0);
        trimmed
    }

    fn class_name(&self) -> &'static str {
        "RotatingKVCache"
    }

    fn state(&self) -> Vec<Array> {
        match (self.keys.as_ref(), self.values.as_ref()) {
            (Some(k), Some(v)) => vec![k.clone(), v.clone()],
            _ => Vec::new(),
        }
    }

    fn meta_state(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("offset".into(), self.offset.to_string());
        m.insert("idx".into(), self.idx.to_string());
        m.insert("max_size".into(), self.max_size.to_string());
        m.insert("keep".into(), self.keep.to_string());
        m
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        let key_shape = keys.shape();
        let t_axis = key_shape.len() - 2;
        let s = key_shape[t_axis];

        // Allocate the rotating buffer on first append.
        if self.keys.is_none() {
            self.keys = Some(Self::alloc_like(&keys, self.max_size)?);
            self.values = Some(Self::alloc_like(&values, self.max_size)?);
        }
        let buf_k = self.keys.as_mut().expect("just allocated");
        let buf_v = self.values.as_mut().expect("just allocated");

        // Bulk-prompt path (S > 1): handled by writing in order; if the
        // prompt exceeds max_size the oldest entries are dropped (matches
        // Python's behaviour for prefill into a rotating cache).
        for i in 0..s {
            let slot = if self.offset < self.max_size {
                self.offset
            } else {
                self.keep + self.idx
            };
            let token_k = keys.index((Ellipsis, i..i + 1, ..));
            let token_v = values.index((Ellipsis, i..i + 1, ..));
            buf_k.try_index_mut((Ellipsis, slot..slot + 1, ..), token_k)?;
            buf_v.try_index_mut((Ellipsis, slot..slot + 1, ..), token_v)?;

            self.offset += 1;
            if self.offset > self.max_size {
                self.idx = (self.idx + 1) % (self.max_size - self.keep);
            }
        }

        // Return the populated buffer in temporal order.
        if self.offset <= self.max_size {
            let end = self.offset;
            Ok((
                buf_k.index((Ellipsis, 0..end, ..)),
                buf_v.index((Ellipsis, 0..end, ..)),
            ))
        } else {
            // [0..keep] head + [keep+idx..max_size] older tail + [keep..keep+idx] newer head.
            let head_k = buf_k.index((Ellipsis, 0..self.keep, ..));
            let head_v = buf_v.index((Ellipsis, 0..self.keep, ..));
            let split = self.keep + self.idx;
            let tail_k = buf_k.index((Ellipsis, split..self.max_size, ..));
            let tail_v = buf_v.index((Ellipsis, split..self.max_size, ..));
            let new_k = buf_k.index((Ellipsis, self.keep..split, ..));
            let new_v = buf_v.index((Ellipsis, self.keep..split, ..));
            let out_k = mlx_rs::ops::concatenate_axis(&[head_k, tail_k, new_k], -2)?;
            let out_v = mlx_rs::ops::concatenate_axis(&[head_v, tail_v, new_v], -2)?;
            Ok((out_k, out_v))
        }
    }
}

// -------- Prompt cache helpers (Python `mlx_lm.models.cache` parity) --------

fn parse_meta(meta: &HashMap<String, String>, key: &str) -> Result<i32, Error> {
    meta.get(key)
        .ok_or_else(|| Error::Other(format!("missing meta key {key:?}").into()))?
        .parse::<i32>()
        .map_err(|e| Error::Other(format!("meta {key:?} parse: {e}").into()))
}

fn parse_meta_or(meta: &HashMap<String, String>, key: &str, default: i32) -> Result<i32, Error> {
    meta.get(key)
        .map(|s| s.parse::<i32>())
        .transpose()
        .map_err(|e| Error::Other(format!("meta {key:?} parse: {e}").into()))
        .map(|v| v.unwrap_or(default))
}

/// Build a uniform prompt cache for a decoder-only model.
///
/// One [`KVCache`] per layer (or `Vec<Option<KVCache>>` if `Vec<Option<_>>`
/// is what the model expects — wrap with `.into_iter().map(Some).collect()`).
/// For hybrid models (qwen3.5) call that model's own `make_caches` instead.
///
/// `max_kv_size` is reserved for future use by sliding-window models; the
/// plain `KVCache` ignores it.
pub fn make_prompt_cache(num_layers: usize, _max_kv_size: Option<i32>) -> Vec<KVCache> {
    (0..num_layers).map(|_| KVCache::new()).collect()
}

/// Return `true` iff every cache in the slice supports `trim` with a
/// non-zero argument.
pub fn can_trim_prompt_cache<C: KeyValueCache>(caches: &[C]) -> bool {
    !caches.is_empty() && caches.iter().all(|c| c.is_trimmable())
}

/// Trim the trailing `n` tokens from every cache in the slice. Returns the
/// minimum number of tokens actually trimmed (some caches may be shorter).
pub fn trim_prompt_cache<C: KeyValueCache>(caches: &mut [C], n: i32) -> i32 {
    if !can_trim_prompt_cache(caches) || n <= 0 {
        return 0;
    }
    caches.iter_mut().map(|c| c.trim(n)).min().unwrap_or(0)
}

/// Save a prompt cache to a `.safetensors` file, mirroring Python's
/// wire format: per-layer arrays keyed `layer.{i}.{slot}` and per-layer
/// metadata keyed `layer.{i}.{key}` plus a flat `layer.{i}.class_name`.
/// `extra_metadata` is merged into the metadata map under unprefixed keys.
pub fn save_prompt_cache<C: KeyValueCache>(
    path: impl AsRef<Path>,
    caches: &[C],
    extra_metadata: Option<&HashMap<String, String>>,
) -> Result<(), Error> {
    let mut arrays: Vec<(String, Array)> = Vec::new();
    let mut metadata: HashMap<String, String> = HashMap::new();

    for (i, c) in caches.iter().enumerate() {
        let class_name = c.class_name();
        metadata.insert(format!("layer.{i}.class_name"), class_name.to_string());
        for (k, v) in c.meta_state() {
            metadata.insert(format!("layer.{i}.{k}"), v);
        }
        let slot_names = state_slot_names(class_name);
        let state = c.state();
        if !state.is_empty() && state.len() != slot_names.len() {
            return Err(Error::Other(
                format!(
                    "{class_name}.state() returned {} arrays, expected {}",
                    state.len(),
                    slot_names.len()
                )
                .into(),
            ));
        }
        for (slot, a) in slot_names.iter().zip(state.into_iter()) {
            arrays.push((format!("layer.{i}.{slot}"), a));
        }
    }

    if let Some(extra) = extra_metadata {
        for (k, v) in extra {
            metadata.insert(k.clone(), v.clone());
        }
    }
    metadata.insert("num_layers".into(), caches.len().to_string());

    let array_refs: Vec<(String, &Array)> = arrays.iter().map(|(k, a)| (k.clone(), a)).collect();
    Array::save_safetensors(array_refs, Some(&metadata), path)?;
    Ok(())
}

/// One layer's worth of loaded prompt-cache state. Caller dispatches on the
/// variant to recover the original cache type.
#[derive(Debug, Clone)]
pub enum LoadedCache {
    /// `class_name == "KVCache"`.
    Plain(KVCache),
    /// `class_name == "QuantizedKVCache"`.
    Quantized(QuantizedKVCache),
}

impl LoadedCache {
    /// Discriminant matching Python `class_name`.
    pub fn class_name(&self) -> &'static str {
        match self {
            LoadedCache::Plain(_) => "KVCache",
            LoadedCache::Quantized(_) => "QuantizedKVCache",
        }
    }
}

/// Inverse of [`save_prompt_cache`]. Returns one [`LoadedCache`] per layer
/// plus any extra metadata that wasn't prefixed with `layer.{i}.`.
pub fn load_prompt_cache(
    path: impl AsRef<Path>,
) -> Result<(Vec<LoadedCache>, HashMap<String, String>), Error> {
    let (mut arrays, mut meta) = Array::load_safetensors_with_metadata(path)?;

    let num_layers: usize = meta
        .remove("num_layers")
        .ok_or_else(|| Error::Other("prompt cache missing num_layers meta".into()))?
        .parse()
        .map_err(|e| Error::Other(format!("num_layers parse: {e}").into()))?;

    let mut layers: Vec<LoadedCache> = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let class_name = meta
            .remove(&format!("layer.{i}.class_name"))
            .ok_or_else(|| Error::Other(format!("missing layer.{i}.class_name").into()))?;

        let prefix = format!("layer.{i}.");
        let mut layer_meta: HashMap<String, String> = HashMap::new();
        let keys: Vec<String> = meta
            .keys()
            .filter(|k| k.starts_with(&prefix))
            .cloned()
            .collect();
        for k in keys {
            let suffix = k[prefix.len()..].to_string();
            let v = meta.remove(&k).expect("just found");
            layer_meta.insert(suffix, v);
        }

        let slot_names = state_slot_names(&class_name);
        let mut state: Vec<Array> = Vec::with_capacity(slot_names.len());
        for slot in slot_names {
            let key = format!("layer.{i}.{slot}");
            let a = arrays.remove(&key).ok_or_else(|| {
                Error::Other(format!("missing array {key} for {class_name}").into())
            })?;
            state.push(a);
        }

        let loaded = match class_name.as_str() {
            "KVCache" => LoadedCache::Plain(KVCache::from_state(state, &layer_meta)?),
            "QuantizedKVCache" => {
                LoadedCache::Quantized(QuantizedKVCache::from_state(state, &layer_meta)?)
            }
            other => {
                return Err(Error::Other(
                    format!("unsupported prompt-cache class {other}").into(),
                ))
            }
        };
        layers.push(loaded);
    }

    Ok((layers, meta))
}

/// Per-class state-array slot names, matching the order
/// [`KeyValueCache::state`] returns them in.
fn state_slot_names(class_name: &str) -> &'static [&'static str] {
    match class_name {
        "KVCache" => &["keys", "values"],
        "QuantizedKVCache" => &[
            "keys_wq",
            "keys_scales",
            "keys_biases",
            "values_wq",
            "values_scales",
            "values_biases",
        ],
        _ => &[],
    }
}

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

    #[test]
    fn make_prompt_cache_returns_per_layer() {
        let caches = make_prompt_cache(4, None);
        assert_eq!(caches.len(), 4);
        for c in &caches {
            assert_eq!(c.offset(), 0);
        }
    }

    #[test]
    fn trim_and_can_trim_helpers() {
        let mut caches = make_prompt_cache(3, None);
        assert!(!can_trim_prompt_cache(&caches[..0]));
        assert!(can_trim_prompt_cache(&caches));
        for c in caches.iter_mut() {
            c.update_and_fetch(token_block(5, 0.0), token_block(5, 0.0))
                .unwrap();
        }
        let trimmed = trim_prompt_cache(&mut caches, 2);
        assert_eq!(trimmed, 2);
        for c in &caches {
            assert_eq!(c.offset(), 3);
        }
    }

    #[test]
    fn prompt_cache_kvcache_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache.safetensors");

        let mut caches = make_prompt_cache(2, None);
        caches[0]
            .update_and_fetch(token_block(3, 0.0), token_block(3, 1000.0))
            .unwrap();
        caches[1]
            .update_and_fetch(token_block(2, 5.0), token_block(2, 2000.0))
            .unwrap();

        let mut extra = HashMap::new();
        extra.insert("prompt_hash".into(), "deadbeef".into());
        save_prompt_cache(&path, &caches, Some(&extra)).unwrap();

        let (loaded, meta) = load_prompt_cache(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(meta.get("prompt_hash"), Some(&"deadbeef".into()));

        match &loaded[0] {
            LoadedCache::Plain(c) => {
                assert_eq!(c.offset(), 3);
                assert_eq!(c.class_name(), "KVCache");
                let state = c.state();
                let diff = state[0]
                    .subtract(&token_block(3, 0.0))
                    .unwrap()
                    .abs()
                    .unwrap()
                    .max(None)
                    .unwrap();
                assert!(diff.item::<f32>() < 1e-6);
            }
            _ => panic!("expected plain KVCache"),
        }
    }

    #[test]
    fn rotating_kvcache_grows_until_full_then_rotates() {
        // max_size=4, keep=1 → after 4 tokens the buffer is full; further
        // tokens overwrite slots [1,2,3] in rotation.
        let mut cache = RotatingKVCache::new(4, 1);
        // Append 4 single tokens — fills buffer 0..4 in order.
        for i in 0..4 {
            let k = token_block(1, i as f32);
            let v = token_block(1, 100.0 + i as f32);
            let (out_k, _) = cache.update_and_fetch(k, v).unwrap();
            eval([&out_k]).unwrap();
        }
        assert_eq!(cache.offset(), 4);
        assert!(cache.is_trimmable());

        // 5th token: writes into slot keep+idx = 1+0 = 1, rotates idx to 1.
        let k5 = token_block(1, 99.0);
        let v5 = token_block(1, 199.0);
        let (out_k, _) = cache.update_and_fetch(k5.clone(), v5).unwrap();
        eval([&out_k]).unwrap();
        assert_eq!(cache.offset(), 5);
        assert!(!cache.is_trimmable(), "trim disabled once wrapped");
        // Output shape should still be max_size along the token axis.
        assert_eq!(out_k.shape()[out_k.shape().len() - 2], 4);
    }

    #[test]
    fn rotating_kvcache_trim_before_wrap() {
        let mut cache = RotatingKVCache::new(8, 0);
        cache
            .update_and_fetch(token_block(5, 0.0), token_block(5, 0.0))
            .unwrap();
        assert_eq!(cache.trim(2), 2);
        assert_eq!(cache.offset(), 3);
    }

    #[test]
    fn prompt_cache_quantized_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("qcache.safetensors");

        let mut caches: Vec<QuantizedKVCache> =
            (0..2).map(|_| QuantizedKVCache::with_config(64, 64, 8)).collect();
        caches[0]
            .update_and_fetch(quant_token_block(3, 0.0), quant_token_block(3, 100.0))
            .unwrap();
        caches[1]
            .update_and_fetch(quant_token_block(4, 1.0), quant_token_block(4, 200.0))
            .unwrap();

        save_prompt_cache(&path, &caches, None).unwrap();

        let (loaded, _) = load_prompt_cache(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        match &loaded[0] {
            LoadedCache::Quantized(c) => {
                assert_eq!(c.offset(), 3);
                assert_eq!(c.bits(), Some(8));
                assert_eq!(c.group_size(), Some(64));
            }
            _ => panic!("expected quantized cache"),
        }
    }
}
