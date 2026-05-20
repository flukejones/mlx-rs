//! [`KVCache`]: pre-allocated, step-grown plain KV cache. Default cache
//! for all decoder-only models.

use std::collections::HashMap;

use mlx_rs::{
    error::Exception,
    fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask},
    ops::{
        indexing::{Ellipsis, IndexOp, TryIndexMutOp},
        zeros_dtype,
    },
    Array,
};

use crate::error::Error;
use crate::steel_attention::{steel_attention_dispatch, SteelAttentionInputs};

use super::io::parse_meta;
use super::kernels::{cached_steel_attention_kernel, STEEL_SUPPORTED_HEAD_DIMS};
use super::trait_def::KeyValueCache;

/// Default step in tokens for [`KVCache`]'s pre-allocated buffer growth.
pub const DEFAULT_KV_CACHE_STEP: i32 = 256;

/// Pre-allocated KV cache with step-based growth.
///
/// Mirrors Python `mlx_lm.models.cache.KVCache`: the underlying `keys` /
/// `values` buffers are `[B, H, capacity, D]` `Array`s pre-allocated in
/// chunks of `step` rows. On each `update_and_fetch` call we grow the
/// buffers if `offset + S > capacity`, slice-write the new tokens into
/// the `[offset:offset+S]` axis, and return a slice view of the populated
/// `[:offset+S]` rows.
///
/// The amortised cost of `step`-sized buffer growth is much smaller
/// than per-token concat, especially at S≥256.
#[derive(Debug, Clone)]
pub struct KVCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
    /// Buffer-growth step in tokens.
    step: i32,
    /// When set, the prefill path (`n_q > 1`) routes through the
    /// steel-attention tiled kernel instead of `fast::SDPA`. Opt-in
    /// via [`Self::with_steel_prefill`]; only D ∈ {128, 256} and
    /// non-mask paths use it (falls back to `fast::SDPA` otherwise).
    use_steel_prefill: bool,
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
            use_steel_prefill: false,
        }
    }

    /// Opt in to the steel-attention tiled prefill kernel. Active only
    /// when `n_q > 1`, `head_dim ∈ {128, 256}`, and the caller passes
    /// no explicit mask. Falls back to `fast::SDPA` otherwise.
    ///
    /// Builder; consumes and returns `self`.
    pub fn with_steel_prefill(mut self) -> Self {
        self.use_steel_prefill = true;
        self
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
            use_steel_prefill: false,
        })
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
        let target_cap = super::trait_def::ceil_step(required, step);

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

    /// Prefill-aware attention: when the cache was built with
    /// [`Self::with_steel_prefill`], route the `n_q > 1` path through
    /// the steel-attention tiled kernel. Falls through to the default
    /// `fast::SDPA` dispatch for decode, masked, or unsupported-shape
    /// calls.
    fn attention(
        &mut self,
        queries: &Array,
        keys: Array,
        values: Array,
        scale: f32,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        // Pre-update offset → steel kernel `ql_off` (causal diagonal shift).
        let ql_off = self.offset;
        let (k_full, v_full) = self.update_and_fetch(keys, values)?;

        let q_shape = queries.shape();
        let n_q = q_shape[q_shape.len() - 2];
        let head_dim = q_shape[q_shape.len() - 1];
        let h_q = q_shape[1];
        let h_kv = k_full.shape()[1];

        let steel_ok = self.use_steel_prefill
            && n_q > 1
            && STEEL_SUPPORTED_HEAD_DIMS.contains(&head_dim)
            && h_q % h_kv == 0;

        // Mirrors qwen3_5::Attention: drop caller mask, force causal+ql_off.
        if steel_ok {
            return steel_attention_dispatch(
                cached_steel_attention_kernel(),
                SteelAttentionInputs {
                    q: queries,
                    k: &k_full,
                    v: &v_full,
                    mask: None,
                    causal: true,
                    ql_off,
                    scale,
                    head_dim,
                    h_q,
                    h_kv,
                },
            );
        }

        super::trait_def::assert_mask_matches_keys(mask, &k_full);
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
