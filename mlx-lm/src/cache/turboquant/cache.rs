//! `TurboQuantKVCache` — a [`KeyValueCache`] implementor using TurboQuant
//! Algorithm 2 for keys and symmetric affine group-quant for values, with
//! a configurable buffer of un-quantised recent tokens for quality.
//!
//! Phase 2a is **dequantise-on-read**: `update_and_fetch` returns dense
//! `(K, V)` arrays so the existing SDPA path (`mlx_lm::utils::
//! scaled_dot_product_attention`) consumes them unchanged. Throughput
//! parity with the existing `QuantizedKVCache` is expected; the win
//! arrives in Phase 3 via the `KeyValueCache::attention` trait method
//! and the fused Metal kernel.
//!
//! Class-name dispatch: returns `"TurboQuantKVCache"` from
//! `KeyValueCache::class_name`. `is_quantized` reports `false` so the
//! SDPA wrapper takes its standard dense path (the convention shared with
//! `QuantizedKVCache` — see commit `4cd27f8` for why).

use std::collections::HashMap;

use mlx_rs::error::Exception;
use mlx_rs::ops::{
    concatenate_axis, dequantize as mlx_dequantize, indexing::Ellipsis, indexing::IndexOp,
    quantize as mlx_quantize,
};
use mlx_rs::{Array, Dtype};

use super::quantizer::{ProdQuantized, TurboQuantProd};
use crate::cache::KeyValueCache;
use crate::error::Error;

fn parse_meta_i32(meta: &HashMap<String, String>, key: &str) -> Result<i32, Error> {
    meta.get(key)
        .ok_or_else(|| Error::Other(format!("missing meta key {key:?}").into()))?
        .parse::<i32>()
        .map_err(|e| Error::Other(format!("meta {key:?} parse: {e}").into()))
}

fn parse_meta_u64(meta: &HashMap<String, String>, key: &str) -> Result<u64, Error> {
    meta.get(key)
        .ok_or_else(|| Error::Other(format!("missing meta key {key:?}").into()))?
        .parse::<u64>()
        .map_err(|e| Error::Other(format!("meta {key:?} parse: {e}").into()))
}

fn parse_dtype(s: &str) -> Option<Dtype> {
    match s {
        "Float32" => Some(Dtype::Float32),
        "Float16" => Some(Dtype::Float16),
        "Bfloat16" => Some(Dtype::Bfloat16),
        _ => None,
    }
}

/// Default TurboQuant configuration matching the 0xSero defaults:
/// 3-bit keys, 2-bit values, group_size 32 for the value affine quant,
/// 128 recent un-quantised tokens.
pub const DEFAULT_KEY_BITS: i32 = 3;
pub const DEFAULT_VALUE_BITS: i32 = 2;
pub const DEFAULT_VALUE_GROUP_SIZE: i32 = 32;
pub const DEFAULT_BUFFER_SIZE: i32 = 128;

/// Build options.
#[derive(Debug, Clone, Copy)]
pub struct TurboQuantConfig {
    pub head_dim: i32,
    pub key_bits: i32,
    pub value_bits: i32,
    pub value_group_size: i32,
    pub buffer_size: i32,
    pub seed: u64,
}

impl TurboQuantConfig {
    /// Reasonable defaults for `head_dim` (Qwen3 / Llama: 128).
    pub fn new(head_dim: i32, seed: u64) -> Self {
        Self {
            head_dim,
            key_bits: DEFAULT_KEY_BITS,
            value_bits: DEFAULT_VALUE_BITS,
            value_group_size: DEFAULT_VALUE_GROUP_SIZE,
            buffer_size: DEFAULT_BUFFER_SIZE,
            seed,
        }
    }
}

/// KV cache using TurboQuant for keys + affine group-quant for values.
///
/// Layout invariant: at any moment the conceptual cache is
/// `concat(dequant(quant_store), buffer)` along the token axis.
/// New tokens land in `buffer`; once `buffer.len() > buffer_size`,
/// the oldest excess rows are flushed into the quantised store.
#[derive(Debug)]
pub struct TurboQuantKVCache {
    cfg: TurboQuantConfig,
    keys_quantizer: TurboQuantProd,
    /// Quantised-K storage. None until the first buffer flush.
    quant_keys: Option<ProdQuantized>,
    /// Quantised-V triples `(wq, scales, biases)`. None until first flush.
    quant_values: Option<(Array, Array, Array)>,
    /// Un-quantised recent K tokens, shape `[B, H, ≤buffer_size, D]`.
    key_buffer: Option<Array>,
    /// Un-quantised recent V tokens, shape `[B, H, ≤buffer_size, D]`.
    value_buffer: Option<Array>,
    /// Total tokens seen.
    offset: i32,
    /// Input dtype, set on first `update_and_fetch`.
    dtype: Option<Dtype>,
}

impl TurboQuantKVCache {
    /// New empty cache with the given config.
    pub fn new(cfg: TurboQuantConfig) -> Result<Self, Error> {
        let keys_quantizer = TurboQuantProd::new(cfg.head_dim, cfg.key_bits, cfg.seed)?;
        Ok(Self {
            cfg,
            keys_quantizer,
            quant_keys: None,
            quant_values: None,
            key_buffer: None,
            value_buffer: None,
            offset: 0,
            dtype: None,
        })
    }

    /// Returns the configured options.
    pub fn config(&self) -> &TurboQuantConfig {
        &self.cfg
    }

    /// Construct from a head_dim with all other config defaults — useful
    /// when wiring with helpers that need a `From<head_dim>` style call.
    pub fn with_head_dim(head_dim: i32) -> Result<Self, Error> {
        Self::new(TurboQuantConfig::new(head_dim, 0))
    }

    /// Reconstruct from previously-persisted `state` + `meta_state`.
    ///
    /// `state` ordering matches [`KeyValueCache::state`]: 11 arrays —
    /// `[Π, S, k_mse, k_signs, k_res_norms, k_norms, v_wq, v_scales,
    /// v_biases, key_buffer, value_buffer]`. Sub-stores recorded with
    /// zero-shape arrays (because nothing had been flushed at save time)
    /// reconstruct as empty `Option<…>`s.
    pub fn from_state(
        mut state: Vec<Array>,
        meta: &HashMap<String, String>,
    ) -> Result<Self, Error> {
        if state.len() != 11 {
            return Err(Error::Other(
                format!(
                    "TurboQuantKVCache::from_state expected 11 arrays, got {}",
                    state.len()
                )
                .into(),
            ));
        }
        let value_buffer_arr = state.pop().unwrap();
        let key_buffer_arr = state.pop().unwrap();
        let v_b = state.pop().unwrap();
        let v_s = state.pop().unwrap();
        let v_wq = state.pop().unwrap();
        let k_norms = state.pop().unwrap();
        let k_res_norms = state.pop().unwrap();
        let k_signs = state.pop().unwrap();
        let k_mse = state.pop().unwrap();
        let _s_matrix = state.pop().unwrap(); // S is rebuilt from seed; serialised copy is informational
        let _pi_matrix = state.pop().unwrap();

        let cfg = TurboQuantConfig {
            head_dim: parse_meta_i32(meta, "head_dim")?,
            key_bits: parse_meta_i32(meta, "key_bits")?,
            value_bits: parse_meta_i32(meta, "value_bits")?,
            value_group_size: parse_meta_i32(meta, "value_group_size")?,
            buffer_size: parse_meta_i32(meta, "buffer_size")?,
            seed: parse_meta_u64(meta, "seed")?,
        };
        let offset = parse_meta_i32(meta, "offset")?;

        // Rebuild the keys_quantizer fresh from seed — produces the same
        // Π/S as save-time.
        let keys_quantizer = TurboQuantProd::new(cfg.head_dim, cfg.key_bits, cfg.seed)?;

        let quant_keys = if k_mse.shape().iter().product::<i32>() > 0 {
            Some(ProdQuantized {
                mse_indices: k_mse,
                qjl_signs: k_signs,
                residual_norms: k_res_norms,
                norms: k_norms,
            })
        } else {
            None
        };
        let quant_values = if v_wq.shape().iter().product::<i32>() > 0 {
            Some((v_wq, v_s, v_b))
        } else {
            None
        };
        let key_buffer = (key_buffer_arr.shape().iter().product::<i32>() > 0)
            .then_some(key_buffer_arr);
        let value_buffer = (value_buffer_arr.shape().iter().product::<i32>() > 0)
            .then_some(value_buffer_arr);

        let dtype = meta.get("dtype").and_then(|s| parse_dtype(s));

        Ok(Self {
            cfg,
            keys_quantizer,
            quant_keys,
            quant_values,
            key_buffer,
            value_buffer,
            offset,
            dtype,
        })
    }

    fn token_axis(shape: &[i32]) -> i32 {
        (shape.len() as i32) - 2
    }

    fn buffer_len(&self) -> i32 {
        match self.key_buffer.as_ref() {
            Some(b) => b.shape()[b.ndim() - 2],
            None => 0,
        }
    }

    /// Flush the oldest `n_flush` rows of the buffer into the quantised
    /// store. Idempotent if `n_flush <= 0`.
    fn flush(&mut self, n_flush: i32) -> Result<(), Exception> {
        if n_flush <= 0 {
            return Ok(());
        }
        let kb = self.key_buffer.as_ref().expect("flush with empty buffer");
        let vb = self.value_buffer.as_ref().expect("flush with empty buffer");

        // Split: oldest n_flush rows go to quantised store; rest stays in buffer.
        let keys_flush = kb.index((Ellipsis, 0..n_flush, ..));
        let values_flush = vb.index((Ellipsis, 0..n_flush, ..));
        let buf_len = kb.shape()[kb.ndim() - 2];
        let keys_keep = kb.index((Ellipsis, n_flush..buf_len, ..));
        let values_keep = vb.index((Ellipsis, n_flush..buf_len, ..));

        // Quantise the flushed K.
        let new_keys_q = self.keys_quantizer.quantize(&keys_flush)?;
        // Quantise V via affine group-quant (mlx-rs `quantize`).
        let (new_v_wq, new_v_s, new_v_b) = mlx_quantize(
            &values_flush,
            self.cfg.value_group_size,
            self.cfg.value_bits,
        )?;

        // Concatenate into the existing quantised stores.
        self.quant_keys = Some(match self.quant_keys.take() {
            None => new_keys_q,
            Some(existing) => ProdQuantized {
                mse_indices: concatenate_axis(
                    &[existing.mse_indices, new_keys_q.mse_indices],
                    -2,
                )?,
                qjl_signs: concatenate_axis(
                    &[existing.qjl_signs, new_keys_q.qjl_signs],
                    -2,
                )?,
                residual_norms: concatenate_axis(
                    &[existing.residual_norms, new_keys_q.residual_norms],
                    -1,
                )?,
                norms: concatenate_axis(&[existing.norms, new_keys_q.norms], -1)?,
            },
        });

        self.quant_values = Some(match self.quant_values.take() {
            None => (new_v_wq, new_v_s, new_v_b),
            Some((wq, s, b)) => (
                concatenate_axis(&[wq, new_v_wq], -2)?,
                concatenate_axis(&[s, new_v_s], -2)?,
                concatenate_axis(&[b, new_v_b], -2)?,
            ),
        });

        self.key_buffer = Some(keys_keep);
        self.value_buffer = Some(values_keep);
        Ok(())
    }

    /// Dequantise the populated quantised stores into dense `(K, V)` at
    /// the cache's input dtype, and concat with the un-quantised buffer.
    /// Returns `(K_full, V_full)` shape `[B, H, offset, D]`.
    fn assemble_dense(&self) -> Result<(Array, Array), Exception> {
        let in_dtype = self.dtype.expect("assemble called before any update");

        // Dequantise K from the quantised store, if present.
        let k_quant = match self.quant_keys.as_ref() {
            None => None,
            Some(q) => Some(self.keys_quantizer.dequantize(q)?.as_dtype(in_dtype)?),
        };
        // Dequantise V.
        let v_quant = match self.quant_values.as_ref() {
            None => None,
            Some((wq, s, b)) => Some(mlx_dequantize(
                wq,
                s,
                b,
                self.cfg.value_group_size,
                self.cfg.value_bits,
            )?),
        };

        // Concat with the recent buffer along the token axis (-2).
        let k_full = match (k_quant, self.key_buffer.as_ref()) {
            (None, None) => unreachable!("assemble called with no data"),
            (Some(q), None) => q,
            (None, Some(b)) => b.clone(),
            (Some(q), Some(b)) => concatenate_axis(&[q, b.clone()], -2)?,
        };
        let v_full = match (v_quant, self.value_buffer.as_ref()) {
            (None, None) => unreachable!("assemble called with no data"),
            (Some(q), None) => q,
            (None, Some(b)) => b.clone(),
            (Some(q), Some(b)) => concatenate_axis(&[q, b.clone()], -2)?,
        };

        Ok((k_full, v_full))
    }
}

impl KeyValueCache for TurboQuantKVCache {
    /// See module-level note — we report `false` because `update_and_fetch`
    /// always returns dense `(K, V)`. The SDPA wrapper takes the dense
    /// branch.
    fn is_quantized(&self) -> bool {
        false
    }

    fn group_size(&self) -> Option<i32> {
        Some(self.cfg.value_group_size)
    }

    fn bits(&self) -> Option<i32> {
        Some(self.cfg.key_bits)
    }

    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    /// Trim is meaningful only while the cache hasn't been forced into the
    /// quantised store — once a token has been flushed and dequantised, the
    /// position information is preserved but rolling back would require
    /// per-row inverse work we don't support today. We allow trimming
    /// within the recent buffer; trims past that point fail loudly.
    fn is_trimmable(&self) -> bool {
        self.quant_keys.is_none()
    }

    fn trim(&mut self, n: i32) -> i32 {
        if !self.is_trimmable() {
            return 0;
        }
        let buf_len = self.buffer_len();
        let trimmed = n.min(buf_len).max(0);
        if trimmed == 0 {
            return 0;
        }
        let kb = self.key_buffer.as_ref().unwrap();
        let vb = self.value_buffer.as_ref().unwrap();
        let new_len = buf_len - trimmed;
        self.key_buffer = Some(kb.index((Ellipsis, 0..new_len, ..)));
        self.value_buffer = Some(vb.index((Ellipsis, 0..new_len, ..)));
        self.offset -= trimmed;
        trimmed
    }

    fn class_name(&self) -> &'static str {
        "TurboQuantKVCache"
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        if self.dtype.is_none() {
            self.dtype = Some(keys.dtype());
        }

        // Append into the buffer (or seed if empty).
        let s = keys.shape()[Self::token_axis(keys.shape()) as usize];
        self.offset += s;
        match (self.key_buffer.take(), self.value_buffer.take()) {
            (None, None) => {
                self.key_buffer = Some(keys);
                self.value_buffer = Some(values);
            }
            (Some(kb), Some(vb)) => {
                self.key_buffer = Some(concatenate_axis(&[kb, keys], -2)?);
                self.value_buffer = Some(concatenate_axis(&[vb, values], -2)?);
            }
            _ => unreachable!("buffer halves desynced"),
        }

        // Flush oldest excess into the quantised store.
        let buf_len = self.buffer_len();
        if buf_len > self.cfg.buffer_size {
            let n_flush = buf_len - self.cfg.buffer_size;
            self.flush(n_flush)?;
        }

        self.assemble_dense()
    }

    fn state(&self) -> Vec<Array> {
        let mut out = Vec::with_capacity(11);
        out.push(self.keys_quantizer.mse_quantizer().rotation().clone());
        out.push(self.keys_quantizer.qjl_matrix().clone());

        let zero_2d = Array::zeros::<f32>(&[0, 0]).unwrap();
        let zero_1d = Array::zeros::<f32>(&[0]).unwrap();

        // Quantised K state — 4 arrays. Substitute empty placeholders if
        // nothing's been flushed yet (load_prompt_cache will see the
        // empty shapes and reconstruct an empty store).
        match self.quant_keys.as_ref() {
            Some(q) => {
                out.push(q.mse_indices.clone());
                out.push(q.qjl_signs.clone());
                out.push(q.residual_norms.clone());
                out.push(q.norms.clone());
            }
            None => {
                out.push(zero_2d.clone());
                out.push(zero_2d.clone());
                out.push(zero_1d.clone());
                out.push(zero_1d.clone());
            }
        }

        // Quantised V state — 3 arrays.
        match self.quant_values.as_ref() {
            Some((wq, s, b)) => {
                out.push(wq.clone());
                out.push(s.clone());
                out.push(b.clone());
            }
            None => {
                out.push(zero_2d.clone());
                out.push(zero_2d.clone());
                out.push(zero_2d.clone());
            }
        }

        // Recent buffer — 2 arrays.
        out.push(self.key_buffer.clone().unwrap_or_else(|| zero_2d.clone()));
        out.push(self.value_buffer.clone().unwrap_or_else(|| zero_2d.clone()));

        out
    }

    fn meta_state(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("offset".into(), self.offset.to_string());
        m.insert("head_dim".into(), self.cfg.head_dim.to_string());
        m.insert("key_bits".into(), self.cfg.key_bits.to_string());
        m.insert("value_bits".into(), self.cfg.value_bits.to_string());
        m.insert(
            "value_group_size".into(),
            self.cfg.value_group_size.to_string(),
        );
        m.insert("buffer_size".into(), self.cfg.buffer_size.to_string());
        m.insert("seed".into(), self.cfg.seed.to_string());
        if let Some(dt) = self.dtype {
            m.insert("dtype".into(), format!("{dt:?}"));
        }
        m
    }
}

/// `Default` is provided only to satisfy the `C: KeyValueCache + Default`
/// bound used by the qwen3 / llama cache-init fallback path
/// (`Qwen3Model::forward`: `if cache.is_empty() { fill with C::default() }`).
///
/// **It returns a placeholder cache with `head_dim = 128, seed = 0` — the
/// caller is expected to pre-populate `Vec<Option<TurboQuantKVCache>>` with
/// the correct config via `make_turboquant_kv_cache(...)` before passing
/// into `Generate::new`.** If a caller passes an empty `Vec::new()` and lets
/// the fallback engage, results will be wrong (every layer shares the same
/// Π, and the head_dim is hard-coded).
impl Default for TurboQuantKVCache {
    fn default() -> Self {
        Self::with_head_dim(128).expect("TurboQuantKVCache::default placeholder")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::KeyValueCache;
    use mlx_rs::transforms::eval;

    fn make_cfg(head_dim: i32, buffer: i32) -> TurboQuantConfig {
        TurboQuantConfig {
            head_dim,
            key_bits: 3,
            value_bits: 2,
            value_group_size: 32,
            buffer_size: buffer,
            seed: 7,
        }
    }

    /// `[B=1, H=2, S, D]` Gaussian sample.
    fn token_block(s: i32, d: i32, seed: u64) -> Array {
        let prng = mlx_rs::random::key(seed).unwrap();
        mlx_rs::random::normal::<f32>(&[1, 2, s, d], None, None, &prng).unwrap()
    }

    #[test]
    fn fresh_cache_has_zero_offset() {
        let c = TurboQuantKVCache::new(make_cfg(64, 128)).unwrap();
        assert_eq!(c.offset(), 0);
        assert_eq!(c.class_name(), "TurboQuantKVCache");
        assert!(!c.is_quantized());
        assert_eq!(c.bits(), Some(3));
        assert_eq!(c.group_size(), Some(32));
    }

    /// First update returns dense (K, V) with correct shape and offset.
    #[test]
    fn first_update_returns_dense_kv() {
        let d = 64;
        let mut c = TurboQuantKVCache::new(make_cfg(d, 128)).unwrap();
        let k = token_block(3, d, 1);
        let v = token_block(3, d, 2);
        let (out_k, out_v) = c.update_and_fetch(k, v).unwrap();
        eval([&out_k, &out_v]).unwrap();
        assert_eq!(out_k.shape(), &[1, 2, 3, d]);
        assert_eq!(out_v.shape(), &[1, 2, 3, d]);
        assert_eq!(c.offset(), 3);
    }

    /// Tokens kept in the recent buffer are returned *exactly* (no quant
    /// loss) — when total tokens ≤ buffer_size, the cache acts like a
    /// plain concat cache.
    #[test]
    fn buffer_only_tokens_are_lossless() {
        let d = 64;
        let mut c = TurboQuantKVCache::new(make_cfg(d, 16)).unwrap();
        let k = token_block(4, d, 3);
        let v = token_block(4, d, 4);
        let (out_k, out_v) = c.update_and_fetch(k.clone(), v.clone()).unwrap();
        eval([&out_k, &out_v]).unwrap();
        let dk = out_k
            .subtract(&k)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        let dv = out_v
            .subtract(&v)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(dk < 1e-6 && dv < 1e-6, "buffer not lossless: dk={dk} dv={dv}");
    }

    /// Flush boundary: prefilling more than `buffer_size` tokens forces
    /// the oldest excess into the quantised store. The returned `K_full`
    /// has the full token count and is finite; quantised tokens differ
    /// from input but reconstruction is bounded.
    #[test]
    fn flush_engages_at_buffer_overflow() {
        let d = 64;
        let mut c = TurboQuantKVCache::new(make_cfg(d, 4)).unwrap();
        // 4 tokens fit; 5th forces flush of the 1st.
        let k = token_block(5, d, 11);
        let v = token_block(5, d, 12);
        let (out_k, out_v) = c.update_and_fetch(k, v).unwrap();
        eval([&out_k, &out_v]).unwrap();
        assert_eq!(out_k.shape(), &[1, 2, 5, d]);
        assert!(c.quant_keys.is_some(), "flush should have engaged");
        assert!(c.quant_values.is_some());
        assert_eq!(c.offset(), 5);
        // Output should be finite.
        let max_k = out_k.abs().unwrap().max(None).unwrap().item::<f32>();
        assert!(max_k.is_finite(), "K_full not finite: {max_k}");
        let max_v = out_v.abs().unwrap().max(None).unwrap().item::<f32>();
        assert!(max_v.is_finite());
    }

    /// Multiple updates accumulate offset and pump tokens through the
    /// buffer→flush→quantised pipeline. Final shape matches the
    /// accumulated count.
    #[test]
    fn multiple_updates_accumulate_offset() {
        let d = 64;
        let mut c = TurboQuantKVCache::new(make_cfg(d, 8)).unwrap();
        for i in 0..5 {
            let k = token_block(3, d, 20 + i);
            let v = token_block(3, d, 30 + i);
            c.update_and_fetch(k, v).unwrap();
        }
        assert_eq!(c.offset(), 15);
        // 15 tokens, buffer_size=8 → 7 are quantised, 8 in buffer.
        assert!(c.quant_keys.is_some());
    }

    /// Trim only within the recent buffer; rejected once anything has been
    /// flushed.
    #[test]
    fn trim_only_within_buffer() {
        let d = 64;
        let mut c = TurboQuantKVCache::new(make_cfg(d, 16)).unwrap();
        c.update_and_fetch(token_block(5, d, 40), token_block(5, d, 41))
            .unwrap();
        assert!(c.is_trimmable());
        assert_eq!(c.trim(2), 2);
        assert_eq!(c.offset(), 3);
        assert_eq!(c.buffer_len(), 3);

        // Now overflow; trim should refuse.
        c.update_and_fetch(token_block(20, d, 42), token_block(20, d, 43))
            .unwrap();
        assert!(!c.is_trimmable());
        assert_eq!(c.trim(2), 0);
    }

    /// State + meta_state round-trip carries the structural shape (no
    /// content check yet — load_prompt_cache dispatch arrives in C9).
    #[test]
    fn state_returns_expected_arity() {
        let d = 64;
        let mut c = TurboQuantKVCache::new(make_cfg(d, 4)).unwrap();
        c.update_and_fetch(token_block(6, d, 50), token_block(6, d, 51))
            .unwrap();
        let state = c.state();
        // 2 (Π, S) + 4 (key quant) + 3 (value quant) + 2 (buffer) = 11
        assert_eq!(state.len(), 11);
        let meta = c.meta_state();
        assert_eq!(meta.get("offset").map(String::as_str), Some("6"));
        assert_eq!(meta.get("key_bits").map(String::as_str), Some("3"));
        assert_eq!(meta.get("value_bits").map(String::as_str), Some("2"));
        assert_eq!(meta.get("buffer_size").map(String::as_str), Some("4"));
    }
}
