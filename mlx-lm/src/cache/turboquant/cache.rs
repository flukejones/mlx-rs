//! `TurboQuantKVCache` — TurboQuant Algorithm 2 for keys + affine group-
//! quant for values, plus a buffer of un-quantised recent tokens. Score
//! path uses the fused tq_attention_score kernel and skips K dequant;
//! values dequant on read.

use std::collections::HashMap;

use mlx_rs::error::Exception;
use mlx_rs::ops::{
    concatenate_axis, dequantize as mlx_dequantize, indexing::Ellipsis, indexing::IndexOp,
    quantize as mlx_quantize, r#where, repeat_axis, softmax_axis,
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

    /// Skips K dequant: scores route through the fused
    /// `tq_score_kernel` for the quant-store + a dense matmul for the
    /// recent buffer. V still dequants for the softmax-weighted attend.
    /// GQA replicates the cached state via `repeat_axis`.
    fn attention(
        &mut self,
        queries: &Array,
        keys: Array,
        values: Array,
        scale: f32,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        // 1: append new (K, V) into the buffer (same shape as
        // update_and_fetch's append phase). The buffer holds dense fp16
        // tokens until it overflows, then the oldest excess flushes into
        // the quant store.
        if self.dtype.is_none() {
            self.dtype = Some(keys.dtype());
        }
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
        let buf_len = self.buffer_len();
        if buf_len > self.cfg.buffer_size {
            let n_flush = buf_len - self.cfg.buffer_size;
            self.flush(n_flush)?;
        }

        let in_dtype = self.dtype.expect("dtype set above");

        // 2: GQA expansion. queries arrive as [B, H_q, n_q, D]; cached
        // K/V live at H_kv heads. If H_q > H_kv, replicate K/V across
        // the query-head dimension.
        let q_shape = queries.shape();
        if q_shape.len() != 4 {
            return Err(Exception::custom(format!(
                "turboquant attention: queries must be 4-D [B,H,S,D], got {q_shape:?}"
            )));
        }
        let n_heads_q = q_shape[1];
        let n_heads_kv = match (self.quant_keys.as_ref(), self.key_buffer.as_ref()) {
            (Some(q), _) => q.mse_indices.shape()[1],
            (None, Some(kb)) => kb.shape()[1],
            (None, None) => return Err(Exception::custom("turboquant attention: no K state")),
        };
        if n_heads_q % n_heads_kv != 0 {
            return Err(Exception::custom(format!(
                "turboquant attention: n_heads_q={n_heads_q} not a multiple of n_heads_kv={n_heads_kv}"
            )));
        }
        let n_rep = n_heads_q / n_heads_kv;

        // 3: scores over the quant store via the kernel — no GQA repeat,
        // no dense materialisation. The kernel reads packed bytes and
        // computes `h_kv = h_q / n_rep` internally.
        let q_f32 = queries.as_dtype(Dtype::Float32)?;
        let mut score_parts: Vec<Array> = Vec::with_capacity(2);
        if let Some(quant_keys) = self.quant_keys.as_ref() {
            let s_quant = self
                .keys_quantizer
                .attention_score(&q_f32, quant_keys, n_heads_kv)?;
            score_parts.push(s_quant);
        }

        // 4: scores over the buffer (dense matmul). The buffer is bounded
        // at `cfg.buffer_size` (default 128) so the `repeat_axis` here is
        // cheap regardless of total context length.
        if let Some(kb) = self.key_buffer.as_ref() {
            let kb_expanded = if n_rep > 1 {
                repeat_axis::<f32>(kb.clone(), n_rep, 1)?
            } else {
                kb.clone()
            };
            let kb_f32 = kb_expanded.as_dtype(Dtype::Float32)?;
            let kb_t = kb_f32.transpose_axes(&[0, 1, 3, 2])?;
            let s_buf = q_f32.matmul(&kb_t)?;
            score_parts.push(s_buf);
        }
        let mut scores = match score_parts.len() {
            0 => return Err(Exception::custom("turboquant attention: no scores")),
            1 => score_parts.pop().unwrap(),
            _ => concatenate_axis(&score_parts, -1)?,
        };

        // 5: scale, mask, softmax.
        scores = scores.multiply(Array::from_f32(scale))?;
        if let Some(m) = mask {
            // `mlx_lm::utils::create_causal_mask` returns a *boolean*
            // mask (true = attention allowed). Convert to the additive
            // form softmax expects: `0` on allowed positions, `-inf` on
            // disallowed. Pre-existing additive masks (fp32 / bf16 with
            // -inf values) round-trip cleanly through this `where`.
            let zero = Array::from_f32(0.0);
            let ninf = Array::from_f32(f32::NEG_INFINITY);
            let additive = if m.dtype() == Dtype::Bool {
                r#where(m, zero, ninf)?
            } else {
                m.as_dtype(Dtype::Float32)?
            };
            scores = scores.add(&additive)?;
        }
        let probs = softmax_axis(&scores, -1, true)?;
        let probs_t = probs.as_dtype(in_dtype)?;

        // 6: assemble V_full (dequant quant-store V + dense buffer V),
        // replicate across query-head groups, multiply by probs.
        let v_quant = match self.quant_values.as_ref() {
            None => None,
            Some((wq, sc, b)) => Some(mlx_dequantize(
                wq,
                sc,
                b,
                self.cfg.value_group_size,
                self.cfg.value_bits,
            )?),
        };
        let v_full = match (v_quant, self.value_buffer.as_ref()) {
            (None, None) => unreachable!("attention with no V"),
            (Some(q), None) => q,
            (None, Some(b)) => b.clone(),
            (Some(q), Some(b)) => concatenate_axis(&[q, b.clone()], -2)?,
        };
        let v_full = if n_rep > 1 {
            repeat_axis::<f32>(v_full, n_rep, 1)?
        } else {
            v_full
        };
        probs_t.matmul(&v_full)
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
    use mlx_rs::random::{key, normal};
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
        let prng = key(seed).unwrap();
        normal::<f32>(&[1, 2, s, d], None, None, &prng).unwrap()
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

    /// GQA case (n_heads_q = 4, n_heads_kv = 2). Override `attention` must
    /// reproduce fp16 SDPA after replicating K/V across query-head groups.
    /// Catches GQA broadcast bugs in the override.
    #[test]
    fn override_buffer_only_matches_fp16_sdpa_gqa() {
        let d = 64;
        let n_kv = 2;
        let n_q = 4;
        let s_k = 8;
        let s_q = 8;
        // Build [1, n_kv, s_k, d] K/V and [1, n_q, s_q, d] Q manually so
        // we can drive an explicit GQA case (token_block uses H=2).
        let prng = key(81).unwrap();
        let k = normal::<f32>(&[1, n_kv, s_k, d], None, None, &prng).unwrap();
        let prng = key(82).unwrap();
        let v = normal::<f32>(&[1, n_kv, s_k, d], None, None, &prng).unwrap();
        let prng = key(83).unwrap();
        let q = normal::<f32>(&[1, n_q, s_q, d], None, None, &prng).unwrap();
        let scale = (d as f32).sqrt().recip();

        let mut c = TurboQuantKVCache::new(make_cfg(d, 128)).unwrap();
        let out_tq = c.attention(&q, k.clone(), v.clone(), scale, None).unwrap();
        mlx_rs::transforms::eval([&out_tq]).unwrap();

        // Reference: fast SDPA handles GQA internally.
        let out_ref = mlx_rs::fast::scaled_dot_product_attention(
            q,
            k,
            v,
            scale,
            None::<mlx_rs::fast::ScaledDotProductAttentionMask>,
            None,
        )
        .unwrap();
        mlx_rs::transforms::eval([&out_ref]).unwrap();

        let err = out_tq
            .subtract(&out_ref)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(
            err < 1e-3,
            "override GQA buffer-only vs fp16 SDPA diverged: max abs = {err}"
        );
    }

    /// Override with a *boolean* causal mask (the format `mlx_lm::utils::
    /// create_causal_mask` returns and the qwen3 prefill path passes
    /// down). The override must internally convert bool → additive
    /// (0 / -inf) before adding to scores.
    #[test]
    fn override_buffer_only_with_bool_mask_matches_fp16_sdpa() {
        let d = 64;
        let n_heads = 4;
        let s = 8;
        let prng = key(111).unwrap();
        let k = normal::<f32>(&[1, n_heads, s, d], None, None, &prng).unwrap();
        let prng = key(112).unwrap();
        let v = normal::<f32>(&[1, n_heads, s, d], None, None, &prng).unwrap();
        let prng = key(113).unwrap();
        let q = normal::<f32>(&[1, n_heads, s, d], None, None, &prng).unwrap();
        let scale = (d as f32).sqrt().recip();

        // Boolean causal mask via the same op `create_causal_mask` uses:
        // linds >= rinds (i.e. lower-triangular).
        let mut bool_data = vec![false; (s * s) as usize];
        for i in 0..s {
            for j in 0..=i {
                bool_data[(i * s + j) as usize] = true;
            }
        }
        let bool_mask =
            Array::from_slice(&bool_data.iter().map(|&b| b as u8).collect::<Vec<_>>(), &[s, s])
                .as_dtype(Dtype::Bool)
                .unwrap();

        let mut c = TurboQuantKVCache::new(make_cfg(d, 128)).unwrap();
        let out_tq = c.attention(&q, k.clone(), v.clone(), scale, Some(&bool_mask)).unwrap();
        eval([&out_tq]).unwrap();

        // fp16 SDPA accepts the bool mask directly via Array variant.
        let out_ref = mlx_rs::fast::scaled_dot_product_attention(
            q,
            k,
            v,
            scale,
            Some(mlx_rs::fast::ScaledDotProductAttentionMask::Array(&bool_mask)),
            None,
        )
        .unwrap();
        eval([&out_ref]).unwrap();

        let err = out_tq
            .subtract(&out_ref)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(
            err < 1e-3,
            "override with bool mask vs fp16 SDPA diverged: max abs = {err}"
        );
    }

    /// GQA + causal mask + bf16 dtype — closest possible match to the
    /// qwen3 prefill path that the parity test exercises.
    #[test]
    fn override_buffer_only_gqa_causal_bf16_matches_fp16_sdpa() {
        let d = 128;
        let n_kv = 8;
        let n_q = 16;
        let s = 32;
        let prng = key(101).unwrap();
        let k_f32 = normal::<f32>(&[1, n_kv, s, d], None, None, &prng).unwrap();
        let prng = key(102).unwrap();
        let v_f32 = normal::<f32>(&[1, n_kv, s, d], None, None, &prng).unwrap();
        let prng = key(103).unwrap();
        let q_f32 = normal::<f32>(&[1, n_q, s, d], None, None, &prng).unwrap();
        let k = k_f32.as_dtype(Dtype::Bfloat16).unwrap();
        let v = v_f32.as_dtype(Dtype::Bfloat16).unwrap();
        let q = q_f32.as_dtype(Dtype::Bfloat16).unwrap();
        let scale = (d as f32).sqrt().recip();

        let mut mask_data = vec![0.0f32; (s * s) as usize];
        for i in 0..s {
            for j in (i + 1)..s {
                mask_data[(i * s + j) as usize] = f32::NEG_INFINITY;
            }
        }
        let mask = Array::from_slice(&mask_data, &[s, s])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();

        let mut c = TurboQuantKVCache::new(make_cfg(d, 128)).unwrap();
        let out_tq = c.attention(&q, k.clone(), v.clone(), scale, Some(&mask)).unwrap();
        mlx_rs::transforms::eval([&out_tq]).unwrap();

        let out_ref = mlx_rs::fast::scaled_dot_product_attention(
            q,
            k,
            v,
            scale,
            Some(mlx_rs::fast::ScaledDotProductAttentionMask::Array(&mask)),
            None,
        )
        .unwrap();
        mlx_rs::transforms::eval([&out_ref]).unwrap();

        let err = out_tq
            .subtract(&out_ref)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        // bf16 tolerance: looser than fp32 paths.
        assert!(
            err < 5e-2,
            "override GQA+causal+bf16 vs fp16 SDPA diverged: max abs = {err}"
        );
    }

    /// Override with a causal mask (the real prefill path). KL on a
    /// model is dominated by mask handling — this catches the mask
    /// shape / additive convention.
    #[test]
    fn override_buffer_only_with_causal_mask_matches_fp16_sdpa() {
        let d = 64;
        let n_heads = 4;
        let s = 8;
        let prng = key(91).unwrap();
        let k = normal::<f32>(&[1, n_heads, s, d], None, None, &prng).unwrap();
        let prng = key(92).unwrap();
        let v = normal::<f32>(&[1, n_heads, s, d], None, None, &prng).unwrap();
        let prng = key(93).unwrap();
        let q = normal::<f32>(&[1, n_heads, s, d], None, None, &prng).unwrap();
        let scale = (d as f32).sqrt().recip();

        // Causal mask: [s, s], -inf above diagonal, 0 on/below.
        let mut mask_data = vec![0.0f32; (s * s) as usize];
        for i in 0..s {
            for j in (i + 1)..s {
                mask_data[(i * s + j) as usize] = f32::NEG_INFINITY;
            }
        }
        let mask = Array::from_slice(&mask_data, &[s, s]);

        let mut c = TurboQuantKVCache::new(make_cfg(d, 128)).unwrap();
        let out_tq = c.attention(&q, k.clone(), v.clone(), scale, Some(&mask)).unwrap();
        mlx_rs::transforms::eval([&out_tq]).unwrap();

        let out_ref = mlx_rs::fast::scaled_dot_product_attention(
            q,
            k,
            v,
            scale,
            Some(mlx_rs::fast::ScaledDotProductAttentionMask::Array(&mask)),
            None,
        )
        .unwrap();
        mlx_rs::transforms::eval([&out_ref]).unwrap();

        let err = out_tq
            .subtract(&out_ref)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(
            err < 1e-3,
            "override with causal mask vs fp16 SDPA diverged: max abs = {err}"
        );
    }

    /// The override `attention` matches `mlx_rs::fast::scaled_dot_product_attention`
    /// on a buffer-only case (no flush, no quantisation engaged — the score
    /// path collapses to a dense matmul). Catches softmax / mask / V matmul
    /// bugs in isolation from the quantiser math.
    #[test]
    fn override_buffer_only_matches_fp16_sdpa() {
        let d = 64;
        let mut c = TurboQuantKVCache::new(make_cfg(d, 128)).unwrap();
        let k = token_block(8, d, 70); // [1, 2, 8, 64]
        let v = token_block(8, d, 71);
        let q = token_block(8, d, 72);
        let scale = (d as f32).sqrt().recip();

        let out_tq = c.attention(&q, k.clone(), v.clone(), scale, None).unwrap();
        mlx_rs::transforms::eval([&out_tq]).unwrap();

        // Reference: dense SDPA on the same q/k/v.
        let out_ref = mlx_rs::fast::scaled_dot_product_attention(
            q,
            k,
            v,
            scale,
            None::<mlx_rs::fast::ScaledDotProductAttentionMask>,
            None,
        )
        .unwrap();
        mlx_rs::transforms::eval([&out_ref]).unwrap();

        let err = out_tq
            .subtract(&out_ref)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(
            err < 1e-3,
            "override buffer-only vs fp16 SDPA diverged: max abs = {err}"
        );
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
