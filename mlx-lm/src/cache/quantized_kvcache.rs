//! [`QuantizedKVCache`]: affine-quantised KV cache with optional Π
//! rotation, packed-matmul path, fused-qsdpa decode kernel, and steel
//! quantised tile prefill kernel.

use std::collections::HashMap;

use mlx_rs::{
    error::Exception,
    fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask},
    ops::{
        dequantize,
        indexing::{Ellipsis, IndexOp, TryIndexMutOp},
        quantize, zeros_dtype,
    },
    Array, Dtype,
};

use crate::error::Error;
use crate::steel_attention::{steel_quant_attention_dispatch, SteelQuantAttentionInputs};
use crate::utils::{quantized_scaled_dot_product_attention, QuantizedKeys, QuantizedValues};

use super::fused_quantized_sdpa::{fused_qsdpa_decode, FusedQsdpaInputs};
use super::io::{parse_meta, parse_meta_or};
use super::kernels::{
    cached_fused_qsdpa_kernel, cached_steel_quant_attention_kernel, STEEL_SUPPORTED_HEAD_DIMS,
};
use super::kvcache::DEFAULT_KV_CACHE_STEP;
use super::rotation;
use super::trait_def::KeyValueCache;

/// Perf crossover for the fused qsdpa kernel. Past this, the per-
/// simdgroup serial K-token processing loses to mlx's tiled
/// `quantized_matmul` ops-composed path. Empirical on Qwen3-1.7B-q4 +
/// KV q8 (Apple M4 Max): fused wins to T=4096; ops-composed wins from
/// T=8192. The actual crossover is somewhere between — 4096 is the
/// safe upper bound. Revisit if the kernel adds intra-simdgroup
/// parallelism over K.
const FUSED_KERNEL_N_K_THRESHOLD: i32 = 4096;

/// Affine-quantised KV cache. Stores K/V as packed-uint32
/// `(wq, scales, biases)` triples in step-allocated buffers.
///
/// Default path is dequantise-on-read. Opt-in builders for the faster
/// paths: [`Self::with_quantized_matmul`] (keep K/V packed across
/// score + attend), [`Self::with_rotation`] (Π pre-quantize for
/// better 4-bit quality), [`Self::with_fused_kernel`] (single-dispatch
/// Metal kernel for n_q=1 decode).
///
/// Memory: `bits=8` → ~2×, `bits=4` → ~4× reduction vs fp16.
/// `head_dim` must be a multiple of `group_size`.
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
    /// Random orthogonal rotation Π applied pre-quantize.
    rotation: Option<Array>,
    /// `rotation` cast to the input dtype (set lazily on first call).
    /// `append_quantised` + the SDPA fallback both need this view;
    /// caching avoids a per-call `as_dtype` launch.
    rotation_input_dtype: Option<Array>,
    /// `rotation` cast to the dequantised dtype (used by the un-rotate
    /// step after fused/steel attention).
    rotation_out_dtype: Option<Array>,
    /// Route scores/attend through `quantized_matmul` (packed-matmul path).
    use_quantized_matmul: bool,
    /// Use the in-tree fused qsdpa Metal kernel for n_q=1 decode.
    use_fused_kernel: bool,
    /// Use the steel quantised tile kernel for n_q > 1 prefill.
    use_steel_prefill: bool,
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
    ///
    /// Default: `use_quantized_matmul = true` (packed-matmul, no
    /// per-step dequant). Packed beats dequant by ~5% at T=1024 and
    /// ~78% at T=8192 on Qwen3-1.7B-q4 + KV q8 (Apple M4 Max). Use
    /// `with_dequant_path()` to opt back in.
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
            rotation: None,
            rotation_input_dtype: None,
            rotation_out_dtype: None,
            use_quantized_matmul: true,
            use_fused_kernel: false,
            use_steel_prefill: false,
        }
    }

    /// Random orthogonal Π applied to K/V pre-quantize for better
    /// 4-bit quality (KL 0.039 vs 0.20 unrotated on Qwen3-1.7B-bf16).
    pub fn with_rotation(mut self, head_dim: i32, seed: u64) -> Result<Self, Error> {
        self.rotation = Some(rotation::generate_rotation_matrix(head_dim, seed)?);
        self.rotation_input_dtype = None;
        self.rotation_out_dtype = None;
        Ok(self)
    }

    /// Get the cached rotation matrix cast to `dtype`, building it on
    /// first access. Used by the rotate / un-rotate paths so the cast
    /// launch happens once per dtype instead of every call.
    fn rotation_for(&mut self, dtype: Dtype) -> Result<Option<&Array>, Exception> {
        let Some(pi) = self.rotation.as_ref() else {
            return Ok(None);
        };
        if pi.dtype() == dtype {
            return Ok(Some(pi));
        }
        let slot = match (self.dtype, dtype) {
            (Some(d), x) if d == x => &mut self.rotation_input_dtype,
            _ => &mut self.rotation_out_dtype,
        };
        if slot.is_none() {
            *slot = Some(pi.as_dtype(dtype)?);
        }
        Ok(slot.as_ref())
    }

    /// Keep K/V packed across score and attend; dispatch through
    /// `quantized_matmul` × 2 instead of dequantising on read.
    /// No-op — packed-matmul is the default; kept for API symmetry.
    pub fn with_quantized_matmul(mut self) -> Self {
        self.use_quantized_matmul = true;
        self
    }

    /// Opt back into the dequant-on-read path. Per-step cost grows
    /// linearly with cache history because the full K/V buffer must be
    /// dequantised every decode step. Use only when packed-matmul is
    /// not viable (e.g. an op composition that requires dense K/V).
    pub fn with_dequant_path(mut self) -> Self {
        self.use_quantized_matmul = false;
        self
    }

    /// Opt into the in-tree fused quantized SDPA Metal kernel for n_q=1
    /// decode. Requires `with_quantized_matmul()`; falls back to the
    /// `quantized_matmul`-composed path for unsupported shapes
    /// (`n_q > 1`, `bits ∉ {4, 8}`, `n_k > 4096`). The n_k threshold is
    /// a perf crossover — at long context mlx's tiled `quantized_matmul`
    /// beats the kernel's per-simdgroup serial processing.
    pub fn with_fused_kernel(mut self) -> Self {
        self.use_fused_kernel = true;
        self
    }

    /// Opt into the steel-attention quantised tile kernel for `n_q > 1`
    /// prefill. Active only when `head_dim ∈ {128, 256}`, `bits ∈ {4, 8}`,
    /// `group_size` divides `head_dim`, and the caller passes no explicit
    /// mask (causal-only). Falls back to the existing
    /// `quantized_scaled_dot_product_attention` ops-composed path
    /// otherwise.
    pub fn with_steel_prefill(mut self) -> Self {
        self.use_steel_prefill = true;
        self
    }

    /// Steel quantised tile kernel applicability check. Active for
    /// `n_q > 1` prefill with supported head_dim, 4/8-bit quantisation,
    /// and group_size dividing head_dim. Caller mask is dropped; the
    /// kernel applies causal+ql_off internally.
    fn can_dispatch_steel_quant(&self, n_q: i32, head_dim: i32, h_q: i32, h_kv: i32) -> bool {
        self.use_steel_prefill
            && n_q > 1
            && STEEL_SUPPORTED_HEAD_DIMS.contains(&head_dim)
            && matches!(self.bits, 4 | 8)
            && head_dim % self.group_size == 0
            && h_q % h_kv == 0
    }

    /// Fused qsdpa kernel applicability check. Active for n_q=1 decode
    /// with 4/8-bit quantisation, packed alignment requirements
    /// (head_dim divisible by elements-per-uint32), and short-enough
    /// history (cached n_k <= FUSED_KERNEL_N_K_THRESHOLD; longer
    /// histories lose to mlx's tiled `quantized_matmul`).
    fn can_dispatch_fused(
        &self,
        n_q: i32,
        head_dim: i32,
        h_q: i32,
        h_kv: i32,
        n_k_cache: i32,
    ) -> bool {
        self.use_fused_kernel
            && n_q == 1
            && matches!(self.bits, 4 | 8)
            && head_dim % (32 / self.bits) == 0
            && head_dim % self.group_size == 0
            && h_q % h_kv == 0
            && n_k_cache <= FUSED_KERNEL_N_K_THRESHOLD
    }

    /// Append (keys, values), quantise into the buffer, return packed views
    /// sliced to populated `[..., :offset, ...]` rows. Shared back-end for
    /// `update_and_fetch` and the packed-matmul `attention` override.
    #[allow(
        clippy::type_complexity,
        reason = "6-tuple of Array refs matches the C-kernel input layout"
    )]
    fn append_quantised(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<((Array, Array, Array), (Array, Array, Array)), Exception> {
        let s = keys.shape()[keys.shape().len() - 2];

        if self.dtype.is_none() {
            self.dtype = Some(keys.dtype());
        }

        let keys_dtype = keys.dtype();
        let (keys_to_q, values_to_q) = match self.rotation_for(keys_dtype)? {
            Some(pi) => {
                let pi_t = pi.transpose_axes(&[1, 0])?;
                (keys.matmul(&pi_t)?, values.matmul(&pi_t)?)
            }
            None => (keys, values),
        };

        let (new_k_wq, new_k_scales, new_k_biases) =
            quantize(&keys_to_q, self.group_size, self.bits)?;
        let (new_v_wq, new_v_scales, new_v_biases) =
            quantize(&values_to_q, self.group_size, self.bits)?;

        self.grow_to_fit(
            &new_k_wq,
            &new_k_scales,
            &new_k_biases,
            &new_v_wq,
            &new_v_scales,
            &new_v_biases,
        )?;

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

        Ok((
            (
                k_wq_buf.index((Ellipsis, 0..end, ..)),
                k_s_buf.index((Ellipsis, 0..end, ..)),
                k_b_buf.index((Ellipsis, 0..end, ..)),
            ),
            (
                v_wq_buf.index((Ellipsis, 0..end, ..)),
                v_s_buf.index((Ellipsis, 0..end, ..)),
                v_b_buf.index((Ellipsis, 0..end, ..)),
            ),
        ))
    }

    /// Reconstruct from previously-persisted `state` + `meta_state`.
    /// `state` order: `[k_wq, k_scales, k_biases, v_wq, v_scales, v_biases]`.
    pub fn from_state(state: Vec<Array>, meta: &HashMap<String, String>) -> Result<Self, Error> {
        let [k_wq, k_s, k_b, v_wq, v_s, v_b]: [Array; 6] =
            state.try_into().map_err(|v: Vec<Array>| {
                Error::Other(
                    format!(
                        "QuantizedKVCache::from_state expected 6 arrays, got {}",
                        v.len()
                    )
                    .into(),
                )
            })?;
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
            rotation: None,
            rotation_input_dtype: None,
            rotation_out_dtype: None,
            use_quantized_matmul: false,
            use_fused_kernel: false,
            use_steel_prefill: false,
        })
    }

    /// Grow each of the six pre-allocated buffers so they have room for
    /// `additional` more tokens beyond the current `offset`. Allocates
    /// fresh zero buffers at the target capacity and copies the populated
    /// `[:offset]` rows over.
    #[allow(
        clippy::too_many_arguments,
        reason = "6 Array refs match the C-kernel quantised-tensor inputs"
    )]
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
        let target_cap = super::trait_def::ceil_step(self.offset + new_tokens, self.step);

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
            for (g, o) in grown.iter_mut().zip(olds) {
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
        let ((k_wq, k_s, k_b), (v_wq, v_s, v_b)) = self.append_quantised(keys, values)?;
        let k_dense = dequantize(&k_wq, &k_s, &k_b, self.group_size, self.bits)?;
        let v_dense = dequantize(&v_wq, &v_s, &v_b, self.group_size, self.bits)?;

        let k_dtype = k_dense.dtype();
        match self.rotation_for(k_dtype)? {
            Some(pi) => Ok((k_dense.matmul(pi)?, v_dense.matmul(pi)?)),
            None => Ok((k_dense, v_dense)),
        }
    }

    /// Packed-matmul path: keep K/V packed and dispatch through
    /// `quantized_scaled_dot_product_attention` (two `quantized_matmul`
    /// kernels). Skips the per-step K/V dequant. Falls through to the
    /// dense-SDPA default when `use_quantized_matmul` is off.
    fn attention(
        &mut self,
        queries: &Array,
        keys: Array,
        values: Array,
        scale: f32,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        if !self.use_quantized_matmul {
            let (k_full, v_full) = self.update_and_fetch(keys, values)?;
            return scaled_dot_product_attention(
                queries.clone(),
                k_full,
                v_full,
                scale,
                mask.map(ScaledDotProductAttentionMask::Array),
                None,
            );
        }

        let ql_off = self.offset;
        let ((k_wq, k_s, k_b), (v_wq, v_s, v_b)) = self.append_quantised(keys, values)?;

        let q_dtype = queries.dtype();
        let queries_for_sdpa = match self.rotation_for(q_dtype)? {
            Some(pi) => {
                let pi_t = pi.transpose_axes(&[1, 0])?;
                queries.matmul(&pi_t)?
            }
            None => queries.clone(),
        };

        let q_shape = queries_for_sdpa.shape();
        let head_dim = q_shape[q_shape.len() - 1];
        let n_q = q_shape[q_shape.len() - 2];
        let h_q = q_shape[1];
        let h_kv = k_wq.shape()[1];
        let n_k_cache = k_wq.shape()[2];

        let steel_quant_supported = self.can_dispatch_steel_quant(n_q, head_dim, h_q, h_kv);
        let fused_supported = self.can_dispatch_fused(n_q, head_dim, h_q, h_kv, n_k_cache);

        let out = if steel_quant_supported {
            steel_quant_attention_dispatch(
                cached_steel_quant_attention_kernel(),
                SteelQuantAttentionInputs {
                    q: &queries_for_sdpa,
                    k_wq: &k_wq,
                    k_scales: &k_s,
                    k_biases: &k_b,
                    v_wq: &v_wq,
                    v_scales: &v_s,
                    v_biases: &v_b,
                    mask: None,
                    causal: true,
                    ql_off,
                    scale,
                    head_dim,
                    h_q,
                    h_kv,
                    bits: self.bits,
                    group_size: self.group_size,
                },
            )?
        } else if fused_supported {
            fused_qsdpa_decode(
                cached_fused_qsdpa_kernel(),
                FusedQsdpaInputs {
                    q: &queries_for_sdpa,
                    k_wq: &k_wq,
                    k_scales: &k_s,
                    k_biases: &k_b,
                    v_wq: &v_wq,
                    v_scales: &v_s,
                    v_biases: &v_b,
                    mask,
                    scale,
                    head_dim,
                    group_size: self.group_size,
                    bits: self.bits,
                    h_q,
                    h_kv,
                },
            )?
        } else {
            let q_keys = QuantizedKeys {
                keys: k_wq,
                scales: k_s,
                biases: k_b,
            };
            let q_values = QuantizedValues {
                values: v_wq,
                scales: v_s,
                biases: v_b,
            };
            quantized_scaled_dot_product_attention(
                queries_for_sdpa,
                q_keys,
                q_values,
                scale,
                mask,
                self.group_size,
                self.bits,
            )?
        };

        let out_dtype = out.dtype();
        match self.rotation_for(out_dtype)? {
            Some(pi) => out.matmul(pi),
            None => Ok(out),
        }
    }
}
