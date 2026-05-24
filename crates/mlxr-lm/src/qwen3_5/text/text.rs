//! Qwen3.5 text decoder building blocks: full-attention layer
//! (`Qwen3_5Attention`) and the SwiGLU MLP.

use mlxr::{
    builder::Builder,
    fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask},
    layers,
    macros::{ModuleParameters, Quantizable},
    module::Module,
    ops::{arange, broadcast_to, reshape, split, transpose_axes},
    quantization::MaybeQuantized,
    Array,
};

use super::config::TextConfig;
use super::rope::{apply_multimodal_rotary_pos_emb, MultimodalRope};
use crate::activations::{attention_gate, swiglu, AttentionGateCache, SwigluCache};
use crate::attention::{attention_dispatch, AttentionInputs};
use crate::cache::kernels::{cached_attention_kernel, SUPPORTED_HEAD_DIMS};
use crate::cache::{KVCache, KeyValueCache};
use crate::error::Error;
use crate::utils::create_attention_mask;

/// SwiGLU feed-forward block: `down(silu(gate(x)) * up(x))`.
#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Mlp {
    #[quantizable]
    #[param]
    pub gate_proj: MaybeQuantized<layers::Linear>,

    #[quantizable]
    #[param]
    pub down_proj: MaybeQuantized<layers::Linear>,

    #[quantizable]
    #[param]
    pub up_proj: MaybeQuantized<layers::Linear>,

    /// Per-layer compiled-graph cache for [`swiglu`].
    swiglu_cache: SwigluCache,
}

impl Mlp {
    /// Build a freshly-initialised MLP with the given inner widths.
    pub fn new(dim: i32, hidden_dim: i32) -> Result<Self, Error> {
        let gate_proj = layers::LinearBuilder::new(dim, hidden_dim)
            .bias(false)
            .build()?;
        let down_proj = layers::LinearBuilder::new(hidden_dim, dim)
            .bias(false)
            .build()?;
        let up_proj = layers::LinearBuilder::new(dim, hidden_dim)
            .bias(false)
            .build()?;
        Ok(Self {
            gate_proj: MaybeQuantized::Original(gate_proj),
            down_proj: MaybeQuantized::Original(down_proj),
            up_proj: MaybeQuantized::Original(up_proj),
            swiglu_cache: SwigluCache::default(),
        })
    }
}

impl Module<&Array> for Mlp {
    type Output = Array;
    type Error = Error;

    /// SwiGLU forward: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
    fn forward(&mut self, x: &Array) -> Result<Array, Error> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = swiglu(&mut self.swiglu_cache, &gate, &up)?;
        Ok(self.down_proj.forward(&activated)?)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
    }
}

/// Full-attention block used at every `full_attention_interval`-th layer.
///
/// Differences from a vanilla GQA attention:
///
/// - `q_proj` outputs `n_heads * head_dim * 2` features and is split into
///   queries and a per-head gate (`attn_output_gate`). The attention output is
///   element-wise multiplied by `sigmoid(gate)` before `o_proj`.
/// - Rotary embedding is the multimodal partial-rotary mrope from
///   [`super::rope::MultimodalRope`]; only the first
///   `head_dim * partial_rotary_factor` features get rotated.
/// - `position_ids` may be supplied explicitly (multimodal flow). When `None`,
///   the cache offset is used to derive a 1-D `[B, S]` range that the mrope
///   broadcasts across the three axes.
#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Attention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

    #[quantizable]
    #[param]
    pub q_proj: MaybeQuantized<layers::Linear>,
    #[quantizable]
    #[param]
    pub k_proj: MaybeQuantized<layers::Linear>,
    #[quantizable]
    #[param]
    pub v_proj: MaybeQuantized<layers::Linear>,
    #[quantizable]
    #[param]
    pub o_proj: MaybeQuantized<layers::Linear>,
    #[param]
    pub q_norm: layers::RmsNorm,
    #[param]
    pub k_norm: layers::RmsNorm,

    /// Not a learnable parameter — kept inline because mrope is stateless
    /// (`inv_freq` and `axis_index` are precomputed at construction).
    rope: MultimodalRope,

    /// Per-layer compiled-graph cache for [`attention_gate`].
    attention_gate_cache: AttentionGateCache,

    /// When set, prefill (`l > 1`, no explicit mask) routes through
    /// `attention_dispatch(causal=true)` instead of
    /// `fast::SDPA(Causal)`. Off by default; toggle with
    /// [`Self::set_use_steel_prefill`].
    use_steel_prefill: bool,
}

impl Attention {
    /// Build a freshly-initialised attention block from a [`TextConfig`].
    pub fn new(cfg: &TextConfig) -> Result<Self, Error> {
        let dim = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let scale = (head_dim as f32).sqrt().recip();

        // q_proj has 2× the head_dim per head — half is queries, half is the gate.
        let q_proj = layers::LinearBuilder::new(dim, n_heads * head_dim * 2)
            .bias(cfg.attention_bias)
            .build()?;
        let k_proj = layers::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(cfg.attention_bias)
            .build()?;
        let v_proj = layers::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(cfg.attention_bias)
            .build()?;
        let o_proj = layers::LinearBuilder::new(n_heads * head_dim, dim)
            .bias(cfg.attention_bias)
            .build()?;

        let q_norm = layers::RmsNormBuilder::new(head_dim)
            .eps(cfg.rms_norm_eps)
            .build()?;
        let k_norm = layers::RmsNormBuilder::new(head_dim)
            .eps(cfg.rms_norm_eps)
            .build()?;

        let rotary_dim =
            (head_dim as f32 * cfg.rope_parameters.partial_rotary_factor).floor() as i32;
        // Unsupported rope variants (yarn / longrope) reject at
        // `config.json` deserialize via `QwenRopeType`, so the only
        // values reaching here are `default` / `mrope`.
        let rope = MultimodalRope::new(
            rotary_dim,
            cfg.rope_parameters.rope_theta,
            &cfg.rope_parameters.mrope_section,
        )?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            q_proj: MaybeQuantized::Original(q_proj),
            k_proj: MaybeQuantized::Original(k_proj),
            v_proj: MaybeQuantized::Original(v_proj),
            o_proj: MaybeQuantized::Original(o_proj),
            q_norm,
            k_norm,
            rope,
            attention_gate_cache: AttentionGateCache::default(),
            use_steel_prefill: false,
        })
    }

    /// Opt this layer into the steel-attention tiled prefill kernel for
    /// `l > 1` calls with no explicit mask. Steel supports head_dim ∈
    /// {128, 256}; layers with other dims silently keep `fast::SDPA`.
    pub fn set_use_steel_prefill(&mut self, on: bool) {
        self.use_steel_prefill = on;
    }

    /// Run the attention block.
    ///
    /// - `x`: `[B, L, hidden_size]`
    /// - `mask`: optional additive causal mask of shape `[..., L, kv_len]`.
    /// - `cache`: optional KV cache — mutated in place.
    /// - `position_ids`: optional `[3, B, L]` (or `[B, L]` text-only) tensor.
    ///   When `None`, derived from `cache.offset`.
    ///
    /// Returns `[B, L, hidden_size]`.
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
        position_ids: Option<&Array>,
    ) -> Result<Array, Error> {
        let shape = x.shape();
        let b = shape[0];
        let l = shape[1];

        // Project once, then split queries from the gate.
        let qp = self.q_proj.forward(x)?;
        let qp = reshape(&qp, &[b, l, self.n_heads, 2 * self.head_dim])?;
        let qg = split(&qp, 2, -1)?;
        let queries = &qg[0];
        let gate = reshape(&qg[1], &[b, l, self.n_heads * self.head_dim])?;

        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        // [B, L, H, D] -> [B, H, L, D] for q/k/v.
        let queries = self.q_norm.forward(queries)?;
        let queries = transpose_axes(&queries, &[0, 2, 1, 3])?;

        let keys = reshape(&keys, &[b, l, self.n_kv_heads, self.head_dim])?;
        let keys = self.k_norm.forward(&keys)?;
        let keys = transpose_axes(&keys, &[0, 2, 1, 3])?;

        let values = reshape(&values, &[b, l, self.n_kv_heads, self.head_dim])?;
        let values = transpose_axes(&values, &[0, 2, 1, 3])?;

        // Resolve position ids. MultimodalRope::cos_sin accepts either
        // 2D [B, L] (auto-expands to text-only mrope) or 3D [3, B, L].
        let offset = cache.as_ref().map(|c| c.offset()).unwrap_or(0);
        let owned_pos;
        let pos: &Array = if let Some(p) = position_ids {
            p
        } else {
            let range = arange::<_, i32>(offset, offset + l, None)?;
            let range = reshape(&range, &[1, l])?;
            owned_pos = broadcast_to(&range, &[b, l])?;
            &owned_pos
        };

        let (cos, sin) = self.rope.cos_sin(pos)?;
        let (queries, keys) = apply_multimodal_rotary_pos_emb(&queries, &keys, &cos, &sin)?;

        let (keys, values) = if let Some(cache) = cache {
            cache.update_and_fetch(keys, values)?
        } else {
            (keys, values)
        };

        // Steel prefill is eligible whenever the toggle is on and the
        // shape is prefill (l > 1) at a supported head_dim. The supplied
        // `mask` is the decoder's standard causal mask
        // (`Qwen35Decoder::build_full_attn_mask`); we replace it with
        // the kernel's `causal=true` semantics rather than feeding the
        // boolean tensor through. `qL_off = offset` shifts the diagonal
        // for prefill calls that continue past prior decode tokens.
        let steel_eligible =
            self.use_steel_prefill && l > 1 && SUPPORTED_HEAD_DIMS.contains(&self.head_dim);

        let output = if steel_eligible {
            attention_dispatch(
                cached_attention_kernel(),
                AttentionInputs {
                    q: &queries,
                    k: &keys,
                    v: &values,
                    mask: None,
                    causal: true,
                    ql_off: offset,
                    scale: self.scale,
                    head_dim: self.head_dim,
                    h_q: self.n_heads,
                    h_kv: self.n_kv_heads,
                },
            )?
        } else {
            match mask {
                Some(m) => scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    self.scale,
                    ScaledDotProductAttentionMask::Array(m),
                    None,
                )?,
                None if l > 1 => scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    self.scale,
                    ScaledDotProductAttentionMask::Causal,
                    None,
                )?,
                None => scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    self.scale,
                    Option::<ScaledDotProductAttentionMask<'_>>::None,
                    None,
                )?,
            }
        };
        let output = transpose_axes(&output, &[0, 2, 1, 3])?;
        let output = reshape(&output, &[b, l, -1])?;

        let gated = attention_gate(&mut self.attention_gate_cache, &output, &gate)?;
        Ok(self.o_proj.forward(&gated)?)
    }

    /// Toggle training mode on every quantisable parameter.
    pub fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
        self.q_norm.training_mode(mode);
        self.k_norm.training_mode(mode);
    }
}

/// Convenience helper: build a causal attention mask for a prefill of length
/// `T`, given the current offset of the layer's KV cache.
///
/// This mirrors `mlxr_lm::utils::create_attention_mask` but specialises the
/// cache type to [`KVCache`] so callers don't need a generic.
pub fn full_attention_mask(h: &Array, cache: &[Option<KVCache>]) -> Result<Option<Array>, Error> {
    create_attention_mask(h, cache)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    #![allow(clippy::print_stdout, reason = "test code")]
    #![allow(clippy::print_stderr, reason = "test code")]
    use super::super::config::RopeParameters;
    use super::*;
    use crate::cache::KeyValueCache;
    use mlxr::{random::uniform, transforms::eval};

    fn synthetic_text_config() -> TextConfig {
        let json = serde_json::json!({
            "model_type": "qwen3_5_text",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rms_norm_eps": 1e-6,
            "vocab_size": 100,
            "max_position_embeddings": 256,
            "layer_types": ["full_attention"],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "rope_parameters": {
                "mrope_section": [2, 1, 1],
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "type": "default"
            }
        });
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn unsupported_rope_type_rejected_at_deserialize() {
        let json = serde_json::json!({
            "mrope_section": [2, 1, 1],
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "type": "yarn"
        });
        let err = serde_json::from_value::<RopeParameters>(json).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("yarn"), "unexpected error: {msg}");
    }

    #[test]
    fn mlp_shape_round_trips() {
        let mut m = Mlp::new(32, 64).unwrap();
        let x = uniform::<_, f32>(0.0, 1.0, &[2, 4, 32], None).unwrap();
        let y = m.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 4, 32]);
    }

    #[test]
    fn attention_shape_round_trips_without_cache() {
        let cfg = synthetic_text_config();
        let mut a = Attention::new(&cfg).unwrap();
        let x = uniform::<_, f32>(0.0, 1.0, &[2, 4, cfg.hidden_size], None).unwrap();
        let y = a.forward(&x, None, None, None).unwrap();
        assert_eq!(y.shape(), &[2, 4, cfg.hidden_size]);
    }

    /// A config whose `head_dim = 128` so the steel-prefill path
    /// actually fires (the default synthetic uses `head_dim = 8`).
    fn steel_eligible_text_config() -> TextConfig {
        let json = serde_json::json!({
            "model_type": "qwen3_5_text",
            "hidden_size": 256,
            "intermediate_size": 256,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "rms_norm_eps": 1e-6,
            "vocab_size": 100,
            "max_position_embeddings": 256,
            "layer_types": ["full_attention"],
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "linear_conv_kernel_dim": 4,
            "rope_parameters": {
                "mrope_section": [32, 16, 16],
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "type": "default"
            }
        });
        serde_json::from_value(json).unwrap()
    }

    /// Toggling `set_use_steel_prefill(true)` should produce numerically
    /// matching prefill output vs the default `fast::SDPA(Causal)` path.
    #[test]
    fn attention_steel_prefill_matches_fast_sdpa() {
        let cfg = steel_eligible_text_config();
        let prompt = uniform::<_, f32>(0.0, 1.0, &[1, 8, cfg.hidden_size], None).unwrap();

        let mut a_ref = Attention::new(&cfg).unwrap();
        let baseline = a_ref.forward(&prompt, None, None, None).unwrap();

        // Reset & rebuild a steel-prefill Attention with identical weights.
        // Easiest: clone the projections via from-the-same-seed init isn't
        // exposed, so we run with the same `Attention` after toggling.
        let mut a_steel = a_ref;
        a_steel.set_use_steel_prefill(true);
        let routed = a_steel.forward(&prompt, None, None, None).unwrap();

        eval([&baseline, &routed]).unwrap();
        let diff = baseline
            .subtract(&routed)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(
            diff < 5e-3,
            "qwen3_5 Attention: steel-prefill vs fast::SDPA(Causal) diverge: max_abs={diff}"
        );
    }

    /// Production path: the decoder builds a `[1, 1, T, T]` boolean
    /// causal mask and passes it to `Attention::forward`. With steel
    /// toggled on, the kernel must ignore the supplied mask and apply
    /// its own `causal=true` logic, yielding the same output as
    /// `fast::SDPA(Array(mask))`.
    #[test]
    fn attention_steel_prefill_ignores_decoder_causal_mask() {
        let cfg = steel_eligible_text_config();
        let l = 8;
        let prompt = uniform::<_, f32>(0.0, 1.0, &[1, l, cfg.hidden_size], None).unwrap();

        // Build a `[1, 1, T, T]` lower-triangular bool mask the same
        // way `Qwen35Decoder::build_full_attn_mask` does for offset=0.
        let mut buf = Vec::with_capacity((l * l) as usize);
        for i in 0..l {
            for j in 0..l {
                buf.push(j <= i);
            }
        }
        let mask_2d = Array::from_slice(&buf, &[l, l]);
        let mask_4d = mask_2d.expand_dims_axes(&[0, 1]).unwrap();

        let mut a_ref = Attention::new(&cfg).unwrap();
        let baseline = a_ref.forward(&prompt, Some(&mask_4d), None, None).unwrap();

        let mut a_steel = a_ref;
        a_steel.set_use_steel_prefill(true);
        let routed = a_steel
            .forward(&prompt, Some(&mask_4d), None, None)
            .unwrap();

        eval([&baseline, &routed]).unwrap();
        let diff = baseline
            .subtract(&routed)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(
            diff < 5e-3,
            "qwen3_5 Attention with decoder mask: steel-prefill vs fast::SDPA(Array) diverge: max_abs={diff}"
        );
    }

    #[test]
    fn attention_prefill_then_decode_extends_cache() {
        let cfg = synthetic_text_config();
        let mut a = Attention::new(&cfg).unwrap();

        let prompt = uniform::<_, f32>(0.0, 1.0, &[1, 5, cfg.hidden_size], None).unwrap();
        let mut cache = KVCache::new();
        let prefill_out = a.forward(&prompt, None, Some(&mut cache), None).unwrap();
        eval([&prefill_out]).unwrap();
        assert_eq!(prefill_out.shape(), &[1, 5, cfg.hidden_size]);
        assert_eq!(cache.offset(), 5);

        let next = uniform::<_, f32>(0.0, 1.0, &[1, 1, cfg.hidden_size], None).unwrap();
        let decode_out = a.forward(&next, None, Some(&mut cache), None).unwrap();
        eval([&decode_out]).unwrap();
        assert_eq!(decode_out.shape(), &[1, 1, cfg.hidden_size]);
        assert_eq!(cache.offset(), 6);
    }
}
