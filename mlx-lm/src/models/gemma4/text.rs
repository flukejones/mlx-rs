//! Gemma 4 text-only model.
//!
//! Sliding+global attention with per-layer head_dim (256/512), partial
//! RoPE, K-eq-V projection, KV sharing across the last
//! `num_kv_shared_layers` layers, final logit softcapping, fp16-safe
//! `clip_residual`. MoE expert routing (26B-A4B / 31B-MoE) and
//! per-layer input embeddings (E2B/E4B) are wired.

use std::collections::HashMap;
use std::sync::OnceLock;

use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::macros::{ModuleParameters, Quantizable};
use mlx_rs::module::{
    Module, ModuleParamMut, ModuleParamRef, ModuleParameters, Param,
};
use mlx_rs::nn::{self, gelu_approximate, RopeInput};
use mlx_rs::fast;
use mlx_rs::ops::clip;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{argpartition_axis, expand_dims_axes, unflatten};
use mlx_rs::quantization::MaybeQuantized;
use mlx_rs::{Array, Dtype};

use crate::activations::{
    geglu, logit_softcap, residual_add_scale, router_post, GegluCache, LogitSoftcapCache,
    ResidualAddScaleCache, RouterPostCache,
};
use crate::cache::KeyValueCache;
use crate::nn::ensure_cache_populated;
use crate::models::gemma4::config::{Gemma4Config, LayerKind};
use crate::models::gemma4::rope::ProportionalRope;
use crate::models::gemma4::switch_layers::SwitchGLU;
use crate::utils::create_attention_mask;
use crate::utils::rope::FloatOrString;

// Re-export canonical input struct at the historical path so external
// consumers (loader, bench, examples) keep compiling.
pub use crate::nn::{AttentionInput, ModelInput};

/// fp16 max — matches `mx.finfo(mx.float16).max`.
const FP16_MAX: f32 = 65504.0;

/// RMS-norm without learnable scale — wraps `fast::rms_norm(x, None, eps)`.
#[derive(Debug, Clone, ModuleParameters)]
pub struct RmsNormNoScale {
    pub eps: f32,
}

impl RmsNormNoScale {
    pub fn new(eps: f32) -> Self {
        Self { eps }
    }
}

impl Module<&Array> for RmsNormNoScale {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        fast::rms_norm(x, None, self.eps)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

/// Per-layer RoPE: proportional (partial rotation) or plain RoPE.
#[derive(Debug, Clone)]
pub enum LayerRope {
    Plain(nn::Rope),
    Proportional(ProportionalRope),
}

impl ModuleParameters for LayerRope {
    fn num_parameters(&self) -> usize {
        match self {
            Self::Plain(r) => r.num_parameters(),
            Self::Proportional(r) => r.num_parameters(),
        }
    }
    fn freeze_parameters(&mut self, r: bool) {
        match self {
            Self::Plain(p) => p.freeze_parameters(r),
            Self::Proportional(p) => p.freeze_parameters(r),
        }
    }
    fn unfreeze_parameters(&mut self, r: bool) {
        match self {
            Self::Plain(p) => p.unfreeze_parameters(r),
            Self::Proportional(p) => p.unfreeze_parameters(r),
        }
    }
    fn parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Plain(p) => p.parameters(),
            Self::Proportional(p) => p.parameters(),
        }
    }
    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        match self {
            Self::Plain(p) => p.parameters_mut(),
            Self::Proportional(p) => p.parameters_mut(),
        }
    }
    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Plain(p) => p.trainable_parameters(),
            Self::Proportional(p) => p.trainable_parameters(),
        }
    }
    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::Plain(p) => p.all_frozen(),
            Self::Proportional(p) => p.all_frozen(),
        }
    }
    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::Plain(p) => p.any_frozen(),
            Self::Proportional(p) => p.any_frozen(),
        }
    }
}

impl LayerRope {
    pub fn forward(&mut self, input: RopeInput<'_>) -> Result<Array, Exception> {
        match self {
            Self::Plain(r) => r.forward(input),
            Self::Proportional(r) => r.forward(input),
        }
    }

    /// Same as [`forward`] but takes a 0-D `Array` offset, routing
    /// through `mlx::fast::rope_dynamic` instead of `rope`. Lets the
    /// per-step decode offset stay on-device so MLX's compile cache
    /// reuses the same kernel across decode steps (Python parity:
    /// `mx.fast.rope(x, offset=mx.array(cache.offset))`).
    pub fn forward_dynamic(
        &self,
        x: &Array,
        offset: &Array,
    ) -> Result<Array, Exception> {
        match self {
            Self::Plain(r) => fast::rope_dynamic(
                x.clone(),
                r.dimensions,
                r.traditional,
                r.base,
                r.scale,
                offset,
                None,
            ),
            Self::Proportional(p) => fast::rope_dynamic(
                x.clone(),
                p.dims,
                p.traditional,
                None,
                1.0_f32,
                offset,
                Some(&p.freqs),
            ),
        }
    }
}

fn build_layer_rope(
    head_dim: i32,
    kind: LayerKind,
    rope_traditional: bool,
    rope_parameters: Option<&HashMap<String, HashMap<String, FloatOrString>>>,
) -> Result<LayerRope, Exception> {
    let layer_key = match kind {
        LayerKind::FullAttention => "full_attention",
        LayerKind::SlidingAttention => "sliding_attention",
    };
    let params = rope_parameters.and_then(|m| m.get(layer_key));
    let rope_theta = params
        .and_then(|p| p.get("rope_theta"))
        .and_then(|v| match v {
            FloatOrString::Float(f) => Some(*f),
            FloatOrString::String(_) => None,
        })
        .unwrap_or(10_000.0);
    let rope_type = params
        .and_then(|p| p.get("rope_type"))
        .and_then(|v| match v {
            FloatOrString::String(s) => Some(s.as_str()),
            FloatOrString::Float(_) => None,
        })
        .unwrap_or("default");
    let partial_rotary_factor = params
        .and_then(|p| p.get("partial_rotary_factor"))
        .and_then(|v| match v {
            FloatOrString::Float(f) => Some(*f),
            FloatOrString::String(_) => None,
        })
        .unwrap_or(1.0);
    let factor = params
        .and_then(|p| p.get("factor"))
        .and_then(|v| match v {
            FloatOrString::Float(f) => Some(*f),
            FloatOrString::String(_) => None,
        })
        .unwrap_or(1.0);

    let rotated_dims = ((head_dim as f32) * partial_rotary_factor) as i32 & !1;

    if rope_type == "proportional" || (rope_type == "default" && rotated_dims < head_dim) {
        Ok(LayerRope::Proportional(ProportionalRope::new(
            head_dim,
            rotated_dims,
            rope_traditional,
            rope_theta,
            factor,
        )?))
    } else {
        let rope = nn::RopeBuilder::new(head_dim)
            .traditional(rope_traditional)
            .base(rope_theta)
            .scale(1.0)
            .build()
            .expect("Infallible");
        Ok(LayerRope::Plain(rope))
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Attention {
    pub layer_idx: i32,
    pub layer_kind: LayerKind,
    pub is_sliding: bool,
    pub has_kv: bool,
    pub use_k_eq_v: bool,

    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

    #[quantizable]
    #[param]
    pub q_proj: MaybeQuantized<nn::Linear>,
    /// `Some` iff this layer owns its own K/V projections (not a
    /// KV-shared layer).
    #[quantizable]
    #[param]
    pub k_proj: Option<MaybeQuantized<nn::Linear>>,
    /// `None` when K == V (full-attention layers with
    /// `attention_k_eq_v=true`), else the dedicated V projection.
    #[quantizable]
    #[param]
    pub v_proj: Option<MaybeQuantized<nn::Linear>>,
    #[quantizable]
    #[param]
    pub o_proj: MaybeQuantized<nn::Linear>,

    #[param]
    pub q_norm: nn::RmsNorm,
    #[param]
    pub k_norm: Option<nn::RmsNorm>,
    #[param]
    pub v_norm: Option<RmsNormNoScale>,

    #[param]
    pub rope: LayerRope,
}

impl Attention {
    pub fn new(args: &Gemma4Config, layer_idx: i32) -> Result<Self, Exception> {
        let layer_types = args.layer_types_resolved();
        let layer_kind = layer_types[layer_idx as usize];
        let is_sliding = matches!(layer_kind, LayerKind::SlidingAttention);

        let first_kv_shared = args.num_hidden_layers - args.num_kv_shared_layers;
        let has_kv = layer_idx < first_kv_shared;

        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let head_dim = if matches!(layer_kind, LayerKind::FullAttention) {
            args.global_head_dim
        } else {
            args.head_dim
        };

        let use_k_eq_v = args.attention_k_eq_v && !is_sliding;
        let n_kv_heads = match (use_k_eq_v, args.num_global_key_value_heads) {
            (true, Some(h)) => h,
            _ => args.num_key_value_heads,
        };

        let scale = 1.0_f32;

        let q_proj = nn::LinearBuilder::new(dim, n_heads * head_dim).bias(false).build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, dim).bias(false).build()?;

        let (k_proj, v_proj) = if has_kv {
            let k = nn::LinearBuilder::new(dim, n_kv_heads * head_dim).bias(false).build()?;
            let v = if use_k_eq_v {
                None
            } else {
                Some(MaybeQuantized::Original(
                    nn::LinearBuilder::new(dim, n_kv_heads * head_dim).bias(false).build()?,
                ))
            };
            (Some(MaybeQuantized::Original(k)), v)
        } else {
            (None, None)
        };

        let q_norm = nn::RmsNormBuilder::new(head_dim).eps(args.rms_norm_eps).build()?;
        let (k_norm, v_norm) = if has_kv {
            let k = nn::RmsNormBuilder::new(head_dim).eps(args.rms_norm_eps).build()?;
            let v = RmsNormNoScale::new(args.rms_norm_eps);
            (Some(k), Some(v))
        } else {
            (None, None)
        };

        let rope = build_layer_rope(
            head_dim,
            layer_kind,
            args.rope_traditional,
            args.rope_parameters.as_ref(),
        )?;

        Ok(Self {
            layer_idx,
            layer_kind,
            is_sliding,
            has_kv,
            use_k_eq_v,
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            q_proj: MaybeQuantized::Original(q_proj),
            k_proj,
            v_proj,
            o_proj: MaybeQuantized::Original(o_proj),
            q_norm,
            k_norm,
            v_norm,
            rope,
        })
    }
}

/// Output of a per-layer forward: hidden state + (k, v) for downstream
/// KV-shared layers + position offset captured *pre-update*.
pub struct AttentionOut {
    pub h: Array,
    pub shared_kv: (Array, Array),
    pub offset: i32,
}

impl Attention {
    #[allow(non_snake_case, reason = "local bindings mirror ML tensor names (Q, K, V)")]
    pub fn attend<C: KeyValueCache + Default>(
        &mut self,
        input: AttentionInput<'_, C>,
    ) -> Result<AttentionOut, Exception> {
        let AttentionInput { x, mask, mut cache, shared_kv, offset } = input;
        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];

        // Pre-update cache offset is what the RoPE applied to fresh
        // queries needs (mirrors mlx_lm Python: `offset = cache.offset`).
        let pre_offset = match (offset, cache.as_ref()) {
            (Some(o), _) => o,
            (None, Some(c)) => c.offset(),
            (None, None) => 0,
        };
        // Python passes `mx.array(cache.offset)` to `mx.fast.rope` so
        // the offset stays on-device; that triggers the dynamic-RoPE
        // kernel which lets MLX reuse the compiled rope graph across
        // decode steps regardless of offset value. Wrapping the i32
        // in a 0-D Array here matches that.
        let pre_offset_arr = Array::from_int(pre_offset);

        let queries = self
            .q_proj
            .forward(x)?
            .reshape(&[B, L, self.n_heads, self.head_dim])?;
        let mut queries = self.q_norm.forward(&queries)?;

        let (keys, values) = if let Some(kv) = shared_kv { kv } else {
            if !self.has_kv {
                return Err(Exception::custom(format!(
                    "gemma4: layer {} is KV-shared but no shared_kv supplied",
                    self.layer_idx
                )));
            }
            let k_proj = self.k_proj.as_mut().expect("has_kv guarantees k_proj");
            let keys = k_proj
                .forward(x)?
                .reshape(&[B, L, self.n_kv_heads, self.head_dim])?;

            let mut k_for_attn = self
                .k_norm
                .as_mut()
                .expect("has_kv guarantees k_norm")
                .forward(&keys)?
                .transpose_axes(&[0, 2, 1, 3])?;
            k_for_attn = self.rope.forward_dynamic(&k_for_attn, &pre_offset_arr)?;

            let values = if self.use_k_eq_v {
                keys.clone()
            } else {
                self.v_proj
                    .as_mut()
                    .expect("non-keqv has v_proj")
                    .forward(x)?
                    .reshape(&[B, L, self.n_kv_heads, self.head_dim])?
            };
            let v_for_attn = self
                .v_norm
                .as_mut()
                .expect("has_kv guarantees v_norm")
                .forward(&values)?
                .transpose_axes(&[0, 2, 1, 3])?;

            (k_for_attn, v_for_attn)
        };

        queries = queries.transpose_axes(&[0, 2, 1, 3])?;
        queries = self.rope.forward_dynamic(&queries, &pre_offset_arr)?;

        // Concat with cache first, then attend. KV-shared downstream
        // layers reuse `(k_full, v_full)`; returning the pre-update
        // per-step projections would lose history across turns.
        let (k_full, v_full) = if let Some(cache) = cache.as_mut() {
            cache.update_and_fetch(keys, values)?
        } else {
            (keys, values)
        };
        let h = fast::scaled_dot_product_attention(
            queries,
            k_full.clone(),
            v_full.clone(),
            self.scale,
            mask.map(fast::ScaledDotProductAttentionMask::Array),
            None,
        )?;

        let h = h.transpose_axes(&[0, 2, 1, 3])?.reshape(&[B, L, -1])?;
        let h = self.o_proj.forward(&h)?;

        Ok(AttentionOut {
            h,
            shared_kv: (k_full, v_full),
            offset: pre_offset,
        })
    }

    pub fn training_mode_set(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        if let Some(k) = self.k_proj.as_mut() { k.training_mode(mode); }
        if let Some(v) = self.v_proj.as_mut() { v.training_mode(mode); }
        self.o_proj.training_mode(mode);
        self.q_norm.training_mode(mode);
        if let Some(k) = self.k_norm.as_mut() { k.training_mode(mode); }
        if let Some(v) = self.v_norm.as_mut() { v.training_mode(mode); }
    }
}

#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Mlp {
    #[quantizable]
    #[param]
    pub gate_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub down_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub up_proj: MaybeQuantized<nn::Linear>,
    /// Per-layer compiled-graph cache for `gelu_approx(gate) * up`.
    /// Filled on first forward; reused across every decode step.
    geglu_cache: GegluCache,
}

impl Mlp {
    pub fn new(args: &Gemma4Config, layer_idx: i32) -> Result<Self, Exception> {
        let first_kv_shared = args.num_hidden_layers - args.num_kv_shared_layers;
        let is_kv_shared_layer =
            args.num_kv_shared_layers > 0 && layer_idx >= first_kv_shared;
        let use_double_wide = args.use_double_wide_mlp && is_kv_shared_layer;
        let intermediate = if use_double_wide {
            args.intermediate_size * 2
        } else {
            args.intermediate_size
        };

        Ok(Self {
            gate_proj: MaybeQuantized::Original(
                nn::LinearBuilder::new(args.hidden_size, intermediate).bias(false).build()?,
            ),
            down_proj: MaybeQuantized::Original(
                nn::LinearBuilder::new(intermediate, args.hidden_size).bias(false).build()?,
            ),
            up_proj: MaybeQuantized::Original(
                nn::LinearBuilder::new(args.hidden_size, intermediate).bias(false).build()?,
            ),
            geglu_cache: GegluCache::default(),
        })
    }
}

impl Module<&Array> for Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = geglu(&mut self.geglu_cache, &gate, &up)?;
        self.down_proj.forward(&activated)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
    }
}

/// fp16-safe additive residual (Python `clip_residual`).
fn clip_residual(x: &Array, y: &Array) -> Result<Array, Exception> {
    if x.dtype() != Dtype::Float16 {
        return x.add(y);
    }
    let xf = x.as_dtype(Dtype::Float32)?;
    let yf = y.as_dtype(Dtype::Float32)?;
    let sum = xf.add(&yf)?;
    clip(&sum, (-FP16_MAX, FP16_MAX))?.as_dtype(Dtype::Float16)
}

/// Top-k expert router (26B-A4B). RMS-norms the hidden, projects to
/// per-expert scores, picks top-`k` via `argpartition`, softmaxes the
/// top-k, and scales by per-expert biases.
#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Router {
    pub eps: f32,
    pub top_k: i32,
    pub root_size: f32,

    #[quantizable]
    #[param]
    pub proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub scale: Param<Array>,
    #[param]
    pub per_expert_scale: Param<Array>,

    /// `scale * root_size`, materialised lazily on first forward and
    /// reused thereafter. Not a learnable parameter.
    scaled_weight: OnceLock<Array>,

    /// `per_expert_scale` re-cast to the scores' dtype. Loaded weights
    /// may differ from the scores dtype (e.g. f32 inits, bf16 scores);
    /// without this cache the `softmax * gathered` multiply would
    /// promote `weights` to f32 every call and poison downstream
    /// ops (the fused MoE kernel needs bf16/f16 inputs).
    per_expert_scale_cast: OnceLock<Array>,

    /// Compiled `softmax(take(scores)) * take(per_expert_scale)` cache —
    /// the whole router post-processing in one fused launch.
    router_post_cache: RouterPostCache,
}

impl Router {
    pub fn new(hidden_size: i32, num_experts: i32, top_k: i32, eps: f32) -> Result<Self, Exception> {
        Ok(Self {
            eps,
            top_k,
            root_size: (hidden_size as f32).powf(-0.5),
            proj: MaybeQuantized::Original(
                nn::LinearBuilder::new(hidden_size, num_experts).bias(false).build()?,
            ),
            scale: Param::new(Array::ones::<f32>(&[hidden_size])?),
            per_expert_scale: Param::new(Array::ones::<f32>(&[num_experts])?),
            scaled_weight: OnceLock::new(),
            per_expert_scale_cast: OnceLock::new(),
            router_post_cache: RouterPostCache::default(),
        })
    }

    /// `x` shape `[B, L, D]` → `(top_k_indices [B, L, K], top_k_weights [B, L, K])`.
    pub fn forward(&mut self, x: &Array) -> Result<(Array, Array), Exception> {
        // Lazily pre-multiply `scale * root_size` once per layer instead
        // of every forward — the weights are loaded once and never change.
        // Stage the f32 scalar into scale's dtype so the multiply keeps
        // bf16/f16 (otherwise `rms_norm(x_bf16, w_f32)` promotes the
        // entire MoE forward to f32).
        let weight = self.scaled_weight.get_or_init(|| {
            let scale = self.scale.as_ref();
            let root_size_arr = Array::from_f32(self.root_size)
                .as_dtype(scale.dtype())
                .expect("root_size cast cannot fail");
            scale
                .multiply(&root_size_arr)
                .expect("scale × root_size cannot fail")
        });
        let normed = fast::rms_norm(x, Some(weight), self.eps)?;
        let scores = self.proj.forward(&normed)?;

        // argpartition along last axis, then slice last K.
        // argpartition along last axis; gives [..., num_experts] reorder
        // with the top-K largest at positions [num_experts-K..num_experts].
        let kth: i32 = -self.top_k;
        let part = argpartition_axis(&scores, kth, -1)?;
        let part_len = *part.shape().last().expect("scores has trailing dim");
        let start = part_len - self.top_k;
        let top_k_indices = part.index((.., .., start..part_len));

        // Cast `per_expert_scale` once to the scores dtype to keep the
        // post-softmax multiply from promoting weights to f32.
        let scores_dtype = scores.dtype();
        let per_expert_scale_cast = self.per_expert_scale_cast.get_or_init(|| {
            self.per_expert_scale
                .as_ref()
                .as_dtype(scores_dtype)
                .expect("per_expert_scale cast cannot fail")
        });
        let top_k_weights = router_post(
            &mut self.router_post_cache,
            &scores,
            &top_k_indices,
            per_expert_scale_cast,
        )?;

        Ok((top_k_indices, top_k_weights))
    }
}

/// Sparse MoE wrapping `SwitchGLU`. Each token's top-k expert outputs
/// are weighted by the router's softmaxed scores and summed.
#[derive(Debug, ModuleParameters)]
pub struct Experts {
    #[param]
    pub switch_glu: SwitchGLU,
}

impl Experts {
    pub fn new(hidden_size: i32, moe_intermediate: i32, num_experts: i32) -> Result<Self, Exception> {
        Ok(Self {
            switch_glu: SwitchGLU::new(hidden_size, moe_intermediate, num_experts, false)?,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        top_k_indices: &Array,
        top_k_weights: &Array,
    ) -> Result<Array, Exception> {
        // Try the fused down+combine path first (quantised + no-sort
        // only). Falls back internally to the legacy 2-launch path
        // for sort/dense.
        self.switch_glu.forward_with_combine(x, top_k_indices, top_k_weights)
    }
}

impl mlx_rs::quantization::Quantizable for Experts {
    type Quantized = Self;
    type QuantizationError = Exception;

    fn try_into_quantized(
        self,
        group_size: i32,
        bits: i32,
    ) -> Result<Self::Quantized, Self::QuantizationError> {
        Ok(Self {
            switch_glu: self.switch_glu.try_into_quantized(group_size, bits)?,
        })
    }
}

#[derive(Debug, ModuleParameters, Quantizable)]
pub struct DecoderLayer {
    pub layer_idx: i32,
    pub layer_kind: LayerKind,
    pub enable_moe: bool,

    #[quantizable]
    #[param]
    pub self_attn: Attention,
    #[quantizable]
    #[param]
    pub mlp: Mlp,

    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
    #[param]
    pub pre_feedforward_layernorm: nn::RmsNorm,
    #[param]
    pub post_feedforward_layernorm: nn::RmsNorm,

    /// Multiplicative per-layer scalar (`layer_scalar` in Python).
    #[param]
    pub layer_scalar: Param<Array>,

    // MoE branch (26B-A4B). All four fields are `None` when
    // `enable_moe_block=false`.
    #[quantizable]
    #[param]
    pub router: Option<Router>,
    #[quantizable]
    #[param]
    pub experts: Option<Experts>,
    #[param]
    pub post_feedforward_layernorm_1: Option<nn::RmsNorm>,
    #[param]
    pub pre_feedforward_layernorm_2: Option<nn::RmsNorm>,
    #[param]
    pub post_feedforward_layernorm_2: Option<nn::RmsNorm>,

    // Per-layer input gating (E2B/E4B). All three fields are `None`
    // when `hidden_size_per_layer_input == 0`.
    #[quantizable]
    #[param]
    pub per_layer_input_gate: Option<MaybeQuantized<nn::Linear>>,
    #[quantizable]
    #[param]
    pub per_layer_projection: Option<MaybeQuantized<nn::Linear>>,
    #[param]
    pub post_per_layer_input_norm: Option<nn::RmsNorm>,

    /// Compiled `(residual + ff_out) * layer_scalar` cache — fuses
    /// the per-layer epilogue's two launches into one on non-fp16
    /// dtypes (bf16 / fp32). Falls back to the unfused path on fp16.
    residual_scale_cache: ResidualAddScaleCache,
}

impl DecoderLayer {
    pub fn new(args: &Gemma4Config, layer_idx: i32) -> Result<Self, Exception> {
        let layer_kind = args.layer_types_resolved()[layer_idx as usize];
        let enable_moe = args.enable_moe_block;

        let (router, experts, post1, pre2, post2) = if enable_moe {
            let num_experts = args.num_experts.ok_or_else(|| {
                Exception::custom("gemma4: enable_moe_block=true requires num_experts")
            })?;
            let top_k = args.top_k_experts.ok_or_else(|| {
                Exception::custom("gemma4: enable_moe_block=true requires top_k_experts")
            })?;
            let moe_int = args.moe_intermediate_size.ok_or_else(|| {
                Exception::custom("gemma4: enable_moe_block=true requires moe_intermediate_size")
            })?;
            (
                Some(Router::new(args.hidden_size, num_experts, top_k, args.rms_norm_eps)?),
                Some(Experts::new(args.hidden_size, moe_int, num_experts)?),
                Some(nn::RmsNormBuilder::new(args.hidden_size).eps(args.rms_norm_eps).build()?),
                Some(nn::RmsNormBuilder::new(args.hidden_size).eps(args.rms_norm_eps).build()?),
                Some(nn::RmsNormBuilder::new(args.hidden_size).eps(args.rms_norm_eps).build()?),
            )
        } else {
            (None, None, None, None, None)
        };

        let (gate, proj, pl_norm) = if args.hidden_size_per_layer_input > 0 {
            let pl = args.hidden_size_per_layer_input;
            (
                Some(MaybeQuantized::Original(
                    nn::LinearBuilder::new(args.hidden_size, pl).bias(false).build()?,
                )),
                Some(MaybeQuantized::Original(
                    nn::LinearBuilder::new(pl, args.hidden_size).bias(false).build()?,
                )),
                Some(nn::RmsNormBuilder::new(args.hidden_size).eps(args.rms_norm_eps).build()?),
            )
        } else {
            (None, None, None)
        };

        Ok(Self {
            layer_idx,
            layer_kind,
            enable_moe,
            self_attn: Attention::new(args, layer_idx)?,
            mlp: Mlp::new(args, layer_idx)?,
            input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            pre_feedforward_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            post_feedforward_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            layer_scalar: Param::new(Array::ones::<f32>(&[1])?),
            router,
            experts,
            post_feedforward_layernorm_1: post1,
            pre_feedforward_layernorm_2: pre2,
            post_feedforward_layernorm_2: post2,
            per_layer_input_gate: gate,
            per_layer_projection: proj,
            post_per_layer_input_norm: pl_norm,
            residual_scale_cache: ResidualAddScaleCache::default(),
        })
    }

    pub fn forward_layer<C: KeyValueCache + Default>(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut C>,
        shared_kv: Option<(Array, Array)>,
        offset: Option<i32>,
        per_layer_input: Option<&Array>,
    ) -> Result<AttentionOut, Exception> {
        let residual = x.clone();

        let h_pre = self.input_layernorm.forward(x)?;
        let AttentionOut { h, shared_kv: kv_out, offset: off_out } =
            self.self_attn.attend(AttentionInput {
                x: &h_pre,
                mask,
                cache,
                shared_kv,
                offset,
            })?;
        let h = self.post_attention_layernorm.forward(&h)?;
        let h = clip_residual(&residual, &h)?;

        let residual = h.clone();
        let ff_mid = if self.enable_moe {
            // h1 = post1(MLP(pre1(h)))
            let h1 = self.pre_feedforward_layernorm.forward(&h)?;
            let h1 = self.mlp.forward(&h1)?;
            let h1 = self
                .post_feedforward_layernorm_1
                .as_mut()
                .expect("moe layer has post_ff_1")
                .forward(&h1)?;

            // h2 = post2(Experts(pre2(h), router(h)))
            let (top_k_indices, top_k_weights) = self
                .router
                .as_mut()
                .expect("moe layer has router")
                .forward(&h)?;
            let h2 = self
                .pre_feedforward_layernorm_2
                .as_mut()
                .expect("moe layer has pre_ff_2")
                .forward(&h)?;
            let h2 = self
                .experts
                .as_mut()
                .expect("moe layer has experts")
                .forward(&h2, &top_k_indices, &top_k_weights)?;
            let h2 = self
                .post_feedforward_layernorm_2
                .as_mut()
                .expect("moe layer has post_ff_2")
                .forward(&h2)?;

            h1.add(&h2)?
        } else {
            let mid = self.pre_feedforward_layernorm.forward(&h)?;
            self.mlp.forward(&mid)?
        };
        // Both branches share the final post_feedforward_layernorm before
        // the residual add — matches Python `gemma4_text.DecoderLayer`.
        let ff_out = self.post_feedforward_layernorm.forward(&ff_mid)?;

        // Fast path: non-fp16 dtype + no per-layer input gating (i.e.
        // 26B-A4B / 31B). Folds `(residual + ff_out) * layer_scalar`
        // into a single compiled launch. fp16 (Gemma 3 vision) and
        // E2B/E4B still go through the unfused legacy path below.
        let pl_inactive = per_layer_input.is_none()
            || self.per_layer_input_gate.is_none()
            || self.per_layer_projection.is_none()
            || self.post_per_layer_input_norm.is_none();
        if pl_inactive && ff_out.dtype() != Dtype::Float16 {
            let h = residual_add_scale(
                &mut self.residual_scale_cache,
                &residual,
                &ff_out,
                self.layer_scalar.as_ref(),
            )?;
            return Ok(AttentionOut { h, shared_kv: kv_out, offset: off_out });
        }

        let mut h = clip_residual(&residual, &ff_out)?;

        // Per-layer input gating (E2B/E4B).
        if let (Some(gate_l), Some(proj_l), Some(norm_l), Some(pl_in)) = (
            self.per_layer_input_gate.as_mut(),
            self.per_layer_projection.as_mut(),
            self.post_per_layer_input_norm.as_mut(),
            per_layer_input,
        ) {
            let residual_pl = h.clone();
            let g = gate_l.forward(&h)?;
            let g = gelu_approximate(&g)?;
            let g = g.multiply(pl_in)?;
            let g = proj_l.forward(&g)?;
            let g = norm_l.forward(&g)?;
            h = residual_pl.add(&g)?;
        }

        h = h.multiply(self.layer_scalar.as_ref())?;

        Ok(AttentionOut { h, shared_kv: kv_out, offset: off_out })
    }
}

#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Gemma4TextModel {
    pub vocab_size: i32,
    pub num_hidden_layers: i32,
    pub sliding_window_pattern: i32,
    pub embed_scale: f32,
    pub hidden_size_per_layer_input: i32,
    pub embed_tokens_per_layer_scale: f32,
    pub per_layer_input_scale: f32,
    pub per_layer_projection_scale: f32,

    /// Cached 0-d Arrays for the scalar constants used per forward.
    /// Replacing per-call `Array::from_f32` with a single allocation
    /// drops 1+ small alloc kernels per token (more in E2B/E4B).
    embed_scale_arr: OnceLock<Array>,
    embed_tokens_per_layer_scale_arr: OnceLock<Array>,
    per_layer_input_scale_arr: OnceLock<Array>,
    per_layer_projection_scale_arr: OnceLock<Array>,
    /// Per-layer source index for KV (each layer's own index unless
    /// it's a shared-KV layer, in which case the most recent
    /// same-kind layer's index < first_kv_shared).
    pub previous_kvs: Vec<usize>,

    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    pub layers: Vec<DecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,

    // Per-layer input embeddings (E2B/E4B). All four fields are `None`
    // when `hidden_size_per_layer_input == 0`.
    #[quantizable]
    #[param]
    pub embed_tokens_per_layer: Option<MaybeQuantized<nn::Embedding>>,
    // mlx-community Gemma 4 E2B/E4B keeps this projection unquantised;
    // staying `nn::Linear` avoids inventing missing `.scales`/`.biases`.
    #[param]
    pub per_layer_model_projection: Option<nn::Linear>,
    #[param]
    pub per_layer_projection_norm: Option<nn::RmsNorm>,
}

impl Gemma4TextModel {
    pub fn new(args: &Gemma4Config) -> Result<Self, Exception> {
        assert!(args.vocab_size > 0);
        let embed_tokens = nn::Embedding::new(args.vocab_size, args.hidden_size)?;
        let layers = (0..args.num_hidden_layers)
            .map(|i| DecoderLayer::new(args, i))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;

        let previous_kvs = compute_previous_kvs(args);

        let pl = args.hidden_size_per_layer_input;
        let (etpl, plproj, plnorm) = if pl > 0 {
            (
                Some(MaybeQuantized::Original(nn::Embedding::new(
                    args.vocab_size_per_layer_input,
                    args.num_hidden_layers * pl,
                )?)),
                Some(
                    nn::LinearBuilder::new(args.hidden_size, args.num_hidden_layers * pl)
                        .bias(false)
                        .build()?,
                ),
                Some(nn::RmsNormBuilder::new(pl).eps(args.rms_norm_eps).build()?),
            )
        } else {
            (None, None, None)
        };

        Ok(Self {
            vocab_size: args.vocab_size,
            num_hidden_layers: args.num_hidden_layers,
            sliding_window_pattern: args.effective_sliding_window_pattern(),
            embed_scale: (args.hidden_size as f32).sqrt(),
            hidden_size_per_layer_input: pl,
            embed_tokens_per_layer_scale: if pl > 0 { (pl as f32).sqrt() } else { 0.0 },
            per_layer_input_scale: (2.0_f32).powf(-0.5),
            per_layer_projection_scale: (args.hidden_size as f32).powf(-0.5),
            previous_kvs,
            embed_tokens: MaybeQuantized::Original(embed_tokens),
            layers,
            norm,
            embed_tokens_per_layer: etpl,
            per_layer_model_projection: plproj,
            per_layer_projection_norm: plnorm,
            embed_scale_arr: OnceLock::new(),
            embed_tokens_per_layer_scale_arr: OnceLock::new(),
            per_layer_input_scale_arr: OnceLock::new(),
            per_layer_projection_scale_arr: OnceLock::new(),
        })
    }
}

/// Build the `previous_kvs` table: `[i]` is the layer-index whose K/V
/// the layer-`i` attention should consume. For non-shared layers
/// it's `i` itself; for shared layers it's the most recent
/// same-kind layer below `first_kv_shared`.
fn compute_previous_kvs(args: &Gemma4Config) -> Vec<usize> {
    let n = args.num_hidden_layers as usize;
    let mut previous_kvs: Vec<usize> = (0..n).collect();
    if args.num_kv_shared_layers <= 0 {
        return previous_kvs;
    }
    let first_kv_shared = (args.num_hidden_layers - args.num_kv_shared_layers) as usize;
    let layer_types = args.layer_types_resolved();
    let mut kvs_by_kind: HashMap<LayerKind, usize> = HashMap::new();
    for (i, k) in layer_types.iter().enumerate().take(first_kv_shared) {
        kvs_by_kind.insert(*k, i);
    }
    for j in first_kv_shared..n {
        let kind = layer_types[j];
        if let Some(&src) = kvs_by_kind.get(&kind) {
            previous_kvs[j] = src;
        }
    }
    previous_kvs
}

impl<C> Module<ModelInput<'_, C>> for Gemma4TextModel
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let ModelInput { inputs, cache, .. } = input;
        let mut h = self.embed_tokens.forward(inputs)?;
        // Stage scale in `h`'s dtype so the multiply doesn't promote
        // bf16 embeddings to f32 (and poison the entire forward).
        let h_dtype = h.dtype();
        let embed_scale_arr = self.embed_scale_arr.get_or_init(|| {
            Array::from_f32(self.embed_scale)
                .as_dtype(h_dtype)
                .expect("embed_scale cast cannot fail")
        });
        h = h.multiply(embed_scale_arr)?;

        ensure_cache_populated(cache, self.layers.len());

        // Per-layer input embeddings (E2B/E4B). When enabled, build a
        // `[B, L, N, D_pl]` tensor and slice axis-2 per layer below.
        let per_layer_inputs: Option<Vec<Array>> = if self.hidden_size_per_layer_input > 0 {
            let pl = self.hidden_size_per_layer_input;
            let etps = self
                .embed_tokens_per_layer_scale_arr
                .get_or_init(|| Array::from_f32(self.embed_tokens_per_layer_scale));
            let pps = self
                .per_layer_projection_scale_arr
                .get_or_init(|| Array::from_f32(self.per_layer_projection_scale));
            let pis = self
                .per_layer_input_scale_arr
                .get_or_init(|| Array::from_f32(self.per_layer_input_scale));
            let etpl = self
                .embed_tokens_per_layer
                .as_mut()
                .expect("hidden_size_per_layer_input>0 requires embed_tokens_per_layer");
            // embed_tokens_per_layer(input_ids) * sqrt(D_pl)
            let raw = etpl.forward(inputs)?;
            let raw = raw.multiply(etps)?;
            // [B, L, N, D_pl]
            let pli = unflatten(&raw, -1, &[self.num_hidden_layers, pl])?;

            // projection = per_layer_model_projection(h) * D^-0.5
            let proj = self
                .per_layer_model_projection
                .as_mut()
                .expect("hidden_size_per_layer_input>0 requires per_layer_model_projection");
            let pproj = proj.forward(&h)?;
            let pproj = pproj.multiply(pps)?;
            let pproj = unflatten(&pproj, -1, &[self.num_hidden_layers, pl])?;
            let pproj = self
                .per_layer_projection_norm
                .as_mut()
                .expect("hidden_size_per_layer_input>0 requires per_layer_projection_norm")
                .forward(&pproj)?;

            // (projection + per_layer_inputs) * 2^-0.5
            let combined = pproj.add(&pli)?;
            let combined = combined.multiply(pis)?;

            // Slice axis 2 per layer: [B, L, D_pl] each.
            let n = self.num_hidden_layers as usize;
            let mut out = Vec::with_capacity(n);
            for i in 0..n as i32 {
                out.push(combined.index((.., .., i, ..)));
            }
            Some(out)
        } else {
            None
        };

        // Build mask per layer-kind (full and sliding share within a
        // forward pass).
        let pattern = self.sliding_window_pattern as usize;
        let global_idx = pattern.saturating_sub(1).min(cache.len() - 1);
        let global_mask = create_attention_mask(&h, &cache[global_idx..=global_idx])?;
        let sliding_mask = if pattern > 1 {
            create_attention_mask(&h, &cache[0..1])?
        } else {
            None
        };
        // Expand 2D `[T, kT]` masks to 4D `[1, 1, T, kT]` so they broadcast
        // against `[B, H, T, D]` activations in the non-fused SDPA path
        // (head_dim ∉ {64, 80, 128} for Gemma 4 falls back to this path).
        let expand_mask = |a: Array| -> Result<Array, Exception> {
            if a.shape().len() == 2 {
                expand_dims_axes(&a, &[0, 1])
            } else {
                Ok(a)
            }
        };
        let global_arr = global_mask.map(expand_mask).transpose()?;
        let sliding_arr = sliding_mask.map(expand_mask).transpose()?;

        // Intermediate KV per layer index for shared lookup.
        let n = self.layers.len();
        let mut intermediates: Vec<Option<(Array, Array, i32)>> = (0..n).map(|_| None).collect();

        // Split borrow: previous_kvs is immutable, layers is mutated.
        let layers = &mut self.layers;
        let previous_kvs = self.previous_kvs.as_slice();

        for i in 0..n {
            let kind = layers[i].layer_kind;
            let mask = match kind {
                LayerKind::FullAttention => global_arr.as_ref(),
                LayerKind::SlidingAttention => sliding_arr.as_ref(),
            };

            let (shared_kv, offset_in) = if previous_kvs[i] != i {
                let src = previous_kvs[i];
                match &intermediates[src] {
                    Some((k, v, off)) => (Some((k.clone(), v.clone())), Some(*off)),
                    None => (None, None),
                }
            } else {
                (None, None)
            };

            let cache_slot = cache.get_mut(i).and_then(|c| c.as_mut());
            let pli_slice = per_layer_inputs.as_ref().map(|v| &v[i]);
            let out = layers[i].forward_layer::<C>(
                &h, mask, cache_slot, shared_kv, offset_in, pli_slice,
            )?;
            h = out.h;
            intermediates[i] = Some((out.shared_kv.0, out.shared_kv.1, out.offset));
        }

        self.norm.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            layer.self_attn.training_mode_set(mode);
            layer.mlp.training_mode(mode);
            layer.input_layernorm.training_mode(mode);
            layer.post_attention_layernorm.training_mode(mode);
            layer.pre_feedforward_layernorm.training_mode(mode);
            layer.post_feedforward_layernorm.training_mode(mode);
        }
        self.norm.training_mode(mode);
    }
}

#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Model {
    pub args: Gemma4Config,
    pub final_logit_softcapping: Option<f32>,

    #[quantizable]
    #[param]
    pub model: Gemma4TextModel,

    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,

    /// `tanh(x / cap) * cap` compiled once and reused.  Pair with
    /// `softcap_array` so `cap` is a stable 0-d input across calls.
    softcap_cache: LogitSoftcapCache,
    /// Cached 0-d Array holding `final_logit_softcapping` — allocated
    /// once instead of `Array::from_f32` per forward.
    softcap_array: OnceLock<Array>,
}

impl Model {
    pub fn new(args: Gemma4Config) -> Result<Self, Exception> {
        let final_logit_softcapping = if args.final_logit_softcapping > 0.0 {
            Some(args.final_logit_softcapping)
        } else {
            None
        };
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(MaybeQuantized::Original(
                nn::LinearBuilder::new(args.hidden_size, args.vocab_size)
                    .bias(false)
                    .build()?,
            ))
        };
        let model = Gemma4TextModel::new(&args)?;
        Ok(Self {
            args,
            final_logit_softcapping,
            model,
            lm_head,
            softcap_cache: LogitSoftcapCache::default(),
            softcap_array: OnceLock::new(),
        })
    }

    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }

    pub fn layer_count(&self) -> usize {
        self.args.num_hidden_layers as usize
    }

    pub fn head_dim(&self) -> i32 {
        self.args.head_dim
    }
}

impl<C> Module<ModelInput<'_, C>> for Model
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let out = self.model.forward(input)?;
        let mut logits = if let Some(lm) = self.lm_head.as_mut() {
            lm.forward(&out)?
        } else {
            match &self.model.embed_tokens {
                MaybeQuantized::Original(e) => e.as_linear(&out)?,
                MaybeQuantized::Quantized(qe) => qe.as_linear(&out)?,
            }
        };
        if let Some(cap) = self.final_logit_softcapping {
            // Stage cap in logits' dtype so `divide(logits, cap)` stays
            // in bf16/f16 instead of promoting to f32.
            let logits_dtype = logits.dtype();
            let cap_arr = self.softcap_array.get_or_init(|| {
                Array::from_f32(cap)
                    .as_dtype(logits_dtype)
                    .expect("cap cast cannot fail")
            });
            logits = logit_softcap(&mut self.softcap_cache, &logits, cap_arr)?;
        }
        Ok(logits)
    }

    fn training_mode(&mut self, mode: bool) {
        <Gemma4TextModel as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
        if let Some(lm) = self.lm_head.as_mut() {
            lm.training_mode(mode);
        }
    }
}
