//! Decoder layer + top-level [`Qwen35Decoder`] / [`Qwen35Model`] wrappers.
//!
//! [`DecoderLayer`] is the per-layer dispatch: it owns either a
//! `linear_attn` Gated DeltaNet block or a `self_attn` full-attention
//! block based on the checkpoint's `layer_types`. Both code paths
//! share the same input / post-attention norms and the SwiGLU MLP.
//!
//! [`Qwen35Decoder`] holds the embeddings, the layer stack, and the
//! final `model.norm`. [`Qwen35Model`] wraps it with the LM head
//! (tied or untied) plus the optional MTP head and runs end-to-end
//! logits.

use mlxr::{
    builder::Builder,
    error::Exception,
    layers,
    macros::{ModuleParameters, Quantizable},
    module::Module,
    ops::{arange, broadcast_to, concatenate_axis, expand_dims},
    quantization::{MaybeQuantized, Quantizable as QuantizableTrait},
    Array,
};

use super::cache::LayerCache;
use super::config::TextConfig;
use super::gated_delta_block::GatedDeltaNet;
use super::text::{Attention, Mlp};

/// One Qwen3.5 decoder layer: either linear-attention (GDN) or full-attention.
/// Generic over the FFN `F`: defaults to dense [`Mlp`] for Qwen3.5/3.6
/// dense; the MoE variants alias `DecoderLayer<Qwen35MoeBlock>`.
///
/// `self_attn` / `linear_attn` are kept in `Option` fields rather than an
/// enum so the derived `ModuleParameters` / `Quantizable` walks both paths
/// in the weight loader. Exactly one is populated per layer.
#[derive(Debug, ModuleParameters, Quantizable)]
pub struct DecoderLayer<F = Mlp>
where
    F: for<'a> Module<&'a Array, Output = Array, Error = Exception>
        + QuantizableTrait<Quantized = F, QuantizationError = Exception>
        + std::fmt::Debug,
{
    pub is_linear: bool,

    #[quantizable]
    #[param]
    pub self_attn: Option<Attention>,

    #[quantizable]
    #[param]
    pub linear_attn: Option<GatedDeltaNet>,

    #[param]
    pub input_layernorm: layers::RmsNorm,

    #[param]
    pub post_attention_layernorm: layers::RmsNorm,

    #[quantizable]
    #[param]
    pub mlp: F,
}

impl<F> DecoderLayer<F>
where
    F: for<'a> Module<&'a Array, Output = Array, Error = Exception>
        + QuantizableTrait<Quantized = F, QuantizationError = Exception>
        + std::fmt::Debug,
{
    /// Build a layer of the right kind for the given index. `make_ffn`
    /// constructs the per-layer FFN (typically `Mlp::new` for dense or
    /// `Qwen35MoeBlock::new` for the MoE variant).
    pub fn new<MakeF>(
        cfg: &TextConfig,
        layer_idx: usize,
        make_ffn: MakeF,
    ) -> Result<Self, Exception>
    where
        MakeF: FnOnce(&TextConfig) -> Result<F, Exception>,
    {
        let is_linear = layer_is_linear(cfg, layer_idx);
        let (self_attn, linear_attn) = if is_linear {
            (None, Some(GatedDeltaNet::new(cfg)?))
        } else {
            (Some(Attention::new(cfg)?), None)
        };
        let input_layernorm = layers::RmsNormBuilder::new(cfg.hidden_size)
            .eps(cfg.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = layers::RmsNormBuilder::new(cfg.hidden_size)
            .eps(cfg.rms_norm_eps)
            .build()?;
        let mlp = make_ffn(cfg)?;
        Ok(Self {
            is_linear,
            self_attn,
            linear_attn,
            input_layernorm,
            post_attention_layernorm,
            mlp,
        })
    }

    /// Run the layer forward.
    pub fn forward(
        &mut self,
        x: &Array,
        full_attn_mask: Option<&Array>,
        ssm_mask: Option<&Array>,
        cache: Option<&mut LayerCache>,
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = if self.is_linear {
            let blk = self
                .linear_attn
                .as_mut()
                .expect("linear_attn missing for linear layer");
            let cache = cache.map(|c| c.as_linear_attention_mut());
            blk.forward(&normed, ssm_mask, cache)?
        } else {
            let blk = self
                .self_attn
                .as_mut()
                .expect("self_attn missing for full-attn layer");
            let cache = cache.map(|c| c.as_full_attention_mut());
            blk.forward(&normed, full_attn_mask, cache, position_ids)?
        };
        let h = x.add(&attn_out)?;
        let mlp_out = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&h)?)?;
        h.add(&mlp_out)
    }

    /// Toggle training mode on every parameter (attention/GDN, norms, FFN).
    pub fn training_mode(&mut self, mode: bool) {
        if let Some(blk) = self.self_attn.as_mut() {
            blk.training_mode(mode);
        }
        if let Some(blk) = self.linear_attn.as_mut() {
            blk.training_mode(mode);
        }
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
    }

    /// Forward the steel-prefill toggle to the full-attention block, if
    /// this layer has one. Linear-attention layers ignore the call.
    pub fn set_use_steel_prefill(&mut self, on: bool) {
        if let Some(blk) = self.self_attn.as_mut() {
            blk.set_use_steel_prefill(on);
        }
    }
}

/// Source-compat alias for the dense (Mlp-FFN) layer.
pub type DenseDecoderLayer = DecoderLayer<Mlp>;

impl DecoderLayer<Mlp> {
    /// Convenience constructor for the dense (Mlp-FFN) layer; mirrors
    /// the pre-generic API.
    pub fn new_dense(cfg: &TextConfig, layer_idx: usize) -> Result<Self, Exception> {
        Self::new(cfg, layer_idx, |c| {
            Mlp::new(c.hidden_size, c.intermediate_size)
        })
    }
}

impl<F> DecoderLayer<F>
where
    F: for<'a> Module<&'a Array, Output = Array, Error = Exception>
        + QuantizableTrait<Quantized = F, QuantizationError = Exception>
        + std::fmt::Debug,
{
    /// Force a `self_attn` layer regardless of the `layer_types` pattern.
    /// MTP heads in Qwen 3.6 always use standard QKV attention even when
    /// the corresponding main-decoder position would have been a linear
    /// (Mamba-style) layer.
    pub fn new_self_attn<MakeF>(cfg: &TextConfig, make_ffn: MakeF) -> Result<Self, Exception>
    where
        MakeF: FnOnce(&TextConfig) -> Result<F, Exception>,
    {
        let input_layernorm = layers::RmsNormBuilder::new(cfg.hidden_size)
            .eps(cfg.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = layers::RmsNormBuilder::new(cfg.hidden_size)
            .eps(cfg.rms_norm_eps)
            .build()?;
        let mlp = make_ffn(cfg)?;
        Ok(Self {
            is_linear: false,
            self_attn: Some(Attention::new(cfg)?),
            linear_attn: None,
            input_layernorm,
            post_attention_layernorm,
            mlp,
        })
    }
}

/// Qwen 3.6 MTP head (Multi-Token Prediction). Takes the prior decoder's
/// last hidden state `h_t` plus the embedding of `token[t+1]`, normalises
/// them independently, concatenates, projects 2H→H, runs through one
/// self-attention decoder layer, and outputs logits for `token[t+2]` via
/// the shared `lm_head`.
///
/// Weight layout (verified against `Qwen/Qwen3.6-35B-A3B`):
///   `mtp.pre_fc_norm_hidden.weight`        [H]
///   `mtp.pre_fc_norm_embedding.weight`     [H]
///   `mtp.fc.weight`                        [H, 2H]
///   `mtp.layers.0.*`                       full self-attn DecoderLayer
///   `mtp.norm.weight`                      [H]
#[derive(Debug, ModuleParameters, Quantizable)]
pub struct MtpHead<F = Mlp>
where
    F: for<'a> Module<&'a Array, Output = Array, Error = Exception>
        + QuantizableTrait<Quantized = F, QuantizationError = Exception>
        + std::fmt::Debug,
{
    #[param]
    pub pre_fc_norm_hidden: layers::RmsNorm,
    #[param]
    pub pre_fc_norm_embedding: layers::RmsNorm,
    #[quantizable]
    #[param]
    pub fc: MaybeQuantized<layers::Linear>,
    #[quantizable]
    #[param]
    pub layers: Vec<DecoderLayer<F>>,
    #[param]
    pub norm: layers::RmsNorm,
}

impl<F> MtpHead<F>
where
    F: for<'a> Module<&'a Array, Output = Array, Error = Exception>
        + QuantizableTrait<Quantized = F, QuantizationError = Exception>
        + std::fmt::Debug,
{
    pub fn new<MakeF>(cfg: &TextConfig, mut make_ffn: MakeF) -> Result<Self, Exception>
    where
        MakeF: FnMut(&TextConfig) -> Result<F, Exception>,
    {
        let h = cfg.hidden_size;
        let pre_fc_norm_hidden = layers::RmsNormBuilder::new(h)
            .eps(cfg.rms_norm_eps)
            .build()?;
        let pre_fc_norm_embedding = layers::RmsNormBuilder::new(h)
            .eps(cfg.rms_norm_eps)
            .build()?;
        let fc = layers::LinearBuilder::new(2 * h, h).bias(false).build()?;
        let n = cfg.mtp_num_hidden_layers.max(0) as usize;
        let mut layers = Vec::with_capacity(n);
        for _ in 0..n {
            layers.push(DecoderLayer::<F>::new_self_attn(cfg, &mut make_ffn)?);
        }
        let norm = layers::RmsNormBuilder::new(h)
            .eps(cfg.rms_norm_eps)
            .build()?;
        Ok(Self {
            pre_fc_norm_hidden,
            pre_fc_norm_embedding,
            fc: MaybeQuantized::Original(fc),
            layers,
            norm,
        })
    }

    /// Run the MTP head. `h_t` is the main decoder's *pre*-final-norm
    /// hidden at the last committed slot; `embed_next` is the
    /// embedding of the sampled `token[t+1]`. The head re-norms both
    /// inputs via its own `pre_fc_norm_*` weights, so passing the
    /// post-norm hidden would double-normalise. Returns the post-norm
    /// hidden ready for the shared `lm_head`.
    pub fn forward(
        &mut self,
        h_t: &Array,
        embed_next: &Array,
        caches: &mut [LayerCache],
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        let h_n = self.pre_fc_norm_hidden.forward(h_t)?;
        let e_n = self.pre_fc_norm_embedding.forward(embed_next)?;
        let combined = concatenate_axis(&[e_n, h_n], -1)?;
        let mut h = self.fc.forward(&combined)?;
        for (layer, cache) in self.layers.iter_mut().zip(caches.iter_mut()) {
            h = layer.forward(&h, None, None, Some(cache), position_ids)?;
        }
        self.norm.forward(&h)
    }

    pub fn training_mode(&mut self, mode: bool) {
        self.pre_fc_norm_hidden.training_mode(mode);
        self.pre_fc_norm_embedding.training_mode(mode);
        self.fc.training_mode(mode);
        for layer in &mut self.layers {
            layer.training_mode(mode);
        }
        self.norm.training_mode(mode);
    }

    pub fn set_use_steel_prefill(&mut self, on: bool) {
        for layer in &mut self.layers {
            layer.set_use_steel_prefill(on);
        }
    }
}

fn layer_is_linear(cfg: &TextConfig, layer_idx: usize) -> bool {
    if !cfg.layer_types.is_empty() {
        return cfg
            .layer_types
            .get(layer_idx)
            .map(|s| s.as_str() == super::config::LAYER_TYPE_LINEAR)
            .unwrap_or(false);
    }
    let interval = cfg.full_attention_interval;
    if interval <= 0 {
        return false;
    }
    ((layer_idx as i32 + 1) % interval) != 0
}

/// Top-level decoder model: embeddings + layers + final norm. Generic
/// over the FFN type (defaults to dense [`Mlp`]).
#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Qwen35Decoder<F = Mlp>
where
    F: for<'a> Module<&'a Array, Output = Array, Error = Exception>
        + QuantizableTrait<Quantized = F, QuantizationError = Exception>
        + std::fmt::Debug,
{
    pub vocab_size: i32,
    pub num_hidden_layers: i32,

    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<layers::Embedding>,

    #[quantizable]
    #[param]
    pub layers: Vec<DecoderLayer<F>>,

    #[param]
    pub norm: layers::RmsNorm,
}

impl<F> Qwen35Decoder<F>
where
    F: for<'a> Module<&'a Array, Output = Array, Error = Exception>
        + QuantizableTrait<Quantized = F, QuantizationError = Exception>
        + std::fmt::Debug,
{
    /// Build a freshly-initialised decoder. `make_ffn` is called once per
    /// layer to construct the FFN block (e.g. dense `Mlp::new` or MoE
    /// `Qwen35MoeBlock::new`).
    pub fn new<MakeF>(cfg: &TextConfig, mut make_ffn: MakeF) -> Result<Self, Exception>
    where
        MakeF: FnMut(&TextConfig) -> Result<F, Exception>,
    {
        let embed_tokens = layers::Embedding::new(cfg.vocab_size, cfg.hidden_size)?;
        let layers = (0..cfg.num_hidden_layers as usize)
            .map(|i| DecoderLayer::<F>::new(cfg, i, &mut make_ffn))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = layers::RmsNormBuilder::new(cfg.hidden_size)
            .eps(cfg.rms_norm_eps)
            .build()?;
        Ok(Self {
            vocab_size: cfg.vocab_size,
            num_hidden_layers: cfg.num_hidden_layers,
            embed_tokens: MaybeQuantized::Original(embed_tokens),
            layers,
            norm,
        })
    }

    /// Run the decoder over a batch of token ids.
    ///
    /// - `inputs`: `[B, S]` int32 token ids.
    /// - `inputs_embeds`: optional `[B, S, hidden]` to bypass `embed_tokens`
    ///   (used by the multimodal path which has already stitched image
    ///   features into the embedding sequence).
    /// - `caches`: one [`LayerCache`] per layer. Mutated in place.
    /// - `position_ids`: `[3, B, S]` mrope position ids; required when
    ///   pixel inputs were stitched in.
    pub fn forward(
        &mut self,
        inputs: Option<&Array>,
        inputs_embeds: Option<&Array>,
        caches: &mut [LayerCache],
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        let mut h = if let Some(e) = inputs_embeds {
            e.clone()
        } else {
            let ids = inputs.ok_or_else(|| {
                Exception::custom("Qwen35Decoder::forward: needs either inputs or inputs_embeds")
            })?;
            self.embed_tokens.forward(ids)?
        };

        if caches.len() != self.layers.len() {
            return Err(Exception::custom(format!(
                "Qwen35Decoder::forward: expected {} caches, got {}",
                self.layers.len(),
                caches.len()
            )));
        }

        // Hybrid mask scheme:
        //
        // - Full-attention layers use a causal mask of shape `[1, 1, T, kv]`
        //   built from the matching layer's KV-cache offset. When `T == 1`
        //   (decode) the mask is dropped — every full-attn block already
        //   relies on `fast::scaled_dot_product_attention` to handle the
        //   trivial 1-row case.
        // - Linear-attention layers don't need a mask in single-batch /
        //   non-chunked prefill, so we always pass `None` (the Python
        //   reference's `create_ssm_mask` returns `None` here too).
        let full_attn_mask = Self::build_full_attn_mask(&h, caches)?;
        let full_attn_mask_ref = full_attn_mask.as_ref();
        let ssm_mask_ref: Option<&Array> = None;

        for (layer, cache) in self.layers.iter_mut().zip(caches.iter_mut()) {
            h = layer.forward(
                &h,
                full_attn_mask_ref,
                ssm_mask_ref,
                Some(cache),
                position_ids,
            )?;
        }
        self.norm.forward(&h)
    }

    /// Like [`Self::forward`] but returns the **pre-final-norm** hidden
    /// state alongside the post-norm one. The MTP head consumes the
    /// pre-norm hidden (it applies its own pre_fc_norm_hidden); the
    /// lm_head projection uses the post-norm.
    pub fn forward_pre_and_post_norm(
        &mut self,
        inputs: Option<&Array>,
        inputs_embeds: Option<&Array>,
        caches: &mut [LayerCache],
        position_ids: Option<&Array>,
    ) -> Result<(Array, Array), Exception> {
        let mut h = if let Some(e) = inputs_embeds {
            e.clone()
        } else {
            let ids = inputs.ok_or_else(|| {
                Exception::custom("Qwen35Decoder::forward: needs either inputs or inputs_embeds")
            })?;
            self.embed_tokens.forward(ids)?
        };
        if caches.len() != self.layers.len() {
            return Err(Exception::custom(format!(
                "Qwen35Decoder::forward: expected {} caches, got {}",
                self.layers.len(),
                caches.len()
            )));
        }
        let full_attn_mask = Self::build_full_attn_mask(&h, caches)?;
        let full_attn_mask_ref = full_attn_mask.as_ref();
        let ssm_mask_ref: Option<&Array> = None;
        for (layer, cache) in self.layers.iter_mut().zip(caches.iter_mut()) {
            h = layer.forward(
                &h,
                full_attn_mask_ref,
                ssm_mask_ref,
                Some(cache),
                position_ids,
            )?;
        }
        let post = self.norm.forward(&h)?;
        Ok((h, post))
    }

    /// Build the additive causal mask the full-attention layers need.
    ///
    /// Hybrid-cache variant of [`crate::utils::create_attention_mask`]:
    /// pulls the offset from the first full-attention slot (its KV cache
    /// tracks the prefill-vs-decode boundary) and emits a `[T, T + offset]`
    /// boolean matrix lifted to a 4-D broadcast-friendly shape.
    ///
    /// Returns `None` for decode steps (`T == 1`) where the implicit causal
    /// handling inside `fast::scaled_dot_product_attention` already covers
    /// the trivial mask shape.
    fn build_full_attn_mask(h: &Array, caches: &[LayerCache]) -> Result<Option<Array>, Exception> {
        let shape = h.shape();
        let t = shape[1];
        if t <= 1 {
            return Ok(None);
        }

        // Find the first full-attention cache and read its offset.
        let offset = caches
            .iter()
            .find_map(|c| match c {
                LayerCache::FullAttention(kv) => {
                    use crate::cache::KeyValueCache;
                    Some(kv.offset())
                }
                LayerCache::LinearAttention(_) => None,
            })
            .unwrap_or(0);

        let total = offset + t;
        let rinds = arange::<_, i32>(0, total, None)?;
        let linds = arange::<_, i32>(offset, total, None)?;
        // [T, 1] >= [1, total]  -> [T, total]
        let linds_b = expand_dims(&linds, 1)?;
        let rinds_b = expand_dims(&rinds, 0)?;
        let mask = linds_b.ge(&rinds_b)?;
        // Lift to [1, 1, T, total] so it broadcasts against [B, H, T, total].
        let mask = expand_dims(&expand_dims(&mask, 0)?, 0)?;
        Ok(Some(broadcast_to(&mask, &[1, 1, t, total])?))
    }

    /// Toggle training mode on every quantisable parameter.
    pub fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            layer.training_mode(mode);
        }
        self.norm.training_mode(mode);
    }

    /// Propagate the steel-prefill toggle to every full-attention layer.
    pub fn set_use_steel_prefill(&mut self, on: bool) {
        for layer in &mut self.layers {
            layer.set_use_steel_prefill(on);
        }
    }
}

/// Source-compat alias for the dense (Mlp-FFN) decoder.
pub type DenseQwen35Decoder = Qwen35Decoder<Mlp>;

impl Qwen35Decoder<Mlp> {
    /// Dense convenience constructor.
    pub fn new_dense(cfg: &TextConfig) -> Result<Self, Exception> {
        Self::new(cfg, |c| Mlp::new(c.hidden_size, c.intermediate_size))
    }
}

/// Qwen3.5 model: decoder stack + final norm + LM head + optional MTP
/// head. Generic over the FFN type (defaults to dense [`Mlp`]; the
/// MoE variant aliases this as [`super::moe::Qwen35MoeModel`]).
///
/// Named `Qwen35Model` (not `LanguageModel`) so the crate-root trait
/// [`crate::LanguageModel`] keeps its single meaning. This struct is
/// the concrete model body the qwen3_5 adapters wrap to implement
/// that trait.
#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Qwen35Model<F = Mlp>
where
    F: for<'a> Module<&'a Array, Output = Array, Error = Exception>
        + QuantizableTrait<Quantized = F, QuantizationError = Exception>
        + std::fmt::Debug,
{
    pub cfg: TextConfig,

    #[quantizable]
    #[param]
    pub model: Qwen35Decoder<F>,

    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<layers::Linear>>,

    /// MTP head (Qwen 3.6 self-speculative decode). Present iff the
    /// checkpoint ships `mtp.*` weights AND `cfg.mtp_num_hidden_layers > 0`.
    #[quantizable]
    #[param]
    pub mtp: Option<MtpHead<F>>,
}

impl<F> Qwen35Model<F>
where
    F: for<'a> Module<&'a Array, Output = Array, Error = Exception>
        + QuantizableTrait<Quantized = F, QuantizationError = Exception>
        + std::fmt::Debug,
{
    /// Build a freshly-initialised language model. `make_ffn` is called
    /// once per decoder layer (and once more per MTP layer when the
    /// config enables an MTP head) to construct the FFN block.
    pub fn new<MakeF>(cfg: TextConfig, mut make_ffn: MakeF) -> Result<Self, Exception>
    where
        MakeF: FnMut(&TextConfig) -> Result<F, Exception>,
    {
        let model = Qwen35Decoder::<F>::new(&cfg, &mut make_ffn)?;
        let lm_head = if !cfg.tie_word_embeddings {
            Some(MaybeQuantized::Original(
                layers::LinearBuilder::new(cfg.hidden_size, cfg.vocab_size)
                    .bias(false)
                    .build()?,
            ))
        } else {
            None
        };
        let mtp = if cfg.mtp_num_hidden_layers > 0 {
            Some(MtpHead::<F>::new(&cfg, &mut make_ffn)?)
        } else {
            None
        };
        Ok(Self {
            cfg,
            model,
            lm_head,
            mtp,
        })
    }

    /// Run the model end-to-end. Returns `[B, S, vocab_size]` logits.
    pub fn forward(
        &mut self,
        inputs: Option<&Array>,
        inputs_embeds: Option<&Array>,
        caches: &mut [LayerCache],
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        let (_, logits) =
            self.forward_hidden_and_logits(inputs, inputs_embeds, caches, position_ids)?;
        Ok(logits)
    }

    /// Like [`Self::forward`] but also returns the **pre-final-norm**
    /// hidden state. The MTP head consumes this pre-norm hidden (it
    /// applies its own `pre_fc_norm_hidden` on top); the lm_head
    /// projection uses the post-norm (returned alongside as logits).
    pub fn forward_hidden_and_logits(
        &mut self,
        inputs: Option<&Array>,
        inputs_embeds: Option<&Array>,
        caches: &mut [LayerCache],
        position_ids: Option<&Array>,
    ) -> Result<(Array, Array), Exception> {
        let (pre_norm, post_norm) =
            self.model
                .forward_pre_and_post_norm(inputs, inputs_embeds, caches, position_ids)?;
        let logits = self.apply_lm_head(&post_norm)?;
        Ok((pre_norm, logits))
    }

    /// Project a hidden state to vocab logits via the LM head (tied
    /// embed lookup or untied `lm_head` linear, whichever the cfg
    /// selected).
    pub fn apply_lm_head(&mut self, hidden: &Array) -> Result<Array, Exception> {
        if let Some(head) = self.lm_head.as_mut() {
            head.forward(hidden)
        } else {
            match &mut self.model.embed_tokens {
                MaybeQuantized::Original(e) => e.as_linear(hidden),
                MaybeQuantized::Quantized(q) => q.as_linear(hidden),
            }
        }
    }

    /// Embed token ids via the (possibly quantised) `embed_tokens` table.
    /// Needed by the MTP head — it consumes `embed(token_t+1)` as one of
    /// its two inputs.
    pub fn embed_tokens(&mut self, ids: &Array) -> Result<Array, Exception> {
        match &mut self.model.embed_tokens {
            MaybeQuantized::Original(e) => e.forward(ids),
            MaybeQuantized::Quantized(q) => q.forward(ids),
        }
    }

    /// Toggle training mode on every quantisable parameter.
    pub fn training_mode(&mut self, mode: bool) {
        self.model.training_mode(mode);
        if let Some(head) = self.lm_head.as_mut() {
            head.training_mode(mode);
        }
    }

    /// Propagate the steel-prefill toggle to every full-attention layer.
    pub fn set_use_steel_prefill(&mut self, on: bool) {
        self.model.set_use_steel_prefill(on);
    }
}

impl Qwen35Model<Mlp> {
    /// Dense convenience constructor; mirrors the pre-generic API.
    pub fn new_dense(cfg: TextConfig) -> Result<Self, Exception> {
        Self::new(cfg, |c| Mlp::new(c.hidden_size, c.intermediate_size))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    #![allow(clippy::print_stdout, reason = "test code")]
    #![allow(clippy::print_stderr, reason = "test code")]
    use super::*;
    use crate::qwen3_5::text::cache::make_caches;
    use mlxr::{random::uniform, transforms::eval, Array};

    fn synthetic_config(layer_types: Vec<&str>) -> super::super::config::ModelConfig {
        let layers: Vec<String> = layer_types.into_iter().map(String::from).collect();
        let n = layers.len() as i32;
        let json = serde_json::json!({
            "model_type": "qwen3_5",
            "tie_word_embeddings": true,
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": n,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "rms_norm_eps": 1e-6,
                "vocab_size": 100,
                "max_position_embeddings": 256,
                "layer_types": layers,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 4,
                "linear_conv_kernel_dim": 4,
                "tie_word_embeddings": true,
                "rope_parameters": {
                    "mrope_section": [2, 1, 1],
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 1.0,
                    "type": "default"
                }
            },
            "vision_config": {
                "depth": 2,
                "hidden_size": 16,
                "intermediate_size": 32,
                "out_hidden_size": 32,
                "num_heads": 2,
                "patch_size": 16,
                "in_channels": 3,
                "spatial_merge_size": 2
            }
        });
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn hybrid_model_end_to_end_shape() {
        let cfg = synthetic_config(vec![
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]);
        let mut lm = Qwen35Model::new_dense(cfg.text_config.clone()).unwrap();
        let mut caches = make_caches(&cfg);

        // Token ids in [0, vocab_size).
        let ids: Vec<i32> = (0..5).collect();
        let inputs = Array::from_slice(&ids, &[1, 5]);
        let logits = lm.forward(Some(&inputs), None, &mut caches, None).unwrap();
        eval([&logits]).unwrap();
        assert_eq!(logits.shape(), &[1, 5, cfg.text_config.vocab_size]);

        // Decode one more token.
        let next = Array::from_slice(&[42_i32], &[1, 1]);
        let logits2 = lm.forward(Some(&next), None, &mut caches, None).unwrap();
        eval([&logits2]).unwrap();
        assert_eq!(logits2.shape(), &[1, 1, cfg.text_config.vocab_size]);

        // The full-attn layer's KV cache should be at offset 6 now.
        let LayerCache::FullAttention(fa) = &caches[3] else {
            panic!("expected FullAttention cache at index 3");
        };
        use crate::cache::KeyValueCache;
        assert_eq!(fa.offset(), 6);

        // The linear-attn layers should each be at offset 6 too.
        for (i, cache) in caches.iter().enumerate().take(3) {
            let LayerCache::LinearAttention(la) = cache else {
                panic!("expected LinearAttention cache at index {i}");
            };
            assert_eq!(la.offset, 6, "linear layer {i} offset");
        }
    }

    #[test]
    fn forward_accepts_inputs_embeds() {
        let cfg = synthetic_config(vec!["linear_attention", "full_attention"]);
        let mut lm = Qwen35Model::new_dense(cfg.text_config.clone()).unwrap();
        let mut caches = make_caches(&cfg);
        let embeds =
            uniform::<_, f32>(0.0, 1.0, &[1, 3, cfg.text_config.hidden_size], None).unwrap();
        let logits = lm.forward(None, Some(&embeds), &mut caches, None).unwrap();
        eval([&logits]).unwrap();
        assert_eq!(logits.shape(), &[1, 3, cfg.text_config.vocab_size]);
    }
}
