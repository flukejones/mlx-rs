//! Decoder layer + top-level `Qwen3_5Model` wrapper.
//!
//! `DecoderLayer` is the per-layer dispatch: it owns either a `linear_attn`
//! Gated DeltaNet block or a `self_attn` full-attention block based on the
//! checkpoint's `layer_types`. Both code paths share the same input/post
//! norms and SwiGLU MLP.
//!
//! `Qwen3_5Model` holds the embeddings, the layer stack, and the final
//! `model.norm`. `LanguageModel` adds the LM head (tied or untied) and runs
//! end-to-end logits.

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn,
    ops::{arange, broadcast_to, expand_dims},
    quantization::MaybeQuantized,
    Array,
};

use super::cache::LayerCache;
use super::config::TextConfig;
use super::gated_delta_block::GatedDeltaNet;
use super::text::{Attention, Mlp};

/// One Qwen3.5 decoder layer: either linear-attention (GDN) or full-attention.
///
/// The two block kinds are kept in `Option` fields rather than an enum so the
/// `#[derive(ModuleParameters, Quantizable)]` macros can walk both paths in
/// the weight loader. Exactly one of `self_attn` / `linear_attn` is populated
/// for each layer.
#[derive(Debug, ModuleParameters, Quantizable)]
pub struct DecoderLayer {
    pub is_linear: bool,

    #[quantizable]
    #[param]
    pub self_attn: Option<Attention>,

    #[quantizable]
    #[param]
    pub linear_attn: Option<GatedDeltaNet>,

    #[param]
    pub input_layernorm: nn::RmsNorm,

    #[param]
    pub post_attention_layernorm: nn::RmsNorm,

    #[quantizable]
    #[param]
    pub mlp: Mlp,
}

impl DecoderLayer {
    /// Build a layer of the right kind for the given index.
    pub fn new(cfg: &TextConfig, layer_idx: usize) -> Result<Self, Exception> {
        let is_linear = layer_is_linear(cfg, layer_idx);
        let (self_attn, linear_attn) = if is_linear {
            (None, Some(GatedDeltaNet::new(cfg)?))
        } else {
            (Some(Attention::new(cfg)?), None)
        };
        let input_layernorm = nn::RmsNormBuilder::new(cfg.hidden_size)
            .eps(cfg.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(cfg.hidden_size)
            .eps(cfg.rms_norm_eps)
            .build()?;
        let mlp = Mlp::new(cfg.hidden_size, cfg.intermediate_size)?;
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

    /// Toggle training mode on every quantisable parameter.
    pub fn training_mode(&mut self, mode: bool) {
        if let Some(blk) = self.self_attn.as_mut() {
            blk.training_mode(mode);
        }
        if let Some(blk) = self.linear_attn.as_mut() {
            blk.training_mode(mode);
        }
        self.mlp.gate_proj.training_mode(mode);
        self.mlp.down_proj.training_mode(mode);
        self.mlp.up_proj.training_mode(mode);
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

/// Top-level decoder model: embeddings + layers + final norm.
#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Qwen35Decoder {
    pub vocab_size: i32,
    pub num_hidden_layers: i32,

    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,

    #[quantizable]
    #[param]
    pub layers: Vec<DecoderLayer>,

    #[param]
    pub norm: nn::RmsNorm,
}

impl Qwen35Decoder {
    /// Build a freshly-initialised decoder from a [`TextConfig`].
    pub fn new(cfg: &TextConfig) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(cfg.vocab_size, cfg.hidden_size)?;
        let layers = (0..cfg.num_hidden_layers as usize)
            .map(|i| DecoderLayer::new(cfg, i))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(cfg.hidden_size)
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
        let mut h = match inputs_embeds {
            Some(e) => e.clone(),
            None => {
                let ids = inputs.ok_or_else(|| {
                    Exception::custom(
                        "Qwen35Decoder::forward: needs either inputs or inputs_embeds",
                    )
                })?;
                self.embed_tokens.forward(ids)?
            }
        };

        if caches.len() != self.layers.len() {
            return Err(Exception::custom(format!(
                "Qwen35Decoder::forward: expected {} caches, got {}",
                self.layers.len(),
                caches.len()
            )));
        }

        // Hybrid mask scheme — matches `mlx_vlm.qwen3_5.language.Qwen3_5Model`:
        //
        // - Full-attention layers use a causal mask of shape `[1, 1, T, kv]`
        //   built from the matching layer's KV-cache offset. When `T == 1`
        //   (decode) the mask is dropped — every full-attn block already
        //   relies on `fast::scaled_dot_product_attention` to handle the
        //   trivial 1-row case.
        // - Linear-attention layers don't need a mask in single-batch /
        //   non-chunked prefill, so we always pass `None` (the Python
        //   reference's `create_ssm_mask` returns `None` here too).
        let full_attn_mask = self.build_full_attn_mask(&h, caches)?;
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

    /// Build the additive causal mask the full-attention layers need.
    ///
    /// Mirrors `mlx_lm.models.base.create_attention_mask` for the hybrid
    /// cache: pulls the offset from the first full-attention slot (its
    /// KV cache tracks the prefill-vs-decode boundary) and emits a `[T, T +
    /// offset]` boolean matrix lifted to a 4-D broadcast-friendly shape.
    ///
    /// Returns `None` for decode steps (`T == 1`) where the implicit causal
    /// handling inside `fast::scaled_dot_product_attention` already covers
    /// the trivial mask shape.
    fn build_full_attn_mask(
        &self,
        h: &Array,
        caches: &[LayerCache],
    ) -> Result<Option<Array>, Exception> {
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
                _ => None,
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

/// LM-head wrapper. Optional `lm_head` linear; if absent, logits are computed
/// by tying with `embed_tokens.as_linear`.
#[derive(Debug, ModuleParameters, Quantizable)]
pub struct LanguageModel {
    pub cfg: TextConfig,

    #[quantizable]
    #[param]
    pub model: Qwen35Decoder,

    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl LanguageModel {
    /// Build a freshly-initialised language model.
    pub fn new(cfg: TextConfig) -> Result<Self, Exception> {
        let model = Qwen35Decoder::new(&cfg)?;
        let lm_head = if !cfg.tie_word_embeddings {
            Some(MaybeQuantized::Original(
                nn::LinearBuilder::new(cfg.hidden_size, cfg.vocab_size)
                    .bias(false)
                    .build()?,
            ))
        } else {
            None
        };
        Ok(Self {
            cfg,
            model,
            lm_head,
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
        let hidden = self
            .model
            .forward(inputs, inputs_embeds, caches, position_ids)?;
        if let Some(head) = self.lm_head.as_mut() {
            head.forward(&hidden)
        } else {
            match &mut self.model.embed_tokens {
                MaybeQuantized::Original(e) => e.as_linear(&hidden),
                MaybeQuantized::Quantized(q) => q.as_linear(&hidden),
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen3_5::cache::make_caches;
    use mlx_rs::{random::uniform, transforms::eval, Array};

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
        let mut lm = LanguageModel::new(cfg.text_config.clone()).unwrap();
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
        let fa = match &caches[3] {
            LayerCache::FullAttention(c) => c,
            _ => panic!("expected FullAttention cache at index 3"),
        };
        use crate::cache::KeyValueCache;
        assert_eq!(fa.offset(), 6);

        // The linear-attn layers should each be at offset 6 too.
        for (i, cache) in caches.iter().enumerate().take(3) {
            let la = match cache {
                LayerCache::LinearAttention(c) => c,
                _ => panic!("expected LinearAttention cache at index {i}"),
            };
            assert_eq!(la.offset, 6, "linear layer {i} offset");
        }
    }

    #[test]
    fn forward_accepts_inputs_embeds() {
        let cfg = synthetic_config(vec!["linear_attention", "full_attention"]);
        let mut lm = LanguageModel::new(cfg.text_config.clone()).unwrap();
        let mut caches = make_caches(&cfg);
        let embeds =
            uniform::<_, f32>(0.0, 1.0, &[1, 3, cfg.text_config.hidden_size], None).unwrap();
        let logits = lm.forward(None, Some(&embeds), &mut caches, None).unwrap();
        eval([&logits]).unwrap();
        assert_eq!(logits.shape(), &[1, 3, cfg.text_config.vocab_size]);
    }
}
