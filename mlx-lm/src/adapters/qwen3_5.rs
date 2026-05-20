//! Qwen3.5 dense [`crate::LanguageModel`] adapter.
//!
//! The dense path: `LanguageModel<Mlp>` with a hybrid linear-attn +
//! full-attn cache stack. Drives prefill / decode by calling the
//! model's `forward` directly. The text-only adapter is built
//! standalone; the VLM path wraps this adapter with the vision
//! tower + multimodal embedding stitch (see `qwen3_5_vlm.rs`).

use mlx_rs::{ops::indexing::IndexOp, Array};

use crate::error::Error;
use crate::language_model::LanguageModel;
use crate::lm_input::{LMInput, LMOutput, PrepareResult};
use crate::models::qwen3_5::cache::{make_caches, LayerCache};
use crate::models::qwen3_5::config::ModelConfig;
use crate::models::qwen3_5::layer::LanguageModel as Qwen35LanguageModel;
use crate::models::qwen3_5::text::Mlp;

pub(crate) struct Qwen35DenseAdapter {
    pub(crate) model: Qwen35LanguageModel<Mlp>,
    pub(crate) cfg: ModelConfig,
    pub(crate) cache: Vec<LayerCache>,
    /// Cumulative position (prompt tokens + decoded tokens). Only
    /// used by the multimodal `step` path to compute mrope `[3,1,1]`
    /// position ids; pure text never reads it.
    pub(crate) cursor: i32,
    /// `Some` after a multimodal prefill: per-step decode position is
    /// `cursor + rope_delta` broadcast across the three mrope axes.
    pub(crate) rope_delta: Option<i32>,
    pub(crate) vocab_size: i32,
}

impl Qwen35DenseAdapter {
    pub(crate) fn new(model: Qwen35LanguageModel<Mlp>, cfg: ModelConfig) -> Self {
        let cache = make_caches(&cfg);
        let vocab_size = cfg.text_config.vocab_size;
        Self {
            model,
            cfg,
            cache,
            cursor: 0,
            rope_delta: None,
            vocab_size,
        }
    }

    /// Multimodal prefill seed: pre-stitched embeddings + mrope
    /// position ids + rope_delta. Public for the VLM adapter.
    pub(crate) fn prefill_embeds(
        &mut self,
        inputs_embeds: Array,
        position_ids: Array,
        rope_delta: i32,
    ) -> Result<Array, Error> {
        let s = inputs_embeds.shape()[1];
        let logits = self.model.forward(
            None,
            Some(&inputs_embeds),
            &mut self.cache,
            Some(&position_ids),
        )?;
        self.cursor = s;
        self.rope_delta = Some(rope_delta);
        Ok(logits.index((.., -1, ..)))
    }
}

impl LanguageModel for Qwen35DenseAdapter {
    fn reset(&mut self) {
        self.cache = make_caches(&self.cfg);
        self.cursor = 0;
        self.rope_delta = None;
    }

    fn prepare(&mut self, input: LMInput) -> Result<PrepareResult, Error> {
        debug_assert!(input.audio.is_none());
        debug_assert!(input.video.is_none());

        // Dense text-only path; the VLM wrapper handles `input.image`
        // before delegating to `prefill_embeds`.
        debug_assert!(input.image.is_none());

        let tokens = input.text.tokens;
        let shape = tokens.shape();
        debug_assert_eq!(shape[0], 1, "batch dim must be 1");
        let s = shape[1];
        let logits = self
            .model
            .forward(Some(&tokens), None, &mut self.cache, None)?;
        self.cursor = s;
        self.rope_delta = None;
        Ok(PrepareResult::Logits(logits.index((.., -1, ..))))
    }

    fn step(&mut self, last_token: &Array) -> Result<LMOutput, Error> {
        let inp = last_token.reshape(&[1, 1])?;

        // Multimodal path needs an explicit `[3,1,1]` mrope position
        // id so the rope keeps advancing past the image block. Pure
        // text leaves `rope_delta = None` and the model derives the
        // position from the cache offset internally.
        let pos_owned;
        let pos = if let Some(delta) = self.rope_delta {
            let p = self.cursor + delta;
            pos_owned = Array::from_slice(&[p, p, p], &[3, 1, 1]);
            Some(&pos_owned)
        } else {
            None
        };
        let logits = self.model.forward(Some(&inp), None, &mut self.cache, pos)?;
        self.cursor += 1;
        Ok(LMOutput {
            logits: logits.index((.., -1, ..)),
        })
    }

    fn vocab_size(&self) -> i32 {
        self.vocab_size
    }
}
