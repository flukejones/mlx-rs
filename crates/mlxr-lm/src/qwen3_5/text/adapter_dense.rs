//! Qwen3.5 dense [`crate::LanguageModel`] adapter.
//!
//! The dense path: `LanguageModel<Mlp>` with a hybrid linear-attn +
//! full-attn cache stack. Drives prefill / decode by calling the
//! model's `forward` directly. The text-only adapter is built
//! standalone; the VLM path wraps this adapter with the vision tower
//! plus multimodal embedding stitch (see
//! [`crate::qwen3_5::image::adapter`]).

use std::path::Path;

use mlxr::{ops::indexing::IndexOp, Array};

use crate::chat_template::ChatTemplate;
use crate::error::Error;
use crate::family::LoadedContext;
use crate::language_model::{LanguageModel, TextOnlyProcessor};
use crate::lm_input::{LMInput, LMOutput, PrepareResult};
use crate::loader::load_tokenizer;
use crate::qwen3_5::text::cache::{make_caches, LayerCache};
use crate::qwen3_5::text::config::ModelConfig;
use crate::qwen3_5::text::layer::Qwen35Model;
use crate::qwen3_5::text::text::Mlp;
use crate::qwen3_5::text::weights::load_language_model;

pub(crate) struct Qwen35DenseAdapter {
    pub(crate) model: Qwen35Model<Mlp>,
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
    pub(crate) fn new(model: Qwen35Model<Mlp>, cfg: ModelConfig) -> Self {
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
    #[cfg(feature = "image")]
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

/// Load a qwen3_5 dense (text-only) checkpoint. Caller is the
/// family-level [`crate::qwen3_5::load_context`] dispatcher; it
/// guarantees the directory carries the dense weights only (no
/// `preprocessor_config.json`).
pub(crate) fn load_context_dense(dir: &Path) -> Result<LoadedContext, Error> {
    let cfg = ModelConfig::from_file(dir.join("config.json"))?;
    let tokenizer = load_tokenizer(dir)?;
    let chat_template = ChatTemplate::from_dir(dir)?;
    let eos_ids = crate::qwen3_5::text::read_qwen3_5_eos_ids(dir, &cfg);

    let (model, leftover) = load_language_model(&cfg, dir)?;
    if !leftover.is_empty() {
        return Err(Error::Other(
            format!(
                "qwen3_5 dense load: {} unbound key(s); first 8: {:?}",
                leftover.len(),
                leftover.iter().take(8).collect::<Vec<_>>()
            )
            .into(),
        ));
    }

    let dense = Qwen35DenseAdapter::new(model, cfg);
    let processor = TextOnlyProcessor::new("qwen3_5", tokenizer, chat_template);
    Ok((Box::new(dense), Box::new(processor), eos_ids))
}
