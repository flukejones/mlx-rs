//! Gemma 4 [`crate::LanguageModel`] adapter.
//!
//! Gemma 4 uses a per-layer sliding/global cache enum
//! ([`crate::models::gemma4::loader::Gemma4LayerCache`]) instead of
//! the bare [`crate::cache::KVCache`] used by llama / qwen3. The
//! `Vec<Option<Gemma4LayerCache>>` slots are built via
//! [`crate::models::gemma4::loader::make_gemma4_caches`] up front
//! because shared-KV layers need pre-allocated `None` slots.

use std::path::Path;

use mlx_rs::{module::Module, ops::indexing::IndexOp, Array};

use crate::adapters::LoadedContext;
use crate::chat_template::ChatTemplate;
use crate::error::Error;
use crate::language_model::{LanguageModel, TextOnlyProcessor};
use crate::lm_input::{LMInput, LMOutput, PrepareResult};
use crate::loader::load_tokenizer;
use crate::models::gemma4::config::Gemma4Config;
use crate::models::gemma4::loader::{load_gemma4_model, make_gemma4_caches, Gemma4LayerCache};
use crate::models::gemma4::text::Model;
use crate::nn::ModelInput;

pub(crate) struct Gemma4Adapter {
    model: Model,
    cache: Vec<Option<Gemma4LayerCache>>,
    args: Gemma4Config,
    vocab_size: i32,
}

impl Gemma4Adapter {
    fn load(dir: &Path) -> Result<Self, Error> {
        let model = load_gemma4_model(dir)?;
        let args = model.args.clone();
        let vocab_size = args.vocab_size;
        let cache = make_gemma4_caches(&args);
        Ok(Self {
            model,
            cache,
            args,
            vocab_size,
        })
    }
}

impl LanguageModel for Gemma4Adapter {
    fn reset(&mut self) {
        self.cache = make_gemma4_caches(&self.args);
    }

    fn prepare(&mut self, input: LMInput) -> Result<PrepareResult, Error> {
        debug_assert!(input.image.is_none());
        debug_assert!(input.audio.is_none());
        debug_assert!(input.video.is_none());

        let logits = self.model.forward(ModelInput {
            inputs: &input.text.tokens,
            mask: None,
            cache: &mut self.cache,
        })?;
        Ok(PrepareResult::Logits(logits.index((.., -1, ..))))
    }

    fn step(&mut self, last_token: i32) -> Result<LMOutput, Error> {
        let inp = Array::from_slice(&[last_token], &[1, 1]);
        let logits = self.model.forward(ModelInput {
            inputs: &inp,
            mask: None,
            cache: &mut self.cache,
        })?;
        Ok(LMOutput {
            logits: logits.index((.., -1, ..)),
        })
    }

    fn vocab_size(&self) -> i32 {
        self.vocab_size
    }

    /// Gemma 4's sliding-attention layers cap each forward pass at
    /// `sliding_window` K/V positions. Prompts longer than that
    /// must be chunked through [`Self::prefill_chunk`]; otherwise
    /// the sliding cache rotates earlier prefill tokens out of
    /// reach of the queries that need them, and SDPA fails the
    /// `[N, K_len]` mask/K shape match.
    fn prefill_chunk_size(&self) -> Option<i32> {
        Some(self.args.sliding_window)
    }

    /// One prefill step: feed the chunk into the model, advance the
    /// per-layer caches, discard the logits. Cache state carries
    /// forward across calls (the rotating cache's early-return
    /// branch keeps the prior window snapshot in scope for each new
    /// chunk's queries).
    fn prefill_chunk(&mut self, tokens: &Array) -> Result<(), Error> {
        let _ = self.model.forward(ModelInput {
            inputs: tokens,
            mask: None,
            cache: &mut self.cache,
        })?;
        Ok(())
    }
}

pub(crate) fn load_context(dir: &Path) -> Result<LoadedContext, Error> {
    let model = Gemma4Adapter::load(dir)?;
    let tokenizer = load_tokenizer(dir)?;
    let chat_template = ChatTemplate::from_dir(dir)?;
    let eos_ids = crate::adapters::read_eos_ids(dir);
    let processor = TextOnlyProcessor::new("gemma4", tokenizer, chat_template);
    Ok((Box::new(model), Box::new(processor), eos_ids))
}
