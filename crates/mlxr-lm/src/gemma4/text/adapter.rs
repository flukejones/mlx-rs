//! Gemma 4 [`crate::LanguageModel`] adapter.
//!
//! Gemma 4 uses a per-layer sliding/global cache enum
//! ([`crate::gemma4::text::loader::LayerCache`]) instead of
//! the bare [`crate::cache::KVCache`] used by llama / qwen3. The
//! `Vec<Option<LayerCache>>` slots are built via
//! [`crate::gemma4::text::loader::make_caches`] up front
//! because shared-KV layers need pre-allocated `None` slots.

use std::path::Path;

use mlxr::{module::Module, ops::indexing::IndexOp, Array};

use crate::cache::{build_rotation, effective_prefill_chunk_opt, CacheOptions};
use crate::chat_template::ChatTemplate;
use crate::config::ModelConfig as Config;
use crate::error::Error;
use crate::family::{EosSpec, LoadedContext};
use crate::gemma4::text::config::{ModelConfig, TextConfig};
use crate::gemma4::text::loader::{make_caches_with_rotation, LayerCache};
use crate::gemma4::text::text::Model;
use crate::gemma4::text::weights::load_model;
use crate::language_model::{LanguageModel, TextOnlyProcessor};
use crate::lm_input::{LMInput, LMOutput, PrepareResult};
use crate::loader::load_tokenizer;
use crate::nn::ModelInput;

pub(crate) struct Gemma4Adapter {
    model: Model,
    cache: Vec<Option<LayerCache>>,
    args: TextConfig,
    /// Default: standard + steel_prefill on (gemma4 full-attn was
    /// always steel-prefill before C8).
    cache_options: CacheOptions,
    /// TurboQuant Π for the full-attn slots; shared across resets.
    /// Built from `global_head_dim` (sliding never goes quantised).
    rotation: Option<Array>,
    vocab_size: i32,
}

impl Gemma4Adapter {
    fn load(cfg: &Config, env: &ModelConfig, dir: &Path) -> Result<Self, Error> {
        let model = load_model(cfg, env, dir)?;
        let args = model.args.clone();
        let vocab_size = args.vocab_size;
        let cache_options = CacheOptions::standard_with_steel_prefill();
        let rotation = build_rotation(cache_options, args.global_head_dim)?;
        let cache = make_caches_with_rotation(&args, cache_options, rotation.as_ref());
        Ok(Self {
            model,
            cache,
            args,
            cache_options,
            rotation,
            vocab_size,
        })
    }
}

impl LanguageModel for Gemma4Adapter {
    fn reset(&mut self) {
        self.cache =
            make_caches_with_rotation(&self.args, self.cache_options, self.rotation.as_ref());
    }

    fn prepare(&mut self, input: LMInput) -> Result<PrepareResult, Error> {
        debug_assert!(input.image.is_none());

        let logits = self.model.forward(ModelInput {
            inputs: &input.text.tokens,
            mask: None,
            cache: &mut self.cache,
        })?;
        Ok(PrepareResult::Logits(logits.index((.., -1, ..))))
    }

    fn step(&mut self, last_token: &Array) -> Result<LMOutput, Error> {
        let inp = last_token.reshape(&[1, 1])?;
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
    /// `sliding_window` K/V positions. Combine with the user cap so
    /// `CacheOptions::max_prefill_chunk` can narrow further but never
    /// exceed the sliding window.
    fn prefill_chunk_size(&self) -> Option<i32> {
        effective_prefill_chunk_opt(&self.cache, self.cache_options.max_prefill_chunk)
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

    fn set_cache_options(&mut self, options: CacheOptions) -> Result<(), Error> {
        let rotation = build_rotation(options, self.args.global_head_dim)?;
        self.cache = make_caches_with_rotation(&self.args, options, rotation.as_ref());
        self.rotation = rotation;
        self.cache_options = options;
        Ok(())
    }
}

pub(crate) fn load_context(
    cfg: &Config,
    env: &ModelConfig,
    dir: &Path,
) -> Result<LoadedContext, Error> {
    let model = Gemma4Adapter::load(cfg, env, dir)?;
    let tokenizer = load_tokenizer(dir)?;
    let chat_template = ChatTemplate::from_dir(dir)?;
    let eos_ids = EosSpec::to_vec(env.eos_token_id.as_ref());
    let processor = TextOnlyProcessor::new("gemma4", tokenizer, chat_template);
    Ok((Box::new(model), Box::new(processor), eos_ids))
}
