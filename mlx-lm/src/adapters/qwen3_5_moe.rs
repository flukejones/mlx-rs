//! Qwen3.5-MoE (35B-A3B) [`crate::LanguageModel`] adapter.
//!
//! Same prefill / decode shape as the dense qwen3.5 adapter; the
//! only difference is the inner FFN type (`Qwen35MoeBlock`). No
//! multimodal path — MoE checkpoints are text-only.

use std::path::Path;

use mlx_rs::{ops::indexing::IndexOp, Array};

use crate::adapters::LoadedContext;
use crate::chat_template::ChatTemplate;
use crate::error::Error;
use crate::language_model::{LanguageModel, TextOnlyProcessor};
use crate::lm_input::{LMInput, LMOutput, PrepareResult};
use crate::loader::load_tokenizer;
use crate::models::qwen3_5::cache::{make_caches, LayerCache};
use crate::models::qwen3_5::config::ModelConfig;
use crate::models::qwen3_5::layer::LanguageModel as Qwen35LanguageModel;
use crate::models::qwen3_5_moe::{load_qwen3_5_moe_model, Qwen35MoeBlock};

pub(crate) struct Qwen35MoeAdapter {
    model: Qwen35LanguageModel<Qwen35MoeBlock>,
    cfg: ModelConfig,
    cache: Vec<LayerCache>,
    vocab_size: i32,
}

impl Qwen35MoeAdapter {
    fn load(dir: &Path) -> Result<Self, Error> {
        let model = load_qwen3_5_moe_model(dir)?;
        // The MoE loader doesn't return the parsed config; re-read.
        let cfg = ModelConfig::from_file(dir.join("config.json"))?;
        let cache = make_caches(&cfg);
        let vocab_size = cfg.text_config.vocab_size;
        Ok(Self {
            model,
            cfg,
            cache,
            vocab_size,
        })
    }
}

impl LanguageModel for Qwen35MoeAdapter {
    fn reset(&mut self) {
        self.cache = make_caches(&self.cfg);
    }

    fn prepare(&mut self, input: LMInput) -> Result<PrepareResult, Error> {
        debug_assert!(input.image.is_none());
        debug_assert!(input.audio.is_none());
        debug_assert!(input.video.is_none());

        let tokens = input.text.tokens;
        let logits = self
            .model
            .forward(Some(&tokens), None, &mut self.cache, None)?;
        Ok(PrepareResult::Logits(logits.index((.., -1, ..))))
    }

    fn step(&mut self, last_token: &Array) -> Result<LMOutput, Error> {
        let inp = last_token.reshape(&[1, 1])?;
        let logits = self
            .model
            .forward(Some(&inp), None, &mut self.cache, None)?;
        Ok(LMOutput {
            logits: logits.index((.., -1, ..)),
        })
    }

    fn vocab_size(&self) -> i32 {
        self.vocab_size
    }
}

pub(crate) fn load_context(dir: &Path) -> Result<LoadedContext, Error> {
    let model = Qwen35MoeAdapter::load(dir)?;
    let cfg = ModelConfig::from_file(dir.join("config.json"))?;
    let tokenizer = load_tokenizer(dir)?;
    let chat_template = ChatTemplate::from_dir(dir)?;
    let eos_ids = crate::models::qwen3_5::read_qwen3_5_eos_ids(dir, &cfg);
    let processor = TextOnlyProcessor::new("qwen3_5_moe", tokenizer, chat_template);
    Ok((Box::new(model), Box::new(processor), eos_ids))
}
