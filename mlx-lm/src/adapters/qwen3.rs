//! Qwen3 (dense, non-VL) [`crate::LanguageModel`] adapter.

use std::path::Path;

use mlx_rs::{module::Module, ops::indexing::IndexOp, Array};

use crate::adapters::LoadedContext;
use crate::cache::KVCache;
use crate::chat_template::ChatTemplate;
use crate::error::Error;
use crate::language_model::{LanguageModel, TextOnlyProcessor};
use crate::lm_input::{LMInput, LMOutput, PrepareResult};
use crate::loader::load_tokenizer;
use crate::models::qwen3::{load_qwen3_model, Model};
use crate::nn::ModelInput;

pub(crate) struct Qwen3Adapter {
    model: Model,
    cache: Vec<Option<KVCache>>,
    vocab_size: i32,
}

impl Qwen3Adapter {
    fn load(dir: &Path) -> Result<Self, Error> {
        let model = load_qwen3_model(dir)?;
        let vocab_size = model.args.vocab_size;
        Ok(Self {
            model,
            cache: Vec::new(),
            vocab_size,
        })
    }
}

impl LanguageModel for Qwen3Adapter {
    fn reset(&mut self) {
        self.cache.clear();
    }

    fn prepare(&mut self, input: LMInput) -> Result<PrepareResult, Error> {
        debug_assert!(input.image.is_none());
        debug_assert!(input.audio.is_none());
        debug_assert!(input.video.is_none());

        let input_arr = input.text.tokens;
        let logits = self.model.forward(ModelInput {
            inputs: &input_arr,
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
}

pub(crate) fn load_context(dir: &Path) -> Result<LoadedContext, Error> {
    let model = Qwen3Adapter::load(dir)?;
    let tokenizer = load_tokenizer(dir)?;
    let chat_template = ChatTemplate::from_dir(dir)?;
    let eos_ids = crate::adapters::read_eos_ids(dir);
    let processor = TextOnlyProcessor::new("qwen3", tokenizer, chat_template);
    Ok((Box::new(model), Box::new(processor), eos_ids))
}
