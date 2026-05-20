//! Llama [`crate::LanguageModel`] adapter.

use std::path::Path;

use mlx_rs::{module::Module, ops::indexing::IndexOp, Array};

use crate::adapters::LoadedContext;
use crate::cache::KVCache;
use crate::chat_template::ChatTemplate;
use crate::error::Error;
use crate::language_model::{LanguageModel, TextOnlyProcessor};
use crate::lm_input::{LMInput, LMOutput, PrepareResult};
use crate::loader::load_tokenizer;
use crate::models::llama::{load_llama_model, Model};
use crate::nn::ModelInput;

pub(crate) struct LlamaAdapter {
    model: Model,
    cache: Vec<Option<KVCache>>,
    vocab_size: i32,
}

impl LlamaAdapter {
    fn load(dir: &Path) -> Result<Self, Error> {
        let model = load_llama_model(dir)?;
        let vocab_size = model.args.vocab_size;
        Ok(Self {
            model,
            cache: Vec::new(),
            vocab_size,
        })
    }
}

impl LanguageModel for LlamaAdapter {
    fn reset(&mut self) {
        self.cache.clear();
    }

    fn prepare(&mut self, input: LMInput) -> Result<PrepareResult, Error> {
        // Llama doesn't accept image / audio / video; if the user
        // pushed any in, the family processor rejected it before we
        // got here. Defence in depth: assert in debug.
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

/// Build the [`crate::LanguageModel`] + [`crate::UserInputProcessor`]
/// plus EOS-id list for a llama checkpoint at `dir`. The tokenizer
/// lives on the processor (it owns the encode + decode round-trip).
pub(crate) fn load_context(dir: &Path) -> Result<LoadedContext, Error> {
    let model = LlamaAdapter::load(dir)?;
    let tokenizer = load_tokenizer(dir)?;
    let chat_template = ChatTemplate::from_dir(dir)?;
    let eos_ids = crate::adapters::read_eos_ids(dir);
    let processor = TextOnlyProcessor::new("llama", tokenizer, chat_template);
    Ok((Box::new(model), Box::new(processor), eos_ids))
}
