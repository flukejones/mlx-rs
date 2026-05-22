//! The two traits every model implements.
//!
//! - [`UserInputProcessor`] preprocesses a [`crate::UserInput`] into
//!   the model-facing [`crate::LMInput`]: tokenises the chat
//!   template, runs the vision tower on attached images, decodes the
//!   audio tower's features, etc. One impl per family.
//! - [`LanguageModel`] owns the parsed model + its KV cache + the
//!   per-step decoder. [`LanguageModel::prepare`] ingests the
//!   `LMInput` to prime the cache; [`LanguageModel::step`] produces
//!   one token's logits at a time.
//!
//! The [`crate::ModelContext`] holds one of each plus a tokenizer
//! and the parsed config, and the top-level [`crate::generate`]
//! drives them.

use crate::error::Error;
use crate::lm_input::{LMInput, LMOutput, PrepareResult, Text};
use crate::user_input::UserInput;

/// Turn a [`UserInput`] into an [`LMInput`].
///
/// One impl per model family. The impl is the *only* place that
/// knows the family's specific preprocessing details — chat-template
/// quirks, the right image-pad token, audio sample rate, etc.
/// Consumers above this trait don't branch on family.
pub trait UserInputProcessor: Send {
    /// A short identifier for the family this processor belongs to
    /// (`"llama"`, `"qwen3_5"`, `"qwen3_5_moe"`, `"gemma4"`, …).
    /// Used only for error messages.
    fn family(&self) -> &'static str;

    /// Convert the user-facing input into the model-facing input.
    ///
    /// Returns [`Error::ModalityUnsupported`] when the input
    /// carries a modality this processor can't handle. Implementors
    /// may carry mutable state (a CPU-side image preprocessor cache,
    /// a Jinja-template compile cache) — hence `&mut self`.
    fn prepare(&mut self, input: UserInput) -> Result<LMInput, Error>;

    /// Decode generated token ids back into UTF-8. The processor
    /// owns the tokenizer (same one it encoded the prompt with), so
    /// this is family-agnostic from the caller's perspective.
    fn decode(&self, ids: &[u32]) -> Result<String, Error>;
}

/// One language model — text-only or multimodal.
///
/// The trait owns generation: implementors hold the parsed module
/// graph + their KV cache as fields, so calls go through the trait
/// rather than threading a cache through the API surface. Concrete
/// model types stay family-specific (qwen3_5's
/// `LanguageModel<Mlp>`, llama's `Model`, etc.); this trait is the
/// thin dispatcher [`crate::ModelContext`] holds as
/// `Box<dyn LanguageModel>`.
pub trait LanguageModel: Send {
    /// Reset the KV cache. Called at the start of every
    /// [`crate::generate`] turn so a fresh `prepare` doesn't reuse
    /// state from a previous request.
    fn reset(&mut self);

    /// Ingest the prompt + any multimodal inputs and prime the KV
    /// cache.
    ///
    /// Returns [`PrepareResult::Primed`] on the normal path (the
    /// caller follows up with [`step`] calls to produce tokens), or
    /// [`PrepareResult::Logits`] if the model has already computed
    /// the next-token distribution as part of prefill.
    ///
    /// [`step`]: LanguageModel::step
    fn prepare(&mut self, input: LMInput) -> Result<PrepareResult, Error>;

    /// Produce one token's logits, given the previously-sampled
    /// token as a `[1]` `int32` `Array` (lives on the GPU; the
    /// driver passes the sampler's output directly so no host
    /// materialisation or device upload happens per decode step).
    ///
    /// The model is responsible for advancing its own cursor /
    /// position state and for reshaping `last_token` to `[1, 1]`
    /// internally if needed.
    fn step(&mut self, last_token: &mlx_rs::Array) -> Result<LMOutput, Error>;

    /// The model's text vocab size, used to validate sampled token
    /// ids before they're decoded.
    fn vocab_size(&self) -> i32;

    /// Maximum number of prompt tokens this model's KV cache can
    /// hold in a single forward pass. `Some(W)` for sliding-window
    /// caches (gemma4); `None` for unbounded caches.
    ///
    /// When `Some(W)` and the rendered prompt exceeds `W` tokens,
    /// [`crate::generate`] splits the prefill into chunks of size
    /// `W` and calls [`Self::prefill_chunk`] on every chunk except
    /// the last. The trailing chunk goes through the normal
    /// [`Self::prepare`] / [`Self::step`] path so its logits seed
    /// the first sampled token.
    fn prefill_chunk_size(&self) -> Option<i32> {
        None
    }

    /// Ingest one prefill chunk and advance the KV cache without
    /// returning logits. Called by [`crate::generate`] only when
    /// [`Self::prefill_chunk_size`] returns `Some(W)` and the
    /// prompt is longer than `W`.
    ///
    /// `tokens` is a `[1, chunk_len]` int32 view into the prompt;
    /// `chunk_len <= prefill_chunk_size().unwrap()`. The
    /// implementation should append the chunk to its KV cache and
    /// discard the logits.
    ///
    /// Default impl returns an error — only models that opted into
    /// chunked prefill (by overriding [`Self::prefill_chunk_size`])
    /// need to implement it.
    fn prefill_chunk(&mut self, _tokens: &mlx_rs::Array) -> Result<(), Error> {
        Err(Error::Other(
            "prefill_chunk called on a model with no prefill_chunk_size override".into(),
        ))
    }

    /// True iff this model has an MTP head loaded — i.e.
    /// [`Self::try_mtp_decode`] will return `Some` on every call.
    /// Default `false`.
    fn has_mtp(&self) -> bool {
        false
    }

    /// MTP self-speculative step. Returns 1 or 2 just-committed
    /// token ids plus the next not-yet-committed pending token (a
    /// `[1]` int32 array).
    ///
    /// `Ok(None)` is reserved for the default impl on models without
    /// an MTP head. Callers that gate on [`Self::has_mtp`] before
    /// calling can treat `None` as unreachable (defence-in-depth
    /// against `has_mtp()` returning a stale value).
    ///
    /// Sampling: `sampler` is the same [`crate::sampler::SamplerState`]
    /// the non-MTP loop uses for this `generate()` call. At
    /// `temperature == 0.0` the adapter takes the greedy fast path
    /// (argmax + accept-if-equal); at `temperature > 0` it runs
    /// Leviathan-2023 rejection sampling against the verify
    /// distribution so the output distribution matches what the
    /// non-MTP path would have produced.
    fn try_mtp_decode(
        &mut self,
        _last_token: &mlx_rs::Array,
        _sampler: &mut crate::sampler::SamplerState,
    ) -> Result<Option<(Vec<u32>, mlx_rs::Array)>, Error> {
        Ok(None)
    }
}

/// Convenience for text-only models: a processor that does nothing
/// but render the chat template + tokenise. Rejects populated
/// `images`/`audios`/`videos` with [`Error::ModalityUnsupported`].
///
/// Each text-only family wraps its own loaded tokenizer + chat
/// template in this struct.
pub struct TextOnlyProcessor {
    family: &'static str,
    tokenizer: tokenizers::Tokenizer,
    chat_template: crate::chat_template::ChatTemplate,
}

impl TextOnlyProcessor {
    /// Build a text-only processor bound to a family identifier.
    pub fn new(
        family: &'static str,
        tokenizer: tokenizers::Tokenizer,
        chat_template: crate::chat_template::ChatTemplate,
    ) -> Self {
        Self {
            family,
            tokenizer,
            chat_template,
        }
    }
}

impl UserInputProcessor for TextOnlyProcessor {
    fn family(&self) -> &'static str {
        self.family
    }

    fn prepare(&mut self, input: UserInput) -> Result<LMInput, Error> {
        #[cfg(feature = "models-vision")]
        if !input.images.is_empty() {
            return Err(Error::ModalityUnsupported {
                family: self.family,
                modality: "image",
            });
        }
        if !input.audios.is_empty() {
            return Err(Error::ModalityUnsupported {
                family: self.family,
                modality: "audio",
            });
        }
        if !input.videos.is_empty() {
            return Err(Error::ModalityUnsupported {
                family: self.family,
                modality: "video",
            });
        }

        let template_kwargs = input.template_kwargs;
        let rendered = match input.prompt {
            crate::user_input::Prompt::Text(s) => s,
            crate::user_input::Prompt::Chat(msgs) => {
                self.chat_template.render(&msgs, true, &template_kwargs)?
            }
        };

        let enc = self
            .tokenizer
            .encode(rendered.as_str(), false)
            .map_err(|e| Error::Other(format!("tokenizer encode: {e}").into()))?;
        let ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
        let len = ids.len() as i32;
        let tokens = mlx_rs::Array::from_slice(&ids, &[1, len]);
        Ok(LMInput {
            text: Text { tokens, mask: None },
            image: None,
            audio: None,
            video: None,
        })
    }

    fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| Error::Other(format!("tokenizer decode: {e}").into()))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    use std::collections::HashMap;

    use super::*;
    use crate::chat_template::{ChatMessage, ChatTemplate};
    use crate::user_input::{Audio, UserInput};

    fn dummy_processor() -> TextOnlyProcessor {
        // Minimal tokenizer + template — exercises the reject paths
        // and the happy path without needing an on-disk model.
        let tok_json = r#"{
            "version":"1.0","truncation":null,"padding":null,
            "added_tokens":[],"normalizer":null,"pre_tokenizer":null,
            "post_processor":null,"decoder":null,
            "model":{"type":"WordLevel","vocab":{"hello":0,"world":1,"<unk>":2},"unk_token":"<unk>"}
        }"#;
        let tokenizer = tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();
        let template = ChatTemplate::from_source(
            "{% for m in messages %}{{ m.role }}={{ m.content }}|{% endfor %}",
        );
        TextOnlyProcessor::new("test", tokenizer, template)
    }

    #[test]
    fn rejects_audio() {
        let mut p = dummy_processor();
        let input = UserInput {
            prompt: crate::user_input::Prompt::Text("hi".into()),
            #[cfg(feature = "models-vision")]
            images: Vec::new(),
            audios: vec![Audio::Pcm {
                samples: vec![0.0],
                sample_rate: 16_000,
            }],
            videos: Vec::new(),
            template_kwargs: HashMap::new(),
        };
        let err = p.prepare(input).unwrap_err();
        assert!(matches!(
            err,
            Error::ModalityUnsupported {
                modality: "audio",
                ..
            }
        ));
    }

    #[test]
    #[cfg(feature = "models-vision")]
    fn rejects_image() {
        use crate::user_input::Image;
        use image::DynamicImage;
        let mut p = dummy_processor();
        let input = UserInput {
            prompt: crate::user_input::Prompt::Text("hi".into()),
            images: vec![Image::Decoded(DynamicImage::new_rgb8(1, 1))],
            audios: Vec::new(),
            videos: Vec::new(),
            template_kwargs: HashMap::new(),
        };
        let err = p.prepare(input).unwrap_err();
        assert!(matches!(
            err,
            Error::ModalityUnsupported {
                modality: "image",
                ..
            }
        ));
    }

    #[test]
    fn text_prompt_round_trips() {
        let mut p = dummy_processor();
        let input = UserInput::text("hello world");
        let lm = p.prepare(input).unwrap();
        assert!(lm.image.is_none());
        assert!(lm.audio.is_none());
        assert!(lm.video.is_none());
        let shape = lm.text.tokens.shape();
        assert_eq!(shape[0], 1, "batch dim should be 1");
        assert!(shape[1] >= 1, "tokenized to nothing: shape={shape:?}");
    }

    #[test]
    fn chat_prompt_renders_through_template() {
        let mut p = dummy_processor();
        let input = UserInput::chat(vec![ChatMessage::user("hello")]);
        let lm = p.prepare(input).unwrap();
        // Template rendered to "user=hello|" — tokenizer sees three
        // unknowns + one known "hello"; what we care about here is
        // that it didn't crash and produced a [1, S] int32 tensor.
        assert_eq!(lm.text.tokens.shape()[0], 1);
        assert!(lm.text.tokens.shape()[1] > 0);
    }
}
