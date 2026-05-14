//! Sampling + iterator-style generation loop for the Qwen3.5 `LanguageModel`.
//!
//! `Generate::new(...)` returns an `Iterator<Item = Result<u32, Exception>>`
//! that yields one token id per call. Internally it prefills on the first
//! `next()` and then decodes one token at a time, reusing the per-layer
//! caches from [`make_caches`].

use mlx_rs::{
    argmax_axis, array, categorical,
    error::Exception,
    ops::{cumsum, indexing::IndexOp, multiply, r#where, softmax_axis, sort_axis},
    Array,
};

use super::cache::{make_caches, LayerCache};
use super::config::{ModelConfig, QWEN_CHAT_EOS_TOKEN_ID};
use super::layer::LanguageModel;

/// Stopping criterion for [`Generate`].
#[derive(Debug, Clone)]
pub struct StopCriteria {
    /// Maximum number of new tokens to produce (excluding the prompt).
    pub max_new_tokens: i32,
    /// Token ids that, when produced, terminate the loop.
    pub eos_ids: Vec<u32>,
}

impl StopCriteria {
    /// Build the canonical stop criteria from a `ModelConfig`. Falls back to
    /// `[QWEN_CHAT_EOS_TOKEN_ID]` when the config has no `eos_token_id`.
    pub fn from_config(cfg: &ModelConfig, max_new_tokens: i32) -> Self {
        let eos_ids = match cfg.eos_token_id.clone() {
            Some(id) => id.into_vec_with_chat_eos(),
            None => vec![QWEN_CHAT_EOS_TOKEN_ID],
        };
        Self {
            max_new_tokens,
            eos_ids,
        }
    }
}

/// Sampling parameters.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Temperature. `0.0` selects argmax (deterministic).
    pub temperature: f32,
    /// Optional top-p (nucleus) value; ignored when `temperature == 0`.
    pub top_p: Option<f32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: None,
        }
    }
}

fn sample(logits: &Array, params: &SamplingParams) -> Result<Array, Exception> {
    if params.temperature == 0.0 {
        return argmax_axis!(logits, -1);
    }
    let scaled = multiply(logits, array!(1.0_f32 / params.temperature))?;
    match params.top_p {
        None => categorical!(scaled),
        Some(p) => top_p_sample(&scaled, p),
    }
}

fn top_p_sample(logits: &Array, p: f32) -> Result<Array, Exception> {
    let probs = softmax_axis(logits, -1, true)?;
    let sorted = sort_axis(&probs, -1)?;
    let csum = cumsum(&sorted, -1, false, false)?;
    let mask = csum.gt(Array::from_f32(1.0 - p))?;
    let zero = Array::from_f32(0.0).as_dtype(sorted.dtype())?;
    let nucleus = r#where(&mask, &sorted, &zero)?;
    categorical!(nucleus)
}

/// Prefill seed: either the raw prompt token ids or a pre-stitched
/// embedding sequence (used by the multimodal path).
enum PrefillSeed {
    Tokens(Array),
    Embeds {
        inputs_embeds: Array,
        position_ids: Array,
    },
}

/// Iterator-style generator.
pub struct Generate<'a> {
    model: &'a mut LanguageModel,
    caches: Vec<LayerCache>,
    stop: StopCriteria,
    params: SamplingParams,
    /// Pending seed for the first `next()`.
    seed: Option<PrefillSeed>,
    /// Last sampled token id; fed back as the next decode input.
    next_token: Option<i32>,
    /// Cumulative tokens consumed (prompt + decode steps); used to derive
    /// the per-step mrope position id when `rope_delta` is set.
    cursor: i32,
    /// `Some` for the multimodal path: per-decode-step position id is
    /// `cursor + rope_delta` (broadcast across the three mrope axes).
    rope_delta: Option<i32>,
    produced: i32,
    finished: bool,
}

impl<'a> Generate<'a> {
    /// Build a new text-only generator. `prompt_ids` is a 1-D `[S]` `int32`
    /// array.
    pub fn new(
        model: &'a mut LanguageModel,
        cfg: &ModelConfig,
        prompt_ids: Array,
        stop: StopCriteria,
        params: SamplingParams,
    ) -> Self {
        let caches = make_caches(cfg);
        Self {
            model,
            caches,
            stop,
            params,
            seed: Some(PrefillSeed::Tokens(prompt_ids)),
            next_token: None,
            cursor: 0,
            rope_delta: None,
            produced: 0,
            finished: false,
        }
    }

    /// Build a multimodal generator. The caller passes the pre-stitched
    /// `inputs_embeds` (with image features already scattered into the
    /// special-token slots) along with the mrope `position_ids` and the
    /// `rope_delta` produced by
    /// [`super::multimodal::get_rope_index_single_batch`].
    pub fn with_inputs_embeds(
        model: &'a mut LanguageModel,
        cfg: &ModelConfig,
        inputs_embeds: Array,
        position_ids: Array,
        rope_delta: i32,
        stop: StopCriteria,
        params: SamplingParams,
    ) -> Self {
        let caches = make_caches(cfg);
        let cursor = inputs_embeds.shape()[1];
        Self {
            model,
            caches,
            stop,
            params,
            seed: Some(PrefillSeed::Embeds {
                inputs_embeds,
                position_ids,
            }),
            next_token: None,
            cursor,
            rope_delta: Some(rope_delta),
            produced: 0,
            finished: false,
        }
    }
}

impl<'a> Iterator for Generate<'a> {
    type Item = Result<u32, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished || self.produced >= self.stop.max_new_tokens {
            return None;
        }

        let logits_full = match self.seed.take() {
            Some(PrefillSeed::Tokens(prompt)) => {
                let s = prompt.shape()[0];
                let input = match prompt.reshape(&[1, s]) {
                    Ok(a) => a,
                    Err(e) => return Some(Err(e)),
                };
                self.cursor = s;
                match self
                    .model
                    .forward(Some(&input), None, &mut self.caches, None)
                {
                    Ok(l) => l,
                    Err(e) => return Some(Err(e)),
                }
            }
            Some(PrefillSeed::Embeds {
                inputs_embeds,
                position_ids,
            }) => {
                match self.model.forward(
                    None,
                    Some(&inputs_embeds),
                    &mut self.caches,
                    Some(&position_ids),
                ) {
                    Ok(l) => l,
                    Err(e) => return Some(Err(e)),
                }
            }
            None => {
                let tok = self.next_token?;
                let input = Array::from_slice(&[tok], &[1, 1]);

                // For the multimodal path, hand the decoder an explicit
                // `[3, 1, 1]` position id so the mrope keeps advancing past
                // the image block in lockstep across the three axes.
                let pos_owned;
                let pos = if let Some(delta) = self.rope_delta {
                    let p = self.cursor + delta;
                    let arr = Array::from_slice(&[p, p, p], &[3, 1, 1]);
                    pos_owned = arr;
                    Some(&pos_owned)
                } else {
                    None
                };
                match self
                    .model
                    .forward(Some(&input), None, &mut self.caches, pos)
                {
                    Ok(l) => l,
                    Err(e) => return Some(Err(e)),
                }
            }
        };

        let last = logits_full.index((.., -1, ..));
        let tok = match sample(&last, &self.params) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };
        // `Array::item` already forces an eval — skip the explicit
        // `eval([&tok])` round-trip to save one host sync per decode step.
        let id = tok.item::<i32>();
        self.next_token = Some(id);
        self.cursor += 1;
        self.produced += 1;

        let id_u32 = id as u32;
        if self.stop.eos_ids.contains(&id_u32) {
            self.finished = true;
        }
        Some(Ok(id_u32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen3_5::weights::load_language_model;

    #[test]
    #[ignore = "requires local model files at ~/MLXModels/chandra2/chandra-ocr-2-mlx-q8"]
    fn greedy_generation_produces_finite_token_stream() {
        let home = std::env::var("HOME").unwrap();
        let dir = std::path::PathBuf::from(home).join("MLXModels/chandra2/chandra-ocr-2-mlx-q8");
        let cfg = ModelConfig::from_file(dir.join("config.json")).unwrap();
        let (mut model, _) = load_language_model(&cfg, &dir).unwrap();

        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
        let enc = tok.encode("Hello", true).unwrap();
        let ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
        let prompt = Array::from_slice(&ids, &[ids.len() as i32]);

        let stop = StopCriteria::from_config(&cfg, 8);
        let params = SamplingParams::default();
        let gen = Generate::new(&mut model, &cfg, prompt, stop, params);

        let mut tokens = Vec::new();
        for r in gen {
            tokens.push(r.expect("generation step"));
        }
        assert!(!tokens.is_empty(), "no tokens generated");
        assert!(tokens.len() <= 8, "produced more than max_new_tokens");
        for t in &tokens {
            assert!((*t as i32) < cfg.text_config.vocab_size, "OOV token {t}");
        }
        eprintln!("greedy tokens: {tokens:?}");
    }
}
