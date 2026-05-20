//! Crate-root entry points: [`load`] + [`generate`].
//!
//! All consumers go through these two functions. [`load`] reads
//! `config.json::model_type`, dispatches to the per-family adapter,
//! and returns a [`ModelContext`] that owns the model + its
//! processor + the EOS ids. [`generate`] runs the full
//! prepare → sample → step loop with optional per-token streaming.

use std::ops::ControlFlow;
use std::path::Path;

use mlx_rs::{ops::indexing::IndexOp, Array};

use crate::error::Error;
use crate::language_model::{LanguageModel, UserInputProcessor};
use crate::lm_input::{LMInput, PrepareResult, Text};
use crate::sampler::{sample_with, SamplingParams};
use crate::user_input::UserInput;

/// Sampling + stopping knobs handed to [`generate`].
#[derive(Debug, Clone)]
pub struct GenerateParams {
    /// Maximum new tokens to produce (excluding the prompt). The
    /// loop exits early when an EOS token is sampled.
    pub max_new_tokens: i32,

    /// Sampling parameters (temperature + optional top-p).
    pub sampling: SamplingParams,

    /// Stop tokens beyond the model-default EOS list. Empty by
    /// default; useful for stop-on-newline or stop-on-`</answer>`
    /// style early exits.
    pub extra_stop_ids: Vec<u32>,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            sampling: SamplingParams::default(),
            extra_stop_ids: Vec::new(),
        }
    }
}

/// Reason a [`generate`] call returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// An EOS token (or a caller-supplied `extra_stop_id`) was
    /// sampled.
    Stop,
    /// `max_new_tokens` was reached without an EOS hit, or the
    /// streaming callback returned `ControlFlow::Break`.
    Length,
}

/// Output of one [`generate`] call.
#[derive(Debug, Clone)]
pub struct GenerateResult {
    /// Decoded UTF-8 text of all generated tokens.
    pub text: String,
    /// Number of tokens in the rendered prompt.
    pub prompt_tokens: i32,
    /// Number of tokens generated (excluding the prompt).
    pub completion_tokens: i32,
    /// Why the loop exited.
    pub finish_reason: FinishReason,
}

/// Per-token streaming callback. Each accepted token id is decoded
/// against the running prefix to produce a UTF-8 delta (BPE merges
/// only stabilise on a full re-decode, so the delta is computed as
/// `decode(accumulated) - decoded_so_far`).
pub type TokenCallback<'cb> = dyn FnMut(u32, &str) -> ControlFlow<()> + 'cb;

/// The loaded model + its preprocessor + the stop-token list. Built
/// by [`load`]; consumed by [`generate`].
pub struct ModelContext {
    /// The boxed language model. Owns its KV cache; [`generate`]
    /// calls `reset()` at the start of each request.
    pub model: Box<dyn LanguageModel>,

    /// The boxed input processor (tokenisation, chat template,
    /// modality handling).
    pub processor: Box<dyn UserInputProcessor>,

    /// EOS token ids the model considers terminal. Loaded from
    /// `config.json::eos_token_id` plus any family default.
    pub eos_ids: Vec<u32>,
}

/// Detect the model family from `<dir>/config.json::model_type` and
/// build the matching [`ModelContext`].
pub fn load(dir: impl AsRef<Path>) -> Result<ModelContext, Error> {
    let dir = dir.as_ref();
    let model_type = detect_model_type(dir)?;
    let (model, processor, eos_ids) = dispatch_load(model_type.as_str(), dir)?;
    Ok(ModelContext {
        model,
        processor,
        eos_ids,
    })
}

/// Run one prompt → tokens loop on `ctx`. Streaming is per-token via
/// `on_token`; pass `&mut |_, _| ControlFlow::Continue(())` to
/// disable streaming.
pub fn generate(
    ctx: &mut ModelContext,
    input: UserInput,
    params: GenerateParams,
    on_token: &mut TokenCallback<'_>,
) -> Result<GenerateResult, Error> {
    ctx.model.reset();

    let lm_input = ctx.processor.prepare(input)?;
    let prompt_tokens = lm_input.text.tokens.shape()[1];

    let initial_logits = run_prefill(ctx.model.as_mut(), lm_input)?;

    let vocab = ctx.model.vocab_size();
    let mut produced: Vec<u32> = Vec::new();
    let mut decoded_prefix = String::new();
    let mut finish_reason = FinishReason::Length;

    let mut next_logits = initial_logits;
    for _ in 0..params.max_new_tokens {
        let token = sample_token(&next_logits, &params.sampling, vocab)?;

        if ctx.eos_ids.contains(&token) || params.extra_stop_ids.contains(&token) {
            finish_reason = FinishReason::Stop;
            break;
        }
        produced.push(token);

        // BPE-incremental decode: re-decode the full id list, take
        // the suffix beyond what we already streamed.
        let full = ctx.processor.decode(&produced)?;
        let delta = full
            .strip_prefix(decoded_prefix.as_str())
            .unwrap_or(full.as_str())
            .to_owned();
        decoded_prefix = full;
        if matches!(on_token(token, &delta), ControlFlow::Break(())) {
            break;
        }

        next_logits = ctx.model.step(token as i32)?.logits;
    }

    Ok(GenerateResult {
        text: decoded_prefix,
        prompt_tokens,
        completion_tokens: produced.len() as i32,
        finish_reason,
    })
}

/// Run the prefill phase. When the model exposes a non-`None`
/// [`LanguageModel::prefill_chunk_size`] and the prompt exceeds it
/// (gemma4's sliding cache), feed all but the trailing chunk
/// through [`LanguageModel::prefill_chunk`], then drive the tail
/// through the normal [`LanguageModel::prepare`] to get the
/// first-step logits. Multimodal inputs (`image`/`audio`/`video`
/// set) bypass chunking — they go straight to `prepare`, which the
/// VLM adapter handles with a single stitched forward pass.
fn run_prefill(
    model: &mut dyn LanguageModel,
    mut input: LMInput,
) -> Result<Array, Error> {
    let chunk_size = model.prefill_chunk_size();
    let is_multimodal = input.image.is_some() || input.audio.is_some() || input.video.is_some();
    let prompt_len = input.text.tokens.shape()[1];

    if let Some(window) = chunk_size {
        if !is_multimodal && prompt_len > window {
            // Feed every chunk except the last through prefill_chunk;
            // the tail (≤ window tokens) goes through prepare so its
            // logits seed the sampler.
            let tokens = input.text.tokens;
            let mut start = 0_i32;
            while prompt_len - start > window {
                let end = start + window;
                let chunk = tokens.index((.., start..end));
                model.prefill_chunk(&chunk)?;
                start = end;
            }
            let tail = tokens.index((.., start..prompt_len));
            input = LMInput {
                text: Text {
                    tokens: tail,
                    mask: None,
                },
                image: None,
                audio: None,
                video: None,
            };
        }
    }

    match model.prepare(input)? {
        PrepareResult::Logits(arr) => Ok(arr),
        PrepareResult::Primed => {
            // No prefill logits returned: drive one step against the
            // sentinel id 0 to get the first usable distribution.
            Ok(model.step(0)?.logits)
        }
    }
}

/// Sample one token id from a logits row, with an OOV check that
/// surfaces a typed error if the sampler hands back an out-of-vocab
/// id (which would corrupt the streaming decode).
fn sample_token(logits: &Array, params: &SamplingParams, vocab: i32) -> Result<u32, Error> {
    let id_arr = sample_with(logits, params)?;
    let id = id_arr.item::<i32>();
    if id < 0 || id >= vocab {
        return Err(Error::Shape(format!(
            "sampler returned out-of-vocab id {id} (vocab = {vocab})"
        )));
    }
    Ok(id as u32)
}

/// Read the top-level `model_type` field from `<dir>/config.json`.
fn detect_model_type(dir: &Path) -> Result<String, Error> {
    let path = dir.join("config.json");
    let raw = std::fs::read_to_string(&path)?;
    let val: serde_json::Value = serde_json::from_str(&raw)?;
    val.get("model_type")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or_else(|| {
            Error::Other(format!("{}: missing `model_type` field", path.display()).into())
        })
}

/// Match `model_type` against the supported families and route to
/// the matching adapter's `load_context`. Single source of truth for
/// the family-to-adapter mapping.
fn dispatch_load(model_type: &str, dir: &Path) -> Result<crate::adapters::LoadedContext, Error> {
    match model_type {
        // Llama / Llama-3 / TinyLlama family.
        "llama" => crate::adapters::llama::load_context(dir),

        // Qwen3 dense (1.7B, 8B, etc.).
        "qwen3" => crate::adapters::qwen3::load_context(dir),

        // Gemma 4 family (text + MoE, no vision tower in this crate).
        "gemma4" | "gemma4_text" | "gemma4textmodel" | "gemma4forcausallm" => {
            crate::adapters::gemma4::load_context(dir)
        }

        // Qwen3.5 / Qwen3.6 dense + VL. The VLM probe inside
        // `qwen3_5_vlm::load_context` looks at
        // `preprocessor_config.json` to decide dense vs VLM.
        "qwen3_5" | "qwen3_5_text" | "qwen3_5forconditionalgeneration" => {
            crate::adapters::qwen3_5_vlm::load_context(dir)
        }

        // Qwen3.5-MoE / Qwen3.6-MoE (35B-A3B and friends).
        "qwen3_5_moe" | "qwen3_5_moe_text" => crate::adapters::qwen3_5_moe::load_context(dir),

        other => Err(Error::Other(
            format!("mlx_lm::load: unsupported model_type {other:?}").into(),
        )),
    }
}
