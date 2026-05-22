//! Language-model runtime and model families on top of `mlxr`.
//!
//! Always compiled: the runtime layer (tokenizer + chat template +
//! sampler + KV-cache + attention kernels + the family-agnostic
//! [`LanguageModel`] / [`ModelContext`] surface).
//!
//! Feature-gated: model families and modalities.
//!
//! - `qwen3_5` — Qwen 3.5 / 3.6 family (dense + MoE).
//! - `gemma4` — Gemma 4 family.
//! - `image` — image modality: `UserInput::images`, [`Image`], and
//!   per-family vision towers (e.g. `qwen3_5::image::*`).
//! - `audio` / `video` — reserved for future families.
//!
//! Default features enable every family this fork ships plus image.

// Always-compiled runtime layer.
pub mod activations;
pub mod attention;
pub mod cache;
pub mod chat_template;
pub mod error;
pub(crate) mod family;
pub mod language_model;
pub mod lm_input;
pub mod loader;
pub mod model_context;
pub mod nn;
pub mod prelude;
pub mod quantization;
pub mod sampler;
pub mod user_input;
pub mod utils;

// Feature-gated model families.
#[cfg(feature = "gemma4")]
pub mod gemma4;
#[cfg(feature = "qwen3_5")]
pub mod qwen3_5;

pub use language_model::{LanguageModel, TextOnlyProcessor, UserInputProcessor};
pub use lm_input::{LMInput, LMOutput, PrepareResult, ProcessedAudio, ProcessedImage, Text};
pub use model_context::{
    generate, load, FinishReason, GenerateParams, GenerateResult, ModelContext, TokenCallback,
};
pub use sampler::{SamplerState, SamplingParams};
#[cfg(feature = "image")]
pub use user_input::Image;
pub use user_input::{Audio, Prompt, UserInput, Video};
