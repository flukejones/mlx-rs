//! Model-facing input.
//!
//! [`crate::UserInputProcessor`] consumes a [`crate::UserInput`] and
//! produces an [`LMInput`]. The model's
//! [`crate::LanguageModel::prepare`] consumes the `LMInput` to seed
//! its KV cache; subsequent [`crate::LanguageModel::step`] calls
//! consume only token ids ([`Text`]) one at a time.
//!
//! Shape mirrors `mlx-swift-lm`'s `LMInput`. Modality fields are
//! `Option`s with `None` meaning "the user didn't supply this
//! modality" and `Some` meaning "the processor preprocessed it and
//! the model is expected to consume it".

use mlx_rs::Array;

/// Output of a [`crate::UserInputProcessor::prepare`] call.
///
/// Every modality slot is independent: a VLM with both image and
/// audio sets `image` and `audio` to `Some`. A text-only request
/// leaves both `None` and the model takes the text-only path.
#[derive(Debug)]
pub struct LMInput {
    /// The tokenised prompt + optional attention mask.
    pub text: Text,

    /// Pre-processed image tensor(s) for the vision tower. `None`
    /// for text-only requests or for models that don't accept image
    /// input.
    pub image: Option<ProcessedImage>,

    /// Pre-processed audio features for the audio tower. `None` for
    /// text-only requests or for models that don't accept audio.
    pub audio: Option<ProcessedAudio>,

    /// Pre-processed video frames. Reserved; currently always `None`.
    pub video: Option<ProcessedVideo>,
}

/// Tokenised text portion of an [`LMInput`]. Same shape as
/// `mlx-swift-lm` `LMInput.Text`.
#[derive(Debug)]
pub struct Text {
    /// `[1, S]` int32 token ids. Batch dim is always 1 (no batched
    /// inference today; the column the model reads is dim 1).
    pub tokens: Array,

    /// Optional `[1, S]` attention mask. `None` lets the model
    /// build its own (causal mask + KV-cache-aware padding).
    pub mask: Option<Array>,
}

/// Pre-processed image tensor(s), ready for the model's vision
/// tower. Layout matches what the tower's `forward` expects — the
/// processor handles per-family normalisation, patch packing, and
/// the temporal/height/width grid metadata.
#[derive(Debug)]
pub struct ProcessedImage {
    /// `[num_patches, feature_dim]` `f32` array. Patches are stacked
    /// across all images in the prompt; `grids` records the per-image
    /// `(t, h, w)` so the model can slice them back apart.
    pub pixels: Array,

    /// One `[t, h, w]` patch-grid per image in the original prompt
    /// (same order as the `UserInput::images` vec).
    pub grids: Vec<[i32; 3]>,
}

/// Pre-processed audio feature tensor, ready for the model's audio
/// tower. No in-tree model consumes this yet; the type is fixed now
/// so the gemma4 audio branch plugs in without breaking the
/// [`LMInput`] surface.
#[derive(Debug)]
pub struct ProcessedAudio {
    /// e.g. `[1, mel_bins, frames]` log-mel spectrogram `f32` array.
    /// Concrete layout is the audio tower's responsibility.
    pub features: Array,

    /// Sample rate the features were extracted at (Hz).
    pub sample_rate: u32,
}

/// Pre-processed video frame tensor. Reserved.
#[derive(Debug)]
pub struct ProcessedVideo {
    /// `[num_frames, channels, height, width]` or similar — exact
    /// layout will be defined when a video-capable family lands.
    pub frames: Array,
}

/// Result of [`crate::LanguageModel::prepare`]: either logits the
/// caller can sample immediately (whole prompt already processed)
/// or "the model consumed the prompt, call `step` to produce
/// tokens" (the normal path).
pub enum PrepareResult {
    /// Prompt has been ingested and the KV cache primed. The next
    /// call to [`crate::LanguageModel::step`] produces the first
    /// generated token.
    Primed,

    /// Prompt was short enough that the model returned the
    /// next-token logits directly. The caller samples from these
    /// and feeds the result into the next `step`.
    Logits(Array),
}

/// One step's output from [`crate::LanguageModel::step`].
pub struct LMOutput {
    /// `[1, 1, vocab_size]` logits over the next token.
    pub logits: Array,
}
