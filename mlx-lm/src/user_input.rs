//! Caller-facing input to [`crate::generate`].
//!
//! Shape mirrors `mlx-swift-lm`'s `UserInput`: one struct carries the
//! prompt (text or chat) plus optional images, audios, and videos. The
//! same struct flows into every model — the model's
//! [`crate::UserInputProcessor`] decides which modalities it accepts.
//! Text-only models reject populated `images`/`audios` with
//! [`crate::error::Error::ModalityUnsupported`].
//!
//! Each modality enum exposes both a "decoded" arm (the helper crate
//! does the preprocessing) and a "pre-processed" arm (caller hands in
//! a tensor that's already in the model's expected layout). The
//! pre-processed arm lets a server avoid double work when the upstream
//! has already resized / normalised the input.

use std::collections::HashMap;
use std::path::PathBuf;

#[cfg(feature = "models-vision")]
use image::DynamicImage;
use mlx_rs::Array;

use crate::chat_template::ChatMessage;

/// Top-level user-facing input. Constructed by consumers
/// (`examples/lm/src/bin/{chat,chat_server}`, library users) and
/// handed to [`crate::generate`]. Every field except `prompt` is
/// optional; an unset field means the modality is absent.
pub struct UserInput {
    /// What the user said. Plain string or structured chat.
    pub prompt: Prompt,

    /// Zero or more images attached to the conversation. Order is
    /// preserved; the chat-template `<image>` slots consume them in
    /// order. Always empty for text-only models.
    #[cfg(feature = "models-vision")]
    pub images: Vec<Image>,

    /// Zero or more audio clips attached to the conversation. Always
    /// empty for non-audio models.
    pub audios: Vec<Audio>,

    /// Zero or more video clips. Reserved for future families; current
    /// processors reject populated `videos`.
    pub videos: Vec<Video>,

    /// Named values forwarded to the chat-template render. Empty by
    /// default. Used by templates that gate on a kwarg — qwen 3.6
    /// reads `enable_thinking` to decide whether to inject `<think>\n`
    /// or `<think>\n\n</think>\n\n` at the assistant turn start.
    pub template_kwargs: HashMap<String, serde_json::Value>,
}

/// Conversation shape. Plain text is the fast path for one-shot
/// completion; `Chat` carries the full structured history that the
/// model's Jinja template renders.
pub enum Prompt {
    /// Single-string prompt, fed verbatim to the tokenizer with no
    /// chat-template render.
    Text(String),

    /// Structured conversation. Rendered through the model's
    /// `tokenizer_config.json` / `chat_template.jinja` template by
    /// the [`crate::UserInputProcessor`].
    Chat(Vec<ChatMessage>),
}

impl UserInput {
    /// Build from a plain-text prompt with no modalities.
    pub fn text(prompt: impl Into<String>) -> Self {
        Self {
            prompt: Prompt::Text(prompt.into()),
            #[cfg(feature = "models-vision")]
            images: Vec::new(),
            audios: Vec::new(),
            videos: Vec::new(),
            template_kwargs: HashMap::new(),
        }
    }

    /// Build from a structured chat conversation with no modalities.
    pub fn chat(messages: Vec<ChatMessage>) -> Self {
        Self {
            prompt: Prompt::Chat(messages),
            #[cfg(feature = "models-vision")]
            images: Vec::new(),
            audios: Vec::new(),
            videos: Vec::new(),
            template_kwargs: HashMap::new(),
        }
    }

    /// Set a single template kwarg by name. Builder-style for
    /// ergonomics: `UserInput::chat(msgs).with_template_kwarg(
    /// "enable_thinking", true.into())`.
    #[must_use]
    pub fn with_template_kwarg(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.template_kwargs.insert(key.into(), value);
        self
    }
}

/// One image attached to a [`UserInput`].
///
/// Two arms cover the trade-off between convenience and avoiding
/// double work:
/// - [`Image::Decoded`] holds a CPU-decoded `DynamicImage`; the
///   processor runs full preprocessing (resize → normalise → patch
///   into the tensor layout the vision tower expects).
/// - [`Image::Pixels`] holds an already-preprocessed pixel array and
///   its `grid_thw` tuple; the processor validates the geometry and
///   feeds it straight to the vision tower. Use this when the
///   upstream (e.g. an HTTP server holding a pool of preprocessed
///   tensors) has already paid the preprocessing cost.
#[cfg(feature = "models-vision")]
pub enum Image {
    /// Raw decoded image. Processor will resize + normalise + pack.
    Decoded(DynamicImage),

    /// Already in the model's pixel-array layout, with `grid_thw`
    /// `[temporal, height_patches, width_patches]` describing the
    /// patch grid. Skips the CPU preprocessing path.
    Pixels {
        /// `[num_patches, feature_dim]` `f32` array as produced by
        /// the qwen3_5 image processor.
        array: Array,

        /// `[t, h, w]` patch counts. Caller must guarantee this
        /// matches the grid the processor would have produced for
        /// the same source image; the processor cross-checks the
        /// product against `array.shape[0]` and errors on mismatch.
        grid: [i32; 3],
    },
}

/// One audio clip attached to a [`UserInput`].
///
/// First-class today even though no in-tree model consumes audio yet
/// — the gemma4 audio tower lands on the next branch and slots in
/// through the existing `UserInputProcessor` trait. Mirrors the
/// `Image` enum's preprocessing-bypass shape:
/// - [`Audio::Wav`] decodes the file from disk.
/// - [`Audio::Pcm`] is already-decoded PCM samples (skip file IO).
/// - [`Audio::Features`] is an already-computed feature tensor
///   (e.g. log-mel spectrogram), skipping both file IO and DSP.
pub enum Audio {
    /// On-disk wav (or any format the audio-decoding helper crate
    /// supports). Reads + decodes when the processor runs.
    Wav(PathBuf),

    /// Decoded PCM samples. `sample_rate` matters: the processor
    /// resamples to the model's expected rate (typically 16 kHz).
    Pcm {
        /// Mono PCM samples in the range `[-1.0, 1.0]`.
        samples: Vec<f32>,

        /// Hertz.
        sample_rate: u32,
    },

    /// Already-computed feature tensor in the model's expected
    /// layout (e.g. a `[1, mel_bins, frames]` log-mel spectrogram).
    /// Skips the helper crate entirely.
    Features(Array),
}

/// Placeholder for video input. No model currently consumes video;
/// processors reject populated `videos` so the surface is stable when
/// a video-capable family lands.
pub enum Video {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_input_constructs() {
        let input = UserInput::text("hi");
        assert!(matches!(input.prompt, Prompt::Text(ref s) if s == "hi"));
        #[cfg(feature = "models-vision")]
        assert!(input.images.is_empty());
        assert!(input.audios.is_empty());
        assert!(input.videos.is_empty());
    }

    #[test]
    fn chat_input_constructs() {
        let input = UserInput::chat(vec![
            ChatMessage::user("hello"),
            ChatMessage::assistant("hi"),
        ]);
        let Prompt::Chat(ref msgs) = input.prompt else {
            panic!("expected Chat prompt");
        };
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[1].role, "assistant");
    }

    #[test]
    fn audio_pcm_carries_sample_rate() {
        let audio = Audio::Pcm {
            samples: vec![0.0; 16_000],
            sample_rate: 16_000,
        };
        let Audio::Pcm { sample_rate, .. } = audio else {
            panic!("expected Pcm");
        };
        assert_eq!(sample_rate, 16_000);
    }

    #[test]
    #[cfg(feature = "models-vision")]
    fn image_decoded_constructs() {
        // 1x1 RGB image as the lightest possible construction.
        let img = DynamicImage::new_rgb8(1, 1);
        let image = Image::Decoded(img);
        assert!(matches!(image, Image::Decoded(_)));
    }
}
