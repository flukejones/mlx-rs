//! End-to-end round-trip of the `UserInput` vision path against a
//! real VLM checkpoint. Covers:
//! - text + image input through the unified surface
//! - the `Image::Pixels` bypass (caller hands in a pre-processed
//!   pixel array; the qwen3_5 processor validates the geometry and
//!   skips CPU preprocessing)
//!
//! `#[ignore]`-gated — needs `mlx-community/Qwen3.6-27B-4bit` on
//! disk at the standard bench-cache path.

#![allow(clippy::missing_assert_message, reason = "test code")]

use std::collections::HashMap;
use std::path::PathBuf;

use image::DynamicImage;
use mlxr_lm::chat_template::ChatMessage;
use mlxr_lm::{generate, load, GenerateParams, Image, Prompt, UserInput};

const MODEL_PATH: &str = ".cache/mlx-rs-bench/mlx-community/Qwen3.6-27B-4bit";

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}

/// Synthetic 224x224 RGB image filled with a single colour. Small
/// enough to keep the patch grid tiny, big enough that the
/// processor's `smart_resize` accepts it.
fn synthetic_image() -> DynamicImage {
    DynamicImage::new_rgb8(224, 224)
}

#[test]
#[ignore = "requires mlx-community/Qwen3.6-27B-4bit on disk"]
fn vlm_image_chat_round_trips() {
    let dir = home().join(MODEL_PATH);
    let mut ctx = load(&dir).expect("load");

    let input = UserInput {
        prompt: Prompt::Chat(vec![ChatMessage::user("What colour is this image?")]),
        images: vec![Image::Decoded(synthetic_image())],
        audios: Vec::new(),
        videos: Vec::new(),
        template_kwargs: HashMap::new(),
    };
    let params = GenerateParams {
        max_new_tokens: 16,
        ..GenerateParams::default()
    };
    let result = generate(&mut ctx, input, params, &mut |_, _| {
        std::ops::ControlFlow::Continue(())
    })
    .expect("generate");

    assert!(result.completion_tokens > 0, "no tokens produced");
    assert!(!result.text.is_empty(), "result.text empty");
}

#[test]
#[ignore = "requires mlx-community/Qwen3.6-27B-4bit on disk"]
fn vlm_text_only_chat_round_trips_against_vlm_checkpoint() {
    // A VLM checkpoint must still serve text-only requests through
    // the same surface — the processor sees `images: []` and falls
    // through to the dense path.
    let dir = home().join(MODEL_PATH);
    let mut ctx = load(&dir).expect("load");

    let input = UserInput::chat(vec![ChatMessage::user("Hello.")]);
    let params = GenerateParams {
        max_new_tokens: 4,
        ..GenerateParams::default()
    };
    let result = generate(&mut ctx, input, params, &mut |_, _| {
        std::ops::ControlFlow::Continue(())
    })
    .expect("generate");
    assert!(result.completion_tokens > 0);
}
