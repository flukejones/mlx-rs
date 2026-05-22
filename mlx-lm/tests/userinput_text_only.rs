//! End-to-end round-trip of the unified `UserInput` → `LMInput` →
//! `LanguageModel` surface against a small text-only model.
//!
//! `#[ignore]`-gated — needs `mlx-community/Qwen3-1.7B-4bit` on disk
//! at the standard bench-cache path.

#![allow(clippy::missing_assert_message, reason = "test code")]

use std::collections::HashMap;
use std::path::PathBuf;

use mlx_lm::chat_template::ChatMessage;
use mlx_lm::{generate, load, GenerateParams, UserInput};

const MODEL_PATH: &str = ".cache/mlx-rs-bench/mlx-community/Qwen3-1.7B-4bit";

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}

#[test]
#[ignore = "requires mlx-community/Qwen3-1.7B-4bit on disk"]
fn text_only_chat_round_trips() {
    let dir = home().join(MODEL_PATH);
    let mut ctx = load(&dir).expect("load");

    let input = UserInput::chat(vec![ChatMessage::user("Say hello briefly.")]);
    let params = GenerateParams {
        max_new_tokens: 16,
        ..GenerateParams::default()
    };

    // Capture streaming deltas to verify they concatenate to the
    // final `text` field (BPE-incremental decoder must produce a
    // delta on every token whose UTF-8 prefix stabilises).
    let mut streamed = String::new();
    let mut token_count = 0_usize;
    let result = generate(&mut ctx, input, params, &mut |_, delta| {
        streamed.push_str(delta);
        token_count += 1;
        std::ops::ControlFlow::Continue(())
    })
    .expect("generate");

    assert!(result.completion_tokens > 0, "no tokens produced");
    assert!(!result.text.is_empty(), "result.text empty");
    assert_eq!(
        streamed.as_str(),
        result.text.as_str(),
        "streaming deltas didn't reassemble to result.text"
    );
    assert_eq!(token_count, result.completion_tokens as usize);
}

#[test]
#[ignore = "requires mlx-community/Qwen3-1.7B-4bit on disk"]
fn text_only_rejects_image_input() {
    // Build manually so we can attach an image without going via
    // UserInput::text/chat (those default images to empty).
    let dir = home().join(MODEL_PATH);
    let mut ctx = load(&dir).expect("load");
    let input = UserInput {
        prompt: mlx_lm::Prompt::Text("hi".into()),
        images: vec![mlx_lm::Image::Decoded(image::DynamicImage::new_rgb8(1, 1))],
        audios: Vec::new(),
        videos: Vec::new(),
        template_kwargs: HashMap::new(),
    };
    let params = GenerateParams {
        max_new_tokens: 4,
        ..GenerateParams::default()
    };
    let err = generate(&mut ctx, input, params, &mut |_, _| {
        std::ops::ControlFlow::Continue(())
    })
    .expect_err("text-only model should reject images");
    let msg = err.to_string();
    assert!(
        msg.contains("image"),
        "expected modality error mentioning image, got: {msg}"
    );
}
