//! Qwen3.6-27B (qwen3_5 architecture) smoke test against the
//! unified `mlx_lm::load` surface.
//!
//! `#[ignore]`-gated — needs the checkpoint on disk.

#![allow(clippy::missing_assert_message, reason = "test code")]

use std::path::PathBuf;

use mlx_lm::{generate, load, GenerateParams, UserInput};

const Q4_PATH: &str = ".cache/mlx-rs-bench/mlx-community/Qwen3.6-27B-4bit";

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}

#[test]
#[ignore = "requires mlx-community/Qwen3.6-27B-4bit on disk"]
fn qwen3_6_27b_loads_and_generates() {
    let dir = home().join(Q4_PATH);
    let mut ctx = load(&dir).expect("load");
    let input = UserInput::text("Hello");
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
