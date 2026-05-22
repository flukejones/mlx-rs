//! Qwen3.6-27B (qwen3_5 architecture) smoke test against the
//! unified `mlxr_lm::load` surface, q4 checkpoint. `#[ignore]`-gated
//! — needs the matching checkpoint on disk under
//! `$HOME/.cache/mlx-rs-bench/`.

#![allow(clippy::missing_assert_message, reason = "test code")]

use std::path::PathBuf;

use mlxr_lm::{generate, load, GenerateParams, UserInput};

const Q4_PATH: &str = ".cache/mlx-rs-bench/mlx-community/Qwen3.6-27B-4bit";
const Q8_PATH: &str = ".cache/mlx-rs-bench/mlx-community/Qwen3.6-27B-8bit";

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}

/// Loads the checkpoint at `rel` through `mlxr_lm::load` and runs a
/// 4-token generation. Passing means: (1) the loader bound every
/// weight, (2) the VLM detection path picked the right adapter, and
/// (3) end-to-end inference produced tokens through the unified
/// surface.
fn smoke(rel: &str) {
    let dir = home().join(rel);
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
    assert!(result.completion_tokens > 0, "no tokens for {rel}");
}

#[test]
#[ignore = "requires mlx-community/Qwen3.6-27B-4bit on disk"]
fn qwen3_6_27b_q4_loads_and_generates() {
    smoke(Q4_PATH);
}

#[test]
#[ignore = "requires mlx-community/Qwen3.6-27B-8bit on disk"]
fn qwen3_6_27b_q8_loads_and_generates() {
    smoke(Q8_PATH);
}
