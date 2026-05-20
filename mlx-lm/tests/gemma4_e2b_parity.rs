//! Gemma 4 E2B / E4B smoke against the unified `mlx_lm::load` +
//! `mlx_lm::generate` surface.
//!
//! All tests `#[ignore]` by default — they need the checkpoint on
//! disk. Loading covers the per-layer-embedding (PLE) binding path;
//! generation covers the steel-prefill / sliding-cache paths.

#![allow(clippy::missing_assert_message, reason = "test code")]

use std::path::PathBuf;

use mlx_lm::{generate, load, GenerateParams, UserInput};

const E2B_PATH: &str = ".cache/mlx-rs-bench/mlx-community/gemma-4-e2b-it-8bit";
const E4B_PATH: &str = ".cache/mlx-rs-bench/mlx-community/gemma-4-e4b-it-8bit";

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}

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
#[ignore = "requires mlx-community/gemma-4-e2b-it-8bit on disk"]
fn e2b_loads_and_generates() {
    smoke(E2B_PATH);
}

#[test]
#[ignore = "requires mlx-community/gemma-4-e4b-it-8bit on disk"]
fn e4b_loads_and_generates() {
    smoke(E4B_PATH);
}
