//! Qwen3.6-35B-A3B MoE smoke test against the unified surface.
//!
//! `#[ignore]`-gated: needs `lmstudio-community/Qwen3.6-35B-A3B-MLX-8bit`
//! on disk (~35 GB). Drives the load + a short generation through
//! `mlx_lm::load` + `mlx_lm::generate`; passing means the
//! per-tensor quantisation overrides bound correctly *and* end-to-end
//! inference produces tokens.

#![allow(clippy::missing_assert_message, reason = "test code")]

use std::path::PathBuf;

use mlx_lm::{generate, load, GenerateParams, UserInput};

const Q8_PATH: &str = ".cache/mlx-rs-bench/lmstudio-community/Qwen3.6-35B-A3B-MLX-8bit";

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}

#[test]
#[ignore = "requires lmstudio-community/Qwen3.6-35B-A3B-MLX-8bit on disk"]
fn loader_and_generate_q8() {
    let dir = home().join(Q8_PATH);
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
