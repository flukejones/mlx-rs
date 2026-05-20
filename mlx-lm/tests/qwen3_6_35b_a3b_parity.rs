//! Qwen3.6-35B-A3B MoE smoke test against the unified surface.
//!
//! `#[ignore]`-gated: needs `mlx-community/Qwen3.6-35B-A3B-4bit` on
//! disk (~17.5 GB). Drives the load + a short generation through
//! `mlx_lm::load` + `mlx_lm::generate`; passing means the
//! per-tensor quantisation overrides (mlp.gate +
//! mlp.shared_expert_gate at 8b on a body=4b checkpoint) bound
//! correctly *and* end-to-end inference produces tokens.

#![allow(clippy::missing_assert_message, reason = "test code")]

use std::path::PathBuf;

use mlx_lm::{generate, load, GenerateParams, UserInput};

const Q4_PATH: &str = ".cache/mlx-rs-bench/mlx-community/Qwen3.6-35B-A3B-4bit";

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}

#[test]
#[ignore = "requires mlx-community/Qwen3.6-35B-A3B-4bit on disk"]
fn loader_and_generate_q4() {
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
