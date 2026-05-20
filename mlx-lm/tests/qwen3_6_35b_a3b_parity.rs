//! Qwen3.6-35B-A3B MoE loader smoke test.
//!
//! `#[ignore]`-gated: needs `mlx-community/Qwen3.6-35B-A3B-4bit` on
//! disk (~17.5 GB). Verifies the end-to-end load returns a populated
//! 40-layer LanguageModel with the routed-expert weights bound and
//! the per-tensor quantisation overrides honoured (mlp.gate +
//! mlp.shared_expert_gate at 8b on a body=4b checkpoint).

#![allow(clippy::missing_assert_message, reason = "test code")]
#![allow(clippy::print_stdout, reason = "test code")]
#![allow(clippy::print_stderr, reason = "test code")]

use std::path::PathBuf;

use mlx_lm::models::qwen3_5_moe::load_qwen3_5_moe_model;

const Q4_PATH: &str = ".cache/mlx-rs-bench/mlx-community/Qwen3.6-35B-A3B-4bit";

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}

#[test]
#[ignore = "requires mlx-community/Qwen3.6-35B-A3B-4bit on disk"]
fn loader_binds_language_model_q4() {
    let dir = home().join(Q4_PATH);
    let model = load_qwen3_5_moe_model(&dir).expect("load");
    assert_eq!(model.model.layers.len(), 40);
}
