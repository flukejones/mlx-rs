//! Qwen3.6-27B (qwen3_5 architecture) loader smoke + parity tests.
//!
//! All tests `#[ignore]` by default — they need the checkpoint on disk.

#![allow(clippy::missing_assert_message, reason = "test code")]
#![allow(clippy::print_stdout, reason = "test code")]
#![allow(clippy::print_stderr, reason = "test code")]

use std::path::PathBuf;

use mlx_lm::models::qwen3_5::{config::ModelConfig, weights::load_language_model};

const Q4_PATH: &str = ".cache/mlx-rs-bench/mlx-community/Qwen3.6-27B-4bit";

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}

#[test]
#[ignore = "requires mlx-community/Qwen3.6-27B-4bit on disk"]
fn qwen3_6_27b_loader_binds_language_model() {
    let dir = home().join(Q4_PATH);
    let cfg = ModelConfig::from_file(dir.join("config.json")).expect("parse config");
    assert_eq!(cfg.text_config.num_hidden_layers, 64);
    assert_eq!(cfg.text_config.hidden_size, 5120);
    assert_eq!(cfg.text_config.linear_num_value_heads, 48);

    let (model, leftover) = load_language_model(&cfg, &dir).expect("load");
    assert_eq!(model.model.layers.len(), 64);

    // Vision-tower keys are bucketed into leftover by load_language_model;
    // anything else is a real binding miss.
    let lm_miss: Vec<&String> = leftover
        .iter()
        .filter(|k| !k.starts_with("vision_tower."))
        .collect();
    assert!(
        lm_miss.is_empty(),
        "unbound LM keys: {lm_miss:?} (first 8 of {} total leftovers)",
        leftover.len()
    );
}
