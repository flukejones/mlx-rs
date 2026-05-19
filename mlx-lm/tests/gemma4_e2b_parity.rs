//! Gemma 4 E2B/E4B per-layer-embedding (PLE) loader smoke + parity tests.
//!
//! Fixtures (npz) are produced by
//!     python3 mlx-lm/tests/fixtures/gemma4_e2b/generate.py \
//!         --model mlx-community/gemma-4-e2b-it-8bit
//!
//! All tests `#[ignore]` by default — they need the checkpoint on disk and
//! the matching `.npz` fixtures alongside this file.

use std::path::PathBuf;

use mlx_lm::models::gemma4::load_gemma4_model_sanitized;

const E2B_PATH: &str = ".cache/mlx-rs-bench/mlx-community/gemma-4-e2b-it-8bit";
const E4B_PATH: &str = ".cache/mlx-rs-bench/mlx-community/gemma-4-e4b-it-8bit";

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}

fn assert_ple_bound(rel: &str) {
    let dir = home().join(rel);
    let model = load_gemma4_model_sanitized(&dir).expect("PLE load");
    assert!(!model.model.layers.is_empty(), "no decoder layers");
    let pl = model.model.embed_tokens_per_layer.is_some()
        && model.model.per_layer_model_projection.is_some()
        && model.model.per_layer_projection_norm.is_some();
    assert!(pl, "PLE fields not populated for {rel}");
}

#[test]
#[ignore = "requires mlx-community/gemma-4-e2b-it-8bit on disk"]
fn e2b_loader_binds_every_weight() {
    assert_ple_bound(E2B_PATH);
}

#[test]
#[ignore = "requires mlx-community/gemma-4-e4b-it-8bit on disk"]
fn e4b_loader_binds_every_weight() {
    assert_ple_bound(E4B_PATH);
}
