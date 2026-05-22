//! Family-agnostic helpers shared by all model-family adapters.
//!
//! [`LoadedContext`] is the shape every family's `load_context_*`
//! returns: the boxed model + processor + EOS-id list. The crate-root
//! `mlxr_lm::load` wraps it into a [`crate::ModelContext`].
//!
//! [`read_eos_ids`] reads `config.json::eos_token_id`, normalising
//! the `int | [int]` shape that Hugging Face configs use.

#[cfg(any(feature = "qwen3_5", feature = "gemma4"))]
use std::path::Path;

#[cfg(any(feature = "qwen3_5", feature = "gemma4"))]
use serde::Deserialize;

use crate::language_model::{LanguageModel, UserInputProcessor};

pub(crate) type LoadedContext = (
    Box<dyn LanguageModel>,
    Box<dyn UserInputProcessor>,
    Vec<u32>,
);

#[cfg(any(feature = "qwen3_5", feature = "gemma4"))]
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EosSpec {
    Single(u32),
    Many(Vec<u32>),
}

/// Read `config.json::eos_token_id` from `dir`. Empty vec on a
/// missing field; the caller may still apply a family default.
#[cfg(any(feature = "qwen3_5", feature = "gemma4"))]
pub(crate) fn read_eos_ids(dir: &Path) -> Vec<u32> {
    let Ok(raw) = std::fs::read_to_string(dir.join("config.json")) else {
        return Vec::new();
    };
    let Ok(val) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return Vec::new();
    };
    val.get("eos_token_id")
        .and_then(|v| serde_json::from_value::<EosSpec>(v.clone()).ok())
        .map(|spec| match spec {
            EosSpec::Single(id) => vec![id],
            EosSpec::Many(ids) => ids,
        })
        .unwrap_or_default()
}
