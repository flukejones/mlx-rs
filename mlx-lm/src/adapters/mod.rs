//! Per-family [`crate::LanguageModel`] adapters.
//!
//! Each family's concrete model type ([`crate::models::gemma4::Model`],
//! [`crate::models::qwen3_5::LanguageModel`], etc.) plus its concrete
//! cache type get wrapped in an adapter struct in this module. The
//! adapter implements [`crate::LanguageModel`] so the rest of the
//! crate can dispatch through `Box<dyn LanguageModel>`.
//!
//! Each adapter is the *only* caller of its family's loader and the
//! *only* driver of its family's prefill/decode primitives.

pub(crate) mod gemma4;
pub(crate) mod qwen3_5;
pub(crate) mod qwen3_5_moe;
pub(crate) mod qwen3_5_vlm;

use std::path::Path;

use serde::Deserialize;

use crate::language_model::{LanguageModel, UserInputProcessor};

/// Per-family adapter payload returned by `load_context`: the boxed
/// model, its processor, and the EOS-id list. The crate-root
/// `mlx_lm::load` wraps these into a [`crate::ModelContext`].
pub(crate) type LoadedContext = (
    Box<dyn LanguageModel>,
    Box<dyn UserInputProcessor>,
    Vec<u32>,
);

/// EOS-token spec accepted by `config.json`: either a single int or
/// an array of ints.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EosSpec {
    Single(u32),
    Many(Vec<u32>),
}

/// Read `config.json::eos_token_id` from `dir`, returning the EOS
/// list. Empty vec on a missing field (caller may still apply a
/// family-default).
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
