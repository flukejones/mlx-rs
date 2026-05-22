//! Gemma 4 family. Text-only today (text decoder + MoE variant).
//!
//! Layout mirrors [`crate::qwen3_5`]: a `text/` sub-module that's
//! always compiled when `gemma4` is on, with room for image/audio/
//! video sub-modules to slot in alongside when those modalities land.

pub mod text;

use std::path::Path;

use crate::error::Error;
use crate::family::LoadedContext;

/// `config.json::model_type` strings handled by this family.
pub const MODEL_TYPES: &[&str] = &[
    "gemma4",
    "gemma4_text",
    "gemma4textmodel",
    "gemma4forcausallm",
];

/// Returns true if this family knows how to load `model_type`.
pub(crate) fn handles(model_type: &str) -> bool {
    MODEL_TYPES.contains(&model_type)
}

/// Family entry point. Gemma 4 has only one path today; future
/// modalities slot in next to the `text::adapter::load_context` call.
pub(crate) fn load_context(model_type: &str, dir: &Path) -> Result<LoadedContext, Error> {
    if MODEL_TYPES.contains(&model_type) {
        return text::adapter::load_context(dir);
    }

    Err(Error::Other(
        format!("gemma4::load_context: unsupported model_type {model_type:?}").into(),
    ))
}
