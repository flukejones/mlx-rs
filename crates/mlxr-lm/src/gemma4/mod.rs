//! Gemma 4 family. Text-only today (text decoder + MoE variant).
//!
//! Layout mirrors [`crate::qwen3_5`]: a `text/` sub-module that's
//! always compiled when `gemma4` is on, with room for image/audio/
//! video sub-modules to slot in alongside when those modalities land.

pub mod text;

use std::path::Path;

use crate::config::ModelConfig;
use crate::error::Error;
use crate::family::LoadedContext;

/// Family entry point. Gemma 4 has only one path today; future
/// modalities slot in next to the `text::adapter::load_context` call.
pub(crate) fn load_context(cfg: &ModelConfig, dir: &Path) -> Result<LoadedContext, Error> {
    let env = cfg
        .family
        .as_gemma4()
        .ok_or_else(|| Error::Other("gemma4::load_context: wrong family".into()))?;
    text::adapter::load_context(cfg, env, dir)
}
