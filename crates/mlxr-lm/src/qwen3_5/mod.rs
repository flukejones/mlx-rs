//! Qwen 3.5 / 3.6 family. Hybrid linear-attention + full-attention LM
//! with optional Qwen3-VL vision tower.
//!
//! Layout:
//! - [`text`] always compiled when `qwen3_5` feature is on. Dense +
//!   MoE adapters, model code, MTP sampling.
//! - [`image`] compiled when the `image` feature is also on. Adds the
//!   ViT tower and the VLM adapter that wraps the dense model with
//!   image-token interleave.

pub mod text;

#[cfg(feature = "image")]
pub mod image;

use std::path::Path;

use crate::config::{Family, ModelConfig};
use crate::error::Error;
use crate::family::LoadedContext;

/// Family entry point. Dispatches on the typed [`Family`] variant,
/// then probes the checkpoint to choose dense-text vs vlm. The probe
/// runs once at load — never per turn.
pub(crate) fn load_context(cfg: &ModelConfig, dir: &Path) -> Result<LoadedContext, Error> {
    if let Family::Qwen35Moe(env) = &cfg.family {
        return text::adapter_moe::load_context_moe(cfg, env, dir);
    }
    let env = cfg
        .family
        .as_qwen35()
        .ok_or_else(|| Error::Other("qwen3_5::load_context: wrong family".into()))?;

    let is_vlm_checkpoint = dir.join("preprocessor_config.json").exists();

    #[cfg(feature = "image")]
    if is_vlm_checkpoint {
        return image::adapter::load_context_vlm(cfg, env, dir);
    }

    #[cfg(not(feature = "image"))]
    if is_vlm_checkpoint {
        log::warn!(
            "qwen3_5: checkpoint at {} carries preprocessor_config.json \
             but the `image` feature is off; loading text-only (vision tower ignored)",
            dir.display()
        );
    }

    text::adapter_dense::load_context_dense(cfg, env, dir)
}
