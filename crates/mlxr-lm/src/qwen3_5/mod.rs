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

use crate::error::Error;
use crate::family::LoadedContext;

/// `config.json::model_type` strings handled by the dense entry point
/// in [`text::adapter_dense::load_context_dense`]. VLM checkpoints
/// also use these — disambiguation happens at load time via the
/// `preprocessor_config.json` probe inside [`load_context`].
pub const MODEL_TYPES_DENSE: &[&str] =
    &["qwen3_5", "qwen3_5_text", "qwen3_5forconditionalgeneration"];

/// MoE model_type strings (Qwen 3.6 35B-A3B and friends).
pub const MODEL_TYPES_MOE: &[&str] = &["qwen3_5_moe", "qwen3_5_moe_text"];

/// Returns true if this family knows how to load `model_type`.
pub(crate) fn handles(model_type: &str) -> bool {
    MODEL_TYPES_DENSE.contains(&model_type) || MODEL_TYPES_MOE.contains(&model_type)
}

/// Family entry point. Routes by `model_type` then probes the
/// checkpoint to choose dense-text vs vlm. The probe runs once at
/// load — never per turn.
pub(crate) fn load_context(model_type: &str, dir: &Path) -> Result<LoadedContext, Error> {
    if MODEL_TYPES_MOE.contains(&model_type) {
        return text::adapter_moe::load_context_moe(dir);
    }

    if MODEL_TYPES_DENSE.contains(&model_type) {
        let is_vlm_checkpoint = dir.join("preprocessor_config.json").exists();

        #[cfg(feature = "image")]
        if is_vlm_checkpoint {
            return image::adapter::load_context_vlm(dir);
        }

        #[cfg(not(feature = "image"))]
        if is_vlm_checkpoint {
            log::warn!(
                "qwen3_5: checkpoint at {} carries preprocessor_config.json \
                 but the `image` feature is off; loading text-only (vision tower ignored)",
                dir.display()
            );
        }

        return text::adapter_dense::load_context_dense(dir);
    }

    Err(Error::Other(
        format!("qwen3_5::load_context: unsupported model_type {model_type:?}").into(),
    ))
}
