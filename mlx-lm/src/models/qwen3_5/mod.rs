//! Qwen3.5 (a.k.a. `qwen3_5` / `Qwen3_5ForConditionalGeneration`) hybrid
//! linear-attention + full-attention multimodal LM.
//!
//! Chandra-OCR-2 base model: 32-layer text decoder with every fourth
//! layer regular grouped-query attention and the rest Gated DeltaNet
//! (Mamba2-style recurrent), paired with a Qwen3-VL ViT tower for
//! image input.

pub mod cache;
pub mod config;
pub mod gated_delta;
pub mod gated_delta_block;
#[cfg(feature = "models-vision")]
pub mod image_processor;
pub mod layer;
pub mod multimodal;
pub mod rope;
pub mod text;
pub mod vision;
pub mod weights;

pub use cache::{make_caches, LayerCache, LinearAttnCache};
pub use config::{ModelConfig, TextConfig, VisionConfig};
pub use gated_delta_block::GatedDeltaNet;
#[cfg(feature = "models-vision")]
pub use image_processor::{
    smart_resize, ImageProcessorConfig, ProcessedImage, Qwen35ImageProcessor,
};
pub use layer::{DecoderLayer, LanguageModel, Qwen35Decoder};
pub use multimodal::{
    get_rope_index_batched, get_rope_index_single_batch, merge_input_ids_with_image_features,
    pack_position_ids,
};
pub use rope::{apply_multimodal_rotary_pos_emb, MultimodalRope};
pub use text::{Attention, Mlp};
pub use vision::{PatchEmbed, PatchMerger, VisionAttention, VisionBlock, VisionMlp, VisionModel};

use std::path::Path;

/// Read the EOS id set for a Qwen 3.5 / 3.6 checkpoint and ensure
/// `<|im_end|>` (the chat-template turn marker) is present. The
/// tokenizer ships it in `added_tokens`, but `config.json::eos_token_id`
/// often points at `<|endoftext|>` only — letting it through causes
/// the model to keep generating role-tagged turns past the assistant
/// reply.
pub(crate) fn read_qwen3_5_eos_ids(dir: &Path, cfg: &ModelConfig) -> Vec<u32> {
    let mut ids = crate::adapters::read_eos_ids(dir);
    if let Some(cfg_eos) = cfg.eos_token_id.clone() {
        for id in cfg_eos.into_vec_with_chat_eos() {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }
    if !ids.contains(&config::QWEN_CHAT_EOS_TOKEN_ID) {
        ids.push(config::QWEN_CHAT_EOS_TOKEN_ID);
    }
    ids
}
