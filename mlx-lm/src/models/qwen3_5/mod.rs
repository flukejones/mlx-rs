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
pub mod generation;
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
pub use generation::{Generate, SamplingParams, StopCriteria};
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
