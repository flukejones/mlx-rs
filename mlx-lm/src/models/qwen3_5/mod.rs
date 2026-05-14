//! Qwen3.5 (a.k.a. `qwen3_5` / `Qwen3_5ForConditionalGeneration`) hybrid
//! linear-attention + full-attention multimodal LM.
//!
//! This is the Chandra-OCR-2 base model: a 32-layer text decoder where every
//! fourth layer is regular grouped-query attention and the rest are Gated
//! DeltaNet (Mamba2-style recurrent) blocks, paired with a Qwen3-VL-style ViT
//! vision tower for image inputs.
//!
//! The module is in active construction — currently only [`config`] is wired
//! up. Subsequent commits will land [`rope`], [`text`], [`vision`],
//! [`multimodal`], [`image_processor`], [`weights`], and [`generation`].

pub mod cache;
pub mod config;
pub mod gated_delta;
pub mod gated_delta_block;
pub mod generation;
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
