//! Qwen 3.5 image path: vision tower, image processor, multimodal
//! embedding stitch, and the VLM adapter that wraps
//! [`crate::qwen3_5::text::adapter_dense::Qwen35DenseAdapter`] with
//! vision-token interleave. Compiled when the `image` feature is on.

pub mod adapter;
pub mod multimodal;
pub mod processor;
pub mod vision;
pub mod weights;
