//! In-tree bf16/fp16 → quantised safetensors converter for mlx-lm-supported
//! checkpoints.
//!
//! The Python `mlx_lm.convert` pipeline is model-coupled and silently drops
//! tensors it doesn't recognise (MTP weights, exotic submodules). This
//! crate is the Rust replacement, driven by a per-model [`Rewriter`] table.
//!
//! Currently supports Qwen 3.6 MoE (with MTP). Other families are deferred
//! until they're actually needed.
//!
//! See [`convert`] for the entry point and `bin/convert.rs` in
//! `examples/lm` for the CLI.

mod plan;
mod quantize;
mod runner;
mod shards;

pub mod qwen3_5;

pub use plan::{QuantClass, RewriteOutput, Rewriter};
pub use runner::{convert, ConvertOptions, ConvertReport};

/// Error type surfaced from the converter.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("safetensors: {0}")]
    Safetensors(#[from] mlx_rs::error::IoError),
    #[error("mlx: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),
    #[error("{0}")]
    Other(String),
}

impl Error {
    pub fn custom(msg: impl Into<String>) -> Self {
        Self::Other(msg.into())
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
