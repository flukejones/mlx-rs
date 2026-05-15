//! TurboQuant KV cache compression (ICLR 2026, arxiv 2504.19874).
//! Π-rotation + Lloyd-Max codebook + optional 1-bit QJL residual.
//! Reference: <https://github.com/0xSero/turboquant>.

pub mod codebook;
pub mod packing;
pub mod quantizer;
pub mod rotation;
pub mod searchsorted_kernel;
