//! Cached one-shot `MetalKernel` factories for the cache's hot-path
//! kernels. Each kernel is built once on first access and reused for
//! the lifetime of the process.

use std::sync::OnceLock;

use mlx_rs::fast::MetalKernel;

use super::fused_quantized_sdpa::make_fused_qsdpa_kernel;
use crate::steel_attention::{make_steel_attention_kernel, make_steel_quant_attention_kernel};

/// Head dims the steel prefill kernel supports. D=128 covers Qwen3,
/// Llama-3.2, Qwen3.5. D=256 covers Qwen 3.6, Gemma 3, Gemma 4 local.
pub(crate) const STEEL_SUPPORTED_HEAD_DIMS: &[i32] = &[128, 256, 512];

pub(crate) fn cached_fused_qsdpa_kernel() -> &'static MetalKernel {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    KERNEL.get_or_init(|| make_fused_qsdpa_kernel().expect("make_fused_qsdpa_kernel"))
}

pub(crate) fn cached_steel_attention_kernel() -> &'static MetalKernel {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    KERNEL.get_or_init(|| make_steel_attention_kernel().expect("make_steel_attention_kernel"))
}

pub(crate) fn cached_steel_quant_attention_kernel() -> &'static MetalKernel {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        make_steel_quant_attention_kernel().expect("make_steel_quant_attention_kernel")
    })
}
