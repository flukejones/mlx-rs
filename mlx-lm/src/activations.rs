//! Shared activation helpers.
//!
//! Mirrors `mlx_lm.models.activations`. Caller-owned cache slots are kept
//! as no-ops at this milestone; the compile-fused variants land alongside
//! `Compile::compile_with_id` infrastructure in a later milestone.

use mlx_rs::{error::Exception, nn, Array};

/// Per-call cache slot for [`swiglu`]. No-op placeholder at this milestone.
#[derive(Debug, Default)]
pub struct SwigluCache;

/// `silu(gate) * x`. Cache argument reserved for the compile-fused variant.
pub fn swiglu(_cache: &mut SwigluCache, gate: &Array, x: &Array) -> Result<Array, Exception> {
    nn::silu(gate)?.multiply(x)
}
