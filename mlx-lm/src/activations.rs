//! Shared activation helpers. Mirrors `mlx_lm.models.activations`.
//!
//! Cache args are unit-struct no-ops: each helper is plain-ops with a
//! `&mut FooCache` placeholder so call sites are stable when the
//! compile-fused variant (caller-owned `Compiled` cache) lands.

use mlx_rs::{error::Exception, nn, ops::sigmoid, Array};

/// Placeholder cache slot for [`swiglu`]. Unit struct — no state.
#[derive(Debug, Default)]
pub struct SwigluCache;

/// `silu(gate) * x`.
pub fn swiglu(_cache: &mut SwigluCache, gate: &Array, x: &Array) -> Result<Array, Exception> {
    nn::silu(gate)?.multiply(x)
}

/// Placeholder cache slot for [`attention_gate`]. Unit struct — no state.
#[derive(Debug, Default)]
pub struct AttentionGateCache;

/// `sigmoid(gate) * output`.
pub fn attention_gate(
    _cache: &mut AttentionGateCache,
    output: &Array,
    gate: &Array,
) -> Result<Array, Exception> {
    sigmoid(gate)?.multiply(output)
}
