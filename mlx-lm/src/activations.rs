//! Shared compiled activation helpers.
//!
//! Mirrors Python `mlx_lm.models.activations`. Functions wrap a
//! `transforms::compile`-fused inner kernel in a thread-local cache so
//! every decoder layer per token reuses one fused graph instead of
//! rebuilding it per call.

use std::cell::RefCell;

use mlx_rs::{
    error::Exception,
    nn,
    ops::sigmoid,
    transforms::compile::{shape::TwoArgs, CallMut, Compile, Compiled},
    Array,
};

type SwigluCompiled = Compiled<
    fn((&Array, &Array)) -> Result<Array, Exception>,
    Box<dyn FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'static>,
    TwoArgs,
>;

thread_local! {
    static SWIGLU_CACHE: RefCell<Option<SwigluCompiled>> = const { RefCell::new(None) };
}

/// `silu(gate) * x`.
///
/// Mirrors Python's `@partial(mx.compile, shapeless=True) swiglu` from
/// `mlx_lm.models.activations`. The compiled graph is cached per thread
/// across every decoder layer of every token.
pub fn swiglu(gate: &Array, x: &Array) -> Result<Array, Exception> {
    SWIGLU_CACHE.with(|cell| {
        let mut borrowed = cell.borrow_mut();
        let compiled = borrowed.get_or_insert_with(|| {
            Compile::<(&Array, &Array), Array, Exception>::compile(
                swiglu_inner as fn((&Array, &Array)) -> Result<Array, Exception>,
                true,
            )
        });
        CallMut::call_mut(compiled, (gate, x))
    })
}

fn swiglu_inner((gate, x): (&Array, &Array)) -> Result<Array, Exception> {
    nn::silu(gate)?.multiply(x)
}

type AttentionGateCompiled = Compiled<
    fn((&Array, &Array)) -> Result<Array, Exception>,
    Box<dyn FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'static>,
    TwoArgs,
>;

thread_local! {
    static ATTENTION_GATE_CACHE: RefCell<Option<AttentionGateCompiled>> =
        const { RefCell::new(None) };
}

/// `output * sigmoid(gate)`.
///
/// Mirrors the trailing two-op pattern at the tail of Qwen3.5's full
/// attention block. Compiled-cached so every full-attention layer of
/// every token reuses one fused graph instead of two separate Metal
/// dispatches (`sigmoid` + `multiply`).
pub fn attention_gate(output: &Array, gate: &Array) -> Result<Array, Exception> {
    ATTENTION_GATE_CACHE.with(|cell| {
        let mut borrowed = cell.borrow_mut();
        let compiled = borrowed.get_or_insert_with(|| {
            Compile::<(&Array, &Array), Array, Exception>::compile(
                attention_gate_inner as fn((&Array, &Array)) -> Result<Array, Exception>,
                true,
            )
        });
        CallMut::call_mut(compiled, (output, gate))
    })
}

fn attention_gate_inner((output, gate): (&Array, &Array)) -> Result<Array, Exception> {
    sigmoid(gate)?.multiply(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swiglu_matches_manual_silu_multiply() {
        let gate = Array::from_slice(&[1.0_f32, -1.0, 0.5, 2.0], &[2, 2]);
        let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
        let fused = swiglu(&gate, &x).unwrap();
        let manual = nn::silu(&gate).unwrap().multiply(&x).unwrap();
        let diff = fused.subtract(&manual).unwrap();
        let max = diff.abs().unwrap().max(None).unwrap().item::<f32>();
        assert!(max < 1e-5, "fused vs manual swiglu diverge: {max}");
    }

    #[test]
    fn attention_gate_matches_manual_sigmoid_multiply() {
        let output = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
        let gate = Array::from_slice(&[0.0_f32, 1.0, -1.0, 2.0], &[2, 2]);
        let fused = attention_gate(&output, &gate).unwrap();
        let manual = sigmoid(&gate).unwrap().multiply(&output).unwrap();
        let diff = fused.subtract(&manual).unwrap();
        let max = diff.abs().unwrap().max(None).unwrap().item::<f32>();
        assert!(max < 1e-5, "fused vs manual attention_gate diverge: {max}");
    }
}
