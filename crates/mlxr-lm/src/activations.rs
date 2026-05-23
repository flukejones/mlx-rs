//! Shared compiled activation helpers.
//!
//! Each function wraps a `transforms::compile`-fused inner kernel in
//! a caller-owned cache so every decoder layer per token reuses one
//! fused graph instead of rebuilding it per call. The cache lives on
//! the owning module struct (e.g. `Mlp::swiglu_cache`) and is borrowed
//! `&mut` per call; mlx core's compile/encoder state is thread-local
//! since mlx 0.31, so the cache must be Drop-bound to the same thread
//! that calls it.

// Each `as fn(...) -> ...` below coerces a zero-sized fn-item to a
// shared fn-pointer type. Without the cast every fn-item would yield a
// distinct `Compiled<F, _>` and the `OnceLock` cache slot could not be
// re-used across activations. Clippy/rustc's trivial_casts diagnostic
// prints identical source/dest types, but the source is the fn-item
// ZST, not a fn-pointer.
#![allow(
    trivial_casts,
    reason = "fn-item ZST → fn-pointer coercion for shared compile cache"
)]

use std::sync::OnceLock;

use mlxr::{
    error::Exception,
    layers,
    ops::{
        indexing::{take_along_axis, take_axis},
        sigmoid, softmax_axis, sum_axes, tanh,
    },
    transforms::compile::{
        allocate_compile_id,
        shape::{ThreeArgs, TwoArgs},
        CallMut, Compile, Compiled,
    },
    Array,
};

/// Process-wide cache ids — one slot per logical activation, shared
/// across every `swiglu`/`geglu`/`attention_gate` cache instance. Lets
/// MLX's `compiler_cache` reuse a single compiled Metal kernel across
/// all 30+ transformer layers instead of JIT-compiling 30 redundant
/// copies.
fn swiglu_id() -> usize {
    static ID: OnceLock<usize> = OnceLock::new();
    *ID.get_or_init(allocate_compile_id)
}
fn attention_gate_id() -> usize {
    static ID: OnceLock<usize> = OnceLock::new();
    *ID.get_or_init(allocate_compile_id)
}
fn geglu_id() -> usize {
    static ID: OnceLock<usize> = OnceLock::new();
    *ID.get_or_init(allocate_compile_id)
}
fn logit_softcap_id() -> usize {
    static ID: OnceLock<usize> = OnceLock::new();
    *ID.get_or_init(allocate_compile_id)
}
fn expert_combine_id() -> usize {
    static ID: OnceLock<usize> = OnceLock::new();
    *ID.get_or_init(allocate_compile_id)
}
fn residual_add_scale_id() -> usize {
    static ID: OnceLock<usize> = OnceLock::new();
    *ID.get_or_init(allocate_compile_id)
}
fn router_post_id() -> usize {
    static ID: OnceLock<usize> = OnceLock::new();
    *ID.get_or_init(allocate_compile_id)
}

pub type SwigluCompiled = Compiled<
    fn((&Array, &Array)) -> Result<Array, Exception>,
    Box<dyn FnMut(&[Array]) -> Result<Vec<Array>, Exception> + Send + 'static>,
    TwoArgs,
>;

pub type AttentionGateCompiled = Compiled<
    fn((&Array, &Array)) -> Result<Array, Exception>,
    Box<dyn FnMut(&[Array]) -> Result<Vec<Array>, Exception> + Send + 'static>,
    TwoArgs,
>;

pub type GegluCompiled = Compiled<
    fn((&Array, &Array)) -> Result<Array, Exception>,
    Box<dyn FnMut(&[Array]) -> Result<Vec<Array>, Exception> + Send + 'static>,
    TwoArgs,
>;

/// Compiled `tanh(x / cap) * cap` — Gemma 3/4 final logit softcap.
/// Same `(&Array, &Array)` signature so it slots into the existing
/// `TwoArgs` compile path; `cap` is passed as a 0-d scalar array.
pub type LogitSoftcapCompiled = Compiled<
    fn((&Array, &Array)) -> Result<Array, Exception>,
    Box<dyn FnMut(&[Array]) -> Result<Vec<Array>, Exception> + Send + 'static>,
    TwoArgs,
>;

/// Compiled `(weights.expand(-1) * y).sum(-2)` — collapses two
/// separate launches (`multiply` + `sum`) into one fused graph used
/// by Gemma 4 MoE `Experts::forward`. `weights` is `[B, L, K, 1]`,
/// `y` is `[B, L, K, D]` → result `[B, L, D]`.
pub type ExpertCombineCompiled = Compiled<
    fn((&Array, &Array)) -> Result<Array, Exception>,
    Box<dyn FnMut(&[Array]) -> Result<Vec<Array>, Exception> + Send + 'static>,
    TwoArgs,
>;

/// Compiled `(residual + ff_out) * layer_scalar` — Gemma 4 per-layer
/// epilogue for non-fp16 dtypes. Folds two launches into one.
pub type ResidualAddScaleCompiled = Compiled<
    fn((&Array, &Array, &Array)) -> Result<Array, Exception>,
    Box<dyn FnMut(&[Array]) -> Result<Vec<Array>, Exception> + Send + 'static>,
    ThreeArgs,
>;

/// Compiled router post-processing: gathers top-k scores, gathers
/// per-expert scales, applies softmax, multiplies. Inputs are
/// (scores `[B, L, E]`, top_k_indices `[B, L, K]`, per_expert_scale
/// `[E]`); output is `[B, L, K]`. Collapses 4 launches into 1.
pub type RouterPostCompiled = Compiled<
    fn((&Array, &Array, &Array)) -> Result<Array, Exception>,
    Box<dyn FnMut(&[Array]) -> Result<Vec<Array>, Exception> + Send + 'static>,
    ThreeArgs,
>;

/// Cached compiled-graph slot for [`swiglu`]. Owned by the calling
/// module (typically a per-layer `Mlp::swiglu_cache`). Initialised lazily
/// on first call. Custom `Debug` is opaque — the inner `Compiled` wraps
/// a `Box<dyn FnMut>` that has no `Debug` impl.
#[derive(Default)]
pub struct SwigluCache(pub Option<SwigluCompiled>);

#[derive(Default)]
pub struct AttentionGateCache(pub Option<AttentionGateCompiled>);

#[derive(Default)]
pub struct GegluCache(pub Option<GegluCompiled>);

#[derive(Default)]
pub struct LogitSoftcapCache(pub Option<LogitSoftcapCompiled>);

#[derive(Default)]
pub struct ExpertCombineCache(pub Option<ExpertCombineCompiled>);

#[derive(Default)]
pub struct ResidualAddScaleCache(pub Option<ResidualAddScaleCompiled>);

#[derive(Default)]
pub struct RouterPostCache(pub Option<RouterPostCompiled>);

impl std::fmt::Debug for SwigluCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SwigluCache")
            .field("filled", &self.0.is_some())
            .finish()
    }
}

impl std::fmt::Debug for AttentionGateCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AttentionGateCache")
            .field("filled", &self.0.is_some())
            .finish()
    }
}

impl std::fmt::Debug for GegluCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GegluCache")
            .field("filled", &self.0.is_some())
            .finish()
    }
}

impl std::fmt::Debug for LogitSoftcapCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogitSoftcapCache")
            .field("filled", &self.0.is_some())
            .finish()
    }
}

impl std::fmt::Debug for ExpertCombineCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExpertCombineCache")
            .field("filled", &self.0.is_some())
            .finish()
    }
}

impl std::fmt::Debug for ResidualAddScaleCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResidualAddScaleCache")
            .field("filled", &self.0.is_some())
            .finish()
    }
}

impl std::fmt::Debug for RouterPostCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RouterPostCache")
            .field("filled", &self.0.is_some())
            .finish()
    }
}

/// `silu(gate) * x` as a compile-fused kernel. Caller passes a
/// `&mut SwigluCache` owned by the surrounding module; the compiled
/// graph is built on first call and reused thereafter.
pub fn swiglu(cache: &mut SwigluCache, gate: &Array, x: &Array) -> Result<Array, Exception> {
    let compiled = cache.0.get_or_insert_with(|| {
        Compile::<(&Array, &Array), Array, Exception>::compile_with_id(
            swiglu_inner as fn((&Array, &Array)) -> Result<Array, Exception>,
            swiglu_id(),
            true,
        )
    });
    CallMut::call_mut(compiled, (gate, x))
}

fn swiglu_inner((gate, x): (&Array, &Array)) -> Result<Array, Exception> {
    layers::silu(gate)?.multiply(x)
}

/// `output * sigmoid(gate)` — trailing fused op of Qwen3.5 full-attention.
/// Caller-owned cache, same shape as [`swiglu`].
pub fn attention_gate(
    cache: &mut AttentionGateCache,
    output: &Array,
    gate: &Array,
) -> Result<Array, Exception> {
    let compiled = cache.0.get_or_insert_with(|| {
        Compile::<(&Array, &Array), Array, Exception>::compile_with_id(
            attention_gate_inner as fn((&Array, &Array)) -> Result<Array, Exception>,
            attention_gate_id(),
            true,
        )
    });
    CallMut::call_mut(compiled, (output, gate))
}

fn attention_gate_inner((output, gate): (&Array, &Array)) -> Result<Array, Exception> {
    sigmoid(gate)?.multiply(output)
}

/// `gelu_approx(gate) * up` as a compile-fused kernel (Gemma 4's
/// GeGLU). Caller passes a `&mut GegluCache` owned by the surrounding
/// module; the compiled graph is built on first call and reused
/// thereafter.
pub fn geglu(cache: &mut GegluCache, gate: &Array, up: &Array) -> Result<Array, Exception> {
    let compiled = cache.0.get_or_insert_with(|| {
        Compile::<(&Array, &Array), Array, Exception>::compile_with_id(
            geglu_inner as fn((&Array, &Array)) -> Result<Array, Exception>,
            geglu_id(),
            true,
        )
    });
    CallMut::call_mut(compiled, (gate, up))
}

fn geglu_inner((gate, up): (&Array, &Array)) -> Result<Array, Exception> {
    gelu_approximate_in_dtype(gate)?.multiply(up)
}

/// Dtype-preserving gelu approximation. mlx-rs's `nn::gelu_approximate`
/// builds its constants as `array!(0.5_f32)` etc., which promotes bf16
/// or f16 inputs to f32 (and cascades that f32 through the rest of the
/// MoE forward). Staging the scalars into the input dtype keeps the
/// graph in-place.
fn gelu_approximate_in_dtype(x: &Array) -> Result<Array, Exception> {
    let dt = x.dtype();
    let cast = |c: f32| -> Result<Array, Exception> { Array::from_f32(c).as_dtype(dt) };
    let half = cast(0.5)?;
    let one = cast(1.0)?;
    let sqrt_2_over_pi = cast((2.0_f32 / std::f32::consts::PI).sqrt())?;
    let k = cast(0.044715)?;
    let x3 = x.multiply(x)?.multiply(x)?;
    let inner = x.add(&k.multiply(&x3)?)?;
    let scaled = sqrt_2_over_pi.multiply(&inner)?;
    let t = tanh(&scaled)?;
    half.multiply(x)?.multiply(&one.add(&t)?)
}

/// `tanh(x / cap) * cap` — Gemma final logit softcap.  Caller owns the
/// cache (per-model `LogitSoftcapCache`); compiled once and reused.
pub fn logit_softcap(
    cache: &mut LogitSoftcapCache,
    x: &Array,
    cap: &Array,
) -> Result<Array, Exception> {
    let compiled = cache.0.get_or_insert_with(|| {
        Compile::<(&Array, &Array), Array, Exception>::compile_with_id(
            logit_softcap_inner as fn((&Array, &Array)) -> Result<Array, Exception>,
            logit_softcap_id(),
            true,
        )
    });
    CallMut::call_mut(compiled, (x, cap))
}

fn logit_softcap_inner((x, cap): (&Array, &Array)) -> Result<Array, Exception> {
    tanh(&x.divide(cap)?)?.multiply(cap)
}

/// `(weights * y).sum(axis=-2)` — Gemma 4 MoE expert combine. Caller
/// passes `weights` shape `[..., K, 1]` (already expanded) and `y`
/// shape `[..., K, D]`; result `[..., D]`. Fuses two launches into one.
pub fn expert_combine(
    cache: &mut ExpertCombineCache,
    weights: &Array,
    y: &Array,
) -> Result<Array, Exception> {
    let compiled = cache.0.get_or_insert_with(|| {
        Compile::<(&Array, &Array), Array, Exception>::compile_with_id(
            expert_combine_inner as fn((&Array, &Array)) -> Result<Array, Exception>,
            expert_combine_id(),
            true,
        )
    });
    CallMut::call_mut(compiled, (weights, y))
}

fn expert_combine_inner((weights, y): (&Array, &Array)) -> Result<Array, Exception> {
    sum_axes(&weights.multiply(y)?, &[-2], false)
}

/// `(residual + ff_out) * layer_scalar` — Gemma 4 per-layer epilogue.
/// Only call on non-fp16 dtypes; the fp16 path needs clip in fp32
/// (see gemma4 `clip_residual`).
pub fn residual_add_scale(
    cache: &mut ResidualAddScaleCache,
    residual: &Array,
    ff_out: &Array,
    layer_scalar: &Array,
) -> Result<Array, Exception> {
    let compiled = cache.0.get_or_insert_with(|| {
        Compile::<(&Array, &Array, &Array), Array, Exception>::compile_with_id(
            residual_add_scale_inner as fn((&Array, &Array, &Array)) -> Result<Array, Exception>,
            residual_add_scale_id(),
            true,
        )
    });
    CallMut::call_mut(compiled, (residual, ff_out, layer_scalar))
}

fn residual_add_scale_inner(
    (residual, ff_out, layer_scalar): (&Array, &Array, &Array),
) -> Result<Array, Exception> {
    residual.add(ff_out)?.multiply(layer_scalar)
}

/// Router post-processing: `softmax(take_along_axis(scores, idx, -1), -1)
/// * per_expert_scale[idx]`. Fuses 4 launches into 1.
pub fn router_post(
    cache: &mut RouterPostCache,
    scores: &Array,
    top_k_indices: &Array,
    per_expert_scale: &Array,
) -> Result<Array, Exception> {
    let compiled = cache.0.get_or_insert_with(|| {
        Compile::<(&Array, &Array, &Array), Array, Exception>::compile_with_id(
            router_post_inner as fn((&Array, &Array, &Array)) -> Result<Array, Exception>,
            router_post_id(),
            true,
        )
    });
    CallMut::call_mut(compiled, (scores, top_k_indices, per_expert_scale))
}

fn router_post_inner(
    (scores, top_k_indices, per_expert_scale): (&Array, &Array, &Array),
) -> Result<Array, Exception> {
    let top_k_scores = take_along_axis(scores, top_k_indices, -1)?;
    let per_expert_gathered = take_axis(per_expert_scale, top_k_indices, 0)?;
    softmax_axis(&top_k_scores, -1, None)?.multiply(&per_expert_gathered)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    #![allow(clippy::print_stdout, reason = "test code")]
    #![allow(clippy::print_stderr, reason = "test code")]
    use super::*;

    #[test]
    fn swiglu_matches_manual_silu_multiply() {
        let gate = Array::from_slice(&[1.0_f32, -1.0, 0.5, 2.0], &[2, 2]);
        let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
        let mut cache = SwigluCache::default();
        let fused = swiglu(&mut cache, &gate, &x).unwrap();
        let manual = layers::silu(&gate).unwrap().multiply(&x).unwrap();
        let diff = fused.subtract(&manual).unwrap();
        let max = diff.abs().unwrap().max(None).unwrap().item::<f32>();
        assert!(max < 1e-5, "fused vs manual swiglu diverge: {max}");
    }

    #[test]
    fn attention_gate_after_swiglu_does_not_collide() {
        // Both activations compile the same `Compile<(&Array, &Array), Array, ...>`
        // signature with `fn` pointer keys; verify their compiled-graph caches
        // remain distinct even when invoked in sequence with the same shapes
        // (regression: chandra-ocr-2 forward returned sigmoid(output)*gate
        //  instead of sigmoid(gate)*output when swiglu warmed mx's compile
        //  cache first).
        let output = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
        let gate = Array::from_slice(&[0.0_f32, 1.0, -1.0, 2.0], &[2, 2]);

        let mut swiglu_cache = SwigluCache::default();
        let _ = swiglu(&mut swiglu_cache, &gate, &output).unwrap();

        let mut ag_cache = AttentionGateCache::default();
        let fused = attention_gate(&mut ag_cache, &output, &gate).unwrap();
        let manual = sigmoid(&gate).unwrap().multiply(&output).unwrap();
        let diff = fused.subtract(&manual).unwrap();
        let max = diff.abs().unwrap().max(None).unwrap().item::<f32>();
        assert!(
            max < 1e-5,
            "attention_gate diverged after warming swiglu: max_abs={max}"
        );
    }

    #[test]
    fn router_post_matches_manual() {
        // scores: [1, 1, 4]; top_k_indices: [1, 1, 2] picks experts (1, 3);
        // per_expert_scale: [4] one per expert.
        let scores = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let top_k_indices = Array::from_slice(&[1_i32, 3], &[1, 1, 2]);
        let per_expert_scale = Array::from_slice(&[10.0_f32, 20.0, 30.0, 40.0], &[4]);

        let mut cache = RouterPostCache::default();
        let fused = router_post(&mut cache, &scores, &top_k_indices, &per_expert_scale).unwrap();

        let top_k_scores = take_along_axis(&scores, &top_k_indices, -1).unwrap();
        let gathered = take_axis(&per_expert_scale, &top_k_indices, 0).unwrap();
        let manual = softmax_axis(&top_k_scores, -1, None)
            .unwrap()
            .multiply(&gathered)
            .unwrap();

        let diff = fused.subtract(&manual).unwrap();
        let max = diff.abs().unwrap().max(None).unwrap().item::<f32>();
        assert!(max < 1e-5, "fused vs manual router_post diverge: {max}");
        assert_eq!(fused.shape(), &[1, 1, 2]);
    }

    #[test]
    fn expert_combine_matches_manual() {
        // [B=1, L=1, K=4, D=3]
        let w = Array::from_slice(&[0.1_f32, 0.2, 0.3, 0.4], &[1, 1, 4, 1]);
        let y = Array::from_slice(
            &[
                1.0_f32, 2.0, 3.0, // expert 0
                4.0, 5.0, 6.0, // expert 1
                7.0, 8.0, 9.0, // expert 2
                10.0, 11.0, 12.0, // expert 3
            ],
            &[1, 1, 4, 3],
        );
        let mut cache = ExpertCombineCache::default();
        let fused = expert_combine(&mut cache, &w, &y).unwrap();
        let manual = sum_axes(w.multiply(&y).unwrap(), &[-2], false).unwrap();
        let diff = fused.subtract(&manual).unwrap();
        let max = diff.abs().unwrap().max(None).unwrap().item::<f32>();
        assert!(max < 1e-5, "fused vs manual expert_combine diverge: {max}");
        // Sanity: shape collapses [1,1,4,3] -> [1,1,3]
        assert_eq!(fused.shape(), &[1, 1, 3]);
    }

    #[test]
    fn logit_softcap_matches_manual() {
        let x = Array::from_slice(&[1.0_f32, 2.0, -3.0, 4.0], &[2, 2]);
        let cap = Array::from_f32(30.0);
        let mut cache = LogitSoftcapCache::default();
        let fused = logit_softcap(&mut cache, &x, &cap).unwrap();
        let manual = tanh(x.divide(&cap).unwrap())
            .unwrap()
            .multiply(&cap)
            .unwrap();
        let diff = fused.subtract(&manual).unwrap();
        let max = diff.abs().unwrap().max(None).unwrap().item::<f32>();
        assert!(max < 1e-5, "fused vs manual logit_softcap diverge: {max}");
    }

    #[test]
    fn attention_gate_matches_manual_sigmoid_multiply() {
        let output = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
        let gate = Array::from_slice(&[0.0_f32, 1.0, -1.0, 2.0], &[2, 2]);
        let mut cache = AttentionGateCache::default();
        let fused = attention_gate(&mut cache, &output, &gate).unwrap();
        let manual = sigmoid(&gate).unwrap().multiply(&output).unwrap();
        let diff = fused.subtract(&manual).unwrap();
        let max = diff.abs().unwrap().max(None).unwrap().item::<f32>();
        assert!(max < 1e-5, "fused vs manual attention_gate diverge: {max}");
    }
}
