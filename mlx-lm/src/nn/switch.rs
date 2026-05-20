//! Switched (expert-routed) FFN primitives shared across MoE models.
//!
//! [`SwitchLinear`] is the packed-expert linear primitive (gemma4 +
//! qwen3_5_moe both consume it); [`QuantizedSwitchLinear`] is the
//! `gather_qmm`-friendly quantised counterpart auto-built via
//! [`Quantizable::try_into_quantized`].
//!
//! Compound types: two concrete shapes — [`PackedSwitchFfn`] for the
//! gemma4 layout (one `gate_up_proj [E, 2H, D]` weight) and
//! [`SplitSwitchFfn`] for qwen3.6-MoE (separate `gate_proj [E, H, D]`
//! + `up_proj [E, H, D]`). Each is parameterised over a
//! [`SwitchActivation`] (geglu / swiglu). Type aliases per model live
//! in the consuming module (e.g. `gemma4::GemmaSwitchGlu`).
//!
//! Fast path: both layouts share [`finish_with_combine`] which feeds
//! the fused `gather_qmm_combine` Metal kernel after the activation
//! step. The fused-down dispatch lives in one place.

use std::sync::OnceLock;

use mlx_rs::error::Exception;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::{ModuleParameters, Param};
use mlx_rs::ops::indexing::take_axis;
use mlx_rs::ops::{
    argsort, expand_dims_axes, gather_mm, gather_qmm, quantize, sum_axes, swap_axes, unflatten,
};
use mlx_rs::quantization::{MaybeQuantized, Quantizable};
use mlx_rs::Array;

use crate::activations::{geglu, swiglu, GegluCache, SwigluCache};
use crate::fused_kernels::{gather_qmm_combine, GatherQmmCombineInputs};

/// Index-count threshold below which `gather_qmm` runs without
/// pre-sorting by expert id. Argsort + take_axis costs more than the
/// gather-locality win on short/medium prompts on M4 Max; very long
/// prompts may benefit but `cache.attention()`'s sliding-window mask
/// caps prefill anyway. Re-measure on new MLX kernels.
pub(crate) const SORT_THRESHOLD: usize = 2048;

// ─── Primitives ───────────────────────────────────────────────────

/// Dense per-expert linear. Weight shape `[num_experts, output_dims,
/// input_dims]`. Quantises into [`QuantizedSwitchLinear`] via
/// [`Quantizable::try_into_quantized`]. Constructed once at model
/// init; the `apply` path runs only on unquantised bf16 checkpoints
/// (none currently shipped by mlx-community / lmstudio-community).
#[derive(Debug, Clone, ModuleParameters)]
pub struct SwitchLinear {
    #[param]
    pub weight: Param<Array>,
    #[param]
    pub bias: Param<Option<Array>>,
}

impl SwitchLinear {
    pub fn new(
        input_dims: i32,
        output_dims: i32,
        num_experts: i32,
        bias: bool,
    ) -> Result<Self, Exception> {
        let scale = (1.0 / input_dims as f32).sqrt();
        let weight = mlx_rs::random::uniform::<_, f32>(
            -scale,
            scale,
            &[num_experts, output_dims, input_dims],
            None,
        )?;
        let bias_arr = if bias {
            Some(Array::zeros::<f32>(&[num_experts, output_dims])?)
        } else {
            None
        };
        Ok(Self {
            weight: Param::new(weight),
            bias: Param::new(bias_arr),
        })
    }

    /// Dense `gather_mm`-based apply. `x` carries the per-token leading
    /// dims plus `[..., 1, 1, input_dims]`; `indices.shape = [..., top_k]`.
    pub fn apply(&self, x: &Array, indices: &Array, sorted: bool) -> Result<Array, Exception> {
        let w = swap_axes(self.weight.as_ref(), -1, -2)?;
        let mut y = gather_mm(x, &w, None, Some(indices), Some(sorted))?;
        if let Some(b) = self.bias.as_ref() {
            let b_gather = take_axis(b, indices, 0)?;
            let b_exp = expand_dims_axes(&b_gather, &[-2])?;
            y = y.add(&b_exp)?;
        }
        Ok(y)
    }
}

impl Quantizable for SwitchLinear {
    type Quantized = QuantizedSwitchLinear;
    type QuantizationError = Exception;

    fn try_into_quantized(
        self,
        group_size: i32,
        bits: i32,
    ) -> Result<Self::Quantized, Self::QuantizationError> {
        QuantizedSwitchLinear::try_from_switch_linear(self, group_size, bits)
    }
}

/// Quantised per-expert linear. Packed weight + per-group scales/biases.
///
/// `inner` carries the packed-uint32 weight and the optional bias slot,
/// matching the `QuantizedLinear { inner: Linear, scales, biases }`
/// shape so the sanitiser's `<prefix>.weight → <prefix>.inner.weight`
/// rewrite lines up.
#[derive(Debug, Clone, ModuleParameters)]
pub struct QuantizedSwitchLinear {
    pub group_size: i32,
    pub bits: i32,

    #[param]
    pub scales: Param<Array>,
    #[param]
    pub biases: Param<Array>,
    #[param]
    pub inner: SwitchLinear,
}

impl QuantizedSwitchLinear {
    pub fn try_from_switch_linear(
        linear: SwitchLinear,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, Exception> {
        let (packed_w, scales, biases) = quantize(linear.weight.as_ref(), group_size, bits)?;
        Ok(Self {
            group_size,
            bits,
            scales: Param::new(scales),
            biases: Param::new(biases),
            inner: SwitchLinear {
                weight: Param::new(packed_w),
                bias: linear.bias,
            },
        })
    }

    /// Hot-path runtime entry on every quantised MoE checkpoint.
    pub fn apply(&self, x: &Array, indices: &Array, sorted: bool) -> Result<Array, Exception> {
        let mut y = gather_qmm(
            x,
            self.inner.weight.as_ref(),
            self.scales.as_ref(),
            Some(self.biases.as_ref()),
            None,
            Some(indices),
            Some(true),
            Some(self.group_size),
            Some(self.bits),
            Some(sorted),
        )?;
        if let Some(b) = self.inner.bias.as_ref() {
            let b_gather = take_axis(b, indices, 0)?;
            let b_exp = expand_dims_axes(&b_gather, &[-2])?;
            y = y.add(&b_exp)?;
        }
        Ok(y)
    }
}

/// Inline dispatch wrapper for `MaybeQuantized<SwitchLinear>`. Hot
/// path on every MoE token; `#[inline]` lets the optimiser fold the
/// match away when the callsite knows the variant.
#[inline]
pub(crate) fn apply_proj(
    proj: &MaybeQuantized<SwitchLinear>,
    x: &Array,
    indices: &Array,
    sorted: bool,
) -> Result<Array, Exception> {
    match proj {
        MaybeQuantized::Original(d) => d.apply(x, indices, sorted),
        MaybeQuantized::Quantized(q) => q.apply(x, indices, sorted),
    }
}

// ─── SwitchActivation trait + concrete activations ────────────────

/// Per-element `activation(gate) * up` for one MoE expert output.
/// Implementors own a compiled-graph cache so the activation kernel
/// is built once per layer and reused across decode steps.
pub trait SwitchActivation: ModuleParameters {
    fn activate(&mut self, gate: &Array, up: &Array) -> Result<Array, Exception>;
}

/// `gelu_approx(gate) * up` — gemma4 MoE activation.
#[derive(Debug, Default, ModuleParameters)]
pub struct GegluActivation {
    cache: GegluCache,
}

impl SwitchActivation for GegluActivation {
    fn activate(&mut self, gate: &Array, up: &Array) -> Result<Array, Exception> {
        geglu(&mut self.cache, gate, up)
    }
}

/// `silu(gate) * up` — qwen3.6-MoE activation.
#[derive(Debug, Default, ModuleParameters)]
pub struct SwigluActivation {
    cache: SwigluCache,
}

impl SwitchActivation for SwigluActivation {
    fn activate(&mut self, gate: &Array, up: &Array) -> Result<Array, Exception> {
        swiglu(&mut self.cache, gate, up)
    }
}

// ─── Shared finish path (fused down + combine) ────────────────────

/// Fused-down dispatch: take an already-activated `[..., K, 1, H]`
/// tensor, squeeze, and feed the fused Metal kernel. Returns
/// `Some(out)` on the fast path or `None` when `down_proj` is dense
/// (caller takes the 2-launch fallback).
fn dispatch_fused_combine(
    activated: &Array,
    down_proj: &MaybeQuantized<SwitchLinear>,
    indices: &Array,
    top_k_weights: &Array,
) -> Result<Option<Array>, Exception> {
    let activated_3d = activated.squeeze_axes(&[-2])?;
    match down_proj {
        MaybeQuantized::Quantized(q) => {
            let inputs = GatherQmmCombineInputs {
                activated: &activated_3d,
                weights: top_k_weights,
                wq: q.inner.weight.as_ref(),
                scales: q.scales.as_ref(),
                biases: q.biases.as_ref(),
                indices,
                group_size: q.group_size,
                bits: q.bits,
            };
            gather_qmm_combine(inputs).map(Some)
        }
        MaybeQuantized::Original(_) => Ok(None),
    }
}

/// Apply the down_proj + sum-combine in the 2-launch path. Shared
/// between both layouts' fallback branches.
fn combine_with_weights(
    activated: &Array,
    down_proj: &MaybeQuantized<SwitchLinear>,
    indices: &Array,
    top_k_weights: &Array,
    sorted: bool,
) -> Result<Array, Exception> {
    let y = apply_proj(down_proj, activated, indices, sorted)?;
    let y = y.squeeze_axes(&[-2])?;
    let w = expand_dims_axes(top_k_weights, &[-1])?;
    sum_axes(&w.multiply(&y)?, &[-2], false)
}

// ─── PackedSwitchFfn: gemma4 layout ───────────────────────────────

/// Gemma4 MoE FFN: gate + up fused into one `gate_up_proj`
/// `[E, 2H, D]` `SwitchLinear`. One `gather_qmm` per token for the
/// up-projection step.
#[derive(Debug, ModuleParameters)]
pub struct PackedSwitchFfn<A: SwitchActivation> {
    #[param]
    pub gate_up_proj: MaybeQuantized<SwitchLinear>,
    #[param]
    pub down_proj: MaybeQuantized<SwitchLinear>,
    pub activation: A,
    /// Inner hidden width per expert (`H`). Splits `gate_up_proj`
    /// output along the last axis.
    hidden_dims: i32,
    /// Cached 0-D `top_k` constant for `gather_sort`'s `floor_divide`.
    top_k_arr: OnceLock<Array>,
}

impl<A: SwitchActivation + Default> PackedSwitchFfn<A> {
    pub fn new(
        input_dims: i32,
        hidden_dims: i32,
        num_experts: i32,
        bias: bool,
    ) -> Result<Self, Exception> {
        Ok(Self {
            gate_up_proj: MaybeQuantized::Original(SwitchLinear::new(
                input_dims,
                2 * hidden_dims,
                num_experts,
                bias,
            )?),
            down_proj: MaybeQuantized::Original(SwitchLinear::new(
                hidden_dims,
                input_dims,
                num_experts,
                bias,
            )?),
            activation: A::default(),
            hidden_dims,
            top_k_arr: OnceLock::new(),
        })
    }
}

impl<A: SwitchActivation> PackedSwitchFfn<A> {
    /// Full MoE forward returning `[..., K, D]` per-expert outputs.
    /// Caller does the `sum_k(weights * y)` combine. Use
    /// [`Self::forward_with_combine`] for the fused decode path.
    pub fn forward(&mut self, x: &Array, indices: &Array) -> Result<Array, Exception> {
        let x_exp = expand_dims_axes(x, &[-2, -3])?;
        let do_sort = indices.size() >= SORT_THRESHOLD;
        let top_k_arr = self.top_k_arr.get_or_init(|| {
            let k = *indices.shape().last().expect("indices has trailing dim");
            Array::from_int(k)
        });
        let sorted = do_sort
            .then(|| gather_sort(&x_exp, indices, top_k_arr))
            .transpose()?;
        let (x_in, idx_in): (&Array, &Array) = match sorted.as_ref() {
            Some((xs, idxs, _)) => (xs, idxs),
            None => (&x_exp, indices),
        };

        let gate_up = apply_proj(&self.gate_up_proj, x_in, idx_in, do_sort)?;
        let parts = mlx_rs::ops::split_sections(&gate_up, &[self.hidden_dims], -1)?;
        let activated = self.activation.activate(&parts[0], &parts[1])?;
        let mut y = apply_proj(&self.down_proj, &activated, idx_in, do_sort)?;

        if let Some((_, _, inv)) = sorted.as_ref() {
            y = scatter_unsort(&y, inv, indices.shape())?;
        }

        y.squeeze_axes(&[-2])
    }

    /// Decode-only fused path: gate+up + activation + fused
    /// `down_proj` + combine. Returns `[..., D]`. Falls back to the
    /// 2-launch path when `down_proj` is dense or sort fired.
    pub fn forward_with_combine(
        &mut self,
        x: &Array,
        indices: &Array,
        top_k_weights: &Array,
    ) -> Result<Array, Exception> {
        let x_exp = expand_dims_axes(x, &[-2, -3])?;
        let do_sort = indices.size() >= SORT_THRESHOLD;
        if do_sort {
            return self.forward_with_combine_fallback(x, indices, top_k_weights);
        }

        let gate_up = apply_proj(&self.gate_up_proj, &x_exp, indices, false)?;
        let parts = mlx_rs::ops::split_sections(&gate_up, &[self.hidden_dims], -1)?;
        let activated = self.activation.activate(&parts[0], &parts[1])?;

        if let Some(out) =
            dispatch_fused_combine(&activated, &self.down_proj, indices, top_k_weights)?
        {
            return Ok(out);
        }
        combine_with_weights(&activated, &self.down_proj, indices, top_k_weights, false)
    }

    fn forward_with_combine_fallback(
        &mut self,
        x: &Array,
        indices: &Array,
        top_k_weights: &Array,
    ) -> Result<Array, Exception> {
        let y = self.forward(x, indices)?;
        let w = expand_dims_axes(top_k_weights, &[-1])?;
        sum_axes(&w.multiply(&y)?, &[-2], false)
    }
}

impl<A: SwitchActivation> Quantizable for PackedSwitchFfn<A> {
    type Quantized = Self;
    type QuantizationError = Exception;

    fn try_into_quantized(
        self,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, Exception> {
        Ok(Self {
            gate_up_proj: self.gate_up_proj.try_into_quantized(group_size, bits)?,
            down_proj: self.down_proj.try_into_quantized(group_size, bits)?,
            activation: self.activation,
            hidden_dims: self.hidden_dims,
            top_k_arr: self.top_k_arr,
        })
    }
}

// ─── SplitSwitchFfn: qwen3.6-MoE layout ───────────────────────────

/// Qwen3.6-MoE FFN: separate `gate_proj` + `up_proj`, each
/// `[E, H, D]`. Two `gather_qmm` calls per token for up-projection.
#[derive(Debug, ModuleParameters)]
pub struct SplitSwitchFfn<A: SwitchActivation> {
    #[param]
    pub gate_proj: MaybeQuantized<SwitchLinear>,
    #[param]
    pub up_proj: MaybeQuantized<SwitchLinear>,
    #[param]
    pub down_proj: MaybeQuantized<SwitchLinear>,
    pub activation: A,
    /// Inner hidden width per expert. Carried for symmetry with
    /// `PackedSwitchFfn`; the split layout doesn't need it for forward.
    hidden_dims: i32,
    /// Cached 0-D `top_k` constant for `gather_sort`'s `floor_divide`.
    top_k_arr: OnceLock<Array>,
}

impl<A: SwitchActivation + Default> SplitSwitchFfn<A> {
    pub fn new(
        input_dims: i32,
        hidden_dims: i32,
        num_experts: i32,
        bias: bool,
    ) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: MaybeQuantized::Original(SwitchLinear::new(
                input_dims,
                hidden_dims,
                num_experts,
                bias,
            )?),
            up_proj: MaybeQuantized::Original(SwitchLinear::new(
                input_dims,
                hidden_dims,
                num_experts,
                bias,
            )?),
            down_proj: MaybeQuantized::Original(SwitchLinear::new(
                hidden_dims,
                input_dims,
                num_experts,
                bias,
            )?),
            activation: A::default(),
            hidden_dims,
            top_k_arr: OnceLock::new(),
        })
    }
}

impl<A: SwitchActivation> SplitSwitchFfn<A> {
    /// Full MoE forward returning `[..., K, D]` per-expert outputs.
    pub fn forward(&mut self, x: &Array, indices: &Array) -> Result<Array, Exception> {
        let x_exp = expand_dims_axes(x, &[-2, -3])?;
        let do_sort = indices.size() >= SORT_THRESHOLD;
        let top_k_arr = self.top_k_arr.get_or_init(|| {
            let k = *indices.shape().last().expect("indices has trailing dim");
            Array::from_int(k)
        });
        let sorted = do_sort
            .then(|| gather_sort(&x_exp, indices, top_k_arr))
            .transpose()?;
        let (x_in, idx_in): (&Array, &Array) = match sorted.as_ref() {
            Some((xs, idxs, _)) => (xs, idxs),
            None => (&x_exp, indices),
        };

        let gate = apply_proj(&self.gate_proj, x_in, idx_in, do_sort)?;
        let up = apply_proj(&self.up_proj, x_in, idx_in, do_sort)?;
        let activated = self.activation.activate(&gate, &up)?;
        let mut y = apply_proj(&self.down_proj, &activated, idx_in, do_sort)?;

        if let Some((_, _, inv)) = sorted.as_ref() {
            y = scatter_unsort(&y, inv, indices.shape())?;
        }

        y.squeeze_axes(&[-2])
    }

    /// Decode-only fused path: same shape contract as
    /// `PackedSwitchFfn::forward_with_combine`.
    pub fn forward_with_combine(
        &mut self,
        x: &Array,
        indices: &Array,
        top_k_weights: &Array,
    ) -> Result<Array, Exception> {
        let x_exp = expand_dims_axes(x, &[-2, -3])?;
        let do_sort = indices.size() >= SORT_THRESHOLD;
        if do_sort {
            return self.forward_with_combine_fallback(x, indices, top_k_weights);
        }

        let gate = apply_proj(&self.gate_proj, &x_exp, indices, false)?;
        let up = apply_proj(&self.up_proj, &x_exp, indices, false)?;
        let activated = self.activation.activate(&gate, &up)?;

        if let Some(out) =
            dispatch_fused_combine(&activated, &self.down_proj, indices, top_k_weights)?
        {
            return Ok(out);
        }
        combine_with_weights(&activated, &self.down_proj, indices, top_k_weights, false)
    }

    fn forward_with_combine_fallback(
        &mut self,
        x: &Array,
        indices: &Array,
        top_k_weights: &Array,
    ) -> Result<Array, Exception> {
        let y = self.forward(x, indices)?;
        let w = expand_dims_axes(top_k_weights, &[-1])?;
        sum_axes(&w.multiply(&y)?, &[-2], false)
    }
}

impl<A: SwitchActivation> Quantizable for SplitSwitchFfn<A> {
    type Quantized = Self;
    type QuantizationError = Exception;

    fn try_into_quantized(
        self,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: self.gate_proj.try_into_quantized(group_size, bits)?,
            up_proj: self.up_proj.try_into_quantized(group_size, bits)?,
            down_proj: self.down_proj.try_into_quantized(group_size, bits)?,
            activation: self.activation,
            hidden_dims: self.hidden_dims,
            top_k_arr: self.top_k_arr,
        })
    }
}

// ─── Sort helpers ─────────────────────────────────────────────────

/// Sort tokens by expert id so `gather_mm` accesses contiguous expert
/// rows. Returns `(sorted_x, sorted_indices, inv_order_to_unsort)`.
/// `top_k_arr` is the cached 0-D `top_k` constant; passing it in lets
/// the caller avoid a per-call `from_int` alloc.
pub(crate) fn gather_sort(
    x: &Array,
    indices: &Array,
    top_k_arr: &Array,
) -> Result<(Array, Array, Array), Exception> {
    let flat_idx = indices.flatten(0, -1)?;
    let order = argsort(&flat_idx)?;
    let inv_order = argsort(&order)?;
    let x_flat = x.flatten(0, -3)?;
    let row_idx = order.floor_divide(top_k_arr)?;
    let x_sorted = take_axis(&x_flat, &row_idx, 0)?;
    let idx_sorted = take_axis(&flat_idx, &order, 0)?;
    Ok((x_sorted, idx_sorted, inv_order))
}

pub(crate) fn scatter_unsort(
    x: &Array,
    inv_order: &Array,
    shape: &[i32],
) -> Result<Array, Exception> {
    let unsorted = take_axis(x, inv_order, 0)?;
    unflatten(&unsorted, 0, shape)
}
