//! Switched (expert-routed) linears for Gemma 4 MoE (26B-A4B).
//!
//! Ports `mlx_lm.models.switch_layers.SwitchLinear` + `SwitchGLU` and
//! the quantised counterpart `QuantizedSwitchLinear`. Each
//! `SwitchLinear` holds `[num_experts, output_dims, input_dims]`
//! weights; quantised form packs the weight matrix and adds
//! `(scales, biases)` siblings. Forward dispatches per-token through
//! `gather_mm` / `gather_qmm` with expert indices.

use std::sync::OnceLock;

use mlx_rs::error::Exception;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::Param;
use mlx_rs::ops::indexing::take_axis;
use mlx_rs::ops::{
    argsort, expand_dims_axes, gather_mm, gather_qmm, quantize, split_sections, sum_axes,
    swap_axes, unflatten,
};
use mlx_rs::quantization::{MaybeQuantized, Quantizable};
use mlx_rs::Array;

use crate::activations::{geglu, GegluCache};
use crate::fused_kernels::{gather_qmm_combine, GatherQmmCombineInputs};

/// Index-count threshold below which `gather_qmm` runs without
/// pre-sorting by expert id. The argsort + take_axis pair costs more
/// than the gather-locality win on short/medium prompts on M4 Max; at
/// very large contexts the sort starts paying off but the
/// `cache.attention()` sliding-window mask caps how far we can push
/// prefill in one pass. Re-measure on new MLX kernels.
const SORT_THRESHOLD: usize = 2048;

/// Dense per-expert linear. Weight shape `[num_experts, output_dims,
/// input_dims]`. Quantises into [`QuantizedSwitchLinear`] via
/// [`Quantizable::try_into_quantized`].
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
/// The dense `inner` carries the packed-uint32 weight and the optional
/// bias slot, matching the `QuantizedLinear { inner: Linear, scales,
/// biases }` shape so the sanitiser's `<prefix>.weight →
/// <prefix>.inner.weight` rewrite lines up.
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

    pub fn apply(&self, x: &Array, indices: &Array, sorted: bool) -> Result<Array, Exception> {
        // gather_qmm(x, w, scales, biases, lhs=None, rhs=indices, transpose=true, ...)
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

/// Dispatch wrapper. Wraps the `MaybeQuantized<SwitchLinear>` pattern so
/// `try_into_quantized` flips dense → quantised cleanly.
fn apply_proj(
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

#[derive(Debug, ModuleParameters)]
pub struct SwitchGLU {
    /// Fused gate+up: weight `[E, 2*H, D]`. One `gather_qmm` per layer
    /// instead of two; split the `2*H` output into (gate, up).
    #[param]
    pub gate_up_proj: MaybeQuantized<SwitchLinear>,
    #[param]
    pub down_proj: MaybeQuantized<SwitchLinear>,
    /// Inner hidden width per expert (`H`). Used to split `gate_up_proj`
    /// output along the last axis.
    hidden_dims: i32,
    /// Per-layer compiled-graph cache for `gelu_approx(gate) * up`.
    /// Filled on first forward; reused across every decode step.
    geglu_cache: GegluCache,
    /// Cached 0-D `top_k` constant for `gather_sort`'s `floor_divide`.
    /// `Array::from_int` allocates a fresh GPU array each call; this
    /// drops that per-layer alloc on the long-prompt sort path.
    top_k_arr: OnceLock<Array>,
}

impl SwitchGLU {
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
            hidden_dims,
            geglu_cache: GegluCache::default(),
            top_k_arr: OnceLock::new(),
        })
    }

    pub fn forward(&mut self, x: &Array, indices: &Array) -> Result<Array, Exception> {
        // Add two singleton axes: one for top_k, one for the inner
        // gather_mm "1 token per expert" slot.
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

        // Fused gate+up: one gather_qmm returns [..., 2*H]; split along
        // the last axis. MLX `split_sections` returns views (no copy);
        // pass the slots by reference into geglu's compiled graph.
        let gate_up = apply_proj(&self.gate_up_proj, x_in, idx_in, do_sort)?;
        let parts = split_sections(&gate_up, &[self.hidden_dims], -1)?;
        let activated = geglu(&mut self.geglu_cache, &parts[0], &parts[1])?;
        let mut y = apply_proj(&self.down_proj, &activated, idx_in, do_sort)?;

        if let Some((_, _, inv)) = sorted.as_ref() {
            y = scatter_unsort(&y, inv, indices.shape())?;
        }

        y.squeeze_axes(&[-2])
    }

    /// Decode-only fused path: runs gate_up + geglu, then collapses the
    /// down_proj `gather_qmm` and the expert_combine `sum(w*y, -2)` into
    /// a single custom Metal kernel. Returns `[..., D]` directly (no
    /// `[K, D]` intermediate). Falls back to the legacy 2-launch path
    /// when the down projection is not quantised or when sort fired.
    ///
    /// `top_k_weights` shape `[..., K]` (post-softmax router weights).
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

        // Gate+up + geglu identical to the un-fused path.
        let gate_up = apply_proj(&self.gate_up_proj, &x_exp, indices, false)?;
        let parts = split_sections(&gate_up, &[self.hidden_dims], -1)?;
        let activated = geglu(&mut self.geglu_cache, &parts[0], &parts[1])?;
        // activated shape: [..., K, 1, H]. The middle 1 is the inner
        // gather_mm "one token per expert" slot. Squeeze it so the
        // fused kernel sees `[..., K, H]`.
        let activated_3d = activated.squeeze_axes(&[-2])?;

        // Only quantised down_proj has the (wq, scales, biases) triple
        // the fused kernel needs. Dense down stays on the legacy path.
        match &self.down_proj {
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
                gather_qmm_combine(inputs)
            }
            MaybeQuantized::Original(_) => {
                self.forward_with_combine_fallback(x, indices, top_k_weights)
            }
        }
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

impl Quantizable for SwitchGLU {
    type Quantized = Self;
    type QuantizationError = Exception;

    fn try_into_quantized(
        self,
        group_size: i32,
        bits: i32,
    ) -> Result<Self::Quantized, Self::QuantizationError> {
        Ok(Self {
            gate_up_proj: self.gate_up_proj.try_into_quantized(group_size, bits)?,
            down_proj: self.down_proj.try_into_quantized(group_size, bits)?,
            hidden_dims: self.hidden_dims,
            geglu_cache: self.geglu_cache,
            top_k_arr: self.top_k_arr,
        })
    }
}

/// Sort tokens by expert id so `gather_mm` accesses contiguous expert
/// rows. Returns `(sorted_x, sorted_indices, inv_order_to_unsort)`.
/// `top_k_arr` is the cached 0-D `top_k` constant; passing it in lets
/// the caller avoid a per-call `from_int` alloc.
fn gather_sort(
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

fn scatter_unsort(x: &Array, inv_order: &Array, shape: &[i32]) -> Result<Array, Exception> {
    let unsorted = take_axis(x, inv_order, 0)?;
    unflatten(&unsorted, 0, shape)
}
