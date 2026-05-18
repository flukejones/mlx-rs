//! Steel attention port — dense Flash-Attention-style tiled SDPA on Apple
//! Silicon via `mlx_rs::fast::metal_kernel`. Targets prefill (n_q > 1) plus
//! head_dim ∈ {128, 256, 512} that the vector v10 kernel cannot reach.
//!
//! Adapted from upstream mlx `mlx/backend/metal/kernels/steel/attn/`.

pub mod kernel_source;
pub mod params;

#[cfg(test)]
mod tests;

use mlx_rs::error::{Exception, Result};
use mlx_rs::fast::{metal_kernel, MetalKernel, MetalKernelConfig};
use mlx_rs::{Array, Dtype, Stream};

use crate::steel_attention::kernel_source::{KERNEL_HEADER, KERNEL_SOURCE, KERNEL_SOURCE_QUANT};
use crate::steel_attention::params::FlatAttnParams;

const KERNEL_NAME: &str = "steel_attention_v0";

/// Per-D kernel-template tuple. D=512 needs a 1-warp tile so
/// `Q_smem + KV_smem` fits Apple's 32 KB TG-mem cap at fp16/bf16:
///
/// - D=128: TG mem at fp16 ≈ 12 KB (BQ=32, BK=16, 4 warps)
/// - D=256: TG mem at fp16 ≈ 24 KB (BQ=32, BK=16, 4 warps)
/// - D=512: TG mem at fp16 ≈ 24 KB (BQ=8, BK=8, 1 warp).
///   BQ=16 just overshoots (33024 > 32768 B).
#[derive(Clone, Copy)]
struct SteelShape {
    bq: i32,
    bk: i32,
    wm: i32,
    wn: i32,
}

fn shape_for_d(d: i32) -> Option<SteelShape> {
    match d {
        128 | 256 => Some(SteelShape { bq: 32, bk: 16, wm: 4, wn: 1 }),
        512 => Some(SteelShape { bq: 8, bk: 8, wm: 1, wn: 1 }),
        _ => None,
    }
}

const INPUT_NAMES: &[&str] = &[
    "q",
    "k",
    "v",
    "mask",
    "b_param",
    "h_param",
    "d_param",
    "q_len_in",
    "k_len_in",
    "gqa_factor",
    "scale_param",
    "mask_present",
    "do_causal_param",
    "nq_param",
    "nk_param",
    "nq_aligned_param",
    "nk_aligned_param",
    "ql_rem_param",
    "kl_rem_param",
    "ql_off_param",
    "q_strides_in",
    "k_strides_in",
    "v_strides_in",
    "o_strides_in",
];

const OUTPUT_NAMES: &[&str] = &["o"];

/// Compile the steel attention kernel. Cache the handle in the calling
/// module — `MetalKernel::Drop` releases the underlying pipeline.
pub fn make_steel_attention_kernel() -> Result<MetalKernel> {
    metal_kernel(
        KERNEL_NAME,
        INPUT_NAMES,
        OUTPUT_NAMES,
        KERNEL_SOURCE,
        KERNEL_HEADER,
        true,
        false,
    )
}

/// Inputs to [`steel_attention_dispatch`].
pub struct SteelAttentionInputs<'a> {
    /// `q`: `[B, H_q, qL, D]`
    pub q: &'a Array,
    /// `k`: `[B, H_kv, kL, D]`
    pub k: &'a Array,
    /// `v`: `[B, H_kv, kL, D]`
    pub v: &'a Array,
    /// Optional bool mask `[B, H_q, qL, kL]`. **Currently unsupported**;
    /// pass `None` and use `causal: true` for prefill.
    pub mask: Option<&'a Array>,
    /// Apply causal masking (lower-triangular S). Set true for standard
    /// autoregressive prefill (`l > 1`).
    pub causal: bool,
    /// Number of KV tokens already cached before this prefill's `qL`
    /// queries. Used by the causal kernel to skip k-blocks fully below
    /// the diagonal. For a fresh prefill, pass 0.
    pub ql_off: i32,
    pub scale: f32,
    pub head_dim: i32,
    pub h_q: i32,
    pub h_kv: i32,
}

/// Launch the steel kernel and return `o = softmax((Q @ K.T) * scale) @ V`
/// with shape `[B, H_q, qL, D]` in the same dtype as `q`.
///
/// **Current scope**: D ∈ {128, 256}, no explicit mask, no causal
/// masking. Parity tests in `tests.rs` enforce the constraints.
pub fn steel_attention_dispatch(
    kernel: &MetalKernel,
    inputs: SteelAttentionInputs<'_>,
) -> Result<Array> {
    let q_shape = inputs.q.shape();
    if q_shape.len() != 4 {
        return Err(Exception::custom(format!(
            "steel_attention: q must be 4-D [B,H_q,qL,D], got {q_shape:?}"
        )));
    }
    let k_shape = inputs.k.shape();
    let v_shape = inputs.v.shape();
    if k_shape.len() != 4 || v_shape.len() != 4 {
        return Err(Exception::custom("steel_attention: k/v must be 4-D"));
    }

    let b = q_shape[0];
    let h_q = q_shape[1];
    let q_len = q_shape[2];
    let d = q_shape[3];
    let h_kv = k_shape[1];
    let k_len = k_shape[2];

    if h_q != inputs.h_q || h_kv != inputs.h_kv {
        return Err(Exception::custom("steel_attention: head count mismatch"));
    }
    if inputs.h_q % inputs.h_kv != 0 {
        return Err(Exception::custom("steel_attention: H_q % H_kv != 0"));
    }
    if d != inputs.head_dim {
        return Err(Exception::custom("steel_attention: head_dim mismatch"));
    }
    if k_shape[3] != d || v_shape[3] != d {
        return Err(Exception::custom(format!(
            "steel_attention: K head_dim {} / V head_dim {} != Q head_dim {d}",
            k_shape[3], v_shape[3]
        )));
    }
    if k_shape[2] != k_len || v_shape[2] != k_len {
        return Err(Exception::custom(format!(
            "steel_attention: K seq {} / V seq {} mismatch",
            k_shape[2], v_shape[2]
        )));
    }
    if k_shape[0] != b || v_shape[0] != b {
        return Err(Exception::custom(format!(
            "steel_attention: K batch {} / V batch {} != Q batch {b}",
            k_shape[0], v_shape[0]
        )));
    }
    let shape = shape_for_d(d).ok_or_else(|| {
        Exception::custom(format!(
            "steel_attention: supported head_dim ∈ {{128, 256, 512}}, got {d}"
        ))
    })?;
    if inputs.mask.is_some() {
        return Err(Exception::custom(
            "steel_attention: explicit masks unsupported (use causal=true)",
        ));
    }
    // D=512 with fp32 exceeds the 32 KB TG-mem budget (Q_smem alone
    // would be 16*520*4 = 33 KB). Reject early.
    if d == 512 && inputs.q.dtype() == Dtype::Float32 {
        return Err(Exception::custom(
            "steel_attention: D=512 requires fp16 or bf16 (fp32 TG-mem exceeds 32 KB)",
        ));
    }

    let out_dtype = inputs.q.dtype();
    if !matches!(out_dtype, Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32) {
        return Err(Exception::custom("steel_attention: unsupported dtype"));
    }

    let flat = FlatAttnParams::from_shapes(
        b,
        h_q,
        h_kv,
        q_len,
        k_len,
        d,
        inputs.scale,
        shape.bq,
        shape.bk,
        inputs.ql_off,
    );

    // Dummy mask — the body never reads it under `mask_present == 0`,
    // but mlx-rs requires every named input to be bound.
    let dummy_mask = Array::zeros::<u8>(&[1])?.as_dtype(Dtype::Bool)?;
    let mask_present_buf = Array::from_int(0);
    let do_causal_buf = Array::from_int(inputs.causal as i32);

    // Grid: one TG per (q-block, head, batch). TG size = WM*WN*32 threads.
    let nq = (q_len + shape.bq - 1) / shape.bq;
    let threads_per_tg = shape.wm * shape.wn * 32;
    let config = MetalKernelConfig::new()
        .add_output(vec![b, h_q, q_len, d], out_dtype)
        .grid(nq * threads_per_tg, h_q, b)
        .thread_group(threads_per_tg, 1, 1)
        .add_template("T", out_dtype)?
        .add_template("BD", d)?
        .add_template("BQ", shape.bq)?
        .add_template("BK", shape.bk)?
        .add_template("WM", shape.wm)?
        .add_template("WN", shape.wn)?;

    let outs = kernel.apply(
        &[
            inputs.q,
            inputs.k,
            inputs.v,
            &dummy_mask,
            flat.b_param(),
            flat.h_param(),
            flat.d_param(),
            flat.q_len_param(),
            flat.k_len_param(),
            flat.gqa_factor_param(),
            flat.scale_param(),
            &mask_present_buf,
            &do_causal_buf,
            flat.nq_param(),
            flat.nk_param(),
            flat.nq_aligned_param(),
            flat.nk_aligned_param(),
            flat.ql_rem_param(),
            flat.kl_rem_param(),
            flat.ql_off_param(),
            flat.q_strides_arr(),
            flat.k_strides_arr(),
            flat.v_strides_arr(),
            flat.o_strides_arr(),
        ],
        config,
        Stream::default(),
    )?;
    outs.into_iter()
        .next()
        .ok_or_else(|| Exception::custom("steel_attention: no outputs"))
}

// ===================== Quantised K/V variant =========================
// Identical kernel body but K and V are read as packed
// (wq, scales, biases) triples via `QuantBlockLoaderT`.

const KERNEL_NAME_QUANT: &str = "steel_attention_quant_v0";

const INPUT_NAMES_QUANT: &[&str] = &[
    "q",
    "k_wq",
    "k_scales",
    "k_biases",
    "v_wq",
    "v_scales",
    "v_biases",
    "mask",
    "b_param",
    "h_param",
    "d_param",
    "q_len_in",
    "k_len_in",
    "gqa_factor",
    "scale_param",
    "mask_present",
    "do_causal_param",
    "nq_param",
    "nk_param",
    "nq_aligned_param",
    "nk_aligned_param",
    "ql_rem_param",
    "kl_rem_param",
    "ql_off_param",
    "q_strides_in",
    "o_strides_in",
];

/// Compile the quantised steel attention kernel. Different mlx-side
/// kernel name from the dense variant so both can be cached in parallel.
pub fn make_steel_quant_attention_kernel() -> Result<MetalKernel> {
    metal_kernel(
        KERNEL_NAME_QUANT,
        INPUT_NAMES_QUANT,
        OUTPUT_NAMES,
        KERNEL_SOURCE_QUANT,
        KERNEL_HEADER,
        true,
        false,
    )
}

/// Inputs to [`steel_quant_attention_dispatch`]. K/V are packed-
/// quantised triples produced by `mlx_rs::ops::quantize`; same wire
/// format as `mlx_lm::cache::QuantizedKVCache`.
pub struct SteelQuantAttentionInputs<'a> {
    /// `q`: `[B, H_q, qL, D]` (dense T).
    pub q: &'a Array,
    /// `k_wq`: `[B, H_kv, kL, D / (32/bits)]` uint32.
    pub k_wq: &'a Array,
    /// `k_scales`, `k_biases`: `[B, H_kv, kL, D / group_size]` same dtype as Q.
    pub k_scales: &'a Array,
    pub k_biases: &'a Array,
    pub v_wq: &'a Array,
    pub v_scales: &'a Array,
    pub v_biases: &'a Array,
    /// Optional bool mask `[B, H_q, qL, kL]`. Currently unsupported.
    pub mask: Option<&'a Array>,
    /// Apply causal masking. Set true for autoregressive prefill.
    pub causal: bool,
    /// KV-tokens already cached before this prefill (see dense version).
    pub ql_off: i32,
    pub scale: f32,
    pub head_dim: i32,
    pub h_q: i32,
    pub h_kv: i32,
    pub bits: i32,
    pub group_size: i32,
}

/// Dispatch the quantised steel attention kernel. Returns `o = softmax(
/// (Q @ dequant(K).T) * scale + causal_mask) @ dequant(V)` in the
/// dtype of `q`.
///
/// Constraints:
/// - `head_dim ∈ {128, 256}`
/// - `bits ∈ {4, 8}`
/// - `group_size` divides `head_dim`
/// - `mask` must be `None`
pub fn steel_quant_attention_dispatch(
    kernel: &MetalKernel,
    inputs: SteelQuantAttentionInputs<'_>,
) -> Result<Array> {
    let q_shape = inputs.q.shape();
    if q_shape.len() != 4 {
        return Err(Exception::custom(format!(
            "steel_quant_attention: q must be 4-D [B,H_q,qL,D], got {q_shape:?}"
        )));
    }
    let k_wq_shape = inputs.k_wq.shape();
    let v_wq_shape = inputs.v_wq.shape();
    if k_wq_shape.len() != 4 || v_wq_shape.len() != 4 {
        return Err(Exception::custom(
            "steel_quant_attention: k_wq / v_wq must be 4-D",
        ));
    }

    let b = q_shape[0];
    let h_q = q_shape[1];
    let q_len = q_shape[2];
    let d = q_shape[3];
    let h_kv = k_wq_shape[1];
    let k_len = k_wq_shape[2];

    if h_q != inputs.h_q || h_kv != inputs.h_kv {
        return Err(Exception::custom("steel_quant_attention: head mismatch"));
    }
    if inputs.h_q % inputs.h_kv != 0 {
        return Err(Exception::custom("steel_quant_attention: H_q % H_kv != 0"));
    }
    if d != inputs.head_dim {
        return Err(Exception::custom("steel_quant_attention: head_dim mismatch"));
    }
    if !matches!(inputs.bits, 4 | 8) {
        return Err(Exception::custom(format!(
            "steel_quant_attention: bits must be 4 or 8, got {}",
            inputs.bits
        )));
    }
    if d % inputs.group_size != 0 {
        return Err(Exception::custom(format!(
            "steel_quant_attention: head_dim {d} not divisible by group_size {}",
            inputs.group_size
        )));
    }
    if inputs.mask.is_some() {
        return Err(Exception::custom(
            "steel_quant_attention: explicit masks unsupported",
        ));
    }

    let pack_factor = 32 / inputs.bits;
    let expected_wq_d = d / pack_factor;
    let expected_meta_d = d / inputs.group_size;
    if k_wq_shape[3] != expected_wq_d || v_wq_shape[3] != expected_wq_d {
        return Err(Exception::custom(format!(
            "steel_quant_attention: K_wq[3]={} V_wq[3]={} != D/pack_factor={expected_wq_d}",
            k_wq_shape[3], v_wq_shape[3]
        )));
    }
    let k_s_shape = inputs.k_scales.shape();
    let v_s_shape = inputs.v_scales.shape();
    if k_s_shape != [b, h_kv, k_len, expected_meta_d]
        || v_s_shape != [b, h_kv, k_len, expected_meta_d]
    {
        return Err(Exception::custom(format!(
            "steel_quant_attention: scales shape mismatch (expected [{b},{h_kv},{k_len},{expected_meta_d}])"
        )));
    }
    // Scales/biases share the `T` template binding with Q on the GPU.
    let q_dtype = inputs.q.dtype();
    if inputs.k_scales.dtype() != q_dtype
        || inputs.k_biases.dtype() != q_dtype
        || inputs.v_scales.dtype() != q_dtype
        || inputs.v_biases.dtype() != q_dtype
    {
        return Err(Exception::custom(format!(
            "steel_quant_attention: scales/biases dtype must match q dtype ({q_dtype:?}); \
             got k_scales={:?}, k_biases={:?}, v_scales={:?}, v_biases={:?}",
            inputs.k_scales.dtype(),
            inputs.k_biases.dtype(),
            inputs.v_scales.dtype(),
            inputs.v_biases.dtype()
        )));
    }

    let shape = shape_for_d(d).ok_or_else(|| {
        Exception::custom(format!(
            "steel_quant_attention: supported head_dim ∈ {{128, 256, 512}}, got {d}"
        ))
    })?;

    let out_dtype = inputs.q.dtype();
    if !matches!(out_dtype, Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32) {
        return Err(Exception::custom(
            "steel_quant_attention: unsupported q dtype",
        ));
    }
    if d == 512 && out_dtype == Dtype::Float32 {
        return Err(Exception::custom(
            "steel_quant_attention: D=512 requires fp16 or bf16 (fp32 TG-mem exceeds 32 KB)",
        ));
    }

    let flat = FlatAttnParams::from_shapes(
        b,
        h_q,
        h_kv,
        q_len,
        k_len,
        d,
        inputs.scale,
        shape.bq,
        shape.bk,
        inputs.ql_off,
    );

    let dummy_mask = Array::zeros::<u8>(&[1])?.as_dtype(Dtype::Bool)?;
    let mask_present_buf = Array::from_int(0);
    let do_causal_buf = Array::from_int(inputs.causal as i32);

    let nq = (q_len + shape.bq - 1) / shape.bq;
    let threads_per_tg = shape.wm * shape.wn * 32;
    let config = MetalKernelConfig::new()
        .add_output(vec![b, h_q, q_len, d], out_dtype)
        .grid(nq * threads_per_tg, h_q, b)
        .thread_group(threads_per_tg, 1, 1)
        .add_template("T", out_dtype)?
        .add_template("BD", d)?
        .add_template("BQ", shape.bq)?
        .add_template("BK", shape.bk)?
        .add_template("WM", shape.wm)?
        .add_template("WN", shape.wn)?
        .add_template("BITS", inputs.bits)?
        .add_template("GROUP_SIZE", inputs.group_size)?;

    let outs = kernel.apply(
        &[
            inputs.q,
            inputs.k_wq,
            inputs.k_scales,
            inputs.k_biases,
            inputs.v_wq,
            inputs.v_scales,
            inputs.v_biases,
            &dummy_mask,
            flat.b_param(),
            flat.h_param(),
            flat.d_param(),
            flat.q_len_param(),
            flat.k_len_param(),
            flat.gqa_factor_param(),
            flat.scale_param(),
            &mask_present_buf,
            &do_causal_buf,
            flat.nq_param(),
            flat.nk_param(),
            flat.nq_aligned_param(),
            flat.nk_aligned_param(),
            flat.ql_rem_param(),
            flat.kl_rem_param(),
            flat.ql_off_param(),
            flat.q_strides_arr(),
            flat.o_strides_arr(),
        ],
        config,
        Stream::default(),
    )?;
    outs.into_iter()
        .next()
        .ok_or_else(|| Exception::custom("steel_quant_attention: no outputs"))
}
