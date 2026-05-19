//! Fused `gather_qmm(down_w) + expert_combine` Metal kernel for the
//! gemma4 MoE down-projection + expert-weighted sum.
//!
//! Replaces the 2-launch sequence:
//!   `y = gather_qmm(activated, down_wq, ..., indices)`     // [..., K, D]
//!   `out = (weights * y).sum(-2)`                          // [..., D]
//! with a single Metal launch that computes:
//!   `out[d] = sum_k weights[k] * sum_h (down_w[idx[k], d, h] * activated[k, h])`
//!
//! Decode-first design (S=1): one threadgroup per output-position group;
//! each thread owns one output dim, iterates K experts, and accumulates
//! per-expert qdot scaled by the router weight. The activated[k, :]
//! vector is shared across all output dims of the TG and loaded into
//! threadgroup memory once per expert.

use std::sync::OnceLock;

use mlx_rs::error::{Exception, Result};
use mlx_rs::fast::{metal_kernel, MetalKernel, MetalKernelConfig};
use mlx_rs::{Array, Dtype, Stream};

const KERNEL_NAME: &str = "fused_gather_qmm_combine_v0";

const KERNEL_SOURCE: &str = r#"
    // Inputs:
    //   activated : [T, K, H]            -- T = product(leading dims), F32-castable
    //   weights   : [T, K]               -- router weights (already softmaxed)
    //   wq        : [E, D, H/pack]       -- packed-uint32 down-projection weight
    //   scales    : [E, D, H/group]      -- per-group dequant scale
    //   biases    : [E, D, H/group]      -- per-group dequant bias
    //   indices   : [T, K]               -- expert id per (token, k)
    // Output:
    //   out       : [T, D]
    //
    // Template ints: D, H, K, GROUP_SIZE, BITS, BD (output dims per TG).
    // BD = 128 by default → 1 TG covers 128 output dims with 128 threads.

    constexpr uint BD_C = uint(BD);
    constexpr uint H_C  = uint(H);
    constexpr uint K_C  = uint(K);
    constexpr uint GS   = uint(GROUP_SIZE);
    constexpr uint BITS_C = uint(BITS);
    constexpr uint PACK_FACTOR = 32u / BITS_C;       // bits=8 -> 4, bits=4 -> 8
    constexpr uint WORDS_PER_ROW = H_C / PACK_FACTOR; // uint32s per (e,d)
    constexpr uint GROUPS_PER_ROW = H_C / GS;
    constexpr uint MASK_BITS = (1u << BITS_C) - 1u;

    // One TG per (t, d_block). Grid laid out: x = d_block * BD_C, y = t.
    uint tid     = thread_index_in_threadgroup;
    uint d_block = threadgroup_position_in_grid.x;
    uint t       = threadgroup_position_in_grid.y;
    uint d       = d_block * BD_C + tid;

    // Shared cache for the current expert's activation row.
    threadgroup float act_smem[H_C];

    float acc = 0.0f;

    // Iterate experts. Each expert k contributes
    //   weights[t,k] * dot(activated[t,k,:], dequant(wq[idx[t,k], d, :]))
    for (uint k = 0; k < K_C; ++k) {
        // Stage activated[t, k, :] into smem cooperatively.
        uint act_base = (t * K_C + k) * H_C;
        for (uint i = tid; i < H_C; i += BD_C) {
            act_smem[i] = float(activated[act_base + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (d < uint(D)) {
            uint expert_id = uint(indices[t * K_C + k]);

            // Per-group accumulators: dot = sum_g (scale_g * raw_sum_g + bias_g * sum_acts_g)
            // where raw_sum_g = sum over h in group of code(h) * act(h)
            // and   sum_acts_g = sum over h in group of act(h)  (for bias term).
            float row_dot = 0.0f;

            uint wq_row_base    = (expert_id * uint(D) + d) * WORDS_PER_ROW;
            uint meta_row_base  = (expert_id * uint(D) + d) * GROUPS_PER_ROW;

            for (uint g = 0; g < GROUPS_PER_ROW; ++g) {
                float scale_v = float(scales[meta_row_base + g]);
                float bias_v  = float(biases[meta_row_base + g]);

                float raw_sum     = 0.0f;
                float sum_acts    = 0.0f;
                uint g_h_start    = g * GS;
                uint words_per_g  = GS / PACK_FACTOR;
                uint wq_g_base    = wq_row_base + g * words_per_g;

                for (uint w = 0; w < words_per_g; ++w) {
                    uint packed = wq[wq_g_base + w];
                    uint h_base = g_h_start + w * PACK_FACTOR;
                    for (uint p = 0; p < PACK_FACTOR; ++p) {
                        uint code = (packed >> (p * BITS_C)) & MASK_BITS;
                        float a = act_smem[h_base + p];
                        raw_sum  += float(code) * a;
                        sum_acts += a;
                    }
                }

                row_dot += scale_v * raw_sum + bias_v * sum_acts;
            }

            float w_k = float(weights[t * K_C + k]);
            acc += w_k * row_dot;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (d < uint(D)) {
        out[t * uint(D) + d] = T(acc);
    }
"#;

const INPUT_NAMES: &[&str] = &["activated", "weights", "wq", "scales", "biases", "indices"];
const OUTPUT_NAMES: &[&str] = &["out"];

pub fn make_gather_qmm_combine_kernel() -> Result<MetalKernel> {
    metal_kernel(
        KERNEL_NAME,
        INPUT_NAMES,
        OUTPUT_NAMES,
        KERNEL_SOURCE,
        "",
        true,
        false,
    )
}

pub fn cached_gather_qmm_combine_kernel() -> &'static MetalKernel {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        make_gather_qmm_combine_kernel().expect("make_gather_qmm_combine_kernel")
    })
}

/// Inputs to [`gather_qmm_combine`].
pub struct GatherQmmCombineInputs<'a> {
    /// `[..., K, H]` activations per (token, expert).
    pub activated: &'a Array,
    /// `[..., K]` router weights (post-softmax).
    pub weights: &'a Array,
    /// `[E, D, H/pack]` packed-uint32 down-projection weight.
    pub wq: &'a Array,
    /// `[E, D, H/group_size]` dequant scales (same dtype as `activated`).
    pub scales: &'a Array,
    /// `[E, D, H/group_size]` dequant biases (same dtype as `activated`).
    pub biases: &'a Array,
    /// `[..., K]` expert ids.
    pub indices: &'a Array,
    pub group_size: i32,
    pub bits: i32,
}

/// Launch the fused down-projection + expert-combine kernel. Returns
/// `[..., D]` in the activation dtype.
pub fn gather_qmm_combine(inputs: GatherQmmCombineInputs<'_>) -> Result<Array> {
    let act_shape = inputs.activated.shape();
    let act_rank = act_shape.len();
    if act_rank < 3 {
        return Err(Exception::custom(format!(
            "gather_qmm_combine: activated must have ≥3 dims [..., K, H], got {act_shape:?}"
        )));
    }
    let h = act_shape[act_rank - 1];
    let k = act_shape[act_rank - 2];
    let t: i32 = act_shape[..act_rank - 2].iter().product();

    let w_shape = inputs.weights.shape();
    if w_shape.last().copied() != Some(k) {
        return Err(Exception::custom(format!(
            "gather_qmm_combine: weights trailing dim {:?} != K={k}",
            w_shape.last()
        )));
    }

    let wq_shape = inputs.wq.shape();
    if wq_shape.len() != 3 {
        return Err(Exception::custom(format!(
            "gather_qmm_combine: wq must be 3-D [E,D,H/pack], got {wq_shape:?}"
        )));
    }
    let e = wq_shape[0];
    let d = wq_shape[1];

    let pack_factor = 32 / inputs.bits;
    let expected_words = h / pack_factor;
    if wq_shape[2] != expected_words {
        return Err(Exception::custom(format!(
            "gather_qmm_combine: wq H/pack={} != H/pack_factor={expected_words}",
            wq_shape[2]
        )));
    }

    let groups_per_row = h / inputs.group_size;
    let scales_shape = inputs.scales.shape();
    if scales_shape != [e, d, groups_per_row] {
        return Err(Exception::custom(format!(
            "gather_qmm_combine: scales shape {scales_shape:?} != [{e}, {d}, {groups_per_row}]"
        )));
    }
    let biases_shape = inputs.biases.shape();
    if biases_shape != [e, d, groups_per_row] {
        return Err(Exception::custom(format!(
            "gather_qmm_combine: biases shape {biases_shape:?} != [{e}, {d}, {groups_per_row}]"
        )));
    }

    let idx_shape = inputs.indices.shape();
    if idx_shape.last().copied() != Some(k) {
        return Err(Exception::custom(format!(
            "gather_qmm_combine: indices trailing dim {:?} != K={k}",
            idx_shape.last()
        )));
    }

    if h % inputs.group_size != 0 {
        return Err(Exception::custom(format!(
            "gather_qmm_combine: H={h} not divisible by group_size={}",
            inputs.group_size
        )));
    }

    let dtype = inputs.activated.dtype();
    if !matches!(dtype, Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32) {
        return Err(Exception::custom(format!(
            "gather_qmm_combine: unsupported activation dtype {dtype:?}"
        )));
    }

    // Output is `[..., D]` — drop the K axis from activated's leading shape.
    let mut out_shape = act_shape[..act_rank - 2].to_vec();
    out_shape.push(d);

    const BD: i32 = 128;
    let d_blocks = (d + BD - 1) / BD;

    let config = MetalKernelConfig::new()
        .add_output(out_shape, dtype)
        .grid(d_blocks * BD, t, 1)
        .thread_group(BD, 1, 1)
        .add_template("T", dtype)?
        .add_template("D", d)?
        .add_template("H", h)?
        .add_template("K", k)?
        .add_template("BD", BD)?
        .add_template("GROUP_SIZE", inputs.group_size)?
        .add_template("BITS", inputs.bits)?;

    let outs = cached_gather_qmm_combine_kernel().apply(
        &[
            inputs.activated,
            inputs.weights,
            inputs.wq,
            inputs.scales,
            inputs.biases,
            inputs.indices,
        ],
        config,
        Stream::default(),
    )?;
    outs.into_iter()
        .next()
        .ok_or_else(|| Exception::custom("gather_qmm_combine: no outputs"))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    #![allow(clippy::print_stdout, reason = "test code")]
    #![allow(clippy::print_stderr, reason = "test code")]
    use super::*;
    use mlx_rs::ops::{expand_dims_axes, gather_qmm, quantize, sum_axes};
    use mlx_rs::random::uniform;
    use mlx_rs::transforms::eval;

    fn max_abs_diff(a: &Array, b: &Array) -> f32 {
        a.subtract(b).unwrap().abs().unwrap().max(None).unwrap().item::<f32>()
    }

    /// Reference: gather_qmm then (weights * y).sum(-2) the slow way.
    fn reference(inputs: GatherQmmCombineInputs<'_>) -> Array {
        let act_exp = expand_dims_axes(inputs.activated, &[-2]).unwrap();
        let y = gather_qmm(
            &act_exp,
            inputs.wq,
            inputs.scales,
            Some(inputs.biases),
            None,
            Some(inputs.indices),
            Some(true),
            Some(inputs.group_size),
            Some(inputs.bits),
            Some(false),
        )
        .unwrap();
        let y = y.squeeze_axes(&[-2]).unwrap();
        let weights_exp = expand_dims_axes(inputs.weights, &[-1]).unwrap();
        sum_axes(weights_exp.multiply(&y).unwrap(), &[-2], false).unwrap()
    }

    #[test]
    fn matches_gather_qmm_path_f32_tiny() {
        // E=4, D=64, H=128, K=2, T=3.
        let e = 4_i32;
        let d = 64_i32;
        let h = 128_i32;
        let k = 2_i32;
        let t = 3_i32;
        let group_size = 64_i32;
        let bits = 8_i32;

        let w_full = uniform::<_, f32>(-0.1, 0.1, &[e, d, h], None).unwrap();
        let (wq, scales, biases) = quantize(&w_full, group_size, bits).unwrap();

        let activated = uniform::<_, f32>(-0.5, 0.5, &[t, k, h], None).unwrap();
        let weights = uniform::<_, f32>(0.0, 1.0, &[t, k], None).unwrap();
        // indices: int32 in [0, E).
        let idx_vec: Vec<i32> = (0..(t * k)).map(|i| i % e).collect();
        let indices = Array::from_slice(&idx_vec, &[t, k]);

        let fused = gather_qmm_combine(GatherQmmCombineInputs {
            activated: &activated,
            weights: &weights,
            wq: &wq,
            scales: &scales,
            biases: &biases,
            indices: &indices,
            group_size,
            bits,
        })
        .unwrap();
        let reference = reference(GatherQmmCombineInputs {
            activated: &activated,
            weights: &weights,
            wq: &wq,
            scales: &scales,
            biases: &biases,
            indices: &indices,
            group_size,
            bits,
        });
        eval([&fused, &reference]).unwrap();

        let err = max_abs_diff(&fused, &reference);
        assert!(err < 1e-3, "fused vs gather_qmm path max_abs_diff={err}");
    }

    #[test]
    fn matches_gather_qmm_path_bf16_decoder_shape() {
        // gemma4 26B-A4B MoE decode shape: T=1, K=8, H=704, D=2816, E=128.
        let e = 128_i32;
        let d = 2816_i32;
        let h = 704_i32;
        let k = 8_i32;
        let t = 1_i32;
        let group_size = 64_i32;
        let bits = 8_i32;

        let w_full = uniform::<_, f32>(-0.05, 0.05, &[e, d, h], None).unwrap();
        let (wq, scales_f32, biases_f32) = quantize(&w_full, group_size, bits).unwrap();
        let scales = scales_f32.as_dtype(Dtype::Bfloat16).unwrap();
        let biases = biases_f32.as_dtype(Dtype::Bfloat16).unwrap();

        let activated = uniform::<_, f32>(-0.5, 0.5, &[t, k, h], None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let weights = uniform::<_, f32>(0.0, 1.0, &[t, k], None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let idx_vec: Vec<i32> = (0..(t * k)).map(|i| i % e).collect();
        let indices = Array::from_slice(&idx_vec, &[t, k]);

        let fused = gather_qmm_combine(GatherQmmCombineInputs {
            activated: &activated,
            weights: &weights,
            wq: &wq,
            scales: &scales,
            biases: &biases,
            indices: &indices,
            group_size,
            bits,
        })
        .unwrap();
        let reference = reference(GatherQmmCombineInputs {
            activated: &activated,
            weights: &weights,
            wq: &wq,
            scales: &scales,
            biases: &biases,
            indices: &indices,
            group_size,
            bits,
        });
        eval([&fused, &reference]).unwrap();

        let err = max_abs_diff(&fused, &reference);
        // bf16 + many-term reduction; allow loose tolerance.
        assert!(err < 1e-1, "bf16 fused vs reference max_abs_diff={err}");
    }
}
