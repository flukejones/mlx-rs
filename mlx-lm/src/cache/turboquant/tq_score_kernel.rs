//! Fused Metal kernel for the TurboQuant asymmetric attention score
//! (paper Algorithm 2's `<q, x̂>` estimator) computed **without
//! dequantising K** *and* without materialising any intermediate dense
//! tensors.
//!
//! Reads the packed cache state directly (bit-extract in registers) and
//! handles grouped-query attention by mapping query heads to key heads
//! via `h_kv = h_q / N_REP`. Both decisions are critical for performance:
//! materialising a dense `[B, H_kv, n_k, D]` uint8 buffer + replicating
//! it across the query-head axis (the v1 design) costs more memory
//! traffic per decode step than the dense fp16 SDPA primitive avoids.
//!
//! Inputs:
//! - `q_pi`: `[B, H_q, n_q, D]` queries already pre-projected by `Πᵀ`.
//! - `q_s`:  `[B, H_q, n_q, D]` queries already pre-projected by `Sᵀ`.
//! - `mse_indices_packed`: `[B, H_kv, n_k, D_PACKED]` uint8 — the bit-
//!   packed MSE indices straight from the cache store. `D_PACKED =
//!   ceil(D * EFF_BITS / 8)`.
//! - `signs_packed`: `[B, H_kv, n_k, D_SIGNS_PACKED]` uint8 — 1-bit
//!   packed QJL signs. `D_SIGNS_PACKED = ceil(D / 8)`.
//! - `centroids`: `[2^MSE_BITS]` fp32 Lloyd-Max table (MSE_BITS = BITS - 1).
//! - `key_norms`: `[B, H_kv, n_k]` fp32.
//! - `residual_norms`: `[B, H_kv, n_k]` fp32.
//!
//! Scalar buffers (passed via `Array::from_*`): `qjl_scale`, `n_q`, `n_k`.
//!
//! Template constants: `T` (output dtype), `D`, `EFF_BITS`, `D_PACKED`,
//! `D_SIGNS_PACKED`, `N_REP`.
//!
//! Output: `scores`: `[B, H_q, n_q, n_k]` fp32.
//!
//! Each thread computes one `(q_idx, k_idx, b*H_q + h_q)` triple.
//! Threadblock is a 16x16 tile in `(q_idx, k_idx)` space.

use mlx_rs::error::{Exception, Result};
use mlx_rs::fast::{metal_kernel, MetalKernel, MetalKernelConfig};
use mlx_rs::{Array, Dtype, Stream};

const KERNEL_NAME: &str = "turboquant_score_packed_gqa";

/// Metal source. The packed-byte reads use compile-time template
/// constants for the slot widths so the inner loop is branch-free and
/// fully unrolled.
const KERNEL_SOURCE: &str = r#"
    uint q_idx = thread_position_in_grid.x;
    uint k_idx = thread_position_in_grid.y;
    uint bh_q  = thread_position_in_grid.z;     // b * H_q + h_q, flattened
    if (q_idx >= uint(n_q) || k_idx >= uint(n_k)) {
        return;
    }

    // Resolve b and the kv-head index via the grouped-query ratio.
    uint h_q   = bh_q % uint(H_Q);
    uint b_idx = bh_q / uint(H_Q);
    uint h_kv  = h_q / uint(N_REP);
    uint bh_kv = b_idx * uint(H_KV) + h_kv;

    uint q_base       = (bh_q * uint(n_q) + q_idx) * uint(D);
    uint k_mse_base   = (bh_kv * uint(n_k) + k_idx) * uint(D_PACKED);
    uint k_signs_base = (bh_kv * uint(n_k) + k_idx) * uint(D_SIGNS_PACKED);
    uint norms_idx    = bh_kv * uint(n_k) + k_idx;

    constexpr uint EFF_MASK = (1u << uint(EFF_BITS)) - 1u;

    float score_mse = 0.0;
    float score_qjl = 0.0;
    for (uint d = 0; d < uint(D); ++d) {
        // MSE: byte-index = d / (8 / EFF_BITS); slot = d % (8 / EFF_BITS).
        uint vpb_mse = 8u / uint(EFF_BITS);
        uint byte_d  = d / vpb_mse;
        uint slot    = d % vpb_mse;
        uint byte    = uint(mse_indices_packed[k_mse_base + byte_d]);
        uint idx     = (byte >> (slot * uint(EFF_BITS))) & EFF_MASK;
        float c      = float(centroids[idx]);
        float qp     = float(q_pi[q_base + d]);
        score_mse   += qp * c;

        // QJL signs: 1 bit per coord; byte-index = d / 8; bit = d % 8.
        uint sbyte_d = d / 8u;
        uint sbit    = d % 8u;
        uint sbyte   = uint(signs_packed[k_signs_base + sbyte_d]);
        float sgn    = ((sbyte >> sbit) & 1u) == 0u ? -1.0 : 1.0;
        float qs     = float(q_s[q_base + d]);
        score_qjl   += qs * sgn;
    }

    float kn = float(key_norms[norms_idx]);
    float rn = float(residual_norms[norms_idx]);
    float result = score_mse * kn + score_qjl * rn * float(qjl_scale);

    uint out_idx = (bh_q * uint(n_q) + q_idx) * uint(n_k) + k_idx;
    scores[out_idx] = T(result);
"#;

/// Compile a fresh kernel handle. The TurboQuant cache stores one per
/// `TurboQuantProd` and reuses it for every decode step.
pub fn make_tq_score_kernel() -> Result<MetalKernel> {
    metal_kernel(
        KERNEL_NAME,
        &[
            "q_pi",
            "q_s",
            "mse_indices_packed",
            "signs_packed",
            "centroids",
            "key_norms",
            "residual_norms",
            "qjl_scale",
            "n_q",
            "n_k",
        ],
        &["scores"],
        KERNEL_SOURCE,
        "",
        true,
        false,
    )
}

/// Inputs to [`tq_attention_score_kernel`].
pub struct TqScoreInputs<'a> {
    /// `q · Πᵀ`, shape `[B, H_q, n_q, D]`.
    pub q_pi: &'a Array,
    /// `q · Sᵀ`, shape `[B, H_q, n_q, D]`.
    pub q_s: &'a Array,
    /// Bit-packed MSE indices, shape `[B, H_kv, n_k, D_PACKED]` uint8.
    pub mse_indices_packed: &'a Array,
    /// Bit-packed 1-bit QJL signs, shape `[B, H_kv, n_k, D_SIGNS_PACKED]`.
    pub signs_packed: &'a Array,
    /// Codebook centroids `[2^MSE_BITS]` fp32.
    pub centroids: &'a Array,
    /// Per-key L2 norms `[B, H_kv, n_k]` fp32.
    pub key_norms: &'a Array,
    /// Per-key residual L2 norms `[B, H_kv, n_k]` fp32.
    pub residual_norms: &'a Array,
    /// QJL scale `√(π/2) / D`.
    pub qjl_scale: f32,
    /// Head dim.
    pub d: i32,
    /// Effective bits per index in the packed store (4 for bits ∈ {3, 4},
    /// 2 for bits=2, 1 for bits=1). Determines how indices unpack from
    /// the bytes inside the kernel.
    pub eff_bits: i32,
    /// Bytes per row in the packed MSE store (`ceil(D * eff_bits / 8)`).
    pub d_packed: i32,
    /// Bytes per row in the packed signs store (`ceil(D / 8)`).
    pub d_signs_packed: i32,
    /// Number of query heads in the model.
    pub h_q: i32,
    /// Number of key/value heads in the model.
    pub h_kv: i32,
}

/// Compute the fused asymmetric attention scores. Returns `[B, H_q,
/// n_q, n_k]` fp32.
pub fn tq_attention_score_kernel(
    kernel: &MetalKernel,
    inputs: TqScoreInputs<'_>,
) -> Result<Array> {
    let q_shape = inputs.q_pi.shape();
    if q_shape.len() != 4 {
        return Err(Exception::custom(format!(
            "tq_attention_score_kernel: q_pi must be 4-D [B, H_q, n_q, D], got {q_shape:?}"
        )));
    }
    let b = q_shape[0];
    let h_q = q_shape[1];
    let n_q = q_shape[2];
    let n_k = inputs.mse_indices_packed.shape()[2];

    if h_q != inputs.h_q {
        return Err(Exception::custom(format!(
            "tq_attention_score_kernel: q_pi[1]={h_q} != cfg h_q={}",
            inputs.h_q
        )));
    }
    if inputs.h_q % inputs.h_kv != 0 {
        return Err(Exception::custom(format!(
            "tq_attention_score_kernel: H_q={} not divisible by H_KV={}",
            inputs.h_q, inputs.h_kv
        )));
    }
    let n_rep = inputs.h_q / inputs.h_kv;

    let dtype = Dtype::Float32;
    let scale_buf = Array::from_f32(inputs.qjl_scale);
    let n_q_buf = Array::from_int(n_q);
    let n_k_buf = Array::from_int(n_k);

    // 16x16 tile in (q_idx, k_idx); grid.z covers b * H_q.
    const TG_X: i32 = 16;
    const TG_Y: i32 = 16;
    let g_x = n_q.saturating_add(TG_X - 1) / TG_X * TG_X;
    let g_y = n_k.saturating_add(TG_Y - 1) / TG_Y * TG_Y;
    let g_z = b * inputs.h_q;

    let config = MetalKernelConfig::new()
        .add_output(vec![b, inputs.h_q, n_q, n_k], dtype)
        .grid(g_x, g_y, g_z)
        .thread_group(TG_X, TG_Y, 1)
        .add_template("T", dtype)?
        .add_template("D", inputs.d)?
        .add_template("EFF_BITS", inputs.eff_bits)?
        .add_template("D_PACKED", inputs.d_packed)?
        .add_template("D_SIGNS_PACKED", inputs.d_signs_packed)?
        .add_template("H_Q", inputs.h_q)?
        .add_template("H_KV", inputs.h_kv)?
        .add_template("N_REP", n_rep)?;

    // Buffer argument order must match the names in make_tq_score_kernel.
    let outs = kernel.apply(
        &[
            inputs.q_pi.clone(),
            inputs.q_s.clone(),
            inputs.mse_indices_packed.clone(),
            inputs.signs_packed.clone(),
            inputs.centroids.clone(),
            inputs.key_norms.clone(),
            inputs.residual_norms.clone(),
            scale_buf,
            n_q_buf,
            n_k_buf,
        ],
        config,
        Stream::default(),
    )?;
    outs.into_iter()
        .next()
        .ok_or_else(|| Exception::custom("tq_attention_score_kernel: no outputs"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::transforms::eval;

    /// Slow scalar reference that mirrors the kernel formula exactly,
    /// reading the same packed buffers via the same bit-extraction.
    /// Used to validate the kernel without depending on the higher-
    /// level `TurboQuantProd` (which would change with the kernel
    /// surface).
    #[allow(clippy::too_many_arguments)]
    fn scalar_reference(
        b: i32,
        h_q: i32,
        h_kv: i32,
        n_q: i32,
        n_k: i32,
        d: i32,
        eff_bits: i32,
        d_packed: i32,
        d_signs_packed: i32,
        q_pi: &[f32],
        q_s: &[f32],
        mse_packed: &[u8],
        signs_packed: &[u8],
        centroids: &[f32],
        key_norms: &[f32],
        res_norms: &[f32],
        qjl_scale: f32,
    ) -> Vec<f32> {
        let n_rep = h_q / h_kv;
        let mask = (1u32 << eff_bits as u32) - 1;
        let vpb_mse = 8 / eff_bits;
        let mut out = vec![0.0f32; (b * h_q * n_q * n_k) as usize];
        for bi in 0..b {
            for h_q_i in 0..h_q {
                let h_kv_i = h_q_i / n_rep;
                let bh_q = (bi * h_q + h_q_i) as usize;
                let bh_kv = (bi * h_kv + h_kv_i) as usize;
                for qi in 0..n_q {
                    for ki in 0..n_k {
                        let q_base = (bh_q * n_q as usize + qi as usize) * d as usize;
                        let k_mse_base = (bh_kv * n_k as usize + ki as usize) * d_packed as usize;
                        let k_signs_base =
                            (bh_kv * n_k as usize + ki as usize) * d_signs_packed as usize;
                        let norms_idx = bh_kv * n_k as usize + ki as usize;
                        let mut s_mse = 0.0f32;
                        let mut s_qjl = 0.0f32;
                        for di in 0..d as usize {
                            let byte_d = di / vpb_mse as usize;
                            let slot = di % vpb_mse as usize;
                            let byte = mse_packed[k_mse_base + byte_d] as u32;
                            let idx = (byte >> (slot as u32 * eff_bits as u32)) & mask;
                            s_mse += q_pi[q_base + di] * centroids[idx as usize];

                            let sbyte_d = di / 8;
                            let sbit = di % 8;
                            let sbyte = signs_packed[k_signs_base + sbyte_d] as u32;
                            let sgn = if (sbyte >> sbit) & 1 == 0 { -1.0 } else { 1.0 };
                            s_qjl += q_s[q_base + di] * sgn;
                        }
                        let out_idx = ((bh_q * n_q as usize) + qi as usize) * n_k as usize
                            + ki as usize;
                        out[out_idx] =
                            s_mse * key_norms[norms_idx] + s_qjl * res_norms[norms_idx] * qjl_scale;
                    }
                }
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn make_random_inputs(
        b: i32,
        h_q: i32,
        h_kv: i32,
        n_q: i32,
        n_k: i32,
        d: i32,
        eff_bits: i32,
    ) -> (
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        f32,
        i32,
        i32,
    ) {
        let mask = (1u32 << eff_bits as u32) - 1;
        let vpb_mse = 8 / eff_bits;
        let d_packed = (d + vpb_mse - 1) / vpb_mse;
        let d_signs_packed = (d + 7) / 8;
        let n_clusters = 1 << eff_bits;

        let prng = mlx_rs::random::key(123).unwrap();
        let q_pi = mlx_rs::random::normal::<f32>(&[b, h_q, n_q, d], None, None, &prng).unwrap();
        let prng = mlx_rs::random::key(456).unwrap();
        let q_s = mlx_rs::random::normal::<f32>(&[b, h_q, n_q, d], None, None, &prng).unwrap();

        // Build packed MSE indices by drawing uint8 in [0, n_clusters) per coord
        // then packing.
        let prng = mlx_rs::random::key(789).unwrap();
        let mse_per_coord = mlx_rs::random::uniform::<_, f32>(
            0.0,
            n_clusters as f32,
            &[b, h_kv, n_k, d],
            &prng,
        )
        .unwrap()
        .as_dtype(Dtype::Uint8)
        .unwrap();
        let mse_packed =
            super::super::packing::pack_indices(&mse_per_coord, eff_bits).unwrap();
        // Build random ±1 signs and pack.
        let prng = mlx_rs::random::key(101).unwrap();
        let raw_signs =
            mlx_rs::random::normal::<f32>(&[b, h_kv, n_k, d], None, None, &prng).unwrap();
        let signs_packed = super::super::packing::pack_signs(&raw_signs).unwrap();

        let cents: Vec<f32> = (0..n_clusters)
            .map(|i| -1.0 + 2.0 * (i as f32 + 0.5) / n_clusters as f32)
            .collect();
        let centroids = Array::from_slice(&cents, &[n_clusters]);
        let prng = mlx_rs::random::key(202).unwrap();
        let key_norms =
            mlx_rs::random::uniform::<_, f32>(0.1f32, 1.0f32, &[b, h_kv, n_k], &prng).unwrap();
        let prng = mlx_rs::random::key(303).unwrap();
        let res_norms =
            mlx_rs::random::uniform::<_, f32>(0.0f32, 0.5f32, &[b, h_kv, n_k], &prng).unwrap();
        let qjl_scale = (std::f32::consts::FRAC_PI_2.sqrt()) / d as f32;
        let _ = mask;
        (
            q_pi,
            q_s,
            mse_packed,
            signs_packed,
            centroids,
            key_norms,
            res_norms,
            qjl_scale,
            d_packed,
            d_signs_packed,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_and_compare(b: i32, h_q: i32, h_kv: i32, n_q: i32, n_k: i32, d: i32, eff_bits: i32) {
        let (
            q_pi,
            q_s,
            mse_packed,
            signs_packed,
            centroids,
            key_norms,
            res_norms,
            qjl_scale,
            d_packed,
            d_signs_packed,
        ) = make_random_inputs(b, h_q, h_kv, n_q, n_k, d, eff_bits);

        // Run scalar reference on host data.
        eval([&q_pi, &q_s, &mse_packed, &signs_packed, &centroids, &key_norms, &res_norms])
            .unwrap();
        let expected = scalar_reference(
            b,
            h_q,
            h_kv,
            n_q,
            n_k,
            d,
            eff_bits,
            d_packed,
            d_signs_packed,
            q_pi.as_slice::<f32>(),
            q_s.as_slice::<f32>(),
            mse_packed.as_slice::<u8>(),
            signs_packed.as_slice::<u8>(),
            centroids.as_slice::<f32>(),
            key_norms.as_slice::<f32>(),
            res_norms.as_slice::<f32>(),
            qjl_scale,
        );

        let kernel = make_tq_score_kernel().unwrap();
        let got = tq_attention_score_kernel(
            &kernel,
            TqScoreInputs {
                q_pi: &q_pi,
                q_s: &q_s,
                mse_indices_packed: &mse_packed,
                signs_packed: &signs_packed,
                centroids: &centroids,
                key_norms: &key_norms,
                residual_norms: &res_norms,
                qjl_scale,
                d,
                eff_bits,
                d_packed,
                d_signs_packed,
                h_q,
                h_kv,
            },
        )
        .unwrap();
        eval([&got]).unwrap();
        let got_vec = got.as_slice::<f32>().to_vec();

        let tol = 1e-3 * (d as f32);
        let mut max_err = 0.0f32;
        for (a, b) in got_vec.iter().zip(expected.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(
            max_err < tol,
            "kernel vs scalar diverged: max abs = {max_err}, tol = {tol}"
        );
    }

    /// 4-bit packing (paper's K=4 V=4 config). H_q == H_kv (non-GQA).
    #[test]
    fn kernel_packed_no_gqa_4bit() {
        run_and_compare(1, 2, 2, 4, 8, 128, 4);
    }

    /// 2-bit packing — every byte holds 4 indices.
    #[test]
    fn kernel_packed_no_gqa_2bit() {
        run_and_compare(1, 2, 2, 4, 8, 128, 2);
    }

    /// GQA: H_q = 4, H_kv = 2, n_rep = 2. The kv-head index inside the
    /// kernel is `h_q / n_rep`.
    #[test]
    fn kernel_packed_gqa_2_to_4_heads() {
        run_and_compare(1, 4, 2, 4, 8, 128, 4);
    }

    /// Heavier GQA: H_q = 16, H_kv = 8, n_rep = 2 (Qwen3-1.7B shape).
    #[test]
    fn kernel_packed_gqa_qwen3_shape() {
        run_and_compare(1, 16, 8, 4, 8, 128, 4);
    }

    /// Long n_k — closer to a real long-prompt shape.
    #[test]
    fn kernel_packed_gqa_long_n_k() {
        run_and_compare(1, 16, 8, 1, 256, 128, 4);
    }

    /// Non-tile-aligned shapes — bounds check engages.
    #[test]
    fn kernel_packed_handles_non_tile_aligned() {
        run_and_compare(1, 4, 2, 7, 11, 128, 4);
    }
}
