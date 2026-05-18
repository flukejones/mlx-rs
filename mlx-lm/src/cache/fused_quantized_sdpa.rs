//! Fused qsdpa Metal kernel: `softmax((Q @ K.T) * scale + mask) @ V`
//! end-to-end with K/V held as packed `(wq, scales, biases)` triples.
//! n_q=1 decode only; prefill falls back to ops-composed.

use mlx_rs::error::{Exception, Result};
use mlx_rs::fast::{metal_kernel, MetalKernel, MetalKernelConfig};
use mlx_rs::{Array, Dtype, Stream};

const KERNEL_NAME: &str = "fused_qsdpa_decode_v10";

const KERNEL_SOURCE: &str = r#"
    // Adapted from mlx sdpa_vector + qdot pattern.
    // BN simdgroups × BD threads-per-simdgroup. One TG per (h_q, b).
    // simd_gid = which K-token stripe this simdgroup is on (advances by BN).
    // simd_lid = which D-slice this thread owns within a K-token (32 lanes).
    constexpr uint BN = 32;
    constexpr uint BD = 32;
    constexpr uint QK_PER_THREAD = uint(D) / BD;
    constexpr uint V_PER_THREAD  = uint(D) / BD;
    constexpr uint PACK_FACTOR   = 32u / uint(BITS);
    constexpr uint GROUPS_PER    = uint(D) / uint(GROUP_SIZE);
    constexpr uint WORDS_PER     = uint(D) / PACK_FACTOR;
    constexpr uint BYTES_PER_PACK = 4u; // uint32 wq -> 4 bytes per pack
    constexpr uint VALS_PER_BYTE = 8u / uint(BITS); // bits=4 -> 2; bits=8 -> 1

    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    uint h_q      = thread_position_in_grid.x / uint(TG_SIZE);
    uint b_idx    = thread_position_in_grid.y;

    uint h_kv  = h_q / uint(N_REP);
    uint bh_q  = b_idx * uint(H_Q) + h_q;
    uint bh_kv = b_idx * uint(H_KV) + h_kv;
    uint nk    = uint(n_k);

    // Each thread owns a D-slice: simd_lid * QK_PER_THREAD .. +QK_PER_THREAD.
    uint d_start = simd_lid * QK_PER_THREAD;

    // Pre-scale Q values for this thread's slice. For 4-bit we also
    // divide each by `1, 16, 256, 4096, ...` so that the `qdot` masking
    // trick can compute scale*code without bit shifts.
    float q_pre[QK_PER_THREAD];
    {
        uint q_base = bh_q * uint(D) + d_start;
        if (uint(BITS) == 4u) {
            // qdot pattern: q[4i+0] keeps scale, q[4i+1]/=16, q[4i+2]/=256, q[4i+3]/=4096
            for (uint i = 0; i < QK_PER_THREAD; i += 4u) {
                q_pre[i + 0] = float(q[q_base + i + 0]) * scale;
                q_pre[i + 1] = float(q[q_base + i + 1]) * scale / 16.0f;
                q_pre[i + 2] = float(q[q_base + i + 2]) * scale / 256.0f;
                q_pre[i + 3] = float(q[q_base + i + 3]) * scale / 4096.0f;
            }
        } else {
            for (uint i = 0; i < QK_PER_THREAD; ++i) {
                q_pre[i] = float(q[q_base + i]) * scale;
            }
        }
    }

    // Running online softmax state (per thread, but only simd_lid==0 of
    // each simdgroup matters before the final aggregation).
    float max_score = -INFINITY;
    float sum_exp_score = 0.0;
    float o_partial[V_PER_THREAD];
    for (uint i = 0; i < V_PER_THREAD; ++i) {
        o_partial[i] = 0.0;
    }

    // Main loop: simdgroup_id strides across n_k by BN.
    for (uint k_idx = simd_gid; k_idx < nk; k_idx += BN) {
        // ----- Load K-slice for this thread's D-coords and compute partial dot -----
        // bytes_per_pack = 4 (uint32 = 4 bytes); each byte holds VALS_PER_BYTE codes.
        // For 4-bit, this thread owns 4 D-coords = 2 bytes = half of one uint32.
        uint k_wq_base   = (bh_kv * nk + k_idx) * WORDS_PER;
        uint k_meta_base = (bh_kv * nk + k_idx) * GROUPS_PER;

        float partial;
        if (uint(BITS) == 4u) {
            // 4 codes per thread → 2 bytes → 1 uint16 lane.
            // qdot masking trick: x[0]*(w & 0x000f) + x[1]*(w & 0x00f0) + ...
            uint code_byte_idx = d_start / VALS_PER_BYTE; // 4 coords = 2 bytes
            // Read 2 bytes as one uint16 from the packed wq.
            // wq is uint32; treat it as uint8 array via byte offset.
            const device uint8_t* k_bytes =
                (const device uint8_t*)(k_wq + k_wq_base);
            uint16_t w16 = uint16_t(k_bytes[code_byte_idx]) |
                           (uint16_t(k_bytes[code_byte_idx + 1]) << 8);
            uint group_d = d_start / uint(GROUP_SIZE);
            float sc = float(k_scales[k_meta_base + group_d]);
            float bi = float(k_biases[k_meta_base + group_d]);
            // Undo the per-lane 1/16/256/4096 pre-scale on q to recover sum_x
            // for the bias term: sum_x = q_pre[0] + q_pre[1]*16 + q_pre[2]*256
            // + q_pre[3]*4096.
            float sum_q = q_pre[0] + q_pre[1] * 16.0f
                        + q_pre[2] * 256.0f + q_pre[3] * 4096.0f;
            // qdot 4-bit:
            float accum = q_pre[0] * float(w16 & 0x000fu)
                        + q_pre[1] * float(w16 & 0x00f0u)
                        + q_pre[2] * float(w16 & 0x0f00u)
                        + q_pre[3] * float(w16 & 0xf000u);
            partial = sc * accum + bi * sum_q;
        } else {
            // 8-bit: pack_factor=4, bytes_per_pack=4. 4 D-coords = 4 bytes.
            const device uint8_t* k_bytes =
                (const device uint8_t*)(k_wq + k_wq_base);
            uint group_d = d_start / uint(GROUP_SIZE);
            float sc = float(k_scales[k_meta_base + group_d]);
            float bi = float(k_biases[k_meta_base + group_d]);
            float accum = 0.0;
            float sum_q = 0.0;
            for (uint i = 0; i < QK_PER_THREAD; ++i) {
                float code = float(k_bytes[d_start + i]);
                accum += q_pre[i] * code;
                sum_q += q_pre[i];
            }
            partial = sc * accum + bi * sum_q;
        }

        // Reduce partial across the 32 lanes of this simdgroup → full score.
        float score = simd_sum(partial);

        // Mask path (bool).
        if (mask_present != 0) {
            uint m_idx = (b_idx * uint(H_Q) + h_q) * nk + k_idx;
            if (mask[m_idx] == 0u) {
                score = -INFINITY;
            }
        }

        // ----- Online softmax + V dequant + output accumulation -----
        // All threads in this simdgroup hold the same score now.
        float new_max = max(max_score, score);
        float factor  = (max_score == -INFINITY) ? 0.0f : exp(max_score - new_max);
        float exp_score = (score == -INFINITY) ? 0.0f : exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        // Dequant V-slice owned by this thread (4 coords).
        uint v_wq_base   = (bh_kv * nk + k_idx) * WORDS_PER;
        uint v_meta_base = (bh_kv * nk + k_idx) * GROUPS_PER;
        float v_thread[V_PER_THREAD];
        if (uint(BITS) == 4u) {
            uint code_byte_idx = d_start / VALS_PER_BYTE;
            const device uint8_t* v_bytes =
                (const device uint8_t*)(v_wq + v_wq_base);
            uint group_d = d_start / uint(GROUP_SIZE);
            float sc = float(v_scales[v_meta_base + group_d]);
            float bi = float(v_biases[v_meta_base + group_d]);
            uint8_t b0 = v_bytes[code_byte_idx];
            uint8_t b1 = v_bytes[code_byte_idx + 1];
            v_thread[0] = sc * float(b0 & 0x0fu) + bi;
            v_thread[1] = sc * float((b0 >> 4) & 0x0fu) + bi;
            v_thread[2] = sc * float(b1 & 0x0fu) + bi;
            v_thread[3] = sc * float((b1 >> 4) & 0x0fu) + bi;
        } else {
            const device uint8_t* v_bytes =
                (const device uint8_t*)(v_wq + v_wq_base);
            uint group_d = d_start / uint(GROUP_SIZE);
            float sc = float(v_scales[v_meta_base + group_d]);
            float bi = float(v_biases[v_meta_base + group_d]);
            for (uint i = 0; i < V_PER_THREAD; ++i) {
                v_thread[i] = sc * float(v_bytes[d_start + i]) + bi;
            }
        }
        for (uint i = 0; i < V_PER_THREAD; ++i) {
            o_partial[i] = o_partial[i] * factor + exp_score * v_thread[i];
        }
    }

    // ----- Cross-simdgroup aggregation -----
    // Each simdgroup has its own (max_score, sum_exp_score, o_partial[4])
    // for its stripe of K-tokens. Need to combine across BN=32 simdgroups.
    threadgroup float tg_max[BN];
    threadgroup float tg_sum[BN];
    threadgroup float tg_out[BN * V_PER_THREAD * BD];
    if (simd_lid == 0) {
        tg_max[simd_gid] = max_score;
        tg_sum[simd_gid] = sum_exp_score;
    }
    // Stash o_partial in TG mem laid out as [BN][BD][V_PER_THREAD].
    for (uint i = 0; i < V_PER_THREAD; ++i) {
        tg_out[(simd_gid * BD + simd_lid) * V_PER_THREAD + i] = o_partial[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Global max + scaled sum across BN simdgroups.
    threadgroup float gmax_scratch;
    threadgroup float gsum_scratch;
    if (simd_gid == 0) {
        float my_max = tg_max[simd_lid];
        float gmax   = simd_max(my_max);
        float factor = exp(my_max - gmax);
        float my_sum = tg_sum[simd_lid] * factor;
        float gsum   = simd_sum(my_sum);
        if (simd_lid == 0) {
            gmax_scratch = gmax;
            gsum_scratch = (gsum > 0.0f) ? gsum : 1.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gmax = gmax_scratch;
    float inv_sum = 1.0f / gsum_scratch;

    // Aggregate o_partial across simdgroups. Each thread aggregates its
    // own (simd_lid, V_PER_THREAD) slice across the BN simdgroups.
    float o_final[V_PER_THREAD];
    for (uint i = 0; i < V_PER_THREAD; ++i) {
        o_final[i] = 0.0f;
    }
    for (uint sg = 0; sg < BN; ++sg) {
        float sg_factor = exp(tg_max[sg] - gmax);
        for (uint i = 0; i < V_PER_THREAD; ++i) {
            o_final[i] += sg_factor * tg_out[(sg * BD + simd_lid) * V_PER_THREAD + i];
        }
    }
    // Only one simdgroup needs to write the output (all simdgroups have
    // the same aggregated o_final by construction); pick simdgroup 0.
    if (simd_gid == 0) {
        uint out_base = bh_q * uint(D) + d_start;
        for (uint i = 0; i < V_PER_THREAD; ++i) {
            o[out_base + i] = T(o_final[i] * inv_sum);
        }
    }
"#;

/// Compile a fresh kernel handle. Caller caches it for the lifetime of
/// the cache layer.
pub fn make_fused_qsdpa_kernel() -> Result<MetalKernel> {
    metal_kernel(
        KERNEL_NAME,
        &[
            "q",
            "k_wq",
            "k_scales",
            "k_biases",
            "v_wq",
            "v_scales",
            "v_biases",
            "mask",
            "scale",
            "n_k",
            "mask_present",
        ],
        &["o"],
        KERNEL_SOURCE,
        "",
        true,
        false,
    )
}

/// Inputs to [`fused_qsdpa_decode`].
pub struct FusedQsdpaInputs<'a> {
    /// `q`: `[B, H_q, 1, D]` (n_q must be 1; caller asserts).
    pub q: &'a Array,
    /// `k_wq`: `[B, H_kv, n_k, D / pack_factor]` uint32.
    pub k_wq: &'a Array,
    /// `k_scales` / `k_biases`: `[B, H_kv, n_k, D / group_size]`.
    pub k_scales: &'a Array,
    pub k_biases: &'a Array,
    pub v_wq: &'a Array,
    pub v_scales: &'a Array,
    pub v_biases: &'a Array,
    /// Optional bool mask `[B, H_q, 1, n_k]`. When `None`, mask is skipped.
    pub mask: Option<&'a Array>,
    pub scale: f32,
    pub head_dim: i32,
    pub group_size: i32,
    pub bits: i32,
    pub h_q: i32,
    pub h_kv: i32,
}

/// Dispatch the fused decode kernel. Returns `[B, H_q, 1, D]` in the
/// same dtype as `q`.
pub fn fused_qsdpa_decode(
    kernel: &MetalKernel,
    inputs: FusedQsdpaInputs<'_>,
) -> Result<Array> {
    let q_shape = inputs.q.shape();
    if q_shape.len() != 4 {
        return Err(Exception::custom(format!(
            "fused_qsdpa_decode: q must be 4-D [B, H_q, 1, D], got {q_shape:?}"
        )));
    }
    if q_shape[2] != 1 {
        return Err(Exception::custom(format!(
            "fused_qsdpa_decode: n_q must be 1 (decode-path only), got {}",
            q_shape[2]
        )));
    }
    let b = q_shape[0];
    let h_q = q_shape[1];
    if h_q != inputs.h_q {
        return Err(Exception::custom(format!(
            "fused_qsdpa_decode: q[1]={h_q} != cfg h_q={}",
            inputs.h_q
        )));
    }
    if inputs.h_q % inputs.h_kv != 0 {
        return Err(Exception::custom(format!(
            "fused_qsdpa_decode: H_q={} not divisible by H_KV={}",
            inputs.h_q, inputs.h_kv
        )));
    }
    let n_rep = inputs.h_q / inputs.h_kv;
    let n_k = inputs.k_wq.shape()[2];
    let pack_factor = 32 / inputs.bits;
    if inputs.head_dim % pack_factor != 0 {
        return Err(Exception::custom(format!(
            "fused_qsdpa_decode: head_dim={} not divisible by pack_factor={}",
            inputs.head_dim, pack_factor
        )));
    }
    if inputs.head_dim % inputs.group_size != 0 {
        return Err(Exception::custom(format!(
            "fused_qsdpa_decode: head_dim={} not divisible by group_size={}",
            inputs.head_dim, inputs.group_size
        )));
    }

    let out_dtype = inputs.q.dtype();
    if !matches!(out_dtype, Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16) {
        return Err(Exception::custom(format!(
            "fused_qsdpa_decode: unsupported q dtype {out_dtype:?}"
        )));
    }

    let scale_buf = Array::from_f32(inputs.scale);
    let n_k_buf = Array::from_int(n_k);
    let mask_present = inputs.mask.is_some() as i32;
    let mask_present_buf = Array::from_int(mask_present);

    // Dummy mask if not provided (kernel reads but `mask_present` gates the load).
    let dummy_mask;
    let mask_arr = match inputs.mask {
        Some(m) => m.clone(),
        None => {
            dummy_mask = Array::zeros::<u8>(&[1])?.as_dtype(Dtype::Bool)?;
            dummy_mask
        }
    };

    // 32 simdgroups × 32 lanes = 1024 threads per TG (Flash-Attention pattern).
    // Online softmax keeps (max, sum, o_partial) in private registers; no
    // TG-mem scales with n_k, so n_k is unbounded by TG memory.
    let threads = 32 * 32;
    let n_k_max = ((n_k as u32).next_power_of_two().max(32)) as i32;
    let config = MetalKernelConfig::new()
        .add_output(vec![b, h_q, 1, inputs.head_dim], out_dtype)
        .grid(threads * h_q, b, 1)
        .thread_group(threads, 1, 1)
        .add_template("T", out_dtype)?
        .add_template("D", inputs.head_dim)?
        .add_template("BITS", inputs.bits)?
        .add_template("GROUP_SIZE", inputs.group_size)?
        .add_template("H_Q", inputs.h_q)?
        .add_template("H_KV", inputs.h_kv)?
        .add_template("N_REP", n_rep)?
        .add_template("N_K_MAX", n_k_max)?
        .add_template("TG_SIZE", threads)?;

    let outs = kernel.apply(
        &[
            inputs.q.clone(),
            inputs.k_wq.clone(),
            inputs.k_scales.clone(),
            inputs.k_biases.clone(),
            inputs.v_wq.clone(),
            inputs.v_scales.clone(),
            inputs.v_biases.clone(),
            mask_arr,
            scale_buf,
            n_k_buf,
            mask_present_buf,
        ],
        config,
        Stream::default(),
    )?;
    outs.into_iter()
        .next()
        .ok_or_else(|| Exception::custom("fused_qsdpa_decode: no outputs"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{dequantize, quantize, softmax_axis};
    use mlx_rs::random::{key, normal};
    use mlx_rs::transforms::eval;

    fn max_abs(a: &Array, b: &Array) -> f32 {
        a.subtract(b).unwrap().abs().unwrap().max(None).unwrap().item::<f32>()
    }

    /// Reference path: full-precision dequant then standard SDPA math.
    fn reference_decode(
        q: &Array,
        k_dense: &Array,
        v_dense: &Array,
        scale: f32,
        h_q: i32,
        h_kv: i32,
    ) -> Array {
        let q_scaled = q.multiply(Array::from_f32(scale)).unwrap();
        // GQA: replicate K, V across query head groups by reshape pattern.
        let n_rep = h_q / h_kv;
        let q_reshape = if n_rep > 1 {
            let s = q_scaled.shape().to_vec();
            q_scaled
                .reshape(&[s[0], h_kv, n_rep, s[2], s[3]])
                .unwrap()
        } else {
            q_scaled.expand_dims(2).unwrap()
        };
        let k_exp = k_dense.expand_dims(2).unwrap();
        let v_exp = v_dense.expand_dims(2).unwrap();
        // scores: q @ k.T over the last axis. k_exp: [B, H_kv, 1, n_k, D]
        let k_t = k_exp.transpose_axes(&[0, 1, 2, 4, 3]).unwrap();
        let scores = q_reshape.matmul(&k_t).unwrap();
        let probs = softmax_axis(&scores, -1, true).unwrap();
        let out = probs.matmul(&v_exp).unwrap();
        // Collapse n_rep back into H_q.
        let s = out.shape().to_vec();
        if n_rep > 1 {
            out.reshape(&[s[0], h_q, s[3], s[4]]).unwrap()
        } else {
            out.squeeze_axes(&[2][..]).unwrap()
        }
    }

    /// Rust scalar reference mirroring the kernel exactly: reads packed
    /// uint32 K/V, extracts codes, dequants in-place, runs softmax,
    /// computes output. Helps isolate kernel-vs-spec bugs from kernel-
    /// vs-mlx-quantize-layout bugs.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn kernel_scalar_reference(
        b: i32, h_q: i32, h_kv: i32, n_k: i32, d: i32,
        bits: i32, group_size: i32,
        q: &[f32], scale: f32,
        k_wq: &[u32], k_scales: &[f32], k_biases: &[f32],
        v_wq: &[u32], v_scales: &[f32], v_biases: &[f32],
    ) -> Vec<f32> {
        let pack_factor = (32 / bits) as usize;
        let groups_per = (d / group_size) as usize;
        let words_per = (d as usize) / pack_factor;
        let n_rep = h_q / h_kv;
        let mask = (1u32 << bits) - 1;
        let nk = n_k as usize;
        let mut out = vec![0.0f32; (b * h_q * d) as usize];
        for bi in 0..b as usize {
            for hq in 0..h_q as usize {
                let hkv = hq / n_rep as usize;
                let bh_q = bi * h_q as usize + hq;
                let bh_kv = bi * h_kv as usize + hkv;
                // Pre-scale Q.
                let mut q_row = vec![0.0f32; d as usize];
                for di in 0..d as usize {
                    q_row[di] = q[bh_q * d as usize + di] * scale;
                }
                // Pass 1: scores.
                let mut scores = vec![0.0f32; nk];
                for k_idx in 0..nk {
                    let k_wq_base = (bh_kv * nk + k_idx) * words_per;
                    let k_meta_base = (bh_kv * nk + k_idx) * groups_per;
                    let mut s = 0.0f32;
                    for di in 0..d as usize {
                        let word_idx = di / pack_factor;
                        let slot = di % pack_factor;
                        let w = k_wq[k_wq_base + word_idx];
                        let code = (w >> (slot * bits as usize)) & mask;
                        let group = di / group_size as usize;
                        let sc = k_scales[k_meta_base + group];
                        let bi_ = k_biases[k_meta_base + group];
                        let k_val = code as f32 * sc + bi_;
                        s += q_row[di] * k_val;
                    }
                    scores[k_idx] = s;
                }
                // Pass 2: softmax.
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut l = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - m).exp();
                    l += *s;
                }
                let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
                // Pass 3: output.
                for di in 0..d as usize {
                    let word_idx = di / pack_factor;
                    let slot = di % pack_factor;
                    let group = di / group_size as usize;
                    let mut acc = 0.0f32;
                    for k_idx in 0..nk {
                        let wgt = scores[k_idx] * inv_l;
                        if wgt == 0.0 { continue; }
                        let v_wq_base = (bh_kv * nk + k_idx) * words_per;
                        let v_meta_base = (bh_kv * nk + k_idx) * groups_per;
                        let w = v_wq[v_wq_base + word_idx];
                        let code = (w >> (slot * bits as usize)) & mask;
                        let sc = v_scales[v_meta_base + group];
                        let bi_ = v_biases[v_meta_base + group];
                        let v_val = code as f32 * sc + bi_;
                        acc += wgt * v_val;
                    }
                    out[bh_q * d as usize + di] = acc;
                }
            }
        }
        out
    }

    fn run(b: i32, h_q: i32, h_kv: i32, n_k: i32, d: i32, bits: i32, group_size: i32) {
        let prng = key(11).unwrap();
        let q = normal::<f32>(&[b, h_q, 1, d], None, None, &prng).unwrap();
        let prng = key(22).unwrap();
        let k = normal::<f32>(&[b, h_kv, n_k, d], None, None, &prng).unwrap();
        let prng = key(33).unwrap();
        let v = normal::<f32>(&[b, h_kv, n_k, d], None, None, &prng).unwrap();

        let (k_wq, k_scales, k_biases) = quantize(&k, group_size, bits).unwrap();
        let (v_wq, v_scales, v_biases) = quantize(&v, group_size, bits).unwrap();
        eval([&k_wq, &k_scales, &k_biases, &v_wq, &v_scales, &v_biases]).unwrap();

        let scale = (d as f32).sqrt().recip();

        // Reference using dequantised K/V via mlx.
        let k_dq = dequantize(&k_wq, &k_scales, &k_biases, group_size, bits).unwrap();
        let v_dq = dequantize(&v_wq, &v_scales, &v_biases, group_size, bits).unwrap();
        let expected = reference_decode(&q, &k_dq, &v_dq, scale, h_q, h_kv);
        eval([&expected]).unwrap();

        // Scalar mirror of the kernel formula (reads packed uint32 bytes).
        eval([&q]).unwrap();
        let scalar_out = kernel_scalar_reference(
            b, h_q, h_kv, n_k, d, bits, group_size,
            q.as_slice::<f32>(), scale,
            k_wq.as_slice::<u32>(), k_scales.as_slice::<f32>(), k_biases.as_slice::<f32>(),
            v_wq.as_slice::<u32>(), v_scales.as_slice::<f32>(), v_biases.as_slice::<f32>(),
        );

        let kernel = make_fused_qsdpa_kernel().unwrap();
        let got = fused_qsdpa_decode(
            &kernel,
            FusedQsdpaInputs {
                q: &q,
                k_wq: &k_wq,
                k_scales: &k_scales,
                k_biases: &k_biases,
                v_wq: &v_wq,
                v_scales: &v_scales,
                v_biases: &v_biases,
                mask: None,
                scale,
                head_dim: d,
                group_size,
                bits,
                h_q,
                h_kv,
            },
        )
        .unwrap();
        eval([&got]).unwrap();

        let got_vec = got.as_slice::<f32>();
        let mut max_err_scalar = 0.0f32;
        let mut worst_i = 0usize;
        for (i, (a, b)) in got_vec.iter().zip(scalar_out.iter()).enumerate() {
            let e = (a - b).abs();
            if e > max_err_scalar {
                max_err_scalar = e;
                worst_i = i;
            }
        }
        let err_dq = max_abs(&got, &expected);
        eprintln!(
            "[n_k={n_k} d={d} bits={bits}] kernel vs scalar-mirror: {max_err_scalar:.5} \
             | kernel vs mlx-dequant ref: {err_dq:.5}"
        );
        if max_err_scalar > 1e-3 {
            eprintln!("  worst idx {worst_i}: kernel={} scalar={}",
                      got_vec[worst_i], scalar_out[worst_i]);
            // Print first 8 vals
            eprintln!("  first 8 kernel: {:?}", &got_vec[..8.min(got_vec.len())]);
            eprintln!("  first 8 scalar: {:?}", &scalar_out[..8.min(scalar_out.len())]);
        }

        let tol = 5e-3 * (n_k as f32).sqrt();
        assert!(
            max_err_scalar < 1e-3,
            "kernel vs scalar mirror diverged: max abs = {max_err_scalar} (n_k={n_k})"
        );
        assert!(
            err_dq < tol,
            "kernel vs mlx-dequant reference diverged: max abs = {err_dq}, tol = {tol}"
        );
    }

    /// Check whether `quantize` output is contiguous (matters because the
    /// kernel uses flat-offset indexing into the buffers).
    #[test]
    fn check_quantize_output_strides() {
        let prng = key(42).unwrap();
        let arr = normal::<f32>(&[2, 4, 8, 128], None, None, &prng).unwrap();
        let (wq, scales, biases) = quantize(&arr, 64, 4).unwrap();
        eval([&wq, &scales, &biases]).unwrap();
        eprintln!("wq:      shape={:?} strides={:?}", wq.shape(), wq.strides());
        eprintln!("scales:  shape={:?} strides={:?}", scales.shape(), scales.strides());
        eprintln!("biases:  shape={:?} strides={:?}", biases.shape(), biases.strides());
    }

    /// Verify mlx's affine-quantize layout matches what the kernel
    /// assumes: 4-bit values packed little-endian within each uint32,
    /// so slot k lives at bits [k*4, k*4+4).
    #[test]
    fn check_quantize_packs_little_endian_per_uint32() {
        // 32 values [0,1,...,31] at group_size=32, bits=4 → 4 uint32s.
        let v: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let arr = Array::from_slice(&v, &[1, 32]);
        let (wq, scales, biases) = quantize(&arr, 32, 4).unwrap();
        eval([&wq, &scales, &biases]).unwrap();
        let words = wq.as_slice::<u32>();
        let sc = scales.as_slice::<f32>()[0];
        let bi = biases.as_slice::<f32>()[0];
        eprintln!("wq[0..4]={:?} sc={sc} bi={bi}",
                  words.iter().map(|w| format!("{w:#010x}")).collect::<Vec<_>>());
        // Decode using little-endian-per-uint32 assumption.
        for i in 0..32usize {
            let w = words[i / 8];
            let slot = i % 8;
            let code = (w >> (slot * 4)) & 0xF;
            let val = (code as f32) * sc + bi;
            eprintln!("  i={i:2}: code={code:2} val={val:7.3} (expected={})",
                      v[i]);
        }
        // Round-trip via mlx dequantize.
        let dq = dequantize(&wq, &scales, &biases, 32, 4).unwrap();
        eval([&dq]).unwrap();
        let dq_vals = dq.as_slice::<f32>();
        eprintln!("dq round-trip vals: {:?}", &dq_vals[..32]);
    }

    #[test]
    fn fused_4bit_d128_n_k_1_no_gqa() {
        // Simplest possible: 1 K/V token. softmax is trivially 1.
        run(1, 1, 1, 1, 128, 4, 64);
    }

    #[test]
    fn fused_4bit_d128_n_k_32_no_gqa() {
        run(1, 2, 2, 32, 128, 4, 64);
    }

    #[test]
    fn fused_4bit_d128_n_k_128_no_gqa() {
        run(1, 2, 2, 128, 128, 4, 64);
    }

    #[test]
    fn fused_4bit_d128_gqa_qwen3() {
        run(1, 16, 8, 64, 128, 4, 64);
    }

    #[test]
    fn fused_8bit_d128_gqa() {
        run(1, 16, 8, 64, 128, 8, 64);
    }

    #[test]
    fn fused_4bit_d128_n_k_257_non_block_aligned() {
        // 257 is BLOCK_K * 8 + 1 — last block partial.
        run(1, 4, 2, 257, 128, 4, 64);
    }

    #[test]
    fn fused_handles_long_context() {
        run(1, 8, 4, 1024, 128, 4, 64);
    }

    #[test]
    fn fused_handles_n_k_4097() {
        // Crosses the old 4096 cap; first non-power-of-2 above 4096.
        run(1, 8, 4, 4097, 128, 4, 64);
    }

    #[test]
    fn fused_handles_n_k_8192_4bit() {
        run(1, 8, 4, 8192, 128, 4, 64);
    }
}
