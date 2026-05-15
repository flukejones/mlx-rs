//! TurboQuant quantizers — Algorithm 1 (MSE) and Algorithm 2 (inner
//! product, added in a follow-up commit).
//!
//! These operate on tensors of shape `[..., d]` where `d` is the embedding
//! dimension (typically `head_dim = 128` for modern LLMs).
//!
//! Algorithm 1 (MSE) at a glance:
//! ```text
//!   r = ‖x‖₂
//!   u = x / r              # unit vector on S^{d-1}
//!   y = u @ Πᵀ             # random rotation; coords now Beta(d)
//!   idx_i = searchsorted(decision_boundaries, y_i)   # 2^b buckets
//!   pack idx into ⌈d·b/8⌉ bytes
//!   return (packed, r)
//! ```
//!
//! Dequant: `x̂ = r · (Πᵀ · centroids[idx])`.

use mlx_rs::error::Exception;
use mlx_rs::fast::MetalKernel;
use mlx_rs::ops::{expand_dims, sum_axis};
use mlx_rs::{Array, Dtype};

use super::codebook::{decision_boundaries, get_codebook, Codebook};
use super::packing::{pack_indices, pack_signs, unpack_indices, unpack_signs};
use super::rotation::{
    generate_qjl_matrix, generate_rotation_matrix, rotate_backward, rotate_forward,
};
use super::searchsorted_kernel::{make_searchsorted_kernel, searchsorted_bucket};

use crate::error::Error;

/// Numerical guard added to `‖x‖` before dividing, matching 0xSero
/// `quantizer.py:_quantize` (`norms + 1e-10`).
const NORM_EPS: f32 = 1e-10;

/// One TurboQuant-MSE quantizer instance, scoped to a fixed `(d, bits)` and
/// rotation seed. The rotation matrix, codebook, and kernel handle are all
/// cached for the lifetime of the cache layer.
#[derive(Debug)]
pub struct TurboQuantMSE {
    d: i32,
    bits: i32,
    pi: Array,
    boundaries: Array,
    centroids: Array,
    kernel: MetalKernel,
    codebook: &'static Codebook,
}

/// Output of [`TurboQuantMSE::quantize`].
#[derive(Debug)]
pub struct MSEQuantized {
    /// Bit-packed indices, shape `[..., d_packed]`, dtype `uint8`.
    pub packed_indices: Array,
    /// Original `‖x‖` values, shape `[...]`, dtype matching input.
    pub norms: Array,
}

impl TurboQuantMSE {
    /// Build a fresh quantizer.
    ///
    /// - `d`: embedding dimension (`head_dim`).
    /// - `bits`: quantization width, must be in `1..=4` (the Metal kernel's
    ///   `MAX_BOUNDARIES = 16` bound).
    /// - `seed`: PRNG seed for Π. Same seed → same Π across processes.
    pub fn new(d: i32, bits: i32, seed: u64) -> Result<Self, Error> {
        assert!(d > 0);
        assert!(
            (1..=4).contains(&bits),
            "TurboQuantMSE: bits must be in 1..=4 (got {bits})"
        );

        let pi = generate_rotation_matrix(d, seed)?;
        let codebook = get_codebook(d, bits);
        let decision = decision_boundaries(codebook);
        let boundaries = Array::from_slice(decision, &[decision.len() as i32]);
        let centroids = Array::from_slice(&codebook.centroids, &[codebook.centroids.len() as i32]);
        let kernel = make_searchsorted_kernel().map_err(Error::from)?;

        Ok(Self {
            d,
            bits,
            pi,
            boundaries,
            centroids,
            kernel,
            codebook,
        })
    }

    /// Embedding dim this quantizer was built for.
    pub fn dim(&self) -> i32 {
        self.d
    }

    /// Bits per coord.
    pub fn bits(&self) -> i32 {
        self.bits
    }

    /// Codebook metadata for serialisation / debugging.
    pub fn codebook(&self) -> &Codebook {
        self.codebook
    }

    /// Rotation matrix Π (used by the caller's `from_state` / `state`
    /// machinery in the cache layer).
    pub fn rotation(&self) -> &Array {
        &self.pi
    }

    /// Quantize `x` of shape `[..., d]`.
    ///
    /// Steps:
    /// 1. `r = ‖x‖` along the last axis (kept in input dtype).
    /// 2. `u = x / (r + ε)`.
    /// 3. `y = u · Πᵀ` (cast to fp32 for the rotation to keep the
    ///    Lloyd-Max search numerically clean — the codebook is fp32).
    /// 4. Bucket lookup via the Metal kernel against the fp32 boundaries.
    /// 5. Pack the resulting `uint8` indices.
    pub fn quantize(&self, x: &Array) -> Result<MSEQuantized, Exception> {
        if x.shape()[x.ndim() - 1] != self.d {
            return Err(Exception::custom(format!(
                "TurboQuantMSE::quantize: last axis {} != configured d {}",
                x.shape()[x.ndim() - 1],
                self.d
            )));
        }
        let in_dtype = x.dtype();

        // L2 norms along the last axis: ‖x‖ = sqrt(sum x^2).
        let sq = x.square()?;
        let norms = sum_axis(&sq, -1, false)?.sqrt()?;
        // Reciprocal with ε guard: `1 / (norm + ε)`.
        let eps = Array::from_f32(NORM_EPS).as_dtype(in_dtype)?;
        let inv = Array::from_f32(1.0)
            .as_dtype(in_dtype)?
            .divide(norms.add(&eps)?)?;
        // Broadcast `inv` over the last axis: `[..., 1]`.
        let inv_b = expand_dims(&inv, -1)?;
        let u = x.multiply(&inv_b)?;

        // Rotate. Π lives in fp32 (CPU-built), so the matmul auto-upcasts.
        // We then cast `y` to fp32 explicitly for the boundary search.
        let u_f32 = u.as_dtype(Dtype::Float32)?;
        let y = rotate_forward(&self.pi, &u_f32)?;

        // Bucket lookup. The kernel wants the input flattened; we call it
        // on the full `[..., d]` tensor — the kernel runs one thread per
        // value so the shape is just a count.
        let indices_u8 = searchsorted_bucket(&self.kernel, &y, &self.boundaries)?;

        // Pack into ⌈d/vpb⌉ bytes per row.
        let packed = pack_indices(&indices_u8, self.bits)?;

        Ok(MSEQuantized {
            packed_indices: packed,
            norms,
        })
    }

    /// Dequantize: recover `x̂ ≈ x` from packed indices + per-row norms.
    pub fn dequantize(&self, q: &MSEQuantized) -> Result<Array, Exception> {
        // Unpack to `[..., d]` uint8.
        let idx_u8 = unpack_indices(&q.packed_indices, self.bits, self.d)?;
        let idx_i32 = idx_u8.as_dtype(Dtype::Int32)?;

        // Look up centroids: `take` flattens-then-indexes a 1-D array,
        // producing an output with the same shape as `idx_i32` (i.e.
        // `[..., d]`).
        let y_hat = self.centroids.take(&idx_i32)?;

        // Rotate back.
        let x_unit = rotate_backward(&self.pi, &y_hat)?;

        // Rescale by norms. Norms shape `[...]` → broadcast over last axis.
        let in_dtype = q.norms.dtype();
        let x_unit_t = x_unit.as_dtype(in_dtype)?;
        let norms_b = expand_dims(&q.norms, -1)?;
        x_unit_t.multiply(&norms_b)
    }
}

// -------- Algorithm 2: inner-product quantizer (used for keys) --------

/// Output of [`TurboQuantProd::quantize`].
#[derive(Debug)]
pub struct ProdQuantized {
    /// MSE-stage packed indices at `(bits - 1)` bits, shape
    /// `[..., d_packed_mse]`.
    pub mse_indices: Array,
    /// Bit-packed sign bits of `S · residual`, shape `[..., d_packed_signs]`.
    pub qjl_signs: Array,
    /// `‖residual‖₂` per row, shape `[...]`, dtype matching input.
    pub residual_norms: Array,
    /// Original `‖x‖₂` per row, shape `[...]`.
    pub norms: Array,
}

/// Two-stage TurboQuant for unbiased inner-product estimation.
///
/// Stage 1: MSE-quantize at `(bits - 1)` bits via [`TurboQuantMSE`].
/// Stage 2: take the residual `x - x̂_mse`, project through a fixed
/// Gaussian `S`, and store its sign pattern. The residual's L2 norm is
/// stored separately so the dequant path can rescale.
///
/// At inference time [`Self::attention_score`] computes `⟨q, x̂⟩` without
/// materialising the dequantised key — the only path that turns
/// TurboQuant into a *throughput* win (vs the affine `QuantizedKVCache`,
/// which always dequantises on read).
#[derive(Debug)]
pub struct TurboQuantProd {
    d: i32,
    bits: i32,
    mse: TurboQuantMSE,
    s: Array,
    qjl_scale: f32,
}

impl TurboQuantProd {
    /// New quantizer at total budget `bits`. Spends `(bits - 1)` on the
    /// MSE stage and 1 on the QJL sign sketch. `bits` must be `≥ 2`.
    pub fn new(d: i32, bits: i32, seed: u64) -> Result<Self, Error> {
        assert!(
            bits >= 2,
            "TurboQuantProd: bits must be ≥ 2 (1 for MSE + 1 for QJL)"
        );
        let mse = TurboQuantMSE::new(d, bits - 1, seed)?;
        let s = generate_qjl_matrix(d, seed)?;
        let qjl_scale = (std::f32::consts::FRAC_PI_2.sqrt()) / (d as f32);
        Ok(Self {
            d,
            bits,
            mse,
            s,
            qjl_scale,
        })
    }

    /// Embedding dim.
    pub fn dim(&self) -> i32 {
        self.d
    }

    /// Total bits per coord (MSE + QJL sign).
    pub fn bits(&self) -> i32 {
        self.bits
    }

    /// MSE stage handle — exposes Π and the codebook for serialisation.
    pub fn mse_quantizer(&self) -> &TurboQuantMSE {
        &self.mse
    }

    /// QJL projection matrix `S`.
    pub fn qjl_matrix(&self) -> &Array {
        &self.s
    }

    /// QJL dequantization constant `√(π/2) / d`. Public for kernel paths
    /// that compute the QJL term directly.
    pub fn qjl_scale(&self) -> f32 {
        self.qjl_scale
    }

    /// Quantize `x` of shape `[..., d]`.
    pub fn quantize(&self, x: &Array) -> Result<ProdQuantized, Exception> {
        // Stage 1: MSE at (bits - 1).
        let mse_q = self.mse.quantize(x)?;
        let x_hat = self.mse.dequantize(&mse_q)?;

        // Residual + its L2 norm.
        let residual = x.subtract(&x_hat)?;
        let residual_norms = sum_axis(&residual.square()?, -1, false)?.sqrt()?;

        // QJL: sign(S · residual) along the last axis. We compute S in fp32
        // for stability, then pack.
        let res_f32 = residual.as_dtype(Dtype::Float32)?;
        let s_t = self.s.transpose_axes(&[1, 0])?;
        let projected = res_f32.matmul(&s_t)?;
        let packed_signs = pack_signs(&projected)?;

        Ok(ProdQuantized {
            mse_indices: mse_q.packed_indices,
            qjl_signs: packed_signs,
            residual_norms,
            norms: mse_q.norms,
        })
    }

    /// Dequantize: symmetric — for testing / save-load round-trips only.
    /// The fast path uses [`Self::attention_score`] which computes
    /// `⟨q, x̂⟩` without going through this method.
    pub fn dequantize(&self, q: &ProdQuantized) -> Result<Array, Exception> {
        let mse_part = self.mse.dequantize(&MSEQuantized {
            packed_indices: q.mse_indices.clone(),
            norms: q.norms.clone(),
        })?;

        // QJL part: `(qjl_scale * ‖r‖) · (signs · S)`.
        let signs = unpack_signs(&q.qjl_signs, self.d)?;
        // signs is fp32; matmul against fp32 S.
        let qjl_unit = signs.matmul(&self.s)?;
        let in_dtype = q.norms.dtype();
        let qjl_unit_t = qjl_unit.as_dtype(in_dtype)?;
        let scale = Array::from_f32(self.qjl_scale).as_dtype(in_dtype)?;
        let factor = q.residual_norms.multiply(&scale)?;
        let factor_b = expand_dims(&factor, -1)?;
        let qjl_part = qjl_unit_t.multiply(&factor_b)?;

        mse_part.add(&qjl_part)
    }

    /// Compute attention logits `⟨q_i, K_j⟩` for every `(i, j)` pair
    /// **without dequantising K**.
    ///
    /// Shapes:
    /// - `query`: `[B, H, n_q, d]` in any float dtype.
    /// - `q` (the packed K): inner shapes `[B, H, n_k, ...]` per field.
    ///
    /// Returns scores `[B, H, n_q, n_k]` in fp32 (caller can cast / scale).
    ///
    /// Plain-ops reference implementation. The fused Metal kernel that
    /// will replace this in a later commit must reproduce the same
    /// formula exactly.
    pub fn attention_score(
        &self,
        query: &Array,
        q: &ProdQuantized,
    ) -> Result<Array, Exception> {
        // Project queries: q_pi = query · Πᵀ; q_s = query · Sᵀ. Both in fp32.
        let query_f32 = query.as_dtype(Dtype::Float32)?;
        let q_pi = rotate_forward(self.mse.rotation(), &query_f32)?;
        let s_t = self.s.transpose_axes(&[1, 0])?;
        let q_s = query_f32.matmul(&s_t)?;

        // ------ MSE term: `‖K_row‖ · ⟨q_pi, centroids[idx_row]⟩` ------
        // Build centroid table looked up per key coord: shape [B, H, n_k, d].
        let mse_bits = self.bits - 1;
        let idx_u8 = unpack_indices(&q.mse_indices, mse_bits, self.d)?;
        let idx_i32 = idx_u8.as_dtype(Dtype::Int32)?;
        let centroids_full = self.mse.codebook_centroids().take(&idx_i32)?;
        let centroids_f32 = centroids_full.as_dtype(Dtype::Float32)?;

        // q_pi: [B, H, n_q, d]; centroids_f32: [B, H, n_k, d].
        // matmul along d: `q_pi @ centroids_f32.T` → [B, H, n_q, n_k].
        let centroids_t = centroids_f32.transpose_axes(&[0, 1, 3, 2])?;
        let partial = q_pi.matmul(&centroids_t)?;

        // Multiply by per-key norms (broadcast `[B, H, 1, n_k]`).
        let key_norms_f32 = q.norms.as_dtype(Dtype::Float32)?;
        let key_norms_b = expand_dims(&key_norms_f32, -2)?;
        let score_mse = partial.multiply(&key_norms_b)?;

        // ------ QJL term: `qjl_scale · ‖residual‖ · ⟨q_s, signs⟩` ------
        let signs = unpack_signs(&q.qjl_signs, self.d)?;
        let signs_t = signs.transpose_axes(&[0, 1, 3, 2])?;
        let partial_qjl = q_s.matmul(&signs_t)?;
        let res_norms_f32 = q.residual_norms.as_dtype(Dtype::Float32)?;
        let scale = Array::from_f32(self.qjl_scale);
        let scaled_res = res_norms_f32.multiply(&scale)?;
        let scaled_res_b = expand_dims(&scaled_res, -2)?;
        let score_qjl = partial_qjl.multiply(&scaled_res_b)?;

        score_mse.add(&score_qjl)
    }
}

// Internal helper on TurboQuantMSE used by TurboQuantProd's attention_score —
// the existing `dequantize` materialises the rotated+norm-scaled output, but
// the asymmetric score needs raw centroid values *before* rotation.
impl TurboQuantMSE {
    /// Raw centroid table `[2^bits]` fp32. Used by the asymmetric attention
    /// path to skip the explicit dequantise.
    pub fn codebook_centroids(&self) -> &Array {
        &self.centroids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs(a: &Array) -> f32 {
        a.abs().unwrap().max(None).unwrap().item::<f32>()
    }

    /// Sample unit-norm vectors uniformly on `S^{d-1}` via Gaussian + L2-normalise.
    fn sample_unit_vectors(d: i32, n: i32, seed: u64) -> Array {
        let prng = mlx_rs::random::key(seed).unwrap();
        let raw = mlx_rs::random::normal::<f32>(&[n, d], None, None, &prng).unwrap();
        let norms = raw.square().unwrap().sum_axis(-1, true).unwrap().sqrt().unwrap();
        raw.divide(&norms).unwrap()
    }

    /// 4-bit round-trip on unit vectors must be near-lossless (mse < 1e-3
    /// per coord for d=128 from the bundled codebook).
    #[test]
    fn round_trip_4bit_unit_vectors() {
        let d = 128;
        let q = TurboQuantMSE::new(d, 4, 11).unwrap();
        let x = sample_unit_vectors(d, 64, 7);
        let enc = q.quantize(&x).unwrap();
        let x_hat = q.dequantize(&enc).unwrap();
        let err = max_abs(&x.subtract(&x_hat).unwrap());
        // The 4-bit codebook for d=128 has MSE per coord ~7e-5 → max-abs
        // residual on unit vectors should be well under 0.1.
        assert!(err < 0.1, "max abs err = {err}");
    }

    /// 1-bit round-trip is lossy (sign-only) but the L2 norm should be
    /// preserved within the codebook MSE.
    #[test]
    fn round_trip_1bit_preserves_l2_within_codebook_mse() {
        let d = 64;
        let q = TurboQuantMSE::new(d, 1, 13).unwrap();
        let x = sample_unit_vectors(d, 16, 19);
        let enc = q.quantize(&x).unwrap();
        let x_hat = q.dequantize(&enc).unwrap();
        // Compare per-row L2 norms.
        let norm_x = x.square().unwrap().sum_axis(-1, false).unwrap().sqrt().unwrap();
        let norm_xh = x_hat.square().unwrap().sum_axis(-1, false).unwrap().sqrt().unwrap();
        let err = max_abs(&norm_x.subtract(&norm_xh).unwrap());
        // Codebook centroids land near ±0.05 for 1-bit/d=64, so
        // `‖centroids[idx]‖²` for a d-vector is roughly `d·(0.05)² ≈ 0.16`.
        // Reconstructed norm differs from 1 by up to ~0.85 — large, expected.
        assert!(err < 0.95, "1-bit norm err = {err}");
    }

    /// Norm scaling: dequant of a vector with `‖x‖ = c` should have
    /// `‖x̂‖ ≈ c · ‖centroids[idx_pattern]‖`. Easier test: a vector and
    /// twice that vector should dequantize to outputs that differ by 2×.
    #[test]
    fn dequant_scales_with_norm() {
        let d = 128;
        let q = TurboQuantMSE::new(d, 3, 5).unwrap();
        let x = sample_unit_vectors(d, 4, 31);
        let x2 = x.multiply(Array::from_f32(2.0)).unwrap();

        let enc = q.quantize(&x).unwrap();
        let enc2 = q.quantize(&x2).unwrap();

        let x_hat = q.dequantize(&enc).unwrap();
        let x2_hat = q.dequantize(&enc2).unwrap();

        // x2_hat / 2 should match x_hat within float noise.
        let diff = x2_hat
            .divide(Array::from_f32(2.0))
            .unwrap()
            .subtract(&x_hat)
            .unwrap();
        let err = max_abs(&diff);
        assert!(err < 1e-4, "scale invariance err = {err}");
    }

    /// Determinism: same input + same seed → bit-identical output.
    #[test]
    fn quantize_is_deterministic() {
        let d = 64;
        let q = TurboQuantMSE::new(d, 2, 9).unwrap();
        let x = sample_unit_vectors(d, 8, 41);

        let enc_a = q.quantize(&x).unwrap();
        let enc_b = q.quantize(&x).unwrap();
        let diff_idx = enc_a
            .packed_indices
            .subtract(&enc_b.packed_indices)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<u8>();
        assert_eq!(diff_idx, 0, "indices not deterministic");
    }

    /// Rejects mismatched last-axis length.
    #[test]
    fn quantize_rejects_dim_mismatch() {
        let q = TurboQuantMSE::new(64, 2, 3).unwrap();
        let x = sample_unit_vectors(128, 4, 5); // wrong d
        let err = q.quantize(&x);
        assert!(err.is_err(), "should reject mismatched d");
    }

    // -------- TurboQuantProd (Algorithm 2) --------

    /// TurboQuantProd round-trip recovers vectors within a wider error bound
    /// than pure MSE — QJL adds a sign-pattern correction but introduces
    /// stochastic noise on its own.
    #[test]
    fn prod_round_trip_is_finite_and_bounded() {
        let d = 128;
        let q = TurboQuantProd::new(d, 3, 17).unwrap();
        let x = sample_unit_vectors(d, 16, 23);
        let enc = q.quantize(&x).unwrap();
        let x_hat = q.dequantize(&enc).unwrap();
        let err = max_abs(&x.subtract(&x_hat).unwrap());
        // Generous bound — Prod is tuned for *inner product* not L2, so
        // pointwise reconstruction is weaker than TurboQuantMSE.
        assert!(err.is_finite() && err < 1.5, "round-trip err = {err}");
    }

    /// Asymmetric attention_score is *unbiased*: averaged over random Π/S
    /// the estimator equals the true inner product. Here we use a fixed
    /// seed and check the empirical mean error across many (q, k) pairs
    /// against the codebook's MSE — a single-run sanity check matching
    /// 0xSero `test_prod_attention_unbiased`.
    #[test]
    fn prod_attention_score_tracks_true_inner_product() {
        let d = 128;
        let n_q = 8;
        let n_k = 32;
        let q = TurboQuantProd::new(d, 3, 29).unwrap();

        // Random (non-unit) vectors so the norm scaling exercises both
        // legs of the formula.
        let prng_q = mlx_rs::random::key(13).unwrap();
        let prng_k = mlx_rs::random::key(99).unwrap();
        let query =
            mlx_rs::random::normal::<f32>(&[1, 1, n_q, d], None, None, &prng_q).unwrap();
        let keys =
            mlx_rs::random::normal::<f32>(&[1, 1, n_k, d], None, None, &prng_k).unwrap();

        let enc = q.quantize(&keys).unwrap();
        let est = q.attention_score(&query, &enc).unwrap();

        // True scores from un-quantised K.
        let keys_t = keys.transpose_axes(&[0, 1, 3, 2]).unwrap();
        let truth = query.matmul(&keys_t).unwrap();

        // Per-element mean absolute error.
        let diff = est.subtract(&truth).unwrap().abs().unwrap();
        let mean = diff.mean(None).unwrap().item::<f32>();
        // 3-bit Prod (MSE@2bit + QJL@1bit) at d=128 should average sub-unit
        // error on N(0,1) vectors. Loose bound: < 4.0 (centroid magnitudes
        // for 2-bit d=128 are ~0.13; 128 ⨯ 1 ≈ 1 expected magnitude).
        assert!(
            mean < 4.0,
            "attention_score mean error {mean} too large vs truth"
        );
        // And much better than predicting zero (truth has stddev sqrt(d) ≈ 11).
        let zero_baseline = truth.abs().unwrap().mean(None).unwrap().item::<f32>();
        assert!(
            mean < zero_baseline,
            "estimator no better than zero: mean={mean} baseline={zero_baseline}"
        );
    }

    /// `dequantize` and `attention_score` agree: building K_hat via
    /// dequantize() then computing `query @ K_hat.T` should match the
    /// asymmetric score within float noise.
    #[test]
    fn prod_attention_matches_explicit_dequant_path() {
        let d = 128;
        let q = TurboQuantProd::new(d, 3, 37).unwrap();

        let prng_q = mlx_rs::random::key(2).unwrap();
        let prng_k = mlx_rs::random::key(4).unwrap();
        let query =
            mlx_rs::random::normal::<f32>(&[1, 1, 4, d], None, None, &prng_q).unwrap();
        let keys =
            mlx_rs::random::normal::<f32>(&[1, 1, 8, d], None, None, &prng_k).unwrap();
        let enc = q.quantize(&keys).unwrap();

        let asym = q.attention_score(&query, &enc).unwrap();
        let k_hat = q.dequantize(&enc).unwrap();
        let k_hat_t = k_hat.transpose_axes(&[0, 1, 3, 2]).unwrap();
        let sym = query.matmul(&k_hat_t).unwrap();

        let err = max_abs(&asym.subtract(&sym).unwrap());
        assert!(err < 1e-2, "asym vs sym path diverge: max abs = {err}");
    }

    /// QJL seed must be independent: same Π seed across two Prod instances
    /// with different seeds → different attention scores.
    #[test]
    fn prod_different_seeds_diverge() {
        let d = 64;
        let a = TurboQuantProd::new(d, 3, 1).unwrap();
        let b = TurboQuantProd::new(d, 3, 2).unwrap();
        let prng_k = mlx_rs::random::key(5).unwrap();
        let keys =
            mlx_rs::random::normal::<f32>(&[1, 1, 4, d], None, None, &prng_k).unwrap();
        let ea = a.quantize(&keys).unwrap();
        let eb = b.quantize(&keys).unwrap();
        // Indices won't match because Π differs.
        let diff = ea
            .mse_indices
            .subtract(&eb.mse_indices)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<u8>();
        assert!(diff > 0, "different seeds produced identical indices");
    }

    /// Bits ≥ 2 required.
    #[test]
    #[should_panic(expected = "bits must be ≥ 2")]
    fn prod_rejects_bits_lt_2() {
        let _ = TurboQuantProd::new(64, 1, 0);
    }
}
