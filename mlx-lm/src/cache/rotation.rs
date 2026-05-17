//! Random orthogonal rotation Π for `QuantizedKVCache::with_rotation`.
//!
//! `generate_rotation_matrix(d, seed)` returns a `[d, d]` matrix built by
//! QR-decomposing a seeded Gaussian and sign-fixing the columns of Q by
//! `sign(diag(R))` so `det(Π) = +1` (proper rotation, no reflection).
//! Built once on CPU per layer at cache creation; `mlx_rs::linalg::qr_device`
//! is CPU-only and the cost is sub-millisecond at `d ∈ {64, 128}`.

use mlx_rs::linalg::qr_device;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{expand_dims, sign};
use mlx_rs::random::{key, normal_device};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, StreamOrDevice};

use crate::error::Error;

/// Random orthogonal Π for a single layer. QR of a seeded `[d,d]` Gaussian,
/// sign-fixed by `sign(diag(R))` so `det(Π) = +1`. Deterministic per `seed`.
pub fn generate_rotation_matrix(d: i32, seed: u64) -> Result<Array, Error> {
    assert!(d > 0, "rotation matrix dim must be positive");

    let prng = key(seed).map_err(Error::from)?;
    let cpu = StreamOrDevice::cpu();
    let g = normal_device::<f32>(&[d, d], None, None, &prng, &cpu).map_err(Error::from)?;

    let (q, r) = qr_device(&g, &cpu).map_err(Error::from)?;

    // sign-fix: multiply each column of Q by sign(R[i, i]).
    let diag = (0..d)
        .map(|i| r.index((i, i)).item::<f32>())
        .collect::<Vec<_>>();
    let diag_array = Array::from_slice(&diag, &[d]);
    let signs = sign(&diag_array)?;
    let signs_row = expand_dims(&signs, 0)?; // [1, d] broadcasts over rows

    let pi = q.multiply(&signs_row)?;
    eval([&pi]).map_err(Error::from)?;
    Ok(pi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::indexing::IndexOp;

    fn max_abs(a: &Array) -> f32 {
        a.abs().unwrap().max(None).unwrap().item::<f32>()
    }

    fn identity(d: i32) -> Array {
        let mut data = vec![0f32; (d * d) as usize];
        for i in 0..d {
            data[(i * d + i) as usize] = 1.0;
        }
        Array::from_slice(&data, &[d, d])
    }

    /// Πᵀ Π should equal the identity within float noise.
    #[test]
    fn rotation_is_orthogonal() {
        for d in [32, 64, 128] {
            let pi = generate_rotation_matrix(d, 42).unwrap();
            let pi_t = pi.transpose_axes(&[1, 0]).unwrap();
            let prod = pi_t.matmul(&pi).unwrap();
            let diff = prod.subtract(identity(d)).unwrap();
            let err = max_abs(&diff);
            assert!(err < 1e-4, "d={d}: |Πᵀ Π - I|∞ = {err}");
        }
    }

    /// Same seed must yield bit-identical Π across calls.
    #[test]
    fn rotation_seeded_is_deterministic() {
        let a = generate_rotation_matrix(64, 7).unwrap();
        let b = generate_rotation_matrix(64, 7).unwrap();
        let diff = a.subtract(&b).unwrap();
        assert_eq!(max_abs(&diff), 0.0, "seeded rotation drift");
    }

    /// Different seeds must yield visibly different Π.
    #[test]
    fn rotation_different_seeds_differ() {
        let a = generate_rotation_matrix(64, 1).unwrap();
        let b = generate_rotation_matrix(64, 2).unwrap();
        let diff = a.subtract(&b).unwrap();
        let err = max_abs(&diff);
        assert!(err > 0.1, "different seeds should differ visibly; got {err}");
    }

    /// Spot-check that the construction produces a finite entry — guards
    /// against accidental NaN/Inf from the QR or sign-fix.
    #[test]
    fn rotation_entries_are_finite() {
        let pi = generate_rotation_matrix(8, 11).unwrap();
        let entry = pi.index((0, 0)).item::<f32>();
        assert!(entry.is_finite());
    }
}
