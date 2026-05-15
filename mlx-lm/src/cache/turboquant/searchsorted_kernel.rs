//! Metal kernel: bucket lookup against a small sorted boundary table.
//!
//! Equivalent to `torch.searchsorted(boundaries, values)` for the small-`k`
//! case the TurboQuant codebooks use (≤ 15 decision boundaries for 4-bit;
//! `MAX_BOUNDARIES = 16` lets the kernel accommodate any future
//! configuration up through 4 bits cleanly). For each input value we
//! return the count of boundary slots `< value`, which is the bucket index
//! the caller's centroid table is indexed by.
//!
//! Each kernel thread processes one input value: a linear scan over the
//! constant-sized boundary table running entirely in registers. Cheaper
//! than a pure-ops `(boundaries <= val).sum() - 1` graph because it avoids
//! the broadcast-and-reduce buffer round-trip.

use mlx_rs::error::{Exception, Result};
use mlx_rs::fast::{metal_kernel, MetalKernel, MetalKernelConfig};
use mlx_rs::{Array, Dtype, Stream};

/// Maximum number of decision boundaries the kernel accepts. The
/// TurboQuant codebook for `bits ∈ {1, 2, 3, 4}` uses 1/3/7/15 decision
/// boundaries respectively; 16 leaves headroom (e.g. a 5-bit codebook
/// would need 31 — out of scope here).
pub const MAX_BOUNDARIES: i32 = 16;

const KERNEL_NAME: &str = "tq_searchsorted_bucket";

/// Metal source. Templated on `T` (the value/boundary dtype) and `K`
/// (the actual number of decision boundaries, ≤ MAX_BOUNDARIES). The
/// total element count `n_total` is passed as a scalar buffer rather
/// than a template — mlx-rs's `metal_kernel` declares each input as a
/// buffer-named parameter, and template names can't collide with buffer
/// names.
///
/// The kernel runs one thread per value. Threads form a flat 1-D grid
/// covering the entire input tensor; bounds-checking against `n_total`
/// handles non-multiple-of-threadgroup tails.
const KERNEL_SOURCE: &str = r#"
    uint gid = thread_position_in_grid.x;
    if (gid >= uint(n_total)) {
        return;
    }
    T v = values[gid];
    uint count = 0;
    // Linear scan over the K decision boundaries — K is small (<=15) so
    // this unrolls nicely and stays in registers.
    for (uint i = 0; i < uint(K); ++i) {
        count += (v >= T(boundaries[i])) ? 1u : 0u;
    }
    indices[gid] = uint8_t(count);
"#;

/// Compile a fresh searchsorted kernel handle. Cache for the lifetime of
/// the cache layer (kernel compilation is amortised by MLX, but holding
/// the handle avoids re-compilation overhead per call).
pub fn make_searchsorted_kernel() -> Result<MetalKernel> {
    metal_kernel(
        KERNEL_NAME,
        &["values", "boundaries", "n_total"],
        &["indices"],
        KERNEL_SOURCE,
        "",
        true,
        false,
    )
}

/// Apply `bucket = #{boundaries[i] <= value}` per element.
///
/// `values` may be any shape; the output has the same shape, dtype `uint8`.
/// `boundaries` is `[K]` sorted ascending with `K ≤ MAX_BOUNDARIES`.
///
/// Threadgroup size is fixed at 256 (a generous Metal SIMD-wave multiple);
/// the grid x-dim covers the full element count rounded up.
pub fn searchsorted_bucket(
    kernel: &MetalKernel,
    values: &Array,
    boundaries: &Array,
) -> Result<Array> {
    if boundaries.ndim() != 1 {
        return Err(Exception::custom(format!(
            "searchsorted_bucket: boundaries must be 1-D, got shape {:?}",
            boundaries.shape()
        )));
    }
    let k = boundaries.shape()[0];
    if !(1..=MAX_BOUNDARIES).contains(&k) {
        return Err(Exception::custom(format!(
            "searchsorted_bucket: K={k} out of range 1..={MAX_BOUNDARIES}"
        )));
    }

    let dtype = values.dtype();
    if dtype != boundaries.dtype() {
        return Err(Exception::custom(format!(
            "searchsorted_bucket: values dtype {dtype:?} != boundaries dtype {:?}",
            boundaries.dtype()
        )));
    }
    if !matches!(dtype, Dtype::Float32 | Dtype::Bfloat16 | Dtype::Float16) {
        return Err(Exception::custom(format!(
            "searchsorted_bucket: unsupported dtype {dtype:?}; expected fp16/bf16/fp32"
        )));
    }

    let total: i32 = values.shape().iter().product();
    const TG: i32 = 256;
    // Round `total` up to a multiple of TG without using unstable `div_ceil`.
    let grid_x = total.saturating_add(TG - 1) / TG * TG;

    let config = MetalKernelConfig::new()
        .add_output(values.shape().to_vec(), Dtype::Uint8)
        .grid(grid_x, 1, 1)
        .thread_group(TG, 1, 1)
        .add_template("T", dtype)?
        .add_template("K", k)?;

    let n_scalar = Array::from_int(total);
    let outs = kernel.apply(
        &[values.clone(), boundaries.clone(), n_scalar],
        config,
        Stream::default(),
    )?;
    outs.into_iter()
        .next()
        .ok_or_else(|| Exception::custom("searchsorted_bucket: kernel returned no outputs"))
}

/// Slow scalar Rust reference (test-only): returns the bucket index for
/// each value. Used to validate the kernel.
#[cfg(test)]
pub fn searchsorted_bucket_scalar(values: &[f32], boundaries: &[f32]) -> Vec<u8> {
    values
        .iter()
        .map(|&v| {
            let mut count = 0u8;
            for &b in boundaries {
                if v >= b {
                    count += 1;
                }
            }
            count
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::transforms::eval;

    fn compare_to_scalar(values: &[f32], boundaries: &[f32]) {
        let expected = searchsorted_bucket_scalar(values, boundaries);
        let kernel = make_searchsorted_kernel().unwrap();
        let v_arr = Array::from_slice(values, &[values.len() as i32]);
        let b_arr = Array::from_slice(boundaries, &[boundaries.len() as i32]);
        let got = searchsorted_bucket(&kernel, &v_arr, &b_arr).unwrap();
        eval([&got]).unwrap();
        let got_bytes = got.as_slice::<u8>().to_vec();
        assert_eq!(
            got_bytes, expected,
            "kernel vs scalar mismatch for boundaries={boundaries:?}"
        );
    }

    /// Uniform values across the whole [-1, 1] range with the canonical
    /// 4-bit boundaries (interior 15 decision points equally spaced).
    #[test]
    fn matches_scalar_for_4bit_boundaries() {
        let boundaries: Vec<f32> = (1..=15).map(|i| -1.0 + (i as f32) * 2.0 / 16.0).collect();
        let values: Vec<f32> = (0..64).map(|i| -1.0 + (i as f32) / 32.0).collect();
        compare_to_scalar(&values, &boundaries);
    }

    /// Edge cases: values below first boundary, exactly on a boundary,
    /// above last boundary.
    #[test]
    fn handles_boundary_edge_cases() {
        let boundaries = vec![-0.5_f32, 0.0, 0.5];
        let values = vec![-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0];
        compare_to_scalar(&values, &boundaries);
    }

    /// Single-boundary kernel (1-bit case).
    #[test]
    fn handles_single_boundary() {
        let boundaries = vec![0.0_f32];
        let values: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        compare_to_scalar(&values, &boundaries);
    }

    /// Maximum-K boundary table (16) for the largest packed-codebook case.
    #[test]
    fn handles_max_boundaries() {
        let boundaries: Vec<f32> = (1..=16).map(|i| -1.0 + (i as f32) * 2.0 / 17.0).collect();
        let values: Vec<f32> = (0..32).map(|i| -1.0 + (i as f32) / 16.0).collect();
        compare_to_scalar(&values, &boundaries);
    }

    /// Rejects K > MAX_BOUNDARIES.
    #[test]
    fn rejects_oversize_boundary_table() {
        let kernel = make_searchsorted_kernel().unwrap();
        let v = Array::from_slice(&[0.0f32; 4], &[4]);
        let b = Array::from_slice(&[0.0f32; 17], &[17]);
        let err = searchsorted_bucket(&kernel, &v, &b);
        assert!(err.is_err(), "should reject K=17 > MAX_BOUNDARIES");
    }

    /// Rejects mismatched dtypes.
    #[test]
    fn rejects_dtype_mismatch() {
        let kernel = make_searchsorted_kernel().unwrap();
        let v = Array::from_slice(&[0.0f32; 4], &[4]);
        let b = Array::from_slice(&[0u8; 4], &[4]);
        let err = searchsorted_bucket(&kernel, &v, &b);
        assert!(err.is_err(), "should reject dtype mismatch");
    }

    /// Batched shape: a `[3, 5, 7]` input should produce a `[3, 5, 7]`
    /// uint8 output.
    #[test]
    fn preserves_batched_shape() {
        let shape = &[3, 5, 7][..];
        let n = 3 * 5 * 7;
        let values: Vec<f32> = (0..n).map(|i| (i as f32 - 50.0) * 0.05).collect();
        let v_arr = Array::from_slice(&values, shape);
        let b_arr = Array::from_slice(&[-1.0_f32, 0.0, 1.0], &[3]);
        let kernel = make_searchsorted_kernel().unwrap();
        let got = searchsorted_bucket(&kernel, &v_arr, &b_arr).unwrap();
        eval([&got]).unwrap();
        assert_eq!(got.shape(), shape);
        assert_eq!(got.dtype(), Dtype::Uint8);
    }
}
