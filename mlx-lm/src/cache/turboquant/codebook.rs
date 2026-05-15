//! Lloyd-Max codebooks for TurboQuant.
//!
//! After the random rotation Π each coordinate of a unit vector in `ℝᵈ`
//! follows the closed-form PDF
//!
//!   f(x; d) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2)   on [-1, 1]
//!
//! a scaled Beta distribution that concentrates near zero as `d` grows.
//! Lloyd-Max iteration on this analytical density gives the MSE-optimal
//! scalar quantizer per coordinate — *data-oblivious*, no calibration set.
//!
//! Codebooks are tiny (≤ 16 centroids + 17 boundaries per (d, b) pair) and
//! identical across all models for matching `(head_dim, bits)`. We ship
//! the four most-likely combinations as JSON bundled via `include_str!`:
//!
//!   - `(d, b)` ∈ ({64, 128}, {1, 2, 3, 4})  — 8 files, each <2 KB.
//!
//! For any other `(d, b)` requested at runtime we recompute (≈ 0.3 s for
//! `d = 128, b = 4` on a single CPU core) and memoise.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};

/// A Lloyd-Max codebook for one `(d, b)` pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Codebook {
    /// Embedding dimension this codebook was computed for.
    pub d: i32,
    /// Bits per coordinate (`n_clusters = 2^bits`).
    pub bits: i32,
    /// Cluster centroids on `[-1, 1]`, sorted ascending. Length = `2^bits`.
    pub centroids: Vec<f32>,
    /// Sorted boundaries delimiting the buckets. Length = `2^bits + 1`,
    /// first element is `-1.0`, last is `1.0`, interior values are the
    /// midpoints between adjacent centroids.
    pub boundaries: Vec<f32>,
    /// Achieved MSE per coordinate (paper Table 1 reference). Diagnostic
    /// only — not used at runtime.
    pub mse_per_coord: f64,
}

/// Interior decision boundaries: `boundaries[1..len-1]` (excludes the
/// `-1`/`+1` endpoints). The `searchsorted` kernel works against these.
pub fn decision_boundaries(cb: &Codebook) -> &[f32] {
    let n = cb.boundaries.len();
    &cb.boundaries[1..n - 1]
}

// -------- Lloyd-Max iteration on closed-form Beta(d) PDF --------

/// Lanczos approximation to `ln Γ(x)` for `x > 0`. ~14-digit accurate over
/// the range we use (`x ≥ 0.5`); no external crate needed.
fn ln_gamma(x: f64) -> f64 {
    // Stirling-derived 8-term Lanczos approximation.
    // Coefficients from Numerical Recipes (3rd ed., §6.1).
    const COEFS: [f64; 8] = [
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];
    let g = 7.0;
    if x < 0.5 {
        // Reflection formula.
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let xm = x - 1.0;
        let mut a = 0.999_999_999_999_809_9;
        for (i, &c) in COEFS.iter().enumerate() {
            a += c / (xm + (i + 1) as f64);
        }
        let t = xm + g + 0.5;
        0.5 * (std::f64::consts::TAU).ln() + (xm + 0.5) * t.ln() - t + a.ln()
    }
}

/// `f(x; d)` Beta(d) PDF on `[-1, 1]`. Computed in `f64` for stability.
fn beta_pdf(x: f64, d: i32) -> f64 {
    if d < 3 {
        // Below this the PDF either degenerates (d=1, point masses at ±1)
        // or is the arcsine (d=2). TurboQuant only targets `d ≥ 32`.
        panic!("beta_pdf: d={d} too small; need d >= 3");
    }
    if !(-1.0..=1.0).contains(&x) {
        return 0.0;
    }
    // Clip to avoid log(0) at boundaries.
    let xc = x.clamp(-1.0 + 1e-15, 1.0 - 1e-15);
    let log_const = ln_gamma(d as f64 / 2.0)
        - 0.5 * std::f64::consts::PI.ln()
        - ln_gamma((d as f64 - 1.0) / 2.0);
    let exponent = (d as f64 - 3.0) / 2.0;
    let log_val = log_const + exponent * (1.0 - xc * xc).ln();
    log_val.exp()
}

/// Simpson's rule over `[lo, hi]` with `n` intervals (must be even).
fn simpson<F: Fn(f64) -> f64>(f: F, lo: f64, hi: f64, n: usize) -> f64 {
    assert!(n.is_multiple_of(2) && n >= 2);
    let h = (hi - lo) / n as f64;
    let mut s = f(lo) + f(hi);
    for i in 1..n {
        let x = lo + (i as f64) * h;
        s += if i % 2 == 0 { 2.0 } else { 4.0 } * f(x);
    }
    s * h / 3.0
}

/// `E[X | lo < X < hi]` under the Beta(d) PDF.
fn conditional_mean(lo: f64, hi: f64, d: i32) -> f64 {
    if hi - lo < 1e-18 {
        return (lo + hi) * 0.5;
    }
    let num = simpson(|x| x * beta_pdf(x, d), lo, hi, 1024);
    let den = simpson(|x| beta_pdf(x, d), lo, hi, 1024);
    if den < 1e-30 {
        (lo + hi) * 0.5
    } else {
        num / den
    }
}

/// MSE of a given centroid placement under the Beta(d) PDF.
fn mse_cost(centroids: &[f64], d: i32) -> f64 {
    let n = centroids.len();
    let mut boundaries = vec![0.0; n + 1];
    boundaries[0] = -1.0;
    boundaries[n] = 1.0;
    for i in 0..n - 1 {
        boundaries[i + 1] = (centroids[i] + centroids[i + 1]) * 0.5;
    }
    let mut cost = 0.0;
    for i in 0..n {
        let c = centroids[i];
        let lo = boundaries[i];
        let hi = boundaries[i + 1];
        cost += simpson(|x| (x - c).powi(2) * beta_pdf(x, d), lo, hi, 1024);
    }
    cost
}

/// Compute the Lloyd-Max codebook for `(d, bits)` from scratch.
///
/// Initialises centroids at quantile midpoints of the empirical CDF (a
/// 10⁴-point grid) and iterates until the MSE change drops below `tol`
/// or `max_iter` steps elapse. For the small `(d, b)` combos we target
/// this converges in <20 iterations.
pub fn compute_codebook(d: i32, bits: i32) -> Codebook {
    assert!(d >= 3, "compute_codebook: need d >= 3");
    assert!((1..=8).contains(&bits), "compute_codebook: bits must be 1..=8");
    let n_clusters = 1usize << bits as u32;
    let max_iter = 200;
    let tol = 1e-12;

    // Quantile-midpoint init using a CDF grid.
    let grid_n = 10_000;
    let x_grid: Vec<f64> = (0..grid_n)
        .map(|i| -1.0 + 2.0 * (i as f64) / (grid_n as f64 - 1.0))
        .collect();
    let pdf_vals: Vec<f64> = x_grid.iter().map(|&x| beta_pdf(x, d)).collect();
    let dx = x_grid[1] - x_grid[0];
    let mut cdf = vec![0.0; grid_n];
    cdf[0] = pdf_vals[0] * dx;
    for i in 1..grid_n {
        cdf[i] = cdf[i - 1] + pdf_vals[i] * dx;
    }
    let total = cdf[grid_n - 1];
    for v in cdf.iter_mut() {
        *v /= total;
    }
    let mut centroids: Vec<f64> = (0..n_clusters)
        .map(|i| {
            let q_mid = (i as f64 + 0.5) / n_clusters as f64;
            // Find first idx with cdf[idx] >= q_mid.
            let idx = cdf.iter().position(|&v| v >= q_mid).unwrap_or(grid_n - 1);
            x_grid[idx]
        })
        .collect();
    centroids.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Lloyd-Max iteration.
    let mut prev_cost = f64::INFINITY;
    for _ in 0..max_iter {
        let mut boundaries = vec![0.0_f64; n_clusters + 1];
        boundaries[0] = -1.0;
        boundaries[n_clusters] = 1.0;
        for i in 0..n_clusters - 1 {
            boundaries[i + 1] = (centroids[i] + centroids[i + 1]) * 0.5;
        }
        let mut new_centroids = vec![0.0_f64; n_clusters];
        for i in 0..n_clusters {
            new_centroids[i] = conditional_mean(boundaries[i], boundaries[i + 1], d);
        }
        let cost = mse_cost(&new_centroids, d);
        centroids = new_centroids;
        if (prev_cost - cost).abs() < tol {
            break;
        }
        prev_cost = cost;
    }

    // Final boundaries.
    let mut boundaries = vec![0.0_f64; n_clusters + 1];
    boundaries[0] = -1.0;
    boundaries[n_clusters] = 1.0;
    for i in 0..n_clusters - 1 {
        boundaries[i + 1] = (centroids[i] + centroids[i + 1]) * 0.5;
    }
    let cost = mse_cost(&centroids, d);

    Codebook {
        d,
        bits,
        centroids: centroids.into_iter().map(|c| c as f32).collect(),
        boundaries: boundaries.into_iter().map(|c| c as f32).collect(),
        mse_per_coord: cost,
    }
}

// -------- Bundled JSON + runtime cache --------

/// Bundled JSON for the most common `(d, b)` combinations. Loaded at
/// build time; falls back to runtime compute for any other pair.
fn bundled_json(d: i32, bits: i32) -> Option<&'static str> {
    match (d, bits) {
        (64, 1) => Some(include_str!("../../../data/turboquant_codebooks/codebook_d64_b1.json")),
        (64, 2) => Some(include_str!("../../../data/turboquant_codebooks/codebook_d64_b2.json")),
        (64, 3) => Some(include_str!("../../../data/turboquant_codebooks/codebook_d64_b3.json")),
        (64, 4) => Some(include_str!("../../../data/turboquant_codebooks/codebook_d64_b4.json")),
        (128, 1) => Some(include_str!("../../../data/turboquant_codebooks/codebook_d128_b1.json")),
        (128, 2) => Some(include_str!("../../../data/turboquant_codebooks/codebook_d128_b2.json")),
        (128, 3) => Some(include_str!("../../../data/turboquant_codebooks/codebook_d128_b3.json")),
        (128, 4) => Some(include_str!("../../../data/turboquant_codebooks/codebook_d128_b4.json")),
        _ => None,
    }
}

type CodebookCache = Mutex<HashMap<(i32, i32), &'static Codebook>>;

fn cache() -> &'static CodebookCache {
    static CACHE: OnceLock<CodebookCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Get a codebook for `(d, bits)`. Loads from bundled JSON if available,
/// otherwise computes once and memoises.
///
/// Returned reference has `'static` lifetime — the codebook is leaked into
/// the cache map and lives for the program lifetime. This is fine: there
/// are at most ~16 codebooks per process and each is <2 KB.
pub fn get_codebook(d: i32, bits: i32) -> &'static Codebook {
    if let Some(existing) = cache().lock().unwrap().get(&(d, bits)).copied() {
        return existing;
    }
    let cb = if let Some(json) = bundled_json(d, bits) {
        serde_json::from_str::<Codebook>(json).expect("bundled codebook JSON parse")
    } else {
        compute_codebook(d, bits)
    };
    let leaked: &'static Codebook = Box::leak(Box::new(cb));
    cache().lock().unwrap().insert((d, bits), leaked);
    leaked
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Beta PDF integrates to 1.
    #[test]
    fn beta_pdf_normalises() {
        for d in [3, 16, 64, 128] {
            let mass = simpson(|x| beta_pdf(x, d), -1.0, 1.0, 4096);
            assert!((mass - 1.0).abs() < 1e-3, "d={d}: mass={mass}");
        }
    }

    /// Lloyd-Max produces sorted centroids inside [-1, 1].
    #[test]
    fn computed_codebook_is_well_formed() {
        for bits in [1, 2, 3, 4] {
            let cb = compute_codebook(64, bits);
            let n = 1usize << bits;
            assert_eq!(cb.centroids.len(), n);
            assert_eq!(cb.boundaries.len(), n + 1);
            for i in 1..n {
                assert!(
                    cb.centroids[i - 1] < cb.centroids[i],
                    "bits={bits}: centroids not sorted"
                );
            }
            assert!((cb.boundaries[0] - -1.0).abs() < 1e-6);
            assert!((cb.boundaries[n] - 1.0).abs() < 1e-6);
            for c in &cb.centroids {
                assert!((-1.0..=1.0).contains(c));
            }
        }
    }

    /// Symmetry: codebook should be symmetric around 0 (the Beta PDF is even).
    #[test]
    fn computed_codebook_is_symmetric() {
        let cb = compute_codebook(64, 3);
        let n = cb.centroids.len();
        for i in 0..n {
            let lhs = cb.centroids[i];
            let rhs = -cb.centroids[n - 1 - i];
            assert!(
                (lhs - rhs).abs() < 1e-4,
                "asymmetric: c[{i}]={lhs} vs -c[{}]={rhs}",
                n - 1 - i,
            );
        }
    }

    /// All bundled codebooks load + parse successfully.
    #[test]
    fn all_bundled_codebooks_load() {
        for d in [64, 128] {
            for bits in [1, 2, 3, 4] {
                let cb = get_codebook(d, bits);
                let n = 1usize << bits;
                assert_eq!(cb.d, d);
                assert_eq!(cb.bits, bits);
                assert_eq!(cb.centroids.len(), n);
                assert_eq!(cb.boundaries.len(), n + 1);
            }
        }
    }

    /// Bundled codebook matches a fresh recomputation within a tight tolerance.
    /// We use d=64, b=2 (small + fast).
    #[test]
    fn bundled_matches_fresh_compute() {
        let bundled = get_codebook(64, 2);
        let fresh = compute_codebook(64, 2);
        for (i, (&b, &f)) in bundled
            .centroids
            .iter()
            .zip(fresh.centroids.iter())
            .enumerate()
        {
            assert!(
                (b - f).abs() < 1e-3,
                "centroid {i} mismatch: bundled={b} fresh={f}"
            );
        }
    }

    /// Interior decision boundaries have the expected count and ordering.
    #[test]
    fn decision_boundaries_well_formed() {
        let cb = get_codebook(128, 4);
        let interior = decision_boundaries(cb);
        assert_eq!(interior.len(), (1 << 4) - 1, "wrong interior count");
        for i in 1..interior.len() {
            assert!(interior[i - 1] < interior[i], "interior not sorted");
        }
    }

    /// On-demand compute for an unbundled `(d, b)` works.
    #[test]
    fn unbundled_pair_computes_on_demand() {
        let cb = get_codebook(32, 2);
        assert_eq!(cb.d, 32);
        assert_eq!(cb.bits, 2);
        assert_eq!(cb.centroids.len(), 4);
    }

    /// Maintenance task: regenerate the bundled JSONs from a fresh compute.
    /// Ignored by default; run with
    ///
    ///   cargo test -p mlx-lm --lib regenerate_bundled_codebooks -- --ignored --nocapture
    ///
    /// to refresh the on-disk files (e.g. after the Lloyd-Max parameters
    /// change). The result must be reviewed by hand — this writes the
    /// files unconditionally.
    #[test]
    #[ignore = "regeneration-only; rerun if codebook math changes"]
    fn regenerate_bundled_codebooks() {
        let out_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("data")
            .join("turboquant_codebooks");
        std::fs::create_dir_all(&out_dir).unwrap();
        for d in [64, 128] {
            for bits in [1, 2, 3, 4] {
                let cb = compute_codebook(d, bits);
                let json = serde_json::to_string_pretty(&cb).unwrap();
                let path = out_dir.join(format!("codebook_d{d}_b{bits}.json"));
                std::fs::write(&path, json).unwrap();
                println!("wrote {} (mse={:.3e})", path.display(), cb.mse_per_coord);
            }
        }
    }
}
