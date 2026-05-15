//! TurboQuant end-to-end quality validation.
//!
//! Loads a real qwen3-1.7B-bf16 checkpoint, runs the same prompt with
//! both the default `ConcatKeyValueCache` (lossless) and a
//! `TurboQuantKVCache(K3, V2)` (quantised), then compares the top-5
//! logits at the last position via KL divergence.
//!
//! Gated `#[ignore]` — requires:
//!   - `mlx-community/Qwen3-1.7B-bf16` cached under `$HOME/.cache/mlx-rs-bench`
//!     or `$MLX_LM_BENCH_CACHE` (same env var the bench harness uses).
//!   - Pulls via `hf download` if missing.
//!
//! Run with:
//!     cargo test -p mlx-lm --test turboquant_parity -- --ignored --nocapture

use std::path::PathBuf;
use std::process::Command;

use mlx_lm::cache::turboquant::cache::{TurboQuantConfig, TurboQuantKVCache};
use mlx_lm::cache::{ConcatKeyValueCache, KeyValueCache};
use mlx_lm::models::qwen3::{load_qwen3_model, Model as Qwen3Model, ModelInput};
use mlx_rs::module::Module;
use mlx_rs::ops::indexing::{Ellipsis, IndexOp};
use mlx_rs::transforms::eval;
use mlx_rs::Array;

const REPO_ID: &str = "mlx-community/Qwen3-1.7B-bf16";

fn bench_cache_root() -> PathBuf {
    if let Some(p) = std::env::var_os("MLX_LM_BENCH_CACHE") {
        return PathBuf::from(p);
    }
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(xdg).join("mlx-rs-bench");
    }
    home.join(".cache").join("mlx-rs-bench")
}

fn ensure_model() -> Option<PathBuf> {
    let cache = bench_cache_root().join(REPO_ID);
    if cache.join("config.json").is_file() {
        return Some(cache);
    }
    if std::env::var_os("MLX_LM_BENCH_NO_DOWNLOAD").is_some() {
        return None;
    }
    let _ = std::fs::create_dir_all(&cache);
    let status = Command::new("hf")
        .args([
            "download",
            REPO_ID,
            "--local-dir",
            cache.to_str().unwrap_or_default(),
        ])
        .status();
    matches!(status, Ok(s) if s.success()).then_some(cache)
}

/// Run a single forward pass on `prompt` with the given pre-populated
/// cache and return the logits row at the last token, shape `[vocab]`.
fn last_token_logits<C>(
    model: &mut Qwen3Model,
    cache: &mut Vec<Option<C>>,
    prompt: &Array,
) -> Vec<f32>
where
    C: KeyValueCache + Default,
{
    let mask: Option<Array> = None;
    let input = ModelInput {
        inputs: prompt,
        mask: mask.as_ref(),
        cache,
    };
    let logits = model.forward(input).expect("model forward");
    // logits shape: [B=1, S, vocab] — take the last position.
    let last = logits.shape()[1] - 1;
    let row = logits.index((Ellipsis, last, ..));
    let row = row.squeeze_axes(&[0][..]).expect("squeeze batch");
    eval([&row]).unwrap();
    row.as_dtype(mlx_rs::Dtype::Float32)
        .expect("cast logits to f32")
        .as_slice::<f32>()
        .to_vec()
}

/// Top-`k` indices of `logits` by descending value.
fn topk_indices(logits: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    idx.truncate(k);
    idx
}

/// KL(p || q) between two softmax distributions derived from `logits_p`,
/// `logits_q`, restricted to the indices in `top_idx`. Doing it over the
/// top-k subset matches Python TurboQuant's per-token quality metric.
fn topk_kl(logits_p: &[f32], logits_q: &[f32], top_idx: &[usize]) -> f32 {
    let max_p = top_idx.iter().map(|&i| logits_p[i]).fold(f32::NEG_INFINITY, f32::max);
    let max_q = top_idx.iter().map(|&i| logits_q[i]).fold(f32::NEG_INFINITY, f32::max);
    let mut exp_p = Vec::with_capacity(top_idx.len());
    let mut exp_q = Vec::with_capacity(top_idx.len());
    for &i in top_idx {
        exp_p.push((logits_p[i] - max_p).exp());
        exp_q.push((logits_q[i] - max_q).exp());
    }
    let sum_p: f32 = exp_p.iter().sum();
    let sum_q: f32 = exp_q.iter().sum();
    let mut kl = 0.0;
    for (p, q) in exp_p.iter().zip(exp_q.iter()) {
        let pn = p / sum_p;
        let qn = q / sum_q;
        if pn > 0.0 && qn > 0.0 {
            kl += pn * (pn / qn).ln();
        }
    }
    kl
}

#[test]
#[ignore = "requires Qwen3-1.7B-bf16 checkpoint + ~3 GB download"]
fn turboquant_k3v2_logits_track_fp16_within_threshold() {
    let dir = ensure_model().expect("Qwen3-1.7B-bf16 unavailable");
    let mut model_fp = load_qwen3_model(&dir).expect("model load (fp16 path)");
    let mut model_tq = load_qwen3_model(&dir).expect("model load (TQ path)");

    // Short synthetic prompt — the test exercises one decode-style
    // forward pass, not a long prefill, so quality only depends on
    // single-token attention.
    let prompt_ids: Vec<i32> = (1000..1064).collect();
    let prompt = Array::from_slice(&prompt_ids, &[1, prompt_ids.len() as i32]);

    let head_dim = model_fp.head_dim();
    let num_layers = model_fp.layer_count();

    let mut cache_fp: Vec<Option<ConcatKeyValueCache>> =
        (0..num_layers).map(|_| Some(ConcatKeyValueCache::new())).collect();
    let mut cache_tq: Vec<Option<TurboQuantKVCache>> = (0..num_layers)
        .map(|i| {
            Some(
                TurboQuantKVCache::new(TurboQuantConfig::new(head_dim, 0x000A_1115 + i as u64))
                    .expect("TurboQuantKVCache::new"),
            )
        })
        .collect();

    let logits_fp = last_token_logits(&mut model_fp, &mut cache_fp, &prompt);
    let logits_tq = last_token_logits(&mut model_tq, &mut cache_tq, &prompt);

    let top_idx = topk_indices(&logits_fp, 32);
    let kl = topk_kl(&logits_fp, &logits_tq, &top_idx);
    eprintln!("top-32 KL(fp16 || tq_k3v2) = {kl:.4}");

    // Loose threshold: KL < 0.5 means TQ stays in the same ballpark as
    // fp16 on the top-32 distribution. The paper's "quality neutral"
    // claim is at 3.5 bits-per-coord (K3.5V3.5 effectively), so K3V2 at
    // d=128 sits below neutrality — we allow more headroom.
    assert!(
        kl < 0.5,
        "TurboQuant K3V2 diverged from fp16 in top-32 distribution: KL = {kl}"
    );
}
