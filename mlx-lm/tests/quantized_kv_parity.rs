//! `QuantizedKVCache` V2-LEAN end-to-end quality validation.
//!
//! Same shape as `turboquant_parity.rs`: load qwen3-1.7B-bf16, run a
//! prompt through both fp16 cache and `QuantizedKVCache(bits=4)`,
//! compare top-32 logit KL.
//!
//! Run with:
//!     cargo test -p mlx-lm --test quantized_kv_parity \
//!         -- --ignored --nocapture

use std::path::PathBuf;
use std::process::Command;

use mlx_lm::cache::{ConcatKeyValueCache, KeyValueCache, QuantizedKVCache};
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
    let last = logits.shape()[1] - 1;
    let row = logits.index((Ellipsis, last, ..));
    let row = row.squeeze_axes(&[0][..]).expect("squeeze batch");
    eval([&row]).unwrap();
    row.as_dtype(mlx_rs::Dtype::Float32)
        .expect("cast logits to f32")
        .as_slice::<f32>()
        .to_vec()
}

fn topk_indices(logits: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    idx.truncate(k);
    idx
}

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
fn quantized_kv_q4_logits_track_fp16_within_threshold() {
    let dir = ensure_model().expect("Qwen3-1.7B-bf16 unavailable");
    let mut model_fp = load_qwen3_model(&dir).expect("model load (fp16 path)");
    let mut model_q = load_qwen3_model(&dir).expect("model load (quant path)");

    let prompt_ids: Vec<i32> = (1000..1064).collect();
    let prompt = Array::from_slice(&prompt_ids, &[1, prompt_ids.len() as i32]);

    let num_layers = model_fp.layer_count();

    let mut cache_fp: Vec<Option<ConcatKeyValueCache>> =
        (0..num_layers).map(|_| Some(ConcatKeyValueCache::new())).collect();
    // 4-bit affine quant, group_size 64. mlx-rs's quantize requires
    // head_dim % group_size == 0 — qwen3 head_dim=128 satisfies.
    let mut cache_q: Vec<Option<QuantizedKVCache>> = (0..num_layers)
        .map(|_| Some(QuantizedKVCache::with_config(256, 64, 4)))
        .collect();

    let logits_fp = last_token_logits(&mut model_fp, &mut cache_fp, &prompt);
    let logits_q = last_token_logits(&mut model_q, &mut cache_q, &prompt);

    let top_idx = topk_indices(&logits_fp, 32);
    let kl = topk_kl(&logits_fp, &logits_q, &top_idx);
    eprintln!("top-32 KL(fp16 || qkv_q4) = {kl:.4}");

    // Threshold: 0.5 (same as the TurboQuant parity test). 4-bit affine
    // group quant at group=64 is normally near-lossless.
    assert!(
        kl < 0.5,
        "QuantizedKVCache q4 diverged from fp16 in top-32 distribution: KL = {kl}"
    );
}

#[test]
#[ignore = "requires Qwen3-1.7B-bf16 checkpoint + ~3 GB download"]
fn quantized_kv_q4_v2lean_logits_match_dequant_path() {
    let dir = ensure_model().expect("Qwen3-1.7B-bf16 unavailable");
    let mut model_a = load_qwen3_model(&dir).expect("model load A");
    let mut model_b = load_qwen3_model(&dir).expect("model load B");

    let prompt_ids: Vec<i32> = (1000..1064).collect();
    let prompt = Array::from_slice(&prompt_ids, &[1, prompt_ids.len() as i32]);
    let num_layers = model_a.layer_count();

    let mut cache_dequant: Vec<Option<QuantizedKVCache>> = (0..num_layers)
        .map(|_| Some(QuantizedKVCache::with_config(256, 64, 4)))
        .collect();
    let mut cache_v2: Vec<Option<QuantizedKVCache>> = (0..num_layers)
        .map(|_| Some(QuantizedKVCache::with_config(256, 64, 4).with_quantized_matmul()))
        .collect();

    let logits_dequant = last_token_logits(&mut model_a, &mut cache_dequant, &prompt);
    let logits_v2 = last_token_logits(&mut model_b, &mut cache_v2, &prompt);

    let top_idx = topk_indices(&logits_dequant, 32);
    let kl = topk_kl(&logits_dequant, &logits_v2, &top_idx);
    eprintln!("top-32 KL(dequant || v2_lean) = {kl:.6}");

    // V2 LEAN and dequant-on-read share the affine quant; they differ
    // only in the matmul path. KL should be small relative to the
    // fp16-vs-q4 baseline (~0.27).
    assert!(kl < 0.05, "V2 LEAN deviates from dequant path: KL = {kl}");
}

#[test]
#[ignore = "requires Qwen3-1.7B-bf16 checkpoint + ~3 GB download"]
fn quantized_kv_q8_logits_near_lossless() {
    let dir = ensure_model().expect("Qwen3-1.7B-bf16 unavailable");
    let mut model_fp = load_qwen3_model(&dir).expect("model load (fp16 path)");
    let mut model_q = load_qwen3_model(&dir).expect("model load (quant path)");

    let prompt_ids: Vec<i32> = (1000..1064).collect();
    let prompt = Array::from_slice(&prompt_ids, &[1, prompt_ids.len() as i32]);

    let num_layers = model_fp.layer_count();

    let mut cache_fp: Vec<Option<ConcatKeyValueCache>> =
        (0..num_layers).map(|_| Some(ConcatKeyValueCache::new())).collect();
    let mut cache_q: Vec<Option<QuantizedKVCache>> = (0..num_layers)
        .map(|_| Some(QuantizedKVCache::with_config(256, 64, 8)))
        .collect();

    let logits_fp = last_token_logits(&mut model_fp, &mut cache_fp, &prompt);
    let logits_q = last_token_logits(&mut model_q, &mut cache_q, &prompt);

    let top_idx = topk_indices(&logits_fp, 32);
    let kl = topk_kl(&logits_fp, &logits_q, &top_idx);
    eprintln!("top-32 KL(fp16 || qkv_q8) = {kl:.4}");

    // q8 should be tight to fp16.
    assert!(
        kl < 0.05,
        "QuantizedKVCache q8 not near-lossless: KL = {kl}"
    );
}
