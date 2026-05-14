//! Decode-throughput bench for mlx-lm text decoders.
//!
//! Covers the decoder-only models that ship with the crate today:
//!   - Qwen3 plain decoder (`mlx_lm::models::qwen3`)
//!   - Llama 3.2 (`mlx_lm::models::llama`)
//!
//! Each variant is benched at a short (13-token) and long (1024-token)
//! prompt, decoding `DECODE_TOKENS` tokens. The long-prompt case stresses
//! the KV cache and the attention path.
//!
//! Model checkpoints are pulled lazily on first use via the `hf` CLI into
//! the cache directory printed at bench start. Individual cells skip
//! silently if `hf` is unavailable or the download fails — CI without
//! `hf` skips the entire bench rather than failing.
//!
//! Run with:
//!
//!     cargo bench -p mlx-lm --bench lm_decode

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mlx_lm::cache::ConcatKeyValueCache;
use mlx_lm::models::{
    llama::{load_llama_model, Generate as LlamaGenerate, Model as LlamaModel},
    qwen3::{load_qwen3_model, Generate as Qwen3Generate, Model as Qwen3Model},
};
use mlx_rs::{
    ops::indexing::{IndexOp, NewAxis},
    transforms::eval,
    Array,
};

const DECODE_TOKENS: i32 = 100;
const LONG_PROMPT_LEN: usize = 1024;
const SHORT_PROMPT_LEN: usize = 13;
const WARMUP_TOKENS: i32 = 4;
const SAMPLE_SIZE: usize = 10;
const MEASUREMENT_SECS: u64 = 20;

/// Resolve `<cache>/<repo_id>`; download via `hf` CLI on first miss.
///
/// Set `MLX_LM_BENCH_NO_DOWNLOAD=1` to skip the download step and drop cells
/// whose checkpoint isn't already present — useful when running a filtered
/// subset to avoid pulling unrelated tiers in the background.
fn ensure_model(repo_id: &str) -> Option<PathBuf> {
    let cache = bench_cache_root().join(repo_id);
    let status = checkpoint_status(&cache);
    match status {
        CheckpointStatus::Complete => return Some(cache),
        CheckpointStatus::Partial { missing } => {
            eprintln!(
                "skipping {repo_id}: partial checkpoint at {} (missing {} shard(s): {}). \
                 Resume with: hf download {repo_id} --local-dir {}",
                cache.display(),
                missing.len(),
                missing.join(", "),
                cache.display(),
            );
            return None;
        }
        CheckpointStatus::Missing => {}
    }
    if std::env::var_os("MLX_LM_BENCH_NO_DOWNLOAD").is_some() {
        return None;
    }
    if std::fs::create_dir_all(&cache).is_err() {
        eprintln!(
            "skipping {repo_id}: could not create cache dir {}",
            cache.display()
        );
        return None;
    }
    let status = Command::new("hf")
        .args([
            "download",
            repo_id,
            "--local-dir",
            cache.to_str().unwrap_or_default(),
        ])
        .status();
    match status {
        Ok(s) if s.success() => Some(cache),
        Ok(s) => {
            eprintln!("skipping {repo_id}: `hf download` exited {s}");
            None
        }
        Err(e) => {
            eprintln!("skipping {repo_id}: `hf` not available ({e})");
            None
        }
    }
}

/// Classification of an on-disk checkpoint directory.
enum CheckpointStatus {
    /// `config.json` is missing or the directory doesn't exist — caller may
    /// download.
    Missing,
    /// Every required file is present.
    Complete,
    /// `config.json` is present but one or more shards listed by
    /// `model.safetensors.index.json` are missing on disk. Caller must surface
    /// this; downloading would silently fail at load time.
    Partial { missing: Vec<String> },
}

fn checkpoint_status(dir: &Path) -> CheckpointStatus {
    if !dir.join("config.json").exists() {
        return CheckpointStatus::Missing;
    }
    if dir.join("model.safetensors").exists() {
        return CheckpointStatus::Complete;
    }
    let index_path = dir.join("model.safetensors.index.json");
    let Ok(json) = std::fs::read_to_string(&index_path) else {
        return CheckpointStatus::Missing;
    };
    let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&json) else {
        return CheckpointStatus::Missing;
    };
    let Some(weight_map) = parsed.get("weight_map").and_then(|v| v.as_object()) else {
        return CheckpointStatus::Missing;
    };
    let mut shards: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for v in weight_map.values() {
        if let Some(s) = v.as_str() {
            shards.insert(s);
        }
    }
    let missing: Vec<String> = shards
        .iter()
        .filter(|s| !dir.join(s).exists())
        .map(|s| (*s).to_string())
        .collect();
    if missing.is_empty() {
        CheckpointStatus::Complete
    } else {
        CheckpointStatus::Partial { missing }
    }
}

/// Root of the bench checkpoint cache. Resolved in order:
///
/// 1. `$MLX_LM_BENCH_CACHE` — explicit override (point at any pre-populated dir).
/// 2. `$XDG_CACHE_HOME/mlx-rs-bench`.
/// 3. `$HOME/.cache/mlx-rs-bench`.
/// 4. `.mlx-rs-bench-cache` (CWD, last-resort fallback).
///
/// Each repo is materialised at `<root>/<repo_id>/` via `hf download
/// --local-dir`, i.e. a flat mirror rather than HF's hash-addressed
/// `models--<org>--<name>/snapshots/<sha>/` layout. This is deliberate:
/// `mlx_lm::models::*::load_*_model` take a plain directory path and the
/// flat layout works without snapshot resolution. The trade-off is that the
/// bench cache does not dedupe against `~/.cache/huggingface/hub/`; set
/// `MLX_LM_BENCH_CACHE` if you already have these checkpoints in your HF
/// cache (or anywhere else) and want to point the bench at them.
fn bench_cache_root() -> PathBuf {
    if let Ok(override_dir) = std::env::var("MLX_LM_BENCH_CACHE") {
        return PathBuf::from(override_dir);
    }
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(xdg).join("mlx-rs-bench");
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".cache").join("mlx-rs-bench");
    }
    PathBuf::from(".mlx-rs-bench-cache")
}

fn synthetic_prompt(len: usize, base_id: i32) -> Array {
    let ids: Vec<i32> = (0..len as i32).map(|i| base_id + (i % 100)).collect();
    Array::from_slice(&ids, &[ids.len() as i32]).index(NewAxis)
}

fn maybe_bench_qwen3(c: &mut Criterion, label: &str, repo_id: &str) {
    let Some(dir) = ensure_model(repo_id) else {
        return;
    };
    let mut model = match load_qwen3_model(&dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("skipping qwen3 {label}: load failed: {e:?}");
            return;
        }
    };

    let short = synthetic_prompt(SHORT_PROMPT_LEN, 1000);
    let long = synthetic_prompt(LONG_PROMPT_LEN, 1000);
    run_qwen3_warmup(&mut model, &short);
    bench_qwen3_group(
        c,
        &format!("qwen3_decode_{label}"),
        &mut model,
        &short,
        &long,
    );
}

fn run_qwen3_warmup(model: &mut Qwen3Model, prompt: &Array) {
    let mut cache: Vec<Option<ConcatKeyValueCache>> = Vec::new();
    let mut tokens = Vec::new();
    let gen = Qwen3Generate::<ConcatKeyValueCache>::new(model, &mut cache, 0.0, prompt);
    for (tok, n) in gen.zip(0..WARMUP_TOKENS) {
        tokens.push(tok.unwrap());
        if n == 0 {
            eval(&tokens).unwrap();
        }
    }
    eval(&tokens).unwrap();
}

fn bench_qwen3_group(
    c: &mut Criterion,
    name: &str,
    model: &mut Qwen3Model,
    short: &Array,
    long: &Array,
) {
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Elements(DECODE_TOKENS as u64));
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));

    let run = |model: &mut Qwen3Model, prompt: &Array| {
        let mut cache: Vec<Option<ConcatKeyValueCache>> = Vec::new();
        let mut tokens = Vec::with_capacity(DECODE_TOKENS as usize);
        let gen = Qwen3Generate::<ConcatKeyValueCache>::new(model, &mut cache, 0.0, prompt);
        for (tok, n) in gen.zip(0..DECODE_TOKENS) {
            tokens.push(tok.unwrap());
            if n == 0 {
                eval(&tokens).unwrap();
            }
        }
        eval(&tokens).unwrap();
    };

    group.bench_function(BenchmarkId::new("short", DECODE_TOKENS), |b| {
        b.iter(|| run(model, short));
    });
    group.bench_function(
        BenchmarkId::new("long_prompt", LONG_PROMPT_LEN as i32),
        |b| {
            b.iter(|| run(model, long));
        },
    );
    group.finish();
}

fn maybe_bench_llama(c: &mut Criterion, label: &str, repo_id: &str) {
    let Some(dir) = ensure_model(repo_id) else {
        return;
    };
    let mut model = match load_llama_model(&dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("skipping llama {label}: load failed: {e:?}");
            return;
        }
    };

    let short = synthetic_prompt(SHORT_PROMPT_LEN, 1000);
    let long = synthetic_prompt(LONG_PROMPT_LEN, 1000);

    let mut warm_cache: Vec<Option<ConcatKeyValueCache>> = Vec::new();
    let mut warm_tokens = Vec::new();
    let gen = LlamaGenerate::<ConcatKeyValueCache>::new(&mut model, &mut warm_cache, 0.0, &short);
    for (tok, n) in gen.zip(0..WARMUP_TOKENS) {
        warm_tokens.push(tok.unwrap());
        if n == 0 {
            eval(&warm_tokens).unwrap();
        }
    }
    eval(&warm_tokens).unwrap();

    let mut group = c.benchmark_group(format!("llama_decode_{label}"));
    group.throughput(Throughput::Elements(DECODE_TOKENS as u64));
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));

    let run = |model: &mut LlamaModel, prompt: &Array| {
        let mut cache: Vec<Option<ConcatKeyValueCache>> = Vec::new();
        let mut tokens = Vec::with_capacity(DECODE_TOKENS as usize);
        let gen = LlamaGenerate::<ConcatKeyValueCache>::new(model, &mut cache, 0.0, prompt);
        for (tok, n) in gen.zip(0..DECODE_TOKENS) {
            tokens.push(tok.unwrap());
            if n == 0 {
                eval(&tokens).unwrap();
            }
        }
        eval(&tokens).unwrap();
    };

    group.bench_function(BenchmarkId::new("short", DECODE_TOKENS), |b| {
        b.iter(|| run(&mut model, &short));
    });
    group.bench_function(
        BenchmarkId::new("long_prompt", LONG_PROMPT_LEN as i32),
        |b| {
            b.iter(|| run(&mut model, &long));
        },
    );
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    eprintln!("lm_decode cache root: {}", bench_cache_root().display());

    // Qwen3 plain decoder — small (0.6B) + large (1.7B), bf16.
    maybe_bench_qwen3(c, "small_bf16", "mlx-community/Qwen3-0.6B-bf16");
    maybe_bench_qwen3(c, "large_bf16", "mlx-community/Qwen3-1.7B-bf16");

    // Llama 3.2 — small (1B) + large (3B), bf16.
    maybe_bench_llama(c, "small_bf16", "mlx-community/Llama-3.2-1B-Instruct-bf16");
    maybe_bench_llama(c, "large_bf16", "mlx-community/Llama-3.2-3B-Instruct-bf16");
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
