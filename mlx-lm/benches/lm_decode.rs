//! Decode-throughput bench for mlx-lm text decoders.
//!
//! Covers the decoder families present at M1:
//!   - Qwen3 plain decoder (`mlx_lm::models::qwen3`)
//!   - Llama 3.2 (`mlx_lm::models::llama`)
//!
//! Each variant is benched at a short (13-token) and long (1024-token)
//! prompt, decoding `DECODE_TOKENS` tokens. The long-prompt case stresses
//! the KV cache and the attention path.
//!
//! Checkpoints are pulled lazily on first use via the `hf` CLI into the
//! cache directory printed at bench start. Cells skip silently if `hf` is
//! unavailable or the download fails.
//!
//! Run with:
//!
//!     cargo bench -p mlx-lm --bench lm_decode

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

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
/// Set `MLX_LM_BENCH_NO_DOWNLOAD=1` to skip download.
fn ensure_model(repo_id: &str) -> Option<PathBuf> {
    let cache = bench_cache_root().join(repo_id);
    match checkpoint_status(&cache) {
        CheckpointStatus::Complete => return Some(cache),
        CheckpointStatus::Partial { missing } => {
            eprintln!(
                "skipping {repo_id}: partial checkpoint at {} (missing {}: {}).",
                cache.display(),
                missing.len(),
                missing.join(", "),
            );
            return None;
        }
        CheckpointStatus::Missing => {}
    }
    if std::env::var_os("MLX_LM_BENCH_NO_DOWNLOAD").is_some() {
        return None;
    }
    if std::fs::create_dir_all(&cache).is_err() {
        eprintln!("skipping {repo_id}: could not create {}", cache.display());
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

enum CheckpointStatus {
    Missing,
    Complete,
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

/// Root of the bench checkpoint cache, in order:
/// `$MLX_LM_BENCH_CACHE` → `$XDG_CACHE_HOME/mlx-rs-bench` →
/// `$HOME/.cache/mlx-rs-bench` → `.mlx-rs-bench-cache`.
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

/// Substring filter on the per-cell group prefix. If
/// `MLX_LM_BENCH_ONLY` is set, any `maybe_bench_*` whose
/// `<family>_<label>` group prefix does not contain the substring will
/// skip even its model load — useful to keep cap-sweep iteration cost
/// down (otherwise every cargo-bench invocation reloads every model).
fn bench_only_skip(group_prefix: &str) -> bool {
    match std::env::var("MLX_LM_BENCH_ONLY") {
        Ok(v) if !v.is_empty() => !group_prefix.contains(&v),
        _ => false,
    }
}

fn maybe_bench_qwen3(c: &mut Criterion, label: &str, repo_id: &str) {
    if bench_only_skip(&format!("qwen3_decode_{label}")) {
        return;
    }
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
    let iter = Qwen3Generate::<ConcatKeyValueCache>::new(model, &mut cache, 0.0, prompt);
    for (tok, n) in (iter).zip(0..WARMUP_TOKENS) {
        tokens.push(tok.unwrap());
        if n == 0 {
            eval(&tokens).unwrap();
        }
    }
    eval(&tokens).unwrap();
}

/// Time only the prompt prefill: one `Generate::next()` call eval'd. The
/// returned token is discarded.
fn time_qwen3_prefill(model: &mut Qwen3Model, prompt: &Array) -> Duration {
    let mut cache: Vec<Option<ConcatKeyValueCache>> = Vec::new();
    let mut iter = Qwen3Generate::<ConcatKeyValueCache>::new(model, &mut cache, 0.0, prompt);
    let t_start = Instant::now();
    let first = iter.next().expect("at least one token").unwrap();
    eval([&first]).unwrap();
    Instant::now() - t_start
}

/// Time only the post-prefill decode steps. The first token's eval is
/// outside the timing window so prefill cost is excluded.
fn time_qwen3_decode(model: &mut Qwen3Model, prompt: &Array, steps: i32) -> Duration {
    let mut cache: Vec<Option<ConcatKeyValueCache>> = Vec::new();
    let mut iter = Qwen3Generate::<ConcatKeyValueCache>::new(model, &mut cache, 0.0, prompt);
    let first = iter.next().expect("at least one token").unwrap();
    eval([&first]).unwrap();
    let mut tokens = Vec::with_capacity(steps as usize);
    let t_start = Instant::now();
    for tok in iter.by_ref().take(steps as usize) {
        tokens.push(tok.unwrap());
    }
    eval(&tokens).unwrap();
    Instant::now() - t_start
}

fn bench_qwen3_group(
    c: &mut Criterion,
    name: &str,
    model: &mut Qwen3Model,
    short: &Array,
    long: &Array,
) {
    let decode_steps = DECODE_TOKENS - 1;
    let mut group = c.benchmark_group(name);
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));

    for (label, prompt) in [
        (BenchmarkId::new("prefill_short", SHORT_PROMPT_LEN as i32), short),
        (BenchmarkId::new("prefill_long", LONG_PROMPT_LEN as i32), long),
    ] {
        let prompt_len = prompt.shape().last().copied().unwrap_or(0) as u64;
        group.throughput(Throughput::Elements(prompt_len));
        group.bench_function(label, |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    total += time_qwen3_prefill(model, prompt);
                }
                total
            });
        });
    }

    group.throughput(Throughput::Elements(decode_steps as u64));
    for (label, prompt) in [
        (BenchmarkId::new("decode_short", decode_steps), short),
        (BenchmarkId::new("decode_long", decode_steps), long),
    ] {
        group.bench_function(label, |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    total += time_qwen3_decode(model, prompt, decode_steps);
                }
                total
            });
        });
    }
    group.finish();
}

fn time_llama_prefill(model: &mut LlamaModel, prompt: &Array) -> Duration {
    let mut cache: Vec<Option<ConcatKeyValueCache>> = Vec::new();
    let mut iter = LlamaGenerate::<ConcatKeyValueCache>::new(model, &mut cache, 0.0, prompt);
    let t_start = Instant::now();
    let first = iter.next().expect("at least one token").unwrap();
    eval([&first]).unwrap();
    Instant::now() - t_start
}

fn time_llama_decode(model: &mut LlamaModel, prompt: &Array, steps: i32) -> Duration {
    let mut cache: Vec<Option<ConcatKeyValueCache>> = Vec::new();
    let mut iter = LlamaGenerate::<ConcatKeyValueCache>::new(model, &mut cache, 0.0, prompt);
    let first = iter.next().expect("at least one token").unwrap();
    eval([&first]).unwrap();
    let mut tokens = Vec::with_capacity(steps as usize);
    let t_start = Instant::now();
    for tok in iter.by_ref().take(steps as usize) {
        tokens.push(tok.unwrap());
    }
    eval(&tokens).unwrap();
    Instant::now() - t_start
}

fn maybe_bench_llama(c: &mut Criterion, label: &str, repo_id: &str) {
    if bench_only_skip(&format!("llama_decode_{label}")) {
        return;
    }
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

    // Warmup: drive WARMUP_TOKENS through the short prompt so any one-shot
    // init costs (kernel compile, lazy weight materialise) are amortised.
    let mut warm_cache: Vec<Option<ConcatKeyValueCache>> = Vec::new();
    let mut warm_tokens = Vec::new();
    let iter = LlamaGenerate::<ConcatKeyValueCache>::new(&mut model, &mut warm_cache, 0.0, &short);
    for (tok, n) in (iter).zip(0..WARMUP_TOKENS) {
        warm_tokens.push(tok.unwrap());
        if n == 0 {
            eval(&warm_tokens).unwrap();
        }
    }
    eval(&warm_tokens).unwrap();

    let decode_steps = DECODE_TOKENS - 1;
    let mut group = c.benchmark_group(format!("llama_decode_{label}"));
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));

    for (id, prompt) in [
        (BenchmarkId::new("prefill_short", SHORT_PROMPT_LEN as i32), &short),
        (BenchmarkId::new("prefill_long", LONG_PROMPT_LEN as i32), &long),
    ] {
        let prompt_len = prompt.shape().last().copied().unwrap_or(0) as u64;
        group.throughput(Throughput::Elements(prompt_len));
        group.bench_function(id, |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    total += time_llama_prefill(&mut model, prompt);
                }
                total
            });
        });
    }

    group.throughput(Throughput::Elements(decode_steps as u64));
    for (id, prompt) in [
        (BenchmarkId::new("decode_short", decode_steps), &short),
        (BenchmarkId::new("decode_long", decode_steps), &long),
    ] {
        group.bench_function(id, |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    total += time_llama_decode(&mut model, prompt, decode_steps);
                }
                total
            });
        });
    }
    group.finish();
}

/// Which cells to register. Trimmed = end-to-end decode only; Full = all
/// llama + qwen3 size × precision combinations. Default trimmed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchSet {
    Trimmed,
    Full,
}

fn bench_set() -> BenchSet {
    match std::env::var("MLX_LM_BENCH_SET").as_deref() {
        Ok("full") | Ok("all") => BenchSet::Full,
        _ => BenchSet::Trimmed,
    }
}

fn bench_decode(c: &mut Criterion) {
    eprintln!("lm_decode cache root: {}", bench_cache_root().display());
    let set = bench_set();
    eprintln!("lm_decode bench set: {set:?} (override with MLX_LM_BENCH_SET={{trimmed,full}})");

    maybe_bench_qwen3(c, "large_bf16", "mlx-community/Qwen3-1.7B-bf16");
    maybe_bench_qwen3(c, "large_q8", "mlx-community/Qwen3-1.7B-8bit");
    maybe_bench_qwen3(c, "large_q4", "mlx-community/Qwen3-1.7B-4bit");
    maybe_bench_llama(c, "small_bf16", "mlx-community/Llama-3.2-1B-Instruct-bf16");
    maybe_bench_llama(c, "small_q8", "mlx-community/Llama-3.2-1B-Instruct-8bit");
    maybe_bench_llama(c, "small_q4", "mlx-community/Llama-3.2-1B-Instruct-4bit");

    if set == BenchSet::Full {
        maybe_bench_qwen3(c, "small_bf16", "mlx-community/Qwen3-0.6B-bf16");
        maybe_bench_qwen3(c, "small_q8", "mlx-community/Qwen3-0.6B-8bit");
        maybe_bench_qwen3(c, "small_q4", "mlx-community/Qwen3-0.6B-4bit");
        maybe_bench_llama(c, "large_bf16", "mlx-community/Llama-3.2-3B-Instruct-bf16");
        maybe_bench_llama(c, "large_q8", "mlx-community/Llama-3.2-3B-Instruct-8bit");
        maybe_bench_llama(c, "large_q4", "mlx-community/Llama-3.2-3B-Instruct-4bit");
    }
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
