//! Decode-throughput bench for mlx-lm text decoders.
//!
//! Covers all the decoder families:
//!   - Qwen3 plain decoder (`mlx_lm::models::qwen3`)
//!   - Llama 3.2 (`mlx_lm::models::llama`)
//!   - Qwen3.5 hybrid SSM + attention (`mlx_lm::models::qwen3_5`)
//!
//! Each variant is benched at a short (13-token) and long (1024-token)
//! prompt, decoding `DECODE_TOKENS` tokens. The long-prompt case stresses
//! the KV cache and the attention path.

#![allow(clippy::unwrap_used, reason = "bench harness")]
#![allow(clippy::print_stdout, reason = "bench output")]
#![allow(clippy::print_stderr, reason = "bench output")]
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
use mlx_lm::cache::{KVCache, QuantizedKVCache};
use mlx_lm::models::{
    gemma4::{
        load_gemma4_model_sanitized, make_gemma4_caches, Gemma4Config, Gemma4LayerCache,
        Generate as Gemma4Generate, Model as Gemma4Model,
    },
    llama::{load_llama_model, Generate as LlamaGenerate, Model as LlamaModel},
    qwen3::{load_qwen3_model, Generate as Qwen3Generate, Model as Qwen3Model},
    qwen3_5::{
        config::ModelConfig as Qwen3_5Config,
        generation::{Generate as Qwen3_5Generate, SamplingParams, StopCriteria},
        layer::LanguageModel as Qwen3_5LanguageModel,
        weights::load_language_model,
    },
    qwen3_5_moe::{load_qwen3_5_moe_model, Qwen35MoeBlock, Qwen35MoeLanguageModel},
};
use mlx_rs::{
    ops::indexing::{IndexOp, NewAxis},
    transforms::eval,
    Array,
};

const DECODE_TOKENS: i32 = 100;
const LONG_PROMPT_LEN: usize = 1024;
const VERY_LONG_PROMPT_LEN: usize = 8192;
const DECODE_ONLY_STEPS: i32 = 50;
const SHORT_PROMPT_LEN: usize = 13;
const WARMUP_TOKENS: i32 = 4;
const SAMPLE_SIZE: usize = 10;
const MEASUREMENT_SECS: u64 = 20;

/// Emit `[mlx_mem] tag active=<MB> cache=<MB> peak=<MB>` to stderr so
/// downstream tools (e.g. `bench_with_temp`) can overlay MLX allocator
/// state on the bench-wall-clock timeline.
fn log_mlx_mem(tag: &str) {
    let active = mlx_rs::memory::active_memory();
    let cache = mlx_rs::memory::cache_memory();
    let peak = mlx_rs::memory::peak_memory();
    eprintln!(
        "[mlx_mem] {tag} active_mb={:.1} cache_mb={:.1} peak_mb={:.1}",
        active as f64 / 1e6,
        cache as f64 / 1e6,
        peak as f64 / 1e6,
    );
}

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
        .map(|s| (*s).to_owned())
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
    let mut cache: Vec<Option<KVCache>> = Vec::new();
    let mut tokens = Vec::new();
    let iter = Qwen3Generate::<KVCache>::new(model, &mut cache, 0.0, prompt);
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
    let mut cache: Vec<Option<KVCache>> = Vec::new();
    let mut iter = Qwen3Generate::<KVCache>::new(model, &mut cache, 0.0, prompt);
    let t_start = Instant::now();
    let first = iter.next().expect("at least one token").unwrap();
    eval([&first]).unwrap();
    Instant::now() - t_start
}

/// Time only the post-prefill decode steps. The first token's eval is
/// outside the timing window so prefill cost is excluded.
fn time_qwen3_decode(model: &mut Qwen3Model, prompt: &Array, steps: i32) -> Duration {
    let mut cache: Vec<Option<KVCache>> = Vec::new();
    let mut iter = Qwen3Generate::<KVCache>::new(model, &mut cache, 0.0, prompt);
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
        (
            BenchmarkId::new("prefill_short", SHORT_PROMPT_LEN as i32),
            short,
        ),
        (
            BenchmarkId::new("prefill_long", LONG_PROMPT_LEN as i32),
            long,
        ),
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
    let mut cache: Vec<Option<KVCache>> = Vec::new();
    let mut iter = LlamaGenerate::<KVCache>::new(model, &mut cache, 0.0, prompt);
    let t_start = Instant::now();
    let first = iter.next().expect("at least one token").unwrap();
    eval([&first]).unwrap();
    Instant::now() - t_start
}

fn time_llama_decode(model: &mut LlamaModel, prompt: &Array, steps: i32) -> Duration {
    let mut cache: Vec<Option<KVCache>> = Vec::new();
    let mut iter = LlamaGenerate::<KVCache>::new(model, &mut cache, 0.0, prompt);
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
    let mut warm_cache: Vec<Option<KVCache>> = Vec::new();
    let mut warm_tokens = Vec::new();
    let iter = LlamaGenerate::<KVCache>::new(&mut model, &mut warm_cache, 0.0, &short);
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
        (
            BenchmarkId::new("prefill_short", SHORT_PROMPT_LEN as i32),
            &short,
        ),
        (
            BenchmarkId::new("prefill_long", LONG_PROMPT_LEN as i32),
            &long,
        ),
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

/// Decode-only (prefill excluded) bench for qwen3 + `QuantizedKVCache`.
/// `with_packed_matmul` toggles fused-kernel vs dequant-on-read.
/// `with_rotation_seed` adds Π pre-quantize.
fn maybe_bench_qwen3_kv_decode_only(
    c: &mut Criterion,
    label: &str,
    repo_id: &str,
    kv_bits: i32,
    with_packed_matmul: bool,
    with_rotation_seed: Option<u64>,
    prompt_len: usize,
) {
    if bench_only_skip(&format!("qwen3_kv_decode_only_{label}")) {
        return;
    }
    let Some(dir) = ensure_model(repo_id) else {
        return;
    };
    let mut model = match load_qwen3_model(&dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("skipping qwen3 kv {label}: load failed: {e:?}");
            return;
        }
    };
    let prompt = synthetic_prompt(prompt_len, 1000);
    let num_layers = model.layer_count();
    const HEAD_DIM: i32 = 128;

    let make_cache = |n: usize| -> Vec<Option<QuantizedKVCache>> {
        (0..n)
            .map(|i| {
                let mut c = QuantizedKVCache::with_config(256, 64, kv_bits);
                c = if with_packed_matmul {
                    c.with_fused_kernel()
                } else {
                    c.with_dequant_path()
                };
                if let Some(base_seed) = with_rotation_seed {
                    c = c
                        .with_rotation(HEAD_DIM, base_seed + i as u64)
                        .expect("with_rotation");
                }
                Some(c)
            })
            .collect()
    };

    // Seed the cache once: prefill + WARMUP_TOKENS so the steady-state
    // decode path is hot before any timed iteration.
    let seeded_cache: Vec<Option<QuantizedKVCache>> = {
        let mut cache = make_cache(num_layers);
        let mut tokens = Vec::new();
        let iter = Qwen3Generate::<QuantizedKVCache>::new(&mut model, &mut cache, 0.0, &prompt);
        for (tok, n) in iter.zip(0..WARMUP_TOKENS) {
            tokens.push(tok.unwrap());
            if n == 0 {
                eval(&tokens).unwrap();
            }
        }
        eval(&tokens).unwrap();
        cache
    };

    let mut group = c.benchmark_group(format!("qwen3_kv_decode_only_{label}"));
    group.throughput(Throughput::Elements(DECODE_ONLY_STEPS as u64));
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    group.bench_function(BenchmarkId::new("post_prefill", DECODE_ONLY_STEPS), |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut cache = seeded_cache.clone();
                let seed = Array::from_slice(&[0i32], &[1, 1]);
                let mut iter =
                    Qwen3Generate::<QuantizedKVCache>::new(&mut model, &mut cache, 0.0, &seed);
                // Drain Generate's Prefill state (two forwards on first
                // next()) before starting the timer so we measure only
                // post-prefill decode steps.
                let warm = iter.next().expect("at least one token").unwrap();
                eval([&warm]).unwrap();

                let start = Instant::now();
                let mut tokens = Vec::with_capacity(DECODE_ONLY_STEPS as usize);
                for tok in iter.by_ref().take(DECODE_ONLY_STEPS as usize) {
                    tokens.push(tok.unwrap());
                }
                eval(&tokens).unwrap();
                total += start.elapsed();
                drop(cache);
            }
            total
        });
    });
    group.finish();
}

/// Chat-rendered "Hello" short prompt for the Qwen3.5 tokenizer.
const QWEN3_5_SHORT_PROMPT: &[i32] = &[
    248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
];

fn maybe_bench_qwen3_5(c: &mut Criterion, family: &str, label: &str, repo_id: &str) {
    if bench_only_skip(&format!("{family}_decode_{label}")) {
        return;
    }
    let Some(dir) = ensure_model(repo_id) else {
        return;
    };

    let cfg = match Qwen3_5Config::from_file(dir.join("config.json")) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping {family} {label}: config parse failed: {e:?}");
            return;
        }
    };
    let (mut model, _) = match load_language_model(&cfg, &dir) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("skipping {family} {label}: load failed: {e:?}");
            return;
        }
    };

    let short = Array::from_slice(QWEN3_5_SHORT_PROMPT, &[QWEN3_5_SHORT_PROMPT.len() as i32]);
    let mut long_ids = Vec::with_capacity(LONG_PROMPT_LEN);
    long_ids.extend_from_slice(QWEN3_5_SHORT_PROMPT);
    let filler = QWEN3_5_SHORT_PROMPT[1];
    while long_ids.len() < LONG_PROMPT_LEN {
        long_ids.push(filler);
    }
    long_ids.truncate(LONG_PROMPT_LEN);
    let long = Array::from_slice(&long_ids, &[long_ids.len() as i32]);

    let warm = Qwen3_5Generate::new(
        &mut model,
        &cfg,
        short.clone(),
        StopCriteria {
            max_new_tokens: WARMUP_TOKENS,
            eos_ids: vec![],
        },
        SamplingParams::default(),
    );
    let tokens: Vec<u32> = warm.map(|r| r.unwrap()).collect();
    let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    if !ids.is_empty() {
        let arr = Array::from_slice(&ids, &[ids.len() as i32]);
        eval([&arr]).unwrap();
    }

    let decode_steps = DECODE_TOKENS - 1;
    let mut group = c.benchmark_group(format!("{family}_decode_{label}"));
    group.throughput(Throughput::Elements(decode_steps as u64));
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));

    // Match qwen3/llama/gemma4 methodology: prefill the first token
    // outside the timing window, then time the remaining decode steps
    // only. `Qwen3_5Generate::next()` already forces a per-step eval
    // via `tok.item::<i32>()`, so the lazy graph collapses each step.
    let time_decode = |model: &mut Qwen3_5LanguageModel, prompt: &Array, steps: i32| -> Duration {
        let mut iter = Qwen3_5Generate::new(
            model,
            &cfg,
            prompt.clone(),
            StopCriteria {
                max_new_tokens: steps + 1,
                eos_ids: vec![],
            },
            SamplingParams::default(),
        );
        let _first = iter.next().expect("at least one token").unwrap();
        let t = Instant::now();
        for tok in iter.by_ref().take(steps as usize) {
            let _ = tok.unwrap();
        }
        Instant::now() - t
    };

    for (id, prompt) in [
        (BenchmarkId::new("decode_short", decode_steps), &short),
        (BenchmarkId::new("decode_long", decode_steps), &long),
    ] {
        group.bench_function(id, |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    total += time_decode(&mut model, prompt, decode_steps);
                }
                total
            });
        });
    }
    group.finish();
}

/// Qwen3.6-MoE (35B-A3B) decode bench. Mirrors `maybe_bench_qwen3_5`
/// but loads via `qwen3_5_moe::load_qwen3_5_moe_model` and uses
/// `LanguageModel<Qwen35MoeBlock>` as the generic specialisation.
fn maybe_bench_qwen3_5_moe(c: &mut Criterion, family: &str, label: &str, repo_id: &str) {
    if bench_only_skip(&format!("{family}_decode_{label}")) {
        return;
    }
    let Some(dir) = ensure_model(repo_id) else {
        return;
    };

    let cfg = match Qwen3_5Config::from_file(dir.join("config.json")) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping {family} {label}: config parse failed: {e:?}");
            return;
        }
    };
    let mut model = match load_qwen3_5_moe_model(&dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("skipping {family} {label}: load failed: {e:?}");
            return;
        }
    };

    let short = Array::from_slice(QWEN3_5_SHORT_PROMPT, &[QWEN3_5_SHORT_PROMPT.len() as i32]);
    let mut long_ids = Vec::with_capacity(LONG_PROMPT_LEN);
    long_ids.extend_from_slice(QWEN3_5_SHORT_PROMPT);
    let filler = QWEN3_5_SHORT_PROMPT[1];
    while long_ids.len() < LONG_PROMPT_LEN {
        long_ids.push(filler);
    }
    long_ids.truncate(LONG_PROMPT_LEN);
    let long = Array::from_slice(&long_ids, &[long_ids.len() as i32]);

    let warm = Qwen3_5Generate::<Qwen35MoeBlock>::new(
        &mut model,
        &cfg,
        short.clone(),
        StopCriteria {
            max_new_tokens: WARMUP_TOKENS,
            eos_ids: vec![],
        },
        SamplingParams::default(),
    );
    let tokens: Vec<u32> = warm.map(|r| r.unwrap()).collect();
    let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    if !ids.is_empty() {
        let arr = Array::from_slice(&ids, &[ids.len() as i32]);
        eval([&arr]).unwrap();
    }

    let decode_steps = DECODE_TOKENS - 1;
    let mut group = c.benchmark_group(format!("{family}_decode_{label}"));
    group.throughput(Throughput::Elements(decode_steps as u64));
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));

    let time_decode =
        |model: &mut Qwen35MoeLanguageModel, prompt: &Array, steps: i32| -> Duration {
            let mut iter = Qwen3_5Generate::<Qwen35MoeBlock>::new(
                model,
                &cfg,
                prompt.clone(),
                StopCriteria {
                    max_new_tokens: steps + 1,
                    eos_ids: vec![],
                },
                SamplingParams::default(),
            );
            let _first = iter.next().expect("at least one token").unwrap();
            let t = Instant::now();
            for tok in iter.by_ref().take(steps as usize) {
                let _ = tok.unwrap();
            }
            Instant::now() - t
        };

    for (id, prompt) in [
        (BenchmarkId::new("decode_short", decode_steps), &short),
        (BenchmarkId::new("decode_long", decode_steps), &long),
    ] {
        group.bench_function(id, |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    total += time_decode(&mut model, prompt, decode_steps);
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

fn maybe_bench_gemma4(c: &mut Criterion, label: &str, repo_id: &str) {
    if bench_only_skip(&format!("gemma4_decode_{label}")) {
        return;
    }
    let Some(dir) = ensure_model(repo_id) else {
        return;
    };
    let cfg = match Gemma4Config::from_file(dir.join("config.json")) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping gemma4 {label}: config parse failed: {e:?}");
            return;
        }
    };
    let mut model = match load_gemma4_model_sanitized(&dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("skipping gemma4 {label}: load failed: {e:?}");
            return;
        }
    };
    log_mlx_mem(&format!("gemma4_{label}/loaded"));

    let short = synthetic_prompt(SHORT_PROMPT_LEN, 1000);
    let long = synthetic_prompt(LONG_PROMPT_LEN, 1000);
    let decode_steps = DECODE_TOKENS - 1;

    let time_prefill = |model: &mut Gemma4Model, prompt: &Array| -> Duration {
        let mut cache = make_gemma4_caches(&cfg);
        let mut iter = Gemma4Generate::<Gemma4LayerCache>::new(model, &mut cache, 0.0, prompt);
        let t = Instant::now();
        let first = iter.next().expect("token").unwrap();
        eval([&first]).unwrap();
        Instant::now() - t
    };
    let time_decode = |model: &mut Gemma4Model, prompt: &Array, steps: i32| -> Duration {
        let mut cache = make_gemma4_caches(&cfg);
        let mut iter = Gemma4Generate::<Gemma4LayerCache>::new(model, &mut cache, 0.0, prompt);
        let first = iter.next().expect("token").unwrap();
        eval([&first]).unwrap();
        // Eval per step so the lazy graph collapses immediately —
        // queuing all `steps` tokens into a Vec then `eval`-ing once
        // materialises every per-layer cache scatter intermediate at
        // peak, blowing up to ~tens of GB on long contexts and forcing
        // the OS into swap (poisoning the timing).
        let t = Instant::now();
        for tok in iter.by_ref().take(steps as usize) {
            let tok = tok.unwrap();
            eval([&tok]).unwrap();
        }
        Instant::now() - t
    };

    let mut group = c.benchmark_group(format!("gemma4_decode_{label}"));
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));

    for (cell_name, id, prompt) in [
        (
            "prefill_short",
            BenchmarkId::new("prefill_short", SHORT_PROMPT_LEN as i32),
            &short,
        ),
        (
            "prefill_long",
            BenchmarkId::new("prefill_long", LONG_PROMPT_LEN as i32),
            &long,
        ),
    ] {
        let prompt_len = prompt.shape().last().copied().unwrap_or(0) as u64;
        group.throughput(Throughput::Elements(prompt_len));
        let tag = format!("gemma4_{label}/{cell_name}");
        group.bench_function(id, |b| {
            log_mlx_mem(&format!("{tag}/start"));
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    total += time_prefill(&mut model, prompt);
                }
                total
            });
            log_mlx_mem(&format!("{tag}/end"));
        });
    }

    group.throughput(Throughput::Elements(decode_steps as u64));
    for (cell_name, id, prompt) in [
        (
            "decode_short",
            BenchmarkId::new("decode_short", decode_steps),
            &short,
        ),
        (
            "decode_long",
            BenchmarkId::new("decode_long", decode_steps),
            &long,
        ),
    ] {
        let tag = format!("gemma4_{label}/{cell_name}");
        group.bench_function(id, |b| {
            log_mlx_mem(&format!("{tag}/start"));
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    total += time_decode(&mut model, prompt, decode_steps);
                }
                total
            });
            log_mlx_mem(&format!("{tag}/end"));
        });
    }
    group.finish();
}

fn bench_set() -> BenchSet {
    match std::env::var("MLX_LM_BENCH_SET").as_deref() {
        Ok("full" | "all") => BenchSet::Full,
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

    // Qwen3.5 hybrid SSM + attention.
    maybe_bench_qwen3_5(c, "qwen3_5", "4b_q8", "mlx-community/Qwen3.5-4B-MLX-8bit");
    maybe_bench_qwen3_5(c, "qwen3_5", "9b_q8", "mlx-community/Qwen3.5-9B-8bit");
    maybe_bench_qwen3_5(c, "qwen3_6", "27b_q4", "mlx-community/Qwen3.6-27B-4bit");
    maybe_bench_qwen3_5(c, "qwen3_6", "27b_q8", "mlx-community/Qwen3.6-27B-8bit");
    maybe_bench_qwen3_5_moe(
        c,
        "qwen3_6_moe",
        "35b_a3b_q4",
        "mlx-community/Qwen3.6-35B-A3B-4bit",
    );
    maybe_bench_qwen3_5_moe(
        c,
        "qwen3_6_moe",
        "35b_a3b_q8",
        "lmstudio-community/Qwen3.6-35B-A3B-MLX-8bit",
    );

    // Gemma 4 (dense + MoE; per-layer-input gating).
    maybe_bench_gemma4(c, "e2b_it_q8", "mlx-community/gemma-4-e2b-it-8bit");
    maybe_bench_gemma4(c, "e4b_it_q8", "mlx-community/gemma-4-e4b-it-8bit");
    maybe_bench_gemma4(c, "26b_a4b_it_q8", "mlx-community/gemma-4-26b-a4b-it-8bit");
    maybe_bench_gemma4(c, "31b_it_q8", "mlx-community/gemma-4-31b-it-8bit");

    if set == BenchSet::Full {
        maybe_bench_gemma4(c, "26b_a4b_it_q4", "mlx-community/gemma-4-26b-a4b-it-4bit");
    }

    // KV-quant decode-only: dequant-on-read vs packed-matmul. T=1024.
    maybe_bench_qwen3_kv_decode_only(
        c,
        "large_q4_kv8_dequant_t1024",
        "mlx-community/Qwen3-1.7B-4bit",
        8,
        false,
        None,
        LONG_PROMPT_LEN,
    );
    maybe_bench_qwen3_kv_decode_only(
        c,
        "large_q4_kv8_packed_matmul_t1024",
        "mlx-community/Qwen3-1.7B-4bit",
        8,
        true,
        None,
        LONG_PROMPT_LEN,
    );

    if set == BenchSet::Full {
        maybe_bench_qwen3(c, "small_bf16", "mlx-community/Qwen3-0.6B-bf16");
        maybe_bench_qwen3(c, "small_q8", "mlx-community/Qwen3-0.6B-8bit");
        maybe_bench_qwen3(c, "small_q4", "mlx-community/Qwen3-0.6B-4bit");
        maybe_bench_llama(c, "large_bf16", "mlx-community/Llama-3.2-3B-Instruct-bf16");
        maybe_bench_llama(c, "large_q8", "mlx-community/Llama-3.2-3B-Instruct-8bit");
        maybe_bench_llama(c, "large_q4", "mlx-community/Llama-3.2-3B-Instruct-4bit");

        // Long-context (T=8192) bandwidth-bound regime.
        maybe_bench_qwen3_kv_decode_only(
            c,
            "large_q4_kv8_dequant_t8192",
            "mlx-community/Qwen3-1.7B-4bit",
            8,
            false,
            None,
            VERY_LONG_PROMPT_LEN,
        );
        maybe_bench_qwen3_kv_decode_only(
            c,
            "large_q4_kv8_packed_matmul_t8192",
            "mlx-community/Qwen3-1.7B-4bit",
            8,
            true,
            None,
            VERY_LONG_PROMPT_LEN,
        );
    }
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
