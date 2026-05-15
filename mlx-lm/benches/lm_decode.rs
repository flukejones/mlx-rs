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
use mlx_lm::cache::{ConcatKeyValueCache, QuantizedKVCache};
use mlx_lm::models::{
    llama::{load_llama_model, Generate as LlamaGenerate, Model as LlamaModel},
    qwen3::{load_qwen3_model, Generate as Qwen3Generate, Model as Qwen3Model},
    qwen3_5::{
        config::ModelConfig as Qwen3_5Config,
        generation::{Generate as Qwen3_5Generate, SamplingParams, StopCriteria},
        image_processor::Qwen35ImageProcessor,
        layer::LanguageModel as Qwen3_5LanguageModel,
        weights::{load_full_model, load_language_model},
    },
};
use mlx_rs::{
    ops::indexing::{IndexOp, NewAxis},
    transforms::eval,
    Array, Dtype,
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

/// Bench a qwen3 cell with a `QuantizedKVCache` of the given `bits`
/// (long_prompt only — KV-quant only moves the needle at long context).
fn maybe_bench_qwen3_kv_quant(c: &mut Criterion, label: &str, repo_id: &str, kv_bits: i32) {
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
    let long = synthetic_prompt(LONG_PROMPT_LEN, 1000);

    let make_cache = |n: usize| -> Vec<Option<QuantizedKVCache>> {
        (0..n)
            .map(|_| Some(QuantizedKVCache::with_config(256, 64, kv_bits)))
            .collect()
    };

    let num_layers = model.layer_count();
    {
        let mut cache = make_cache(num_layers);
        let mut tokens = Vec::new();
        let gen = Qwen3Generate::<QuantizedKVCache>::new(&mut model, &mut cache, 0.0, &long);
        for (tok, n) in gen.zip(0..WARMUP_TOKENS) {
            tokens.push(tok.unwrap());
            if n == 0 {
                eval(&tokens).unwrap();
            }
        }
        eval(&tokens).unwrap();
    }

    let mut group = c.benchmark_group(format!("qwen3_decode_{label}"));
    group.throughput(Throughput::Elements(DECODE_TOKENS as u64));
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    group.bench_function(
        BenchmarkId::new("long_prompt", LONG_PROMPT_LEN as i32),
        |b| {
            b.iter(|| {
                let mut cache = make_cache(num_layers);
                let mut tokens = Vec::with_capacity(DECODE_TOKENS as usize);
                let gen =
                    Qwen3Generate::<QuantizedKVCache>::new(&mut model, &mut cache, 0.0, &long);
                for (tok, n) in gen.zip(0..DECODE_TOKENS) {
                    tokens.push(tok.unwrap());
                    if n == 0 {
                        eval(&tokens).unwrap();
                    }
                }
                eval(&tokens).unwrap();
            });
        },
    );
    group.finish();
}

/// Mirror of [`maybe_bench_qwen3_kv_quant`] for the Llama path.
fn maybe_bench_llama_kv_quant(c: &mut Criterion, label: &str, repo_id: &str, kv_bits: i32) {
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
    let long = synthetic_prompt(LONG_PROMPT_LEN, 1000);

    let make_cache = |n: usize| -> Vec<Option<QuantizedKVCache>> {
        (0..n)
            .map(|_| Some(QuantizedKVCache::with_config(256, 64, kv_bits)))
            .collect()
    };

    let num_layers = model.layer_count();
    {
        let mut cache = make_cache(num_layers);
        let mut tokens = Vec::new();
        let gen = LlamaGenerate::<QuantizedKVCache>::new(&mut model, &mut cache, 0.0, &long);
        for (tok, n) in gen.zip(0..WARMUP_TOKENS) {
            tokens.push(tok.unwrap());
            if n == 0 {
                eval(&tokens).unwrap();
            }
        }
        eval(&tokens).unwrap();
    }

    let mut group = c.benchmark_group(format!("llama_decode_{label}"));
    group.throughput(Throughput::Elements(DECODE_TOKENS as u64));
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    group.bench_function(
        BenchmarkId::new("long_prompt", LONG_PROMPT_LEN as i32),
        |b| {
            b.iter(|| {
                let mut cache = make_cache(num_layers);
                let mut tokens = Vec::with_capacity(DECODE_TOKENS as usize);
                let gen =
                    LlamaGenerate::<QuantizedKVCache>::new(&mut model, &mut cache, 0.0, &long);
                for (tok, n) in gen.zip(0..DECODE_TOKENS) {
                    tokens.push(tok.unwrap());
                    if n == 0 {
                        eval(&tokens).unwrap();
                    }
                }
                eval(&tokens).unwrap();
            });
        },
    );
    group.finish();
}

/// Chat-rendered "Hello" short prompt for the Qwen3.5 tokenizer.
const QWEN3_5_SHORT_PROMPT: &[i32] = &[
    248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
];

fn maybe_bench_qwen3_5(c: &mut Criterion, label: &str, repo_id: &str) {
    let Some(dir) = ensure_model(repo_id) else {
        return;
    };

    let cfg = match Qwen3_5Config::from_file(dir.join("config.json")) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping qwen3_5 {label}: config parse failed: {e:?}");
            return;
        }
    };
    let (mut model, _) = match load_language_model(&cfg, &dir) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("skipping qwen3_5 {label}: load failed: {e:?}");
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

    let mut group = c.benchmark_group(format!("qwen3_5_decode_{label}"));
    group.throughput(Throughput::Elements(DECODE_TOKENS as u64));
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));

    let run = |model: &mut Qwen3_5LanguageModel, prompt: &Array| {
        let gen = Qwen3_5Generate::new(
            model,
            &cfg,
            prompt.clone(),
            StopCriteria {
                max_new_tokens: DECODE_TOKENS,
                eos_ids: vec![],
            },
            SamplingParams::default(),
        );
        let tokens: Vec<u32> = gen.map(|r| r.unwrap()).collect();
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let arr = Array::from_slice(&ids, &[ids.len() as i32]);
        eval([&arr]).unwrap();
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

/// Vision-tower (ViT-24 + merger) forward pass on a single image.
///
/// Loads the full multimodal qwen3_5 checkpoint, runs the image processor on
/// the bundled fixture `tests/fixtures/qwen3_5/test_image.png` once, then
/// benches the ViT forward in isolation. Throughput is one image per call.
///
/// The `head_dim = 72` in this ViT is exactly the case that falls outside
/// MLX's fused SDPA set, so this cell is sensitive to the
/// `scaled_dot_product_attention_pad_to_fused` helper.
fn maybe_bench_chandra_vision(c: &mut Criterion, label: &str, repo_id: &str) {
    let Some(dir) = ensure_model(repo_id) else {
        return;
    };

    let cfg = match Qwen3_5Config::from_file(dir.join("config.json")) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping vision {label}: config parse failed: {e:?}");
            return;
        }
    };
    if cfg.vision_config.depth == 0 {
        eprintln!("skipping vision {label}: checkpoint has no vision tower");
        return;
    }
    let (_, mut vision, _) = match load_full_model(&cfg, &dir) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("skipping vision {label}: load failed: {e:?}");
            return;
        }
    };

    let processor = match Qwen35ImageProcessor::from_dir(&dir) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("skipping vision {label}: image processor init failed: {e:?}");
            return;
        }
    };
    let img_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3_5/test_image.png");
    let processed = match processor.preprocess_path(&img_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("skipping vision {label}: preprocess failed: {e:?}");
            return;
        }
    };
    let num_patches = (processed.pixel_values.len() / processed.feature_dim as usize) as i32;
    let pixel_array = Array::from_slice(
        &processed.pixel_values,
        &[num_patches, processed.feature_dim],
    )
    .as_dtype(Dtype::Bfloat16)
    .unwrap();
    let grid_thw = processed.grid_thw;

    // Warm-up: ensure Metal kernels + alloc state are stable.
    let warm = vision.forward(&pixel_array, &[grid_thw]).unwrap();
    eval([&warm]).unwrap();

    let mut group = c.benchmark_group(format!("vision_prefill_{label}"));
    group.throughput(Throughput::Elements(1));
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    group.bench_function(BenchmarkId::new("one_image", num_patches), |b| {
        b.iter(|| {
            let out = vision.forward(&pixel_array, &[grid_thw]).unwrap();
            eval([&out]).unwrap();
        });
    });
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    eprintln!("lm_decode cache root: {}", bench_cache_root().display());

    // Qwen3 plain decoder — small (0.6B) + large (1.7B) × {bf16, q8, q4}.
    maybe_bench_qwen3(c, "small_bf16", "mlx-community/Qwen3-0.6B-bf16");
    maybe_bench_qwen3(c, "small_q8", "mlx-community/Qwen3-0.6B-8bit");
    maybe_bench_qwen3(c, "small_q4", "mlx-community/Qwen3-0.6B-4bit");
    maybe_bench_qwen3(c, "large_bf16", "mlx-community/Qwen3-1.7B-bf16");
    maybe_bench_qwen3(c, "large_q8", "mlx-community/Qwen3-1.7B-8bit");
    maybe_bench_qwen3(c, "large_q4", "mlx-community/Qwen3-1.7B-4bit");

    // Llama 3.2 — small (1B) + large (3B) × {bf16, q8, q4}.
    maybe_bench_llama(c, "small_bf16", "mlx-community/Llama-3.2-1B-Instruct-bf16");
    maybe_bench_llama(c, "small_q8", "mlx-community/Llama-3.2-1B-Instruct-8bit");
    maybe_bench_llama(c, "small_q4", "mlx-community/Llama-3.2-1B-Instruct-4bit");
    maybe_bench_llama(c, "large_bf16", "mlx-community/Llama-3.2-3B-Instruct-bf16");
    maybe_bench_llama(c, "large_q8", "mlx-community/Llama-3.2-3B-Instruct-8bit");
    maybe_bench_llama(c, "large_q4", "mlx-community/Llama-3.2-3B-Instruct-4bit");

    // Qwen3.5 hybrid SSM+attention. The 4B-8bit checkpoint is the smallest
    // public mlx-community Qwen3.5 build and exercises the same code paths as
    // the chandra-ocr-2 text-only model.
    maybe_bench_qwen3_5(c, "4b_q8", "mlx-community/Qwen3.5-4B-MLX-8bit");
    maybe_bench_qwen3_5(c, "9b_q8", "mlx-community/Qwen3.5-9B-8bit");

    // KV-quant cells: largest decoder-only quant base × {KV q8, KV q4},
    // long_prompt only (KV quant only moves the needle at long context).
    // Plus a bf16-base cell to isolate KV-quant effect from weight quant.
    maybe_bench_qwen3_kv_quant(
        c,
        "large_bf16_kv8",
        "mlx-community/Qwen3-1.7B-bf16",
        8,
    );
    maybe_bench_qwen3_kv_quant(
        c,
        "large_q4_kv8",
        "mlx-community/Qwen3-1.7B-4bit",
        8,
    );
    maybe_bench_qwen3_kv_quant(
        c,
        "large_q4_kv4",
        "mlx-community/Qwen3-1.7B-4bit",
        4,
    );
    maybe_bench_llama_kv_quant(
        c,
        "large_q4_kv8",
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        8,
    );
    maybe_bench_llama_kv_quant(
        c,
        "large_q4_kv4",
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        4,
    );

    // Vision-tower prefill on the chandra-ocr-2-8bit-mlx checkpoint (the only
    // public mlx conversion of chandra). Measures `VisionModel::forward` for
    // one image — the hot path #348's `scaled_dot_product_attention_pad_to_fused`
    // is meant to accelerate (head_dim=72 falls outside the fused SDPA set).
    maybe_bench_chandra_vision(c, "chandra_q8", "jwindle47/chandra-ocr-2-8bit-mlx");
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
