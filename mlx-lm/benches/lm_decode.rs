//! Decode-throughput bench for `mlx_lm`.
//!
//! Drives every family through the unified [`mlx_lm::load`] surface.
//! Each cell measures:
//! - **prefill_*** — first call to [`LanguageModel::prepare`] on a
//!   short / long synthetic prompt. Prefill latency, single shot.
//! - **decode_*** — per-token [`LanguageModel::step`] cost averaged
//!   over `DECODE_TOKENS - 1` steps after the prefill primer. Cache
//!   is reset before every `iter_custom` iteration.
//!
//! Both phases sit inside `iter_custom` so prefill cost does not
//! pollute decode measurements. Cells skip silently if the
//! checkpoint isn't on disk and can't be fetched via the `hf` CLI.

#![allow(clippy::unwrap_used, reason = "bench harness")]
#![allow(clippy::print_stdout, reason = "bench output")]
#![allow(clippy::print_stderr, reason = "bench output")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mlx_lm::lm_input::{LMInput, PrepareResult, Text};
use mlx_lm::{load, ModelContext, SamplerState, SamplingParams};
use mlx_rs::{
    ops::indexing::IndexOp,
    transforms::{async_eval, eval},
    Array,
};

const DECODE_TOKENS: i32 = 100;
const SHORT_PROMPT_LEN: usize = 13;
const LONG_PROMPT_LEN: usize = 1024;
const SAMPLE_SIZE: usize = 10;
const MEASUREMENT_SECS: u64 = 20;

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
    let Ok(json) = std::fs::read_to_string(dir.join("model.safetensors.index.json")) else {
        return CheckpointStatus::Missing;
    };
    let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&json) else {
        return CheckpointStatus::Missing;
    };
    let Some(weight_map) = parsed.get("weight_map").and_then(|v| v.as_object()) else {
        return CheckpointStatus::Missing;
    };
    let shards: std::collections::HashSet<&str> =
        weight_map.values().filter_map(|v| v.as_str()).collect();
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

/// Substring filter on the per-cell group prefix; if
/// `MLX_LM_BENCH_ONLY` is set, cells whose `<family>_<label>` prefix
/// does not contain the substring skip even loading the model.
fn bench_only_skip(group_prefix: &str) -> bool {
    match std::env::var("MLX_LM_BENCH_ONLY") {
        Ok(v) if !v.is_empty() => !group_prefix.contains(&v),
        _ => false,
    }
}

/// `[1, len]` int32 prompt of synthetic ids `100..(100+len)`. The
/// model never sees a real tokenizer here — we're measuring forward
/// throughput, not text quality.
fn synthetic_prompt(len: usize) -> Array {
    let ids: Vec<i32> = (0..len as i32).map(|i| 100 + (i % 100)).collect();
    Array::from_slice(&ids, &[1, ids.len() as i32])
}

fn make_lm_input(prompt: &Array) -> LMInput {
    LMInput {
        text: Text {
            tokens: prompt.clone(),
            mask: None,
        },
        image: None,
        audio: None,
        video: None,
    }
}

/// One full prefill (cache reset to drop any prior state) on
/// `prompt`, eval-fenced. Mirrors the chunking that
/// `mlx_lm::generate`'s `run_prefill` does — required for
/// sliding-window adapters (gemma 4) whose `prepare` would otherwise
/// build a `[L, L]` causal mask that the K-capped attention rejects.
fn time_prefill(ctx: &mut ModelContext, prompt: &Array) -> Duration {
    ctx.model.reset();
    let prompt_len = prompt.shape()[1];
    let chunk_size = ctx.model.prefill_chunk_size();
    let t = Instant::now();
    if let Some(window) = chunk_size {
        if prompt_len > window {
            let mut start = 0_i32;
            while prompt_len - start > window {
                let end = start + window;
                let chunk = prompt.index((.., start..end));
                ctx.model.prefill_chunk(&chunk).unwrap();
                start = end;
            }
            let tail = prompt.index((.., start..prompt_len));
            let res = ctx.model.prepare(make_lm_input(&tail)).unwrap();
            match res {
                PrepareResult::Logits(arr) => {
                    eval([&arr]).unwrap();
                }
                PrepareResult::Primed => {}
            }
            return t.elapsed();
        }
    }
    let res = ctx.model.prepare(make_lm_input(prompt)).unwrap();
    match res {
        PrepareResult::Logits(arr) => {
            eval([&arr]).unwrap();
        }
        PrepareResult::Primed => {}
    }
    t.elapsed()
}

/// Decode-only timing: prefill outside the window, then `steps`
/// consecutive `step` calls inside. Cache is reset before each
/// iteration so the long-prompt cell measures the same hot-path each
/// time. Pipelined async_eval mirrors the production
/// `mlx_lm::generate` loop: every step's argmax is submitted to the
/// GPU before the previous step's id is `.item()`-resolved, so the
/// host's `.item()` block overlaps with the next step's compute.
/// Cache-reset + chunked prefill of `prompt`, returning the seed
/// logits the decode loop should sample from. Shared between greedy
/// and sampled decode timers so both pick up the gemma-4 chunked
/// prefill path.
fn prefill_for_decode(ctx: &mut ModelContext, prompt: &Array) -> Array {
    ctx.model.reset();
    let prompt_len = prompt.shape()[1];
    let chunk_size = ctx.model.prefill_chunk_size();
    let tail = if let Some(window) = chunk_size {
        if prompt_len > window {
            let mut start = 0_i32;
            while prompt_len - start > window {
                let end = start + window;
                let chunk = prompt.index((.., start..end));
                ctx.model.prefill_chunk(&chunk).unwrap();
                start = end;
            }
            prompt.index((.., start..prompt_len))
        } else {
            prompt.clone()
        }
    } else {
        prompt.clone()
    };
    match ctx.model.prepare(make_lm_input(&tail)).unwrap() {
        PrepareResult::Logits(arr) => arr,
        PrepareResult::Primed => {
            let seed = Array::from_slice::<i32>(&[0], &[1]);
            ctx.model.step(&seed).unwrap().logits
        }
    }
}

fn time_decode(ctx: &mut ModelContext, prompt: &Array, steps: i32) -> Duration {
    let initial = prefill_for_decode(ctx, prompt);

    // Submit the first sample, eval-fence it so the prefill graph
    // is fully resolved before timing starts. `pending` stays on
    // device — the per-step `step(&pending)` reshapes it via a
    // view, no host materialisation or device upload per token.
    let mut pending = mlx_rs::argmax_axis!(&initial, -1).unwrap();
    eval([&pending]).unwrap();

    let t = Instant::now();
    for _ in 0..steps {
        // Submit step N+1's graph + argmax via async_eval before
        // sync-waiting the prior step. Use vector eval rather than
        // `.item()` — `.item()` reads the int back to host which
        // forces an extra unified-memory coherence barrier on top
        // of the graph eval, ~3 ms per call on M4 Max.
        let logits = ctx.model.step(&pending).unwrap().logits;
        let next = mlx_rs::argmax_axis!(&logits, -1).unwrap();
        async_eval([&next]).unwrap();
        eval([&pending]).unwrap();
        pending = next;
    }
    eval([&pending]).unwrap();
    t.elapsed()
}

/// Decode-only timing through [`SamplerState`] at temp=0.1 + top_p=0.95
/// to exercise the cached scalar-array hot path. Same pipelining
/// shape as [`time_decode`] (async_eval N+1 before syncing on N).
fn time_decode_sampled(ctx: &mut ModelContext, prompt: &Array, steps: i32) -> Duration {
    let initial = prefill_for_decode(ctx, prompt);
    let mut sampler = SamplerState::new(SamplingParams {
        temperature: 0.1,
        top_p: Some(0.95),
    });
    let mut pending = sampler.sample(&initial).unwrap();
    eval([&pending]).unwrap();

    let t = Instant::now();
    for _ in 0..steps {
        let logits = ctx.model.step(&pending).unwrap().logits;
        let next = sampler.sample(&logits).unwrap();
        async_eval([&next]).unwrap();
        eval([&pending]).unwrap();
        pending = next;
    }
    eval([&pending]).unwrap();
    t.elapsed()
}

/// Generic per-family bench: prefill_short, prefill_long,
/// decode_short, decode_long, decode_short_sampled. `label` becomes
/// the criterion group prefix (`<family>_decode_<label>`).
///
/// Model is loaded **inside** this fn and dropped + the mlx-core
/// buffer cache cleared on return, so back-to-back `bench_one()`
/// calls don't keep the prior model's weights or its allocator
/// free-list in RAM. Important when chaining 35B + 27B + 26B
/// cells in one bench run — without the explicit drop + cache
/// clear, peak resident memory would be sum-of-models instead
/// of max-of-models.
fn bench_one(c: &mut Criterion, family: &str, label: &str, repo_id: &str) {
    let group_prefix = format!("{family}_decode_{label}");
    if bench_only_skip(&group_prefix) {
        return;
    }
    let Some(dir) = ensure_model(repo_id) else {
        return;
    };
    eprintln!("loading {repo_id}");
    log_mlx_mem(&format!("{group_prefix}/pre_load"));
    let mut ctx = match load(&dir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping {repo_id}: load failed: {e}");
            return;
        }
    };
    log_mlx_mem(&format!("{group_prefix}/post_load"));

    let short = synthetic_prompt(SHORT_PROMPT_LEN);
    let long = synthetic_prompt(LONG_PROMPT_LEN);
    let decode_steps = DECODE_TOKENS - 1;

    {
        let mut group = c.benchmark_group(group_prefix.clone());
        group.sample_size(SAMPLE_SIZE);
        group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));

        for (label, prompt) in [
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
            group.bench_function(label, |b| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        total += time_prefill(&mut ctx, prompt);
                    }
                    total
                });
            });
        }

        group.throughput(Throughput::Elements(decode_steps as u64));
        for (label, prompt) in [
            (BenchmarkId::new("decode_short", decode_steps), &short),
            (BenchmarkId::new("decode_long", decode_steps), &long),
        ] {
            group.bench_function(label, |b| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        total += time_decode(&mut ctx, prompt, decode_steps);
                    }
                    total
                });
            });
        }

        // Sampled cell: same shape as decode_short but routes through
        // SamplerState (temp=0.1 + top_p=0.95). Measures the cost of
        // the cached scalar arrays + top-p chain that greedy bypasses.
        group.bench_function(
            BenchmarkId::new("decode_short_sampled", decode_steps),
            |b| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        total += time_decode_sampled(&mut ctx, &short, decode_steps);
                    }
                    total
                });
            },
        );
        group.finish();
    }

    // Release the model + unmap mlx-core's buffer cache so peak
    // RAM in a multi-model run is max-of-models, not sum.
    ctx.unload();
    mlx_rs::memory::reset_peak_memory();
    log_mlx_mem(&format!("{group_prefix}/post_unload"));
}

fn bench_decode(c: &mut Criterion) {
    eprintln!("lm_decode cache root: {}", bench_cache_root().display());

    bench_one(c, "qwen3_5", "4b_q8", "mlx-community/Qwen3.5-4B-MLX-8bit");
    bench_one(c, "qwen3_5", "9b_q8", "mlx-community/Qwen3.5-9B-8bit");
    bench_one(c, "qwen3_6", "27b_q4", "mlx-community/Qwen3.6-27B-4bit");
    bench_one(
        c,
        "qwen3_6_moe",
        "35b_a3b_q8",
        "lmstudio-community/Qwen3.6-35B-A3B-MLX-8bit",
    );

    bench_one(
        c,
        "gemma4",
        "e2b_it_q8",
        "mlx-community/gemma-4-e2b-it-8bit",
    );
    bench_one(
        c,
        "gemma4",
        "e4b_it_q8",
        "mlx-community/gemma-4-e4b-it-8bit",
    );
    bench_one(
        c,
        "gemma4",
        "26b_a4b_it_q8",
        "mlx-community/gemma-4-26b-a4b-it-8bit",
    );
    bench_one(c, "gemma4", "31b_it_q4", "mlx-community/gemma-4-31b-it-4bit");
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
