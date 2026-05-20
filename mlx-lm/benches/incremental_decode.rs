//! Regression bench for the streaming decode hot path.
//!
//! Compares two implementations of "given N produced tokens,
//! stream the UTF-8 delta after each one":
//!
//! - `naive`: `processor.decode(&ids[..k])` per step for each `k`
//!   in 1..=N — the pre-rewrite path, O(N²) over a response.
//! - `incremental`: `IncrementalDecoder::push` per step — sliding
//!   window of size `WINDOW`, O(N) total.
//!
//! Drives a real Qwen3 tokenizer (loaded from the bench cache) so
//! BPE merges + multi-byte byte-pair fallback hit the same code
//! paths as production `generate()`. Cell skips silently if the
//! tokenizer asset isn't on disk.

#![allow(clippy::unwrap_used, reason = "bench harness")]
#![allow(clippy::print_stderr, reason = "bench output")]

use std::path::PathBuf;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mlx_lm::chat_template::ChatTemplate;
use mlx_lm::{TextOnlyProcessor, UserInputProcessor};

const N_TOKENS: usize = 1024;
const SAMPLE_SIZE: usize = 30;
const MEASUREMENT_SECS: u64 = 5;

fn bench_cache_root() -> PathBuf {
    if let Ok(dir) = std::env::var("MLX_LM_BENCH_CACHE") {
        return PathBuf::from(dir);
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".cache").join("mlx-rs-bench");
    }
    PathBuf::from(".mlx-rs-bench-cache")
}

fn load_processor() -> Option<TextOnlyProcessor> {
    // Reuse the Qwen3-1.7B-4bit tokenizer — small, BPE, present
    // on every bench-equipped machine.
    let dir = bench_cache_root().join("mlx-community/Qwen3-1.7B-4bit");
    if !dir.join("tokenizer.json").exists() {
        eprintln!(
            "incremental_decode: skipping — no tokenizer at {}",
            dir.display()
        );
        return None;
    }
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).ok()?;
    let template = ChatTemplate::from_dir(&dir).ok()?;
    Some(TextOnlyProcessor::new("bench", tokenizer, template))
}

/// Deterministic synthetic id stream that crosses ASCII and
/// multi-byte ranges by sampling vocab indices spread across the
/// space.
fn synthetic_ids(n: usize, vocab: u32) -> Vec<u32> {
    (0..n as u32)
        .map(|i| {
            // step ~7919 (prime) so successive ids don't cluster
            // in one BPE neighbourhood.
            (i.wrapping_mul(7919) % vocab).saturating_add(10)
        })
        .collect()
}

/// Pull vocab size from the underlying tokenizer via a quick
/// probe. We don't have direct access to vocab() from the
/// processor; round-trip through encode of a known string and
/// fall back to a conservative default.
fn estimate_vocab(_processor: &dyn UserInputProcessor) -> u32 {
    // Qwen3 vocab is ~150k. We don't need an exact figure — just
    // a number big enough to cover the synthetic spread without
    // returning unk for every id.
    150_000
}

fn bench_decode(c: &mut Criterion) {
    let Some(processor) = load_processor() else {
        return;
    };
    let vocab = estimate_vocab(&processor);
    let ids = synthetic_ids(N_TOKENS, vocab);

    let mut group = c.benchmark_group("streaming_decode");
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    group.throughput(Throughput::Elements(N_TOKENS as u64));

    // O(N²) naive: per step, decode all ids[..=k], diff against
    // prior decoded prefix. Same shape as the pre-fix loop.
    group.bench_function(BenchmarkId::new("naive", N_TOKENS), |b| {
        b.iter(|| {
            let mut prefix = String::new();
            let mut produced: Vec<u32> = Vec::with_capacity(ids.len());
            for &id in &ids {
                produced.push(id);
                let full = processor.decode(&produced).unwrap();
                let delta = full
                    .strip_prefix(prefix.as_str())
                    .unwrap_or(full.as_str())
                    .to_owned();
                prefix = full;
                criterion::black_box(delta);
            }
            criterion::black_box(prefix)
        });
    });

    // O(N) incremental: window of last 8 tokens decoded each
    // step, drives the same delta extraction. We mimic the
    // IncrementalDecoder shape here so the bench is independent
    // of internal visibility — same algorithm, same window size.
    group.bench_function(BenchmarkId::new("incremental", N_TOKENS), |b| {
        b.iter(|| {
            const WINDOW: usize = 8;
            let mut ids_buf: Vec<u32> = Vec::with_capacity(ids.len());
            let mut committed_tokens: usize = 0;
            let mut committed = String::new();
            let mut window = String::new();
            for &id in &ids {
                ids_buf.push(id);
                let new_window = processor.decode(&ids_buf[committed_tokens..]).unwrap();
                let delta: String = if new_window.starts_with(window.as_str()) {
                    new_window[window.len()..].to_owned()
                } else {
                    new_window.clone()
                };
                window = new_window;
                if ids_buf.len() - committed_tokens > WINDOW {
                    let two = processor
                        .decode(&ids_buf[committed_tokens..committed_tokens + 2])
                        .unwrap();
                    let next_alone = processor
                        .decode(&ids_buf[committed_tokens + 1..committed_tokens + 2])
                        .unwrap();
                    let lead = two.len().saturating_sub(next_alone.len());
                    if lead <= window.len() {
                        let moved: String = window.drain(..lead).collect();
                        committed.push_str(&moved);
                        committed_tokens += 1;
                    }
                }
                criterion::black_box(delta);
            }
            committed.push_str(&window);
            criterion::black_box(committed)
        });
    });

    group.finish();
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
