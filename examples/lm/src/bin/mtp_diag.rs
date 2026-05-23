//! One-shot diagnostic: measure MTP acceptance rate + tok/s for the
//! 35B-A3B-q8-mtp model. Compares MTP vs greedy step throughput at
//! short (13) + long (1024) prefill using a real-text prompt tiled to
//! the target length.

#![allow(clippy::unwrap_used)]
#![allow(clippy::print_stderr)]

use std::path::PathBuf;
use std::time::Instant;

use mlxr::{
    ops::indexing::IndexOp,
    transforms::{async_eval, eval},
    Array,
};
use mlxr_lm::language_model::UserInputProcessor;
use mlxr_lm::lm_input::{LMInput, PrepareResult, Text};
use mlxr_lm::{load, SamplerState, SamplingParams, UserInput};

const REAL_TEXT_SEED: &str = "Rust is a better systems language than Python for most non-glue \
     code. The type system catches whole categories of bug at compile \
     time that Python only catches at runtime — null derefs, use-after- \
     move, data races, accidental implicit conversions. Cargo gives you \
     reproducible builds, a single test runner, a single benchmarking \
     framework, and a single doc tool without ten years of accumulated \
     packaging archaeology. Performance is closer to C than to CPython \
     by a factor of fifty or more on tight loops, and the language has \
     no global interpreter lock, so spawning threads actually buys you \
     parallelism. Memory layout is predictable: structs are flat, \
     enums are tagged unions the size of their largest variant, and \
     allocations only happen where you write them. Python wins on \
     iteration speed at the REPL and on libraries for one-off data \
     work, but those wins shrink when the code outgrows a notebook. \
     For everything that ships, Rust gives you fewer surprises in \
     production. ";

fn real_text_prompt(processor: &mut dyn UserInputProcessor, len: usize) -> Array {
    let mut text = String::with_capacity(REAL_TEXT_SEED.len() * 8);
    while text.len() < REAL_TEXT_SEED.len() * 8 {
        text.push_str(REAL_TEXT_SEED);
    }
    let lm_input = processor.prepare(UserInput::text(text)).unwrap();
    lm_input.text.tokens.index((.., ..len as i32))
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

const MODEL_REPO: &str = "mlx-community/Qwen3.6-35B-A3B-q8-mtp";

fn bench_cache_root() -> PathBuf {
    if let Ok(dir) = std::env::var("MLX_LM_BENCH_CACHE") {
        return PathBuf::from(dir);
    }
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(xdg).join("mlx-rs-bench");
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".cache").join("mlx-rs-bench");
    }
    PathBuf::from(".mlx-rs-bench-cache")
}

fn main() {
    let dir = bench_cache_root().join(MODEL_REPO);
    eprintln!("loading {}", dir.display());
    let mut ctx = load(&dir).expect("load");
    eprintln!("has_mtp={}", ctx.model.has_mtp());

    // Warmup: a single greedy + MTP pair at short prefill to JIT the
    // Metal kernel cache + mlx-c compile cache. Without this the first
    // measured cell pays cold-start overhead that distorts the
    // comparison.
    eprintln!("warmup pass…");
    let warm = real_text_prompt(ctx.processor.as_mut(), 13);
    run_greedy(&mut ctx, &warm, 10);
    ctx.model.set_mtp_depth(1);
    run_mtp(&mut ctx, &warm, 10);

    let steps = 50i32;
    for &prompt_len in &[13usize, 1024usize] {
        let prompt = real_text_prompt(ctx.processor.as_mut(), prompt_len);
        let greedy_toks = run_greedy(&mut ctx, &prompt, steps);

        ctx.model.set_mtp_depth(1);
        let (mtp1_toks, c1, a1) = run_mtp(&mut ctx, &prompt, steps);
        ctx.model.set_mtp_depth(2);
        let (mtp2_toks, c2, a2) = run_mtp(&mut ctx, &prompt, steps);
        ctx.model.set_mtp_depth(3);
        let (mtp3_toks, c3, a3) = run_mtp(&mut ctx, &prompt, steps);

        eprintln!(
            "prefill={prompt_len:>4} greedy={greedy_toks:>6.2}  \
             d1={mtp1_toks:>6.2} ({:+.1}%, {:.1}% accept, {c1} calls)  \
             d2={mtp2_toks:>6.2} ({:+.1}%, {:.1}% per-slot, {c2} calls)  \
             d3={mtp3_toks:>6.2} ({:+.1}%, {:.1}% per-slot, {c3} calls)",
            100.0 * (mtp1_toks / greedy_toks - 1.0),
            100.0 * a1 as f32 / c1 as f32,
            100.0 * (mtp2_toks / greedy_toks - 1.0),
            100.0 * a2 as f32 / (c2 * 2) as f32,
            100.0 * (mtp3_toks / greedy_toks - 1.0),
            100.0 * a3 as f32 / (c3 * 3) as f32,
        );
    }
}

fn run_greedy(ctx: &mut mlxr_lm::ModelContext, prompt: &Array, steps: i32) -> f32 {
    ctx.model.reset();
    let initial = match ctx.model.prepare(make_lm_input(prompt)).unwrap() {
        PrepareResult::Logits(arr) => arr,
        PrepareResult::Primed => unreachable!(),
    };
    let mut pending = mlxr::argmax_axis!(&initial, -1).unwrap();
    eval([&pending]).unwrap();
    let t = Instant::now();
    for _ in 0..steps {
        let logits = ctx.model.step(&pending).unwrap().logits;
        let next = mlxr::argmax_axis!(&logits, -1).unwrap();
        async_eval([&next]).unwrap();
        eval([&pending]).unwrap();
        pending = next;
    }
    eval([&pending]).unwrap();
    steps as f32 / t.elapsed().as_secs_f32()
}

/// Returns `(tok_per_sec, calls, accepted_drafts)`. `accepted_drafts`
/// is the number of draft tokens the main model accepted across all
/// calls — per call this is `committed.len() - 1`. Divide by
/// `calls * depth` for the per-draft-slot acceptance rate.
fn run_mtp(ctx: &mut mlxr_lm::ModelContext, prompt: &Array, steps: i32) -> (f32, i32, i32) {
    ctx.model.reset();
    let initial = match ctx.model.prepare(make_lm_input(prompt)).unwrap() {
        PrepareResult::Logits(arr) => arr,
        PrepareResult::Primed => unreachable!(),
    };
    let mut sampler = SamplerState::new(SamplingParams {
        temperature: 0.0,
        top_p: None,
    });
    let mut pending = sampler.sample(&initial).unwrap();
    eval([&pending]).unwrap();
    let t = Instant::now();
    let mut produced = 0i32;
    let mut calls = 0i32;
    let mut accepted_drafts = 0i32;
    while produced < steps {
        let (tokens, next_pending) = ctx
            .model
            .try_mtp_decode(&pending, &mut sampler)
            .unwrap()
            .unwrap();
        calls += 1;
        accepted_drafts += tokens.len() as i32 - 1;
        async_eval([&next_pending]).unwrap();
        eval([&pending]).unwrap();
        pending = next_pending;
        produced += tokens.len() as i32;
    }
    eval([&pending]).unwrap();
    (
        produced as f32 / t.elapsed().as_secs_f32(),
        calls,
        accepted_drafts,
    )
}
