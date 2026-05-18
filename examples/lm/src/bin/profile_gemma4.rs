//! Instrumented gemma4 decode profiler. `eval`-fenced wall-clock per
//! decode step; reports prefill + per-step mean/median/p95 +
//! per-launch budget across all layers. Use to spot regressions and
//! decide where to fuse.
//!
//! Usage:
//!   profile_gemma4 --model ~/.cache/mlx-rs-bench/mlx-community/gemma-4-26b-a4b-it-8bit \
//!                  [--steps 32] [--prompt "..."]

use std::path::{Path, PathBuf};
use std::time::Instant;

use mlx_lm::models::gemma4::{
    self,
    loader::{load_gemma4_model, make_gemma4_caches, Gemma4LayerCache},
};
use mlx_lm_utils::tokenizer::Tokenizer;
use mlx_rs::{
    ops::indexing::{IndexOp, NewAxis},
    transforms::eval,
    Array,
};

type BoxError = Box<dyn std::error::Error + Send + Sync>;
type Result<T> = std::result::Result<T, BoxError>;

struct Args {
    model: PathBuf,
    prompt: String,
    steps: usize,
}

fn parse_args() -> Result<Args> {
    let mut model: Option<PathBuf> = None;
    let mut prompt = "Tell me a fact about whales.".to_string();
    let mut steps: usize = 32;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => model = Some(PathBuf::from(it.next().ok_or("--model needs a value")?)),
            "--prompt" => prompt = it.next().ok_or("--prompt needs a value")?,
            "--steps" => steps = it.next().ok_or("--steps needs a value")?.parse()?,
            "-h" | "--help" => {
                eprintln!("usage: profile_gemma4 --model DIR [--prompt TEXT] [--steps 32]");
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}").into()),
        }
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        prompt,
        steps,
    })
}

fn encode_prompt(model_dir: &Path, prompt: &str) -> Result<Array> {
    let tok = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| format!("{e:?}"))?;
    let enc = tok.encode(prompt, true).map_err(|e| format!("{e:?}"))?;
    let ids: Vec<u32> = enc.get_ids().to_vec();
    Ok(Array::from(&ids[..]).index(NewAxis))
}

fn main() -> Result<()> {
    let args = parse_args()?;
    eprintln!("[loading {}]", args.model.display());
    let mut model = load_gemma4_model(&args.model)?;
    let cfg = gemma4::loader::get_gemma4_model_args(&args.model)?;
    let mut cache: Vec<Option<Gemma4LayerCache>> = make_gemma4_caches(&cfg);

    let prompt = encode_prompt(&args.model, &args.prompt)?;
    let n_prompt = prompt.shape()[1] as usize;

    eprintln!("[prompt: {n_prompt} tokens, decode steps: {}]", args.steps);

    // Warm-up: run prefill + 4 decode steps to compile graphs.
    {
        let mut gen = gemma4::Generate::<Gemma4LayerCache>::new(
            &mut model,
            &mut cache,
            0.0,
            &prompt,
        );
        for tok in gen.by_ref().take(5) {
            let t = tok?;
            eval([&t])?;
        }
    }

    // Reset cache, do timed pass.
    cache = make_gemma4_caches(&cfg);
    let mut gen = gemma4::Generate::<Gemma4LayerCache>::new(
        &mut model,
        &mut cache,
        0.0,
        &prompt,
    );

    // Prefill (1 forward of n_prompt tokens).
    let t0 = Instant::now();
    let first = gen.next().ok_or("no first token")??;
    eval([&first])?;
    let t_prefill = t0.elapsed();

    // Decode loop: time each step individually so we can spot variance.
    let mut step_us: Vec<u128> = Vec::with_capacity(args.steps);
    for _ in 0..args.steps {
        let t = Instant::now();
        let tok = gen.next().ok_or("decode exhausted")??;
        eval([&tok])?;
        step_us.push(t.elapsed().as_micros());
    }

    let total_us: u128 = step_us.iter().sum();
    let mean_us = total_us as f64 / step_us.len() as f64;
    let mut sorted = step_us.clone();
    sorted.sort_unstable();
    let median = sorted[sorted.len() / 2];
    let p95 = sorted[(sorted.len() * 95 / 100).min(sorted.len() - 1)];

    eprintln!();
    eprintln!("=== gemma4 instrumented decode ===");
    eprintln!(
        "prefill:   {:>7.2} ms  ({:>6.1} tok/s, {n_prompt} tokens)",
        t_prefill.as_secs_f64() * 1e3,
        n_prompt as f64 / t_prefill.as_secs_f64()
    );
    eprintln!(
        "decode:    mean {:>5.2} ms  median {:>5.2} ms  p95 {:>5.2} ms  ({:>6.1} tok/s)",
        mean_us / 1e3,
        median as f64 / 1e3,
        p95 as f64 / 1e3,
        1e6 / mean_us
    );
    eprintln!("layers:    {}", cfg.num_hidden_layers);
    eprintln!(
        "per-launch budget at observed mean: {:.1} us / layer",
        mean_us / cfg.num_hidden_layers as f64
    );

    Ok(())
}
