//! MLX allocator memory introspection for gemma4.
//!
//! Samples mlx-core's allocator state (active / cache / peak) at each
//! phase of a decode: idle → model loaded → cache built → prefill →
//! per-decode-step. Helps separate "model weights" from "MLX
//! buffer-pool cache" from "live working set".
//!
//! Wraps the same `mlx_sys` raw bindings mlx-rs doesn't yet expose.

use std::path::PathBuf;
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

type BoxErr = Box<dyn std::error::Error + Send + Sync>;
type Res<T> = std::result::Result<T, BoxErr>;

struct MemSnapshot {
    active_mb: f64,
    cache_mb: f64,
    peak_mb: f64,
}

fn snapshot() -> MemSnapshot {
    MemSnapshot {
        active_mb: mlx_rs::memory::active_memory() as f64 / 1e6,
        cache_mb: mlx_rs::memory::cache_memory() as f64 / 1e6,
        peak_mb: mlx_rs::memory::peak_memory() as f64 / 1e6,
    }
}

fn print_row(label: &str, t: f64, s: &MemSnapshot) {
    println!(
        "{:>40}  t={:>6.2}s  active={:>7.1} MB  cache={:>7.1} MB  peak={:>7.1} MB",
        label, t, s.active_mb, s.cache_mb, s.peak_mb
    );
}

fn reset_peak() {
    mlx_rs::memory::reset_peak_memory();
}

fn clear_cache() {
    mlx_rs::memory::clear_cache();
}

fn encode_prompt(model_dir: &PathBuf, prompt: &str) -> Res<Array> {
    let tok = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| format!("{e:?}"))?;
    let enc = tok.encode(prompt, true).map_err(|e| format!("{e:?}"))?;
    let ids: Vec<u32> = enc.get_ids().to_vec();
    Ok(Array::from(&ids[..]).index(NewAxis))
}

fn main() -> Res<()> {
    let mut model_dir: Option<PathBuf> = None;
    let mut prompt = "Tell me a fact about whales.".to_string();
    let mut steps: usize = 30;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => model_dir = Some(PathBuf::from(it.next().ok_or("--model needs a value")?)),
            "--prompt" => prompt = it.next().ok_or("--prompt needs a value")?,
            "--steps" => steps = it.next().ok_or("--steps needs a value")?.parse()?,
            "-h" | "--help" => {
                eprintln!("mem_profile --model DIR [--prompt TEXT] [--steps 30]");
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}").into()),
        }
    }
    let model_dir = model_dir.ok_or("--model required")?;

    let t0 = Instant::now();
    println!(
        "{:>40}  t={:>6}s  active={:>7} MB  cache={:>7} MB  peak={:>7} MB",
        "label", "t", "", "", ""
    );
    print_row("idle (process start)", t0.elapsed().as_secs_f64(), &snapshot());

    let mut model = load_gemma4_model(&model_dir)?;
    print_row("after load_gemma4_model", t0.elapsed().as_secs_f64(), &snapshot());

    let cfg = gemma4::loader::get_gemma4_model_args(&model_dir)?;
    let mut cache: Vec<Option<Gemma4LayerCache>> = make_gemma4_caches(&cfg);
    print_row(
        "after make_gemma4_caches (empty)",
        t0.elapsed().as_secs_f64(),
        &snapshot(),
    );

    let prompt_arr = encode_prompt(&model_dir, &prompt)?;
    let n_prompt = prompt_arr.shape()[1];
    print_row(
        &format!("encoded prompt ({n_prompt} tokens)"),
        t0.elapsed().as_secs_f64(),
        &snapshot(),
    );

    // Reset peak counter so we measure the prefill watermark fresh.
    reset_peak();
    let mut gen = gemma4::Generate::<Gemma4LayerCache>::new(
        &mut model,
        &mut cache,
        0.0,
        &prompt_arr,
    );
    let first = gen.next().ok_or("no first token")??;
    eval([&first])?;
    print_row(
        "after prefill + eval(first)",
        t0.elapsed().as_secs_f64(),
        &snapshot(),
    );

    for n in 0..steps {
        let tok = gen.next().ok_or("decode exhausted")??;
        eval([&tok])?;
        if n == 0 || n == 4 || n == 9 || n == steps - 1 {
            print_row(
                &format!("after decode step {n}"),
                t0.elapsed().as_secs_f64(),
                &snapshot(),
            );
        }
    }

    // Drop the iterator, force a fresh snapshot.
    drop(gen);
    print_row("after dropping Generate iter", t0.elapsed().as_secs_f64(), &snapshot());

    // What does the MLX buffer pool hold onto after the iter goes away?
    let before_clear = snapshot();
    clear_cache();
    let after_clear = snapshot();
    println!();
    println!(
        "mlx_clear_cache() reclaimed: cache {:.1} → {:.1} MB ({:+.1})  active {:.1} → {:.1} MB ({:+.1})",
        before_clear.cache_mb,
        after_clear.cache_mb,
        after_clear.cache_mb - before_clear.cache_mb,
        before_clear.active_mb,
        after_clear.active_mb,
        after_clear.active_mb - before_clear.active_mb,
    );
    print_row("after mlx_clear_cache", t0.elapsed().as_secs_f64(), &snapshot());

    Ok(())
}
