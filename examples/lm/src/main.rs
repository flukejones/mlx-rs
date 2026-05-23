//! Minimal example driver: load a Qwen3 checkpoint and run one
//! `mlxr_lm::generate` call. For interactive use see `chat`; for
//! one-shot completion with full CLI options see `generate`.

#![allow(clippy::print_stderr)]
#![allow(clippy::print_stdout)]

use std::io::Write;
use std::path::PathBuf;

use mlxr_lm::chat_template::ChatMessage;
use mlxr_lm::{generate, load, GenerateParams, Sampler, UserInput};

/// Default checkpoint relative to the bench-cache root. The cache root
/// resolves via `MLX_LM_BENCH_CACHE` → `XDG_CACHE_HOME/mlx-rs-bench` →
/// `~/.cache/mlx-rs-bench` (matching the bench harness). Override the
/// full path with `MLX_LM_MODEL_DIR`.
const DEFAULT_MODEL_REPO: &str = "mlx-community/Qwen3-1.7B-4bit";

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

fn main() -> anyhow::Result<()> {
    let dir = std::env::var("MLX_LM_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| bench_cache_root().join(DEFAULT_MODEL_REPO));
    eprintln!("[loading {}]", dir.display());
    let mut ctx = load(&dir)?;

    let input = UserInput::chat(vec![ChatMessage::user("what's your name?")]);
    let params = GenerateParams {
        max_new_tokens: 256,
        sampling: Sampler::Temperature(0.2),
        ..GenerateParams::default()
    };

    let mut stdout = std::io::stdout().lock();
    let result = generate(&mut ctx, input, params, &mut |_, delta| {
        let _ = stdout.write_all(delta.as_bytes());
        let _ = stdout.flush();
        std::ops::ControlFlow::Continue(())
    })?;
    println!();
    eprintln!(
        "[prompt={} completion={}]",
        result.prompt_tokens, result.completion_tokens
    );
    Ok(())
}
