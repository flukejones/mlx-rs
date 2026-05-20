//! Minimal example driver: load a Qwen3 checkpoint and run one
//! `mlx_lm::generate` call. For interactive use see `chat`; for
//! one-shot completion with full CLI options see `generate`.

#![allow(clippy::print_stderr)]
#![allow(clippy::print_stdout)]

use std::io::Write;
use std::path::Path;

use mlx_lm::chat_template::ChatMessage;
use mlx_lm::{generate, load, GenerateParams, SamplingParams, UserInput};

const DEFAULT_MODEL_DIR: &str = "./cache/Qwen3-4B-bf16";

fn main() -> anyhow::Result<()> {
    let dir = std::env::var("MLX_LM_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| Path::new(DEFAULT_MODEL_DIR).to_path_buf());
    eprintln!("[loading {}]", dir.display());
    let mut ctx = load(&dir)?;

    let input = UserInput::chat(vec![ChatMessage::user("what's your name?")]);
    let params = GenerateParams {
        max_new_tokens: 256,
        sampling: SamplingParams {
            temperature: 0.2,
            top_p: None,
        },
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
