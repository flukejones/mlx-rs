//! One-shot text completion against any `mlxr_lm` checkpoint.
//!
//! Builds a [`mlxr_lm::ModelContext`] via [`mlxr_lm::load`] and runs
//! one [`mlxr_lm::generate`] call against a single prompt, streaming
//! tokens to stdout.

#![allow(clippy::print_stderr, reason = "CLI binary logs to stderr")]
#![allow(clippy::print_stdout, reason = "CLI binary prints to stdout")]

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use argh::FromArgs;
use mlxr_lm::chat_template::ChatMessage;
use mlxr_lm::{generate, load, GenerateParams, Sampler, UserInput};

/// One-shot text completion against any `mlxr_lm` checkpoint.
#[derive(FromArgs)]
struct Args {
    /// checkpoint directory (must contain config.json + safetensors)
    #[argh(option)]
    model: PathBuf,

    /// user prompt
    #[argh(option)]
    prompt: String,

    /// sampling temperature; 0.0 = greedy (default)
    #[argh(option, default = "0.0")]
    temperature: f32,

    /// nucleus-sampling top-p cutoff
    #[argh(option)]
    top_p: Option<f32>,

    /// maximum new tokens to generate (default 256)
    #[argh(option, default = "256")]
    max_tokens: i32,

    /// skip chat-template rendering; feed the prompt as raw text
    #[argh(switch)]
    no_chat_template: bool,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let args: Args = argh::from_env();

    eprintln!("[loading {}]", args.model.display());
    let mut ctx = load(&args.model)?;

    let input = if args.no_chat_template {
        UserInput::text(args.prompt)
    } else {
        UserInput::chat(vec![ChatMessage::user(args.prompt)])
    };
    let sampling = match (args.temperature, args.top_p) {
        (0.0, _) => Sampler::Greedy,
        (t, None) => Sampler::Temperature(t),
        (t, Some(p)) => Sampler::TopP { temperature: t, p },
    };
    let params = GenerateParams {
        max_new_tokens: args.max_tokens,
        sampling,
        ..GenerateParams::default()
    };

    let mut stdout = std::io::stdout().lock();
    let t_start = Instant::now();
    let mut t_first: Option<Instant> = None;
    let result = generate(&mut ctx, input, params, &mut |_, delta| {
        if t_first.is_none() {
            t_first = Some(Instant::now());
        }
        let _ = stdout.write_all(delta.as_bytes());
        let _ = stdout.flush();
        std::ops::ControlFlow::Continue(())
    })?;
    let t_end = Instant::now();
    println!();
    let t_first = t_first.unwrap_or(t_end);
    let prefill_s = (t_first - t_start).as_secs_f64();
    let decode_s = (t_end - t_first).as_secs_f64();
    let prefill_tps = result.prompt_tokens as f64 / prefill_s;
    let decode_tps = (result.completion_tokens.saturating_sub(1)) as f64 / decode_s.max(1e-9);
    eprintln!(
        "[prompt tokens={} completion={} finish={:?} | prefill {:.2}s ({:.1} tok/s) | decode {:.2}s ({:.1} tok/s)]",
        result.prompt_tokens,
        result.completion_tokens,
        result.finish_reason,
        prefill_s,
        prefill_tps,
        decode_s,
        decode_tps
    );
    Ok(())
}
