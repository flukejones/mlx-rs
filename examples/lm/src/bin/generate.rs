//! One-shot text completion against any `mlxr_lm` checkpoint.
//!
//! Builds a [`mlxr_lm::ModelContext`] via [`mlxr_lm::load`] and runs
//! one [`mlxr_lm::generate`] call against a single prompt, streaming
//! tokens to stdout.

#![allow(clippy::print_stderr, reason = "CLI binary logs to stderr")]
#![allow(clippy::print_stdout, reason = "CLI binary prints to stdout")]

use std::io::Write;
use std::path::PathBuf;

use mlxr_lm::chat_template::ChatMessage;
use mlxr_lm::{generate, load, GenerateParams, Sampler, UserInput};

type BoxError = Box<dyn std::error::Error + Send + Sync>;
type Result<T> = std::result::Result<T, BoxError>;

const DEFAULT_MAX_TOKENS: i32 = 256;

struct Args {
    model: PathBuf,
    prompt: String,
    temperature: f32,
    top_p: Option<f32>,
    max_tokens: i32,
    no_chat_template: bool,
}

fn parse_args() -> Result<Args> {
    let mut model: Option<PathBuf> = None;
    let mut prompt: Option<String> = None;
    let mut temperature: f32 = 0.0;
    let mut top_p: Option<f32> = None;
    let mut max_tokens: i32 = DEFAULT_MAX_TOKENS;
    let mut no_chat_template = false;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => model = Some(PathBuf::from(it.next().ok_or("--model needs a path")?)),
            "--prompt" => prompt = Some(it.next().ok_or("--prompt needs a value")?),
            "--temp" | "--temperature" => {
                temperature = it.next().ok_or("--temp needs a value")?.parse()?;
            }
            "--top-p" => top_p = Some(it.next().ok_or("--top-p needs a value")?.parse()?),
            "--max-tokens" | "--max_tokens" => {
                max_tokens = it.next().ok_or("--max-tokens needs a value")?.parse()?;
            }
            "--no-chat-template" => no_chat_template = true,
            "-h" | "--help" => {
                println!(
                    "generate --model <dir> --prompt <s> [--temp 0.0] [--top-p <f>] \
                     [--max-tokens {DEFAULT_MAX_TOKENS}] [--no-chat-template]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        prompt: prompt.ok_or("--prompt is required")?,
        temperature,
        top_p,
        max_tokens,
        no_chat_template,
    })
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let args = parse_args()?;
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
    let t_start = std::time::Instant::now();
    let mut t_first: Option<std::time::Instant> = None;
    let result = generate(&mut ctx, input, params, &mut |_, delta| {
        if t_first.is_none() {
            t_first = Some(std::time::Instant::now());
        }
        let _ = stdout.write_all(delta.as_bytes());
        let _ = stdout.flush();
        std::ops::ControlFlow::Continue(())
    })?;
    let t_end = std::time::Instant::now();
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
