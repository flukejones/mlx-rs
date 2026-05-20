//! Interactive REPL against any `mlx_lm` checkpoint.
//!
//! Builds a [`mlx_lm::ModelContext`] via [`mlx_lm::load`] and drives
//! it through [`mlx_lm::generate`]. Family detection happens inside
//! `load`; the REPL is family-agnostic. Multi-turn state is kept by
//! re-rendering the full chat history on every turn — the model's KV
//! cache is reset between turns (matches the upstream Python
//! `mlx-chat` shape, simpler to reason about than incremental cache
//! reuse).

#![allow(clippy::print_stderr, reason = "CLI binary logs to stderr")]
#![allow(clippy::print_stdout, reason = "CLI binary prints to stdout")]

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use mlx_lm::chat_template::ChatMessage;
use mlx_lm::{generate, load, GenerateParams, ModelContext, SamplingParams, UserInput};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

type BoxError = Box<dyn std::error::Error + Send + Sync>;
type Result<T> = std::result::Result<T, BoxError>;

const DEFAULT_MAX_TOKENS: i32 = 1024;

// ANSI colour codes for the REPL prompt + assistant turn + stats.
// `eprintln!`/`println!` writes them through; terminals that don't
// support ANSI render them as text — the rest of the output stays
// readable.
const C_BOT: &str = "\x1b[1;32m";
const C_DIM: &str = "\x1b[2m";
const C_RESET: &str = "\x1b[0m";

struct Args {
    model: PathBuf,
    temperature: f32,
    top_p: Option<f32>,
    max_tokens: i32,
}

fn parse_args() -> Result<Args> {
    let mut model: Option<PathBuf> = None;
    let mut temperature: f32 = 0.0;
    let mut top_p: Option<f32> = None;
    let mut max_tokens: i32 = DEFAULT_MAX_TOKENS;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => {
                model = Some(PathBuf::from(it.next().ok_or("--model needs a path")?));
            }
            "--temp" | "--temperature" => {
                temperature = it.next().ok_or("--temp needs a value")?.parse()?;
            }
            "--top-p" => {
                top_p = Some(it.next().ok_or("--top-p needs a value")?.parse()?);
            }
            "--max-tokens" | "--max_tokens" => {
                max_tokens = it.next().ok_or("--max-tokens needs a value")?.parse()?;
            }
            "-h" | "--help" => {
                println!(
                    "chat --model <dir> [--temp 0.0] [--top-p <f>] [--max-tokens {DEFAULT_MAX_TOKENS}]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        temperature,
        top_p,
        max_tokens,
    })
}

fn main() -> Result<()> {
    let args = parse_args()?;
    eprintln!("[loading {}]", args.model.display());
    let mut ctx = load(&args.model)?;

    let mut history: Vec<ChatMessage> = Vec::new();
    let mut editor = DefaultEditor::new().map_err(|e| format!("rustyline init: {e}"))?;
    eprintln!("[ready. /exit to quit. /reset to clear history.]");

    loop {
        let input = match editor.readline("> ") {
            Ok(s) => s,
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => break,
            Err(e) => return Err(format!("readline: {e}").into()),
        };
        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == "/exit" || trimmed == "/quit" {
            break;
        }
        if trimmed == "/reset" {
            history.clear();
            eprintln!("[history cleared]");
            continue;
        }
        let _ = editor.add_history_entry(trimmed);

        history.push(ChatMessage::user(trimmed));
        let user_input = UserInput::chat(history.clone());
        let params = GenerateParams {
            max_new_tokens: args.max_tokens,
            sampling: SamplingParams {
                temperature: args.temperature,
                top_p: args.top_p,
            },
            ..GenerateParams::default()
        };

        let outcome = run_turn(&mut ctx, user_input, params);
        match outcome {
            Ok(text) => {
                history.push(ChatMessage::assistant(text));
            }
            Err(e) => {
                // Pop the unanswered user turn so the next prompt
                // isn't a duplicate of the failed one.
                history.pop();
                eprintln!("[error: {e}]");
            }
        }
        println!();
    }
    Ok(())
}

/// Run one generation turn:
/// - colour the assistant's streamed output in bot-green
/// - measure prefill (start → first token delta) and decode
///   (first delta → end) wall-clock independently, print per-phase
///   token rates after the turn closes
fn run_turn(ctx: &mut ModelContext, input: UserInput, params: GenerateParams) -> Result<String> {
    let mut stdout = std::io::stdout().lock();
    let _ = stdout.write_all(C_BOT.as_bytes());
    let _ = stdout.flush();

    let t_start = Instant::now();
    let mut t_first: Option<Instant> = None;
    let result = generate(ctx, input, params, &mut |_, delta| {
        if t_first.is_none() {
            t_first = Some(Instant::now());
        }
        let _ = stdout.write_all(delta.as_bytes());
        let _ = stdout.flush();
        std::ops::ControlFlow::Continue(())
    })?;
    let t_end = Instant::now();

    let _ = stdout.write_all(C_RESET.as_bytes());
    let _ = stdout.write_all(b"\n");
    let _ = stdout.flush();
    drop(stdout);

    let t_first = t_first.unwrap_or(t_end);
    let prefill_s = (t_first - t_start).as_secs_f64();
    let decode_s = (t_end - t_first).as_secs_f64();
    let prefill_tps = if prefill_s > 0.0 {
        result.prompt_tokens as f64 / prefill_s
    } else {
        0.0
    };
    let decode_steps = result.completion_tokens.saturating_sub(1);
    let decode_tps = if decode_s > 0.0 {
        decode_steps as f64 / decode_s
    } else {
        0.0
    };
    eprintln!(
        "{C_DIM}[prefill: {n_prompt} tok in {prefill_s:.2}s ({prefill_tps:.1} tok/s) | \
         decode: {decode_steps} tok in {decode_s:.2}s ({decode_tps:.1} tok/s)]{C_RESET}",
        n_prompt = result.prompt_tokens,
    );
    Ok(result.text)
}
