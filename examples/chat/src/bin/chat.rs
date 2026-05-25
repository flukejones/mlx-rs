//! Interactive REPL against any `mlxr_lm` checkpoint. KV cache resets
//! between turns; the full chat history is re-rendered each request.

#![allow(clippy::print_stderr, reason = "CLI binary logs to stderr")]
#![allow(clippy::print_stdout, reason = "CLI binary prints to stdout")]

use std::io::Write;
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use argh::FromArgs;
use chat::think_stream::ThinkStream;
use mlxr_lm::cache::{CacheKind, CacheOptions};
use mlxr_lm::chat_template::ChatMessage;
use mlxr_lm::{generate, load, GenerateParams, ModelContext, Sampler, UserInput};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

const DEFAULT_MAX_TOKENS: i32 = 1024;

const C_BOT: &str = "\x1b[1;32m";
const C_DIM: &str = "\x1b[2m";
const C_RESET: &str = "\x1b[0m";

/// Interactive REPL against any `mlxr_lm` checkpoint.
#[derive(FromArgs)]
struct Args {
    /// path to a loadable model directory (config.json + safetensors)
    #[argh(option)]
    model: PathBuf,

    /// sampling temperature; 0.0 = greedy (default 0.0)
    #[argh(option, default = "0.0")]
    temperature: f32,

    /// nucleus top-p threshold; omit for pure temperature sampling
    #[argh(option, long = "top-p")]
    top_p: Option<f32>,

    /// maximum new tokens per assistant turn (default 1024)
    #[argh(option, default = "DEFAULT_MAX_TOKENS")]
    max_tokens: i32,

    /// thinking mode: on | off | default (template's `enable_thinking`)
    #[argh(option, default = "ThinkMode::Default", from_str_fn(parse_think_mode))]
    think: ThinkMode,

    /// KV cache backing: standard | q8 | q4 (default standard)
    #[argh(
        option,
        long = "kv-cache",
        default = "KvCacheArg::Dense",
        from_str_fn(parse_kv_cache)
    )]
    kv_cache: KvCacheArg,

    /// turboquant Π rotation (quantised cache only). Ignored unless --kv-cache is q4/q8.
    #[argh(switch, long = "turbo-quant")]
    turbo_quant: bool,

    /// max tokens per prefill chunk (default 2048). 0 disables chunking.
    #[argh(option, long = "prefill-chunk-size")]
    prefill_chunk_size: Option<i32>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ThinkMode {
    On,
    Off,
    Default,
}

fn parse_think_mode(s: &str) -> std::result::Result<ThinkMode, String> {
    match s {
        "on" | "true" | "1" => Ok(ThinkMode::On),
        "off" | "false" | "0" => Ok(ThinkMode::Off),
        "default" => Ok(ThinkMode::Default),
        other => Err(format!("--think: expected on|off|default, got {other}")),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum KvCacheArg {
    Dense,
    Q8,
    Q4,
}

fn parse_kv_cache(s: &str) -> std::result::Result<KvCacheArg, String> {
    match s {
        "standard" | "std" | "dense" => Ok(KvCacheArg::Dense),
        "q8" | "quantized-q8" => Ok(KvCacheArg::Q8),
        "q4" | "quantized-q4" => Ok(KvCacheArg::Q4),
        other => Err(format!("--kv-cache: expected standard|q8|q4, got {other}")),
    }
}

impl KvCacheArg {
    fn to_options(
        self,
        turbo_quant_seed: Option<u64>,
        max_prefill_chunk: Option<i32>,
    ) -> CacheOptions {
        let base = match self {
            Self::Dense => CacheOptions::default(),
            Self::Q8 => CacheOptions {
                kind: CacheKind::quantized_q8(),
                turbo_quant_seed,
                ..CacheOptions::default()
            },
            Self::Q4 => CacheOptions {
                kind: CacheKind::quantized_q4(),
                turbo_quant_seed,
                ..CacheOptions::default()
            },
        };
        CacheOptions {
            max_prefill_chunk,
            ..base
        }
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let args: Args = argh::from_env();

    eprintln!("[loading {}]", args.model.display());
    let mut ctx = load(&args.model).context("load model")?;
    let turbo_quant_seed = args.turbo_quant.then(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before UNIX_EPOCH")
            .as_nanos() as u64
    });
    // 0 → disable chunking. Otherwise the user-provided value (or
    // `CacheOptions::default()`'s 2048 fallback) is used.
    let max_prefill_chunk = match args.prefill_chunk_size {
        Some(0) => None,
        Some(n) => Some(n),
        None => CacheOptions::default().max_prefill_chunk,
    };
    ctx.model
        .set_cache_options(
            args.kv_cache
                .to_options(turbo_quant_seed, max_prefill_chunk),
        )
        .context("set kv-cache options")?;

    let mut history: Vec<ChatMessage> = Vec::new();
    let mut editor = DefaultEditor::new().context("rustyline init")?;
    eprintln!("[ready. /exit to quit. /reset to clear history.]");

    loop {
        let input = match editor.readline("> ") {
            Ok(s) => s,
            Err(ReadlineError::Interrupted | ReadlineError::Eof) => break,
            Err(e) => return Err(anyhow::anyhow!("readline: {e}")),
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
        editor.add_history_entry(trimmed).ok();

        history.push(ChatMessage::user(trimmed));
        let mut user_input = UserInput::chat(history.clone());
        match args.think {
            ThinkMode::On => {
                user_input = user_input
                    .with_template_kwarg("enable_thinking", serde_json::Value::Bool(true));
            }
            ThinkMode::Off => {
                user_input = user_input
                    .with_template_kwarg("enable_thinking", serde_json::Value::Bool(false));
            }
            ThinkMode::Default => {}
        }
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

        match run_turn(&mut ctx, user_input, params) {
            Ok(text) => history.push(ChatMessage::assistant(text)),
            Err(e) => {
                // Pop the unanswered user turn so the next prompt
                // isn't a duplicate of the failed one.
                history.pop();
                eprintln!("[error: {e:#}]");
            }
        }
        println!();
    }
    Ok(())
}

fn run_turn(ctx: &mut ModelContext, input: UserInput, params: GenerateParams) -> Result<String> {
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(C_BOT.as_bytes())?;
    stdout.flush()?;
    let mut md = ThinkStream::new(stdout);

    let t_start = Instant::now();
    let mut t_first: Option<Instant> = None;
    let mut push_err: Option<std::io::Error> = None;
    let result = generate(ctx, input, params, &mut |_, delta| {
        if t_first.is_none() {
            t_first = Some(Instant::now());
        }
        if let Err(e) = md.push(delta) {
            push_err = Some(e);
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    })?;
    let t_end = Instant::now();

    md.finish()?;
    if let Some(e) = push_err {
        return Err(e.into());
    }
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(C_RESET.as_bytes())?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    drop(stdout);

    let t_first = t_first.unwrap_or(t_end);
    let prefill_s = (t_first - t_start).as_secs_f64();
    let decode_s = (t_end - t_first).as_secs_f64();
    let prefill_tps = safe_rate(result.prompt_tokens as f64, prefill_s);
    let decode_steps = result.completion_tokens.saturating_sub(1);
    let decode_tps = safe_rate(decode_steps as f64, decode_s);
    eprintln!(
        "{C_DIM}[prefill: {n_prompt} tok in {prefill_s:.2}s ({prefill_tps:.1} tok/s) | \
         decode: {decode_steps} tok in {decode_s:.2}s ({decode_tps:.1} tok/s)]{C_RESET}",
        n_prompt = result.prompt_tokens,
    );
    Ok(result.text)
}

/// Token-rate `n / seconds`, returning 0.0 for the degenerate
/// zero-duration case (single-token prompt + zero-token decode).
fn safe_rate(n: f64, seconds: f64) -> f64 {
    if seconds > 0.0 {
        n / seconds
    } else {
        0.0
    }
}
