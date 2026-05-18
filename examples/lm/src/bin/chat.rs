//! Multiturn chat REPL for gemma4. KV cache persists across turns so
//! prefill of the second-and-later turns only reprocesses the new
//! user/assistant tokens.
//!
//! Usage:
//!   chat --model ~/.cache/mlx-rs-bench/mlx-community/gemma-4-26b-a4b-it-8bit
//!
//! Type a message and press Enter. Empty line submits the current message.
//! `/exit` or Ctrl-D quits. `/reset` clears the conversation + KV cache.

use std::fs::File;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use mlx_lm::models::gemma4::{
    self,
    loader::{load_gemma4_model, make_gemma4_caches, Gemma4LayerCache},
};

/// ANSI colours. Bright cyan for the user prompt, bright green for the
/// model output, dim grey for the speed line, reset at end of each
/// region so a Ctrl-C mid-stream doesn't leak colour into the shell.
const C_USER: &str = "\x1b[1;36m";
const C_BOT: &str = "\x1b[1;32m";
const C_DIM: &str = "\x1b[2m";
const C_RESET: &str = "\x1b[0m";
use mlx_lm_utils::tokenizer::{
    load_model_chat_template_from_file, load_special_tokens_from_file, ApplyChatTemplateArgs,
    Conversation, Role, Tokenizer,
};
use mlx_rs::{
    ops::indexing::{IndexOp, NewAxis},
    Array,
};

type BoxError = Box<dyn std::error::Error + Send + Sync>;
type Result<T> = std::result::Result<T, BoxError>;

const DEFAULT_MAX_TOKENS: usize = 1024;

struct Args {
    model: PathBuf,
    temp: f32,
    max_tokens: usize,
}

fn parse_args() -> Result<Args> {
    let mut model: Option<PathBuf> = None;
    let mut temp: f32 = 0.0;
    let mut max_tokens: usize = DEFAULT_MAX_TOKENS;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => model = Some(PathBuf::from(it.next().ok_or("--model needs a value")?)),
            "--temp" => temp = it.next().ok_or("--temp needs a value")?.parse()?,
            "--max-tokens" | "--max_tokens" => {
                max_tokens = it.next().ok_or("--max-tokens needs a value")?.parse()?
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: chat --model DIR [--temp 0.0] [--max-tokens {DEFAULT_MAX_TOKENS}]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}").into()),
        }
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        temp,
        max_tokens,
    })
}

fn load_eos_ids(model_dir: &Path) -> Vec<u32> {
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum EosField {
        One(u32),
        Many(Vec<u32>),
    }
    #[derive(serde::Deserialize)]
    struct Stub {
        #[serde(default)]
        eos_token_id: Option<EosField>,
    }
    for name in ["generation_config.json", "config.json"] {
        let Ok(f) = File::open(model_dir.join(name)) else { continue };
        let Ok(stub): std::result::Result<Stub, _> = serde_json::from_reader(f) else {
            continue;
        };
        if let Some(f) = stub.eos_token_id {
            return match f {
                EosField::One(x) => vec![x],
                EosField::Many(v) => v,
            };
        }
    }
    Vec::new()
}

fn encode_turn(
    model_dir: &Path,
    history: &[(Role, String)],
    add_generation_prompt: bool,
) -> Result<Vec<u32>> {
    let tok_path = model_dir.join("tokenizer.json");
    let mut tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| format!("{e:?}"))?;

    let cfg_path = model_dir.join("tokenizer_config.json");
    let jinja_path = model_dir.join("chat_template.jinja");
    let model_id = model_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    let template = if jinja_path.is_file() {
        std::fs::read_to_string(&jinja_path)?
    } else {
        load_model_chat_template_from_file(&cfg_path)?
            .ok_or("no chat template in tokenizer_config.json or chat_template.jinja")?
    };

    let conv: Vec<Conversation<Role, &str>> = history
        .iter()
        .map(|(role, content)| Conversation {
            role: *role,
            content: content.as_str(),
        })
        .collect();
    let special_tokens = load_special_tokens_from_file(&cfg_path).unwrap_or_default();
    let args = ApplyChatTemplateArgs {
        conversations: vec![conv.into()],
        documents: None,
        model_id: &model_id,
        chat_template_id: None,
        add_generation_prompt: Some(add_generation_prompt),
        continue_final_message: None,
        special_tokens,
    };
    let encodings = tokenizer
        .apply_chat_template_and_encode(template, args)
        .map_err(|e| format!("{e:?}"))?;
    Ok(encodings
        .iter()
        .flat_map(|enc| enc.get_ids())
        .copied()
        .collect())
}

fn load_decode_tokenizer(model_dir: &Path) -> Result<tokenizers::Tokenizer> {
    tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| format!("{e:?}").into())
}

fn read_user_message() -> Result<Option<String>> {
    print!("\n{C_USER}you> ");
    std::io::stdout().flush()?;
    let stdin = std::io::stdin();
    let mut line = String::new();
    let n = stdin.lock().read_line(&mut line)?;
    print!("{C_RESET}");
    std::io::stdout().flush()?;
    if n == 0 {
        return Ok(None);
    }
    Ok(Some(line.trim_end_matches(['\n', '\r']).to_string()))
}

/// First Ctrl-C cancels the in-flight generation (sets the flag).
/// Second Ctrl-C while the flag is still set hard-exits.
fn install_sigint_handler() -> Arc<AtomicBool> {
    let flag = Arc::new(AtomicBool::new(false));
    let flag_clone = flag.clone();
    ctrlc::set_handler(move || {
        if flag_clone.swap(true, Ordering::SeqCst) {
            std::process::exit(130);
        }
    })
    .expect("install SIGINT handler");
    flag
}

fn main() -> Result<()> {
    let args = parse_args()?;
    eprintln!("[loading {}]", args.model.display());

    let mut model = load_gemma4_model(&args.model)?;
    let cfg = gemma4::loader::get_gemma4_model_args(&args.model)?;
    let mut cache: Vec<Option<Gemma4LayerCache>> = make_gemma4_caches(&cfg);

    let tokenizer = load_decode_tokenizer(&args.model)?;
    let eos_ids = load_eos_ids(&args.model);
    let mut history: Vec<(Role, String)> = Vec::new();
    // Tokens already absorbed by the cache (kept in sync with cache offset).
    let mut prefix_tokens: Vec<u32> = Vec::new();
    let interrupted = install_sigint_handler();

    eprintln!(
        "[ready — type /exit or Ctrl-D to quit, /reset to clear history, Ctrl-C cancels generation]"
    );

    loop {
        let user = match read_user_message()? {
            Some(s) if s.trim().is_empty() => continue,
            Some(s) => s,
            None => {
                eprintln!();
                break;
            }
        };
        match user.trim() {
            "/exit" | "/quit" => break,
            "/reset" => {
                history.clear();
                prefix_tokens.clear();
                cache = make_gemma4_caches(&cfg);
                eprintln!("[history cleared]");
                continue;
            }
            _ => {}
        }

        history.push((Role::User, user));

        // Full templated token stream for current history (incl. generation
        // prompt). The delta vs `prefix_tokens` is what gets fed to prefill.
        let full = encode_turn(&args.model, &history, true)?;
        if full.len() <= prefix_tokens.len() || full[..prefix_tokens.len()] != prefix_tokens[..] {
            // Template can't be prefix-extended (e.g. system prompt rebuilt).
            // Cheapest correct fallback: rebuild cache + reprocess everything.
            cache = make_gemma4_caches(&cfg);
            prefix_tokens.clear();
        }
        let new_tokens = &full[prefix_tokens.len()..];
        let prompt_arr = Array::from(new_tokens).index(NewAxis);
        let n_prompt = new_tokens.len();

        let mut generate = gemma4::Generate::<Gemma4LayerCache>::new(
            &mut model,
            &mut cache,
            args.temp,
            &prompt_arr,
        );

        print!("{C_BOT}bot> ");
        std::io::stdout().flush()?;

        let mut ids: Vec<u32> = Vec::new();
        let mut last_decoded_len = 0;
        let flush = |ids: &[u32], last_decoded_len: &mut usize| {
            if let Ok(text) = tokenizer.decode(ids, true) {
                let delta = &text[*last_decoded_len..];
                print!("{delta}");
                let _ = std::io::stdout().flush();
                *last_decoded_len = text.len();
            }
        };

        interrupted.store(false, Ordering::SeqCst);
        let t_start = std::time::Instant::now();
        let mut t_first: Option<std::time::Instant> = None;
        let mut cancelled = false;
        for (n, token) in generate.by_ref().enumerate() {
            if interrupted.load(Ordering::SeqCst) {
                cancelled = true;
                break;
            }
            let arr = token.map_err(|e| format!("{e:?}"))?;
            let id = arr.item::<u32>();
            if t_first.is_none() {
                t_first = Some(std::time::Instant::now());
            }
            if eos_ids.contains(&id) || n + 1 >= args.max_tokens {
                break;
            }
            ids.push(id);
            if ids.len() % 4 == 0 {
                flush(&ids, &mut last_decoded_len);
            }
        }
        flush(&ids, &mut last_decoded_len);
        print!("{C_RESET}");
        if cancelled {
            print!(" {C_DIM}[cancelled]{C_RESET}");
        }
        println!();

        let t_first = t_first.unwrap_or(t_start);
        let prefill_s = (t_first - t_start).as_secs_f64();
        let decode_s = t_start.elapsed().as_secs_f64() - prefill_s;
        let decode_steps = ids.len().saturating_sub(1);
        let prefill_tps = if prefill_s > 0.0 { n_prompt as f64 / prefill_s } else { 0.0 };
        let decode_tps = if decode_s > 0.0 { decode_steps as f64 / decode_s } else { 0.0 };
        eprintln!(
            "{C_DIM}[prefill: {n_prompt} tok in {prefill_s:.2}s ({prefill_tps:.1} tok/s) | decode: {decode_steps} tok in {decode_s:.2}s ({decode_tps:.1} tok/s)]{C_RESET}"
        );

        if cancelled {
            // Cache holds partial decode state; drop the user turn from
            // history and rebuild the cache from scratch on next prompt.
            history.pop();
            cache = make_gemma4_caches(&cfg);
            prefix_tokens.clear();
            continue;
        }

        // Push the assistant turn into history + advance prefix_tokens to
        // include both the user prompt and the assistant response that the
        // cache now holds. Re-templating with `add_generation_prompt=false`
        // gives the canonical token stream for the completed turn.
        let assistant = tokenizer.decode(&ids, true).map_err(|e| format!("{e:?}"))?;
        history.push((Role::Assistant, assistant));
        prefix_tokens = encode_turn(&args.model, &history, false)?;
    }

    Ok(())
}
