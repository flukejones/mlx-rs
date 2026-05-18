//! Generic multiturn chat REPL for llama / qwen3 / qwen3.5 / gemma4.
//! Dispatches on `config.json::model_type`. KV cache persists across
//! turns; only the delta vs the prior prefix is prefilled per turn.
//!
//! Usage: `chat --model DIR [--temp 0.0] [--max-tokens 1024]`
//! `/exit` quits. `/reset` clears history + cache. Ctrl-C cancels the
//! in-flight generation (second hit hard-exits).

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use mlx_lm::{
    cache::KVCache,
    models::{
        gemma4::{
            self,
            loader::{load_gemma4_model, make_gemma4_caches, Gemma4LayerCache},
            Model as Gemma4Model,
        },
        llama::{load_llama_model, Generate as LlamaGenerate, Model as LlamaModel},
        qwen3::{load_qwen3_model, Generate as Qwen3Generate, Model as Qwen3Model},
        qwen3_5::{
            cache::{make_caches as make_qwen35_caches, LayerCache as Qwen35LayerCache},
            generation::{Generate as Qwen35Generate, StopCriteria as Qwen35StopCriteria},
            layer::LanguageModel as Qwen35Model,
            weights::load_language_model as load_qwen35_lm,
            ModelConfig as Qwen35Config, SamplingParams as Qwen35SamplingParams,
        },
    },
};
use mlx_lm_utils::tokenizer::{
    load_model_chat_template_from_file, load_special_tokens_from_file, ApplyChatTemplateArgs,
    Conversation, Role, Tokenizer,
};
use mlx_rs::{
    ops::indexing::{IndexOp, NewAxis},
    Array,
};

const C_USER: &str = "\x1b[1;36m";
const C_BOT: &str = "\x1b[1;32m";
const C_DIM: &str = "\x1b[2m";
const C_RESET: &str = "\x1b[0m";

type BoxError = Box<dyn std::error::Error + Send + Sync>;
type Result<T> = std::result::Result<T, BoxError>;

const DEFAULT_MAX_TOKENS: usize = 1024;

struct Args {
    model: PathBuf,
    temp: f32,
    max_tokens: usize,
    /// `Some(true)` / `Some(false)` set the template's `enable_thinking`
    /// kwarg (qwen 3 / qwen 3.5 / deepseek-r1 / glm-4); `None` leaves
    /// the template default in place.
    thinking: Option<bool>,
}

fn parse_args() -> Result<Args> {
    let mut model: Option<PathBuf> = None;
    let mut temp: f32 = 0.0;
    let mut max_tokens: usize = DEFAULT_MAX_TOKENS;
    let mut thinking: Option<bool> = None;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => model = Some(PathBuf::from(it.next().ok_or("--model needs a value")?)),
            "--temp" => temp = it.next().ok_or("--temp needs a value")?.parse()?,
            "--max-tokens" | "--max_tokens" => {
                max_tokens = it.next().ok_or("--max-tokens needs a value")?.parse()?
            }
            "--think" => {
                let v = it.next().ok_or("--think needs on|off|default")?;
                thinking = match v.as_str() {
                    "on" | "true" | "1" => Some(true),
                    "off" | "false" | "0" => Some(false),
                    "default" | "auto" => None,
                    other => return Err(format!("--think: expected on|off|default, got {other}").into()),
                };
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: chat --model DIR [--temp 0.0] [--max-tokens {DEFAULT_MAX_TOKENS}] \
                     [--think on|off|default]"
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
        thinking,
    })
}

fn detect_family(model_dir: &Path) -> Result<String> {
    #[derive(serde::Deserialize)]
    struct ConfigStub {
        #[serde(default)]
        model_type: String,
        #[serde(default)]
        text_config: Option<TextStub>,
        #[serde(default)]
        architectures: Vec<String>,
    }
    #[derive(serde::Deserialize)]
    struct TextStub {
        #[serde(default)]
        model_type: String,
    }
    let f = File::open(model_dir.join("config.json"))?;
    let cfg: ConfigStub = serde_json::from_reader(f)?;
    if !cfg.model_type.is_empty() {
        return Ok(cfg.model_type);
    }
    if let Some(t) = cfg.text_config.as_ref() {
        if !t.model_type.is_empty() {
            return Ok(t.model_type.clone());
        }
    }
    if let Some(arch) = cfg.architectures.first() {
        return Ok(arch.to_lowercase());
    }
    Err("could not detect model family from config.json".into())
}

fn load_eos_ids(model_dir: &Path, tokenizer: &tokenizers::Tokenizer) -> Vec<u32> {
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum EosField {
        One(u32),
        Many(Vec<u32>),
    }
    #[derive(serde::Deserialize)]
    struct ConfigStub {
        #[serde(default)]
        eos_token_id: Option<EosField>,
    }
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum TokenEntry {
        Bare(String),
        Wrapped { content: String },
    }
    #[derive(serde::Deserialize)]
    struct TokConfig {
        #[serde(default)]
        eos_token: Option<TokenEntry>,
    }

    let mut ids: Vec<u32> = Vec::new();
    for name in ["generation_config.json", "config.json"] {
        let Ok(f) = File::open(model_dir.join(name)) else { continue };
        let Ok(stub): std::result::Result<ConfigStub, _> = serde_json::from_reader(f) else {
            continue;
        };
        if let Some(f) = stub.eos_token_id {
            match f {
                EosField::One(x) => ids.push(x),
                EosField::Many(v) => ids.extend(v),
            }
            break;
        }
    }

    // Chat models commonly use `<|im_end|>` (or similar) as the
    // turn-end token while `config.json::eos_token_id` points at a
    // different `<|endoftext|>` id. The tokenizer's `eos_token` is the
    // authoritative end-of-turn marker — encode it and merge.
    if let Ok(f) = File::open(model_dir.join("tokenizer_config.json")) {
        if let Ok(tc) = serde_json::from_reader::<_, TokConfig>(f) {
            let tok_str = tc.eos_token.map(|t| match t {
                TokenEntry::Bare(s) => s,
                TokenEntry::Wrapped { content } => content,
            });
            if let Some(s) = tok_str {
                if let Ok(enc) = tokenizer.encode(s.as_str(), false) {
                    for &id in enc.get_ids() {
                        if !ids.contains(&id) {
                            ids.push(id);
                        }
                    }
                }
            }
        }
    }
    ids
}

fn encode_turn(
    model_dir: &Path,
    history: &[(Role, String)],
    add_generation_prompt: bool,
    template_kwargs: &HashMap<String, serde_json::Value>,
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
        template_kwargs: template_kwargs.clone(),
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

/// Result of a single readline turn.
enum InputOutcome {
    /// User submitted a line (post-trim it may still be empty; caller
    /// decides how to handle that).
    Line(String),
    /// Ctrl-C while editing: prompt cleared, REPL stays open.
    Cleared,
    /// Ctrl-D / EOF: exit the REPL.
    Eof,
}

fn read_user_message(rl: &mut rustyline::DefaultEditor) -> Result<InputOutcome> {
    let prompt = format!("\n{C_USER}you> {C_RESET}");
    match rl.readline(&prompt) {
        Ok(line) => {
            if !line.trim().is_empty() {
                let _ = rl.add_history_entry(line.as_str());
            }
            Ok(InputOutcome::Line(line))
        }
        Err(rustyline::error::ReadlineError::Interrupted) => Ok(InputOutcome::Cleared),
        Err(rustyline::error::ReadlineError::Eof) => Ok(InputOutcome::Eof),
        Err(e) => Err(format!("readline: {e:?}").into()),
    }
}

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

enum Backend {
    Llama {
        model: LlamaModel,
        cache: Vec<Option<KVCache>>,
    },
    Qwen3 {
        model: Qwen3Model,
        cache: Vec<Option<KVCache>>,
    },
    Gemma4 {
        model: Box<Gemma4Model>,
        cfg: gemma4::config::Gemma4Config,
        cache: Vec<Option<Gemma4LayerCache>>,
    },
    Qwen35 {
        model: Box<Qwen35Model>,
        cfg: Qwen35Config,
        cache: Vec<Qwen35LayerCache>,
    },
}

impl Backend {
    fn load(family: &str, model_dir: &Path) -> Result<Self> {
        match family {
            "llama" | "llamaforcausallm" => Ok(Backend::Llama {
                model: load_llama_model(model_dir)?,
                cache: Vec::new(),
            }),
            "qwen3" | "qwen3forcausallm" => Ok(Backend::Qwen3 {
                model: load_qwen3_model(model_dir)?,
                cache: Vec::new(),
            }),
            "gemma4" | "gemma4_text" | "gemma4textmodel" | "gemma4forcausallm" => {
                let cfg = gemma4::loader::get_gemma4_model_args(model_dir)?;
                let cache = make_gemma4_caches(&cfg);
                let model = Box::new(load_gemma4_model(model_dir)?);
                Ok(Backend::Gemma4 { model, cfg, cache })
            }
            "qwen3_5" | "qwen3_5_text" | "qwen3_5forconditionalgeneration" => {
                let cfg = Qwen35Config::from_file(model_dir.join("config.json"))?;
                let cache = make_qwen35_caches(&cfg);
                let (model, _leftover) = load_qwen35_lm(&cfg, model_dir)?;
                Ok(Backend::Qwen35 {
                    model: Box::new(model),
                    cfg,
                    cache,
                })
            }
            other => Err(format!("unsupported model family: {other}").into()),
        }
    }

    fn reset_cache(&mut self) {
        match self {
            Backend::Llama { cache, .. } | Backend::Qwen3 { cache, .. } => cache.clear(),
            Backend::Gemma4 { cfg, cache, .. } => {
                *cache = make_gemma4_caches(cfg);
            }
            Backend::Qwen35 { cfg, cache, .. } => {
                *cache = make_qwen35_caches(cfg);
            }
        }
    }

    fn run_turn(
        &mut self,
        prompt_tokens: &[u32],
        temp: f32,
        max_tokens: usize,
        eos_ids: &[u32],
        interrupted: &Arc<AtomicBool>,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<(Vec<u32>, bool)> {
        let prompt_arr = Array::from(prompt_tokens).index(NewAxis);
        let mut ids: Vec<u32> = Vec::new();
        let mut cancelled = false;

        match self {
            Backend::Llama { model, cache } => {
                let mut gen = LlamaGenerate::<KVCache>::new(model, cache, temp, &prompt_arr);
                for (n, token) in gen.by_ref().enumerate() {
                    if interrupted.load(Ordering::SeqCst) {
                        cancelled = true;
                        break;
                    }
                    let arr = token.map_err(|e| format!("{e:?}"))?;
                    let id = arr.item::<u32>();
                    if eos_ids.contains(&id) || n + 1 >= max_tokens {
                        break;
                    }
                    on_token(id);
                    ids.push(id);
                }
            }
            Backend::Qwen3 { model, cache } => {
                let mut gen = Qwen3Generate::<KVCache>::new(model, cache, temp, &prompt_arr);
                for (n, token) in gen.by_ref().enumerate() {
                    if interrupted.load(Ordering::SeqCst) {
                        cancelled = true;
                        break;
                    }
                    let arr = token.map_err(|e| format!("{e:?}"))?;
                    let id = arr.item::<u32>();
                    if eos_ids.contains(&id) || n + 1 >= max_tokens {
                        break;
                    }
                    on_token(id);
                    ids.push(id);
                }
            }
            Backend::Gemma4 { model, cache, .. } => {
                let mut gen = gemma4::Generate::<Gemma4LayerCache>::new(
                    model.as_mut(),
                    cache,
                    temp,
                    &prompt_arr,
                );
                for (n, token) in gen.by_ref().enumerate() {
                    if interrupted.load(Ordering::SeqCst) {
                        cancelled = true;
                        break;
                    }
                    let arr = token.map_err(|e| format!("{e:?}"))?;
                    let id = arr.item::<u32>();
                    if eos_ids.contains(&id) || n + 1 >= max_tokens {
                        break;
                    }
                    on_token(id);
                    ids.push(id);
                }
            }
            Backend::Qwen35 { model, cache, .. } => {
                let prompt_1d = Array::from(prompt_tokens);
                let stop = Qwen35StopCriteria {
                    max_new_tokens: max_tokens as i32,
                    eos_ids: eos_ids.to_vec(),
                };
                let params = Qwen35SamplingParams { temperature: temp, top_p: None };
                let owned = std::mem::take(cache);
                let mut gen =
                    Qwen35Generate::with_caches(model.as_mut(), prompt_1d, owned, stop, params);
                for (n, token) in gen.by_ref().enumerate() {
                    if interrupted.load(Ordering::SeqCst) {
                        cancelled = true;
                        break;
                    }
                    let id = token.map_err(|e| format!("{e:?}"))?;
                    if eos_ids.contains(&id) || n + 1 >= max_tokens {
                        break;
                    }
                    on_token(id);
                    ids.push(id);
                }
                *cache = gen.into_caches();
            }
        }
        Ok((ids, cancelled))
    }
}

fn main() -> Result<()> {
    let args = parse_args()?;
    eprintln!("[loading {}]", args.model.display());

    let family = detect_family(&args.model)?;
    eprintln!("[model_type = {family}]");
    let mut backend = Backend::load(&family, &args.model)?;

    let tokenizer = load_decode_tokenizer(&args.model)?;
    let eos_ids = load_eos_ids(&args.model, &tokenizer);
    eprintln!("[eos_ids = {eos_ids:?}]");
    let mut history: Vec<(Role, String)> = Vec::new();
    let mut prefix_tokens: Vec<u32> = Vec::new();
    let interrupted = install_sigint_handler();
    let mut rl = rustyline::DefaultEditor::new().map_err(|e| format!("readline init: {e:?}"))?;

    let mut template_kwargs: HashMap<String, serde_json::Value> = HashMap::new();
    if let Some(b) = args.thinking {
        template_kwargs.insert("enable_thinking".into(), serde_json::Value::Bool(b));
        eprintln!("[enable_thinking = {b}]");
    }

    eprintln!(
        "[ready — ↑/↓ prompt history, /exit or Ctrl-D quits, /reset clears chat, \
         Ctrl-C clears the current prompt (or cancels generation)]"
    );

    loop {
        let user = match read_user_message(&mut rl)? {
            InputOutcome::Line(s) if s.trim().is_empty() => continue,
            InputOutcome::Line(s) => s,
            InputOutcome::Cleared => continue,
            InputOutcome::Eof => {
                eprintln!();
                break;
            }
        };
        match user.trim() {
            "/exit" | "/quit" => break,
            "/reset" => {
                history.clear();
                prefix_tokens.clear();
                backend.reset_cache();
                eprintln!("[history cleared]");
                continue;
            }
            _ => {}
        }

        history.push((Role::User, user));

        let full = encode_turn(&args.model, &history, true, &template_kwargs)?;
        if full.len() <= prefix_tokens.len() || full[..prefix_tokens.len()] != prefix_tokens[..] {
            // Template diverged from the cached prefix (e.g. system prompt
            // rebuilt mid-conversation). Reprocess from scratch.
            backend.reset_cache();
            prefix_tokens.clear();
        }
        let new_tokens = &full[prefix_tokens.len()..];
        let n_prompt = new_tokens.len();

        print!("{C_BOT}bot> ");
        std::io::stdout().flush()?;

        let mut last_decoded_len = 0;
        let mut streamed_ids: Vec<u32> = Vec::new();
        interrupted.store(false, Ordering::SeqCst);
        let t_start = std::time::Instant::now();
        let mut t_first: Option<std::time::Instant> = None;

        let mut on_token = |id: u32| {
            if t_first.is_none() {
                t_first = Some(std::time::Instant::now());
            }
            streamed_ids.push(id);
            if streamed_ids.len().is_multiple_of(4) {
                if let Ok(text) = tokenizer.decode(&streamed_ids, true) {
                    let delta = &text[last_decoded_len..];
                    print!("{delta}");
                    let _ = std::io::stdout().flush();
                    last_decoded_len = text.len();
                }
            }
        };
        let (ids, cancelled) = backend.run_turn(
            new_tokens,
            args.temp,
            args.max_tokens,
            &eos_ids,
            &interrupted,
            &mut on_token,
        )?;
        if let Ok(text) = tokenizer.decode(&streamed_ids, true) {
            let delta = &text[last_decoded_len..];
            print!("{delta}");
            let _ = std::io::stdout().flush();
        }
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
            // Cache holds partial decode state; can't extend cleanly.
            history.pop();
            backend.reset_cache();
            prefix_tokens.clear();
            continue;
        }

        let assistant = tokenizer.decode(&ids, true).map_err(|e| format!("{e:?}"))?;
        history.push((Role::Assistant, assistant));
        prefix_tokens = encode_turn(&args.model, &history, false, &template_kwargs)?;
    }

    Ok(())
}
