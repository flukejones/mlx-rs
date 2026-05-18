//! Generic text-generation example for llama / qwen3 / gemma4.
//!
//! Detects the model family from `config.json::model_type`, loads the
//! checkpoint via the per-family loader, and runs greedy or
//! temperature-sampled decoding.
//!
//! Examples:
//!   generate --model ~/.cache/mlx-rs-bench/mlx-community/Qwen3-1.7B-4bit \
//!            --prompt "Once upon a time"
//!   generate --model ~/.cache/mlx-rs-bench/mlx-community/Llama-3.2-3B-Instruct-4bit \
//!            --prompt "What is the capital of France?" --temp 0.7
//!   generate --model ~/.cache/mlx-rs-bench/mlx-community/gemma-4-e4b-it-8bit \
//!            --prompt "Write a haiku."
//!   generate --model ~/.cache/mlx-rs-bench/mlx-community/gemma-4-26b-a4b-it-8bit \
//!            --prompt "Explain MoE in two sentences."
//!
//! qwen3_5 (chandra) has a bespoke `LanguageModel` with `Option<&Array>`
//! inputs + multimodal cache — out of scope for this CLI; use the
//! `chandra` example instead.

use std::fs::File;
use std::path::{Path, PathBuf};

use mlx_lm::{
    cache::KVCache,
    models::{
        gemma4::{
            self,
            loader::{load_gemma4_model, make_gemma4_caches, Gemma4LayerCache},
        },
        llama::{load_llama_model, Generate as LlamaGenerate},
        qwen3::{load_qwen3_model, Generate as Qwen3Generate},
        qwen3_5::{
            generation::{Generate as Qwen35Generate, StopCriteria as Qwen35StopCriteria},
            weights::load_language_model as load_qwen35_lm,
            ModelConfig as Qwen35ModelConfig, SamplingParams as Qwen35SamplingParams,
        },
    },
};
use mlx_lm_utils::tokenizer::{
    load_model_chat_template_from_file, load_special_tokens_from_file, ApplyChatTemplateArgs,
    Conversation, Role, Tokenizer,
};
use mlx_rs::{
    ops::indexing::{IndexOp, NewAxis},
    transforms::eval,
    Array,
};

type BoxError = Box<dyn std::error::Error + Send + Sync>;
type Result<T> = std::result::Result<T, BoxError>;

const DEFAULT_MAX_TOKENS: usize = 256;

struct Args {
    model: PathBuf,
    prompt: String,
    temp: f32,
    max_tokens: usize,
    no_chat_template: bool,
    ids: Option<String>,
}

fn parse_args() -> Result<Args> {
    let mut model: Option<PathBuf> = None;
    let mut prompt: Option<String> = None;
    let mut temp: f32 = 0.0;
    let mut max_tokens: usize = DEFAULT_MAX_TOKENS;
    let mut no_chat_template = false;
    let mut ids: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model = Some(PathBuf::from(args.next().ok_or("--model needs a value")?)),
            "--prompt" => prompt = Some(args.next().ok_or("--prompt needs a value")?),
            "--temp" => temp = args.next().ok_or("--temp needs a value")?.parse()?,
            "--max-tokens" | "--max_tokens" => {
                max_tokens = args.next().ok_or("--max-tokens needs a value")?.parse()?
            }
            "--no-chat-template" => no_chat_template = true,
            "--ids" => ids = Some(args.next().ok_or("--ids needs a value")?),
            "-h" | "--help" => {
                eprintln!("usage: generate --model DIR --prompt TEXT [--temp 0.0] [--max-tokens 256] [--no-chat-template]");
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}").into()),
        }
    }

    let prompt_required = ids.is_none();
    Ok(Args {
        model: model.ok_or("--model is required")?,
        prompt: if prompt_required { prompt.ok_or("--prompt is required")? } else { prompt.unwrap_or_default() },
        temp,
        max_tokens,
        no_chat_template,
        ids,
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

/// Parse `eos_token_id` from generation_config.json or config.json.
/// Accepts both a single int and a list of ints.
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

fn build_prompt(model_dir: &Path, user_message: &str, no_chat_template: bool) -> Result<Array> {
    let tok_path = model_dir.join("tokenizer.json");
    let mut tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| format!("{e:?}"))?;

    if no_chat_template {
        let enc = tokenizer
            .encode(user_message, true)
            .map_err(|e| format!("{e:?}"))?;
        let ids: Vec<u32> = enc.get_ids().to_vec();
        return Ok(Array::from(&ids[..]).index(NewAxis));
    }

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
        match load_model_chat_template_from_file(&cfg_path) {
            Ok(Some(t)) => t,
            Ok(None) => {
                let enc = tokenizer
                    .encode(user_message, true)
                    .map_err(|e| format!("{e:?}"))?;
                let ids: Vec<u32> = enc.get_ids().to_vec();
                return Ok(Array::from(&ids[..]).index(NewAxis));
            }
            Err(e) => return Err(format!("load chat template: {e:?}").into()),
        }
    };

    let conv = vec![Conversation {
        role: Role::User,
        content: user_message,
    }];
    let special_tokens = load_special_tokens_from_file(&cfg_path).unwrap_or_default();
    let args = ApplyChatTemplateArgs {
        conversations: vec![conv.into()],
        documents: None,
        model_id: &model_id,
        chat_template_id: None,
        add_generation_prompt: Some(true),
        continue_final_message: None,
        special_tokens,
    };
    let encodings = tokenizer
        .apply_chat_template_and_encode(template, args)
        .map_err(|e| format!("{e:?}"))?;
    let ids: Vec<u32> = encodings
        .iter()
        .flat_map(|enc| enc.get_ids())
        .copied()
        .collect();
    Ok(Array::from(&ids[..]).index(NewAxis))
}

fn load_decode_tokenizer(model_dir: &Path) -> Result<tokenizers::Tokenizer> {
    tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| format!("{e:?}").into())
}

fn run_llama(model_dir: &Path, prompt_tokens: Array, temp: f32, max_tokens: usize) -> Result<()> {
    let n_prompt = prompt_tokens.shape()[1] as usize;
    let mut model = load_llama_model(model_dir)?;
    let mut cache: Vec<Option<KVCache>> = Vec::new();
    let mut generate = LlamaGenerate::<KVCache>::new(&mut model, &mut cache, temp, &prompt_tokens);
    drive_generate_array(&mut generate, model_dir, max_tokens, n_prompt)
}

fn run_qwen3(model_dir: &Path, prompt_tokens: Array, temp: f32, max_tokens: usize) -> Result<()> {
    let n_prompt = prompt_tokens.shape()[1] as usize;
    let mut model = load_qwen3_model(model_dir)?;
    let mut cache: Vec<Option<KVCache>> = Vec::new();
    let mut generate = Qwen3Generate::<KVCache>::new(&mut model, &mut cache, temp, &prompt_tokens);
    drive_generate_array(&mut generate, model_dir, max_tokens, n_prompt)
}

fn run_qwen3_5(model_dir: &Path, prompt_tokens: Array, temp: f32, max_tokens: usize) -> Result<()> {
    let cfg = Qwen35ModelConfig::from_file(model_dir.join("config.json"))?;
    let (mut model, _leftover) = load_qwen35_lm(&cfg, model_dir)?;
    // prompt_tokens arrives shaped [1, S]; Qwen35Generate wants 1-D [S].
    let s = prompt_tokens.shape()[1];
    let prompt_1d = prompt_tokens.reshape(&[s]).map_err(|e| format!("{e:?}"))?;
    let stop = Qwen35StopCriteria::from_config(&cfg, max_tokens as i32);
    let params = Qwen35SamplingParams { temperature: temp, top_p: None };
    let mut gen = Qwen35Generate::new(&mut model, &cfg, prompt_1d, stop, params);
    let tokenizer = load_decode_tokenizer(model_dir)?;
    let eos_ids = load_eos_ids(model_dir);
    let mut ids: Vec<u32> = Vec::new();
    let mut last_decoded_len = 0;
    use std::io::Write;
    let flush_chunk = |ids: &[u32], last_decoded_len: &mut usize, tok: &tokenizers::Tokenizer| {
        if let Ok(text) = tok.decode(ids, true) {
            let delta = &text[*last_decoded_len..];
            print!("{delta}");
            let _ = std::io::stdout().flush();
            *last_decoded_len = text.len();
        }
    };
    let t_start = std::time::Instant::now();
    let mut t_first: Option<std::time::Instant> = None;
    for token in gen.by_ref() {
        let id = token.map_err(|e| format!("{e:?}"))?;
        if t_first.is_none() {
            t_first = Some(std::time::Instant::now());
        }
        if eos_ids.contains(&id) {
            break;
        }
        ids.push(id);
        if ids.len() % 8 == 0 {
            flush_chunk(&ids, &mut last_decoded_len, &tokenizer);
        }
    }
    flush_chunk(&ids, &mut last_decoded_len, &tokenizer);
    println!();
    report_speed(s as usize, ids.len(), t_start, t_first);
    Ok(())
}

fn run_gemma4(model_dir: &Path, prompt_tokens: Array, temp: f32, max_tokens: usize) -> Result<()> {
    let n_prompt = prompt_tokens.shape()[1] as usize;
    let mut model = load_gemma4_model(model_dir)?;
    let cfg = gemma4::loader::get_gemma4_model_args(model_dir)?;
    let mut cache: Vec<Option<Gemma4LayerCache>> = make_gemma4_caches(&cfg);
    let mut generate =
        gemma4::Generate::<Gemma4LayerCache>::new(&mut model, &mut cache, temp, &prompt_tokens);
    drive_generate_array(&mut generate, model_dir, max_tokens, n_prompt)
}

fn drive_generate_array<I>(
    generate: &mut I,
    model_dir: &Path,
    max_tokens: usize,
    prompt_tokens: usize,
) -> Result<()>
where
    I: Iterator<Item = std::result::Result<Array, mlx_rs::error::Exception>>,
{
    let tokenizer = load_decode_tokenizer(model_dir)?;
    let eos_ids = load_eos_ids(model_dir);
    let mut ids: Vec<u32> = Vec::new();
    let mut last_decoded_len = 0;
    use std::io::Write;

    let flush_chunk = |ids: &[u32], last_decoded_len: &mut usize, tok: &tokenizers::Tokenizer| {
        if let Ok(text) = tok.decode(ids, true) {
            let delta = &text[*last_decoded_len..];
            print!("{delta}");
            let _ = std::io::stdout().flush();
            *last_decoded_len = text.len();
        }
    };

    let t_start = std::time::Instant::now();
    let mut t_first: Option<std::time::Instant> = None;
    for (n, token) in generate.enumerate() {
        if n >= max_tokens {
            break;
        }
        let token = token?;
        eval([&token]).map_err(|e| format!("{e:?}"))?;
        if t_first.is_none() {
            t_first = Some(std::time::Instant::now());
        }
        let id = token.item::<i32>() as u32;
        if eos_ids.contains(&id) {
            break;
        }
        ids.push(id);
        if ids.len() % 8 == 0 {
            flush_chunk(&ids, &mut last_decoded_len, &tokenizer);
        }
    }
    flush_chunk(&ids, &mut last_decoded_len, &tokenizer);
    println!();
    let n_gen = ids.len();
    report_speed(prompt_tokens, n_gen, t_start, t_first);
    Ok(())
}

fn report_speed(
    prompt_tokens: usize,
    generated: usize,
    t_start: std::time::Instant,
    t_first: Option<std::time::Instant>,
) {
    let t_end = std::time::Instant::now();
    match t_first {
        Some(t_first) => {
            let prefill_s = (t_first - t_start).as_secs_f64();
            let decode_s = (t_end - t_first).as_secs_f64();
            let prefill_tps = if prefill_s > 0.0 {
                prompt_tokens as f64 / prefill_s
            } else {
                f64::INFINITY
            };
            // Prefill processes the entire prompt and emits token #1 of
            // the generation. Decode covers the remaining steps.
            let decode_steps = generated.saturating_sub(1);
            let decode_tps = if decode_s > 0.0 && decode_steps > 0 {
                decode_steps as f64 / decode_s
            } else {
                0.0
            };
            eprintln!(
                "\n--- speed ---\n  prefill: {prompt_tokens} prompt tokens in {prefill_s:.2}s ({prefill_tps:.1} tok/s)\n  decode:  {decode_steps} tokens in {decode_s:.2}s ({decode_tps:.1} tok/s)\n  total:   {generated} tokens generated",
            );
        }
        None => eprintln!("[generated 0 tokens]"),
    }
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let family = detect_family(&args.model)?;
    eprintln!("[model_type = {family}]");

    let prompt = if let Some(ids_csv) = args.ids.as_deref() {
        let ids: Vec<u32> = ids_csv
            .split(',')
            .map(|s| s.trim().parse::<u32>())
            .collect::<std::result::Result<_, _>>()?;
        Array::from(&ids[..]).index(NewAxis)
    } else {
        build_prompt(&args.model, &args.prompt, args.no_chat_template)?
    };

    match family.as_str() {
        "llama" | "llamaforcausallm" => {
            run_llama(&args.model, prompt, args.temp, args.max_tokens)
        }
        "qwen3" | "qwen3forcausallm" => {
            run_qwen3(&args.model, prompt, args.temp, args.max_tokens)
        }
        "gemma4" | "gemma4_text" | "gemma4textmodel" | "gemma4forcausallm" => {
            run_gemma4(&args.model, prompt, args.temp, args.max_tokens)
        }
        "qwen3_5" | "qwen3_5_text" | "qwen3_5forconditionalgeneration" => {
            run_qwen3_5(&args.model, prompt, args.temp, args.max_tokens)
        }
        other => Err(format!("unsupported model family: {other}").into()),
    }
}
