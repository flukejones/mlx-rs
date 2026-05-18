//! Text generation CLI for llama / qwen3. Detects family from
//! `config.json::model_type` and routes through that family's loader
//! + `Generate` iterator.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

use mlx_lm::cache::KVCache;
use mlx_lm::models::{
    llama::{load_llama_model, sample as llama_sample, Generate as LlamaGenerate},
    qwen3::{load_qwen3_model, sample as qwen3_sample, Generate as Qwen3Generate},
    qwen3_5::{
        generation::{Generate as Qwen35Generate, SamplingParams, StopCriteria},
        weights::load_language_model as load_qwen35_lm,
        ModelConfig as Qwen35Config,
    },
};
use mlx_lm_utils::tokenizer::{
    load_model_chat_template_from_file, ApplyChatTemplateArgs, Conversation, Role, Tokenizer,
};
use mlx_rs::{
    ops::indexing::{IndexOp, NewAxis},
    transforms::eval,
    Array,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

const DEFAULT_MAX_TOKENS: usize = 256;

struct Args {
    model: PathBuf,
    prompt: String,
    temp: f32,
    max_tokens: usize,
}

fn parse_args() -> Result<Args> {
    let mut model = None;
    let mut prompt = None;
    let mut temp = 0.0_f32;
    let mut max_tokens = DEFAULT_MAX_TOKENS;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => model = Some(PathBuf::from(it.next().ok_or("--model needs a value")?)),
            "--prompt" => prompt = Some(it.next().ok_or("--prompt needs a value")?),
            "--temp" => temp = it.next().ok_or("--temp needs a value")?.parse()?,
            "--max-tokens" => {
                max_tokens = it.next().ok_or("--max-tokens needs a value")?.parse()?
            }
            "--help" | "-h" => {
                eprintln!(
                    "usage: generate --model DIR --prompt TEXT [--temp 0.0] [--max-tokens 256]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}").into()),
        }
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        prompt: prompt.ok_or("--prompt is required")?,
        temp,
        max_tokens,
    })
}

#[derive(serde::Deserialize)]
struct ConfigPeek {
    model_type: String,
}

fn detect_family(model_dir: &Path) -> Result<String> {
    let cfg: ConfigPeek = serde_json::from_reader(File::open(model_dir.join("config.json"))?)?;
    Ok(cfg.model_type)
}

fn encode_prompt(model_dir: &Path, prompt: &str) -> Result<Vec<u32>> {
    let mut tok = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| format!("tokenizer load: {e:?}"))?;
    let tmpl_path = model_dir.join("tokenizer_config.json");
    let convs = vec![Conversation { role: Role::User, content: prompt }];
    let model_id = model_dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".to_string());
    let args = ApplyChatTemplateArgs {
        conversations: vec![convs.into()],
        documents: None,
        model_id: &model_id,
        chat_template_id: None,
        add_generation_prompt: Some(true),
        continue_final_message: None,
    };
    match load_model_chat_template_from_file(&tmpl_path)? {
        Some(tmpl) => {
            let encs = tok.apply_chat_template_and_encode(tmpl, args)?;
            Ok(encs.iter().flat_map(|e| e.get_ids()).copied().collect())
        }
        None => {
            let enc = tok.encode(prompt, true).map_err(|e| format!("encode: {e:?}"))?;
            Ok(enc.get_ids().to_vec())
        }
    }
}

fn detok(model_dir: &Path, ids: &[u32]) -> Result<String> {
    let tok = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| format!("tokenizer load: {e:?}"))?;
    tok.decode(ids, true).map_err(|e| format!("decode: {e:?}").into())
}

fn run_qwen3(args: &Args) -> Result<()> {
    let mut model = load_qwen3_model(&args.model)?;
    let prompt_ids = encode_prompt(&args.model, &args.prompt)?;
    let prompt = Array::from(&prompt_ids[..]).index(NewAxis);

    let mut cache: Vec<Option<KVCache>> = Vec::new();
    let gen = Qwen3Generate::<KVCache>::new(&mut model, &mut cache, args.temp, &prompt);

    decode_and_stream(gen, args.max_tokens, &args.model, prompt_ids.len(), qwen3_sample)
}

fn run_llama(args: &Args) -> Result<()> {
    let mut model = load_llama_model(&args.model)?;
    let prompt_ids = encode_prompt(&args.model, &args.prompt)?;
    let prompt = Array::from(&prompt_ids[..]).index(NewAxis);

    let mut cache: Vec<Option<KVCache>> = Vec::new();
    let gen = LlamaGenerate::<KVCache>::new(&mut model, &mut cache, args.temp, &prompt);

    decode_and_stream(gen, args.max_tokens, &args.model, prompt_ids.len(), llama_sample)
}

fn run_qwen3_5(args: &Args) -> Result<()> {
    let cfg = Qwen35Config::from_file(args.model.join("config.json"))?;
    let (mut model, _leftover) = load_qwen35_lm(&cfg, &args.model)?;
    let prompt_ids = encode_prompt(&args.model, &args.prompt)?;
    let prompt = Array::from(&prompt_ids[..]);
    let stop = StopCriteria::from_config(&cfg, args.max_tokens as i32);
    let params = SamplingParams { temperature: args.temp, top_p: None };
    let iter = Qwen35Generate::new(&mut model, &cfg, prompt, stop, params);

    let t_start = Instant::now();
    let mut t_first: Option<Instant> = None;
    let mut ids: Vec<u32> = Vec::with_capacity(args.max_tokens);
    for tok in iter {
        let id = tok?;
        if t_first.is_none() {
            t_first = Some(Instant::now());
        }
        ids.push(id);
    }

    let text = detok(&args.model, &ids)?;
    println!("{text}");

    let t_end = Instant::now();
    let t_first = t_first.unwrap_or(t_end);
    let prefill_s = (t_first - t_start).as_secs_f64();
    let decode_s = (t_end - t_first).as_secs_f64();
    let n_decode = ids.len().saturating_sub(1);
    let n_prompt = prompt_ids.len();
    eprintln!();
    eprintln!("--- speed ---");
    eprintln!(
        "  prefill: {n_prompt} prompt tokens in {prefill_s:.2}s ({:.1} tok/s)",
        n_prompt as f64 / prefill_s.max(1e-6)
    );
    eprintln!(
        "  decode:  {n_decode} tokens in {decode_s:.2}s ({:.1} tok/s)",
        n_decode as f64 / decode_s.max(1e-6)
    );
    Ok(())
}

fn decode_and_stream<I>(
    gen: I,
    max_tokens: usize,
    model_dir: &Path,
    n_prompt: usize,
    _sample_unused: fn(&Array, f32) -> mlx_rs::error::Result<Array>,
) -> Result<()>
where
    I: Iterator<Item = mlx_rs::error::Result<Array>>,
{
    let t_start = Instant::now();
    let mut t_first: Option<Instant> = None;
    let mut tokens = Vec::with_capacity(max_tokens);

    for (tok, n) in gen.zip(0..max_tokens) {
        let tok = tok?;
        tokens.push(tok);
        if n == 0 {
            eval(&tokens)?;
            t_first = Some(Instant::now());
        }
    }
    eval(&tokens)?;

    let ids: Vec<u32> = tokens.iter().map(|t| t.item::<u32>()).collect();
    let text = detok(model_dir, &ids)?;
    println!("{text}");

    let t_end = Instant::now();
    let t_first = t_first.unwrap_or(t_end);
    let prefill_s = (t_first - t_start).as_secs_f64();
    let decode_s = (t_end - t_first).as_secs_f64();
    let n_decode = ids.len().saturating_sub(1);
    eprintln!();
    eprintln!("--- speed ---");
    eprintln!(
        "  prefill: {n_prompt} prompt tokens in {prefill_s:.2}s ({:.1} tok/s)",
        n_prompt as f64 / prefill_s.max(1e-6)
    );
    eprintln!(
        "  decode:  {n_decode} tokens in {decode_s:.2}s ({:.1} tok/s)",
        n_decode as f64 / decode_s.max(1e-6)
    );
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let family = detect_family(&args.model)?;
    eprintln!("[model_type = {family}]");
    match family.as_str() {
        "qwen3" => run_qwen3(&args),
        "llama" => run_llama(&args),
        "qwen3_5_moe_omni" | "qwen3_5_omni" | "qwen3_5" => run_qwen3_5(&args),
        other => Err(format!("unsupported model_type: {other}").into()),
    }
}
