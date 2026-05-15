//! Chandra-OCR-2 / Qwen3.5 inference CLI + OpenAI-compatible HTTP server.
//!
//! Supports text-only generation against any Qwen3.5 mlx-format checkpoint,
//! plus the full vision path (image -> ViT -> multimodal stitch -> hybrid
//! decoder) against `chandra-ocr-2-mlx-q8` and compatible checkpoints.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use image::DynamicImage;
use mlx_lm::chat_template::{ChatMessage, ChatTemplate};
use mlx_lm::models::qwen3_5::{
    config::ModelConfig,
    generation::{Generate, SamplingParams, StopCriteria},
    image_processor::{ProcessedImage, Qwen35ImageProcessor},
    multimodal::{get_rope_index_single_batch, merge_input_ids_with_image_features, pack_position_ids},
    vision::VisionModel,
    weights::load_full_model,
    LanguageModel,
};
use mlx_rs::{module::Module, Array};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

mod server;

#[derive(Parser)]
#[command(about = "Chandra-OCR-2 / Qwen3.5 hybrid SSM + full-attention inference")]
struct Cli {
    /// Path to the local MLX-format checkpoint directory.
    #[clap(long)]
    model: PathBuf,

    /// Prompt to run one-shot. Mutually exclusive with --serve.
    #[clap(long)]
    prompt: Option<String>,

    /// Optional image attached to the prompt. The image processor renders it
    /// into the vision tower input; the vision features replace the
    /// `<|image_pad|>` placeholders inserted by the chat template.
    #[clap(long)]
    image: Option<PathBuf>,

    /// Maximum new tokens to generate.
    #[clap(long, default_value_t = 512)]
    max_tokens: i32,

    /// Sampling temperature. `0.0` selects greedy.
    #[clap(long, default_value_t = 0.0)]
    temperature: f32,

    /// Top-p (nucleus) sampling. Ignored when `temperature == 0`.
    #[clap(long)]
    top_p: Option<f32>,

    /// Seed for the PRNG (used when `temperature > 0`).
    #[clap(long)]
    seed: Option<u64>,

    /// Start an OpenAI-compatible HTTP server instead of running one-shot.
    #[clap(long)]
    serve: bool,

    /// Port to bind when `--serve` is set.
    #[clap(long, default_value_t = 8088)]
    port: u16,
}

/// Bundle owned by the request handler.
pub struct AppState {
    pub model: Mutex<LanguageModel>,
    pub vision: Mutex<VisionModel>,
    pub image_processor: Qwen35ImageProcessor,
    pub tokenizer: Tokenizer,
    pub cfg: ModelConfig,
    pub chat_template: ChatTemplate,
}

impl AppState {
    fn from_dir(dir: &Path) -> Result<Self> {
        let cfg_path = dir.join("config.json");
        let cfg = ModelConfig::from_file(&cfg_path)
            .with_context(|| format!("parsing {}", cfg_path.display()))?;
        let (model, vision, leftover) = load_full_model(&cfg, dir)
            .with_context(|| format!("loading weights from {}", dir.display()))?;
        if !leftover.is_empty() {
            tracing::warn!(
                "{} weight keys left after full-model load:",
                leftover.len()
            );
            for k in leftover.iter().take(10) {
                tracing::warn!("  {k}");
            }
        }
        let image_processor = Qwen35ImageProcessor::from_dir(dir)
            .with_context(|| "loading preprocessor_config.json")?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("loading tokenizer.json: {e}"))?;
        let chat_template = ChatTemplate::from_dir(dir)
            .with_context(|| "loading chat template")?;
        Ok(Self {
            model: Mutex::new(model),
            vision: Mutex::new(vision),
            image_processor,
            tokenizer,
            cfg,
            chat_template,
        })
    }

    /// Render a user prompt through the checkpoint's Jinja chat template.
    /// When `image_present` is true, the user message uses the parts-list
    /// content form with a single image placeholder; the template emits
    /// `<|image_pad|>` tokens that the vision tower replaces at runtime.
    pub fn render_chat_prompt(
        &self,
        user_message: &str,
        image_present: bool,
    ) -> Result<String> {
        let msg = if image_present {
            ChatMessage::user_with_image(user_message)
        } else {
            ChatMessage::user(user_message)
        };
        Ok(self.chat_template.render(&[msg], true)?)
    }

    /// Run text-only generation.
    pub fn generate_text(
        &self,
        prompt: &str,
        max_tokens: i32,
        temperature: f32,
        top_p: Option<f32>,
    ) -> Result<GenerationResult> {
        self.stream_text(prompt, max_tokens, temperature, top_p, |_, _| {
            std::ops::ControlFlow::Continue(())
        })
    }

    /// Streaming variant of [`generate_text`]. `on_token(token_id, delta_text)`
    /// is called for every newly-produced token (after re-decoding to handle
    /// BPE merges). Return `ControlFlow::Break(())` to stop early.
    pub fn stream_text<F>(
        &self,
        prompt: &str,
        max_tokens: i32,
        temperature: f32,
        top_p: Option<f32>,
        mut on_token: F,
    ) -> Result<GenerationResult>
    where
        F: FnMut(u32, &str) -> std::ops::ControlFlow<()>,
    {
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("tokenizer encode: {e}"))?;
        let ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
        let prompt_len = ids.len();
        let prompt_array = Array::from_slice(&ids, &[ids.len() as i32]);
        let stop = StopCriteria::from_config(&self.cfg, max_tokens);
        let params = SamplingParams { temperature, top_p };
        let (new_ids, finish_reason) = {
            let mut model = self.model.lock().expect("model lock");
            let gen = Generate::new(&mut model, &self.cfg, prompt_array, stop.clone(), params);
            self.run_stream(gen, &stop, &mut on_token)?
        };
        let text = self
            .tokenizer
            .decode(&new_ids, true)
            .map_err(|e| anyhow!("tokenizer decode: {e}"))?;
        Ok(GenerationResult {
            text,
            prompt_tokens: prompt_len as i32,
            completion_tokens: new_ids.len() as i32,
            finish_reason,
        })
    }

    fn run_stream<F>(
        &self,
        gen: Generate<'_>,
        stop: &StopCriteria,
        on_token: &mut F,
    ) -> Result<(Vec<u32>, FinishReason)>
    where
        F: FnMut(u32, &str) -> std::ops::ControlFlow<()>,
    {
        let mut new_ids: Vec<u32> = Vec::new();
        let mut decoded_so_far = String::new();
        let mut finish_reason = FinishReason::Length;
        for r in gen {
            let tok = r.map_err(|e| anyhow!("generation step: {e}"))?;
            if stop.eos_ids.contains(&tok) {
                finish_reason = FinishReason::Stop;
                break;
            }
            new_ids.push(tok);
            // BPE merges mean the latest token's text only stabilises after
            // a re-decode of the whole accumulated id list. Compute the
            // delta against `decoded_so_far`.
            let full = self
                .tokenizer
                .decode(&new_ids, true)
                .map_err(|e| anyhow!("tokenizer decode: {e}"))?;
            let delta = full
                .strip_prefix(&decoded_so_far)
                .unwrap_or(full.as_str())
                .to_string();
            decoded_so_far = full;
            if matches!(on_token(tok, &delta), std::ops::ControlFlow::Break(())) {
                break;
            }
        }
        Ok((new_ids, finish_reason))
    }

    /// Run multimodal generation. The image is pre-processed, run through
    /// the vision tower, and its features are scattered into the embedding
    /// sequence at every `<|image_pad|>` slot. `position_ids` are derived
    /// from `get_rope_index_single_batch`.
    pub fn generate_multimodal(
        &self,
        prompt: &str,
        image: DynamicImage,
        max_tokens: i32,
        temperature: f32,
        top_p: Option<f32>,
    ) -> Result<GenerationResult> {
        self.stream_multimodal(prompt, image, max_tokens, temperature, top_p, |_, _| {
            std::ops::ControlFlow::Continue(())
        })
    }

    /// Streaming variant of [`generate_multimodal`].
    pub fn stream_multimodal<F>(
        &self,
        prompt: &str,
        image: DynamicImage,
        max_tokens: i32,
        temperature: f32,
        top_p: Option<f32>,
        mut on_token: F,
    ) -> Result<GenerationResult>
    where
        F: FnMut(u32, &str) -> std::ops::ControlFlow<()>,
    {
        // Run the image processor.
        let ProcessedImage {
            pixel_values,
            grid_thw,
            feature_dim,
        } = self.image_processor.preprocess_image(image)?;
        let num_patches = (pixel_values.len() / feature_dim as usize) as i32;
        let pixel_array = Array::from_slice(&pixel_values, &[num_patches, feature_dim]);
        let merge = self.cfg.vision_config.spatial_merge_size;
        let expected_image_tokens = (grid_thw[0] as usize)
            * ((grid_thw[1] / merge) as usize)
            * ((grid_thw[2] / merge) as usize);

        // The Jinja chat template emits a single `<|image_pad|>` placeholder
        // per image; mlx-vlm's `Qwen3VLProcessor` post-processes that into
        // `expected_image_tokens` copies before tokenising. Do the same.
        let image_pad_str = "<|image_pad|>";
        let pad_replacement = image_pad_str.repeat(expected_image_tokens);
        let expanded_prompt = prompt.replacen(image_pad_str, &pad_replacement, 1);

        let enc = self
            .tokenizer
            .encode(expanded_prompt.as_str(), false)
            .map_err(|e| anyhow!("tokenizer encode: {e}"))?;
        let ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();

        // Sanity-check the expansion landed the right number of image-pad
        // tokens after tokenisation.
        let n_image_pad = ids
            .iter()
            .filter(|&&t| (t as u32) == self.cfg.image_token_id)
            .count();
        if n_image_pad != expected_image_tokens {
            return Err(anyhow!(
                "rendered prompt has {n_image_pad} image-pad tokens but the image \
                 produces {expected_image_tokens} merged patches"
            ));
        }

        // Run the vision tower.
        let image_features = {
            let mut vm = self.vision.lock().expect("vision lock");
            vm.forward(&pixel_array, &[grid_thw])
                .map_err(|e| anyhow!("vision forward: {e}"))?
        };

        // Stitch features into embeddings.
        let stop = StopCriteria::from_config(&self.cfg, max_tokens);
        let params = SamplingParams { temperature, top_p };
        let prompt_len = ids.len();
        let s = prompt_len as i32;
        let input_ids = Array::from_slice(&ids, &[1, s]);

        let (new_ids, finish_reason) = {
            let mut lm = self.model.lock().expect("model lock");

            let embeds = lm
                .model
                .embed_tokens
                .forward(&input_ids)
                .map_err(|e| anyhow!("embed_tokens: {e}"))?;
            let stitched = merge_input_ids_with_image_features(
                &image_features,
                &embeds,
                &input_ids,
                self.cfg.image_token_id,
                self.cfg.video_token_id,
            )
            .map_err(|e| anyhow!("merge_input_ids: {e}"))?;

            let (t_pos, h_pos, w_pos, rope_delta) = get_rope_index_single_batch(
                &ids,
                &[grid_thw],
                merge,
                self.cfg.image_token_id,
                self.cfg.video_token_id,
                self.cfg.vision_start_token_id,
            )
            .map_err(|e| anyhow!("get_rope_index: {e}"))?;
            let position_ids = pack_position_ids(&t_pos, &h_pos, &w_pos)
                .map_err(|e| anyhow!("pack_position_ids: {e}"))?;

            let gen = Generate::with_inputs_embeds(
                &mut lm,
                &self.cfg,
                stitched,
                position_ids,
                rope_delta,
                stop.clone(),
                params,
            );
            self.run_stream(gen, &stop, &mut on_token)?
        };

        let text = self
            .tokenizer
            .decode(&new_ids, true)
            .map_err(|e| anyhow!("tokenizer decode: {e}"))?;
        Ok(GenerationResult {
            text,
            prompt_tokens: prompt_len as i32,
            completion_tokens: new_ids.len() as i32,
            finish_reason,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GenerationResult {
    pub text: String,
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let cli = Cli::parse();

    if let Some(seed) = cli.seed {
        mlx_rs::random::seed(seed)?;
    }

    if cli.serve {
        return server::run(cli);
    }

    let prompt_text = cli
        .prompt
        .as_deref()
        .ok_or_else(|| anyhow!("--prompt is required when --serve is not set"))?;
    let state = AppState::from_dir(&cli.model)?;

    let result = if let Some(image_path) = &cli.image {
        let img = image::open(image_path)
            .with_context(|| format!("opening image {}", image_path.display()))?;
        let rendered = state.render_chat_prompt(prompt_text, true)?;
        tracing::debug!("rendered prompt: {rendered:?}");
        state.generate_multimodal(
            &rendered,
            img,
            cli.max_tokens,
            cli.temperature,
            cli.top_p,
        )?
    } else {
        let rendered = state.render_chat_prompt(prompt_text, false)?;
        tracing::debug!("rendered prompt: {rendered:?}");
        state.generate_text(&rendered, cli.max_tokens, cli.temperature, cli.top_p)?
    };

    println!("{}", result.text);
    eprintln!(
        "tokens: prompt={} completion={} finish={:?}",
        result.prompt_tokens, result.completion_tokens, result.finish_reason
    );
    Ok(())
}
