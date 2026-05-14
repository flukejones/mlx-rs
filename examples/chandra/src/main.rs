//! Chandra-OCR-2 / Qwen3.5 inference CLI + OpenAI-compatible HTTP server.
//!
//! Supports text-only generation against any Qwen3.5 mlx-format checkpoint,
//! plus the full vision path (image -> ViT -> multimodal stitch -> hybrid
//! decoder) against `chandra-ocr-2-mlx-q8` and compatible checkpoints.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use argh::FromArgs;

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T, E = BoxError> = std::result::Result<T, E>;

#[macro_export]
macro_rules! err {
    ($($arg:tt)*) => {
        crate::BoxError::from(format!($($arg)*))
    };
}
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

#[derive(FromArgs)]
/// Chandra-OCR-2 / Qwen3.5 hybrid SSM + full-attention inference.
pub struct Cli {
    /// path to the local MLX-format checkpoint directory
    #[argh(option)]
    pub model: PathBuf,

    /// prompt to run one-shot; defaults to the chandra OCR prompt when
    /// --image is supplied; mutually exclusive with --serve
    #[argh(option)]
    pub prompt: Option<String>,

    /// optional image attached to the prompt
    #[argh(option)]
    pub image: Option<PathBuf>,

    /// maximum new tokens to generate (default 4096; OCR pages can run
    /// long, especially layout-heavy documents)
    #[argh(option, default = "4096")]
    pub max_tokens: i32,

    /// sampling temperature; 0.0 selects greedy (default 0.0)
    #[argh(option, default = "0.0")]
    pub temperature: f32,

    /// top-p (nucleus) sampling; ignored when temperature == 0.
    /// Override the per-mode default (OCR uses 0.1 to match the chandra
    /// upstream baseline; text-only uses no top_p).
    #[argh(option)]
    pub top_p: Option<f32>,

    /// seed for the PRNG (used when temperature > 0)
    #[argh(option)]
    pub seed: Option<u64>,

    /// start an OpenAI-compatible HTTP server instead of running one-shot
    #[argh(switch)]
    pub serve: bool,

    /// port to bind when --serve is set (default 8088)
    #[argh(option, default = "8088")]
    pub port: u16,
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
            .map_err(|e| err!("parsing {}: {e}", cfg_path.display()))?;
        let (model, vision, leftover) = load_full_model(&cfg, dir)
            .map_err(|e| err!("loading weights from {}: {e}", dir.display()))?;
        if !leftover.is_empty() {
            eprintln!(
                "warn: {} weight keys left after full-model load:",
                leftover.len()
            );
            for k in leftover.iter().take(10) {
                eprintln!("warn:   {k}");
            }
        }
        let image_processor = Qwen35ImageProcessor::from_dir(dir)
            .map_err(|e| err!("loading preprocessor_config.json: {e}"))?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| err!("loading tokenizer.json: {e}"))?;
        let chat_template = ChatTemplate::from_dir(dir)
            .map_err(|e| err!("loading chat template: {e}"))?;
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
            .map_err(|e| err!("tokenizer encode: {e}"))?;
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
            .map_err(|e| err!("tokenizer decode: {e}"))?;
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
            let tok = r.map_err(|e| err!("generation step: {e}"))?;
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
                .map_err(|e| err!("tokenizer decode: {e}"))?;
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
            .map_err(|e| err!("tokenizer encode: {e}"))?;
        let ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();

        // Sanity-check the expansion landed the right number of image-pad
        // tokens after tokenisation.
        let n_image_pad = ids
            .iter()
            .filter(|&&t| (t as u32) == self.cfg.image_token_id)
            .count();
        if n_image_pad != expected_image_tokens {
            return Err(err!(
                "rendered prompt has {n_image_pad} image-pad tokens but the image \
                 produces {expected_image_tokens} merged patches"
            ));
        }

        // Run the vision tower.
        let image_features = {
            let mut vm = self.vision.lock().expect("vision lock");
            vm.forward(&pixel_array, &[grid_thw])
                .map_err(|e| err!("vision forward: {e}"))?
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
                .map_err(|e| err!("embed_tokens: {e}"))?;
            let stitched = merge_input_ids_with_image_features(
                &image_features,
                &embeds,
                &input_ids,
                self.cfg.image_token_id,
                self.cfg.video_token_id,
            )
            .map_err(|e| err!("merge_input_ids: {e}"))?;

            let (t_pos, h_pos, w_pos, rope_delta) = get_rope_index_single_batch(
                &ids,
                &[grid_thw],
                merge,
                self.cfg.image_token_id,
                self.cfg.video_token_id,
                self.cfg.vision_start_token_id,
            )
            .map_err(|e| err!("get_rope_index: {e}"))?;
            let position_ids = pack_position_ids(&t_pos, &h_pos, &w_pos)
                .map_err(|e| err!("pack_position_ids: {e}"))?;

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
            .map_err(|e| err!("tokenizer decode: {e}"))?;
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

/// Chandra OCR prompt (verbatim from upstream `chandra/model/vllm.py`).
/// Used as the default prompt when `--image` is supplied without `--prompt`.
const CHANDRA_OCR_PROMPT: &str = "OCR this image to HTML, arranged as layout blocks.  Each layout block should be a div with the data-bbox attribute representing the bounding box of the block in x0 y0 x1 y1 format.  Bboxes are normalized 0-1000. The data-label attribute is the label for the block.\n\nUse the following labels:\n- Caption\n- Footnote\n- Equation-Block\n- List-Group\n- Page-Header\n- Page-Footer\n- Image\n- Section-Header\n- Table\n- Text\n- Title\n- Form\n- Equation\n- Handwriting\n\nDo not output formatting tags around the divs.  Always use a div with data-bbox/data-label attributes.";

fn main() -> Result<()> {
    let cli: Cli = argh::from_env();

    if let Some(seed) = cli.seed {
        mlx_rs::random::seed(seed)?;
    }

    if cli.serve {
        return server::run(cli);
    }

    let has_image = cli.image.is_some();
    let prompt_text = match cli.prompt.as_deref() {
        Some(s) => s,
        None if has_image => CHANDRA_OCR_PROMPT,
        None => return Err(err!("--prompt is required when --image is not set")),
    };

    // OCR baseline: top_p = 0.1 when not overridden, matching upstream
    // chandra. Text-only stays None unless caller passes it.
    let effective_top_p = cli.top_p.or(if has_image { Some(0.1) } else { None });

    let state = AppState::from_dir(&cli.model)?;

    let result = if let Some(image_path) = &cli.image {
        let img = image::open(image_path)
            .map_err(|e| err!("opening image {}: {e}", image_path.display()))?;
        let rendered = state.render_chat_prompt(prompt_text, true)?;
        eprintln!("debug: rendered prompt: {rendered:?}");
        state.generate_multimodal(
            &rendered,
            img,
            cli.max_tokens,
            cli.temperature,
            effective_top_p,
        )?
    } else {
        let rendered = state.render_chat_prompt(prompt_text, false)?;
        eprintln!("debug: rendered prompt: {rendered:?}");
        state.generate_text(&rendered, cli.max_tokens, cli.temperature, effective_top_p)?
    };

    println!("{}", result.text);
    eprintln!(
        "tokens: prompt={} completion={} finish={:?}",
        result.prompt_tokens, result.completion_tokens, result.finish_reason
    );
    Ok(())
}
