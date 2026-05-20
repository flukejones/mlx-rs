//! Qwen3.5 vision-language [`crate::LanguageModel`] + processor.
//!
//! The VLM path wraps the [`Qwen35DenseAdapter`] dense decoder, adds
//! the vision tower, and routes `prepare()` through a multimodal
//! prefill: pre-process images → vision tower → stitch features into
//! the prompt embedding sequence → call
//! [`Qwen35DenseAdapter::prefill_embeds`].
//!
//! [`Qwen35Processor`] is the matching `UserInputProcessor`: renders
//! the chat template (with image placeholders), expands
//! `<|image_pad|>` to one-per-merged-patch, tokenises, and
//! preprocesses every image (or accepts the
//! [`crate::Image::Pixels`] bypass).

use std::path::Path;

use mlx_rs::{module::Module, Array};

use crate::adapters::qwen3_5::Qwen35DenseAdapter;
use crate::adapters::LoadedContext;
use crate::chat_template::{ChatMessage, ChatTemplate, ContentPart, MessageContent};
use crate::error::Error;
use crate::language_model::{LanguageModel, UserInputProcessor};
use crate::lm_input::{LMInput, LMOutput, PrepareResult, ProcessedImage, Text};
use crate::loader::load_tokenizer;
use crate::models::qwen3_5::config::ModelConfig;
use crate::models::qwen3_5::image_processor::{
    ProcessedImage as VlmRawImage, Qwen35ImageProcessor,
};
use crate::models::qwen3_5::multimodal::{
    get_rope_index_single_batch, merge_input_ids_with_image_features, pack_position_ids,
};
use crate::models::qwen3_5::vision::VisionModel;
use crate::models::qwen3_5::weights::load_full_model;
use crate::user_input::{Image, Prompt, UserInput};

const IMAGE_PAD_STR: &str = "<|image_pad|>";

/// Adapter for qwen3_5 VLM checkpoints (Qwen3.5-VL, Qwen3.6-VL,
/// chandra). Owns the text decoder + the vision tower + the
/// preprocessed-image staging state.
pub(crate) struct Qwen35VlmAdapter {
    dense: Qwen35DenseAdapter,
    vision: VisionModel,
}

impl Qwen35VlmAdapter {
    pub(crate) fn new(dense: Qwen35DenseAdapter, vision: VisionModel) -> Self {
        Self { dense, vision }
    }
}

impl LanguageModel for Qwen35VlmAdapter {
    fn reset(&mut self) {
        self.dense.reset();
    }

    fn prepare(&mut self, input: LMInput) -> Result<PrepareResult, Error> {
        debug_assert!(input.audio.is_none());
        debug_assert!(input.video.is_none());

        let Some(image) = input.image else {
            // Text-only request against a VLM checkpoint: defer to
            // the dense path. Same as if the processor produced no
            // image (no <|image_pad|> in the prompt).
            return self.dense.prepare(LMInput {
                text: input.text,
                image: None,
                audio: None,
                video: None,
            });
        };

        // Vision tower runs over the per-image patches.
        let image_features = self.vision.forward(&image.pixels, image.grids.as_slice())?;

        // Embed the input ids into the model's embedding space.
        let input_ids = input.text.tokens;
        let embeds = match &mut self.dense.model.model.embed_tokens {
            mlx_rs::quantization::MaybeQuantized::Original(e) => e.forward(&input_ids)?,
            mlx_rs::quantization::MaybeQuantized::Quantized(q) => q.forward(&input_ids)?,
        };

        let cfg = &self.dense.cfg;
        let stitched = merge_input_ids_with_image_features(
            &image_features,
            &embeds,
            &input_ids,
            cfg.image_token_id,
            cfg.video_token_id,
        )?;

        // Build [3, 1, S] mrope position ids + per-image rope_delta.
        // `get_rope_index_single_batch` works on host-side `&[i32]`,
        // so flatten the input ids first.
        let s = input_ids.shape()[1];
        let host_ids: Vec<i32> = input_ids.reshape(&[s])?.as_slice::<i32>().to_vec();
        let merge = cfg.vision_config.spatial_merge_size;
        let (t_pos, h_pos, w_pos, rope_delta) = get_rope_index_single_batch(
            &host_ids,
            image.grids.as_slice(),
            merge,
            cfg.image_token_id,
            cfg.video_token_id,
            cfg.vision_start_token_id,
        )?;
        let position_ids = pack_position_ids(&t_pos, &h_pos, &w_pos)?;

        let logits = self
            .dense
            .prefill_embeds(stitched, position_ids, rope_delta)?;
        Ok(PrepareResult::Logits(logits))
    }

    fn step(&mut self, last_token: i32) -> Result<LMOutput, Error> {
        self.dense.step(last_token)
    }

    fn vocab_size(&self) -> i32 {
        self.dense.vocab_size()
    }
}

/// The qwen3_5 `UserInputProcessor`. Renders the chat template,
/// expands image-pad placeholders to the per-patch count, tokenises,
/// and runs image preprocessing (or validates the [`Image::Pixels`]
/// bypass shape).
pub(crate) struct Qwen35Processor {
    tokenizer: tokenizers::Tokenizer,
    chat_template: ChatTemplate,
    image_processor: Qwen35ImageProcessor,
    cfg: ModelConfig,
}

impl Qwen35Processor {
    pub(crate) fn new(
        tokenizer: tokenizers::Tokenizer,
        chat_template: ChatTemplate,
        image_processor: Qwen35ImageProcessor,
        cfg: ModelConfig,
    ) -> Self {
        Self {
            tokenizer,
            chat_template,
            image_processor,
            cfg,
        }
    }
}

impl UserInputProcessor for Qwen35Processor {
    fn family(&self) -> &'static str {
        "qwen3_5"
    }

    fn prepare(&mut self, input: UserInput) -> Result<LMInput, Error> {
        if !input.audios.is_empty() {
            return Err(Error::ModalityUnsupported {
                family: "qwen3_5",
                modality: "audio",
            });
        }
        if !input.videos.is_empty() {
            return Err(Error::ModalityUnsupported {
                family: "qwen3_5",
                modality: "video",
            });
        }

        let merge = self.cfg.vision_config.spatial_merge_size;
        let mut grids: Vec<[i32; 3]> = Vec::with_capacity(input.images.len());
        let mut pixel_arrays: Vec<Array> = Vec::with_capacity(input.images.len());
        let mut expected_pad_total = 0_usize;

        for image in input.images {
            let (array, grid) = match image {
                Image::Decoded(img) => {
                    let processed = self.image_processor.preprocess_image(img)?;
                    pixels_array_from(processed)
                }
                Image::Pixels { array, grid } => {
                    validate_bypass_geometry(&array, grid, merge, &self.image_processor)?;
                    (array, grid)
                }
            };
            let expected =
                (grid[0] as usize) * ((grid[1] / merge) as usize) * ((grid[2] / merge) as usize);
            expected_pad_total += expected;
            grids.push(grid);
            pixel_arrays.push(array);
        }

        // Render chat template. Images route to the parts-list form
        // (one ContentPart::Image per attached image followed by the
        // text part); plain text goes through verbatim.
        let prompt_text = render_prompt(&self.chat_template, input.prompt, grids.len())?;

        // Per-image, expand the single `<|image_pad|>` placeholder
        // emitted by the template into one-per-merged-patch (matches
        // mlx-vlm's `Qwen3VLProcessor` post-step).
        let mut expanded = prompt_text;
        for grid in &grids {
            let expected =
                (grid[0] as usize) * ((grid[1] / merge) as usize) * ((grid[2] / merge) as usize);
            let replacement = IMAGE_PAD_STR.repeat(expected);
            expanded = expanded.replacen(IMAGE_PAD_STR, &replacement, 1);
        }

        let enc = self
            .tokenizer
            .encode(expanded.as_str(), false)
            .map_err(|e| Error::Other(format!("tokenizer encode: {e}").into()))?;
        let ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();

        // Sanity-check that the template + tokenise round-trip
        // produced the expected number of image-pad tokens.
        let observed = ids
            .iter()
            .filter(|&&t| (t as u32) == self.cfg.image_token_id)
            .count();
        if observed != expected_pad_total {
            return Err(Error::Shape(format!(
                "qwen3_5 vlm: rendered prompt has {observed} image-pad tokens \
                 but {} image(s) expand to {expected_pad_total} merged patches",
                grids.len()
            )));
        }

        let s = ids.len() as i32;
        let tokens = Array::from_slice(&ids, &[1, s]);

        let image = if grids.is_empty() {
            None
        } else {
            // Concatenate per-image pixel arrays along the patch axis
            // so the vision tower sees one `[total_patches, feature_dim]`
            // input. `vision.forward` slices them back apart using
            // `grids`.
            let pixels = concat_patches(&pixel_arrays)?;
            Some(ProcessedImage { pixels, grids })
        };

        Ok(LMInput {
            text: Text { tokens, mask: None },
            image,
            audio: None,
            video: None,
        })
    }

    fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| Error::Other(format!("tokenizer decode: {e}").into()))
    }
}

/// Convert a CPU-side [`VlmRawImage`] into the `(array, grid)` pair
/// the LMInput pipeline consumes.
fn pixels_array_from(processed: VlmRawImage) -> (Array, [i32; 3]) {
    let VlmRawImage {
        pixel_values,
        grid_thw,
        feature_dim,
    } = processed;
    let num_patches = (pixel_values.len() / feature_dim as usize) as i32;
    let array = Array::from_slice(&pixel_values, &[num_patches, feature_dim]);
    (array, grid_thw)
}

/// Validate that a caller-supplied `Image::Pixels` array matches the
/// shape the processor would have produced for the same grid. Catches
/// the most common bypass-path mistakes (wrong feature_dim, patch
/// count that doesn't equal `t*h*w`) up front instead of letting the
/// vision tower crash with a shape error.
fn validate_bypass_geometry(
    array: &Array,
    grid: [i32; 3],
    _merge: i32,
    processor: &Qwen35ImageProcessor,
) -> Result<(), Error> {
    let shape = array.shape();
    if shape.len() != 2 {
        return Err(Error::Shape(format!(
            "Image::Pixels: array must be 2-D [num_patches, feature_dim], got {shape:?}"
        )));
    }
    let expected_patches = grid[0] * grid[1] * grid[2];
    if shape[0] != expected_patches {
        return Err(Error::Shape(format!(
            "Image::Pixels: array.shape[0] = {} but grid t*h*w = {}*{}*{} = {}",
            shape[0], grid[0], grid[1], grid[2], expected_patches
        )));
    }
    let cfg = &processor.config;
    let expected_dim = cfg.patch_size * cfg.patch_size * cfg.temporal_patch_size * 3;
    if shape[1] != expected_dim {
        return Err(Error::Shape(format!(
            "Image::Pixels: array.shape[1] = {} but processor expects feature_dim = {}",
            shape[1], expected_dim
        )));
    }
    Ok(())
}

/// Concatenate per-image patch tensors along axis 0.
fn concat_patches(arrays: &[Array]) -> Result<Array, Error> {
    if arrays.len() == 1 {
        return Ok(arrays[0].clone());
    }
    mlx_rs::ops::concatenate_axis(arrays, 0).map_err(Error::Exception)
}

/// Render the chat template against `prompt`. When `num_images > 0`,
/// the prompt is wrapped in the parts-list form with one image
/// placeholder per attached image, matching the template's
/// expectations.
fn render_prompt(
    template: &ChatTemplate,
    prompt: Prompt,
    num_images: usize,
) -> Result<String, Error> {
    match prompt {
        Prompt::Text(text) => {
            if num_images == 0 {
                let msg = ChatMessage::user(text);
                template.render(&[msg], true)
            } else {
                let mut parts: Vec<ContentPart> =
                    (0..num_images).map(|_| ContentPart::Image).collect();
                parts.push(ContentPart::Text { text });
                let msg = ChatMessage {
                    role: "user".into(),
                    content: MessageContent::Parts(parts),
                };
                template.render(&[msg], true)
            }
        }
        Prompt::Chat(mut messages) => {
            // Caller-built chat: trust the messages as-is. If they
            // didn't add image parts, that's their decision.
            if num_images > 0
                && messages.iter().all(|m| {
                    !matches!(&m.content, MessageContent::Parts(parts)
                    if parts.iter().any(|p| matches!(p, ContentPart::Image)))
                })
            {
                // No image parts in any message but images were
                // supplied — splice into the last user message.
                if let Some(last_user) = messages.iter_mut().rev().find(|m| m.role == "user") {
                    let existing_text = match std::mem::replace(
                        &mut last_user.content,
                        MessageContent::Text(String::new()),
                    ) {
                        MessageContent::Text(t) => t,
                        MessageContent::Parts(parts) => parts
                            .into_iter()
                            .filter_map(|p| match p {
                                ContentPart::Text { text } => Some(text),
                                ContentPart::Image => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n"),
                    };
                    let mut new_parts: Vec<ContentPart> =
                        (0..num_images).map(|_| ContentPart::Image).collect();
                    new_parts.push(ContentPart::Text {
                        text: existing_text,
                    });
                    last_user.content = MessageContent::Parts(new_parts);
                }
            }
            template.render(&messages, true)
        }
    }
}

/// Top-level loader for a qwen3_5 family checkpoint at `dir`.
/// Detects vision support from `cfg.vision_config` (always present on
/// VLM checkpoints, parsed as a stub on dense-only ones). Builds the
/// VLM adapter when vision weights are present, else the dense
/// adapter with a text-only processor.
pub(crate) fn load_context(dir: &Path) -> Result<LoadedContext, Error> {
    let cfg_path = dir.join("config.json");
    let cfg = ModelConfig::from_file(&cfg_path)?;

    let tokenizer = load_tokenizer(dir)?;
    let chat_template = ChatTemplate::from_dir(dir)?;
    let eos_ids = crate::models::qwen3_5::read_qwen3_5_eos_ids(dir, &cfg);

    // VLM checkpoints have a `preprocessor_config.json` alongside the
    // text weights; dense checkpoints don't. Probe for the file.
    if dir.join("preprocessor_config.json").exists() {
        let (model, vision, leftover) = load_full_model(&cfg, dir)?;
        if !leftover.is_empty() {
            return Err(Error::Other(
                format!(
                    "qwen3_5 vlm load: {} unbound key(s); first 8: {:?}",
                    leftover.len(),
                    leftover.iter().take(8).collect::<Vec<_>>()
                )
                .into(),
            ));
        }
        let image_processor = Qwen35ImageProcessor::from_dir(dir)?;
        let dense = Qwen35DenseAdapter::new(model, cfg.clone());
        let vlm = Qwen35VlmAdapter::new(dense, vision);
        let processor = Qwen35Processor::new(tokenizer, chat_template, image_processor, cfg);
        Ok((Box::new(vlm), Box::new(processor), eos_ids))
    } else {
        let (model, leftover) = crate::models::qwen3_5::weights::load_language_model(&cfg, dir)?;
        if !leftover.is_empty() {
            return Err(Error::Other(
                format!(
                    "qwen3_5 dense load: {} unbound key(s); first 8: {:?}",
                    leftover.len(),
                    leftover.iter().take(8).collect::<Vec<_>>()
                )
                .into(),
            ));
        }
        let dense = Qwen35DenseAdapter::new(model, cfg);
        let processor =
            crate::language_model::TextOnlyProcessor::new("qwen3_5", tokenizer, chat_template);
        Ok((Box::new(dense), Box::new(processor), eos_ids))
    }
}
