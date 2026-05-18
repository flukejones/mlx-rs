//! Weight loader for Qwen3.5 / Chandra OCR-2 checkpoints.
//!
//! Mirrors the Python `Model.sanitize` from `mlx_vlm.models.qwen3_5.qwen3_5`:
//!
//! - `model.language_model.X`  -> `language_model.model.X`
//! - `model.visual.X`          -> `vision_tower.X` (vision tower not yet wired)
//! - `lm_head.X`               -> `language_model.lm_head.X`
//! - `conv1d.weight` whose last axis != 1 is sanitised with `moveaxis(2, 1)`.
//! - Norm weights (`*.input_layernorm.weight`,
//!   `*.post_attention_layernorm.weight`, `model.norm.weight`,
//!   `*.q_norm.weight`, `*.k_norm.weight`) get `+1.0` if their dtype is a
//!   floating-point 1-D tensor — the Python implementation does the same to
//!   recover the standard RMSNorm parameterisation from the centred form
//!   stored in the checkpoint.
//! - `mtp.*` (multi-token-prediction) keys are skipped.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use mlx_rs::{
    module::ModuleParameters, ops::move_axis, quantization::Quantizable as _,
    transforms::eval_params, Array, Dtype,
};
use serde::Deserialize;

use super::config::{ModelConfig, QuantizationConfig};
use super::layer::LanguageModel;
use super::vision::VisionModel;
use crate::error::Error;

const NORM_SUFFIXES: &[&str] = &[
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    "model.norm.weight",
];

/// `model.safetensors.index.json` schema.
#[derive(Debug, Deserialize)]
struct WeightIndex {
    weight_map: HashMap<String, String>,
}

/// Returns `true` if the safetensors file's header metadata advertises
/// `format == "mlx"`. mlx-format checkpoints already have the
/// norm-weight `+1.0` shift baked into the stored tensors, so the
/// `sanitize` step in `mlx_vlm.utils.load_model` is skipped for them —
/// and our loader must skip it too.
fn safetensors_is_mlx_format(path: &Path) -> Result<bool, Error> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut len_bytes = [0_u8; 8];
    if f.read_exact(&mut len_bytes).is_err() {
        return Ok(false);
    }
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    if header_len == 0 || header_len > 64 * 1024 * 1024 {
        return Ok(false);
    }
    let mut buf = vec![0_u8; header_len];
    if f.read_exact(&mut buf).is_err() {
        return Ok(false);
    }
    let header = std::str::from_utf8(&buf)
        .map_err(|e| Error::Other(format!("safetensors header is not utf-8: {e}").into()))?;
    // The metadata lives under `"__metadata__"`. Just scan for the
    // `"format":"mlx"` literal — full JSON parsing of the header is
    // overkill for one boolean.
    Ok(header.contains("\"__metadata__\"") && header.contains("\"format\":\"mlx\""))
}

/// Apply the Python `sanitize_key` mapping in place.
fn sanitize_key(key: &str) -> String {
    if key.contains("model.language_model") {
        return key.replace("model.language_model", "language_model.model");
    }
    if key.contains("model.visual") {
        return key.replace("model.visual", "vision_tower");
    }
    if let Some(rest) = key.strip_prefix("lm_head") {
        return format!("language_model.lm_head{rest}");
    }
    key.to_string()
}

/// Strip the `language_model.` prefix to match the Rust LanguageModel's
/// parameter paths.
#[cfg(test)]
fn strip_language_model_prefix(key: &str) -> &str {
    key.strip_prefix("language_model.").unwrap_or(key)
}

/// Bucket a sanitised key into either the language-model or vision-tower
/// param-path namespace, with the appropriate prefix stripped.
#[derive(Debug)]
pub enum Bucketed {
    /// Routes to `LanguageModel` under the returned path.
    LanguageModel(String),
    /// Routes to `VisionModel` under the returned path.
    Vision(String),
    /// Neither bucket — typically a `mtp.*` or unknown key that should be
    /// dropped or surfaced in the loader's `leftover` list.
    Other(String),
}

fn bucket_key(key: String) -> Bucketed {
    if let Some(rest) = key.strip_prefix("language_model.") {
        return Bucketed::LanguageModel(rest.to_string());
    }
    if let Some(rest) = key.strip_prefix("vision_tower.") {
        return Bucketed::Vision(rest.to_string());
    }
    Bucketed::Other(key)
}

/// Apply the `+1.0` centring fix to a norm weight tensor.
fn add_one_to_norm(value: &Array) -> Result<Array, Error> {
    let dt = value.dtype();
    let one = Array::from_f32(1.0)
        .as_dtype(dt)
        .map_err(Error::Exception)?;
    value.add(&one).map_err(Error::Exception)
}

/// Apply the conv1d moveaxis sanitisation: `[out, in, k]` -> `[out, k, in]`
/// when the last axis is not already `1`.
fn sanitize_conv1d_weight(value: &Array) -> Result<Array, Error> {
    let shape = value.shape();
    if shape.len() != 3 {
        return Ok(value.clone());
    }
    if shape[2] == 1 {
        return Ok(value.clone());
    }
    let moved = move_axis(value, 2, 1).map_err(Error::Exception)?;
    Ok(moved)
}

/// Returns the `Array` after applying any per-key sanitisation rules.
///
/// `is_mlx_format` tracks whether the source safetensors carry the
/// `format == "mlx"` metadata flag — when true, the conv1d moveaxis and the
/// norm `+1.0` shift are *already* baked into the stored weights, and
/// re-applying them here doubles the bias / mis-orients the kernel.
fn sanitize_value(key: &str, value: Array, is_mlx_format: bool) -> Result<Array, Error> {
    if !is_mlx_format && key.contains("conv1d.weight") {
        return sanitize_conv1d_weight(&value);
    }
    if is_mlx_format {
        return Ok(value);
    }
    let needs_plus_one = NORM_SUFFIXES.iter().any(|sfx| key.ends_with(sfx));
    if needs_plus_one && value.ndim() == 1 && is_float(value.dtype()) {
        return add_one_to_norm(&value);
    }
    Ok(value)
}

fn is_float(dtype: Dtype) -> bool {
    matches!(
        dtype,
        Dtype::Float16 | Dtype::Float32 | Dtype::Float64 | Dtype::Bfloat16
    )
}

/// Load and sanitise every shard listed in `model.safetensors.index.json`.
///
/// Returns a flat map keyed by the **fully-qualified** sanitised path
/// (`language_model.model.layers.0.self_attn.q_proj.weight`,
/// `vision_tower.patch_embed.proj.weight`, ...) with `mtp.*` filtered out.
/// Caller buckets the result into the LM / vision-tower parameter walks.
pub fn load_sanitized_weights(
    model_dir: impl AsRef<Path>,
) -> Result<HashMap<String, Array>, Error> {
    let model_dir = model_dir.as_ref();
    let shards = list_shards(model_dir)?;
    let mut raw: HashMap<String, Array> = HashMap::new();
    for shard in shards {
        let path = model_dir.join(shard);
        let is_mlx_format = safetensors_is_mlx_format(&path)?;
        let loaded = Array::load_safetensors(&path).map_err(Error::LoadWeights)?;
        for (k, v) in loaded {
            if k.contains("mtp.") {
                continue;
            }
            let san_k = sanitize_key(&k);
            let san_v = sanitize_value(&san_k, v, is_mlx_format)?;
            raw.insert(san_k, san_v);
        }
    }

    // Build the set of prefixes whose tensor is quantised (i.e. there's a
    // matching `*.scales` companion). Each such `<prefix>.weight` is then
    // remapped to `<prefix>.inner.weight` so it lines up with the Rust
    // QuantizedLinear's param path.
    let quantised_prefixes: HashSet<String> = raw
        .keys()
        .filter_map(|k| k.strip_suffix(".scales").map(|p| p.to_string()))
        .collect();

    let mut out: HashMap<String, Array> = HashMap::with_capacity(raw.len());
    for (mut k, v) in raw {
        // Naming alignment between the Python checkpoint and the Rust
        // parameter walk:
        //
        //  - the GDN `norm` submodule stores its scale as `norm.weight` —
        //    Rust collapses it into a single `norm_weight` Param.
        //  - the GDN `A_log` scalar parameter follows Rust's lowercase
        //    convention as `a_log`.
        k = k.replace(".linear_attn.norm.weight", ".linear_attn.norm_weight");
        k = k.replace(".linear_attn.A_log", ".linear_attn.a_log");

        // For quantised linears, rewrite the underlying weight slot so it
        // matches the `MaybeQuantized::Quantized(QuantizedLinear { inner })`
        // shape that `parameters_mut().flatten()` exposes.
        if let Some(prefix) = k.strip_suffix(".weight") {
            if quantised_prefixes.contains(prefix) {
                k = format!("{prefix}.inner.weight");
            }
        }

        out.insert(k, v);
    }
    Ok(out)
}

fn list_shards(model_dir: &Path) -> Result<Vec<String>, Error> {
    let single = model_dir.join("model.safetensors");
    if single.is_file() {
        return Ok(vec!["model.safetensors".to_string()]);
    }
    let index_path = model_dir.join("model.safetensors.index.json");
    if !index_path.is_file() {
        return Err(Error::Other(
            format!(
                "weights: neither model.safetensors nor model.safetensors.index.json present in {}",
                model_dir.display()
            )
            .into(),
        ));
    }
    let f = std::fs::File::open(index_path)?;
    let index: WeightIndex = serde_json::from_reader(f)?;
    let unique: HashSet<&String> = index.weight_map.values().collect();
    let mut shards: Vec<String> = unique.into_iter().cloned().collect();
    shards.sort();
    Ok(shards)
}

/// Load weights into a Rust [`LanguageModel`] only. Vision-tower keys are
/// returned in the `leftover` list so callers can decide whether to ignore
/// them or feed them through [`load_full_model`] instead.
///
/// The model is rebuilt as quantised first when the checkpoint declares
/// `quantization_config`. Returns the loaded model and the list of
/// fully-qualified sanitised paths that did not bind to a model parameter.
pub fn load_language_model(
    cfg: &ModelConfig,
    model_dir: impl AsRef<Path>,
) -> Result<(LanguageModel, Vec<String>), Error> {
    let mut model = LanguageModel::new(cfg.text_config.clone()).map_err(Error::Exception)?;
    if let Some(q) = cfg.effective_quantization() {
        quantize_language_model(&mut model, q)?;
    }
    let weights = load_sanitized_weights(model_dir)?;

    let mut leftover = Vec::new();
    {
        let mut params = model.parameters_mut().flatten();
        for (k, v) in weights {
            match bucket_key(k) {
                Bucketed::LanguageModel(p) => {
                    if let Some(slot) = params.get_mut(&*p) {
                        **slot = v;
                    } else {
                        leftover.push(format!("language_model.{p}"));
                    }
                }
                Bucketed::Vision(p) => leftover.push(format!("vision_tower.{p}")),
                Bucketed::Other(p) => leftover.push(p),
            }
        }
    }

    eval_params(model.parameters()).map_err(Error::Exception)?;
    crate::loader::apply_post_load_memory_policy();
    leftover.sort();
    Ok((model, leftover))
}

/// Load both the language model and the vision tower from the same
/// checkpoint. Vision weights are bf16 (not quantised in chandra-ocr-2) so
/// the vision module is not run through [`quantize_language_model`].
pub fn load_full_model(
    cfg: &ModelConfig,
    model_dir: impl AsRef<Path>,
) -> Result<(LanguageModel, VisionModel, Vec<String>), Error> {
    let mut lm = LanguageModel::new(cfg.text_config.clone()).map_err(Error::Exception)?;
    if let Some(q) = cfg.effective_quantization() {
        quantize_language_model(&mut lm, q)?;
    }
    let mut vision = VisionModel::new(&cfg.vision_config).map_err(Error::Exception)?;
    let weights = load_sanitized_weights(model_dir)?;

    let mut leftover = Vec::new();
    {
        let mut lm_params = lm.parameters_mut().flatten();
        let mut vision_params = vision.parameters_mut().flatten();
        for (k, v) in weights {
            match bucket_key(k) {
                Bucketed::LanguageModel(p) => {
                    if let Some(slot) = lm_params.get_mut(&*p) {
                        **slot = v;
                    } else {
                        leftover.push(format!("language_model.{p}"));
                    }
                }
                Bucketed::Vision(p) => {
                    if let Some(slot) = vision_params.get_mut(&*p) {
                        **slot = v;
                    } else {
                        leftover.push(format!("vision_tower.{p}"));
                    }
                }
                Bucketed::Other(p) => leftover.push(p),
            }
        }
    }

    eval_params(lm.parameters()).map_err(Error::Exception)?;
    eval_params(vision.parameters()).map_err(Error::Exception)?;
    crate::loader::apply_post_load_memory_policy();
    leftover.sort();
    Ok((lm, vision, leftover))
}

fn quantize_language_model(model: &mut LanguageModel, q: &QuantizationConfig) -> Result<(), Error> {
    let original = std::mem::replace(
        model,
        LanguageModel::new(model.cfg.clone()).map_err(Error::Exception)?,
    );
    let quantized = original
        .try_into_quantized(q.group_size, q.bits)
        .map_err(Error::Exception)?;
    *model = quantized;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::vision::VisionModel;
    use super::*;

    #[test]
    fn sanitize_key_rewrites_language_model_prefix() {
        assert_eq!(
            sanitize_key("model.language_model.embed_tokens.weight"),
            "language_model.model.embed_tokens.weight"
        );
        assert_eq!(
            sanitize_key("model.visual.patch_embed.proj.weight"),
            "vision_tower.patch_embed.proj.weight"
        );
        assert_eq!(
            sanitize_key("lm_head.weight"),
            "language_model.lm_head.weight"
        );
        assert_eq!(
            sanitize_key("model.embed_tokens.weight"),
            "model.embed_tokens.weight"
        );
    }

    #[test]
    fn strip_prefix_drops_language_model_segment() {
        assert_eq!(
            strip_language_model_prefix("language_model.model.embed_tokens.weight"),
            "model.embed_tokens.weight"
        );
        assert_eq!(
            strip_language_model_prefix("vision_tower.patch_embed.proj.weight"),
            "vision_tower.patch_embed.proj.weight"
        );
    }

    #[test]
    #[ignore = "diagnostic: dump expected language-model parameter paths"]
    fn dump_vision_param_keys() {
        let cfg_json = r#"{
            "model_type": "qwen3_5",
            "depth": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "out_hidden_size": 128,
            "num_heads": 2,
            "patch_size": 16,
            "in_channels": 3,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 16
        }"#;
        let cfg: super::super::config::VisionConfig = serde_json::from_str(cfg_json).unwrap();
        let mut vm = VisionModel::new(&cfg).unwrap();
        let params = vm.parameters_mut().flatten();
        let mut keys: Vec<String> = params.keys().map(|k| k.to_string()).collect();
        keys.sort();
        for k in &keys {
            eprintln!("VKEY: {k}");
        }
    }

    #[test]
    #[ignore = "diagnostic: dumps the loaded LanguageModel parameter paths"]
    fn dump_expected_param_keys() {
        let cfg_json = r#"{
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 64,
                "rms_norm_eps": 1e-6,
                "vocab_size": 128,
                "max_position_embeddings": 256,
                "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_key_head_dim": 64,
                "linear_value_head_dim": 64,
                "linear_conv_kernel_dim": 4,
                "tie_word_embeddings": true,
                "rope_parameters": {
                    "mrope_section": [16, 8, 8],
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 1.0,
                    "type": "default"
                }
            },
            "vision_config": {
                "depth": 2,
                "hidden_size": 16,
                "intermediate_size": 32,
                "out_hidden_size": 32,
                "num_heads": 2,
                "patch_size": 16,
                "in_channels": 3,
                "spatial_merge_size": 2
            }
        }"#;
        let cfg: ModelConfig = serde_json::from_str(cfg_json).unwrap();
        let mut model = LanguageModel::new(cfg.text_config.clone()).unwrap();
        // Quantize like the chandra checkpoint would.
        let q = QuantizationConfig {
            group_size: 64,
            bits: 8,
            mode: "affine".to_string(),
        };
        quantize_language_model(&mut model, &q).unwrap();
        let params = model.parameters_mut().flatten();
        let mut keys: Vec<String> = params.keys().map(|k| k.to_string()).collect();
        keys.sort();
        for k in &keys {
            eprintln!("KEY: {k}");
        }
    }

    #[test]
    #[ignore = "requires local model files at ~/MLXModels/chandra2/chandra-ocr-2-mlx-q8"]
    fn text_only_prefill_runs_on_loaded_chandra_q8() {
        use crate::models::qwen3_5::cache::make_caches;
        use mlx_rs::transforms::eval;
        use tokenizers::Tokenizer;

        let home = std::env::var("HOME").unwrap();
        let dir = std::path::PathBuf::from(home).join("MLXModels/chandra2/chandra-ocr-2-mlx-q8");
        let cfg = ModelConfig::from_file(dir.join("config.json")).expect("parse config");
        let (mut model, _leftover) = load_language_model(&cfg, &dir).expect("load weights");

        let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");
        let enc = tok.encode("Hello, world!", true).expect("encode");
        let ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
        let s = ids.len() as i32;
        let inputs = Array::from_slice(&ids, &[1, s]);

        let mut caches = make_caches(&cfg);
        let logits = model
            .forward(Some(&inputs), None, &mut caches, None)
            .expect("forward");
        eval([&logits]).expect("eval");
        assert_eq!(
            logits.shape(),
            &[1, s, cfg.text_config.vocab_size],
            "logits shape mismatch"
        );
        // No NaNs anywhere.
        let any_nan: Array = logits
            .as_dtype(Dtype::Float32)
            .unwrap()
            .ne(logits.as_dtype(Dtype::Float32).unwrap())
            .unwrap()
            .any(None)
            .unwrap();
        eval([&any_nan]).unwrap();
        assert!(!any_nan.item::<bool>(), "logits contain NaN");
    }

    #[test]
    #[ignore = "requires local model files at ~/MLXModels/chandra2/chandra-ocr-2-mlx-q8"]
    fn loads_chandra_q8_full_model_with_vision() {
        let home = std::env::var("HOME").unwrap();
        let dir = std::path::PathBuf::from(home).join("MLXModels/chandra2/chandra-ocr-2-mlx-q8");
        let cfg = ModelConfig::from_file(dir.join("config.json")).expect("parse config");
        let (lm, vision, leftover) = load_full_model(&cfg, &dir).expect("load full model");
        // No leftover keys at all after wiring both buckets.
        if !leftover.is_empty() {
            eprintln!("unexpected leftover keys ({}):", leftover.len());
            for k in &leftover[..leftover.len().min(20)] {
                eprintln!("  {k}");
            }
            panic!("unexpected leftover keys");
        }
        use mlx_rs::module::ModuleParametersExt;
        lm.eval().expect("eval LM");
        vision.eval().expect("eval vision");
    }

    #[test]
    #[ignore = "requires local model files at ~/MLXModels/chandra2/chandra-ocr-2-mlx-q8"]
    fn loads_chandra_q8_weights_into_language_model() {
        let home = std::env::var("HOME").unwrap();
        let dir = std::path::PathBuf::from(home).join("MLXModels/chandra2/chandra-ocr-2-mlx-q8");
        let cfg = ModelConfig::from_file(dir.join("config.json")).expect("parse config");
        let (model, leftover) = load_language_model(&cfg, &dir).expect("load weights");

        // We should at least have layer 0 weights populated. The exact param
        // ergonomics get exercised in subsequent commits; for now we sanity-
        // check that the model survived the load and there are no
        // *unexpected* leftover keys: only vision-tower keys, which are not
        // wired up yet, may remain.
        let mut unexpected: Vec<&String> = leftover
            .iter()
            .filter(|k| !k.starts_with("vision_tower"))
            .collect();
        unexpected.sort();
        if !unexpected.is_empty() {
            eprintln!("first 30 unexpected leftover keys:");
            for k in unexpected.iter().take(30) {
                eprintln!("  {k}");
            }
            panic!(
                "{} unexpected leftover safetensors keys (see stderr)",
                unexpected.len()
            );
        }
        // Smoke-eval all parameters to catch loader errors.
        use mlx_rs::module::ModuleParametersExt;
        model.eval().expect("eval params");
    }
}
