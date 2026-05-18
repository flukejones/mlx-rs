//! Gemma 4 weight loader + safetensors sanitiser.
//!
//! Mirrors `mlx_lm.models.gemma4.Model.sanitize` (outer multimodal
//! wrapper) chained with `mlx_lm.models.gemma4_text.Model.sanitize`
//! (inner text model):
//!
//! - Drop `vision_tower.*`, `multi_modal_projector.*`, `audio_tower.*`,
//!   `embed_audio.*`, `embed_vision.*`, plus quantiser stats keys
//!   (`input_max`, `input_min`, `output_max`, `output_min`) and
//!   `self_attn.rotary_emb` (RoPE freqs are precomputed in code).
//! - `model.language_model.X` → `model.X` (collapse the multimodal
//!   `language_model.` middle hop).
//! - `*.experts.gate_up_proj` → `*.switch_glu.{gate_proj,up_proj}.weight`
//!   via `split(..., 2, axis=-2)` + `contiguous`.
//! - `*.experts.down_proj` → `*.switch_glu.down_proj.weight`.
//! - Quantised `<prefix>.weight` (with a `<prefix>.scales` sibling) is
//!   remapped to `<prefix>.inner.weight` to line up with the Rust
//!   `MaybeQuantized::Quantized` param path.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use mlx_rs::module::ModuleParameters;
use mlx_rs::ops::{contiguous, split};
use mlx_rs::transforms::eval_params;
use mlx_rs::Array;

use crate::error::Error;
use crate::models::gemma4::config::Gemma4Config;
use crate::models::gemma4::text::Model;
use crate::quantization::resolve_quantization;
use mlx_rs::quantization::Quantizable;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
struct WeightIndex {
    weight_map: HashMap<String, String>,
}

/// Substrings that mark a checkpoint key for unconditional removal.
const DROP_SUBSTRINGS: &[&str] = &[
    "vision_tower",
    "multi_modal_projector",
    "audio_tower",
    "embed_audio",
    "embed_vision",
    "self_attn.rotary_emb",
    "input_max",
    "input_min",
    "output_max",
    "output_min",
];

fn should_drop(key: &str) -> bool {
    DROP_SUBSTRINGS.iter().any(|s| key.contains(s))
}

/// Strip the multimodal-wrapper prefix(es) so a text-only `Model` can
/// consume the keys: `language_model.model.X` → `model.X` (the form
/// mlx-community Gemma 4 checkpoints actually use), and
/// `model.language_model.X` → `model.X` for any variant that kept the
/// outer `model.` wrapper.
fn rewrite_outer_key(key: &str) -> String {
    if let Some(rest) = key.strip_prefix("language_model.model.") {
        return format!("model.{rest}");
    }
    if let Some(rest) = key.strip_prefix("model.language_model.") {
        return format!("model.{rest}");
    }
    if let Some(rest) = key.strip_prefix("language_model.") {
        // Bare `language_model.X` (lm_head etc.). Drop the prefix.
        return rest.to_string();
    }
    key.to_string()
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
                "gemma4 weights: neither model.safetensors nor model.safetensors.index.json in {}",
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

/// Sanitise one (key, tensor) pair. Returns a vec because a single
/// `experts.gate_up_proj` entry expands into two `switch_glu` entries.
fn sanitize_entry(key: &str, value: Array) -> Result<Vec<(String, Array)>, Error> {
    if should_drop(key) {
        return Ok(Vec::new());
    }
    let key = rewrite_outer_key(key);

    if let Some(base) = key.strip_suffix(".experts.gate_up_proj") {
        // Split `[num_experts, 2*moe_hidden, hidden]` along axis -2 into
        // `(gate, up)`, each `[num_experts, moe_hidden, hidden]`. Force
        // contiguous so each tensor is row-major in the rebuilt map.
        let parts = split(&value, 2, -2).map_err(Error::Exception)?;
        if parts.len() != 2 {
            return Err(Error::Other(
                format!("gemma4: experts.gate_up_proj split into {} parts", parts.len()).into(),
            ));
        }
        let mut it = parts.into_iter();
        let gate = contiguous(it.next().unwrap(), None).map_err(Error::Exception)?;
        let up = contiguous(it.next().unwrap(), None).map_err(Error::Exception)?;
        return Ok(vec![
            (format!("{base}.switch_glu.gate_proj.weight"), gate),
            (format!("{base}.switch_glu.up_proj.weight"), up),
        ]);
    }
    if let Some(base) = key.strip_suffix(".experts.down_proj") {
        return Ok(vec![(format!("{base}.switch_glu.down_proj.weight"), value)]);
    }

    Ok(vec![(key, value)])
}

/// Load every shard, sanitise per-entry, then rewrite quantised
/// `<prefix>.weight` → `<prefix>.inner.weight` for Rust param paths.
pub fn load_sanitized_gemma4_weights(
    model_dir: impl AsRef<Path>,
) -> Result<HashMap<String, Array>, Error> {
    let model_dir = model_dir.as_ref();
    let shards = list_shards(model_dir)?;

    let mut raw: HashMap<String, Array> = HashMap::new();
    for shard in shards {
        let path = model_dir.join(shard);
        let loaded = Array::load_safetensors(&path).map_err(Error::LoadWeights)?;
        for (k, v) in loaded {
            for (san_k, san_v) in sanitize_entry(&k, v)? {
                raw.insert(san_k, san_v);
            }
        }
    }

    // Quantised tensors carry `<prefix>.scales` (and `.biases`)
    // siblings; the `<prefix>.weight` slot must be redirected to
    // `<prefix>.inner.weight` to land on the
    // `MaybeQuantized::Quantized(QuantizedLinear { inner })` param.
    let quantised_prefixes: HashSet<String> = raw
        .keys()
        .filter_map(|k| k.strip_suffix(".scales").map(|p| p.to_string()))
        .collect();

    let mut out: HashMap<String, Array> = HashMap::with_capacity(raw.len());
    for (mut k, v) in raw {
        if let Some(prefix) = k.strip_suffix(".weight") {
            if quantised_prefixes.contains(prefix) {
                k = format!("{prefix}.inner.weight");
            }
        }
        out.insert(k, v);
    }
    Ok(out)
}

/// True if `key` targets a `model.layers.N.self_attn.{k,v}_*` slot on a
/// KV-shared layer (`N >= num_layers - num_shared`). The Rust model
/// builds those slots as `None`; the checkpoint still ships the unused
/// weights so we drop them at load time.
fn is_shared_kv_layer_key(key: &str, num_layers: i32, num_shared: i32) -> bool {
    if num_shared <= 0 {
        return false;
    }
    let first_shared = num_layers - num_shared;
    let rest = match key.strip_prefix("model.layers.") {
        Some(r) => r,
        None => return false,
    };
    let dot = match rest.find('.') {
        Some(i) => i,
        None => return false,
    };
    let layer_idx: i32 = match rest[..dot].parse() {
        Ok(i) => i,
        Err(_) => return false,
    };
    if layer_idx < first_shared {
        return false;
    }
    let tail = &rest[dot + 1..];
    tail.starts_with("self_attn.k_") || tail.starts_with("self_attn.v_")
}


/// End-to-end load: build `Model::new`, apply quantisation config, load
/// sanitised weights into the parameter walk, then `eval_params`.
pub fn load_gemma4_model_sanitized(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let cfg = Gemma4Config::from_file(model_dir.join("config.json"))?;
    let quant = resolve_quantization(&cfg.quantization, &cfg.quantization_config).cloned();
    let num_layers = cfg.num_hidden_layers;
    let num_shared = cfg.num_kv_shared_layers;
    let mut model = Model::new(cfg)?;
    if let Some(q) = quant {
        model = model.try_into_quantized(q.group_size, q.bits)?;
    }

    let weights = load_sanitized_gemma4_weights(model_dir)?;

    let mut leftover: Vec<String> = Vec::new();
    {
        let mut params = model.parameters_mut().flatten();
        for (k, v) in weights {
            if is_shared_kv_layer_key(&k, num_layers, num_shared) {
                // KV-shared layers reuse an earlier layer's K/V; the
                // checkpoint keeps the unused weights to mirror Python.
                continue;
            }
            if let Some(slot) = params.get_mut(&*k) {
                **slot = v;
            } else {
                leftover.push(k);
            }
        }
    }

    if !leftover.is_empty() {
        leftover.sort();
        return Err(Error::Other(
            format!(
                "gemma4 loader: {} unbound key(s); first 8: {:?}",
                leftover.len(),
                &leftover.iter().take(8).collect::<Vec<_>>()
            )
            .into(),
        ));
    }
    eval_params(model.parameters()).map_err(Error::Exception)?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_vision_and_quant_stats() {
        assert!(should_drop("vision_tower.encoder.layer.0.attn.q.weight"));
        assert!(should_drop("multi_modal_projector.proj.weight"));
        assert!(should_drop("audio_tower.layer.0.proj.weight"));
        assert!(should_drop("embed_audio.weight"));
        assert!(should_drop("embed_vision.weight"));
        assert!(should_drop("layers.0.self_attn.rotary_emb.inv_freq"));
        assert!(should_drop("layers.0.self_attn.q_proj.input_max"));
        assert!(!should_drop("model.layers.0.self_attn.q_proj.weight"));
    }

    #[test]
    fn rewrites_language_model_prefix() {
        // mlx-community canonical form: `language_model.model.X`.
        assert_eq!(
            rewrite_outer_key("language_model.model.layers.0.self_attn.q_proj.weight"),
            "model.layers.0.self_attn.q_proj.weight"
        );
        // Outer-`model.` multimodal variant.
        assert_eq!(
            rewrite_outer_key("model.language_model.layers.0.self_attn.q_proj.weight"),
            "model.layers.0.self_attn.q_proj.weight"
        );
        // Bare `language_model.X` (lm_head etc.) — drop prefix entirely.
        assert_eq!(
            rewrite_outer_key("language_model.lm_head.weight"),
            "lm_head.weight"
        );
        // Already-flat keys pass through.
        assert_eq!(
            rewrite_outer_key("model.layers.0.self_attn.q_proj.weight"),
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(rewrite_outer_key("lm_head.weight"), "lm_head.weight");
    }
}
