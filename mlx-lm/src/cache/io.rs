//! Prompt-cache helpers + safetensors persistence (Python
//! `mlx_lm.models.cache` parity).

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;

use crate::error::Error;

use super::kvcache::KVCache;
use super::quantized_kvcache::QuantizedKVCache;
use super::trait_def::KeyValueCache;

pub(super) fn parse_meta(meta: &HashMap<String, String>, key: &str) -> Result<i32, Error> {
    meta.get(key)
        .ok_or_else(|| Error::Other(format!("missing meta key {key:?}").into()))?
        .parse::<i32>()
        .map_err(|e| Error::Other(format!("meta {key:?} parse: {e}").into()))
}

pub(super) fn parse_meta_or(
    meta: &HashMap<String, String>,
    key: &str,
    default: i32,
) -> Result<i32, Error> {
    meta.get(key)
        .map(|s| s.parse::<i32>())
        .transpose()
        .map_err(|e| Error::Other(format!("meta {key:?} parse: {e}").into()))
        .map(|v| v.unwrap_or(default))
}

/// One [`KVCache`] per layer for a decoder-only model. For hybrid
/// models (qwen3.5) use that model's own `make_caches`. `max_kv_size`
/// is reserved for sliding-window models.
pub fn make_prompt_cache(num_layers: usize, _max_kv_size: Option<i32>) -> Vec<KVCache> {
    (0..num_layers).map(|_| KVCache::new()).collect()
}

/// Return `true` iff every cache in the slice supports `trim` with a
/// non-zero argument.
pub fn can_trim_prompt_cache<C: KeyValueCache>(caches: &[C]) -> bool {
    !caches.is_empty() && caches.iter().all(|c| c.is_trimmable())
}

/// Trim the trailing `n` tokens from every cache in the slice. Returns the
/// minimum number of tokens actually trimmed (some caches may be shorter).
pub fn trim_prompt_cache<C: KeyValueCache>(caches: &mut [C], n: i32) -> i32 {
    if !can_trim_prompt_cache(caches) || n <= 0 {
        return 0;
    }
    caches.iter_mut().map(|c| c.trim(n)).min().unwrap_or(0)
}

/// Save a prompt cache to a `.safetensors` file, mirroring Python's
/// wire format: per-layer arrays keyed `layer.{i}.{slot}` and per-layer
/// metadata keyed `layer.{i}.{key}` plus a flat `layer.{i}.class_name`.
/// `extra_metadata` is merged into the metadata map under unprefixed keys.
pub fn save_prompt_cache<C: KeyValueCache>(
    path: impl AsRef<Path>,
    caches: &[C],
    extra_metadata: Option<&HashMap<String, String>>,
) -> Result<(), Error> {
    let mut arrays: Vec<(String, Array)> = Vec::new();
    let mut metadata: HashMap<String, String> = HashMap::new();

    for (i, c) in caches.iter().enumerate() {
        let class_name = c.class_name();
        metadata.insert(format!("layer.{i}.class_name"), class_name.to_string());
        for (k, v) in c.meta_state() {
            metadata.insert(format!("layer.{i}.{k}"), v);
        }
        let slot_names = state_slot_names(class_name);
        let state = c.state();
        if !state.is_empty() && state.len() != slot_names.len() {
            return Err(Error::Other(
                format!(
                    "{class_name}.state() returned {} arrays, expected {}",
                    state.len(),
                    slot_names.len()
                )
                .into(),
            ));
        }
        for (slot, a) in slot_names.iter().zip(state) {
            arrays.push((format!("layer.{i}.{slot}"), a));
        }
    }

    if let Some(extra) = extra_metadata {
        for (k, v) in extra {
            metadata.insert(k.clone(), v.clone());
        }
    }
    metadata.insert("num_layers".into(), caches.len().to_string());

    let array_refs: Vec<(String, &Array)> = arrays.iter().map(|(k, a)| (k.clone(), a)).collect();
    Array::save_safetensors(array_refs, Some(&metadata), path)?;
    Ok(())
}

/// One layer's worth of loaded prompt-cache state. Caller dispatches on the
/// variant to recover the original cache type.
#[derive(Debug)]
pub enum LoadedCache {
    /// `class_name == "KVCache"`.
    Plain(KVCache),
    /// `class_name == "QuantizedKVCache"`.
    Quantized(QuantizedKVCache),
}

impl LoadedCache {
    /// Discriminant matching Python `class_name`.
    pub fn class_name(&self) -> &'static str {
        match self {
            LoadedCache::Plain(_) => "KVCache",
            LoadedCache::Quantized(_) => "QuantizedKVCache",
        }
    }
}

/// Inverse of [`save_prompt_cache`]. Returns one [`LoadedCache`] per layer
/// plus any extra metadata that wasn't prefixed with `layer.{i}.`.
pub fn load_prompt_cache(
    path: impl AsRef<Path>,
) -> Result<(Vec<LoadedCache>, HashMap<String, String>), Error> {
    let (mut arrays, mut meta) = Array::load_safetensors_with_metadata(path)?;

    let num_layers: usize = meta
        .remove("num_layers")
        .ok_or_else(|| Error::Other("prompt cache missing num_layers meta".into()))?
        .parse()
        .map_err(|e| Error::Other(format!("num_layers parse: {e}").into()))?;

    let mut layers: Vec<LoadedCache> = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let class_name = meta
            .remove(&format!("layer.{i}.class_name"))
            .ok_or_else(|| Error::Other(format!("missing layer.{i}.class_name").into()))?;

        let prefix = format!("layer.{i}.");
        let mut layer_meta: HashMap<String, String> = HashMap::new();
        let keys: Vec<String> = meta
            .keys()
            .filter(|k| k.starts_with(&prefix))
            .cloned()
            .collect();
        for k in keys {
            let suffix = k[prefix.len()..].to_string();
            let v = meta.remove(&k).expect("just found");
            layer_meta.insert(suffix, v);
        }

        let slot_names = state_slot_names(&class_name);
        let mut state: Vec<Array> = Vec::with_capacity(slot_names.len());
        for slot in slot_names {
            let key = format!("layer.{i}.{slot}");
            let a = arrays.remove(&key).ok_or_else(|| {
                Error::Other(format!("missing array {key} for {class_name}").into())
            })?;
            state.push(a);
        }

        let loaded = match class_name.as_str() {
            "KVCache" => LoadedCache::Plain(KVCache::from_state(state, &layer_meta)?),
            "QuantizedKVCache" => {
                LoadedCache::Quantized(QuantizedKVCache::from_state(state, &layer_meta)?)
            }
            other => {
                return Err(Error::Other(
                    format!("unsupported prompt-cache class {other}").into(),
                ))
            }
        };
        layers.push(loaded);
    }

    Ok((layers, meta))
}

/// Per-class state-array slot names, matching the order
/// [`KeyValueCache::state`] returns them in.
fn state_slot_names(class_name: &str) -> &'static [&'static str] {
    match class_name {
        "KVCache" => &["keys", "values"],
        "QuantizedKVCache" => &[
            "keys_wq",
            "keys_scales",
            "keys_biases",
            "values_wq",
            "values_scales",
            "values_biases",
        ],
        _ => &[],
    }
}
