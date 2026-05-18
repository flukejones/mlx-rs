//! Shared safetensors loader for direct-path models (llama, qwen3).
//! Sanitised loaders (gemma4, qwen3_5) keep bespoke per-key transforms
//! in their own modules.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use mlx_rs::module::{ModuleParameters, ModuleParametersExt};
use mlx_rs::transforms::eval_params;
use mlx_rs::Array;
use serde::{de::DeserializeOwned, Deserialize};
use tokenizers::Tokenizer;

use crate::error::Error;

/// Load `tokenizer.json` from a checkpoint directory.
pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    Tokenizer::from_file(model_dir.as_ref().join("tokenizer.json")).map_err(Into::into)
}

/// Deserialize `config.json` from a checkpoint directory.
pub fn load_config<C: DeserializeOwned>(model_dir: impl AsRef<Path>) -> Result<C, Error> {
    let file = std::fs::File::open(model_dir.as_ref().join("config.json"))?;
    Ok(serde_json::from_reader(file)?)
}

/// `model.safetensors.index.json` schema. `metadata` is optional.
#[derive(Debug, Clone, Deserialize)]
pub struct ShardIndex {
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    pub weight_map: HashMap<String, String>,
}

/// Load a safetensors checkpoint into `model`. Handles sharded and
/// single-file layouts. Rewrites `<prefix>.weight` →
/// `<prefix>.inner.weight` for any key whose checkpoint has a
/// `.scales` sibling (matches `QuantizedLinear`'s inner-Linear wrap).
/// Errors if a model param has no matching checkpoint key.
pub fn load_sharded<M: ModuleParametersExt>(model: &mut M, model_dir: &Path) -> Result<(), Error> {
    let shards = list_shards(model_dir)?;

    let mut raw: HashMap<String, Array> = HashMap::new();
    for shard in shards {
        let loaded = Array::load_safetensors(&shard).map_err(Error::LoadWeights)?;
        for (k, v) in loaded {
            raw.insert(k, v);
        }
    }

    let quantised_prefixes: HashSet<String> = raw
        .keys()
        .filter_map(|k| k.strip_suffix(".scales").map(|p| p.to_string()))
        .collect();

    let mut filled: HashSet<String> = HashSet::new();
    {
        let mut params = model.parameters_mut().flatten();
        for (k, v) in raw {
            let key = if let Some(prefix) = k.strip_suffix(".weight") {
                if quantised_prefixes.contains(prefix) {
                    format!("{prefix}.inner.weight")
                } else {
                    k
                }
            } else {
                k
            };
            // Extra checkpoint keys (e.g. `lm_head.weight` on a
            // tied-embedding model) are silently ignored; the
            // post-loop coverage check catches missing model params.
            if let Some(slot) = params.get_mut(&*key) {
                **slot = v;
                filled.insert(key);
            }
        }
    }

    let missing: Vec<String> = model
        .parameters()
        .flatten()
        .keys()
        .filter(|k| !filled.contains(k.as_ref()))
        .map(|k| k.to_string())
        .collect();
    if !missing.is_empty() {
        return Err(Error::Other(
            format!(
                "load_sharded: {} model param(s) not present in checkpoint; first 5: {:?}",
                missing.len(),
                &missing.iter().take(5).collect::<Vec<_>>(),
            )
            .into(),
        ));
    }

    eval_params(model.parameters()).map_err(Error::Exception)?;
    Ok(())
}

fn list_shards(model_dir: &Path) -> Result<Vec<PathBuf>, Error> {
    let index = model_dir.join("model.safetensors.index.json");
    if index.exists() {
        let json = std::fs::read_to_string(index)?;
        let map: ShardIndex = serde_json::from_str(&json)?;
        let files: HashSet<&String> = map.weight_map.values().collect();
        Ok(files.into_iter().map(|f| model_dir.join(f)).collect())
    } else {
        Ok(vec![model_dir.join("model.safetensors")])
    }
}
