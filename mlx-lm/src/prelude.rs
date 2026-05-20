//! Convenience re-exports for downstream consumers.
//!
//! `use mlx_lm::prelude::*;` brings the canonical building blocks
//! (cache, nn, sampler, loader) into scope. Models stay namespaced
//! (use `mlx_lm::models::<family>` explicitly) since the per-model
//! `Model`, `ModelArgs`, `Generate`, `load_<name>_model` names would
//! collide.

pub use crate::cache::{
    can_trim_prompt_cache, load_prompt_cache, make_prompt_cache, save_prompt_cache,
    trim_prompt_cache, KVCache, KeyValueCache, LoadedCache, QuantizedKVCache, RotatingKVCache,
    DEFAULT_KV_CACHE_STEP,
};

pub use crate::loader::{load_config, load_sharded, load_tokenizer, ShardIndex};

pub use crate::nn::{ensure_cache_populated, AttentionInput, ModelInput, SwigluMlp};

pub use crate::sampler::{sample, sample_with, top_p_sample, SamplerState, SamplingParams};

pub use crate::tri;
