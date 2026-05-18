//! Canonical building blocks shared across decoder models.
//! `qwen3_5` keeps its own attention/multimodal pipeline
//! (see `models::qwen3_5::generation`).

pub mod attention_input;
pub mod model_input;
pub mod swiglu_mlp;

pub use attention_input::AttentionInput;
pub use model_input::ModelInput;
pub use swiglu_mlp::SwigluMlp;

use crate::cache::KeyValueCache;

/// Populate `cache` with `len` default-constructed slots if empty.
/// No-op once populated.
pub fn ensure_cache_populated<C>(cache: &mut Vec<Option<C>>, len: usize)
where
    C: KeyValueCache + Default,
{
    if cache.is_empty() {
        *cache = (0..len).map(|_| Some(C::default())).collect();
    }
}
