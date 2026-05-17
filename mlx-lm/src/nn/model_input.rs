//! Canonical model-level input.

use mlx_rs::Array;

/// Top-level model input.
///
/// `mask` is `None` for models that build the attention mask
/// internally (gemma3, gemma4 — they derive sliding/global masks from
/// the per-layer cache state). llama/qwen3 accept a caller-provided
/// mask via the `Some` branch. The dead `Option` slot costs one
/// pointer per call, dwarfed by the model forward itself.
pub struct ModelInput<'a, C> {
    pub inputs: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut Vec<Option<C>>,
}
