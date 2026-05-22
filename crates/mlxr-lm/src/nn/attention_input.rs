//! Canonical per-layer attention input.

use mlxr::Array;

/// Per-layer attention input used by `llama`, `qwen3`, `gemma3`, and
/// `gemma4` (`gemma4::Attention::attend` returns `AttentionOut` but
/// takes this struct as its input).
///
/// `shared_kv` / `offset` are populated only by Gemma 4 KV-sharing
/// layers; `None` everywhere else. Hot-path-static branches dominated
/// by projection + RoPE cost — see plan notes for the perf justification
/// to keep them as `Option` rather than splitting into two structs.
pub struct AttentionInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: Option<&'a mut C>,
    /// Gemma 4 KV-sharing: K/V tensors produced by an earlier layer of
    /// the same `layer_kind`. `Some` for KV-shared layers, `None`
    /// otherwise.
    pub shared_kv: Option<(Array, Array)>,
    /// Cache offset captured from the source layer, paired with
    /// `shared_kv`.
    pub offset: Option<i32>,
}

impl<'a, C> AttentionInput<'a, C> {
    /// Construct an input for the common case (no KV sharing).
    pub fn plain(x: &'a Array, mask: Option<&'a Array>, cache: Option<&'a mut C>) -> Self {
        Self {
            x,
            mask,
            cache,
            shared_kv: None,
            offset: None,
        }
    }
}
