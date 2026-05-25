//! Family-agnostic helpers shared by all model-family adapters.
//!
//! [`LoadedContext`] is the shape every family's `load_context_*`
//! returns: the boxed model + processor + EOS-id list. The crate-root
//! `mlxr_lm::load` wraps it into a [`crate::ModelContext`].
//!
//! [`EosSpec`] normalises the `int | [int]` shape that Hugging Face
//! configs use for `eos_token_id`. Each family envelope owns its own
//! `eos_token_id: Option<EosSpec>` field that serde populates at the
//! one-and-only `config.json` parse.

use serde::Deserialize;

use crate::language_model::{LanguageModel, UserInputProcessor};

pub(crate) type LoadedContext = (
    Box<dyn LanguageModel>,
    Box<dyn UserInputProcessor>,
    Vec<u32>,
);

/// `config.json::eos_token_id` — either a single id or a list of ids.
/// Each family envelope carries `Option<EosSpec>` so the value is
/// parsed once at load and read off the typed struct afterwards.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EosSpec {
    /// Single EOS token id.
    Single(u32),
    /// Multiple acceptable EOS token ids.
    Many(Vec<u32>),
}

impl EosSpec {
    /// Flatten to a `Vec<u32>`. Empty when `spec` is `None`.
    pub(crate) fn to_vec(spec: Option<&Self>) -> Vec<u32> {
        match spec {
            Some(Self::Single(id)) => vec![*id],
            Some(Self::Many(ids)) => ids.clone(),
            None => Vec::new(),
        }
    }
}
