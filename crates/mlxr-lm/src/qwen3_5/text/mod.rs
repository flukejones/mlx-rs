//! Qwen 3.5 text path: dense + MoE model code, adapters, and the
//! MTP rejection-sampling helper. Always compiled when the `qwen3_5`
//! feature is on.

pub mod adapter_dense;
pub mod adapter_moe;
pub mod cache;
pub mod config;
pub mod gated_delta;
pub mod gated_delta_block;
pub mod layer;
pub mod moe;
pub mod rope;
pub mod sampling;
#[allow(
    clippy::module_inception,
    reason = "text-family core type lives in text.rs"
)]
pub mod text;
pub mod weights;

pub use adapter_moe::{Qwen35MoeAdapter, MAX_MTP_DEPTH};
pub use cache::{make_caches, LayerCache, LinearAttnCache};
pub use config::{ModelConfig, TextConfig};
pub use gated_delta_block::GatedDeltaNet;
pub use layer::{DecoderLayer, Qwen35Decoder, Qwen35Model};
pub use rope::{apply_multimodal_rotary_pos_emb, MultimodalRope};
pub use text::{Attention, Mlp};

use std::path::Path;

/// Read the EOS id set for a Qwen 3.5 / 3.6 checkpoint and ensure
/// `<|im_end|>` (the chat-template turn marker) is present. The
/// tokenizer ships it in `added_tokens`, but `config.json::eos_token_id`
/// often points at `<|endoftext|>` only — letting it through causes
/// the model to keep generating role-tagged turns past the assistant
/// reply.
pub(crate) fn read_qwen3_5_eos_ids(dir: &Path, cfg: &ModelConfig) -> Vec<u32> {
    let mut ids = crate::family::read_eos_ids(dir);
    if let Some(cfg_eos) = cfg.eos_token_id.clone() {
        for id in cfg_eos.into_vec_with_chat_eos() {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }
    if !ids.contains(&config::QWEN_CHAT_EOS_TOKEN_ID) {
        ids.push(config::QWEN_CHAT_EOS_TOKEN_ID);
    }
    ids
}

/// Load the shared prelude every Qwen 3.5 / 3.6 adapter (dense, MoE,
/// VLM) needs: parsed `config.json`, tokenizer, chat template, and
/// the resolved EOS-id set.
pub(crate) fn load_common(
    dir: &Path,
) -> Result<
    (
        ModelConfig,
        tokenizers::Tokenizer,
        crate::chat_template::ChatTemplate,
        Vec<u32>,
    ),
    crate::error::Error,
> {
    let cfg = ModelConfig::from_file(dir.join("config.json"))?;
    let tokenizer = crate::loader::load_tokenizer(dir)?;
    let chat_template = crate::chat_template::ChatTemplate::from_dir(dir)?;
    let eos_ids = read_qwen3_5_eos_ids(dir, &cfg);
    Ok((cfg, tokenizer, chat_template, eos_ids))
}

/// Build a typed error for an adapter load that left safetensors keys
/// unbound. Truncates to the first 8 keys for log readability.
pub(crate) fn leftover_keys_error(family: &str, leftover: &[String]) -> crate::error::Error {
    crate::error::Error::Other(
        format!(
            "qwen3_5 {family} load: {} unbound key(s); first 8: {:?}",
            leftover.len(),
            leftover.iter().take(8).collect::<Vec<_>>()
        )
        .into(),
    )
}
