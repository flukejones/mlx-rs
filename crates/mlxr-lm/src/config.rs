//! Top-level workspace config schema.
//!
//! Every checkpoint is parsed exactly once at [`load`](crate::load)
//! into [`ModelConfig`]. The `model_type` field drives the
//! [`Family`] discriminant via serde's internally-tagged enum
//! shape — no second probe pass, no string match in the dispatcher.
//!
//! Family-specific text configs (`qwen3_5::text::config::TextConfig`,
//! `gemma4::text::config::TextConfig`) hang off the variant via the
//! per-family `ModelConfig` envelope.
//! Anything common-to-all (quantization, image tokens, EOS) lives on
//! the outer struct.

use std::path::Path;

use serde::Deserialize;

#[cfg(feature = "gemma4")]
use crate::gemma4::text::config::ModelConfig as Gemma4Envelope;
#[cfg(feature = "qwen3_5")]
use crate::qwen3_5::text::config::ModelConfig as Qwen35Envelope;

use crate::error::Error;
use crate::quantization::QuantizationConfig;

/// Parsed `config.json` for any supported family. Exactly one
/// allocation per load — there is no second probe of the file.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    /// Family-tagged config body. Dispatched on `model_type` by serde.
    #[serde(flatten)]
    pub family: Family,

    /// Body quantisation + per-tensor overrides. mlx-community
    /// checkpoints sometimes emit both `quantization` and a sibling
    /// `quantization_config` with identical contents; the second is
    /// captured separately and folded into `quantization` if the
    /// primary is absent.
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
    #[serde(default, rename = "quantization_config")]
    quantization_legacy: Option<QuantizationConfig>,
}

impl ModelConfig {
    /// Parse a `config.json` at `<dir>/config.json`. The whole file is
    /// read and validated against the typed schema once — every
    /// subsequent consumer reads fields off the typed struct.
    pub fn from_dir(dir: &Path) -> Result<Self, Error> {
        let path = dir.join("config.json");
        let raw = std::fs::read_to_string(&path)
            .map_err(|e| Error::Other(format!("read {}: {e}", path.display()).into()))?;
        let mut cfg: Self = serde_json::from_str(&raw)
            .map_err(|e| Error::Other(format!("parse {}: {e}", path.display()).into()))?;
        if cfg.quantization.is_none() {
            cfg.quantization = cfg.quantization_legacy.take();
        }
        Ok(cfg)
    }

    /// Family name (the canonical `model_type` value).
    pub fn family_name(&self) -> &'static str {
        self.family.name()
    }

    /// Effective quantisation settings, preferring the modern
    /// `quantization` field and falling back to the legacy
    /// `quantization_config` block.
    pub fn quantization(&self) -> Option<&QuantizationConfig> {
        self.quantization
            .as_ref()
            .or(self.quantization_legacy.as_ref())
    }
}

/// Per-family config body. Each variant owns its own typed text (and
/// future image / audio) sub-configs; serde's `tag = "model_type"`
/// reads the discriminant from the top-level `model_type` field at
/// deserialize time. Unknown discriminants fail with
/// `unknown variant`, not silent default.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "model_type")]
pub enum Family {
    /// Qwen 3.5 / 3.6 dense and Qwen3-VL.
    #[cfg(feature = "qwen3_5")]
    #[serde(
        rename = "qwen3_5",
        alias = "qwen3_5_text",
        alias = "qwen3_5forconditionalgeneration"
    )]
    Qwen35(Qwen35Envelope),

    /// Qwen 3.6 MoE (35B-A3B). Same envelope shape as
    /// [`Self::Qwen35`]; `text_config.num_experts > 0`.
    #[cfg(feature = "qwen3_5")]
    #[serde(rename = "qwen3_5_moe", alias = "qwen3_5_moe_text")]
    Qwen35Moe(Qwen35Envelope),

    /// Gemma 4 (E2B, E4B, 26B-A4B, 31B).
    #[cfg(feature = "gemma4")]
    #[serde(
        rename = "gemma4",
        alias = "gemma4_text",
        alias = "gemma4textmodel",
        alias = "gemma4forcausallm"
    )]
    Gemma4(Gemma4Envelope),
}

impl Family {
    /// Canonical family name. Logs + errors use this; never a raw
    /// `model_type` string from the file.
    pub fn name(&self) -> &'static str {
        match self {
            #[cfg(feature = "qwen3_5")]
            Self::Qwen35(_) => "qwen3_5",
            #[cfg(feature = "qwen3_5")]
            Self::Qwen35Moe(_) => "qwen3_5_moe",
            #[cfg(feature = "gemma4")]
            Self::Gemma4(_) => "gemma4",
        }
    }

    /// Return the qwen3_5 envelope or `None` for any other family.
    /// Both dense and MoE share `Qwen35Envelope`.
    #[cfg(feature = "qwen3_5")]
    pub fn as_qwen35(&self) -> Option<&Qwen35Envelope> {
        match self {
            Self::Qwen35(env) | Self::Qwen35Moe(env) => Some(env),
            #[cfg(feature = "gemma4")]
            _ => None,
        }
    }

    /// Return the gemma4 envelope or `None`.
    #[cfg(feature = "gemma4")]
    pub fn as_gemma4(&self) -> Option<&Gemma4Envelope> {
        match self {
            Self::Gemma4(env) => Some(env),
            #[cfg(feature = "qwen3_5")]
            _ => None,
        }
    }
}
