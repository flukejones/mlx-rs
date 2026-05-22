//! Gemma 4 text path: config, weights, model code, dense + MoE
//! variants, adapter. Always compiled when `gemma4` is on.

pub mod adapter;
pub mod config;
pub mod loader;
pub mod rope;
#[allow(
    clippy::module_inception,
    reason = "text-family core type lives in text.rs"
)]
pub mod text;
pub mod weights;

pub use config::{Gemma4Config, LayerKind};
pub use loader::Gemma4LayerCache;
pub use rope::ProportionalRope;
pub use text::{
    Attention, AttentionInput, AttentionOut, DecoderLayer, Gemma4TextModel, LayerRope, Mlp, Model,
    ModelInput, RmsNormNoScale,
};

/// Gemma 4 routed-expert FFN: gelu-approx activation + packed
/// `gate_up_proj` layout. Concrete alias of
/// [`crate::nn::switch::PackedSwitchFfn`].
pub type GemmaSwitchGlu = crate::nn::switch::PackedSwitchFfn<crate::nn::switch::GegluActivation>;
