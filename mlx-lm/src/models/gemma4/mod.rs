//! Gemma 4 text model. Covers dense 31B + MoE 26B-A4B + E2B/E4B
//! per-layer-input variants.

pub mod config;
pub mod loader;
pub mod rope;
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
