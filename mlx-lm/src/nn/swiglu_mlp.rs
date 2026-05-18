//! Shared SwiGLU MLP used by llama and qwen3 (and any other LLaMA-family
//! decoder).
//!
//! Three Linear projections (gate, up, down) + a compiled `swiglu` op:
//! `down(swiglu(gate(x), up(x)))`. The `swiglu` op is `silu(gate) * up`
//! compiled via `mx.compile` and cached per layer (see
//! [`crate::activations::swiglu`]).

use mlx_rs::{
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn::{Linear, LinearBuilder},
    builder::Builder,
    quantization::MaybeQuantized,
    Array,
};

use crate::activations::{swiglu, SwigluCache};

/// Three-projection SwiGLU MLP. Bias on the projections is configurable
/// (`true` for llama, `false` for qwen3).
#[derive(Debug, ModuleParameters, Quantizable)]
pub struct SwigluMlp {
    #[quantizable]
    #[param]
    pub gate_proj: MaybeQuantized<Linear>,

    #[quantizable]
    #[param]
    pub down_proj: MaybeQuantized<Linear>,

    #[quantizable]
    #[param]
    pub up_proj: MaybeQuantized<Linear>,

    /// Per-layer compiled-graph cache for [`swiglu`]. Filled on first
    /// forward; reused across every decode step.
    swiglu_cache: SwigluCache,
}

impl SwigluMlp {
    /// Build a new MLP. `bias=true` matches llama's `mlp_bias` config;
    /// `bias=false` matches qwen3's hardcoded behaviour.
    pub fn new(dim: i32, hidden_dim: i32, bias: bool) -> Result<Self, Exception> {
        let gate_proj = LinearBuilder::new(dim, hidden_dim).bias(bias).build()?;
        let down_proj = LinearBuilder::new(hidden_dim, dim).bias(bias).build()?;
        let up_proj = LinearBuilder::new(dim, hidden_dim).bias(bias).build()?;

        Ok(Self {
            gate_proj: MaybeQuantized::Original(gate_proj),
            down_proj: MaybeQuantized::Original(down_proj),
            up_proj: MaybeQuantized::Original(up_proj),
            swiglu_cache: SwigluCache::default(),
        })
    }
}

impl Module<&Array> for SwigluMlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array) -> Result<Self::Output, Self::Error> {
        let gate = self.gate_proj.forward(input)?;
        let up = self.up_proj.forward(input)?;
        let activated = swiglu(&mut self.swiglu_cache, &gate, &up)?;
        self.down_proj.forward(&activated)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
    }
}
