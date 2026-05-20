//! Qwen3.6-MoE (35B-A3B): sparse MoE FFN over the qwen3_5 hybrid
//! GDN + full-attention spine.
//!
//! Reuses every piece of the qwen3_5 dense module except the FFN:
//! [`qwen3_5::layer::LanguageModel<Qwen35MoeBlock>`] gives us the
//! shared decoder skeleton + KV cache + generation iterator.
//! [`Qwen35MoeBlock`] implements the DeepSeek-style shared+routed
//! MoE: 256 routed experts (silu-gated [`SplitSwitchFfn`]), one
//! dense shared expert ([`SwigluMlp`]), one `[hidden, 1]`
//! sigmoid gate on the shared output, plus the linear router.
//!
//! Loader honours per-tensor quantisation overrides from
//! [`QuantizationConfig::for_path`] so the `mlp.gate` +
//! `mlp.shared_expert_gate` slots land at the right bit width
//! (Qwen3.6-MoE ships both at 8-bit even when the body is 4-bit).

use std::path::Path;

use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::macros::{ModuleParameters, Quantizable};
use mlx_rs::module::{Module, ModuleParameters};
use mlx_rs::nn;
use mlx_rs::nn::sigmoid;
use mlx_rs::ops::indexing::{take_along_axis, IndexOp};
use mlx_rs::ops::{argpartition_axis, softmax_axis, sum_axis};
use mlx_rs::quantization::{MaybeQuantized, Quantizable as _};
use mlx_rs::transforms::eval_params;
use mlx_rs::Array;

use crate::error::Error;
use crate::models::qwen3_5::{
    self,
    config::ModelConfig,
    layer::LanguageModel,
    weights::{bucket_key, load_sanitized_weights, Bucketed},
};
use crate::nn::switch::{SplitSwitchFfn, SwigluActivation};
use crate::nn::SwigluMlp;
use crate::quantization::QuantizationConfig;

/// Sparse MoE FFN block for Qwen3.6-MoE.
///
/// Forward pass:
/// 1. `shared = shared_expert(x) * sigmoid(shared_expert_gate(x))`
/// 2. `probs  = softmax(gate(x), -1)`
/// 3. `(w, idx) = topk(probs, num_experts_per_tok)` then renormalise
/// 4. `routed = sum_k w[k] * experts[idx[k]](x)`
/// 5. `out = shared + routed`
#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Qwen35MoeBlock {
    /// Linear router. `[num_experts, hidden]` — quantised independently
    /// at 8-bit on every Qwen3.6-MoE checkpoint we ship.
    #[quantizable]
    #[param]
    pub gate: MaybeQuantized<nn::Linear>,

    /// Routed switch_mlp: silu-gated SwiGLU, split `gate_proj` + `up_proj`.
    /// Field name matches the HF safetensors path
    /// (`mlp.switch_mlp.{gate,up,down}_proj.*`).
    #[quantizable]
    #[param]
    pub switch_mlp: SplitSwitchFfn<SwigluActivation>,

    /// Always-on dense shared expert (SwiGLU MLP).
    #[quantizable]
    #[param]
    pub shared_expert: SwigluMlp,

    /// Scalar gate on the shared-expert output. `[1, hidden]`,
    /// quantised at 8-bit independently of the body.
    #[quantizable]
    #[param]
    pub shared_expert_gate: MaybeQuantized<nn::Linear>,

    num_experts_per_tok: i32,
}

impl Qwen35MoeBlock {
    /// Build a freshly-initialised MoE block from dims.
    pub fn new(
        hidden_size: i32,
        moe_intermediate_size: i32,
        shared_expert_intermediate_size: i32,
        num_experts: i32,
        num_experts_per_tok: i32,
    ) -> Result<Self, Exception> {
        let gate = nn::LinearBuilder::new(hidden_size, num_experts)
            .bias(false)
            .build()?;
        let switch_mlp = SplitSwitchFfn::<SwigluActivation>::new(
            hidden_size,
            moe_intermediate_size,
            num_experts,
            false,
        )?;
        let shared_expert =
            SwigluMlp::new(hidden_size, shared_expert_intermediate_size, false)?;
        let shared_expert_gate = nn::LinearBuilder::new(hidden_size, 1)
            .bias(false)
            .build()?;
        Ok(Self {
            gate: MaybeQuantized::Original(gate),
            switch_mlp,
            shared_expert,
            shared_expert_gate: MaybeQuantized::Original(shared_expert_gate),
            num_experts_per_tok,
        })
    }
}

impl Module<&Array> for Qwen35MoeBlock {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // Shared branch.
        let shared = self.shared_expert.forward(x)?;
        let sg = self.shared_expert_gate.forward(x)?;
        let shared = shared.multiply(&sigmoid(&sg)?)?;

        // Router → softmax → topk.
        let logits = self.gate.forward(x)?;
        let probs = softmax_axis(&logits, -1, true)?;

        // argpartition along last axis; top-K largest land at
        // positions [num_experts - K .. num_experts].
        let kth: i32 = -self.num_experts_per_tok;
        let part = argpartition_axis(&probs, kth, -1)?;
        let part_len = *part.shape().last().expect("probs has trailing dim");
        let start = part_len - self.num_experts_per_tok;
        let top_k_indices = part.index((.., .., start..part_len));

        // Gather the matching probs, then renormalise so the K weights
        // sum to 1 — standard Qwen3-MoE / DeepSeek routing.
        let top_k_probs = take_along_axis(&probs, &top_k_indices, -1)?;
        let denom = sum_axis(&top_k_probs, -1, true)?;
        let top_k_weights = top_k_probs.divide(&denom)?;

        // Routed experts — fused down+combine on decode.
        let routed = self
            .switch_mlp
            .forward_with_combine(x, &top_k_indices, &top_k_weights)?;

        shared.add(&routed)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate.training_mode(mode);
        // SwigluMlp/SplitSwitchFfn have no internal training-state to
        // toggle (their MaybeQuantized fields are scalar slots).
        self.shared_expert_gate.training_mode(mode);
        let _ = mode;
    }
}

/// Concrete alias: Qwen3.6-MoE language model is the dense decoder
/// generic specialised on the MoE FFN.
pub type Qwen35MoeLanguageModel = LanguageModel<Qwen35MoeBlock>;

/// Re-export the dense cache helpers — qwen3_5 cache machinery handles
/// both hybrid (linear-attn + full-attn) layer kinds and works
/// unchanged on the MoE variant.
pub use qwen3_5::cache::{make_caches, LayerCache};
pub use qwen3_5::generation::{Generate, SamplingParams, StopCriteria};

/// End-to-end loader: parse config, build the model with per-tensor
/// quant overrides, sanitise + bind weights, hard-error on unbound
/// LM keys.
pub fn load_qwen3_5_moe_model(
    model_dir: impl AsRef<Path>,
) -> Result<Qwen35MoeLanguageModel, Error> {
    let model_dir = model_dir.as_ref();
    let cfg = ModelConfig::from_file(model_dir.join("config.json"))?;
    if !cfg.text_config.is_moe() {
        return Err(Error::Other(
            format!(
                "qwen3_5_moe loader: config {} declares num_experts={} \
                 (not MoE); use qwen3_5::weights::load_language_model",
                model_dir.display(),
                cfg.text_config.num_experts,
            )
            .into(),
        ));
    }

    let mut model = make_moe_language_model(&cfg.text_config)?;
    if let Some(q) = cfg.effective_quantization() {
        quantize_with_overrides(&mut model, q)?;
    }

    let weights = load_sanitized_weights(model_dir)?;

    let mut leftover: Vec<String> = Vec::new();
    {
        let mut params = model.parameters_mut().flatten();
        for (k, v) in weights {
            match bucket_key(k) {
                Bucketed::LanguageModel(p) => {
                    if let Some(slot) = params.get_mut(&*p) {
                        **slot = v;
                    } else {
                        leftover.push(format!("language_model.{p}"));
                    }
                }
                Bucketed::Vision(_) => {
                    // Some Qwen3.6-MoE checkpoints ship vision_tower.*
                    // weights even though the config declares no
                    // vision_config. The text-only loader drops them.
                }
                Bucketed::Other(p) => leftover.push(p),
            }
        }
    }

    if !leftover.is_empty() {
        leftover.sort();
        return Err(Error::Other(
            format!(
                "qwen3_5_moe loader: {} unbound key(s); first 8: {:?}",
                leftover.len(),
                &leftover.iter().take(8).collect::<Vec<_>>()
            )
            .into(),
        ));
    }
    eval_params(model.parameters()).map_err(Error::Exception)?;
    crate::loader::apply_post_load_memory_policy();
    Ok(model)
}

/// Build the MoE model with the dense `LanguageModel::new` generic,
/// supplying a `Qwen35MoeBlock` factory closure per layer.
fn make_moe_language_model(
    cfg: &qwen3_5::config::TextConfig,
) -> Result<Qwen35MoeLanguageModel, Error> {
    LanguageModel::<Qwen35MoeBlock>::new(cfg.clone(), |c| {
        Qwen35MoeBlock::new(
            c.hidden_size,
            c.moe_intermediate_size,
            c.shared_expert_intermediate_size,
            c.num_experts,
            c.num_experts_per_tok,
        )
    })
    .map_err(Error::Exception)
}

/// Quantise per-slot, honouring per-tensor overrides for `mlp.gate`
/// and `mlp.shared_expert_gate`. The rest of the model body uses
/// `(q.group_size, q.bits)`.
///
/// Implementation strategy: bulk-quantise the whole model at the body
/// settings first, then re-quantise just the override slots if their
/// `(group_size, bits)` differ. The override slots' weights have not
/// been bound yet (loader runs after this), so re-quantising the
/// freshly-initialised dense template is safe — it doesn't corrupt
/// any loaded data.
fn quantize_with_overrides(
    model: &mut Qwen35MoeLanguageModel,
    q: &QuantizationConfig,
) -> Result<(), Error> {
    // Stash the cfg before std::mem::replace consumes the model.
    let cfg = model.cfg.clone();
    let original = std::mem::replace(model, make_moe_language_model(&cfg)?);
    let body_q = original
        .try_into_quantized(q.group_size, q.bits)
        .map_err(Error::Exception)?;
    *model = body_q;

    // Per-layer override pass: re-build override slots at the
    // override (group_size, bits) if it differs from the body.
    for (layer_idx, layer) in model.model.layers.iter_mut().enumerate() {
        let raw_gate_prefix =
            format!("language_model.model.layers.{layer_idx}.mlp.gate");
        let (gate_gs, gate_bits) = q.for_path(&raw_gate_prefix);
        if (gate_gs, gate_bits) != (q.group_size, q.bits) {
            requantise_linear(&mut layer.mlp.gate, gate_gs, gate_bits)?;
        }

        let raw_sgate_prefix = format!(
            "language_model.model.layers.{layer_idx}.mlp.shared_expert_gate"
        );
        let (sgate_gs, sgate_bits) = q.for_path(&raw_sgate_prefix);
        if (sgate_gs, sgate_bits) != (q.group_size, q.bits) {
            requantise_linear(&mut layer.mlp.shared_expert_gate, sgate_gs, sgate_bits)?;
        }
    }
    Ok(())
}

/// Tiny helper: drop the existing `MaybeQuantized<Linear>` (whose
/// shape was carried over from `try_into_quantized` at the body
/// bits) and rebuild at the override `(group_size, bits)`. The
/// rebuilt linear is dense — the loader's `.weight` →
/// `.inner.weight` rewrite + scales/biases bind on the next pass.
fn requantise_linear(
    slot: &mut MaybeQuantized<nn::Linear>,
    group_size: i32,
    bits: i32,
) -> Result<(), Error> {
    let dummy = nn::LinearBuilder::new(1, 1)
        .bias(false)
        .build()
        .map_err(Error::Exception)?;
    let old = std::mem::replace(slot, MaybeQuantized::Original(dummy));
    // Re-extract the original Linear shape so we can rebuild at the
    // right dims before quantising at the override bits.
    let linear = match old {
        MaybeQuantized::Original(l) => l,
        MaybeQuantized::Quantized(q) => {
            // Already quantised at body bits: roll back via the
            // inner Linear weight shape. The actual weight values
            // get overwritten by the loader; only the slot's
            // (group_size, bits) and shape contract matters.
            let shape = q.inner.weight.as_ref().shape();
            // QuantizedLinear inner weight is packed uint32 with
            // shape `[out_features, in_features / pack_factor]`.
            // We need the original `[out_features, in_features]`.
            let out_features = shape[0];
            let body_pack = 32 / q.bits;
            let in_features = shape[1] * body_pack;
            nn::LinearBuilder::new(in_features, out_features)
                .bias(false)
                .build()
                .map_err(Error::Exception)?
        }
    };
    let requant = MaybeQuantized::Original(linear)
        .try_into_quantized(group_size, bits)
        .map_err(Error::Exception)?;
    *slot = requant;
    Ok(())
}

