use std::{collections::HashMap, path::Path};

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn,
    ops::indexing::{IndexOp, NewAxis},
    quantization::{MaybeQuantized, Quantizable as _},
    transforms::async_eval,
    Array,
};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::{
    cache::KeyValueCache,
    error::Error,
    quantization::{resolve_quantization, QuantizationConfig},
    utils::rope::{initialize_rope, FloatOrString, RopeVariant},
};

pub use crate::nn::{AttentionInput, ModelInput};

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub num_key_value_heads: i32,
    pub max_position_embeddings: i32,
    pub rope_theta: f32,
    pub head_dim: i32,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    pub rope_scaling: Option<HashMap<String, FloatOrString>>,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
    #[serde(default)]
    pub quantization_config: Option<QuantizationConfig>,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Attention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub scale: f32,

    #[quantizable]
    #[param]
    pub q_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub k_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub v_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub rope: RopeVariant,
}

impl Attention {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;

        let head_dim = args.head_dim;
        let scale = (head_dim as f32).sqrt().recip();

        let q_proj = nn::LinearBuilder::new(dim, n_heads * head_dim)
            .bias(args.attention_bias)
            .build()?;
        let k_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(args.attention_bias)
            .build()?;
        let v_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(args.attention_bias)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, dim)
            .bias(args.attention_bias)
            .build()?;

        let rope = initialize_rope(
            head_dim,
            args.rope_theta,
            false,
            &args.rope_scaling,
            args.max_position_embeddings,
        )?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            scale,
            q_proj: MaybeQuantized::Original(q_proj),
            k_proj: MaybeQuantized::Original(k_proj),
            v_proj: MaybeQuantized::Original(v_proj),
            o_proj: MaybeQuantized::Original(o_proj),
            rope,
        })
    }
}

impl<C> Module<AttentionInput<'_, C>> for Attention
where
    C: KeyValueCache + Default,
{
    type Output = Array;

    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, mut cache, .. } = input;

        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];

        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        let mut queries = queries
            .reshape(&[B, L, self.n_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut keys = keys
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let values = values
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let output = if let Some(cache) = cache.as_mut() {
            let q_input = nn::RopeInputBuilder::new(&queries)
                .offset(cache.offset())
                .build()?;
            queries = self.rope.forward(q_input)?;
            let k_input = nn::RopeInputBuilder::new(&keys)
                .offset(cache.offset())
                .build()?;
            keys = self.rope.forward(k_input)?;

            // Dispatch through the cache so quantised caches can fuse
            // update + attention without dequantising K.
            cache.attention(&queries, keys, values, self.scale, mask)?
        } else {
            queries = self.rope.forward(nn::RopeInput::new(&queries))?;
            keys = self.rope.forward(nn::RopeInput::new(&keys))?;
            mlx_rs::fast::scaled_dot_product_attention(
                queries,
                keys,
                values,
                self.scale,
                mask.map(mlx_rs::fast::ScaledDotProductAttentionMask::Array),
                None,
            )?
        };

        let output = output.transpose_axes(&[0, 2, 1, 3])?.reshape(&[B, L, -1])?;
        self.o_proj.forward(&output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
        <RopeVariant as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
    }
}

/// Re-export the canonical SwiGLU MLP at the historical path.
/// Llama's `Mlp::new(dim, hidden_dim, mlp_bias)` is `SwigluMlp::new(...,
/// bias: mlp_bias)`.
pub use crate::nn::SwigluMlp as Mlp;

#[derive(Debug, ModuleParameters, Quantizable)]
pub struct TransformerBlock {
    pub num_attention_heads: i32,
    pub hidden_size: i32,

    #[quantizable]
    #[param]
    pub self_attn: Attention,

    #[quantizable]
    #[param]
    pub mlp: Mlp,

    #[param]
    pub input_layernorm: nn::RmsNorm,

    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

impl TransformerBlock {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let num_attention_heads = args.num_attention_heads;
        let hidden_size = args.hidden_size;

        let self_attn = Attention::new(args)?;
        let mlp = Mlp::new(args.hidden_size, args.intermediate_size, args.mlp_bias)?;
        let input_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;

        Ok(Self {
            num_attention_heads,
            hidden_size,
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

impl<C> Module<AttentionInput<'_, C>> for TransformerBlock
where
    C: KeyValueCache + Default,
{
    type Output = Array;

    type Error = Exception;

    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, cache, .. } = input;

        let normalized = self.input_layernorm.forward(x)?;
        let self_attn_input = AttentionInput::plain(&normalized, mask, cache);
        let r = self.self_attn.forward(self_attn_input)?;
        let h = x.add(r)?;

        let r = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&h)?)?;
        h.add(r)
    }

    fn training_mode(&mut self, mode: bool) {
        <Attention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
    }
}

#[derive(Debug, ModuleParameters, Quantizable)]
pub struct LlamaModel {
    pub vocab_size: i32,
    pub num_hidden_layers: i32,

    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,

    #[quantizable]
    #[param]
    pub layers: Vec<TransformerBlock>,

    #[param]
    pub norm: nn::RmsNorm,
}

impl LlamaModel {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        assert!(args.vocab_size.is_positive());

        let vocab_size = args.vocab_size;
        let num_hidden_layers = args.num_hidden_layers;

        let embed_tokens = nn::Embedding::new(args.vocab_size, args.hidden_size)?;
        let layers = (0..num_hidden_layers)
            .map(|_| TransformerBlock::new(args))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;

        Ok(Self {
            vocab_size,
            num_hidden_layers,
            embed_tokens: MaybeQuantized::Original(embed_tokens),
            layers,
            norm,
        })
    }
}

impl<C> Module<ModelInput<'_, C>> for LlamaModel
where
    C: KeyValueCache + Default,
{
    type Output = Array;

    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let ModelInput {
            inputs,
            mask,
            cache,
        } = input;

        let mut h = self.embed_tokens.forward(inputs)?;

        crate::nn::ensure_cache_populated(cache, self.layers.len());

        // Cache-aware mask: shape `[L_q, cache.offset() + L_q]`, matching
        // the K/V history that update_and_fetch returns to attention.
        let mask = match mask {
            Some(m) => Some(m.clone()),
            None => crate::utils::create_attention_mask(&h, cache)?
                .map(|m| m.as_dtype(h.dtype()))
                .transpose()?,
        };

        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            let layer_input = AttentionInput::plain(&h, mask.as_ref(), c.as_mut());
            h = layer.forward(layer_input)?;
        }

        self.norm.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            <TransformerBlock as Module<AttentionInput<'_, C>>>::training_mode(layer, mode);
        }
        self.norm.training_mode(mode);
    }
}

#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Model {
    pub args: ModelArgs,

    #[quantizable]
    #[param]
    pub model: LlamaModel,

    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Exception> {
        let model = LlamaModel::new(&args)?;
        let lm_head = if !args.tie_word_embeddings {
            Some(MaybeQuantized::Original(
                nn::LinearBuilder::new(args.hidden_size, args.vocab_size)
                    .bias(false)
                    .build()?,
            ))
        } else {
            None
        };

        Ok(Self {
            args,
            model,
            lm_head,
        })
    }

    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }

    /// Number of transformer layers — the length any per-layer cache
    /// `Vec<Option<C>>` must have.
    pub fn layer_count(&self) -> usize {
        self.args.num_hidden_layers as usize
    }

    /// Per-head dimension. Required by quantised KV caches whose state
    /// arrays are shaped `[B, H, S, D]`.
    pub fn head_dim(&self) -> i32 {
        self.args.head_dim
    }
}

impl<C> Module<ModelInput<'_, C>> for Model
where
    C: KeyValueCache + Default,
{
    type Output = Array;

    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let out = self.model.forward(input)?;

        match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(&out),
            None => match &mut self.model.embed_tokens {
                MaybeQuantized::Original(embed_tokens) => embed_tokens.as_linear(&out),
                MaybeQuantized::Quantized(q_embed_tokens) => q_embed_tokens.as_linear(&out),
            },
        }
    }

    fn training_mode(&mut self, mode: bool) {
        <LlamaModel as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

pub fn load_llama_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    crate::loader::load_tokenizer(model_dir)
}

pub fn get_llama_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    crate::loader::load_config(model_dir)
}

pub fn load_llama_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let model_args = get_llama_model_args(model_dir)?;
    let quant =
        resolve_quantization(&model_args.quantization, &model_args.quantization_config).cloned();
    let mut model = Model::new(model_args)?;
    if let Some(q) = quant {
        model = model.try_into_quantized(q.group_size, q.bits)?;
    }

    crate::loader::load_sharded(&mut model, model_dir)?;
    Ok(model)
}

pub use crate::sampler::sample;

pub struct Generate<'a, C> {
    model: &'a mut Model,
    cache: &'a mut Vec<Option<C>>,
    temp: f32,
    state: GenerateState<'a>,
}

impl<'a, C> Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    pub fn new(
        model: &'a mut Model,
        cache: &'a mut Vec<Option<C>>,
        temp: f32,
        prompt_token: &'a Array,
    ) -> Self {
        Self {
            model,
            cache,
            temp,
            state: GenerateState::Prefill { prompt_token },
        }
    }
}

pub enum GenerateState<'a> {
    Prefill { prompt_token: &'a Array },
    /// `pending` is the next token to hand out; its predecessor has
    /// already been returned to the caller.
    Decode { pending: Array },
}

use crate::tri;

impl<'a, C> Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    fn step(&mut self, inputs: &Array) -> Result<Array, Exception> {
        let input = ModelInput {
            inputs,
            mask: None,
            cache: self.cache,
        };
        let logits = self.model.forward(input)?;
        sample(&logits.index((.., -1, ..)), self.temp)
    }
}

impl<'a, C> Iterator for Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        match std::mem::replace(
            &mut self.state,
            GenerateState::Decode {
                pending: Array::from_int(0),
            },
        ) {
            GenerateState::Prefill { prompt_token } => {
                let y0 = tri!(self.step(prompt_token));
                tri!(async_eval([&y0]));
                let inputs = y0.index((.., NewAxis));
                let y1 = tri!(self.step(&inputs));
                tri!(async_eval([&y1]));
                self.state = GenerateState::Decode { pending: y1 };
                Some(Ok(y0))
            }
            GenerateState::Decode { pending } => {
                let inputs = pending.index((.., NewAxis));
                let next_y = tri!(self.step(&inputs));
                tri!(async_eval([&next_y]));
                self.state = GenerateState::Decode { pending: next_y };
                Some(Ok(pending))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{env::home_dir, fs};

    use lazy_static::lazy_static;
    use mlx_rs::{
        ops::indexing::{IndexOp, NewAxis},
        transforms::eval,
        Array,
    };

    use crate::{
        cache::KVCache,
        models::llama::{load_llama_model, load_llama_tokenizer},
    };

    /// Resolve the HuggingFace cache directory to the actual snapshot path.
    /// The structure is:
    ///   models--<org>--<name>/
    ///     refs/
    ///       main  (contains the commit hash)
    ///     snapshots/
    ///       <commit_hash>/  (actual model files)
    fn resolve_hf_cache_dir(model_cache_dir: &str) -> String {
        let refs_main = std::path::Path::new(model_cache_dir)
            .join("refs")
            .join("main");
        let commit_hash = fs::read_to_string(&refs_main)
            .unwrap_or_default()
            .trim()
            .to_string();
        std::path::Path::new(model_cache_dir)
            .join("snapshots")
            .join(commit_hash)
            .to_string_lossy()
            .into_owned()
    }

    lazy_static! {
        static ref CACHED_TEST_MODEL_DIR: String = {
            let cache_dir = home_dir()
                .map(|p| {
                    p.join(".cache")
                        .join("huggingface")
                        .join("hub")
                        .join("models--meta-llama--Llama-3.2-1B-Instruct")
                        .to_string_lossy()
                        .into_owned()
                })
                .unwrap_or_default();

            resolve_hf_cache_dir(&cache_dir)
        };
        static ref CACHED_QUANT_TEST_MODEL_DIR: String = {
            let cache_dir = home_dir()
                .map(|p| {
                    p.join(".cache")
                        .join("huggingface")
                        .join("hub")
                        .join("models--mlx-community--Llama-3.2-1B-Instruct-4bit")
                        .to_string_lossy()
                        .into_owned()
                })
                .unwrap_or_default();

            resolve_hf_cache_dir(&cache_dir)
        };
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_llama_model() {
        use mlx_rs::module::ModuleParameters;

        let model_dir = CACHED_TEST_MODEL_DIR.as_str();
        let model_args = super::get_llama_model_args(model_dir).unwrap();
        let model = super::Model::new(model_args).unwrap();

        // Print some model parameter keys
        let params = model.parameters().flatten();
        let mut param_keys: Vec<_> = params.keys().map(|k| k.to_string()).collect();
        param_keys.sort();
        println!("=== Model parameter keys (first 20) ===");
        for key in param_keys.iter().take(20) {
            println!("  {key}");
        }

        // Print some safetensor keys
        let weights_path = std::path::Path::new(model_dir).join("model.safetensors");
        let loaded = mlx_rs::Array::load_safetensors(&weights_path).unwrap();
        let mut weight_keys: Vec<_> = loaded.keys().map(|k| k.to_string()).collect();
        weight_keys.sort();
        println!("=== Safetensor weight keys (first 20) ===");
        for key in weight_keys.iter().take(20) {
            println!("  {key}");
        }

        // Find unmatched keys
        let param_set: std::collections::HashSet<_> = param_keys.iter().collect();
        let weight_set: std::collections::HashSet<_> = weight_keys.iter().collect();
        let unloaded: Vec<_> = weight_set.difference(&param_set).collect();
        let missing: Vec<_> = param_set.difference(&weight_set).collect();
        println!(
            "=== Weight keys NOT in model params ({}) ===",
            unloaded.len()
        );
        for key in unloaded.iter().take(10) {
            println!("  {key}");
        }
        println!(
            "=== Model param keys NOT in weights ({}) ===",
            missing.len()
        );
        for key in missing.iter().take(10) {
            println!("  {key}");
        }
        println!(
            "Total model params: {}, Total weight keys: {}",
            param_keys.len(),
            weight_keys.len()
        );
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_tokenizer() {
        let tokenizer = load_llama_tokenizer(CACHED_TEST_MODEL_DIR.as_str()).unwrap();

        let _encoding = tokenizer.encode("Hello, world!", true).unwrap();
    }

    /// Regression guard for the quantised-checkpoint loader path: without
    /// `try_into_quantized` before `load_safetensors`, the packed-uint32
    /// weights overwrite the unquantised `Linear.weight` slot and the first
    /// forward fires `[rms_norm] weight has K elements but x's last dim is D`.
    #[test]
    #[ignore = "requires local quantised model files"]
    fn quantized_llama_model_loads_and_forwards() {
        use mlx_rs::module::Module;

        let mut model = load_llama_model(CACHED_QUANT_TEST_MODEL_DIR.as_str()).unwrap();
        let prompt = Array::from_slice(&[1i32, 2, 3, 4], &[1, 4]);
        let mut cache: Vec<Option<KVCache>> = Vec::new();
        let input = super::ModelInput {
            inputs: &prompt,
            mask: None,
            cache: &mut cache,
        };
        let logits = Module::forward(&mut model, input).unwrap();
        eval([&logits]).unwrap();
        assert_eq!(logits.shape()[2], model.args.vocab_size);
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_and_run_llama_with_concat_cache() {
        let tokenizer = load_llama_tokenizer(CACHED_TEST_MODEL_DIR.as_str()).unwrap();
        let mut model = load_llama_model(CACHED_TEST_MODEL_DIR.as_str()).unwrap();

        let prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
        let encoding = tokenizer.encode(prompt, false).unwrap();
        let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);
        let mut cache = Vec::new();

        let eos_token_id = 128001u32;
        let eot_token_id = 128009u32;

        let mut token_ids = Vec::new();
        let generate = super::Generate::<KVCache>::new(
            &mut model,
            &mut cache,
            0.0,
            &prompt_tokens,
        );
        for (token, _ntoks) in generate.zip(0..50) {
            let token = token.unwrap();
            eval([&token]).unwrap();
            let token_id = token.item::<u32>();
            print!("[{}]", token_id);
            if token_id == eos_token_id || token_id == eot_token_id {
                break;
            }
            token_ids.push(token_id);
        }
        println!();

        let output = tokenizer.decode(&token_ids, true).unwrap();
        println!("Response: {output}");
        println!("------");
    }
}
