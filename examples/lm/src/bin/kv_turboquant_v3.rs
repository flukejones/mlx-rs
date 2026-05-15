//! TurboQuant V3 — paper-correct (ICLR 2026, arxiv 2504.19874).
//!
//! What:  `TurboQuantKVCache` with K3V2 default config.
//! Why:   data-oblivious vector quantisation of K (3-bit Lloyd-Max +
//!        1-bit QJL sign) and V (2-bit affine). ~4.7× memory reduction
//!        vs fp16, top-32 KL = 0.0012 on Qwen3-1.7B-bf16.
//!        Currently slower than V2 LEAN on Apple Silicon (~50%); use
//!        this when quality is the constraint, not speed.
//! How:   different cache type from V2 — own struct, own
//!        `KeyValueCache::attention` override routing through the
//!        `tq_attention_score` fused kernel. From the model's POV it's
//!        still just `cache.attention(...)`.

use std::path::Path;

use mlx_lm::{
    cache::turboquant::cache::{TurboQuantConfig, TurboQuantKVCache},
    models::qwen3::{load_qwen3_model, Generate},
};
use mlx_rs::{
    ops::indexing::{IndexOp, NewAxis},
    transforms::eval,
    Array,
};

const MODEL_DIR: &str = "./cache/mlx-community/Qwen3-1.7B-bf16";
const PROMPT_TOKEN_IDS: &[u32] = &[
    1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009,
];
const DECODE_TOKENS: i32 = 64;
const HEAD_DIM: i32 = 128;

fn main() -> anyhow::Result<()> {
    let mut model = load_qwen3_model(Path::new(MODEL_DIR))?;
    let num_layers = model.layer_count();

    // TurboQuantConfig::new(head_dim, seed) gives the paper-default K3V2
    // (3-bit keys, 2-bit values, group_size=32, 128-token recent buffer).
    // Vary `seed` per layer if you want independent rotation matrices;
    // re-using one seed across layers is fine for inference.
    let mut caches: Vec<Option<TurboQuantKVCache>> = (0..num_layers)
        .map(|_| {
            let cfg = TurboQuantConfig::new(HEAD_DIM, 0);
            Some(TurboQuantKVCache::new(cfg).expect("TurboQuantKVCache::new"))
        })
        .collect();

    let prompt = Array::from(PROMPT_TOKEN_IDS).index(NewAxis);
    let gen = Generate::<TurboQuantKVCache>::new(&mut model, &mut caches, 0.0, &prompt);

    let mut tokens = Vec::new();
    for (tok, n) in gen.zip(0..DECODE_TOKENS) {
        tokens.push(tok?);
        if n == 0 {
            eval(&tokens)?;
        }
    }
    eval(&tokens)?;
    println!("decoded {} tokens with TurboQuant V3 K3V2", tokens.len());
    Ok(())
}
