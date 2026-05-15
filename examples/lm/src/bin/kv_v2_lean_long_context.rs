//! V2 LEAN at long context — when the fused kernel falls back.
//!
//! What:  `QuantizedKVCache::with_quantized_matmul()` (no fused-kernel flag).
//! Why:   at `n_k > 4096` the fused kernel's threadgroup `scores` buffer
//!        would exceed Apple's 32 KB TG-memory budget, so it falls back
//!        to ops-composed. V2 LEAN still wins: at T=8192 it delivers
//!        ~152 tok/s vs ~77 tok/s for naive dequant-on-read (+96%).
//! How:   identical model code to the fused example — the cache's
//!        `attention` override picks the path internally. Setting
//!        `with_fused_kernel()` here is harmless (would just fall back),
//!        but omitting it makes the long-context-is-the-winner intent
//!        explicit at the call site.

use std::path::Path;

use mlx_lm::{
    cache::QuantizedKVCache,
    models::qwen3::{load_qwen3_model, Generate},
};
use mlx_rs::{
    ops::indexing::{IndexOp, NewAxis},
    transforms::eval,
    Array,
};

const MODEL_DIR: &str = "./cache/mlx-community/Qwen3-1.7B-4bit";
const LONG_PROMPT_LEN: usize = 8192;
const DECODE_TOKENS: i32 = 50;

fn main() -> anyhow::Result<()> {
    let mut model = load_qwen3_model(Path::new(MODEL_DIR))?;
    let num_layers = model.layer_count();

    let mut caches: Vec<Option<QuantizedKVCache>> = (0..num_layers)
        .map(|_| Some(QuantizedKVCache::with_config(256, 64, 8).with_quantized_matmul()))
        .collect();

    // Synthetic long prompt; in production this is whatever the
    // tokeniser produced. T=8192 is where V2 LEAN's bandwidth win is
    // most visible.
    let ids: Vec<u32> = (0..LONG_PROMPT_LEN as u32).map(|i| 1000 + (i % 100)).collect();
    let prompt = Array::from(&ids[..]).index(NewAxis);

    let gen = Generate::<QuantizedKVCache>::new(&mut model, &mut caches, 0.0, &prompt);

    let mut tokens = Vec::new();
    for (tok, n) in gen.zip(0..DECODE_TOKENS) {
        tokens.push(tok?);
        if n == 0 {
            eval(&tokens)?;
        }
    }
    eval(&tokens)?;
    println!(
        "decoded {} tokens at T={} with V2 LEAN (ops-composed at long ctx)",
        tokens.len(),
        LONG_PROMPT_LEN
    );
    Ok(())
}
