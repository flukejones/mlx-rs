//! Packed-matmul + fused qsdpa kernel — recommended **short-context
//! decode** path for quantised KV.
//!
//! What:  `QuantizedKVCache::with_quantized_matmul().with_fused_kernel()`.
//! Why:   single-dispatch Metal kernel (`softmax((Q @ K.T) * scale + mask) @ V`)
//!        with K/V held as packed `(wq, scales, biases)` triples — no K/V
//!        dequant on the decode hot path. Beats fp16 dequant-on-read by
//!        ~5% at T=1024 on Qwen3-1.7B-q4 (Apple M4 Max).
//! How:   the cache's `attention` override inspects each call's shapes
//!        (`n_q`, `bits`, `n_k`) and dispatches to the fused kernel when
//!        supported (`n_q == 1`, `bits ∈ {4, 8}`, `n_k ≤ 4096`); otherwise
//!        falls back to the ops-composed packed-matmul path automatically.
//!        Model code is unchanged — `cache.attention(...)` does the right thing.

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
const PROMPT_TOKEN_IDS: &[u32] = &[
    // a tiny synthetic prompt; replace with a real tokeniser in production.
    1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009,
];
const DECODE_TOKENS: i32 = 64;

fn main() -> anyhow::Result<()> {
    let mut model = load_qwen3_model(Path::new(MODEL_DIR))?;
    let num_layers = model.layer_count();

    // KV q8 (2× memory reduction, near-lossless). `with_config(step,
    // group_size, bits)` matches the default `QuantizedKVCache::new`.
    // The two builder flags chain — order doesn't matter:
    //   - with_quantized_matmul: keep K/V packed across score + attend,
    //     dispatch via mlx_rs::ops::quantized_matmul × 2.
    //   - with_fused_kernel: prefer the in-tree fused Metal kernel when
    //     supported; falls back to the above otherwise.
    let mut caches: Vec<Option<QuantizedKVCache>> = (0..num_layers)
        .map(|_| {
            Some(
                QuantizedKVCache::with_config(256, 64, 8)
                    .with_quantized_matmul()
                    .with_fused_kernel(),
            )
        })
        .collect();

    let prompt = Array::from(PROMPT_TOKEN_IDS).index(NewAxis);
    let gen = Generate::<QuantizedKVCache>::new(&mut model, &mut caches, 0.0, &prompt);

    let mut tokens = Vec::new();
    for (tok, n) in gen.zip(0..DECODE_TOKENS) {
        let tok = tok?;
        tokens.push(tok);
        // First step pays the prefill cost; eval forces the graph so the
        // next decode steps see a materialised cache.
        if n == 0 {
            eval(&tokens)?;
        }
    }
    eval(&tokens)?;
    println!("decoded {} tokens with packed-matmul + fused qsdpa", tokens.len());
    Ok(())
}
