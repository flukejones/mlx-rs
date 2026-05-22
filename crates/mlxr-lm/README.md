# mlxr-lm

Language-model runtime + model families on top of [`mlxr`](../mlxr/).
Provides the KV cache machinery, sampler, prompt-cache save/load,
sliding-window prefill chunking, the unified `UserInput` / `LMInput` /
`LanguageModel` surface, and the `mlxr_lm::load` / `mlxr_lm::generate`
entry points.

Default features: `qwen3_5`, `gemma4`, `image`.

- **`qwen3_5`** — Qwen 3.5 / 3.6 dense + MoE (hybrid GDN + full-attn,
  optional MTP self-speculative decode). Cascades `mlxr/layers`.
- **`gemma4`** — Gemma 4 text (dense + MoE + per-layer-input variants;
  sliding-window attention; chunked prefill). Cascades `mlxr/layers`.
- **`image`** — modality flag. Pulls the `image` crate dep and unlocks
  per-family vision towers (today: Qwen 3-VL via `qwen3_5::image`,
  covering Qwen 3.5-VL / 3.6-VL / chandra-ocr-2).
- `audio` / `video` — reserved for future families.

## Install

```toml
[dependencies]
mlxr-lm = "0.27"
```

Disable families you don't need:

```toml
mlxr-lm = { version = "0.27", default-features = false, features = ["qwen3_5"] }
```

## Quick start

```rust
use std::path::PathBuf;
use mlxr_lm::{chat_template::ChatMessage, generate, load, GenerateParams, UserInput};

let home = std::env::var("HOME").expect("HOME");
let dir = PathBuf::from(home).join(".cache/mlx-community/Qwen3-1.7B-4bit");
let mut ctx = load(&dir)?;

let input = UserInput::chat(vec![ChatMessage::user("Explain MLX in one sentence.")]);
generate(&mut ctx, input, GenerateParams::default(), &mut |_id, delta| {
    print!("{delta}");
    std::ops::ControlFlow::Continue(())
})?;
```

## KV cache variants

All caches implement [`mlxr_lm::cache::KeyValueCache`]. The `attention`
trait method is the fused entry point — caches that hold packed
quantised state override it to skip K/V dequant on the hot path.

Numbers below: Qwen3-1.7B-q4 weights + KV q8 on Apple M4 Max,
decode-only methodology (criterion 10×20s, prefill outside the timing
band).

| # | Type | Memory | T=1024 | T=4096 | T=8192 | Quality | When to use |
|---|---|---|---:|---:|---:|---|---|
| 1 | `KVCache` (fp16 KV) | fp16 dense | **237** | **218** | 143 | exact | **short / medium context (T≤4K)**; memory not a constraint |
| 2 | `RotatingKVCache` | fp16, fixed `max_size` | ≈ #1 | ≈ #1 | ≈ #1 | exact | sliding-window models |
| 3 | `QuantizedKVCache` (dequant-on-read) | 2–4× smaller | 193 | — | 78 | KL 0.20 (q4) / 0.006 (q8) | memory-constrained, but slower than #1 |
| 4 | `with_quantized_matmul()` | same as 3 | 201 | 173 | **148** | same as 3 (modulo fp32 accum order) | **long context (T ≥ 6-8K)** |
| 5 | `…with_rotation(d, seed)` | same as 3 | 173 | — | 141 | **KL 0.039 @ 4-bit (5.2× tighter than unrotated)** | quality-critical 4-bit |
| 6 | `…with_quantized_matmul().with_fused_kernel()` | same as 3 | 201 | (#4 fallback) | (#4 fallback) | same as 4 | **n_q=1 decode, n_k ≤ 4096, bits ∈ {4,8}** |

The fp16-vs-packed-matmul **crossover is between T=4096 and T=8192**.
fp16 KV holds the lead at T=2048 (244 vs 174) and T=4096 (218 vs 173).

### Quick selection guide

- **T ≤ 4K, no memory pressure** → `KVCache` (fp16). 15-29% faster
  than packed-matmul depending on context.
- **T ≤ 4K, need 3-4× smaller KV** → chain
  `with_quantized_matmul().with_fused_kernel()`. ~85% of fp16 speed
  at T=1024.
- **T ≥ 8K** → chain `with_quantized_matmul()`. Packed-matmul beats
  fp16 by ~3% at T=8192 and dequant by ~90%. Fused kernel falls back
  to ops-composed past `n_k = 4096`.
- **4-bit quality boost** → add `.with_rotation(head_dim, seed)`.
  Random orthogonal Π pre-quantize gives KL 0.039 vs 0.20 unrotated
  (5.2× tighter top-32 distribution). Throughput cost: ~14% at T=1024,
  ~5% at T=8192.

### Flash-attention prefill

`KVCache::with_steel_prefill()` routes the `n_q > 1` prefill path
through the upstream MLX `steel` flash-attention tiled kernel
(method name kept for parity with the upstream symbol). Active only
for `head_dim ∈ {128, 256}` and unmasked dispatch; falls back to
`fast::SDPA` for decode (`n_q = 1`), masked, or unsupported head dims.

```rust
let cache = KVCache::new().with_steel_prefill();
```

Unlocks the `head_dim = 256` path needed for Qwen 3.6, Gemma 3, and
Gemma 4 local layers. Parity-tested vs `fast::SDPA` across aligned,
unaligned, GQA, and bf16 shapes (see
[`src/attention/tests.rs`](src/attention/tests.rs)).

### Usage

```rust
use mlxr_lm::cache::{KVCache, QuantizedKVCache};

// 1. Default
let cache = KVCache::new();

// 1a. Flash-attention prefill (D ∈ {128, 256}, unmasked)
let cache = KVCache::new().with_steel_prefill();

// 3. Affine quant, dequant-on-read
let cache = QuantizedKVCache::new(); // group_size=64, bits=8

// 4. Packed-matmul (long context)
let cache = QuantizedKVCache::with_config(256, 64, 8)
    .with_quantized_matmul();

// 5. Packed-matmul + rotation (better 4-bit quality)
let cache = QuantizedKVCache::with_config(256, 64, 4)
    .with_quantized_matmul()
    .with_rotation(/* head_dim */ 128, /* seed */ 0)?;

// 6. Packed-matmul + fused kernel (short context, n_q=1 decode)
let cache = QuantizedKVCache::with_config(256, 64, 4)
    .with_quantized_matmul()
    .with_fused_kernel();
```

Per-layer factory: `make_prompt_cache(num_layers, max_kv_size)`.
Hybrid models (qwen3_5) build their own via
`qwen3_5::text::cache::make_caches`.

## Prompt cache save/load

`save_prompt_cache(path, caches, extra)` /
`load_prompt_cache(path)` — safetensors file with
`layer.{i}.{slot}` arrays + `layer.{i}.{key}` metadata. All six
cache variants round-trip.

## Models

- `qwen3_5::text` — Qwen 3.5 / 3.6 hybrid SSM + attention (4B / 9B /
  27B / MoE 35B-A3B); dense + MoE adapters; MTP rejection-sampling.
- `qwen3_5::image` (requires the `image` feature) — Qwen 3-VL ViT
  tower + multimodal stitching; covers Qwen 3.5-VL, Qwen 3.6-VL,
  chandra-ocr-2.
- `gemma4::text` — Gemma 4 dense + MoE + per-layer-input
  (E2B / E4B / 26B-A4B / 31B). Sliding-window every Nth layer; chunked
  prefill when prompt > window.

Consumers go through `mlxr_lm::load(&Path) -> ModelContext` +
`mlxr_lm::generate(&mut ctx, UserInput, GenerateParams, on_token)`.
Family dispatch happens inside `load` via `config.json::model_type`.

## Benchmarks

```sh
cargo bench -p mlxr-lm --bench lm_decode
```

Cells filter via `MLX_LM_BENCH_ONLY=<family_label_prefix>`. Cache dir
via `MLX_LM_BENCH_CACHE`. See [`benches/`](benches/) for methodology
notes (cold + serial; `iter_custom` with prefill outside the timing
window).

## License

Dual-licensed under MIT and Apache 2.0.
