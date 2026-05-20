# mlx-lm

Decoder-language-model glue on top of `mlx-rs`: gemma 4 (dense + MoE +
per-layer-input), qwen 3.5 / 3.6 (hybrid SSM + attention + VL),
qwen 3.5-MoE / 3.6-MoE (35B-A3B). Provides the KV cache machinery,
sampling loop, prompt-cache save/load, sliding-window prefill
chunking, the unified `UserInput` / `LMInput` / `LanguageModel`
surface, and the `mlx_lm::load` / `mlx_lm::generate` entry points.

## KV cache variants

All caches implement `mlx_lm::cache::KeyValueCache`. The `attention`
trait method is the fused entry point — caches that hold packed
quantised state override it to skip K/V dequant on the hot path.

Numbers below: Qwen3-1.7B-q4 weights + KV q8 on Apple M4 Max,
decode-only methodology (criterion 10×20s, prefill outside the timing
band). See `~/Projects/mlx-rs-bench-results.md` for full bench tables
across model families, KV bits, and bf16 / q4 weight bases.

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
  than packed-matmul depending on context (237 / 244 / 218 at
  T=1024 / 2048 / 4096). Quant is only worth it for **memory** here.
- **T ≤ 4K, need 3-4× smaller KV** → chain
  `with_quantized_matmul().with_fused_kernel()`. ~85% of fp16 speed
  at T=1024 (201 vs 237).
- **T ≥ 8K** → chain `with_quantized_matmul()`. At T=8192 packed-matmul
  beats fp16 by ~3% (148 vs 143) and dequant by ~90%. Fused kernel
  falls back to ops-composed past `n_k = 4096` (perf crossover —
  mlx's tiled `quantized_matmul` wins beyond that point).
- **4-bit quality boost** → add `.with_rotation(head_dim, seed)`.
  Random orthogonal Π pre-quantize gives KL 0.039 vs 0.20 unrotated
  on Qwen3-1.7B-bf16 (5.2× tighter top-32 distribution). Throughput
  cost: ~14% at T=1024, only ~5% at T=8192 (basically free at long
  context).

### Steel-attention prefill (Phase A)

`KVCache::with_steel_prefill()` routes the `n_q > 1` prefill path
through the upstream `mlx::steel` flash-attention tiled kernel. Active
only for head_dim ∈ {128, 256} and unmasked dispatch; falls back to
`fast::SDPA` for decode (n_q = 1), masked, or unsupported head dims.

```rust
let cache = KVCache::new().with_steel_prefill();
```

Unlocks the long-blocked **head_dim = 256** path needed for Qwen 3.6,
Gemma 3, and Gemma 4 local layers. Parity-tested vs `fast::SDPA`
across aligned, unaligned, GQA, and bf16 shapes
(`mlx-lm/src/steel_attention/tests.rs`).

### Usage

```rust
use mlx_lm::cache::{KVCache, QuantizedKVCache};

// 1. Default
let cache = KVCache::new();

// 1a. Steel-attention prefill (D ∈ {128, 256}, unmasked)
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
Hybrid models (qwen3.5) build their own via
`models::qwen3_5::cache::make_caches`.

### Worked examples

Runnable binaries under `examples/lm/src/bin/`:

- `kv_packed_matmul_fused` — packed-matmul + fused qsdpa kernel (short-context).
- `kv_packed_matmul_long_context` — packed-matmul at T=8192 (fused kernel falls back).

Test/bench references (no model loading needed for the bench helper):

- `mlx-lm/benches/lm_decode.rs::maybe_bench_qwen3_kv_decode_only` —
  end-to-end packed-matmul + fused construction, prefill outside criterion
  timing band.
- `mlx-lm/tests/quantized_kv_parity.rs` — quality parity (fp16 vs
  dequant vs packed-matmul vs rotated) on Qwen3-1.7B-bf16.

`#[ignore]`-gated tests require Qwen3-1.7B-bf16 in the bench cache.

## Prompt cache save/load

`save_prompt_cache(path, caches, extra)` / `load_prompt_cache(path)`
match the Python `mlx_lm.models.cache` wire format (safetensors with
`layer.{i}.{slot}` arrays + `layer.{i}.{key}` metadata). All six
cache variants round-trip.

## Models

- `models::gemma4` — Gemma 4 dense + MoE + per-layer-input
  (E2B / E4B / 26B-A4B / 31B). Sliding-window attention every
  Nth layer; chunked prefill when prompt > window.
- `models::qwen3_5` — Qwen 3.5 / 3.6 hybrid SSM + attention
  (4B / 9B / 27B). Includes the vision tower (Qwen 3.5-VL,
  Qwen 3.6-VL, chandra-ocr-2).
- `models::qwen3_5_moe` — Qwen 3.6-MoE (35B-A3B) atop the
  qwen3_5 hybrid spine with switch-FFN experts and per-tensor
  quantisation overrides.

All loaders are private. Consumers use
`mlx_lm::load(&Path) -> ModelContext` +
`mlx_lm::generate(&mut ctx, UserInput, GenerateParams, on_token)`.
Family dispatch happens inside `load` via `config.json::model_type`.

## Benchmarks

```sh
cargo bench -p mlx-lm --bench lm_decode
```

See `mlx-lm/benches/README.md` for cell filtering, cache-dir config,
and methodology notes.
