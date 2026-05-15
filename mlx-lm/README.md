# mlx-lm

Decoder-language-model glue on top of `mlx-rs`: qwen3, qwen3.5 (hybrid
SSM + attention), llama 3.2. Provides the KV cache machinery, sampling
loop, prompt-cache save/load, and a few model loaders.

## KV cache variants

All caches implement `mlx_lm::cache::KeyValueCache`. The `attention`
trait method is the fused entry point — caches that hold packed
quantised state override it to skip K/V dequant on the hot path.

| # | Type | Memory | Decode speed (T=1024, Qwen3-1.7B-q4 + KV q8) | Quality | When to use |
|---|---|---|---:|---|---|
| 1 | `KVCache` (= `ConcatKeyValueCache`) | fp16 dense | baseline | exact | default; small contexts |
| 2 | `RotatingKVCache` | fp16 dense, fixed `max_size` | ≈ baseline | exact | sliding-window models |
| 3 | `QuantizedKVCache` (default) | 2–4× smaller | 193 tok/s | KL 0.20 (q4) / 0.006 (q8) | memory-constrained, T < 4K |
| 4 | `QuantizedKVCache::with_quantized_matmul()` | same as 3 | 186 tok/s @ T=1024, **152 tok/s @ T=8192 (+96% vs 3)** | same as 3 (modulo fp32 accum order) | **T ≥ 4K** — V2 LEAN |
| 5 | `…with_quantized_matmul().with_rotation(d, seed)` | same as 3 | ~70% of fp16 | better @ 4-bit | quality-critical 4-bit at long context |
| 6 | `…with_quantized_matmul().with_fused_kernel()` | same as 3 | **202 tok/s (+4.7% vs fp16 dequant)** | same as 4 | **n_q=1 decode, n_k ≤ 4096, bits ∈ {4,8}** |
| 7 | `TurboQuantKVCache` (V3, Lloyd-Max + QJL) | ~4.7× smaller (b ∈ {1..4}) | ~50% of V2 LEAN | KL 0.0012 (K3V2) | research / quality fallback |

Numbers from `~/Projects/mlx-rs-bench-results.md` (Apple M4 Max,
criterion 10×20s, decode-only methodology — prefill outside timing).

### Quick selection guide

- **Default** (`KVCache`) — until you hit memory or long-context
  bandwidth limits.
- **Short-context quant decode** — chain `with_quantized_matmul().with_fused_kernel()`.
  Single Metal dispatch, beats fp16 by ~5% at T=1024.
- **Long-context decode** — chain `with_quantized_matmul()`. Fused
  kernel falls back to ops-composed past `n_k = 4096` (TG-memory cap),
  V2 LEAN still wins by ~2× vs naive dequant-on-read.
- **4-bit quality** — add `.with_rotation(head_dim, seed)` for the
  V2 rotated path; ~5–10% perplexity improvement at 4-bit per
  sharpner/turboquant-mlx.
- **Research / paper-correct TurboQuant** — `TurboQuantKVCache` with
  Lloyd-Max codebook + QJL residual. Slower than V2 LEAN on Apple
  Silicon; quality is the win.

### Usage

```rust
use mlx_lm::cache::{KVCache, QuantizedKVCache, turboquant::cache::{TurboQuantConfig, TurboQuantKVCache}};

// 1. Default
let cache = KVCache::new();

// 3. Affine quant, dequant-on-read
let cache = QuantizedKVCache::new(); // group_size=64, bits=8

// 4. V2 LEAN (long context)
let cache = QuantizedKVCache::with_config(256, 64, 8)
    .with_quantized_matmul();

// 6. V2 LEAN + fused kernel (short context, n_q=1 decode)
let cache = QuantizedKVCache::with_config(256, 64, 4)
    .with_quantized_matmul()
    .with_fused_kernel();

// 7. TurboQuant V3 (K3V2 default)
let cfg = TurboQuantConfig::new(/* head_dim */ 128, /* seed */ 0);
let cache = TurboQuantKVCache::new(cfg)?;
```

Per-layer factories: `make_prompt_cache(num_layers, max_kv_size)` for
the default path; `make_turboquant_kv_cache(num_layers, head_dim, seed)`
for V3. Hybrid models (qwen3.5) build their own via
`models::qwen3_5::cache::make_caches` / `make_caches_with_tq`.

### Worked examples

Runnable binaries under `examples/lm/src/bin/`:

- `kv_v2_lean_fused` — V2 LEAN + fused qsdpa kernel (short-context).
- `kv_v2_lean_long_context` — V2 LEAN at T=8192 (fused kernel falls back).
- `kv_turboquant_v3` — TurboQuant V3 K3V2.

Test/bench references (no model loading needed for the bench helper):

- `mlx-lm/benches/lm_decode.rs::maybe_bench_qwen3_kv_decode_only` —
  end-to-end V2 LEAN + fused construction, prefill outside criterion
  timing band.
- `mlx-lm/tests/quantized_kv_parity.rs` — quality parity (fp16 vs
  dequant vs V2 LEAN) on Qwen3-1.7B-bf16.
- `mlx-lm/tests/turboquant_parity.rs` — V3 K3V2 quality validation.

Both test files are `#[ignore]`-gated; they require Qwen3-1.7B-bf16
in the bench cache.

## Prompt cache save/load

`save_prompt_cache(path, caches, extra)` / `load_prompt_cache(path)`
match the Python `mlx_lm.models.cache` wire format (safetensors with
`layer.{i}.{slot}` arrays + `layer.{i}.{key}` metadata). All seven
cache types round-trip.

## Models

- `models::qwen3` — Qwen3 base (0.6B / 1.7B / …)
- `models::llama` — Llama 3.2 (1B / 3B)
- `models::qwen3_5` — Qwen3.5 hybrid SSM + attention (4B / 9B);
  includes vision tower

Each exposes `load_*_model(&Path)` and a `Generate<C: KeyValueCache>`
streaming iterator. Qwen3.5 also has `Generate::with_caches(...)` for
mixed cache types (e.g. TurboQuant on the attention layers, default
linear-attn cache on the SSM layers).

## Benchmarks

```sh
cargo bench -p mlx-lm --bench lm_decode
```

See `mlx-lm/benches/README.md` for cell filtering, cache-dir config,
and methodology notes.
