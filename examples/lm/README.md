# lm examples

Small runnable demos of `mlx-lm`'s decode loop and KV-cache variants.

## Binaries

| Bin | Cache variant | Demonstrates |
|---|---|---|
| `lm` | `ConcatKeyValueCache` (fp16 dense) | minimal qwen3 chat-template + decode loop |
| `kv_v2_lean_fused` | `QuantizedKVCache` + V2 LEAN + fused qsdpa kernel | short-context quant decode (beats fp16 dequant by ~5% at T=1024) |
| `kv_v2_lean_long_context` | `QuantizedKVCache` + V2 LEAN | long-context quant decode (~+96% over dequant at T=8192; fused kernel falls back past `n_k=4096`) |
| `kv_turboquant_v3` | `TurboQuantKVCache` K3V2 | paper-correct Lloyd-Max + QJL (quality-first, slower than V2 LEAN on Apple Silicon) |

Each example is a single file with header comments explaining
*what / why / how* for its variant.

## Running

Each example expects the relevant model checkpoint already on disk at
`./cache/<repo-id>/`. Easy way to populate:

```sh
hf download mlx-community/Qwen3-1.7B-4bit --local-dir ./cache/mlx-community/Qwen3-1.7B-4bit
hf download mlx-community/Qwen3-1.7B-bf16 --local-dir ./cache/mlx-community/Qwen3-1.7B-bf16
```

Then:

```sh
cargo run --release --bin kv_v2_lean_fused
cargo run --release --bin kv_v2_lean_long_context
cargo run --release --bin kv_turboquant_v3
```

The synthetic prompts in the KV-cache demos are token-id arrays so the
examples have zero tokenizer setup; copy the cache-construction lines
into your own decode loop to use real prompts.

## Where to look next

- `mlx-lm/README.md` — full KV cache variant table + selection guide.
- `mlx-lm/benches/lm_decode.rs::maybe_bench_qwen3_kv_decode_only` —
  the bench cell that produced the numbers cited above.
- `mlx-lm/tests/quantized_kv_parity.rs` — quality validation of V2 LEAN
  vs dequant-on-read.
- `mlx-lm/tests/turboquant_parity.rs` — quality validation of V3.
