# lm examples

Runnable demos and tooling for `mlxr-lm`.

## Binaries

| Bin                | Purpose                                                                                                  |
|--------------------|----------------------------------------------------------------------------------------------------------|
| `lm` (`main.rs`)   | Minimal driver — load a Qwen 3 checkpoint and run one `mlxr_lm::generate` call. Useful as a smoke test.  |
| `generate`         | One-shot text completion with full CLI options (model dir, prompt, temperature, top-p, max-tokens, chat-template toggle); streams tokens to stdout. |
| `bench_with_temp`  | Wraps `cargo bench` with `macmon raw` sampling and emits a CSV + PNG of GPU/CPU temp and power vs time, with per-bench-cell boundaries marked. |

## Running

Each example expects a checkpoint already on disk. Convenient way to
populate the bench cache:

```sh
hf download mlx-community/Qwen3-1.7B-4bit \
  --local-dir ~/.cache/mlx-rs-bench/mlx-community/Qwen3-1.7B-4bit
```

Then:

```sh
cargo run --release --bin generate -- \
  --model ~/.cache/mlx-rs-bench/mlx-community/Qwen3-1.7B-4bit \
  --prompt "Explain MLX in one sentence."
```

For interactive chat see [`examples/chat/`](../chat/).

## Where to look next

- [`crates/mlxr-lm/README.md`](../../crates/mlxr-lm/README.md) — full
  KV cache variant table + selection guide.
- [`crates/mlxr-lm/benches/lm_decode.rs`](../../crates/mlxr-lm/benches/lm_decode.rs) —
  decode-only bench harness (criterion `iter_custom`, prefill outside
  the timing band). Run with `cargo bench -p mlxr-lm --bench lm_decode`.
- [`crates/mlxr-lm/tests/qwen3_6_35b_a3b_parity.rs`](../../crates/mlxr-lm/tests/qwen3_6_35b_a3b_parity.rs) —
  end-to-end loader + MTP decode parity test (`--ignored`; needs the
  checkpoint on disk).
