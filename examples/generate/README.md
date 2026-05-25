# generate

One-shot text completion against any `mlxr_lm` checkpoint.

Loads a checkpoint via `mlxr_lm::load`, drives a single
`mlxr_lm::generate` call against one prompt, streams tokens to stdout.

## Running

Populate the bench cache with a checkpoint, then run `generate`:

```sh
hf download mlx-community/Qwen3-1.7B-4bit \
  --local-dir ~/.cache/mlx-rs-bench/mlx-community/Qwen3-1.7B-4bit

cargo run --release --bin generate -- \
  --model ~/.cache/mlx-rs-bench/mlx-community/Qwen3-1.7B-4bit \
  --prompt "Explain MLX in one sentence."
```

Flags: `--model <dir>`, `--prompt <s>`, `--temp <f>` (default 0.0 = greedy),
`--top-p <f>`, `--max-tokens <n>` (default 256), `--no-chat-template`.

For interactive chat see [`examples/chat/`](../chat/).
