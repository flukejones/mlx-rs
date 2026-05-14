# mlx-lm benches

Criterion benches for the text-decoder models in `mlx_lm::models`.

Currently covers:

- `qwen3_decode_*` — `mlx_lm::models::qwen3` on small (0.6B) + large (1.7B)
  Qwen3 checkpoints × `{bf16, q8, q4}` quantisation tiers.
- `llama_decode_*` — `mlx_lm::models::llama` on small (1B) + large (3B)
  Llama-3.2-Instruct checkpoints × `{bf16, q8, q4}`.
- `qwen3_5_decode_{4b_q8,9b_q8}` — `mlx_lm::models::qwen3_5` on the
  `mlx-community/Qwen3.5-{4B,9B}` hybrid checkpoints (Gated DeltaNet +
  full-attention layers). Same architecture as chandra-ocr-2 (vision-free).
- `vision_prefill_chandra_q8` — ViT-24 forward on `tests/fixtures/qwen3_5/test_image.png`
  using the chandra-ocr-2 multimodal checkpoint (`jwindle47/chandra-ocr-2-8bit-mlx`,
  only public mlx conversion). One image per iteration. `head_dim = 72` falls
  outside MLX's fused SDPA set; sensitive to the
  `scaled_dot_product_attention_pad_to_fused` helper.

Each variant is benched with a 13-token short prompt and a 1024-token long
prompt, decoding 100 tokens per iteration; both prompts use synthetic
in-vocab ids (we measure tok/s, not generation quality).

## Run

```
cargo bench -p mlx-lm --bench lm_decode
```

Pass a filter to run a subset:

```
cargo bench -p mlx-lm --bench lm_decode -- qwen3_decode_small
cargo bench -p mlx-lm --bench lm_decode -- llama_decode_large_q4/short
```

## Model cache

Checkpoints are downloaded lazily on first use via the
[`hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli). Cache
root is resolved in order:

1. `$MLX_LM_BENCH_CACHE` — explicit override. Point at any pre-populated dir.
2. `$XDG_CACHE_HOME/mlx-rs-bench/`.
3. `$HOME/.cache/mlx-rs-bench/`.

Each repo lives at `<root>/<repo_id>/`. The bench uses `hf download
--local-dir <root>/<repo_id>`, which produces a flat mirror of the repo —
**not** Hugging Face's standard hash-addressed `~/.cache/huggingface/hub/`
layout. The flat layout is deliberate: `mlx_lm::models::*::load_*_model`
take a plain directory path and the bench can pass `<root>/<repo_id>`
directly without resolving `refs/main` → `snapshots/<sha>/`. The trade-off
is that the bench cache does not dedupe against the system HF cache.

If you already have these checkpoints elsewhere (e.g. `~/MLXModels/`),
point `MLX_LM_BENCH_CACHE` at that root and the bench will reuse them —
provided each checkpoint sits at `<root>/<repo_id>/` matching the IDs used
in the bench file.

Total disk after a full population is ~42 GB (text decoders ~16 GB +
Qwen3.5 ~22 GB + chandra-ocr-2 multimodal ~3 GB). Cells skip if `hf` is
missing or download fails; CI without `hf` will run zero cells rather
than fail. Partial checkpoints (an interrupted `hf download` leaving the
index file plus only some shards) are detected and reported with a resume
command — they're never silently treated as complete.

Set `MLX_LM_BENCH_NO_DOWNLOAD=1` to suppress downloads entirely — any cell
whose checkpoint isn't already cached will be dropped silently. Useful
when filtering to one cell so unrelated tiers don't pull in the
background.
