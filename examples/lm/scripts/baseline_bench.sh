#!/usr/bin/env bash
# Serial bench across HQ-quant models with 60s cooldown.
# Each invocation runs ONE model's decode cells. Logs to $OUT_DIR.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/Users/lukejones/Projects/mlx-rs}"
COOLDOWN_S="${COOLDOWN_S:-60}"
OUT_DIR="${OUT_DIR:-/tmp/baseline_bench}"
mkdir -p "$OUT_DIR"

# (label  cell_filter)
MODELS=(
  "llama_small_bf16     llama_decode_small_bf16/decode_"
  "qwen3_large_bf16     qwen3_decode_large_bf16/decode_"
  "qwen3_5_4b_q8        qwen3_5_decode_4b_q8/"
  "gemma4_26b_a4b_q8    gemma4_decode_26b_a4b_it_q8/decode_"
)

cd "$REPO_ROOT"
cargo bench -p mlx-lm --bench lm_decode --no-run

for entry in "${MODELS[@]}"; do
  read -r label filter <<<"$entry"
  echo
  echo "[baseline_bench] $label"
  MLX_LM_BENCH_NO_DOWNLOAD=1 MLX_LM_BENCH_SET=trimmed \
    MLX_LM_BENCH_ONLY="${filter%%/*}" \
    cargo bench -p mlx-lm --bench lm_decode -- "$filter" \
    > "$OUT_DIR/${label}.log" 2>&1 || echo "[baseline_bench] $label exited non-zero"
  echo "[baseline_bench] $label done; cooling ${COOLDOWN_S}s"
  sleep "$COOLDOWN_S"
done

echo
echo "[baseline_bench] all done — logs at $OUT_DIR"
for entry in "${MODELS[@]}"; do
  read -r label _ <<<"$entry"
  echo
  echo "=== $label ==="
  grep -E "^\s+time:" "$OUT_DIR/${label}.log" || echo "(no time: lines)"
done
