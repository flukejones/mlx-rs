#!/usr/bin/env bash
# Cache-cap sweep across HQ-quant models with 30s cooldown between runs.
# Each invocation: 1 model × 1 cap, runs all 4 bench cells of that model.
#
# Output layout under $OUT_DIR (default /tmp/cap_sweep):
#   <model>_<cap>.svg          — temp/power/memory chart
#   <model>_<cap>.csv          — readings + bench events + mlx_mem stamps
#   <model>_<cap>.bench.log    — full stderr/stdout
#   summary.tsv                — model	cap	cell	time_s	cache_mb
#
# Requires:
#   - examples/lm/target/release/bench_with_temp (pre-built)
#   - mlx-rs root at $REPO_ROOT (auto-detected)
#   - macmon (no sudo) on PATH

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/Users/lukejones/Projects/mlx-rs}"
OUT_DIR="${OUT_DIR:-/tmp/cap_sweep}"
COOLDOWN_S="${COOLDOWN_S:-30}"
BENCH_BIN="$REPO_ROOT/examples/lm/target/release/bench_with_temp"

mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.tsv"
echo -e "model\tcap\tcell\tmedian_s\tcache_mb_end" > "$SUMMARY"

# (model_label  bench_only_substring  cell_filter)
# Restrict to HQ quants (bf16 / q8). Skip q4 + 31B to keep runtime
# manageable (~30 min at 3 caps × 4 models × decode-only).
# `cell_filter` is the criterion regex; differs by family because
# qwen3.5 uses short/long_prompt and the others use decode_short/long.
MODELS=(
  "llama_small_bf16        llama_decode_small_bf16        llama_decode_small_bf16/decode_"
  "qwen3_large_bf16        qwen3_decode_large_bf16        qwen3_decode_large_bf16/decode_"
  "qwen3_5_4b_q8           qwen3_5_decode_4b_q8           qwen3_5_decode_4b_q8/"
  "gemma4_26b_a4b_q8       gemma4_decode_26b_a4b_it_q8    gemma4_decode_26b_a4b_it_q8/decode_"
)

# (cap_label  bytes)
CAPS=(
  "cache0      0"
  "cache20mb   $((20 * 1024 * 1024))"
  "default     -1"
)

cooldown() {
  local s="$1"
  echo "[driver] cooldown ${s}s…"
  sleep "$s"
}

run_one() {
  local model_label="$1"
  local bench_only="$2"
  local cell_filter="$3"
  local cap_label="$4"
  local cap_bytes="$5"
  local out_prefix="$OUT_DIR/${model_label}_${cap_label}"

  echo
  echo "==========================================================="
  echo "[driver] $model_label / $cap_label (bytes=$cap_bytes)"
  echo "==========================================================="

  if [[ "$cap_bytes" == "-1" ]]; then
    unset MLX_LM_CACHE_LIMIT_BYTES
  else
    export MLX_LM_CACHE_LIMIT_BYTES="$cap_bytes"
  fi

  export MLX_LM_BENCH_ONLY="$bench_only"
  "$BENCH_BIN" \
    --bench-args "-- $cell_filter" \
    --interval-ms 500 \
    --out "$out_prefix"
  unset MLX_LM_BENCH_ONLY

  # Append summary rows: for every "time: [.. <median> ..]" line, find
  # the previous "Benchmarking <id>" header and the next "/end" stamp.
  python3 - "$out_prefix.bench.log" "$model_label" "$cap_label" "$SUMMARY" <<'PY'
import re, sys
log_path, model, cap, summary = sys.argv[1:]
hdr_re = re.compile(r"Benchmarking ([^:\s]+)$")
time_re = re.compile(r"time:\s+\[\S+\s+\S+\s+(\S+)\s+(\S+)\s+\S+\s+\S+\]")
end_re = re.compile(r"\[mlx_mem\]\s+(\S+)\s+active_mb=\S+\s+cache_mb=(\S+)")
last_hdr = None
end_cache = {}
rows = []
with open(log_path) as f:
    for line in f:
        m = hdr_re.search(line)
        if m:
            last_hdr = m.group(1)
            continue
        m = time_re.search(line)
        if m and last_hdr:
            median = m.group(1)
            unit = m.group(2)
            scale = {"ns": 1e-9, "µs": 1e-6, "us": 1e-6, "ms": 1e-3, "s": 1.0}.get(unit, 1.0)
            rows.append((last_hdr, float(median) * scale))
            last_hdr = None
            continue
        m = end_re.search(line)
        if m and "/end" in m.group(1):
            end_cache[m.group(1).replace("/end", "")] = m.group(2)
with open(summary, "a") as f:
    for hdr, med in rows:
        cache_mb = end_cache.get(hdr, end_cache.get(hdr.split('/')[0] + '/' + hdr.split('/')[1], ""))
        f.write(f"{model}\t{cap}\t{hdr}\t{med:.3f}\t{cache_mb}\n")
PY
}

# Build first so the bench compile cost is paid once.
( cd "$REPO_ROOT" && cargo bench -p mlx-lm --bench lm_decode --no-run )

for model_entry in "${MODELS[@]}"; do
  read -r model_label bench_only cell_filter <<<"$model_entry"
  for cap_entry in "${CAPS[@]}"; do
    read -r cap_label cap_bytes <<<"$cap_entry"
    run_one "$model_label" "$bench_only" "$cell_filter" "$cap_label" "$cap_bytes"
    cooldown "$COOLDOWN_S"
  done
done

echo
echo "[driver] all done — summary at $SUMMARY"
column -t -s $'\t' "$SUMMARY"
