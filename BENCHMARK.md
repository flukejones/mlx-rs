# mlxr benchmarks

Decode throughput for `mlxr-lm`. One section per system, each holding
system details + latest measurements.

A perf-affecting commit moves the old **Current** into **Previous**
and writes the new measurement into **Current**. Commits with no
measurable change leave the table alone.

## Methodology

- Harness: `crates/mlxr-lm/benches/lm_decode.rs`.
- Framework: criterion, 10 samples × 20 s per cell.
- Decode-only: criterion `iter_custom` runs prefill once unmeasured,
  then loops `N` decode steps inside the timed band.
- Single-threaded (`MLX_LM_BENCH_NO_DOWNLOAD=1` cell loop).
- Throughput in tok/s = elements/sec; higher is better.

## Repro

```sh
MLX_LM_BENCH_NO_DOWNLOAD=1 \
  cargo bench -p mlxr-lm --bench lm_decode -- <group>
```

Bench cache root resolution (first hit wins):
`$MLX_LM_BENCH_CACHE` → `$XDG_CACHE_HOME/mlx-rs-bench/` →
`$HOME/.cache/mlx-rs-bench/`. Set `MLX_LM_BENCH_NO_DOWNLOAD=1` to
skip cells whose checkpoint isn't cached.

## How to update

Only perf-affecting commits touch this file. For each cell the
commit changes:

1. Re-run until the median is stable (rerun if criterion `change:`
   flips sign or the CI is wide).
2. Move the old **Current** to **Previous**; write the new value to
   **Current**; record Δ as `(new − old) / old`.
3. > 2% regression needs justification in the commit body or
   reversion. The bench commit lands first in any perf series.

New system → add a new top-level section; do not overwrite existing
ones.

---

## Apple M4 Max, 64 GB, macOS 26.5

- Chip: M4 Max — 12P + 4E CPU, 40 GPU
- Memory: 64 GB unified, ~410 GB/s
- Toolchain: rustc 1.95.0 (2026-04-14)
- Date: 2026-05-23

### gemma-4-e2b-it-8bit

| Sub-cell               |    N | Previous |  Current |  Δ |
|---|---:|---:|---:|---:|
| `prefill_short`        |   13 |        — |   538.05 |  — |
| `prefill_long`         | 1024 |        — |  2475.30 |  — |
| `prefill_xlong`        | 2048 |        — |  3215.20 |  — |
| `decode_short`         |   99 |        — |    97.62 |  — |
| `decode_long`          |   99 |        — |    94.06 |  — |
| `decode_short_sampled` |   99 |        — |    98.35 |  — |

### gemma-4-e4b-it-8bit

| Sub-cell               |    N | Previous |  Current |  Δ |
|---|---:|---:|---:|---:|
| `prefill_short`        |   13 |        — |   284.07 |  — |
| `prefill_long`         | 1024 |        — |  1125.10 |  — |
| `prefill_xlong`        | 2048 |        — |  1284.00 |  — |
| `decode_short`         |   99 |        — |    60.67 |  — |
| `decode_long`          |   99 |        — |    56.68 |  — |
| `decode_short_sampled` |   99 |        — |    60.72 |  — |

### gemma-4-26b-a4b-it-8bit

| Sub-cell               |    N | Previous |  Current |  Δ |
|---|---:|---:|---:|---:|
| `prefill_short`        |   13 |        — |   228.88 |  — |
| `prefill_long`         | 1024 |        — |   967.42 |  — |
| `prefill_xlong`        | 2048 |        — |  1070.30 |  — |
| `decode_short`         |   99 |        — |    78.89 |  — |
| `decode_long`          |   99 |        — |    72.82 |  — |
| `decode_short_sampled` |   99 |        — |    75.25 |  — |

### gemma-4-31b-it-4bit

| Sub-cell               |    N | Previous |  Current |  Δ |
|---|---:|---:|---:|---:|
| `prefill_short`        |   13 |        — |    61.69 |  — |
| `prefill_long`         | 1024 |        — |   148.13 |  — |
| `prefill_xlong`        | 2048 |        — |   161.77 |  — |
| `decode_short`         |   99 |        — |    24.38 |  — |
| `decode_long`          |   99 |        — |    22.21 |  — |
| `decode_short_sampled` |   99 |        — |    24.12 |  — |

### Qwen3.5-4B-8bit

| Sub-cell               |    N | Previous |  Current |  Δ |
|---|---:|---:|---:|---:|
| `prefill_short`        |   13 |        — |   414.97 |  — |
| `prefill_long`         | 1024 |        — |  1180.80 |  — |
| `prefill_xlong`        | 2048 |        — |  1154.20 |  — |
| `decode_short`         |   99 |        — |    87.48 |  — |
| `decode_long`          |   99 |        — |    86.42 |  — |
| `decode_short_sampled` |   99 |        — |    85.77 |  — |

### Qwen3.5-9B-8bit

| Sub-cell               |    N | Previous |  Current |  Δ |
|---|---:|---:|---:|---:|
| `prefill_short`        |   13 |        — |   230.68 |  — |
| `prefill_long`         | 1024 |        — |   638.27 |  — |
| `prefill_xlong`        | 2048 |        — |   641.75 |  — |
| `decode_short`         |   99 |        — |    51.15 |  — |
| `decode_long`          |   99 |        — |    50.07 |  — |
| `decode_short_sampled` |   99 |        — |    50.30 |  — |

### Qwen3.6-27B-4bit

| Sub-cell               |    N | Previous |  Current |  Δ |
|---|---:|---:|---:|---:|
| `prefill_short`        |   13 |        — |    73.08 |  — |
| `prefill_long`         | 1024 |        — |   195.79 |  — |
| `prefill_xlong`        | 2048 |        — |   194.46 |  — |
| `decode_short`         |   99 |        — |    27.58 |  — |
| `decode_long`          |   99 |        — |    26.62 |  — |
| `decode_short_sampled` |   99 |        — |    27.57 |  — |

### Qwen3.6-35B-A3B-q8-mtp

| Sub-cell               |    N | Previous |  Current |  Δ |
|---|---:|---:|---:|---:|
| `prefill_short`        |   13 |        — |   174.84 |  — |
| `prefill_long`         | 1024 |        — |  1259.50 |  — |
| `prefill_xlong`        | 2048 |        — |  1255.00 |  — |
| `decode_short`         |   99 |        — |    88.62 |  — |
| `decode_long`          |   99 |        — |    84.49 |  — |
| `decode_short_sampled` |   99 |        — |    86.66 |  — |
| `decode_short_mtp`     |   99 |        — |    99.92 |  — |
| `decode_long_mtp`      |   99 |        — |    74.55 |  — |
