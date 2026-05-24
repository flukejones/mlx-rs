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
- Date: 2026-05-24

### gemma-4-e2b-it-8bit

| Sub-cell               |    N | Previous |  Current |       Δ |
|---|---:|---:|---:|---:|
| `prefill_short`        |   13 |   538.05 |   521.12 |   −3.1% |
| `prefill_long`         | 1024 |  2475.30 |  2498.00 |   +0.9% |
| `prefill_xlong`        | 2048 |  3215.20 |  3261.90 |   +1.5% |
| `decode_short`         |   99 |    97.62 |   100.27 |   +2.7% |
| `decode_long`          |   99 |    94.06 |    95.44 |   +1.5% |
| `decode_short_sampled` |   99 |    98.35 |   100.28 |   +2.0% |

### gemma-4-e4b-it-8bit

| Sub-cell               |    N | Previous |  Current |       Δ |
|---|---:|---:|---:|---:|
| `prefill_short`        |   13 |   284.07 |   291.07 |   +2.5% |
| `prefill_long`         | 1024 |  1125.10 |  1142.50 |   +1.5% |
| `prefill_xlong`        | 2048 |  1284.00 |  1315.60 |   +2.5% |
| `decode_short`         |   99 |    60.67 |    62.73 |   +3.4% |
| `decode_long`          |   99 |    56.68 |    58.03 |   +2.4% |
| `decode_short_sampled` |   99 |    60.72 |    61.61 |   +1.5% |

### gemma-4-26b-a4b-it-8bit

| Sub-cell               |    N | Previous |  Current |       Δ |
|---|---:|---:|---:|---:|
| `prefill_short`        |   13 |   228.88 |   220.14 |   −3.8% |
| `prefill_long`         | 1024 |   967.42 |   988.38 |   +2.2% |
| `prefill_xlong`        | 2048 |  1070.30 |  1089.50 |   +1.8% |
| `decode_short`         |   99 |    78.89 |    80.66 |   +2.2% |
| `decode_long`          |   99 |    72.82 |    76.98 |   +5.7% |
| `decode_short_sampled` |   99 |    75.25 |    79.23 |   +5.3% |

### gemma-4-31b-it-4bit

| Sub-cell               |    N | Previous |  Current |       Δ |
|---|---:|---:|---:|---:|
| `prefill_short`        |   13 |    61.69 |    65.71 |   +6.5% |
| `prefill_long`         | 1024 |   148.13 |   161.02 |   +8.7% |
| `prefill_xlong`        | 2048 |   161.77 |   166.84 |   +3.1% |
| `decode_short`         |   99 |    24.38 |    24.51 |   +0.5% |
| `decode_long`          |   99 |    22.21 |    22.39 |   +0.8% |
| `decode_short_sampled` |   99 |    24.12 |    24.34 |   +0.9% |

### Qwen3.5-4B-8bit

| Sub-cell               |    N | Previous |  Current |       Δ |
|---|---:|---:|---:|---:|
| `prefill_short`        |   13 |   414.97 |   407.87 |   −1.7% |
| `prefill_long`         | 1024 |  1180.80 |  1177.00 |   −0.3% |
| `prefill_xlong`        | 2048 |  1154.20 |  1149.40 |   −0.4% |
| `decode_short`         |   99 |    87.48 |    87.69 |   +0.2% |
| `decode_long`          |   99 |    86.42 |    85.93 |   −0.6% |
| `decode_short_sampled` |   99 |    85.77 |    85.87 |   +0.1% |

### Qwen3.5-9B-8bit

| Sub-cell               |    N | Previous |  Current |       Δ |
|---|---:|---:|---:|---:|
| `prefill_short`        |   13 |   230.68 |   230.18 |   −0.2% |
| `prefill_long`         | 1024 |   638.27 |   648.26 |   +1.6% |
| `prefill_xlong`        | 2048 |   641.75 |   640.71 |   −0.2% |
| `decode_short`         |   99 |    51.15 |    50.99 |   −0.3% |
| `decode_long`          |   99 |    50.07 |    49.84 |   −0.5% |
| `decode_short_sampled` |   99 |    50.30 |    49.95 |   −0.7% |

### Qwen3.6-27B-4bit

Decode cells rerun cold after the first pass showed −3 to
−4% drift attributable to thermal load. Reported `Current`
is the rerun median.

| Sub-cell               |    N | Previous |  Current |       Δ |
|---|---:|---:|---:|---:|
| `prefill_short`        |   13 |    73.08 |    77.04 |   +5.4% |
| `prefill_long`         | 1024 |   195.79 |   192.66 |   −1.6% |
| `prefill_xlong`        | 2048 |   194.46 |   200.09 |   +2.9% |
| `decode_short`         |   99 |    27.58 |    27.76 |   +0.7% |
| `decode_long`          |   99 |    26.62 |    27.29 |   +2.5% |
| `decode_short_sampled` |   99 |    27.57 |    27.86 |   +1.1% |

### Qwen3.6-35B-A3B-q8-mtp

Decode cells rerun cold after the first pass showed −3 to
−4% drift attributable to thermal load on this 35B MoE.
Reported `Current` is the rerun median; the prior pass at
174.84 / 1259.50 / 1255.00 prefill and 88.62 / 84.49 /
86.66 decode is the `Previous` baseline.

| Sub-cell                 |    N | Previous |  Current |       Δ |
|---|---:|---:|---:|---:|
| `prefill_short`          |   13 |   174.84 |   187.02 |   +7.0% |
| `prefill_long`           | 1024 |  1259.50 |  1263.60 |   +0.3% |
| `prefill_xlong`          | 2048 |  1255.00 |  1275.60 |   +1.6% |
| `decode_short`           |   99 |    88.62 |    89.39 |   +0.9% |
| `decode_long`            |   99 |    84.49 |    86.51 |   +2.4% |
| `decode_short_sampled`   |   99 |    86.66 |    88.53 |   +2.2% |
| `decode_short_mtp`       |   99 |    97.01 |    98.17 |   +1.2% |
| `decode_long_mtp`        |   99 |    92.95 |    93.81 |   +0.9% |
| `decode_short_mtp_depth2`|   99 |   110.83 |   113.91 |   +2.8% |
| `decode_long_mtp_depth2` |   99 |    95.26 |    95.96 |   +0.7% |
| `decode_short_mtp_depth3`|   99 |   109.08 |   108.45 |   −0.6% |
| `decode_long_mtp_depth3` |   99 |    87.01 |    87.96 |   +1.1% |
