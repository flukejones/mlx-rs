# mlx-rs recipes — `just <recipe>` (run `just` with no args to list).
# Inspired by retypst (terse, per-recipe) and JTE evolve-hmi-rust (hook
# install + strict-lint gates).

default:
	@just --list

# --- one-time setup ---

# Point git at .githooks/ for pre-commit + pre-push. Run once after clone.
# Ensures the hook scripts are executable then sets `core.hooksPath`.
install-hooks:
	chmod +x .githooks/commit-msg .githooks/pre-commit .githooks/pre-push
	git config core.hooksPath .githooks
	@echo "git core.hooksPath -> .githooks"
	@echo "  pre-commit: rustfmt staged-only (auto-folded into commit) + cargo clippy --workspace -- -D warnings"
	@echo "  pre-push:   cargo clippy --workspace -- -D warnings"

# --- check / fmt / lint ---

# Quick compile-check for whole workspace.
check:
	cargo check --workspace --message-format=short

# Compile-check one crate. Usage: just check-crate mlx-lm
check-crate crate:
	cargo check -p {{crate}} --message-format=short

# rustfmt across the workspace (disabled in pre-commit while refactor churn high).
fmt:
	cargo fmt --all

# rustfmt --check only (no writes). CI gate.
check-fmt:
	cargo fmt --all -- --check

# Workspace clippy with -D warnings. Matches the pre-commit / pre-push
# hook semantics so `just lint` reproduces the gate locally.
lint:
	cargo clippy --workspace --all-targets -- -D warnings

# Same as `lint` but scoped to one crate. Faster for iteration.
lint-crate crate:
	cargo clippy -p {{crate}} --all-targets -- -D warnings

# --- tests ---

# Workspace unit tests, single-threaded (MLX/Metal kernels require it).
test:
	cargo test --workspace --lib -- --test-threads=1

# Tests for one crate, single-threaded.
test-crate crate:
	cargo test -p {{crate}} --lib -- --test-threads=1

# Workspace integration tests (mlx-tests + parity tests).
test-integration:
	cargo test --workspace --tests -- --test-threads=1

# --- benches ---

# Run mlx-lm decode bench. Pre-req: kill background compiles, ≥60s cooldown.
bench:
	cargo bench -p mlx-lm --bench lm_decode

# Run one criterion group, skipping load of every other model. Pre-req:
# kill background compiles. Usage:
#   just bench-one gemma4_decode_26b_a4b_it_q8
#   just bench-one gemma4_decode_26b_a4b_it_q8 decode_short
#   just bench-one qwen3_5_decode_27b_q8 decode_long full
# `cell` (optional) narrows to one BenchmarkId under the group
# (`prefill_short`, `prefill_long`, `decode_short`, `decode_long`).
# `set=full` opts into the MLX_LM_BENCH_SET=full gated cells.
# MLX_LM_BENCH_ONLY is set to <group> so every other `bench_one()`
# call short-circuits before its model load — the bench only ever
# loads the checkpoint you asked for.
bench-one group cell="" set="trimmed":
	MLX_LM_BENCH_NO_DOWNLOAD=1 \
	MLX_LM_BENCH_ONLY={{group}} \
	MLX_LM_BENCH_SET={{set}} \
	  cargo bench -p mlx-lm --bench lm_decode -- \
	  '{{ if cell == "" { group } else { group + "/" + cell } }}'
	@just bench-results {{group}}

# Print median tok/s for every cell under a criterion group.
# Usage: just bench-results gemma4_decode_26b_a4b_it_q8
bench-results group:
	@for f in target/criterion/{{group}}/*/*/new/estimates.json; do \
	  [ -f "$f" ] || { echo "no estimates under target/criterion/{{group}}"; exit 1; }; \
	  ns=$(jq -r .mean.point_estimate "$f"); \
	  n=$(echo "$f" | awk -F/ '{print $(NF-2)}'); \
	  cell=$(echo "$f" | awk -F/ '{print $(NF-3)}'); \
	  awk -v c="$cell" -v p="$n" -v ns="$ns" 'BEGIN{printf "%-16s %5s\t%7.2f tok/s\n", c, p, p*1e9/ns}'; \
	done

# Profile one bench cell under Apple Instruments. Usage:
#   just profile-bench gemma4_decode_26b_a4b_it_q8 decode_short
#   just profile-bench gemma4_decode_26b_a4b_it_q8 decode_short cpu
#   just profile-bench gemma4_decode_26b_a4b_it_q8 decode_short gpu 10s
# `template` (3rd arg) picks the xctrace template:
#   `gpu`  -> Metal System Trace  (default; GPU command queue +
#             kernel timeline + shader timing)
#   `cpu`  -> Time Profiler       (host samples per thread)
#   any other string is passed through to xctrace as-is (see
#   `xcrun xctrace list templates`).
# `limit` (4th arg) caps the recording duration. xctrace stops
# recording at the limit even if the bench is still running —
# essential for the GPU template, which buffers per-kernel events
# at hundreds of thousands per second and finalises slowly on long
# captures. Decode is steady-state; 5s is enough to see per-step
# kernel cost. Pass `none` to record until the bench exits.
#
# Output: /tmp/<group>_<cell>_<template>.trace, opened in Instruments.app.
# The bench binary is built once via `cargo bench --no-run` (release +
# line-table debuginfo from [profile.bench] in the workspace Cargo.toml)
# and its path is extracted from cargo's json output — no `cargo bench`
# wrapper, so xctrace attaches to the unwrapped binary and the trace
# only captures the bench process.
profile-bench group cell template="gpu" limit="5s":
	@template_name="{{ if template == 'gpu' { 'Metal System Trace' } else if template == 'cpu' { 'Time Profiler' } else { template } }}"; \
	echo "+ building bench binary (release + line-tables-only debuginfo)…"; \
	bin=$(cargo bench -p mlx-lm --bench lm_decode --no-run --message-format=json 2>/dev/null \
	      | jq -r 'select(.target.name == "lm_decode" and .executable != null) | .executable' \
	      | tail -1); \
	[ -n "$bin" ] || { echo "could not locate bench binary"; exit 1; }; \
	out="/tmp/{{group}}_{{cell}}_{{template}}.trace"; \
	rm -rf "$out"; \
	limit_args=""; \
	if [ "{{limit}}" != "none" ]; then limit_args="--time-limit {{limit}}"; fi; \
	echo "+ xctrace template: $template_name"; \
	echo "+ bench filter:     {{group}}/{{cell}}"; \
	echo "+ time limit:       {{limit}}"; \
	echo "+ output:           $out"; \
	MLX_LM_BENCH_NO_DOWNLOAD=1 MLX_LM_BENCH_ONLY={{group}} \
	  xcrun xctrace record --template "$template_name" --output "$out" \
	    $limit_args \
	    --launch -- "$bin" --bench '{{group}}/{{cell}}'; \
	open "$out"

# Run mlx-rs compile-overhead bench.
bench-compile:
	cargo bench -p mlx-rs --bench compile_overhead

# --- run / examples ---

# Run a generate example. Usage: just generate <model> <prompt...>
generate model *args:
	cargo run --release -p lm --bin generate -- --model {{model}} {{args}}

# Run the chat REPL example.
chat *args:
	cargo run --release -p lm --bin chat -- {{args}}

# --- maintenance ---

# Wipe build artefacts. Use when diagnosing perf regressions or after mlx-sys / mlx-c bumps.
clean:
	cargo clean

# Update Cargo.lock for the entire workspace.
update:
	cargo update
