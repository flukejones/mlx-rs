# mlx-rs recipes — `just <recipe>` (run `just` with no args to list).
# Inspired by retypst (terse, per-recipe) and JTE evolve-hmi-rust (hook
# install + strict-lint gates).

default:
	@just --list

# --- one-time setup ---

# Point git at .githooks/ for pre-commit + pre-push. Run once after clone.
install-hooks:
	git config core.hooksPath .githooks
	@echo "git core.hooksPath -> .githooks"

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

# Workspace clippy, warnings only.
lint:
	cargo clippy --workspace --all-targets -- -W clippy::all

# Clippy with -D warnings, scoped to one crate. Used by pre-commit hook.
# Per-crate scope avoids upstream-main clippy debt (see CLAUDE.md).
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

# Run mlx-rs compile-overhead bench.
bench-compile:
	cargo bench -p mlx-rs --bench compile_overhead

# --- run / examples ---

# Run a generate example. Usage: just generate <model> <prompt...>
generate model *args:
	cargo run --release --manifest-path examples/lm/Cargo.toml --bin generate -- --model {{model}} {{args}}

# Run the chat REPL example.
chat *args:
	cargo run --release --manifest-path examples/lm/Cargo.toml --bin chat -- {{args}}

# --- maintenance ---

# Wipe build artefacts. Use when diagnosing perf regressions or after mlx-sys / mlx-c bumps.
clean:
	cargo clean

# Update Cargo.lock for the entire workspace.
update:
	cargo update
