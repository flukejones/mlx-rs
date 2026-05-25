# Code Review Guidelines — mlxr

What to look for when reviewing a PR or commit in this workspace. Each item is a check, not an assertion.

Items already enforced by clippy / rustc lints or `cargo-deny` are not repeated here. Trust CI for those; review covers what tools cannot.

Related docs in this repo:

- [`CLAUDE.md`](CLAUDE.md) — project conventions (git, build, MLX/Metal rules, code style).
- [`BENCHMARK.md`](BENCHMARK.md) — the perf reference for all hot-path changes.

## Workspace and crate boundaries

- **Verify new code lands in the right crate.** `mlxr` is the tensor library + autograd; `mlxr-lm` is the LM runtime + model families; `mlxr-sys` is the FFI shim. Anything that imports `mlxr_sys::` from outside `mlxr` is a layering violation.
- **Look for new optional sub-crates.** The published surface is 5 crates (`mlxr-sys`, `mlxr-macros`, `mlxr-codegen`, `mlxr`, `mlxr-lm`); adding a sixth carries a SemVer + changelog + `cargo publish` tax forever. A new `mlxr-attention` or `mlxr-kv-cache` crate needs a real second consumer to justify it — internal modules + feature gates are the default.
- **Confirm `publish = false`** on every new crate that isn't user-facing (test crates, converters, examples, xtask).
- **Check sibling deps use `{ workspace = true }`.** Never `path = "../foo"` inside a member crate.
- **Verify mlx-c version pin is bumped in one place only** — `[workspace.metadata.mlx]` in root `Cargo.toml`. README claims and submodule SHA must match.

## Feature gates

- **Verify the feature surface compiles in isolation.** A change that adds a `#[cfg(feature = "qwen3_5")]` mod needs verification via `cargo check -p mlxr-lm --no-default-features --features qwen3_5`. Same for gemma4-only, `qwen3_5 + image`, and the bare-minimum `mlxr --no-default-features`.
- **Check feature cascades make sense.** `layers = ["fast"]` because RmsNorm/LayerNorm/Rope use fused kernels. A new gated module that calls into `crate::fast::*` must declare the cascade in `[features]`, not push the burden onto consumers.
- **Flag features that gate at the symbol level.** Per-module `#[cfg(feature = "X")] pub mod foo;` is the project rule; scattering `#[cfg]` inside fn bodies of an always-compiled module is wrong. One file per gate.
- **Flag `#[cfg]` on imports without the matching `#[cfg]` on the consumers.** Easy way to produce "unused import" warnings only in the no-default-features build.

## Module structure

- **Look for type-only files** — any `.rs` with no `impl` blocks and no `fn` bodies. Ask whether it earns its boundary or should fold into the consumer.
- **Look for single-file folders** — `foo/mod.rs` with no siblings. Should be flat `foo.rs`.
- **Check for mutual sibling imports** — if `a.rs` and `b.rs` import each other, the boundary is wrong.
- **Watch for 1000-LOC files mixing 3+ concerns** — model code + weight loading + adapter glue in one file is the same anti-pattern as fragmentation. The per-family layout (`text/`, `image/`, `adapter_dense.rs`, `adapter_moe.rs`, `weights.rs`) exists for a reason.
- **Verify `mod.rs` stays small** — re-exports + the family-level `handles()` / `load_context()` dispatch. If `mod.rs` is 700 LOC, the structure is wrong.
- **Look for two-level dispatch** — folder split by family over a `model_type` string, where each `<family>/mod.rs` does another match on the same string. One dispatch should suffice; the second match means the boundary is in the wrong place.
- **Flag new model families that don't follow the `<family>/{text,image,audio,video}/` shape.** Modality consistency across families is what lets the dispatcher stay flat.

## Naming

- **Check for naming collisions across the two `nn` namespaces.** `mlxr::layers` (was `mlxr::nn`) and `mlxr-lm`'s crate-local `nn/` (runtime helpers) coexist. A bare `use nn::*` is ambiguous-looking; require an explicit `use mlxr::layers::*` or `use crate::nn::*`.
- **Flag function names with type/variant suffixes** — `decode_f32`, `forward_v2`, `mlp_fast`. Rename the call site or split modules. (CLAUDE.md rule.)
- **Flag the `steel_` prefix** on any new symbol. `attention/` was renamed deliberately because "steel" is an upstream MLX kernel-pack name with no standalone meaning. `with_steel_prefill` is grandfathered (matches the upstream method name).
- **Verify variable names describe role, not type** — an `Array`-typed key matrix is `k`, not `arr_k` or `key_array`.
- **Audit `model_type` strings.** Each family owns its list (`MODEL_TYPES_DENSE`, `MODEL_TYPES_MOE`, etc.). Adding a new variant means appending to the list AND verifying the dispatcher routes it. A typo silently falls through to the "unsupported model_type" error.

## Typed configs vs stringly-typed

Closed-form value sets should be enums; unknown inputs should fail at the type-system boundary (deserialize / `TryFrom`), not at the kernel call site where the error message is `"unknown rope_type 'yarn'"` from inside a 30-layer forward.

- **Flag `&str` / `String` fields that accept a finite set of values.** `MatrixNorm`, `MatrixTriangle`, `QwenRopeType`, `QwenLayerKind`, `QuantMode`, `LayerKind` (gemma4) are the in-tree references. New ones go with `#[derive(Deserialize)] #[serde(rename_all = "snake_case")] enum Foo`.
- **Flag `parse(s: &str) -> Self` with a silent `_` fallback** (e.g. an old `LayerKind::parse` that defaults unknown strings to one variant). Use `TryFrom<&str>` returning `Result<_, ErrorKind>`; let serde reject at config-load time when the value is a config field.
- **Flag `match str { … _ => default }`** in loader-adjacent code. The same pattern, harder to spot.
- **Flag `Option<bool>` for binary toggles with a known default.** `cholesky(upper: Option<bool>)` is the same anti-pattern as `Option<&str>`: three states with two distinct meanings. Use the enum (`MatrixTriangle`) or a plain `bool` with serde default.
- **Flag mutually-exclusive `Option<f32>` + `Option<f32>` fields** that one branch silently ignores at runtime — `temperature == 0.0 ` ignoring `top_p`. Collapse into an enum (`Sampler::{Greedy, Temperature(f32), TopP { temperature, p }}`) so the type system enforces "set together or not at all".
- **Flag `meta_state: HashMap<String, String>` round-tripping typed scalars.** The KV-cache wire format does this today (parse-string-back-to-i32 on load); a future change should land typed serde structs + a version field. Flag any new consumer that adds keys to the map.

## Public API

- **Audit every new `pub use`.** Confirm an external caller actually reaches it. Unused re-exports inflate the SemVer surface.
- **Prefer `pub(crate)` over `pub`** for items used only inside the crate. `family::LoadedContext` is `pub(crate)` because no external consumer should hand-construct it.
- **Renaming a published item is a breaking change.** The `mlxr`/`mlxr-lm` crates carry the SemVer surface; bump appropriately and update the changelog. Internal renames (inside a crate) are free.
- **Flag new `pub` fields on Cargo-exposed structs.** `GenerateParams`, `SamplingParams`, `ModelContext` — adding a field breaks struct-update syntax for consumers using `..Default::default()` only if the field is `pub` without `#[non_exhaustive]`. Note that `#[non_exhaustive]` is itself a breaking change for external callers: under E0639, `..Default::default()` tail syntax stops working from outside the defining crate. The remedy is a builder (`GenerateParamsBuilder::new().max_new_tokens(n).build()`), not a bare `#[non_exhaustive]` annotation — adding the attribute without supplying a builder breaks every external `let p = GenerateParams { x, ..Default::default() }` call site at once.

## Tests

- **Verify tests live in the same file** as the code under test — `#[cfg(test)] mod tests`.
- **When a file moves, check tests followed the code** — tests for moved items should be in the new file, not orphaned in the old one. The reorg commit caught several of these via cargo check; future ones may not.
- **Verify `mlxr-lm` lib tests run with `--test-threads=1`.** mlx-c shared state crashes under parallel execution. A test that passes under `--test-threads=1` but fails in the default run is mlx-c noise, not a real failure.
- **Flag tests that depend on global state** — env vars, cwd, fixed temp paths. Use `env::temp_dir().join(format!("..._{}", process::id()))`.
- **Flag mocked tensors that bypass the real loader.** Model integration tests should round-trip through `mlxr_lm::load`, not hand-construct a `ModelContext`.
- **Check `#[ignore]` gates on tests that need real checkpoints.** Those tests live alongside cheap ones; mark them clearly and document the `hf download` line in the test header.
- **Flag tests asserting on log output** — brittle; test the side effect, not the log line.

## MLX-specific: state, threading, Metal

- **No `thread_local!` for `Compile`-cache state.** mlx-c v0.31 SIGSEGVs when the thread_local destructor races the GPU stream during thread exit. Caches live as fields on the owning struct (e.g. `Mlp::swiglu_cache`, `GatedDeltaNet::compute_g_cache`), held as `Option<Compiled<…>>` and passed `&mut` to the helper. The cache then drops deterministically with the model.
- **Module-level statics: only FFI callback slots.** Comment them as such. Any other persistent state crosses into `Arc<Mutex<T>>` or struct ownership.
- **Cache kernel names; never rebuild per-call.** `mlx` caches compiled kernels by name. Bump the version suffix (`_v10` → `_v11`) on every body change, or the stale binary persists across runs.
- **Verify Metal grid semantics: grid = total threads.** `.grid(N, …)` is `threadgroup_size × num_groups`, not `num_groups`. Wrong = only thread 0 runs.
- **Reject f64 in kernel-adjacent code.** `Array::from_f64` lands on the Metal stream and is rejected. Build sentinel values as f32 then `as_dtype(target)`.
- **Convert bool masks to additive form before adding to scores.** `create_causal_mask` returns bool. Adding bool directly to scores gives silently-broken KL ≈ 7. Use `where(mask, scores, neg_inf_sentinel)`.
- **Flag `simd_sum` shadowed by local var.** `float simd_sum = simd_sum(x)` shadows the intrinsic. Name locals `lane_sum`.
- **TG-mem global-max stash bug.** When reducing across simdgroups, use dedicated TG slots (`gmax_scratch`); don't overwrite `tg_max[0]` (it's simdgroup-0's local max and gets reread).

## FFI and bindgen

- **`mlxr-sys` is the only crate that talks to mlx-c directly.** Anything else importing `mlx_sys::*` types is a layering violation; wrap in `mlxr` first.
- **Verify new bindings come from the bindgen step** (`cargo xtask mlx-c-diff`), not hand-written `extern "C"` blocks.
- **Flag FFI calls outside `unsafe` blocks.** Even when bindgen wraps them, raw mlx-c calls returning status codes must check the result before treating output pointers as valid.
- **Check `Drop` impls on FFI-owned handles.** `mlx_*_free` must be called exactly once. Double-free crashes are silent until tests run under address sanitizer.
- **Verify mlx-c version bump touches `[workspace.metadata.mlx]` + the submodule SHA + README claim simultaneously.** Drift between them is a recurring bug. The two pins are for *separate* upstream projects — `[workspace.metadata.mlx]` pins `ml-explore/mlx` (the C++ kernel library), and the submodule at `crates/mlxr-sys/src/mlx-c` pins `ml-explore/mlx-c` (the C ABI shim built on top of it). Two different SHAs by design; an audit that flags "SHA mismatch between metadata and submodule" is a misread.

## Performance and benchmarks

- **Flag perf-sensitive changes that lack a bench run.** Attention paths, KV-cache, generation loop, kernel work — `cargo bench -p mlxr-lm --bench lm_decode` must be run before commit, with results compared against the **Current** column in `BENCHMARK.md`. A regression > 2% on any cell needs justification or reversion; an accepted improvement updates the table on the same commit (old **Current** becomes **Previous**).
- **Verify benches respect the decode-only methodology** — criterion `iter_custom`, prefill outside the timing band, single-threaded execution, 60 s cooldown between cells. See `BENCHMARK.md` for the full methodology and repro.
- **Flag `O(N²)` allocations or copies inside the decode hot path**, even when the local microbench bypasses them. "Not measured here" ≠ "free". The per-step `processor.decode(&full_id_list)` regression rebuilt the whole decoded string every token — 25× speedup at N=1024 once replaced with a sliding-window decoder. The fix would have been invisible to the bench; every real consumer paid for it.
- **Check that mlx-c version bumps trigger a `cargo clean` + full bench rerun.** Cross-version comparisons against the cached old artefacts lie.
- **No `lmstudio-community/*` bench models** (one documented exception: q8 MoE Qwen3.6 35B-A3B). Bench models are `mlx-community/*` or official MLX repos.
- **Rerun any single-pass regression on a >20B model.** Thermal load on the 35B MoE and 27B q4 cells produces ±5% run-to-run swings on the M4 Max even with the documented 60 s cooldown. A `-3%` first-pass result is not real until a cold rerun reproduces it. Apply the same rule to any commit before claiming a regression — single-run criterion deltas on these cells are noise.
- **The bench harness must drain GPU + heap state between models** in a multi-cell run. `lm_decode.rs` asserts `active_memory == 0 && cache_memory == 0` before each model loads (via `between_models_quiesce`) and double-drains the buffer-reuse pool after `ctx.unload()`. Skipping this lets a prior model's residual state bleed into the next cell's timing — hours-long sweeps that chain 8 models will measure pool-fragmentation drift rather than the change you're testing. If the pre-load assertion panics, the panicking cell is **not** the regression — it's pointing at the previous cell that didn't drain. For gold-standard isolation, run one model per `cargo bench` invocation via `MLX_LM_BENCH_ONLY=<cell>` — each invocation is a fresh process, fresh kernel cache, fresh mlx-c global state.
- **Don't run anything else on the machine while a bench cell is timing.** Any concurrent work — a parallel `cargo build`, an editor save triggering rust-analyzer, an unrelated process — steals P-core cycles from criterion's `iter_custom` window. Kill background compiles before starting (`pkill -f 'cargo|rust-analyzer'`).
- **BENCHMARK.md "Current" is a captured snapshot, not ground truth.** If a comparison shows a 2-5% regression vs Current, the safe check is rerun-baseline + rerun-candidate in the **same session**, not "candidate vs Current" — the captured baseline may have been measured on a colder/quieter machine. Cross-session deltas on Qwen 27B / 35B and Gemma 31B routinely show ±5% with no code change.

## Array clones, ownership, and FFI roundtrips

Every `Array::clone()` is a refcount-bump FFI call across the mlx-c boundary. Each saved clone is one fewer `mlx_clone` round-trip per token per layer; with 60+ layers, costs compound fast. Two documented perf passes have specifically targeted clone removal in the decode hot path (commits `547209d` and the broader `decode hot-path overhaul`).

- **Flag any `.clone()` on `Array` inside a per-token or per-layer loop.** First fix is ownership: if the source isn't re-used after the call, move it. The `mixed_qkv` chain in `gated_delta_block.rs` ships `concatenate_axis(&[mixed_qkv, …])` (move, last use), not `&[mixed_qkv.clone(), …]`.
- **Flag `concatenate_axis(&[x.clone(), x], -1)` / `stack_axis(&[x.clone(), …])` patterns.** The fn signatures take `impl AsRef<Array>` or `&[&Array]`; pass `&x` borrows instead. `qwen3_5::text::rope` and `qwen3_5::image::vision` got this fix; new code must follow the pattern.
- **Flag clones inserted to "satisfy the borrow checker".** First try restructuring the call site: consume the source by move, take a borrow earlier, or rebind. `gemma4/text.rs`'s `residual = h.clone()` per layer per token was unnecessary — `h` wasn't rebound between the assignment and the consumer, so passing `&h` worked. The clone was 5% of decode CPU per the Time Profiler.
- **Flag fn signatures that take `Array` by value when they only read it.** `quantized_scaled_dot_product_attention` was changed from owned to `queries: &Array` precisely so cache `attention()` impls could pass a borrow straight through. New attention helpers must follow.
- **Flag silent eager evaluation.** Some Array methods (`item()`, `as_slice()`, `save_*`) force a sync barrier. Calling these inside the decode loop kills GPU pipelining. The `Array::try_item` duplicate `eval()` fix (`decode hot-path overhaul`) cut two FFI sync barriers per scalar read; new sync-forcing calls must be justified.
- **Verify async_eval scheduling in any new decode loop.** Submit step N+1's forward + sample via `async_eval` BEFORE the host blocks on N's `.item()` for the EOS check. The unified-userinput rewrite lost this pattern and the recovery commit restored it; the pattern is non-obvious and easy to break.
- **Flag step functions that take `i32` token IDs.** The `LanguageModel::step(&Array)` signature exists so the sampler's device tensor passes by reference; reshape to `[1, 1]` inside the adapter via a graph view. `step(i32)` forces a per-step `Array::from_slice(&[id], &[1, 1])` host→device upload + a refcount round-trip.
- **Flag `Vec<u32>` (or any growable buffer) that pushes inside the decode loop without pre-allocation.** The `produced` vec in `generate()` was reallocating geometrically until pre-sized to `max_new_tokens`. Same rule for logit buffers, token-id windows, anything that grows per step.
- **Flag stateless per-forward compute held as a per-layer field.** `MultimodalRope` was a `rope: MultimodalRope` field on `Attention`, instantiated once per layer (64 instances for Qwen 3.6) but called with identical inputs every forward. `cos_sin` ran 64× per token. Hoist any compute that depends only on per-forward inputs (rope tables, masks, dtype-bound scalars, frequency cosines) to the decoder level, compute once, thread `(cos, sin)` / mask borrows through the layer loop. The fix on `qwen3_5/text/{layer,rope,text}.rs` moved the rope to `Qwen35Decoder` + `MtpHead` — pattern is reusable for any per-layer / per-forward compute. Look for any `*::new(cfg)` field that takes only the config (no learnable params) — strong candidate for hoisting.

## Dtype management and silent precision loss

mlx is dtype-strict: ops between mismatched dtypes either error or silently promote. Promotion is the dangerous case — promoting bf16/fp16 to f32 mid-graph poisons every downstream op (gemm, softmax, residuals) for the rest of the forward pass and quietly halves throughput.

- **Flag `Array::from_f32(scalar) * inputs` without an explicit `as_dtype(target)`.** `queries * f32_scale` promotes bf16/fp16 inputs to f32. The `quantized_scaled_dot_product_attention` fix stages the scale into the input dtype first: `Array::from_f32(scale).as_dtype(q_dtype)?`. New scalar multiplies in attention/SDPA paths must follow.
- **Flag `Array::from_f64(_)`** anywhere kernel-adjacent. Metal rejects f64; the array lands on the Metal stream and the kernel dispatch fails. Build sentinel values as f32, then `.as_dtype(target_dtype)`. The bool-mask sentinel fix in `utils/mod.rs::quantized_scaled_dot_product_attention` is the reference.
- **Flag dtype-cast chains inside loops.** Each `.as_dtype()` allocates a new graph node. Caching the dtype-promoted constant outside the loop is the `SamplerState` pattern: `inv_temp`, `top_p` threshold, `neg_inf` sentinel cached as Arrays bound to the logits dtype on the first sample, reused per token.
- **Flag bool masks added directly to scores.** `create_causal_mask` returns bool; `scores + bool_mask` silently broadcasts and gives KL ≈ 7 (silently broken — model still runs, output is garbage). Use `where(mask, scores, neg_inf_sentinel)` or convert bool → (0 / -inf) explicitly.
- **Verify quant scales/biases stay in the dtype the kernel expects.** Per-tensor dtype overrides on quantised checkpoints (Qwen 3.6-MoE) propagate through `MaybeQuantized<T>`; flag any path that strips the override.
- **Flag `f32` accumulators added to bf16/fp16 graphs without an explicit cast back.** mlx will promote silently. The end-of-op `as_dtype(input_dtype)` keeps the rest of the graph in the original precision.
- **Flag deliberate f32 round-trip in a hot path "for precision".** The Qwen rope `rotate()` cast q/k bf16→f32, ran all of `multiply/add/rotate_half` in f32, then cast back to input dtype — twice per layer × 64 layers per token. The intent was numerical safety on the rope rotation; the cost was three full-tensor cast launches per layer per token plus 2× memory bandwidth in the f32 body. mlx-lm upstream multiplies at input dtype; numerics agree to ≤5e-3 max_abs in the steel-prefill parity test. Pattern: if a cast pair brackets a fused multiply-add chain in a per-layer hot path, prove the f32 width is needed against the input-dtype baseline before accepting the cost. Compute frequency tables (cos/sin) in f32 once, but cast the *output* to input dtype before threading into per-layer compute.

## Allocation patterns (fixed-N, hot paths)

When N is known at compile time, draining FFI vectors or iterators through a `Vec<T>` is wasteful — heap alloc + dealloc per call. On hot paths (per-token KV cache writes run `quantize()` ~60× per token across all layers) this is measurable: gemma-4-26b q8 decode went 68.13 → 68.82 tok/s after replacing one `Vec` round-trip in `VectorArray::try_into_array<N>`.

- **Flag `Vec<T>` → `[T; N]` via `try_into` or `collect`** when N is a compile-time constant. Replace with a `MaybeUninit<[T; N]>` slot filled by index, with drop-on-error cleanup that drops already-written slots. `VectorArray::try_into_array<N>` is the in-tree reference; gemma-4-26b q8 decode went 68.13 → 68.82 tok/s after one such rewrite.
- **Flag any new helper that takes `Vec<T>` when callers always pass fixed-size data.** `&[T; N]` or `[T; N]` on the signature pushes the allocation off the hot path.
- **Flag `[None; N]` then fill-and-unwrap.** Same `MaybeUninit` rewrite applies.
- **Don't apply this rule when N varies at runtime** (variable sequence length, variable layer count). `Vec` is the right tool there.
- **Flag per-step `Array::from_slice(&[scalar], &[1])` or `Array::from_iter`** in the decode loop. Each is a host→device upload + a fresh graph node. Stash the scalar on a struct field (`SamplerState`, the layer module) and reuse.
- **Flag iterator chains that materialise intermediate `Vec`s** when the final consumer takes `&[T]` or `impl Iterator`. `.collect::<Vec<_>>().iter().map(…)` should be `.iter().map(…)` (or vice versa, depending on chain shape).

## Error handling

- **`anyhow` is bin-only; `thiserror` is lib-only.** Library crates (`mlxr`, `mlxr-lm`, `mlxr-convert`, `mlxr-sys`, `mlxr-macros`, `mlxr-codegen`) define a typed `Error` enum via `thiserror`. Bins (`examples/*/src/bin/*.rs`, `examples/*/src/main.rs`) consume the lib's `Error` and may use `anyhow::Result` for top-level glue. Flag a lib `Cargo.toml` pulling in `anyhow`, any `anyhow::` in lib source, or a bin declaring its own thiserror enum.
- **Each lib crate has a `pub(crate) type Result<T>` alias.** Defined in its `error.rs` as `pub(crate) type Result<T> = std::result::Result<T, Error>;`. `pub(crate)` deliberately — a `pub` alias forces consumers to disambiguate `mlxr::Result` vs `mlxr_lm::Result` at every call site. External callers spell `Result<_, mlxr_lm::Error>` or use `anyhow`.
- **Inside the crate, write `Result<T>` (the alias).** Bare `std::result::Result<T, MyError>` in a lib fn signature is the std-spelling exception, reserved for the alias defn itself and the rare case where the error genuinely differs from the crate default (`impl TryFrom<&str> for Foo` whose `Error = UnknownFoo`). **Mass-rewrite hazard**: never `use crate::error::Result;` into a file that also contains `Result<X, BuilderError>` 2-arg returns — the import shadows the std prelude, turning every 2-arg `Result` into a 1-arg alias and breaking the builder fns. The mlxr optimizer modules (adafactor, adam, adagrad) all carry mixed `Result<T>` + `Result<X, BuilderError>` shapes; the alias hoist there has to be per-fn, not file-top. When migrating a file, grep for `Result<` first and confirm every site uses the crate `Error` before adding the import.
- **`mlxr::error::Exception` is constructed only inside `mlxr`.** Downstream crates use their own `Error` variants (`Error::config`, `Error::shape`, `Error::out_of_bounds`, `Error::MissingInput`, `Error::config_missing`). mlxr op failures auto-convert via `From<Exception> for Error` on `?`. Grep `mlxr-lm` for `Exception::custom` / `Exception::from` — zero hits in source.
- **`Module` and `Quantizable` associated error types are per-impl free.** mlxr-lm impls set `type Error = crate::error::Error;` and `type QuantizationError = crate::error::Error;` — not `Exception`. The mlxr trait only requires `: std::error::Error`. Copying `type Error = Exception` from mlxr's own impls is a leak.
- **`From<Error> for Exception` is mandatory** in any crate whose helpers return the local `Error` while pre-existing impls remain locked to `type Error = Exception`. Without it, `?` won't lift. Lossy by necessity — non-`Exception` variants collapse to `Exception::custom(self.to_string())`. The reverse (`From<Exception> for Error`) is a trivial `#[from]` arm.
- **`Compile<F, _, _>` cache types force `Result<_, Exception>`.** The compile trait bakes the inner fn's error type into a generic. Fns wired through `mlxr::transforms::compile::Compile` (mlxr-lm: `activations.rs`, `gated_delta::compute_g`, `gated_delta_block::precise_swiglu`) cannot move off Exception without an mlxr-side trait change. Don't flag them as unconverted.
- **`Quantizable` derive unifies `QuantizationError` across all `#[quantizable]` fields.** A struct mixing `MaybeQuantized<Linear>` (Exception, mlxr's choice) and `SplitSwitchFfn<_>` (Error, mlxr-lm) won't compile. Resolution: pick `Error` in the mlxr-lm impl, use `.map_err(Into::into)` in the body to absorb Linear's Exception.
- **Flag silent `.ok()`** that discards a `Result` from a meaningful op. Channel sends, I/O writes, weight-loader probes, EOS-id parsing should `log::warn!` on error.
- **Verify error variants carry context.** `Error::Other(_)` with a bare string loses the call-site detail; prefer typed variants where the recovery path differs. `Exception::custom(format!(...))` messages surface to the user when a model fails to load.
- **Flag `?` in `main`** that swallows error context — prefer explicit handling at the top level.
- **Fallible conversions use `TryFrom`, not `From`.** A `Dtype` parsed from a config string must be `TryFrom`; never silent-default on unknown bytes.

## Constants and magic values

- **Flag magic numbers** — buffer sizes, timeouts, head dims, group sizes, KV bits, sliding-window length. Should be named `const` at module top.
- **Flag `const` declarations inside fn bodies or closures.** `const` belongs at module top.
- **Check tensor-shape literals are named** — `[B, n_kv_heads, n_repeats, L, D]` chains in attention code are easy to typo. Either name the axis count (`const DK_AXIS: usize = ...`) or assert shape at the entry point.

## Type system

- **Flag `.clone()` calls on plain Rust types that exist only to satisfy the borrow checker.** Restructure ownership instead — drop the field, consume by move (`into_parts()`/`into_inner()`), or rework call sites. (For `Array` specifically see the "Array clones, ownership, and FFI roundtrips" section above — those are FFI-expensive and have their own rules.)
- **Flag `Option<&T>` returns** that could be `&Option<T>` (or vice versa) — picking the wrong shape forces awkward call sites.
- **Check trait bounds on generics** — `fn foo<T>(x: T) where T: Clone + Send + 'static` belongs at the impl/fn level, not duplicated everywhere.
- **Flag wrapper struct + parallel data** — change the base type or define a richer local type. (CLAUDE.md rule.)

## Lifetimes and ownership

- **Flag `'static` bounds added without justification** — usually a workaround for an ownership problem, not a real requirement.
- **Check for `Arc<Mutex<T>>` where `&mut T` would suffice** — wrapping in shared ownership for single-threaded code is overkill, and on mlx-c the shared-state cost is real.
- **Verify `Drop` impls don't panic** — panic-in-drop is undefined behaviour during unwinding. On FFI-owned handles it's catastrophic.

## Comments and docs

- **Flag verbose doc comments** — multi-paragraph rambles on trivial fns. One short line max for non-obvious *why*.
- **Flag noise comments** — narrating the diff (`// removed unused field`, `// renamed from steel_attention`), describing the obvious (`// increment counter`), or referencing call sites that will rot (`// used by load_full_model`). Delete on review.
- **Flag decorative section dividers** (`// ──── Section ────`). They lie when sections drift.
- **Verify comments are self-contained** — readable to someone opening the file in 6 months with no PR context. "TODO: ask user" or "see PR #123" belongs in chat/PR, not the file.
- **Check public API docs explain *why*, not *what*** — well-named identifiers already say what. The KV-cache trait docs are a good reference: each variant says when to use it, not what fields it has.
- **Flag stale Python references.** Inline comments quoting Python (`# def quantized_scaled_dot_product_attention(...)`) are tolerated only when they're the spec we're implementing. Drop them once the Rust path is the source of truth.

## Imports and paths

- **Flag inline multi-segment path qualifiers** in fn signatures, type annotations, struct fields, generic args, callsites. `fn foo(b: std::collections::HashMap<K, V>)` should be `use std::collections::HashMap;` at the top + `fn foo(b: HashMap<K, V>)`. Single-segment `crate::Foo` / `super::Foo` is fine. **Enforced by `cargo run -q -p xtask -- check-paths`**; the pre-commit hook (install once via `cargo run -q -p xtask -- install-hooks`) blocks any new violations in the staged file set. Bypass for a single commit with `git commit --no-verify` only after deliberate discussion. The check skips `use` statements, doc/line/block comments, attributes, and vendored `mlx-c/`.
- **Flag `use` statements inside fn bodies.** Imports belong at the file top. Exceptions: `#[cfg(test)] mod tests { use super::*; … }` and `#[cfg(...)]`-gated fns whose imports are also `#[cfg(...)]`-only.
- **Check for `use std::fmt::Result`** — shadows the prelude `Result`. Should be `use std::fmt;` then `fmt::Result`. The `Result<T>` alias defn at `error.rs` is the one place `std::result::Result` is allowed inline (see Error handling).

## Concurrency

- **Verify channel send/recv errors are handled** — silent `.ok()` on a `Sender::send` is a dropped token in the streaming path.
- **Flag blocking calls inside async fns** — `std::fs::read`, `std::thread::sleep`, sync `Mutex::lock` blocks the executor.
- **Flag use of `tokio` or async runtimes inside `mlxr-lm` core.** The generation loop is sync by design; mlx-c isn't async-safe.

## Model loading and weight files

- **`config.json` is parsed exactly once** at [`crate::config::ModelConfig::from_dir`] in `mlxr_lm::load`. Every adapter takes `(&ModelConfig, &Envelope, &Path)` — no `cfg.from_file()` calls inside `load_context_*`, no second `serde_json::from_str` pass anywhere downstream. Flag any new loader that re-reads `config.json` or unwraps a fresh JSON `Value` from the file. The rule applies to every consumer: a helper like `read_eos_ids(dir: &Path)` that re-opens `config.json` from disk is the same violation as a second `from_file` — replace with `read_eos_ids(cfg: &ModelConfig)` or similar reading off the typed struct.
- **Flag fields that exist in multiple struct paths.** Qwen had `eos_token_id` declared on both `Qwen35Envelope::ModelConfig` and `TextConfig` (the latter dead), and was *also* re-parsed via disk re-read by `family::read_eos_ids`. Three sources for one config value, with no merge-conflict signal if they ever disagreed. Generalisation: any value reachable via more than one field/path is a source-of-truth ambiguity; pick the canonical location (typically the outermost envelope) and delete the duplicates.
- **Dispatch through the typed `Family` enum**, never on a raw `model_type` string. The internally-tagged enum (`#[serde(tag = "model_type")]`) reads the discriminant at deserialize; aliases (`qwen3_5_text`, `gemma4textmodel`, …) live on the enum variants. `MODEL_TYPES_*` const arrays + `handles(&str)` predicates are the old shape — flag them on sight.
- **Verify family envelopes (`qwen3_5::text::config::ModelConfig`, `gemma4::text::config::Gemma4Envelope`) don't re-declare `quantization`, `model_type`, or other outer-envelope fields.** Those live on `crate::config::ModelConfig` and are accessed via `cfg.quantization()`.
- **Check `eos_token_id` normalisation handles both `int` and `[int]` forms.** `family::read_eos_ids` is the canonical helper.
- **Flag hand-rolled safetensors-key probing.** The weight-loader walk pattern in `qwen3_5/text/weights.rs` is the reference; copying its skeleton for a new family beats re-inventing.
- **Verify quantised checkpoints carry per-tensor overrides where needed** — Qwen 3.6-MoE has them; missing them silently downgrades expert weights to defaults.
- **Check VLM probe order in `load_context`.** `preprocessor_config.json` existence → VLM path; without the `image` feature, log::warn + fall through to dense. The probe runs once at load, never per turn.

## Duplication

- **Search for similar fns before approving a new one** — grep the obvious names. Re-implementing an existing parser, cache helper, or attention dispatch wrapper is a frequent failure mode.
- **Three near-identical lines is fine. A premature helper is not** — but a fourth duplicate means it's time to extract.
- **Cross-family copy-paste of model code is the default**, not a bug. Qwen3.5 and Gemma 4 share patterns but not types; trying to unify them prematurely produces a "framework" that no one wants.

## Git hygiene (project-specific)

- **Verify every commit leaves the repo green** — `cargo check` + `cargo clippy -p <crate> --all-targets -- -D warnings` + `cargo test -p <crate>`.
- **No fixup commits stacked on top.** Fold fixes into the commit that introduced the bug. (`git commit --amend --no-edit` after the fix.)
- **Commit messages: no forward refs, no session context, no `Co-Authored-By` trailers.** "for Phase B", "as discussed", "addressing review feedback" are all out.
- **Conventional prefix mandatory** — `feat(scope):`, `fix(scope):`, `refactor(scope):`, etc.
- **`cargo fmt` + clippy fixes get folded into the commit that introduced them**, not landed as a separate "fmt fix" commit.
