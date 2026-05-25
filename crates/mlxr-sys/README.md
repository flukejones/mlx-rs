# mlxr-sys

Rust FFI bindings to the [mlx-c](https://github.com/ml-explore/mlx-c)
API, generated with bindgen.

Implementation detail of [`mlxr`](../mlxr/) — consumers should depend
on `mlxr` directly. The high-level Rust API, layer library, and
language-model runtime are built on top of this crate.

## Version pinning

Two upstream projects are pinned. They move together but each lives in
its own place.

- `[workspace.metadata.mlx]` in the root `Cargo.toml` pins
  [`ml-explore/mlx`](https://github.com/ml-explore/mlx) — the C++
  kernel library that mlx-c links against. Read by build scripts in
  `mlxr-sys` (cmake FetchContent) and `mlxr-lm` (steel-attention
  preamble generator).
- The git submodule at `crates/mlxr-sys/src/mlx-c` pins
  [`ml-explore/mlx-c`](https://github.com/ml-explore/mlx-c) — the C ABI
  shim built on top of `mlx`, from which `bindgen` generates the FFI
  bindings.

```toml
[workspace.metadata.mlx]
version = "v0.31.2"
sha = "68cf2fddd8de5edd8ab3d926391772b2e2cedad8" # ml-explore/mlx
```

Bumping the pair → bump the `[workspace.metadata.mlx]` entry and run
`cargo xtask mlx-c-diff [<tag>]` to advance the mlx-c submodule to the
matching tag (defaults to the latest upstream tag) and regenerate the
bindings.

## Features

- `metal` — Apple Metal GPU backend (forwarded from `mlxr/metal`).
- `accelerate` — Apple Accelerate CPU backend.

## License

Dual-licensed under MIT and Apache 2.0.
