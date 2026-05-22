# mlxr-sys

Rust FFI bindings to the [mlx-c](https://github.com/ml-explore/mlx-c)
API, generated with bindgen.

Implementation detail of [`mlxr`](../mlxr/) — consumers should depend
on `mlxr` directly. The high-level Rust API, layer library, and
language-model runtime are built on top of this crate.

## Version pinning

`mlxr-sys` tracks upstream mlx-c. The pinned version + commit live in
the workspace root `Cargo.toml`:

```toml
[workspace.metadata.mlx]
version = "v0.31.2"
sha = "68cf2fddd8de5edd8ab3d926391772b2e2cedad8"
```

Bumping mlx-c → bump that single source of truth and run
`cargo xtask [<tag>]` to advance the git submodule to the matching
tag (defaults to the latest upstream tag) and regenerate the bindings.

## Features

- `metal` — Apple Metal GPU backend (forwarded from `mlxr/metal`).
- `accelerate` — Apple Accelerate CPU backend.

## License

Dual-licensed under MIT and Apache 2.0.
