# mlxr

Rust port of Apple's [MLX](https://github.com/ml-explore/mlx) plus a
language-model runtime (Qwen 3.5/3.6, Gemma 4) built on top.

A hard fork of [oxideai/mlx-rs](https://github.com/oxideai/mlx-rs) /
[oxiglade/mlx-rs](https://github.com/oxiglade/mlx-rs) — diverged ~25k
LOC with model-family adapters, MTP self-speculative decode, a fused
flash-attention prefill kernel (`head_dim ∈ {128, 256}`), and quantised
KV cache machinery upstream doesn't ship.

## Workspace

| Crate                                       | Role                                                                                  |
|---------------------------------------------|---------------------------------------------------------------------------------------|
| [`mlxr-sys`](crates/mlxr-sys/)              | FFI bindings (bindgen against vendored mlx-c)                                         |
| [`mlxr-macros`](crates/mlxr-macros/)        | Consumer-facing derive macros (`ModuleParameters`, `Quantizable`)                     |
| [`mlxr-codegen`](crates/mlxr-codegen/)      | Internal codegen proc-macros (op-fn variants, builders, default-device)               |
| [`mlxr`](crates/mlxr/)                      | Tensor library: array, ops, transforms, layers, fft, linalg, fast Metal kernels      |
| [`mlxr-lm`](crates/mlxr-lm/)                | LM runtime + model families (Qwen 3.5/3.6 dense + MoE + VL, Gemma 4) + KV cache       |
| `mlxr-convert`                              | `bf16 → qN` checkpoint converter (bin, `publish = false`)                             |
| `mlxr-tests`                                | Cross-crate integration tests (`publish = false`)                                     |
| `examples/{chat,lm,mnist}`                  | Runnable demos                                                                        |

## Quick start

```toml
[dependencies]
mlxr = "0.27"
mlxr-lm = "0.27"
```

```rust
use std::path::PathBuf;
use mlxr_lm::{chat_template::ChatMessage, generate, load, GenerateParams, UserInput};

let home = std::env::var("HOME").expect("HOME");
let dir = PathBuf::from(home).join(".cache/mlx-community/Qwen3-1.7B-4bit");
let mut ctx = load(&dir)?;

let input = UserInput::chat(vec![ChatMessage::user("Explain MLX in one sentence.")]);
generate(&mut ctx, input, GenerateParams::default(), &mut |_id, delta| {
    print!("{delta}");
    std::ops::ControlFlow::Continue(())
})?;
```

See [examples/chat/](examples/chat/) for a REPL + OpenAI-compatible
HTTP server, [examples/generate/](examples/generate/) for one-shot
completion, [examples/mnist/](examples/mnist/) for a minimal training
example.

## Features

`mlxr` defaults pull in everything: `accelerate`, `metal`, `layers`,
`fft`, `linalg`, `fast`, `losses`, `optimizers`. Backends gate the
mlx-c codepath; the higher-level libraries each toggle a `mlxr::*`
module.

`mlxr-lm` defaults: `qwen3_5`, `gemma4`, `image`. Each model family
cascades the `mlxr` features it needs (e.g. `qwen3_5` pulls in
`mlxr/layers`). The `image` feature gates per-family vision towers
(currently the Qwen 3-VL tower).

## Platform

Apple Silicon (M-series). Metal-only GPU path; CPU fallback works
elsewhere but is not the target. Tested on macOS 14+.

## License

Dual-licensed under MIT and Apache 2.0. See [LICENSE-MIT](LICENSE-MIT)
and [LICENSE-APACHE](LICENSE-APACHE).
