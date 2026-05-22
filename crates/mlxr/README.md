# mlxr

Rust port of MLX's high-level API: tensors, layers, autograd,
transforms, FFT, linalg, fast Metal kernels.

This is the consumer-facing tensor library. Builds on
[`mlxr-sys`](../mlxr-sys/) (the bindgen FFI shim against mlx-c).
[`mlxr-lm`](../mlxr-lm/) builds the LM runtime + model families on
top of this crate.

## Install

```toml
[dependencies]
mlxr = "0.27"
```

## Features

Default features: `accelerate`, `metal`, `layers`, `fft`, `linalg`,
`fast`, `losses`, `optimizers`.

| Feature       | Surface                                                                 |
|---------------|-------------------------------------------------------------------------|
| `accelerate`  | Forwarded to `mlxr-sys/accelerate` — Apple Accelerate CPU backend        |
| `metal`       | Forwarded to `mlxr-sys/metal` — Apple Metal GPU backend                  |
| `layers`      | `mlxr::layers::*` — Linear, Conv, RmsNorm, Embedding, RoPE, transformer (cascades `fast`) |
| `fft`         | `mlxr::fft::*` — fft / ifft / rfft / shifts                              |
| `linalg`      | `mlxr::linalg::*` — qr, svd, norms                                       |
| `fast`        | `mlxr::fast::*` — fused Metal kernels (sdpa, rms_norm, metal_kernel)     |
| `losses`      | `mlxr::losses::*` — MSE, cross-entropy, etc.                             |
| `optimizers`  | `mlxr::optimizers::*` — Adam, AdamW, AdaFactor, etc.                     |
| `safetensors` | `Array ↔ safetensors::TensorView` conversion                             |

Disable subsystems your build doesn't need:

```toml
mlxr = { version = "0.27", default-features = false, features = ["metal", "layers"] }
```

Tensors (`array`, `ops`, `transforms`, `quantization`, `memory`,
`module`) are always compiled.

## Quick start

```rust
use mlxr::{array, ops, Dtype};

let a = array!([1.0_f32, 2.0, 3.0, 4.0]);
let b = array!([5.0_f32, 6.0, 7.0, 8.0]);
let c = ops::add(&a, &b)?;
c.eval()?;
assert_eq!(c.as_slice::<f32>(), &[6.0, 8.0, 10.0, 12.0]);
```

See [`examples/`](examples/) for `tutorial.rs` (broad API walk) and
`linear_regression.rs` (one-shot training with `value_and_grad`).

## Lazy evaluation

Operations build a compute graph; nothing executes until `eval()`
(implicit on `print`, `as_slice`, `item`, `save_*`). Idiomatic
training loops `eval` once per outer iteration:

```rust
for batch in dataset {
    let (loss, grad) = value_and_grad_fn(&mut model, batch)?;
    optimizer.update(&mut model, grad)?;
    eval_params(model.parameters())?;
}
```

## Platform

Apple Silicon (M-series), macOS 14+. Other targets work for CPU-only
codepaths but are not the development focus.

## License

Dual-licensed under MIT and Apache 2.0.
