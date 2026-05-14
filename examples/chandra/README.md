# Chandra

Rust inference for the [Chandra-OCR-2](https://huggingface.co/datalab-to/chandra-ocr-2-mlx)
checkpoint (Qwen3.5 hybrid Gated-DeltaNet + sparse full-attention), built on
top of [`mlx-rs`](https://github.com/oxideai/mlx-rs) and the new `mlx-lm`
`models::qwen3_5` module.

## Status

- **Text-only generation** matches Python `mlx_vlm.generate` token-for-token
  against the on-disk `chandra-ocr-2-mlx-q8` checkpoint.
- **Vision / OCR** runs the full Qwen3-VL vision tower + multimodal
  stitching against the same checkpoint. Pass `--image <path>` for the CLI,
  or send a data-URL `image_url` content-part to the HTTP server.

## Build

```bash
cargo build --release -p chandra
```

## Run modes

### One-shot generation

```bash
cargo run --release -p chandra -- \
    --model ~/MLXModels/chandra2/chandra-ocr-2-mlx-q8 \
    --prompt "Hello" \
    --max-tokens 32
```

Expected output:

```
Hello! How can I help you?
tokens: prompt=13 completion=8 finish=Stop
```

### OpenAI-compatible HTTP server

```bash
cargo run --release -p chandra -- \
    --model ~/MLXModels/chandra2/chandra-ocr-2-mlx-q8 \
    --serve --port 8088
```

`POST /v1/chat/completions`:

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chandra",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 32
  }'
```

Returns the standard OpenAI shape with `choices[0].message.content`,
`finish_reason`, and `usage`.

Image-bearing requests of the form
`{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`
are decoded, fed through the vision tower, and stitched into the prompt
embeddings before generation. `http(s)://` URLs are also supported and
fetched server-side with a 30 s timeout and a 64 MiB body cap.

Setting `"stream": true` switches the response to OpenAI-style
Server-Sent Events: one `delta.role` chunk, per-token `delta.content`
chunks, a final `finish_reason` chunk, and `data: [DONE]`.

## CLI flags

| Flag             | Default | Description                                                          |
| ---------------- | ------- | -------------------------------------------------------------------- |
| `--model`        | —       | Path to a local MLX-format checkpoint directory.                     |
| `--prompt`       | —       | Prompt string for one-shot mode (mutually exclusive with `--serve`). |
| `--image`        | —       | Image to OCR. Decoded with the `image` crate (PNG/JPEG/BMP/TIFF/WebP/GIF). |
| `--max-tokens`   | `512`   | Maximum new tokens to generate.                                      |
| `--temperature`  | `0.0`   | Sampling temperature. `0.0` selects greedy / argmax.                 |
| `--top-p`        | —       | Top-p (nucleus) value; ignored when `temperature == 0`.              |
| `--seed`         | —       | PRNG seed for sampling.                                              |
| `--serve`        | `false` | Start the HTTP server instead of running one-shot.                   |
| `--port`         | `8088`  | Port to bind when `--serve` is set.                                  |
