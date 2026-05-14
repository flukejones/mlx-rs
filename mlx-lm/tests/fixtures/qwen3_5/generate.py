"""Generate Python reference fixtures for the Qwen3.5 / Chandra-OCR-2 port.

Run inside a virtualenv that has `mlx`, `mlx_vlm` and `numpy` installed. The
fixtures are written to the directory this script lives in:

    python3 mlx-lm/tests/fixtures/qwen3_5/generate.py \
        --model ~/MLXModels/chandra2/chandra-ocr-2-mlx-q8

Outputs:

- `first_logits_hello.npz`  — `prompt_ids`, `logits[-1]` after a single
  forward pass on the chat-rendered `Hello` prompt.
- `greedy_hello_64.json`    — first 64 greedy tokens decoded from the same
  prompt, plus the rendered prompt string.

Both fixtures are deterministic at temperature 0 and pin the Python
reference for the Rust parity tests in `mlx-lm/tests/qwen3_5_parity.rs`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def _chat_prompt_for_hello() -> str:
    return (
        "<|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to a chandra-ocr-2 / Qwen3.5 MLX checkpoint directory.",
    )
    parser.add_argument(
        "--max-tokens",
        default=64,
        type=int,
        help="Number of greedy tokens to materialise for the second fixture.",
    )
    args = parser.parse_args()

    try:
        import mlx.core as mx
        from mlx_vlm import load, generate
        from tokenizers import Tokenizer
    except ImportError as e:
        print(f"missing dependency: {e}", file=sys.stderr)
        return 1

    model, processor = load(str(args.model))
    tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))

    prompt = _chat_prompt_for_hello()
    ids = tok.encode(prompt, add_special_tokens=False).ids
    print(f"prompt tokens: {len(ids)} -> {ids}")

    # First-logit fixture: forward the prompt once, capture the last logit row.
    input_ids = mx.array([ids], dtype=mx.int32)
    lm = model.language_model
    cache = lm.make_cache()
    out = lm(input_ids, cache=cache)
    last_logits = np.array(out.logits[0, -1, :].astype(mx.float32))
    fixtures_dir = Path(__file__).resolve().parent
    np.savez(
        fixtures_dir / "first_logits_hello.npz",
        prompt_ids=np.array(ids, dtype=np.int32),
        last_logits=last_logits,
    )
    print(
        "saved first_logits_hello.npz:",
        f"prompt_ids={len(ids)} logits.shape={last_logits.shape}",
    )

    # Layer-bisection fixture: embeddings only.
    embed = lm.model.embed_tokens(input_ids)
    embed_np = np.array(embed.astype(mx.float32))
    np.savez(
        fixtures_dir / "embeddings_hello.npz",
        prompt_ids=np.array(ids, dtype=np.int32),
        embeddings=embed_np,
    )
    print(f"saved embeddings_hello.npz: shape={embed_np.shape}")

    # Per-layer fixtures: capture hidden after layers 0..3 and the final norm.
    cache2 = lm.make_cache()
    h = embed
    h0 = lm.model.layers[0](h, None, cache2[0])
    np.savez(
        fixtures_dir / "post_layer_0_hello.npz",
        hidden=np.array(h0.astype(mx.float32)),
    )
    print(f"saved post_layer_0_hello.npz: shape={tuple(h0.shape)}")

    # Vision parity fixture: preprocess `test_image.png`, run the vision
    # tower, and save the merger output. The Rust test reads the same
    # image through `Qwen35ImageProcessor`, runs `VisionModel.forward`,
    # and compares.
    from PIL import Image
    test_image_path = fixtures_dir / "test_image.png"
    if test_image_path.exists():
        from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor
        proc = Qwen3VLImageProcessor(
            patch_size=model.config.vision_config.patch_size,
            temporal_patch_size=model.config.vision_config.temporal_patch_size,
            merge_size=model.config.vision_config.spatial_merge_size,
        )
        out = proc([Image.open(test_image_path)])
        pixel_values = mx.array(out["pixel_values"])
        grid_thw = out["image_grid_thw"]  # ndarray shape (1, 3)
        grid_thw_mx = mx.array(grid_thw, dtype=mx.int32)
        features, _deepstack = model.vision_tower(
            pixel_values.astype(mx.bfloat16), grid_thw_mx
        )
        np.savez(
            fixtures_dir / "vision_features_test.npz",
            grid_thw=np.array(grid_thw, dtype=np.int32),
            pixel_values=np.array(pixel_values),
            features=np.array(features.astype(mx.float32)),
        )
        print(
            f"saved vision_features_test.npz: grid_thw={grid_thw.tolist()} "
            f"features.shape={tuple(features.shape)}"
        )

    # Intra-block bisection for layer 0 (a Gated DeltaNet block).
    layer0 = lm.model.layers[0]
    inputs = layer0.input_layernorm(embed)
    blk = layer0.linear_attn
    mixed_qkv = blk.in_proj_qkv(inputs)
    z = blk.in_proj_z(inputs).reshape(*inputs.shape[:-1], -1, blk.head_v_dim)
    bv = blk.in_proj_b(inputs)
    av = blk.in_proj_a(inputs)
    # Conv prep (no mask).
    B, S, _ = inputs.shape
    conv_state = mx.zeros((B, blk.conv_kernel_size - 1, blk.conv_dim), dtype=inputs.dtype)
    conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
    conv_out_raw = blk.conv1d(conv_input)
    import mlx.nn as nn
    conv_out = nn.silu(conv_out_raw)
    np.savez(
        fixtures_dir / "gdn_l0_post_conv_hello.npz",
        post_input_layernorm=np.array(inputs.astype(mx.float32)),
        mixed_qkv=np.array(mixed_qkv.astype(mx.float32)),
        z=np.array(z.astype(mx.float32)),
        bv=np.array(bv.astype(mx.float32)),
        av=np.array(av.astype(mx.float32)),
        conv_input=np.array(conv_input.astype(mx.float32)),
        conv_out_raw=np.array(conv_out_raw.astype(mx.float32)),
        conv_out_silu=np.array(conv_out.astype(mx.float32)),
    )
    print(f"saved gdn_l0_post_conv_hello.npz")

    # Greedy fixture: decode `max_tokens` tokens with temperature 0.
    result = generate(
        model,
        processor,
        prompt,
        max_tokens=args.max_tokens,
        temperature=0.0,
        verbose=False,
    )
    greedy_path = fixtures_dir / "greedy_hello_64.json"
    payload = {
        "prompt": prompt,
        "prompt_ids": ids,
        "max_tokens": args.max_tokens,
        "text": result.text,
    }
    greedy_path.write_text(json.dumps(payload, indent=2))
    print(f"saved {greedy_path.name}: text={result.text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
