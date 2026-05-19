//! Cross-validation parity tests against Python `mlx_vlm` reference fixtures.
//!
//! Regenerate the fixtures with
//!
//!     python3 mlx-lm/tests/fixtures/qwen3_5/generate.py \
//!         --model ~/MLXModels/chandra2/chandra-ocr-2-mlx-q8
//!
//! These tests are all `#[ignore]`-gated; they require the chandra-ocr-2
//! q8 checkpoint on disk and the npz/json fixtures alongside this file.

#![allow(clippy::print_stderr, reason = "test output")]
#![allow(clippy::unwrap_used, reason = "test code")]
#![allow(clippy::missing_assert_message, reason = "test code")]
#![allow(clippy::print_stdout, reason = "test code")]

use std::path::{Path, PathBuf};

use mlx_lm::models::qwen3_5::{
    cache::make_caches,
    config::ModelConfig,
    image_processor::Qwen35ImageProcessor,
    weights::{load_full_model, load_language_model},
};
use mlx_rs::{transforms::eval, Array, Dtype};

const FIXTURES: &str = "tests/fixtures/qwen3_5";

fn flatten_f32(arr: &Array) -> Vec<f32> {
    let total: i32 = arr.shape().iter().product();
    let flat = arr.reshape(&[total]).unwrap();
    let evald = flat.add(Array::from_f32(0.0)).unwrap();
    eval([&evald]).unwrap();
    let f = evald.as_dtype(Dtype::Float32).unwrap();
    f.as_slice::<f32>().to_vec()
}

fn read_npz_last_logits(path: &Path) -> Vec<f32> {
    // The fixture is `np.savez(..., last_logits=...)` -> deflate `.npz` archive
    // of plain `.npy` files. Decoding `.npy` v1.0 with a flat f32 array is a
    // short, dependency-free read.
    use std::io::Read;
    let f = std::fs::File::open(path).expect("open npz");
    let mut zip = zip::ZipArchive::new(f).expect("open zip");
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).unwrap();
        let name = entry.name().to_owned();
        if !name.contains("last_logits") {
            continue;
        }
        let mut buf = Vec::new();
        entry.read_to_end(&mut buf).unwrap();
        return parse_npy_f32(&buf);
    }
    panic!("last_logits not found in {path:?}");
}

fn parse_npy_f32(buf: &[u8]) -> Vec<f32> {
    assert_eq!(&buf[0..6], b"\x93NUMPY");
    let major = buf[6];
    assert!(major == 1 || major == 2);
    let header_len = if major == 1 {
        u16::from_le_bytes([buf[8], buf[9]]) as usize
    } else {
        u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize
    };
    let header_end = if major == 1 { 10 } else { 12 } + header_len;
    let body = &buf[header_end..];
    assert!(body.len().is_multiple_of(4), "non-f32 npy body");
    body.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_npz_field(path: &Path, field: &str) -> Vec<f32> {
    use std::io::Read;
    let f = std::fs::File::open(path).expect("open npz");
    let mut zip = zip::ZipArchive::new(f).expect("open zip");
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).unwrap();
        let name = entry.name().to_owned();
        if !name.contains(field) {
            continue;
        }
        let mut buf = Vec::new();
        entry.read_to_end(&mut buf).unwrap();
        return parse_npy_f32(&buf);
    }
    panic!("{field} not found in {path:?}");
}

#[test]
#[ignore = "requires chandra-ocr-2-mlx-q8 + regenerated fixtures"]
fn embeddings_match_python_reference() {
    let home = std::env::var("HOME").unwrap();
    let model_dir = PathBuf::from(&home).join("MLXModels/chandra2/chandra-ocr-2-mlx-q8");
    let cfg = ModelConfig::from_file(model_dir.join("config.json")).unwrap();
    let (mut model, _) = load_language_model(&cfg, &model_dir).unwrap();

    let prompt_ids: Vec<i32> = vec![
        248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
    ];
    let s = prompt_ids.len() as i32;
    let inputs = Array::from_slice(&prompt_ids, &[1, s]);

    use mlx_rs::module::Module;
    let emb = model.model.embed_tokens.forward(&inputs).unwrap();
    eval([&emb]).unwrap();
    let rust = flatten_f32(&emb);

    let py = read_npz_field(
        &PathBuf::from(FIXTURES).join("embeddings_hello.npz"),
        "embeddings",
    );
    assert_eq!(rust.len(), py.len(), "embedding length mismatch");
    let (max_abs, _max_rel, mean_abs) = stats(&rust, &py);
    eprintln!("embedding diff: max_abs={max_abs:.6} mean_abs={mean_abs:.6}");
    assert!(max_abs < 1e-3, "embeddings diverged: max_abs={max_abs}");
}

#[test]
#[ignore = "diagnostic: dumps the loaded input_layernorm weight statistics"]
fn dump_input_layernorm_weight_stats() {
    let home = std::env::var("HOME").unwrap();
    let model_dir = PathBuf::from(&home).join("MLXModels/chandra2/chandra-ocr-2-mlx-q8");
    let cfg = ModelConfig::from_file(model_dir.join("config.json")).unwrap();
    let (model, _) = load_language_model(&cfg, &model_dir).unwrap();
    let w = &model.model.layers[0].input_layernorm.weight.value;
    let v = flatten_f32(w);
    let mean: f64 = v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64;
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min = v.iter().copied().fold(f32::INFINITY, f32::min);
    eprintln!(
        "rust input_layernorm.weight first10={:?} mean={mean:.6} min={min} max={max}",
        &v[..10]
    );
}

#[test]
#[ignore = "requires chandra-ocr-2-mlx-q8 + regenerated fixtures"]
fn vision_features_match_python_reference() {
    use mlx_rs::Dtype;
    let home = std::env::var("HOME").unwrap();
    let model_dir = PathBuf::from(&home).join("MLXModels/chandra2/chandra-ocr-2-mlx-q8");
    let cfg = ModelConfig::from_file(model_dir.join("config.json")).unwrap();
    let (_, mut vision, _) = load_full_model(&cfg, &model_dir).unwrap();

    let processor = Qwen35ImageProcessor::from_dir(&model_dir).unwrap();
    let img_path = PathBuf::from(FIXTURES).join("test_image.png");
    let processed = processor.preprocess_path(&img_path).unwrap();
    let num_patches = (processed.pixel_values.len() / processed.feature_dim as usize) as i32;
    let pixel_array = Array::from_slice(
        &processed.pixel_values,
        &[num_patches, processed.feature_dim],
    )
    .as_dtype(Dtype::Bfloat16)
    .unwrap();

    let features = vision.forward(&pixel_array, &[processed.grid_thw]).unwrap();
    eval([&features]).unwrap();
    let rust = flatten_f32(&features);

    let py = read_npz_field(
        &PathBuf::from(FIXTURES).join("vision_features_test.npz"),
        "features",
    );
    assert_eq!(rust.len(), py.len(), "feature length mismatch");
    let (max_abs, _, mean_abs) = stats(&rust, &py);
    eprintln!("vision features diff: max_abs={max_abs:.6} mean_abs={mean_abs:.6}");
    // ViT depth 24 in bf16 — allow up to ~0.5 max_abs and ~0.05 mean_abs.
    assert!(max_abs < 0.5, "vision features diverged: max_abs={max_abs}");
    assert!(
        mean_abs < 0.05,
        "vision features diverged: mean_abs={mean_abs}"
    );
}

#[test]
#[ignore = "requires chandra-ocr-2-mlx-q8 + regenerated fixtures"]
fn gdn_l0_post_input_layernorm_matches_python() {
    let home = std::env::var("HOME").unwrap();
    let model_dir = PathBuf::from(&home).join("MLXModels/chandra2/chandra-ocr-2-mlx-q8");
    let cfg = ModelConfig::from_file(model_dir.join("config.json")).unwrap();
    let (mut model, _) = load_language_model(&cfg, &model_dir).unwrap();

    let prompt_ids: Vec<i32> = vec![
        248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
    ];
    let inputs = Array::from_slice(&prompt_ids, &[1, prompt_ids.len() as i32]);
    use mlx_rs::module::Module;
    let emb = model.model.embed_tokens.forward(&inputs).unwrap();
    let layer0 = &mut model.model.layers[0];
    let normed = layer0.input_layernorm.forward(&emb).unwrap();
    eval([&normed]).unwrap();
    let rust = flatten_f32(&normed);
    let py = read_npz_field(
        &PathBuf::from(FIXTURES).join("gdn_l0_post_conv_hello.npz"),
        "post_input_layernorm",
    );
    let (max_abs, _, mean_abs) = stats(&rust, &py);
    eprintln!("input_layernorm diff: max_abs={max_abs:.6} mean_abs={mean_abs:.6}");
    assert!(
        max_abs < 1e-3,
        "input_layernorm diverged: max_abs={max_abs}"
    );
}

#[test]
#[ignore = "requires chandra-ocr-2-mlx-q8 + regenerated fixtures"]
fn gdn_l0_post_conv_matches_python() {
    let home = std::env::var("HOME").unwrap();
    let model_dir = PathBuf::from(&home).join("MLXModels/chandra2/chandra-ocr-2-mlx-q8");
    let cfg = ModelConfig::from_file(model_dir.join("config.json")).unwrap();
    let (mut model, _) = load_language_model(&cfg, &model_dir).unwrap();

    let prompt_ids: Vec<i32> = vec![
        248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
    ];
    let inputs = Array::from_slice(&prompt_ids, &[1, prompt_ids.len() as i32]);
    use mlx_rs::module::Module;
    let emb = model.model.embed_tokens.forward(&inputs).unwrap();
    let layer0 = &mut model.model.layers[0];
    let normed = layer0.input_layernorm.forward(&emb).unwrap();
    let blk = layer0.linear_attn.as_mut().expect("linear_attn at layer 0");

    // Replicate the conv prep of the block forward up to conv_out (without
    // dispatching the gated_delta scan).
    use mlx_rs::ops::{concatenate_axis, zeros};
    let mixed_qkv = blk.in_proj_qkv.forward(&normed).unwrap();
    let b = mixed_qkv.shape()[0];
    let history_len = blk.conv_kernel_size - 1;
    let conv_dim = blk.conv_dim;
    let conv_state = zeros::<f32>(&[b, history_len, conv_dim]).unwrap();
    let conv_input = concatenate_axis(&[conv_state, mixed_qkv.clone()], 1).unwrap();
    let conv_out = blk.conv1d.forward(&conv_input).unwrap();
    let conv_out_silu = mlx_rs::nn::silu(conv_out.clone()).unwrap();
    eval([&conv_out, &conv_out_silu]).unwrap();

    let rust_qkv = flatten_f32(&mixed_qkv);
    let py_qkv = read_npz_field(
        &PathBuf::from(FIXTURES).join("gdn_l0_post_conv_hello.npz"),
        "mixed_qkv",
    );
    let (a, _, m) = stats(&rust_qkv, &py_qkv);
    eprintln!("mixed_qkv      diff: max_abs={a:.6} mean_abs={m:.6}");

    let rust_co = flatten_f32(&conv_out);
    let py_co = read_npz_field(
        &PathBuf::from(FIXTURES).join("gdn_l0_post_conv_hello.npz"),
        "conv_out_raw",
    );
    let (a, _, m) = stats(&rust_co, &py_co);
    eprintln!("conv_out_raw   diff: max_abs={a:.6} mean_abs={m:.6}");

    let rust_silu = flatten_f32(&conv_out_silu);
    let py_silu = read_npz_field(
        &PathBuf::from(FIXTURES).join("gdn_l0_post_conv_hello.npz"),
        "conv_out_silu",
    );
    let (a, _, m) = stats(&rust_silu, &py_silu);
    eprintln!("conv_out_silu  diff: max_abs={a:.6} mean_abs={m:.6}");
    // bf16 ops with rounding mode differences from upstream mlx — accept up
    // to ~0.2 max_abs on the conv path (mean is still ~1e-4).
    assert!(a < 0.2, "conv_out_silu diverged: max_abs={a}");
}

#[test]
#[ignore = "requires chandra-ocr-2-mlx-q8 + regenerated fixtures"]
fn post_layer_0_matches_python_reference() {
    let home = std::env::var("HOME").unwrap();
    let model_dir = PathBuf::from(&home).join("MLXModels/chandra2/chandra-ocr-2-mlx-q8");
    let cfg = ModelConfig::from_file(model_dir.join("config.json")).unwrap();
    let (mut model, _) = load_language_model(&cfg, &model_dir).unwrap();

    let prompt_ids: Vec<i32> = vec![
        248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
    ];
    let s = prompt_ids.len() as i32;
    let inputs = Array::from_slice(&prompt_ids, &[1, s]);

    use mlx_rs::module::Module;
    let emb = model.model.embed_tokens.forward(&inputs).unwrap();
    let mut caches = make_caches(&cfg);

    // Run only the first decoder layer.
    let layer0 = &mut model.model.layers[0];
    let cache0 = &mut caches[0];
    let h0 = layer0
        .forward(&emb, None, None, Some(cache0), None)
        .unwrap();
    eval([&h0]).unwrap();
    let rust = flatten_f32(&h0);

    let py = read_npz_field(
        &PathBuf::from(FIXTURES).join("post_layer_0_hello.npz"),
        "hidden",
    );
    assert_eq!(rust.len(), py.len(), "post-layer-0 length mismatch");
    let (max_abs, _max_rel, mean_abs) = stats(&rust, &py);
    eprintln!("post-layer-0 diff: max_abs={max_abs:.6} mean_abs={mean_abs:.6}");
    // bf16 numerics — accept ~0.05 max_abs at the residual-stream entry to
    // layer 1. Mean stays four orders of magnitude lower.
    assert!(max_abs < 0.05, "post-layer-0 diverged: max_abs={max_abs}");
}

#[test]
#[ignore = "requires chandra-ocr-2-mlx-q8 + regenerated fixtures"]
fn first_token_logits_match_python_reference() {
    let home = std::env::var("HOME").unwrap();
    let model_dir = PathBuf::from(&home).join("MLXModels/chandra2/chandra-ocr-2-mlx-q8");
    let cfg = ModelConfig::from_file(model_dir.join("config.json")).unwrap();
    let (mut model, _) = load_language_model(&cfg, &model_dir).unwrap();

    let prompt_ids: Vec<i32> = vec![
        248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
    ];
    let s = prompt_ids.len() as i32;
    let inputs = Array::from_slice(&prompt_ids, &[1, s]);
    let mut caches = make_caches(&cfg);
    let logits = model
        .forward(Some(&inputs), None, &mut caches, None)
        .unwrap();
    eval([&logits]).unwrap();
    assert_eq!(logits.shape(), &[1, s, cfg.text_config.vocab_size]);

    use mlx_rs::ops::indexing::IndexOp;
    let last = logits.index((0, -1, ..));
    let rust = flatten_f32(&last);

    let py = read_npz_last_logits(&PathBuf::from(FIXTURES).join("first_logits_hello.npz"));
    assert_eq!(rust.len(), py.len(), "logits length mismatch");

    let (top_rust, top_py) = (argmax(&rust), argmax(&py));
    eprintln!(
        "rust top1: id={} val={:.4}; python top1: id={} val={:.4}",
        top_rust.0, top_rust.1, top_py.0, top_py.1
    );

    let (max_abs, max_rel, mean_abs) = stats(&rust, &py);
    eprintln!("logits diff: max_abs={max_abs:.4} max_rel={max_rel:.4} mean_abs={mean_abs:.4}",);

    // Greedy-equivalence is the load-bearing assertion — once top-1 matches
    // the generation loop produces the same string as Python `mlx_vlm`.
    // The numeric tolerance is loose because we're 32 bf16 layers deep.
    assert!(
        top_rust.0 == top_py.0,
        "top-1 disagreement: rust={} python={}",
        top_rust.0,
        top_py.0
    );
    assert!(max_abs < 0.5, "max_abs={max_abs} exceeded 0.5");
    let _ = (max_rel, mean_abs);
}

fn argmax(v: &[f32]) -> (usize, f32) {
    v.iter()
        .enumerate()
        .fold((0_usize, f32::NEG_INFINITY), |acc, (i, &x)| {
            if x > acc.1 {
                (i, x)
            } else {
                acc
            }
        })
}

fn stats(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    let mut max_abs = 0_f32;
    let mut max_rel = 0_f32;
    let mut sum = 0_f64;
    for (&x, &y) in a.iter().zip(b) {
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
        let denom = y.abs().max(1e-6);
        let r = d / denom;
        if r > max_rel {
            max_rel = r;
        }
        sum += d as f64;
    }
    (max_abs, max_rel, (sum / a.len() as f64) as f32)
}
