//! Steel-attention parity tests.
//!
//! **A3 scope**: D=128, non-causal, no mask. Validates the kernel
//! against `mlxr::fast::scaled_dot_product_attention` (no-mask,
//! no-causal) at fp16/bf16 tolerances.

#![allow(clippy::unwrap_used, reason = "test code")]
#![allow(clippy::missing_assert_message, reason = "test code")]
#![allow(clippy::print_stdout, reason = "test code")]
#![allow(clippy::print_stderr, reason = "test code")]

use mlxr::error::Result;
use mlxr::fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlxr::random::{key, normal};
use mlxr::transforms::eval;
use mlxr::{Array, Dtype};

use mlxr::ops::{dequantize, quantize};

use crate::attention::{
    attention_dispatch, make_attention_kernel, make_quant_attention_kernel,
    quant_attention_dispatch, AttentionInputs, QuantAttentionInputs,
};

fn max_abs(a: &Array, b: &Array) -> Result<f32> {
    Ok(a.subtract(b)?.abs()?.max(None)?.item::<f32>())
}

fn reference_sdpa(q: &Array, k: &Array, v: &Array, scale: f32) -> Result<Array> {
    scaled_dot_product_attention(
        q,
        k,
        v,
        scale,
        Option::<ScaledDotProductAttentionMask<'_>>::None,
        None,
    )
}

fn reference_sdpa_causal(q: &Array, k: &Array, v: &Array, scale: f32) -> Result<Array> {
    scaled_dot_product_attention(q, k, v, scale, ScaledDotProductAttentionMask::Causal, None)
}

fn rand_4d(shape: &[i32], dtype: Dtype, seed: u64) -> Result<Array> {
    let kctx = key(seed)?;
    let a = normal::<f32>(shape, None, None, &kctx)?;
    a.as_dtype(dtype)
}

#[allow(clippy::too_many_arguments, reason = "test helper: shape parameters")]
fn parity_case_inner(
    d: i32,
    b: i32,
    h_q: i32,
    h_kv: i32,
    q_len: i32,
    k_len: i32,
    dtype: Dtype,
    tolerance: f32,
    causal: bool,
) -> Result<()> {
    let kernel = make_attention_kernel()?;
    let q = rand_4d(&[b, h_q, q_len, d], dtype, 1)?;
    let k = rand_4d(&[b, h_kv, k_len, d], dtype, 2)?;
    let v = rand_4d(&[b, h_kv, k_len, d], dtype, 3)?;
    eval([&q, &k, &v])?;

    let scale = 1.0 / (d as f32).sqrt();
    let steel = attention_dispatch(
        &kernel,
        AttentionInputs {
            q: &q,
            k: &k,
            v: &v,
            mask: None,
            causal,
            ql_off: 0,
            scale,
            head_dim: d,
            h_q,
            h_kv,
        },
    )?;
    let reference = if causal {
        reference_sdpa_causal(&q, &k, &v, scale)?
    } else {
        reference_sdpa(&q, &k, &v, scale)?
    };
    eval([&steel, &reference])?;

    assert_eq!(steel.shape(), reference.shape());
    let diff = max_abs(&steel, &reference)?;
    assert!(
        diff <= tolerance,
        "steel vs fast::SDPA diverge ({b},{h_q},{q_len},{k_len}, {dtype:?}, causal={causal}): max_abs={diff} > {tolerance}"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments, reason = "test helper: shape parameters")]
fn parity_case(
    d: i32,
    b: i32,
    h_q: i32,
    h_kv: i32,
    q_len: i32,
    k_len: i32,
    dtype: Dtype,
    tolerance: f32,
) -> Result<()> {
    parity_case_inner(d, b, h_q, h_kv, q_len, k_len, dtype, tolerance, false)
}

#[allow(clippy::too_many_arguments, reason = "test helper: shape parameters")]
fn causal_case(
    d: i32,
    b: i32,
    h_q: i32,
    h_kv: i32,
    q_len: i32,
    k_len: i32,
    dtype: Dtype,
    tolerance: f32,
) -> Result<()> {
    parity_case_inner(d, b, h_q, h_kv, q_len, k_len, dtype, tolerance, true)
}

// ===== D=128 =====

#[test]
fn d128_aligned_short_fp16() {
    parity_case(128, 1, 8, 8, 64, 64, Dtype::Float16, 5e-3).unwrap();
}

#[test]
fn d128_aligned_long_fp16() {
    parity_case(128, 1, 8, 8, 256, 256, Dtype::Float16, 5e-3).unwrap();
}

#[test]
fn d128_gqa_fp16() {
    // GQA: H_q=16, H_kv=8 → groups of 2 q-heads per kv-head.
    parity_case(128, 1, 16, 8, 128, 128, Dtype::Float16, 5e-3).unwrap();
}

#[test]
fn d128_unaligned_q_fp16() {
    // qL = 65 (one tail Q-block past 64 = 2*BQ).
    parity_case(128, 1, 8, 8, 65, 128, Dtype::Float16, 5e-3).unwrap();
}

#[test]
fn d128_unaligned_k_fp16() {
    // kL = 70 (one tail K-block past 64 = 4*BK).
    parity_case(128, 1, 8, 8, 128, 70, Dtype::Float16, 5e-3).unwrap();
}

#[test]
fn d128_aligned_short_bf16() {
    // bf16 has 7 mantissa bits → coarser tolerance.
    parity_case(128, 1, 8, 8, 64, 64, Dtype::Bfloat16, 5e-2).unwrap();
}

#[test]
fn d128_batch_fp16() {
    parity_case(128, 2, 8, 8, 64, 64, Dtype::Float16, 5e-3).unwrap();
}

// ===== D=256 (Qwen 3.6, Gemma 3, Gemma 4 local) =====
// TG memory budget: (BQ + 2*BK) * BD * sizeof(T) = (32 + 32) * 256 * 2
// = 32 KB. At Apple's 32 KB limit; passes on M-series.

#[test]
fn d256_aligned_short_fp16() {
    parity_case(256, 1, 8, 8, 64, 64, Dtype::Float16, 5e-3).unwrap();
}

#[test]
fn d256_aligned_long_fp16() {
    parity_case(256, 1, 8, 8, 256, 256, Dtype::Float16, 5e-3).unwrap();
}

#[test]
fn d256_gqa_fp16() {
    parity_case(256, 1, 16, 8, 128, 128, Dtype::Float16, 5e-3).unwrap();
}

#[test]
fn d256_unaligned_q_fp16() {
    parity_case(256, 1, 8, 8, 65, 128, Dtype::Float16, 5e-3).unwrap();
}

#[test]
fn d256_unaligned_k_fp16() {
    parity_case(256, 1, 8, 8, 128, 70, Dtype::Float16, 5e-3).unwrap();
}

#[test]
fn d256_aligned_short_bf16() {
    parity_case(256, 1, 8, 8, 64, 64, Dtype::Bfloat16, 5e-2).unwrap();
}

// ===== Causal mask =====
// Standard autoregressive prefill: qL == kL, lower-triangular mask.

#[test]
fn d128_causal_aligned_fp16() {
    causal_case(128, 1, 8, 8, 64, 64, Dtype::Float16, 5e-3).unwrap();
}

#[test]
fn d128_causal_aligned_long_fp16() {
    causal_case(128, 1, 8, 8, 256, 256, Dtype::Float16, 5e-3).unwrap();
}

#[test]
fn d128_causal_gqa_fp16() {
    causal_case(128, 1, 16, 8, 128, 128, Dtype::Float16, 5e-3).unwrap();
}

#[test]
fn d128_causal_unaligned_fp16() {
    causal_case(128, 1, 8, 8, 65, 65, Dtype::Float16, 5e-3).unwrap();
}

#[test]
fn d256_causal_aligned_fp16() {
    causal_case(256, 1, 8, 8, 64, 64, Dtype::Float16, 5e-3).unwrap();
}

#[test]
fn d256_causal_aligned_long_fp16() {
    causal_case(256, 1, 8, 8, 256, 256, Dtype::Float16, 5e-3).unwrap();
}

#[test]
fn d128_causal_bf16() {
    causal_case(128, 1, 8, 8, 64, 64, Dtype::Bfloat16, 5e-2).unwrap();
}

// ===== D=512 =====
// Smaller tile (BQ=8, BK=8, WM=1, WN=1) so TG mem fits 32 KB cap.

#[test]
fn d512_aligned_short_fp16() {
    parity_case(512, 1, 4, 4, 16, 16, Dtype::Float16, 5e-3).unwrap();
}

#[test]
fn d512_aligned_long_fp16() {
    parity_case(512, 1, 4, 4, 64, 64, Dtype::Float16, 5e-3).unwrap();
}

#[test]
fn d512_gqa_fp16() {
    parity_case(512, 1, 8, 4, 32, 32, Dtype::Float16, 5e-3).unwrap();
}

#[test]
fn d512_unaligned_q_fp16() {
    parity_case(512, 1, 4, 4, 17, 32, Dtype::Float16, 5e-3).unwrap();
}

#[test]
fn d512_unaligned_k_fp16() {
    parity_case(512, 1, 4, 4, 32, 19, Dtype::Float16, 5e-3).unwrap();
}

#[test]
fn d512_aligned_short_bf16() {
    parity_case(512, 1, 4, 4, 16, 16, Dtype::Bfloat16, 5e-2).unwrap();
}

#[test]
fn d512_causal_aligned_fp16() {
    causal_case(512, 1, 4, 4, 16, 16, Dtype::Float16, 5e-3).unwrap();
}

#[test]
fn d512_causal_aligned_long_fp16() {
    causal_case(512, 1, 4, 4, 64, 64, Dtype::Float16, 5e-3).unwrap();
}

// ===== Quantised K/V =====

#[allow(clippy::too_many_arguments, reason = "test helper: shape parameters")]
fn quant_case(
    d: i32,
    b: i32,
    h_q: i32,
    h_kv: i32,
    q_len: i32,
    k_len: i32,
    dtype: Dtype,
    bits: i32,
    group_size: i32,
    causal: bool,
    tolerance: f32,
) -> Result<()> {
    let kernel = make_quant_attention_kernel()?;
    let q = rand_4d(&[b, h_q, q_len, d], dtype, 11)?;
    let k_dense = rand_4d(&[b, h_kv, k_len, d], dtype, 12)?;
    let v_dense = rand_4d(&[b, h_kv, k_len, d], dtype, 13)?;
    eval([&q, &k_dense, &v_dense])?;

    // Quantise K/V the same way QuantizedKVCache does.
    let (k_wq, k_s, k_b) = quantize(&k_dense, group_size, bits)?;
    let (v_wq, v_s, v_b) = quantize(&v_dense, group_size, bits)?;

    let scale = 1.0 / (d as f32).sqrt();
    let routed = quant_attention_dispatch(
        &kernel,
        QuantAttentionInputs {
            q: &q,
            k_wq: &k_wq,
            k_scales: &k_s,
            k_biases: &k_b,
            v_wq: &v_wq,
            v_scales: &v_s,
            v_biases: &v_b,
            mask: None,
            causal,
            ql_off: 0,
            scale,
            head_dim: d,
            h_q,
            h_kv,
            bits,
            group_size,
        },
    )?;

    // Reference: dequant to dense, then run standard SDPA. This matches
    // what the kernel computes (the kernel dequants in-place during the
    // tile load), modulo accumulation-order rounding.
    let k_ref = dequantize(&k_wq, &k_s, &k_b, group_size, bits)?;
    let v_ref = dequantize(&v_wq, &v_s, &v_b, group_size, bits)?;
    let reference = if causal {
        reference_sdpa_causal(&q, &k_ref, &v_ref, scale)?
    } else {
        reference_sdpa(&q, &k_ref, &v_ref, scale)?
    };
    eval([&routed, &reference])?;

    assert_eq!(routed.shape(), reference.shape());
    let diff = max_abs(&routed, &reference)?;
    assert!(
        diff <= tolerance,
        "steel_quant vs dequant+SDPA diverge \
         (d={d}, b={b}, hq={h_q}, hkv={h_kv}, qL={q_len}, kL={k_len}, \
          {dtype:?}, bits={bits}, group={group_size}, causal={causal}): \
         max_abs={diff} > {tolerance}"
    );
    Ok(())
}

#[test]
fn quant_d128_b8_g64_fp16_noncausal() {
    quant_case(128, 1, 8, 8, 64, 64, Dtype::Float16, 8, 64, false, 5e-3).unwrap();
}

#[test]
fn quant_d128_b8_g64_fp16_causal() {
    quant_case(128, 1, 8, 8, 64, 64, Dtype::Float16, 8, 64, true, 5e-3).unwrap();
}

#[test]
fn quant_d128_b4_g64_fp16_noncausal() {
    quant_case(128, 1, 8, 8, 64, 64, Dtype::Float16, 4, 64, false, 5e-3).unwrap();
}

#[test]
fn quant_d128_b4_g64_fp16_causal() {
    quant_case(128, 1, 8, 8, 64, 64, Dtype::Float16, 4, 64, true, 5e-3).unwrap();
}

#[test]
fn quant_d128_b8_gqa_fp16() {
    quant_case(128, 1, 16, 8, 128, 128, Dtype::Float16, 8, 64, true, 5e-3).unwrap();
}

#[test]
fn quant_d128_b8_long_fp16() {
    quant_case(128, 1, 8, 8, 256, 256, Dtype::Float16, 8, 64, true, 5e-3).unwrap();
}

#[test]
fn quant_d256_b8_g64_fp16() {
    quant_case(256, 1, 8, 8, 64, 64, Dtype::Float16, 8, 64, true, 5e-3).unwrap();
}

#[test]
fn quant_d256_b4_g64_fp16() {
    quant_case(256, 1, 8, 8, 64, 64, Dtype::Float16, 4, 64, true, 5e-3).unwrap();
}

#[test]
fn quant_d128_b8_bf16() {
    quant_case(128, 1, 8, 8, 64, 64, Dtype::Bfloat16, 8, 64, true, 5e-2).unwrap();
}

#[test]
fn quant_d512_b8_g64_fp16() {
    quant_case(512, 1, 4, 4, 16, 16, Dtype::Float16, 8, 64, true, 5e-3).unwrap();
}

#[test]
fn quant_d512_b4_g64_fp16() {
    quant_case(512, 1, 4, 4, 16, 16, Dtype::Float16, 4, 64, true, 5e-3).unwrap();
}
