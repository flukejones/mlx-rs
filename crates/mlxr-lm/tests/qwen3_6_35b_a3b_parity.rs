//! Qwen3.6-35B-A3B MoE smoke test against the unified surface.
//!
//! `#[ignore]`-gated: needs `lmstudio-community/Qwen3.6-35B-A3B-MLX-8bit`
//! on disk (~35 GB). Drives the load + a short generation through
//! `mlxr_lm::load` + `mlxr_lm::generate`; passing means the
//! per-tensor quantisation overrides bound correctly *and* end-to-end
//! inference produces tokens.

#![allow(clippy::missing_assert_message, reason = "test code")]

use std::path::PathBuf;

use mlxr_lm::chat_template::ChatMessage;
use mlxr_lm::{generate, load, GenerateParams, Sampler, UserInput};

const Q8_PATH: &str = ".cache/mlx-rs-bench/lmstudio-community/Qwen3.6-35B-A3B-MLX-8bit";
const Q8_MTP_PATH: &str = ".cache/mlx-rs-bench/mlx-community/Qwen3.6-35B-A3B-q8-mtp";

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}

#[test]
#[ignore = "requires lmstudio-community/Qwen3.6-35B-A3B-MLX-8bit on disk"]
fn loader_and_generate_q8() {
    let dir = home().join(Q8_PATH);
    let mut ctx = load(&dir).expect("load");
    let input = UserInput::text("Hello");
    let params = GenerateParams {
        max_new_tokens: 4,
        ..GenerateParams::default()
    };
    let result = generate(&mut ctx, input, params, &mut |_, _| {
        std::ops::ControlFlow::Continue(())
    })
    .expect("generate");
    assert!(result.completion_tokens > 0);
}

/// Same shape as `loader_and_generate_q8` but against a checkpoint
/// that ships the MTP head weights (`mtp.fc`, `mtp.layers.0.*`, etc.).
/// Passing means the MtpHead struct + the loader's MTP binding both
/// work; failure here is an unbound-key error from the qwen3_5_moe loader.
#[test]
#[ignore = "requires mlx-community/Qwen3.6-35B-A3B-q8-mtp on disk"]
fn loader_and_generate_q8_mtp() {
    let dir = home().join(Q8_MTP_PATH);
    let mut ctx = load(&dir).expect("load");
    let input = UserInput::text("Hello");
    let params = GenerateParams {
        max_new_tokens: 4,
        ..GenerateParams::default()
    };
    let result = generate(&mut ctx, input, params, &mut |_, _| {
        std::ops::ControlFlow::Continue(())
    })
    .expect("generate");
    assert!(result.completion_tokens > 0);
}

/// Greedy MTP must produce byte-identical output to greedy non-MTP.
/// Runs the same 32-token prompt twice: once with `disable_mtp: true`
/// (forces the per-token path), once with `false` (speculative path).
/// The `result.text` must match exactly. Any divergence is a verify
/// bug — the speculative path's accept/restore is mis-handling some
/// state (cache, hidden, or sampler).
#[test]
#[ignore = "requires mlx-community/Qwen3.6-35B-A3B-q8-mtp on disk"]
fn mtp_greedy_matches_non_mtp_greedy_q8() {
    let dir = home().join(Q8_MTP_PATH);
    let mut ctx = load(&dir).expect("load");

    fn run(ctx: &mut mlxr_lm::ModelContext, disable_mtp: bool) -> String {
        let input = UserInput::text("Once upon a time");
        let params = GenerateParams {
            max_new_tokens: 32,
            disable_mtp,
            ..GenerateParams::default()
        };
        generate(ctx, input, params, &mut |_, _| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("generate")
        .text
    }

    let non_mtp = run(&mut ctx, true);
    let with_mtp = run(&mut ctx, false);

    assert_eq!(
        non_mtp, with_mtp,
        "greedy MTP diverged from greedy non-MTP:\n  non-mtp: {non_mtp:?}\n  with-mtp: {with_mtp:?}"
    );
}

/// Same as the depth-1 greedy parity test but with the MTP draft
/// chained to depth 2 (two MTP forwards per speculative call, then a
/// 3-token main verify). All cache-rollback / re-prime paths in the
/// walk-back accept must keep output bit-identical to non-MTP greedy.
#[test]
#[ignore = "requires mlx-community/Qwen3.6-35B-A3B-q8-mtp on disk"]
fn mtp_greedy_depth2_matches_non_mtp_greedy_q8() {
    let dir = home().join(Q8_MTP_PATH);
    let mut ctx = load(&dir).expect("load");

    fn run(ctx: &mut mlxr_lm::ModelContext, disable_mtp: bool, depth: u32) -> String {
        ctx.model.set_mtp_depth(depth);
        let input = UserInput::text("Once upon a time");
        let params = GenerateParams {
            max_new_tokens: 32,
            disable_mtp,
            ..GenerateParams::default()
        };
        generate(ctx, input, params, &mut |_, _| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("generate")
        .text
    }

    let non_mtp = run(&mut ctx, true, 1);
    let with_mtp_d2 = run(&mut ctx, false, 2);

    assert_eq!(
        non_mtp, with_mtp_d2,
        "greedy MTP depth=2 diverged from greedy non-MTP:\n  non-mtp:  {non_mtp:?}\n  depth=2:  {with_mtp_d2:?}"
    );
}

/// Same as the depth-2 greedy parity test but at depth 3. Verifies
/// the walk-back / cache-rollback logic at the new cap.
#[test]
#[ignore = "requires mlx-community/Qwen3.6-35B-A3B-q8-mtp on disk"]
fn mtp_greedy_depth3_matches_non_mtp_greedy_q8() {
    let dir = home().join(Q8_MTP_PATH);
    let mut ctx = load(&dir).expect("load");

    fn run(ctx: &mut mlxr_lm::ModelContext, disable_mtp: bool, depth: u32) -> String {
        ctx.model.set_mtp_depth(depth);
        let input = UserInput::text("Once upon a time");
        let params = GenerateParams {
            max_new_tokens: 32,
            disable_mtp,
            ..GenerateParams::default()
        };
        generate(ctx, input, params, &mut |_, _| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("generate")
        .text
    }

    let non_mtp = run(&mut ctx, true, 1);
    let with_mtp_d3 = run(&mut ctx, false, 3);

    assert_eq!(
        non_mtp, with_mtp_d3,
        "greedy MTP depth=3 diverged from greedy non-MTP:\n  non-mtp:  {non_mtp:?}\n  depth=3:  {with_mtp_d3:?}"
    );
}

/// Sampled (temp > 0, top-p) generation through the MTP path. The
/// gate no longer requires `temperature == 0` — the adapter runs
/// Leviathan-2023 rejection sampling so the output distribution
/// matches the non-MTP per-step path. Exact bit-parity against the
/// non-MTP run is *not* expected (RNG draws happen in different
/// places). Smoke: assert >0 tokens and non-empty text. Early-EOS
/// is legitimate under sampling once `<|im_end|>` is in the stop
/// set, so we can't assert the full budget was used.
#[test]
#[ignore = "requires mlx-community/Qwen3.6-35B-A3B-q8-mtp on disk"]
fn mtp_sampled_q8_produces_text() {
    let dir = home().join(Q8_MTP_PATH);
    let mut ctx = load(&dir).expect("load");

    let input = UserInput::text("The capital of France is");
    let params = GenerateParams {
        max_new_tokens: 32,
        sampling: Sampler::TopP {
            temperature: 0.7,
            p: 0.95,
        },
        ..GenerateParams::default()
    };
    let result = generate(&mut ctx, input, params, &mut |_, _| {
        std::ops::ControlFlow::Continue(())
    })
    .expect("generate");
    assert!(
        result.completion_tokens > 0,
        "sampled MTP produced 0 tokens ({:?})",
        result.finish_reason
    );
    assert!(!result.text.is_empty(), "sampled MTP text empty");
}

/// MTP path under sampling without top-p. Exercises the no-mask
/// branch of `mtp_step_sampled` (the union-mask code is skipped
/// when `params.top_p.is_none()`).
#[test]
#[ignore = "requires mlx-community/Qwen3.6-35B-A3B-q8-mtp on disk"]
fn mtp_sampled_no_top_p_q8_produces_text() {
    let dir = home().join(Q8_MTP_PATH);
    let mut ctx = load(&dir).expect("load");

    let input = UserInput::text("Write a haiku about Rust:");
    let params = GenerateParams {
        max_new_tokens: 24,
        sampling: Sampler::Temperature(1.0),
        ..GenerateParams::default()
    };
    let result = generate(&mut ctx, input, params, &mut |_, _| {
        std::ops::ControlFlow::Continue(())
    })
    .expect("generate");
    assert!(
        result.completion_tokens > 0,
        "sampled MTP (no top-p) produced 0 tokens ({:?})",
        result.finish_reason
    );
    assert!(!result.text.is_empty(), "sampled MTP (no top-p) text empty");
}

/// Same parity check via the chat-template path. The chat input goes
/// through `ChatTemplate::render` + tokeniser before reaching the
/// model, so the prompt is longer and includes role markers — exercises
/// the MTP loop's interaction with prompt prefix length / starting
/// hidden state, not just raw-text decode.
#[test]
#[ignore = "requires mlx-community/Qwen3.6-35B-A3B-q8-mtp on disk"]
fn mtp_greedy_matches_non_mtp_greedy_q8_chat() {
    let dir = home().join(Q8_MTP_PATH);
    let mut ctx = load(&dir).expect("load");

    fn run(ctx: &mut mlxr_lm::ModelContext, disable_mtp: bool) -> String {
        let input = UserInput::chat(vec![ChatMessage::user("Write a haiku about Rust.")]);
        let params = GenerateParams {
            max_new_tokens: 32,
            disable_mtp,
            ..GenerateParams::default()
        };
        generate(ctx, input, params, &mut |_, _| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("generate")
        .text
    }

    let non_mtp = run(&mut ctx, true);
    let with_mtp = run(&mut ctx, false);

    assert_eq!(
        non_mtp, with_mtp,
        "greedy MTP diverged from greedy non-MTP (chat input):\n  non-mtp: {non_mtp:?}\n  with-mtp: {with_mtp:?}"
    );
}
