//! Qwen3.5-MoE (35B-A3B) [`crate::LanguageModel`] adapter.
//!
//! Same prefill / decode shape as the dense qwen3.5 adapter; the
//! only difference is the inner FFN type (`Qwen35MoeBlock`). No
//! multimodal path — MoE checkpoints are text-only.

use std::path::Path;

use mlxr::ops::indexing::{take_along_axis, IndexOp};
use mlxr::ops::{concatenate_axis, exp, log, maximum, r#where, sum_axis};
use mlxr::random::uniform;
use mlxr::{argmax_axis, categorical, Array};

use crate::chat_template::ChatTemplate;
use crate::error::Error;
use crate::family::LoadedContext;
use crate::language_model::{LanguageModel, TextOnlyProcessor};
use crate::lm_input::{LMInput, LMOutput, PrepareResult};
use crate::loader::load_tokenizer;
use crate::qwen3_5::text::cache::{make_caches, make_mtp_caches, LayerCache};
use crate::qwen3_5::text::config::ModelConfig;
use crate::qwen3_5::text::layer::Qwen35Model;
use crate::qwen3_5::text::moe::{load_qwen3_5_moe_model, Qwen35MoeBlock};
use crate::qwen3_5::text::sampling::top_p_keep_mask;
use crate::sampler::SamplerState;

pub(crate) struct Qwen35MoeAdapter {
    model: Qwen35Model<Qwen35MoeBlock>,
    cfg: ModelConfig,
    cache: Vec<LayerCache>,
    /// Per-MTP-layer caches. Empty when the checkpoint has no MTP head.
    mtp_cache: Vec<LayerCache>,
    /// *Pre*-final-norm hidden at the last decoded position, sliced
    /// to `[B=1, 1, hidden]`. The MTP head re-norms via its own
    /// `pre_fc_norm_hidden`, so it must receive the unnormed input.
    /// `None` before the first prepare/step.
    prev_hidden: Option<Array>,
    vocab_size: i32,
}

impl Qwen35MoeAdapter {
    fn load(dir: &Path) -> Result<Self, Error> {
        let model = load_qwen3_5_moe_model(dir)?;
        let cfg = ModelConfig::from_file(dir.join("config.json"))?;
        let cache = make_caches(&cfg);
        let mtp_cache = make_mtp_caches(&cfg);
        let vocab_size = cfg.text_config.vocab_size;
        Ok(Self {
            model,
            cfg,
            cache,
            mtp_cache,
            prev_hidden: None,
            vocab_size,
        })
    }
}

impl LanguageModel for Qwen35MoeAdapter {
    fn reset(&mut self) {
        self.cache = make_caches(&self.cfg);
        self.mtp_cache = make_mtp_caches(&self.cfg);
        self.prev_hidden = None;
    }

    fn prepare(&mut self, input: LMInput) -> Result<PrepareResult, Error> {
        debug_assert!(input.image.is_none());
        debug_assert!(input.audio.is_none());
        debug_assert!(input.video.is_none());

        let tokens = input.text.tokens;
        let (hidden, logits) =
            self.model
                .forward_hidden_and_logits(Some(&tokens), None, &mut self.cache, None)?;
        self.prev_hidden = Some(hidden.index((.., -1..)));
        Ok(PrepareResult::Logits(logits.index((.., -1, ..))))
    }

    fn step(&mut self, last_token: &Array) -> Result<LMOutput, Error> {
        let inp = last_token.reshape(&[1, 1])?;
        let (hidden, logits) =
            self.model
                .forward_hidden_and_logits(Some(&inp), None, &mut self.cache, None)?;
        self.prev_hidden = Some(hidden.index((.., -1..)));
        Ok(LMOutput {
            logits: logits.index((.., -1, ..)),
        })
    }

    fn vocab_size(&self) -> i32 {
        self.vocab_size
    }

    fn has_mtp(&self) -> bool {
        self.model.mtp.is_some()
    }

    fn try_mtp_decode(
        &mut self,
        last_token: &Array,
        sampler: &mut SamplerState,
    ) -> Result<Option<(Vec<u32>, Array)>, Error> {
        if self.model.mtp.is_none() {
            return Ok(None);
        }
        if sampler.params().temperature == 0.0 {
            mtp_step_greedy(self, last_token).map(Some)
        } else {
            mtp_step_sampled(self, last_token, sampler).map(Some)
        }
    }
}

/// Greedy speculative step (temperature = 0).
///
/// Inputs:
/// - `last_token`: candidate for the next-to-commit slot. Its KV is
///   not yet in the cache.
/// - `self.prev_hidden`: *pre*-final-norm hidden at the most-recently
///   committed cache slot. The MTP head applies its own
///   `pre_fc_norm_hidden` so it needs the pre-norm input.
///
/// Algorithm:
/// 1. MTP forward on `prev_hidden` + embed(last_token) → draft token
///    (candidate for the slot AFTER last_token).
/// 2. Snapshot caches.
/// 3. Two-token main forward `[last_token, draft]`. Commits both
///    KV slots. Returns logits at both positions:
///      - `verify_logits[0]`: what the model thinks comes after
///        last_token (apples-to-apples vs the draft).
///      - `verify_logits[1]`: what comes after draft (the next
///        not-yet-committed pending if we accept).
/// 4. Compare `argmax(verify_logits[0])` to draft.
///    - MATCH: accept both. Emit `[last_token, draft]`. New
///      `prev_hidden` is the hidden at slot 1. Next pending =
///      `argmax(verify_logits[1])` (not yet committed).
///    - MISMATCH: roll back. The correct token after last_token is
///      `corrected = argmax(verify_logits[0])`. Re-run a single
///      forward on `last_token` to commit just its slot. Emit
///      `[last_token]`. New `prev_hidden` is the hidden at the
///      committed slot. Next pending = `corrected`.
fn mtp_step_greedy(
    adapter: &mut Qwen35MoeAdapter,
    last_token: &Array,
) -> Result<(Vec<u32>, Array), Error> {
    let prev_hidden = adapter.prev_hidden.clone().ok_or_else(|| {
        Error::Other("mtp_step_greedy: prev_hidden unset; call prepare first".into())
    })?;

    let last_token_2d = last_token.reshape(&[1, 1])?;
    let last_host = last_token.item::<i32>();
    let last_u32 = host_id_to_u32(last_host, adapter.vocab_size)?;

    let draft_logits = run_mtp(adapter, &prev_hidden, &last_token_2d)?;
    let draft_id = argmax_axis!(&draft_logits, -1)?.reshape(&[1])?;
    let draft_host = draft_id.item::<i32>();

    // Snapshot caches before the 2-token verify forward so reject can
    // roll back both. `mtp_cache` advances by 1 in `run_mtp` above; the
    // snapshot here freezes that post-MTP state.
    let cache_snapshot = adapter.cache.clone();
    let mtp_snapshot = adapter.mtp_cache.clone();

    let pair = concatenate_axis(&[&last_token_2d, &draft_id.reshape(&[1, 1])?], 1)?;
    let (verify_hidden, verify_logits) =
        adapter
            .model
            .forward_hidden_and_logits(Some(&pair), None, &mut adapter.cache, None)?;
    let verify_first = verify_logits.index((.., 0, ..));
    let verify_first_id = argmax_axis!(&verify_first, -1)?.reshape(&[1])?;
    let verify_first_host = verify_first_id.item::<i32>();

    if verify_first_host == draft_host {
        adapter.prev_hidden = Some(verify_hidden.index((.., -1..)));
        let draft_u32 = host_id_to_u32(draft_host, adapter.vocab_size)?;
        let next_pending = argmax_axis!(&verify_logits.index((.., 1, ..)), -1)?.reshape(&[1])?;
        return Ok((vec![last_u32, draft_u32], next_pending));
    }

    adapter.cache = cache_snapshot;
    adapter.mtp_cache = mtp_snapshot;
    let (rehidden, _) = adapter.model.forward_hidden_and_logits(
        Some(&last_token_2d),
        None,
        &mut adapter.cache,
        None,
    )?;
    adapter.prev_hidden = Some(rehidden.index((.., -1..)));
    Ok((vec![last_u32], verify_first_id))
}

/// Sampled speculative step (temperature > 0).
///
/// Same prepare/forward shape as `mtp_step_greedy`, but the accept
/// rule is full Leviathan-2023 rejection sampling so the output
/// distribution matches what the non-MTP per-step path would have
/// drawn from. Cross-checked against AirRunner's mlx-lm MTP fork.
///
/// 1. MTP forward → `draft_logits` ([1, vocab]).
/// 2. Build a shared *union* top-p keep mask from
///    `draft_logits` ∪ `verify_logits[0]` (when top-p is set, else
///    no mask) so the per-side log-probs share a domain.
/// 3. Scale + mask + log_softmax both distributions:
///    `draft_lp = log_softmax((draft_logits / temp) ∧ keep)`,
///    `verify_lp_0 = log_softmax((verify_logits[0] / temp) ∧ keep)`.
/// 4. Sample `d ~ draft_lp` (categorical on the masked scaled
///    logits, by `exp(log_softmax) = softmax`).
/// 5. Two-token verify forward `[last_token, d]`.
/// 6. Accept iff `verify_lp_0[d] - draft_lp[d] >= log(u)` for
///    `u ~ U(0, 1)`.
///    - Accept: emit `[last_token, d]`, sample bonus next-pending
///      from `verify_logits[1]` via the caller's `SamplerState`.
///    - Reject: roll back caches, sample `corrected` from the
///      residual `r = max(0, exp(verify_lp_0) - exp(draft_lp))`
///      (fall back to `verify_lp_0` if `r.sum() == 0`), re-run a
///      single forward on `last_token` to commit its slot, emit
///      `[last_token]`, next pending = `corrected`.
fn mtp_step_sampled(
    adapter: &mut Qwen35MoeAdapter,
    last_token: &Array,
    sampler: &mut SamplerState,
) -> Result<(Vec<u32>, Array), Error> {
    let prev_hidden = adapter.prev_hidden.clone().ok_or_else(|| {
        Error::Other("mtp_step_sampled: prev_hidden unset; call prepare first".into())
    })?;

    let last_token_2d = last_token.reshape(&[1, 1])?;
    let last_host = last_token.item::<i32>();
    let last_u32 = host_id_to_u32(last_host, adapter.vocab_size)?;

    let top_p = sampler.params().top_p;

    let draft_logits = run_mtp(adapter, &prev_hidden, &last_token_2d)?;

    // Snapshot caches before the 2-token verify forward so reject can
    // roll back both.
    let cache_snapshot = adapter.cache.clone();
    let mtp_snapshot = adapter.mtp_cache.clone();

    // Verify forward first, so we have `verify_logits[0]` available
    // to build the *union* top-p mask. The verify pass uses a
    // placeholder draft `d=0` for now and we'll re-issue after
    // sampling the real `d` — except we can't, since the cache is
    // already advanced. Instead: sample `d` from draft_logits using
    // just the draft-side mask, do verify with the real `d`, then
    // do the accept test with the union mask.
    let draft_only_mask = if let Some(p) = top_p {
        Some(top_p_keep_mask(&draft_logits, p)?)
    } else {
        None
    };
    let draft_lp_for_sample = sampler.masked_log_probs(&draft_logits, draft_only_mask.as_ref())?;
    // `categorical!` samples from logits proportional to exp(x). The
    // log-probs ARE logits up to an additive constant that softmax
    // absorbs, so sampling from them directly is equivalent to
    // sampling from the scaled+masked distribution.
    let draft_id = categorical!(&draft_lp_for_sample)?.reshape(&[1])?;
    let draft_host = draft_id.item::<i32>();

    let pair = concatenate_axis(&[&last_token_2d, &draft_id.reshape(&[1, 1])?], 1)?;
    let (verify_hidden, verify_logits) =
        adapter
            .model
            .forward_hidden_and_logits(Some(&pair), None, &mut adapter.cache, None)?;
    let verify_first = verify_logits.index((.., 0, ..));

    // Now build the union mask over both sides and re-derive
    // log-probs the accept test will use. Without top-p the mask is
    // None and both distributions span the full vocab.
    let keep_mask = if let Some(p) = top_p {
        let verify_mask = top_p_keep_mask(&verify_first, p)?;
        let draft_mask = draft_only_mask
            .as_ref()
            .expect("top_p_keep_mask was built above when top_p is Some");
        Some(draft_mask.logical_or(&verify_mask)?)
    } else {
        None
    };
    let draft_lp = sampler.masked_log_probs(&draft_logits, keep_mask.as_ref())?;
    let verify_lp_0 = sampler.masked_log_probs(&verify_first, keep_mask.as_ref())?;

    let draft_id_2d = draft_id.reshape(&[1, 1])?;
    let lp_v = take_along_axis(&verify_lp_0, &draft_id_2d, -1)?.reshape(&[])?;
    let lp_d = take_along_axis(&draft_lp, &draft_id_2d, -1)?.reshape(&[])?;
    let log_ratio = lp_v.subtract(&lp_d)?;
    let u = uniform::<_, f32>(0.0_f32, 1.0_f32, &[], None)?;
    let log_u = log(&u)?.as_dtype(log_ratio.dtype())?;
    let accept = log_ratio.ge(&log_u)?.item::<bool>();

    if accept {
        adapter.prev_hidden = Some(verify_hidden.index((.., -1..)));
        let draft_u32 = host_id_to_u32(draft_host, adapter.vocab_size)?;
        // Bonus next-pending from verify_logits[1]: same sampler the
        // main loop would have used, so top-p/temp apply identically.
        let next_logits = verify_logits.index((.., 1, ..));
        let next_pending = sampler.sample(&next_logits)?;
        return Ok((vec![last_u32, draft_u32], next_pending));
    }

    // Reject. Restore caches and re-run a single forward on
    // `last_token` to commit only its slot.
    adapter.cache = cache_snapshot;
    adapter.mtp_cache = mtp_snapshot;

    let p_v = exp(&verify_lp_0)?;
    let p_d = exp(&draft_lp)?;
    let zero = Array::from_f32(0.0).as_dtype(p_v.dtype())?;
    let residual = maximum(&p_v.subtract(&p_d)?, &zero)?;
    let z = sum_axis(&residual, -1, true)?;
    // Host sync on z: the residual either has non-zero mass (sample
    // from it) or sums to zero (degenerate case where the union
    // top-p mask collapsed the support — fall back to verify). The
    // branch is cheap and only fires on the reject path.
    let z_host = z.reshape(&[])?.item::<f32>();
    let corrected = if z_host > 0.0 {
        // Sample from residual. Pass log(residual) to `categorical!`
        // so it samples proportional to residual probs. residual
        // has exact zeros where verify-prob <= draft-prob; replace
        // those with `0` (so `log` produces -inf) to mask them out.
        let mask = residual.gt(&zero)?;
        let safe = r#where(&mask, &residual, &zero)?;
        let log_r = log(&safe)?;
        categorical!(&log_r)?.reshape(&[1])?
    } else {
        // Fallback: sample from verify distribution directly.
        categorical!(&verify_lp_0)?.reshape(&[1])?
    };

    let (rehidden, _) = adapter.model.forward_hidden_and_logits(
        Some(&last_token_2d),
        None,
        &mut adapter.cache,
        None,
    )?;
    adapter.prev_hidden = Some(rehidden.index((.., -1..)));
    Ok((vec![last_u32], corrected))
}

/// Drive the MTP head for one draft step. Returns the projected logits
/// at position t+2 (a `[1, vocab]` array).
fn run_mtp(
    adapter: &mut Qwen35MoeAdapter,
    prev_hidden: &Array,
    last_token_2d: &Array,
) -> Result<Array, Error> {
    let embed_next = adapter.model.embed_tokens(last_token_2d)?;
    let mtp = adapter
        .model
        .mtp
        .as_mut()
        .expect("run_mtp: caller checked mtp.is_some()");
    let mtp_hidden = mtp.forward(prev_hidden, &embed_next, &mut adapter.mtp_cache, None)?;
    let logits = adapter.model.apply_lm_head(&mtp_hidden)?;
    Ok(logits.index((.., -1, ..)))
}

fn host_id_to_u32(id: i32, vocab: i32) -> Result<u32, Error> {
    if id < 0 || id >= vocab {
        return Err(Error::Shape(format!(
            "mtp_step: out-of-vocab id {id} (vocab = {vocab})"
        )));
    }
    Ok(id as u32)
}

pub(crate) fn load_context_moe(dir: &Path) -> Result<LoadedContext, Error> {
    let model = Qwen35MoeAdapter::load(dir)?;
    let cfg = ModelConfig::from_file(dir.join("config.json"))?;
    let tokenizer = load_tokenizer(dir)?;
    let chat_template = ChatTemplate::from_dir(dir)?;
    let eos_ids = crate::qwen3_5::text::read_qwen3_5_eos_ids(dir, &cfg);
    let processor = TextOnlyProcessor::new("qwen3_5_moe", tokenizer, chat_template);
    Ok((Box::new(model), Box::new(processor), eos_ids))
}
