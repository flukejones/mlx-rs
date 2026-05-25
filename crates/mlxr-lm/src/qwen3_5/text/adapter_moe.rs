//! Qwen3.5-MoE (35B-A3B) [`crate::LanguageModel`] adapter.
//!
//! Same prefill / decode shape as the dense qwen3.5 adapter; the
//! only difference is the inner FFN type (`Qwen35MoeBlock`). No
//! multimodal path — MoE checkpoints are text-only.

use std::path::Path;

use mlxr::ops::indexing::{take_along_axis, IndexOp};
use mlxr::ops::{concatenate_axis, exp, log, maximum, r#where, stack_axis, sum_axis};
use mlxr::random::uniform;
use mlxr::{argmax_axis, categorical, Array};

use crate::cache::{build_rotation, CacheOptions};
use crate::config::ModelConfig as Config;
use crate::error::Error;
use crate::family::LoadedContext;
use crate::language_model::{LanguageModel, TextOnlyProcessor};
use crate::lm_input::{LMInput, LMOutput, PrepareResult};
use crate::qwen3_5::text::cache::{
    make_caches_with_rotation, make_mtp_caches_with_rotation, LayerCache,
};
use crate::qwen3_5::text::config::ModelConfig;
use crate::qwen3_5::text::layer::Qwen35Model;
use crate::qwen3_5::text::load_common;
use crate::qwen3_5::text::moe::{load_qwen3_5_moe_model, Qwen35MoeBlock};
use crate::qwen3_5::text::sampling::top_p_keep_mask;
use crate::sampler::{Sampler, SamplerState};

/// Upper bound on MTP draft depth. The walk-back algorithm is
/// depth-generic; this cap reflects the depth past which adding
/// drafts stops paying its cost on the per-call wall clock. On a
/// bandwidth-bound 35B MoE at 89% per-slot acceptance, depth-3
/// double-reject probability is `0.11² ≈ 1%` and triple-reject is
/// `0.11³ ≈ 0.1%`, so the expected cache-restore + re-prime cost
/// stays small. Past 3 the marginal D→D+1 acceptance ratio
/// (`accept²` for each added slot) makes the verify-forward cost
/// dominate.
pub const MAX_MTP_DEPTH: u32 = 3;

pub struct Qwen35MoeAdapter {
    model: Qwen35Model<Qwen35MoeBlock>,
    cfg: ModelConfig,
    cache: Vec<LayerCache>,
    /// Per-MTP-layer caches. Empty when the checkpoint has no MTP head.
    mtp_cache: Vec<LayerCache>,
    cache_options: CacheOptions,
    /// TurboQuant rotation matrix, shared across `cache` and
    /// `mtp_cache`. `None` for non-quantised or no-seed configs.
    rotation: Option<Array>,
    /// Post-final-norm hidden at the last decoded position, sliced
    /// to `[B=1, 1, hidden]`. Fed into the MTP head, which applies
    /// its own `pre_fc_norm_hidden` on top. `None` before the first
    /// prepare/step.
    prev_hidden: Option<Array>,
    vocab_size: i32,
    /// Number of tokens the MTP head drafts ahead per speculative
    /// call. Default 1; bump via [`Self::set_mtp_depth`]. Clamped to
    /// `1..=MAX_MTP_DEPTH`.
    mtp_depth: u32,
}

impl Qwen35MoeAdapter {
    pub fn load(cfg: &Config, env: &ModelConfig, dir: &Path) -> Result<Self, Error> {
        let model = load_qwen3_5_moe_model(cfg, env, dir)?;
        let cache_options = CacheOptions::default();
        let rotation = build_rotation(cache_options, env.text_config.head_dim)?;
        let cache = make_caches_with_rotation(env, cache_options, rotation.as_ref());
        let mtp_cache = make_mtp_caches_with_rotation(env, cache_options, rotation.as_ref());
        let vocab_size = env.text_config.vocab_size;
        Ok(Self {
            model,
            cfg: env.clone(),
            cache,
            mtp_cache,
            cache_options,
            rotation,
            prev_hidden: None,
            vocab_size,
            mtp_depth: 1,
        })
    }

    /// Current MTP draft depth.
    pub fn mtp_depth(&self) -> u32 {
        self.mtp_depth
    }
}

impl LanguageModel for Qwen35MoeAdapter {
    fn reset(&mut self) {
        self.cache =
            make_caches_with_rotation(&self.cfg, self.cache_options, self.rotation.as_ref());
        self.mtp_cache =
            make_mtp_caches_with_rotation(&self.cfg, self.cache_options, self.rotation.as_ref());
        self.prev_hidden = None;
    }

    fn prepare(&mut self, input: LMInput) -> Result<PrepareResult, Error> {
        debug_assert!(input.image.is_none());

        let tokens = input.text.tokens;
        let (hidden, logits) =
            self.model
                .forward_hidden_and_logits(Some(&tokens), None, &mut self.cache, None)?;
        prime_mtp_cache(&mut self.model, &tokens, &hidden, &mut self.mtp_cache)?;
        self.prev_hidden = Some(hidden.index((.., -1..)));
        Ok(PrepareResult::Logits(logits.index((.., -1, ..))))
    }

    fn step(&mut self, last_token: &Array) -> Result<LMOutput, Error> {
        let inp = last_token.reshape(&[1, 1])?;
        let prior_hidden = self.prev_hidden.clone();
        let (hidden, logits) =
            self.model
                .forward_hidden_and_logits(Some(&inp), None, &mut self.cache, None)?;
        // Keep the MTP cache offset in lockstep with the main cache so a
        // subsequent `try_mtp_decode` call sees matching RoPE positions.
        // No-op when the model ships no MTP head.
        if self.model.mtp.is_some() {
            if let Some(prior) = prior_hidden.as_ref() {
                let embed_next = self.model.embed_tokens(&inp)?;
                let mtp = self.model.mtp.as_mut().expect("checked mtp.is_some()");
                mtp.update_cache(prior, &embed_next, &mut self.mtp_cache)?;
            }
        }
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
        mtp_step(self, last_token, sampler).map(Some)
    }

    fn set_mtp_depth(&mut self, n: u32) {
        self.mtp_depth = n.clamp(1, MAX_MTP_DEPTH);
    }

    fn prefill_chunk_size(&self) -> Option<i32> {
        // Qwen3.5 caches are unbounded; user cap wins.
        self.cache_options.max_prefill_chunk
    }

    fn prefill_chunk(&mut self, tokens: &Array) -> Result<(), Error> {
        // MoE prefill chunks must also advance the MTP cache —
        // otherwise speculative decode after the final chunk sees
        // out-of-sync RoPE positions.
        let (hidden, _logits) =
            self.model
                .forward_hidden_and_logits(Some(tokens), None, &mut self.cache, None)?;
        prime_mtp_cache(&mut self.model, tokens, &hidden, &mut self.mtp_cache)?;
        // Track `prev_hidden` so the final `prepare` chunk's MTP step
        // has the right anchor.
        self.prev_hidden = Some(hidden.index((.., -1..)));
        Ok(())
    }

    fn set_cache_options(&mut self, options: CacheOptions) -> Result<(), Error> {
        let rotation = build_rotation(options, self.cfg.text_config.head_dim)?;
        let cache = make_caches_with_rotation(&self.cfg, options, rotation.as_ref());
        let mtp_cache = make_mtp_caches_with_rotation(&self.cfg, options, rotation.as_ref());
        self.cache = cache;
        self.mtp_cache = mtp_cache;
        self.rotation = rotation;
        self.cache_options = options;
        Ok(())
    }
}

/// One speculative MTP step.
///
/// Inputs:
/// - `last_token`: candidate for the next-to-commit slot. Its KV is
///   not yet in the cache.
/// - `self.prev_hidden`: post-final-norm hidden at the most-recently
///   committed cache slot. The Qwen 3.6 MTP head was trained against
///   the model's final-norm output and re-normalises via its own
///   `pre_fc_norm_hidden` on top.
/// - `sampler`: at `temperature == 0.0` the helpers below take the
///   greedy fast path (argmax draft, argmax-equality accept, argmax
///   resample). Above 0 they use Leviathan rejection sampling with a
///   shared union top-p mask between draft and verify distributions.
///
/// Algorithm:
/// 1. MTP forward on `prev_hidden` + embed(last_token) → draft logits.
///    Pick the draft token via [`sample_draft`].
/// 2. Snapshot caches.
/// 3. Two-token main forward `[last_token, draft]`. Returns logits at
///    both positions: `verify_logits[0]` (what comes after last_token)
///    and `verify_logits[1]` (what comes after draft).
/// 4. [`accept_draft`] tests `draft` against `verify_logits[0]`.
///    - Accept: emit `[last_token, draft]`. New `prev_hidden` is the
///      hidden at slot 1. Next pending sampled from `verify_logits[1]`.
///    - Reject: roll back caches, [`resample_on_reject`] picks the
///      corrected token from `verify_logits[0]`. Re-run a single
///      forward on `last_token` to commit just its slot. Emit
///      `[last_token]`. New `prev_hidden` is the hidden at the
///      committed slot. Next pending = corrected.
fn mtp_step(
    adapter: &mut Qwen35MoeAdapter,
    last_token: &Array,
    sampler: &mut SamplerState,
) -> Result<(Vec<u32>, Array), Error> {
    let depth = adapter.mtp_depth as usize;
    debug_assert!((1..=MAX_MTP_DEPTH as usize).contains(&depth));

    let prev_hidden = adapter
        .prev_hidden
        .clone()
        .ok_or_else(|| Error::Other("mtp_step: prev_hidden unset; call prepare first".into()))?;

    let last_token_2d = last_token.reshape(&[1, 1])?;
    // Host-read of `last_token` is deferred to after the verify forward
    // submission below so the GPU→host sync overlaps with the draft +
    // verify pipeline instead of blocking before it. `last_u32` is only
    // needed when building the `committed` return vec.

    // Snapshot caches BEFORE any draft so partial-reject can roll back
    // both main + mtp to the pre-step state and re-commit only the
    // accepted prefix. Snapshot clone is shared-ptr cheap (the Arrays
    // are `mlx::core::array` shared_ptr handles). The guard restores
    // both caches if dropped without `.commit()`, including the `?`
    // early-exit paths in the verify forward + accept_draft below.
    let mut cache_guard = CacheSnapshot::new(&adapter.cache, &adapter.mtp_cache);

    // Build the chained draft: drafts[i] predicts the token at slot
    // `last_token + i + 1`. Each MTP forward advances mtp_cache by 1
    // and produces both the draft's logits and the post-norm hidden
    // that feeds the NEXT chained MTP call as its `prev_hidden`.
    let mut draft_ids: Vec<Array> = Vec::with_capacity(depth);
    let mut draft_logits: Vec<Array> = Vec::with_capacity(depth);
    let mut prev_h = prev_hidden;
    let mut token_in = last_token_2d.clone();
    for _ in 0..depth {
        let (logits_i, mtp_post_norm_i) = run_mtp(adapter, &prev_h, &token_in)?;
        let id_i = sample_draft(sampler, &logits_i)?;
        token_in = id_i.reshape(&[1, 1])?;
        prev_h = mtp_post_norm_i;
        draft_ids.push(id_i);
        draft_logits.push(logits_i);
    }

    // Verify forward on [last_token, draft_0, .., draft_{depth-1}].
    // Main cache advances by depth+1.
    let mut verify_inputs: Vec<&Array> = Vec::with_capacity(depth + 1);
    verify_inputs.push(&last_token_2d);
    let draft_ids_2d: Vec<Array> = draft_ids
        .iter()
        .map(|d| d.reshape(&[1, 1]))
        .collect::<Result<_, _>>()?;
    for d in &draft_ids_2d {
        verify_inputs.push(d);
    }
    let verify_input = concatenate_axis(&verify_inputs, 1)?;
    let (verify_hidden, verify_logits) = adapter.model.forward_hidden_and_logits(
        Some(&verify_input),
        None,
        &mut adapter.cache,
        None,
    )?;

    // Now sync `last_token` to host — verify forward has been submitted,
    // so this read overlaps with its dispatch instead of blocking it.
    let last_u32 = host_id_to_u32(last_token.item::<i32>(), adapter.vocab_size)?;

    // Materialise host ids for every draft in one sync, instead of
    // re-syncing each `draft_ids[i]` individually inside the commit
    // loop below. Verify forward above already evaluated the chain,
    // so this stack is cheap. argmax returns u32; per-slot bounds
    // check mirrors `host_id_to_u32`.
    let draft_ids_stacked = stack_axis(&draft_ids, 0)?.reshape(&[depth as i32])?;
    let vocab_u32 = u32::try_from(adapter.vocab_size).map_err(|_| {
        Error::Shape(format!(
            "mtp_step: vocab_size {} negative",
            adapter.vocab_size
        ))
    })?;
    let draft_ids_host: Vec<u32> = draft_ids_stacked
        .as_slice::<u32>()
        .iter()
        .map(|&id| {
            if id >= vocab_u32 {
                return Err(Error::Shape(format!(
                    "mtp_step: out-of-vocab id {id} (vocab = {vocab_u32})"
                )));
            }
            Ok(id)
        })
        .collect::<Result<_, _>>()?;

    // Walk-back accept: find the first level k where verify rejects
    // the draft. k == depth means all-accept.
    let mut k = depth;
    for i in 0..depth {
        let verify_i = verify_logits.index((.., i as i32, ..));
        if !accept_draft(sampler, &draft_ids[i], &draft_logits[i], &verify_i)? {
            k = i;
            break;
        }
    }

    if k == depth {
        // All-accept: commit last_token + every draft.
        cache_guard.commit();
        adapter.prev_hidden = Some(verify_hidden.index((.., -1..)));
        let next_logits = verify_logits.index((.., depth as i32, ..));
        let next_pending = sampler.sample(&next_logits)?;
        let mut committed = Vec::with_capacity(depth + 1);
        committed.push(last_u32);
        committed.extend_from_slice(&draft_ids_host);
        return Ok((committed, next_pending));
    }

    // Partial reject at level k (0 <= k < depth). Both caches are
    // currently over-committed: main by depth+1, mtp by depth. Roll
    // back to the pre-step snapshot via the guard, then re-commit
    // exactly the accepted prefix (k+1 tokens) on the main side, plus
    // matching MTP-cache priming so the next call's RoPE positions
    // line up.
    cache_guard.rollback_into(&mut adapter.cache, &mut adapter.mtp_cache);

    let corrected = {
        let verify_k = verify_logits.index((.., k as i32, ..));
        resample_on_reject(sampler, &draft_logits[k], &verify_k)?
    };

    // Re-commit the accepted prefix in one main forward.
    let mut accept_inputs: Vec<&Array> = Vec::with_capacity(k + 1);
    accept_inputs.push(&last_token_2d);
    for d in draft_ids_2d.iter().take(k) {
        accept_inputs.push(d);
    }
    let accept_tokens = concatenate_axis(&accept_inputs, 1)?;
    let (rehidden, _) = adapter.model.forward_hidden_and_logits(
        Some(&accept_tokens),
        None,
        &mut adapter.cache,
        None,
    )?;
    // Re-prime MTP cache to match the new main cache offset. The
    // prime helper writes positions `1..k+1` (it skips position 0;
    // for k=0 the helper is a no-op since the accepted segment is
    // just `last_token` with no successor).
    prime_mtp_cache(
        &mut adapter.model,
        &accept_tokens,
        &rehidden,
        &mut adapter.mtp_cache,
    )?;
    adapter.prev_hidden = Some(rehidden.index((.., -1..)));

    let mut committed = Vec::with_capacity(k + 1);
    committed.push(last_u32);
    committed.extend_from_slice(&draft_ids_host[..k]);
    Ok((committed, corrected))
}

/// Pick the draft token. Greedy at `temperature == 0` (`argmax`);
/// categorical on the masked log-probs otherwise. The `[1]`-shape
/// output is what the rest of the speculative step expects.
fn sample_draft(sampler: &mut SamplerState, draft_logits: &Array) -> Result<Array, Error> {
    if matches!(sampler.sampler(), Sampler::Greedy) {
        return Ok(argmax_axis!(draft_logits, -1)?.reshape(&[1])?);
    }
    let top_p_mask = match sampler.sampler().top_p() {
        Some(p) => Some(top_p_keep_mask(draft_logits, p)?),
        None => None,
    };
    let lp = sampler.masked_log_probs(draft_logits, top_p_mask.as_ref())?;
    Ok(categorical!(&lp)?.reshape(&[1])?)
}

/// Decide whether the draft token is accepted at this verify position.
/// At `temperature == 0` this is exact argmax equality. Above 0 it is
/// Leviathan-2023 rejection: build a shared union top-p mask between
/// draft and verify, derive log-probs on both sides, accept iff
/// `log p_verify(draft) - log p_draft(draft) >= log u`.
fn accept_draft(
    sampler: &mut SamplerState,
    draft_id: &Array,
    draft_logits: &Array,
    verify_logits_i: &Array,
) -> Result<bool, Error> {
    if matches!(sampler.sampler(), Sampler::Greedy) {
        let verify_first_id = argmax_axis!(verify_logits_i, -1)?.reshape(&[1])?;
        return Ok(verify_first_id.eq(draft_id)?.item::<bool>());
    }
    let keep_mask = match sampler.sampler().top_p() {
        Some(p) => {
            let draft_mask = top_p_keep_mask(draft_logits, p)?;
            let verify_mask = top_p_keep_mask(verify_logits_i, p)?;
            Some(draft_mask.logical_or(&verify_mask)?)
        }
        None => None,
    };
    let draft_lp = sampler.masked_log_probs(draft_logits, keep_mask.as_ref())?;
    let verify_lp = sampler.masked_log_probs(verify_logits_i, keep_mask.as_ref())?;
    let draft_id_2d = draft_id.reshape(&[1, 1])?;
    let lp_v = take_along_axis(&verify_lp, &draft_id_2d, -1)?.reshape(&[])?;
    let lp_d = take_along_axis(&draft_lp, &draft_id_2d, -1)?.reshape(&[])?;
    let log_ratio = lp_v.subtract(&lp_d)?;
    let u = uniform::<_, f32>(0.0_f32, 1.0_f32, &[], None)?;
    let log_u = log(&u)?.as_dtype(log_ratio.dtype())?;
    Ok(log_ratio.ge(&log_u)?.item::<bool>())
}

/// Pick the corrected token at the rejected position. At
/// `temperature == 0` this is `argmax(verify_logits_i)`. Above 0 it
/// is a categorical draw from the Leviathan residual
/// `max(0, exp(verify_lp) - exp(draft_lp))`, falling back to the
/// verify distribution when the residual sums to zero (degenerate
/// case where the union top-p mask collapsed the support).
fn resample_on_reject(
    sampler: &mut SamplerState,
    draft_logits: &Array,
    verify_logits_i: &Array,
) -> Result<Array, Error> {
    if matches!(sampler.sampler(), Sampler::Greedy) {
        return Ok(argmax_axis!(verify_logits_i, -1)?.reshape(&[1])?);
    }
    let keep_mask = match sampler.sampler().top_p() {
        Some(p) => {
            let draft_mask = top_p_keep_mask(draft_logits, p)?;
            let verify_mask = top_p_keep_mask(verify_logits_i, p)?;
            Some(draft_mask.logical_or(&verify_mask)?)
        }
        None => None,
    };
    let draft_lp = sampler.masked_log_probs(draft_logits, keep_mask.as_ref())?;
    let verify_lp = sampler.masked_log_probs(verify_logits_i, keep_mask.as_ref())?;
    let p_v = exp(&verify_lp)?;
    let p_d = exp(&draft_lp)?;
    let zero = Array::from_f32(0.0).as_dtype(p_v.dtype())?;
    let residual = maximum(&p_v.subtract(&p_d)?, &zero)?;
    let z = sum_axis(&residual, -1, true)?;
    // Host sync on z: residual either has mass (sample from it) or
    // sums to zero (top-p mask collapsed the support — fall back to
    // verify). The branch is cheap and only fires on reject.
    let z_host = z.reshape(&[])?.item::<f32>();
    if z_host > 0.0 {
        let mask = residual.gt(&zero)?;
        let safe = r#where(&mask, &residual, &zero)?;
        let log_r = log(&safe)?;
        Ok(categorical!(&log_r)?.reshape(&[1])?)
    } else {
        Ok(categorical!(&verify_lp)?.reshape(&[1])?)
    }
}

/// Drive the MTP head for one draft step. Returns the projected logits
/// at position t+2 (a `[1, vocab]` array).
/// Run one MTP-head forward. Returns `(logits, mtp_post_norm)` — both
/// sliced to the last position. `logits` is `[1, vocab]` for sampling;
/// `mtp_post_norm` is `[1, 1, hidden]` for chained drafts (the next
/// MTP forward in a depth>1 chain consumes the prior level's post-norm
/// hidden as its `prev_hidden`).
fn run_mtp(
    adapter: &mut Qwen35MoeAdapter,
    prev_hidden: &Array,
    last_token_2d: &Array,
) -> Result<(Array, Array), Error> {
    let embed_next = adapter.model.embed_tokens(last_token_2d)?;
    let mtp = adapter
        .model
        .mtp
        .as_mut()
        .expect("run_mtp: caller checked mtp.is_some()");
    let mtp_hidden = mtp.forward(prev_hidden, &embed_next, &mut adapter.mtp_cache, None)?;
    let logits = adapter.model.apply_lm_head(&mtp_hidden)?;
    Ok((logits.index((.., -1, ..)), mtp_hidden.index((.., -1..))))
}

/// Populate `mtp_cache` so its offset matches the main cache after
/// prefill. Without this, the first `try_mtp_decode` call runs the
/// MTP attention block at RoPE position 0 while the main model is at
/// position `prompt_len` — the position-frequency mismatch collapses
/// MTP acceptance at long context. Mirrors what the standalone decode
/// loop would have done across the prompt one token at a time, but
/// folded into a single forward over the whole sequence.
///
/// `prompt_tokens` is `[1, N]`; `hidden_full` is the main model's
/// post-final-norm hidden over the same N positions. For N < 2 the
/// MTP head has nothing to predict from, so this is a no-op.
fn prime_mtp_cache(
    model: &mut Qwen35Model<Qwen35MoeBlock>,
    prompt_tokens: &Array,
    hidden_full: &Array,
    mtp_cache: &mut [LayerCache],
) -> Result<(), Error> {
    if model.mtp.is_none() {
        return Ok(());
    }
    let n = prompt_tokens.shape()[1];
    if n < 2 {
        return Ok(());
    }
    let next_tokens = prompt_tokens.index((.., 1..n));
    let next_embeds = model.embed_tokens(&next_tokens)?;
    let prime_hidden = hidden_full.index((.., ..n - 1));
    let mtp = model.mtp.as_mut().expect("checked mtp.is_some()");
    mtp.update_cache(&prime_hidden, &next_embeds, mtp_cache)?;
    Ok(())
}

fn host_id_to_u32(id: i32, vocab: i32) -> Result<u32, Error> {
    if id < 0 || id >= vocab {
        return Err(Error::Shape(format!(
            "mtp_step: out-of-vocab id {id} (vocab = {vocab})"
        )));
    }
    Ok(id as u32)
}

/// Pre-MTP snapshot of `(main_cache, mtp_cache)`. Holding it forces a
/// caller to choose between [`Self::commit`] (keep the over-committed
/// state — full accept path) and [`Self::rollback_into`] (restore
/// before re-committing the accepted prefix — partial reject path).
/// Dropping it without either is a logic bug; the destructor logs.
struct CacheSnapshot {
    main: Option<Vec<LayerCache>>,
    mtp: Option<Vec<LayerCache>>,
}

impl CacheSnapshot {
    fn new(main: &[LayerCache], mtp: &[LayerCache]) -> Self {
        Self {
            main: Some(main.to_vec()),
            mtp: Some(mtp.to_vec()),
        }
    }

    /// Discard the snapshot — the post-MTP cache state is what we
    /// want.
    fn commit(&mut self) {
        self.main = None;
        self.mtp = None;
    }

    /// Restore both caches from the snapshot. Consumes the snapshot
    /// fields so [`Drop`] below doesn't double-warn.
    fn rollback_into(&mut self, main: &mut Vec<LayerCache>, mtp: &mut Vec<LayerCache>) {
        if let Some(s) = self.main.take() {
            *main = s;
        }
        if let Some(s) = self.mtp.take() {
            *mtp = s;
        }
    }
}

impl Drop for CacheSnapshot {
    fn drop(&mut self) {
        if self.main.is_some() || self.mtp.is_some() {
            log::warn!(
                "CacheSnapshot dropped without commit() or rollback_into(); \
                 KV cache state may be inconsistent"
            );
        }
    }
}

pub(crate) fn load_context_moe(
    cfg: &Config,
    env: &ModelConfig,
    dir: &Path,
) -> Result<LoadedContext, Error> {
    let model = Qwen35MoeAdapter::load(cfg, env, dir)?;
    let (tokenizer, chat_template, eos_ids) = load_common(env, dir)?;
    let processor = TextOnlyProcessor::new("qwen3_5_moe", tokenizer, chat_template);
    Ok((Box::new(model), Box::new(processor), eos_ids))
}
