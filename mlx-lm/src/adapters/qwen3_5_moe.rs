//! Qwen3.5-MoE (35B-A3B) [`crate::LanguageModel`] adapter.
//!
//! Same prefill / decode shape as the dense qwen3.5 adapter; the
//! only difference is the inner FFN type (`Qwen35MoeBlock`). No
//! multimodal path — MoE checkpoints are text-only.

use std::path::Path;

use mlx_rs::ops::concatenate_axis;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{argmax_axis, Array};

use crate::adapters::LoadedContext;
use crate::chat_template::ChatTemplate;
use crate::error::Error;
use crate::language_model::{LanguageModel, TextOnlyProcessor};
use crate::lm_input::{LMInput, LMOutput, PrepareResult};
use crate::loader::load_tokenizer;
use crate::models::qwen3_5::cache::{make_caches, make_mtp_caches, LayerCache};
use crate::models::qwen3_5::config::ModelConfig;
use crate::models::qwen3_5::layer::LanguageModel as Qwen35LanguageModel;
use crate::models::qwen3_5_moe::{load_qwen3_5_moe_model, Qwen35MoeBlock};

pub(crate) struct Qwen35MoeAdapter {
    model: Qwen35LanguageModel<Qwen35MoeBlock>,
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

    fn try_mtp_decode_greedy(
        &mut self,
        last_token: &Array,
    ) -> Result<Option<(Vec<u32>, Array)>, Error> {
        if self.model.mtp.is_none() {
            return Ok(None);
        }
        mtp_step(self, last_token).map(Some)
    }
}

/// One speculative step.
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
fn mtp_step(
    adapter: &mut Qwen35MoeAdapter,
    last_token: &Array,
) -> Result<(Vec<u32>, Array), Error> {
    let prev_hidden = adapter
        .prev_hidden
        .clone()
        .ok_or_else(|| Error::Other("mtp_step: prev_hidden unset; call prepare first".into()))?;

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

pub(crate) fn load_context(dir: &Path) -> Result<LoadedContext, Error> {
    let model = Qwen35MoeAdapter::load(dir)?;
    let cfg = ModelConfig::from_file(dir.join("config.json"))?;
    let tokenizer = load_tokenizer(dir)?;
    let chat_template = ChatTemplate::from_dir(dir)?;
    let eos_ids = crate::models::qwen3_5::read_qwen3_5_eos_ids(dir, &cfg);
    let processor = TextOnlyProcessor::new("qwen3_5_moe", tokenizer, chat_template);
    Ok((Box::new(model), Box::new(processor), eos_ids))
}
