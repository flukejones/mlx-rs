//! Crate-root entry points: [`load`] + [`generate`].
//!
//! All consumers go through these two functions. [`load`] reads
//! `config.json::model_type`, dispatches to the per-family adapter,
//! and returns a [`ModelContext`] that owns the model + its
//! processor + the EOS ids. [`generate`] runs the full
//! prepare → sample → step loop with optional per-token streaming.

use std::ops::ControlFlow;
use std::path::Path;

use mlx_rs::{ops::indexing::IndexOp, transforms::async_eval, Array};

use crate::error::Error;
use crate::language_model::{LanguageModel, UserInputProcessor};
use crate::lm_input::{LMInput, PrepareResult, Text};
use crate::sampler::{SamplerState, SamplingParams};
use crate::user_input::UserInput;

/// Sampling + stopping knobs handed to [`generate`].
#[derive(Debug, Clone)]
pub struct GenerateParams {
    /// Maximum new tokens to produce (excluding the prompt). The
    /// loop exits early when an EOS token is sampled.
    pub max_new_tokens: i32,

    /// Sampling parameters (temperature + optional top-p).
    pub sampling: SamplingParams,

    /// Stop tokens beyond the model-default EOS list. Empty by
    /// default; useful for stop-on-newline or stop-on-`</answer>`
    /// style early exits.
    pub extra_stop_ids: Vec<u32>,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            sampling: SamplingParams::default(),
            extra_stop_ids: Vec::new(),
        }
    }
}

/// Reason a [`generate`] call returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// An EOS token (or a caller-supplied `extra_stop_id`) was
    /// sampled.
    Stop,
    /// `max_new_tokens` was reached without an EOS hit, or the
    /// streaming callback returned `ControlFlow::Break`.
    Length,
}

/// Output of one [`generate`] call.
#[derive(Debug, Clone)]
pub struct GenerateResult {
    /// Decoded UTF-8 text of all generated tokens.
    pub text: String,
    /// Number of tokens in the rendered prompt.
    pub prompt_tokens: i32,
    /// Number of tokens generated (excluding the prompt).
    pub completion_tokens: i32,
    /// Why the loop exited.
    pub finish_reason: FinishReason,
}

/// Per-token streaming callback. Each accepted token id is decoded
/// against the running prefix to produce a UTF-8 delta (BPE merges
/// only stabilise on a full re-decode, so the delta is computed as
/// `decode(accumulated) - decoded_so_far`).
pub type TokenCallback<'cb> = dyn FnMut(u32, &str) -> ControlFlow<()> + 'cb;

/// The loaded model + its preprocessor + the stop-token list. Built
/// by [`load`]; consumed by [`generate`].
pub struct ModelContext {
    /// The boxed language model. Owns its KV cache; [`generate`]
    /// calls `reset()` at the start of each request.
    pub model: Box<dyn LanguageModel>,

    /// The boxed input processor (tokenisation, chat template,
    /// modality handling).
    pub processor: Box<dyn UserInputProcessor>,

    /// EOS token ids the model considers terminal. Loaded from
    /// `config.json::eos_token_id` plus any family default.
    pub eos_ids: Vec<u32>,
}

impl ModelContext {
    /// Release the loaded model + processor and unmap mlx-core's
    /// buffer cache.
    ///
    /// Dropping a `ModelContext` (or letting it fall out of scope)
    /// already releases every parameter `Array` via
    /// `mlx_array_free`, returning the backing Metal buffers to
    /// mlx-core's free-list pool. The pool is **kept alive** so
    /// the next allocation can reuse buffers without paying
    /// another Metal driver round-trip — correct for long-running
    /// consumers (REPL, server) that re-use one model for the
    /// session.
    ///
    /// Consumers that load + drop multiple distinct models
    /// (multi-model dispatcher, bench harness, hot-swap server)
    /// want the free-list pool to actually unmap between models
    /// so peak resident memory is `max(models)`, not
    /// `sum(models)`. Call this in that case.
    ///
    /// `self`-by-value statically prevents reusing the context
    /// after unload.
    pub fn unload(self) {
        // Explicit drop is the same as letting `self` fall out
        // of scope; named here so the ordering vs `clear_cache`
        // is obvious — the parameter Arrays' refcounts must hit
        // zero before `clear_cache` can release the buffers
        // they were pinning.
        drop(self);
        mlx_rs::memory::clear_cache();
    }
}

/// Detect the model family from `<dir>/config.json::model_type` and
/// build the matching [`ModelContext`].
pub fn load(dir: impl AsRef<Path>) -> Result<ModelContext, Error> {
    let dir = dir.as_ref();
    let model_type = detect_model_type(dir)?;
    let (model, processor, eos_ids) = dispatch_load(model_type.as_str(), dir)?;
    Ok(ModelContext {
        model,
        processor,
        eos_ids,
    })
}

/// Per-token streaming decoder. Sliding window of the last
/// `WINDOW` tokens: decode the window each push, diff vs the
/// prior decoded window to extract the new bytes, drain leading
/// tokens into `committed` once the window overflows. Bounded
/// work per token instead of the naive O(N²) full re-decode.
struct IncrementalDecoder {
    ids: Vec<u32>,
    committed_tokens: usize,
    committed: String,
    window: String,
}

impl IncrementalDecoder {
    /// Must be ≥ the longest BPE merge that can reach back into
    /// earlier tokens. 4-5 in practice for Qwen / Llama / Gemma 4.
    const WINDOW: usize = 8;

    fn with_capacity(cap: usize) -> Self {
        Self {
            ids: Vec::with_capacity(cap),
            committed_tokens: 0,
            committed: String::new(),
            window: String::new(),
        }
    }

    /// Push a token, return the new UTF-8 delta to stream.
    fn push(&mut self, token: u32, processor: &dyn UserInputProcessor) -> Result<String, Error> {
        self.ids.push(token);

        let new_window = processor.decode(&self.ids[self.committed_tokens..])?;
        // BPE-merge fallback: if older bytes shifted, emit the
        // whole window as a corrective re-render.
        let delta: String = if new_window.starts_with(self.window.as_str()) {
            new_window[self.window.len()..].to_owned()
        } else {
            new_window.clone()
        };
        self.window = new_window;

        if self.ids.len() - self.committed_tokens > Self::WINDOW {
            // Lead's byte contribution = decode(window) -
            // decode(window without lead). Defer if it lands
            // mid-codepoint (sub-glyph BPE token).
            let lead_idx = self.committed_tokens;
            let after_lead = processor.decode(&self.ids[lead_idx + 1..])?;
            let mut lead_byte_len = self.window.len().saturating_sub(after_lead.len());
            while lead_byte_len > 0 && !self.window.is_char_boundary(lead_byte_len) {
                lead_byte_len -= 1;
            }

            if lead_byte_len > 0 {
                let moved = self.window.drain(..lead_byte_len).collect::<String>();
                self.committed.push_str(&moved);
                self.committed_tokens += 1;
            }
        }

        Ok(delta)
    }

    fn into_text(mut self) -> String {
        self.committed.push_str(&self.window);
        self.committed
    }

    fn len(&self) -> usize {
        self.ids.len()
    }
}

/// Run one prompt → tokens loop on `ctx`. Streaming is per-token via
/// `on_token`; pass `&mut |_, _| ControlFlow::Continue(())` to
/// disable streaming.
pub fn generate(
    ctx: &mut ModelContext,
    input: UserInput,
    params: GenerateParams,
    on_token: &mut TokenCallback<'_>,
) -> Result<GenerateResult, Error> {
    ctx.model.reset();

    let lm_input = ctx.processor.prepare(input)?;
    let prompt_tokens = lm_input.text.tokens.shape()[1];

    let initial_logits = run_prefill(ctx.model.as_mut(), lm_input)?;

    let vocab = ctx.model.vocab_size();
    let cap = params.max_new_tokens.max(0) as usize;
    let mut decoder = IncrementalDecoder::with_capacity(cap);
    let mut finish_reason = FinishReason::Length;

    if params.max_new_tokens == 0 {
        return Ok(GenerateResult {
            text: decoder.into_text(),
            prompt_tokens,
            completion_tokens: 0,
            finish_reason: FinishReason::Length,
        });
    }

    let mut sampler = SamplerState::new(params.sampling.clone());
    let mut pending_id = sampler.sample(&initial_logits)?;
    async_eval([&pending_id])?;

    for _ in 0..params.max_new_tokens {
        // Submit N+1 before syncing on N — overlap host coherence
        // sync with N+1 GPU compute.
        let next_logits = ctx.model.step(&pending_id)?.logits;
        let next_pending = sampler.sample(&next_logits)?;
        async_eval([&next_pending])?;

        let id_i32 = pending_id.item::<i32>();
        if id_i32 < 0 || id_i32 >= vocab {
            return Err(Error::Shape(format!(
                "sampler returned out-of-vocab id {id_i32} (vocab = {vocab})"
            )));
        }
        let token = id_i32 as u32;
        pending_id = next_pending;

        if ctx.eos_ids.contains(&token) || params.extra_stop_ids.contains(&token) {
            finish_reason = FinishReason::Stop;
            break;
        }

        let delta = decoder.push(token, ctx.processor.as_ref())?;
        if matches!(on_token(token, &delta), ControlFlow::Break(())) {
            break;
        }
    }

    let completion_tokens = decoder.len() as i32;
    Ok(GenerateResult {
        text: decoder.into_text(),
        prompt_tokens,
        completion_tokens,
        finish_reason,
    })
}

/// Run the prefill phase. When the model exposes a non-`None`
/// [`LanguageModel::prefill_chunk_size`] and the prompt exceeds it
/// (gemma4's sliding cache), feed all but the trailing chunk
/// through [`LanguageModel::prefill_chunk`], then drive the tail
/// through the normal [`LanguageModel::prepare`] to get the
/// first-step logits. Multimodal inputs (`image`/`audio`/`video`
/// set) bypass chunking — they go straight to `prepare`, which the
/// VLM adapter handles with a single stitched forward pass.
fn run_prefill(model: &mut dyn LanguageModel, mut input: LMInput) -> Result<Array, Error> {
    let chunk_size = model.prefill_chunk_size();
    let is_multimodal = input.image.is_some() || input.audio.is_some() || input.video.is_some();
    let prompt_len = input.text.tokens.shape()[1];

    if let Some(window) = chunk_size {
        if !is_multimodal && prompt_len > window {
            // Feed every chunk except the last through prefill_chunk;
            // the tail (≤ window tokens) goes through prepare so its
            // logits seed the sampler.
            let tokens = input.text.tokens;
            let mut start = 0_i32;
            while prompt_len - start > window {
                let end = start + window;
                let chunk = tokens.index((.., start..end));
                model.prefill_chunk(&chunk)?;
                start = end;
            }
            let tail = tokens.index((.., start..prompt_len));
            input = LMInput {
                text: Text {
                    tokens: tail,
                    mask: None,
                },
                image: None,
                audio: None,
                video: None,
            };
        }
    }

    match model.prepare(input)? {
        PrepareResult::Logits(arr) => Ok(arr),
        PrepareResult::Primed => {
            // No prefill logits returned: drive one step against the
            // sentinel id 0 to get the first usable distribution.
            // Allocates one `[1]` int32 array; this branch only
            // fires on models that don't return prefill logits, so
            // it's a one-shot.
            let seed = Array::from_slice::<i32>(&[0], &[1]);
            Ok(model.step(&seed)?.logits)
        }
    }
}

/// Read the top-level `model_type` field from `<dir>/config.json`.
fn detect_model_type(dir: &Path) -> Result<String, Error> {
    let path = dir.join("config.json");
    let raw = std::fs::read_to_string(&path)?;
    let val: serde_json::Value = serde_json::from_str(&raw)?;
    val.get("model_type")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or_else(|| {
            Error::Other(format!("{}: missing `model_type` field", path.display()).into())
        })
}

/// Match `model_type` against the supported families and route to
/// the matching adapter's `load_context`. Single source of truth for
/// the family-to-adapter mapping.
fn dispatch_load(model_type: &str, dir: &Path) -> Result<crate::adapters::LoadedContext, Error> {
    match model_type {
        // Gemma 4 family (text + MoE, no vision tower in this crate).
        "gemma4" | "gemma4_text" | "gemma4textmodel" | "gemma4forcausallm" => {
            crate::adapters::gemma4::load_context(dir)
        }

        // Qwen3.5 / Qwen3.6 dense + VL (incl. chandra-ocr-2).
        // The VLM probe inside `qwen3_5_vlm::load_context` looks
        // at `preprocessor_config.json` to decide dense vs VLM.
        "qwen3_5" | "qwen3_5_text" | "qwen3_5forconditionalgeneration" => {
            crate::adapters::qwen3_5_vlm::load_context(dir)
        }

        // Qwen3.5-MoE / Qwen3.6-MoE (35B-A3B and friends).
        "qwen3_5_moe" | "qwen3_5_moe_text" => crate::adapters::qwen3_5_moe::load_context(dir),

        other => Err(Error::Other(
            format!("mlx_lm::load: unsupported model_type {other:?}").into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]

    use super::*;
    use crate::lm_input::Text;

    /// Test processor that decodes tokens via a fixed lookup table.
    /// Tokens map to byte slices; `decode` concatenates them. This
    /// lets the test simulate BPE-style merges (a token that
    /// decodes differently in the presence of a neighbour) by
    /// modelling each `decode` of a slice as plain concatenation
    /// of the per-token byte slices — the simplest model that
    /// still exposes O(N²) vs O(N) accounting.
    struct FakeProcessor {
        // Per-token decoded form.
        pieces: Vec<&'static str>,
    }

    impl UserInputProcessor for FakeProcessor {
        fn family(&self) -> &'static str {
            "fake"
        }
        fn prepare(&mut self, _input: UserInput) -> Result<LMInput, Error> {
            Ok(LMInput {
                text: Text {
                    tokens: Array::from_slice::<i32>(&[], &[1, 0]),
                    mask: None,
                },
                image: None,
                audio: None,
                video: None,
            })
        }
        fn decode(&self, ids: &[u32]) -> Result<String, Error> {
            let mut out = String::new();
            for &id in ids {
                let id = id as usize;
                if id < self.pieces.len() {
                    out.push_str(self.pieces[id]);
                }
            }
            Ok(out)
        }
    }

    fn assert_incremental_matches_naive(pieces: &[&'static str], ids: &[u32]) {
        let processor = FakeProcessor {
            pieces: pieces.to_vec(),
        };
        let naive_full = processor.decode(ids).unwrap();
        let mut dec = IncrementalDecoder::with_capacity(ids.len());
        let mut streamed = String::new();
        for &id in ids {
            let delta = dec.push(id, &processor).unwrap();
            streamed.push_str(&delta);
        }
        let final_text = dec.into_text();
        assert_eq!(
            streamed, final_text,
            "concat of streamed deltas should equal final into_text()"
        );
        assert_eq!(
            naive_full, final_text,
            "incremental decode should match naive decode byte-for-byte"
        );
    }

    #[test]
    fn incremental_matches_naive_ascii() {
        let pieces = vec!["hello", " ", "world", ".", " ", "foo", " ", "bar"];
        let ids: Vec<u32> = (0..pieces.len() as u32).collect();
        assert_incremental_matches_naive(&pieces, &ids);
    }

    #[test]
    fn incremental_matches_naive_long_run() {
        // 1024 tokens — exercises the window-advance path
        // repeatedly. Each id maps to a single-char piece so we
        // can verify byte count exactly.
        let pieces: Vec<&'static str> = "abcdefghijklmnopqrstuvwxyz0123456789"
            .split("")
            .filter(|s| !s.is_empty())
            .collect();
        let ids: Vec<u32> = (0..1024).map(|i| i % pieces.len() as u32).collect();
        let processor = FakeProcessor {
            pieces: pieces.clone(),
        };
        let mut dec = IncrementalDecoder::with_capacity(ids.len());
        let mut streamed = String::new();
        for &id in &ids {
            streamed.push_str(&dec.push(id, &processor).unwrap());
        }
        let final_text = dec.into_text();
        assert_eq!(streamed.len(), 1024);
        assert_eq!(streamed, final_text);
        assert_eq!(processor.decode(&ids).unwrap(), final_text);
    }

    #[test]
    fn incremental_matches_naive_multibyte() {
        // CJK + emoji tokens — multi-byte UTF-8. Window advance
        // must move byte counts (not char counts) and never slice
        // mid-byte.
        let pieces = vec!["你", "好", "世", "界", "🎉", " ", "🍕", "!"];
        let ids: Vec<u32> = (0..pieces.len() as u32).collect();
        assert_incremental_matches_naive(&pieces, &ids);
    }

    #[test]
    fn incremental_handles_empty_run() {
        let processor = FakeProcessor { pieces: vec![] };
        let dec = IncrementalDecoder::with_capacity(0);
        assert_eq!(dec.into_text(), "");
        let _ = processor;
    }

    /// Lead token is a sub-glyph byte fragment that only forms a
    /// valid UTF-8 char when merged with the next token. `decode`
    /// returns lossy UTF-8 so isolated continuation bytes become
    /// U+FFFD. Regression: pre-fix this panicked at the
    /// `String::drain(..lead_byte_len)` mid-codepoint split.
    struct SubglyphProcessor {
        bytes: Vec<&'static [u8]>,
    }

    impl UserInputProcessor for SubglyphProcessor {
        fn family(&self) -> &'static str {
            "subglyph"
        }
        fn prepare(&mut self, _input: UserInput) -> Result<LMInput, Error> {
            Ok(LMInput {
                text: Text {
                    tokens: Array::from_slice::<i32>(&[], &[1, 0]),
                    mask: None,
                },
                image: None,
                audio: None,
                video: None,
            })
        }
        fn decode(&self, ids: &[u32]) -> Result<String, Error> {
            let mut buf: Vec<u8> = Vec::new();
            for &id in ids {
                let id = id as usize;
                if id < self.bytes.len() {
                    buf.extend_from_slice(self.bytes[id]);
                }
            }
            Ok(String::from_utf8_lossy(&buf).into_owned())
        }
    }

    #[test]
    fn incremental_does_not_panic_on_subglyph_token() {
        // WINDOW=8 padding + a 3-byte ✓ split across the last two
        // tokens. Push #9 triggers the commit-lead path. With the
        // old `two.len() - next_alone.len()` heuristic, this
        // panicked on `String::drain` at a non-char-boundary.
        let pieces: Vec<&'static [u8]> = vec![
            b"a",
            b"a",
            b"a",
            b"a",
            b"a",
            b"a",
            b"a",
            b"a",
            &[0xE2, 0x9C], // first 2 bytes of ✓ (U+2713)
            &[0x93],       // last byte of ✓
        ];
        let processor = SubglyphProcessor { bytes: pieces };
        let ids: Vec<u32> = (0..10).collect();
        let mut dec = IncrementalDecoder::with_capacity(ids.len());
        for &id in &ids {
            dec.push(id, &processor).expect("push should not panic");
        }
        let final_text = dec.into_text();
        assert!(
            final_text.ends_with('\u{2713}'),
            "✓ glyph not preserved: {final_text:?}"
        );
    }

    #[test]
    fn incremental_window_advances_across_long_response() {
        // Verify committed_tokens actually advances past zero
        // when ids.len() exceeds WINDOW. Without this the
        // optimisation is dead.
        let pieces: Vec<&'static str> = vec!["a"; 100];
        let processor = FakeProcessor { pieces };
        let mut dec = IncrementalDecoder::with_capacity(100);
        for id in 0..100_u32 {
            dec.push(id, &processor).unwrap();
        }
        // After 100 pushes with WINDOW=8, committed_tokens
        // should be ~92 (100 - 8). Tolerate ±1 for the exact
        // boundary semantics.
        assert!(
            dec.committed_tokens >= 100 - IncrementalDecoder::WINDOW - 1,
            "committed_tokens did not advance: {} of 100",
            dec.committed_tokens,
        );
    }
}
