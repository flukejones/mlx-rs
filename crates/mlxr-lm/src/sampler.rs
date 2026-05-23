use mlxr::{
    argmax_axis, array, categorical,
    error::Exception,
    layers::log_softmax,
    ops::{
        argsort_axis, cumsum,
        indexing::{take_along_axis, Ellipsis, IndexOp, NewAxis},
        multiply, r#where, softmax_axis,
    },
    Array, Dtype,
};

/// Argmax at `temp == 0.0`, categorical sampling otherwise.
pub fn sample(logits: &Array, temp: f32) -> Result<Array, Exception> {
    if temp == 0.0 {
        argmax_axis!(logits, -1)
    } else {
        let logits = logits.multiply(array!(1.0 / temp))?;
        categorical!(logits)
    }
}

/// Sampling strategy. The variants are mutually exclusive; nucleus
/// (`top_p`) can never silently override greedy decode because
/// `Greedy` has no `p` field. Default is `Greedy` (argmax) for parity
/// with `temperature == 0.0` in prior shapes.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Sampler {
    /// Argmax. No temperature, no top-p.
    #[default]
    Greedy,
    /// Plain categorical sampling at the given temperature.
    /// `temperature` must be `> 0.0`; `0.0` is a logic bug, use [`Self::Greedy`].
    Temperature(f32),
    /// Categorical sampling with a nucleus (top-p) mask applied after
    /// temperature scaling. `temperature` must be `> 0.0`.
    TopP { temperature: f32, p: f32 },
}

impl Sampler {
    /// `None` for [`Self::Greedy`], else the temperature.
    pub fn temperature(self) -> Option<f32> {
        match self {
            Self::Greedy => None,
            Self::Temperature(t) | Self::TopP { temperature: t, .. } => Some(t),
        }
    }

    /// `Some(p)` for [`Self::TopP`]; `None` otherwise.
    pub fn top_p(self) -> Option<f32> {
        match self {
            Self::TopP { p, .. } => Some(p),
            _ => None,
        }
    }
}

/// Sample with full parameter support: argmax for [`Sampler::Greedy`],
/// plain categorical for [`Sampler::Temperature`], or nucleus mask for
/// [`Sampler::TopP`].
///
/// One-shot convenience: each call re-allocates the temperature and
/// top-p scalar arrays. Use [`SamplerState`] in a hot decode loop.
pub fn sample_with(logits: &Array, sampler: Sampler) -> Result<Array, Exception> {
    match sampler {
        Sampler::Greedy => argmax_axis!(logits, -1),
        Sampler::Temperature(t) => {
            let scaled = multiply(logits, array!(1.0_f32 / t))?;
            categorical!(scaled)
        }
        Sampler::TopP { temperature, p } => {
            let scaled = multiply(logits, array!(1.0_f32 / temperature))?;
            top_p_sample(&scaled, p)
        }
    }
}

/// Per-decode-loop sampler with cached scalar constants. Avoids the
/// per-token host→device allocations that [`sample_with`] performs
/// for inverse-temperature, top-p threshold, and the −∞ mask.
///
/// Constants are bound to the logits dtype on first sample and reused
/// for every subsequent call. Rebuild the state if the dtype changes
/// (it does not, across a single generation).
pub struct SamplerState {
    sampler: Sampler,
    /// `1.0 / temperature` materialised at logits dtype. `None` for
    /// greedy decode.
    inv_temp: Option<Array>,
    /// `top_p` threshold as f32 (compared against an f32 softmax,
    /// dtype is fixed). `None` when top-p is disabled.
    top_p_threshold: Option<Array>,
    /// `-inf` cast to logits dtype, used as the nucleus mask
    /// sentinel. `None` until first use.
    neg_inf: Option<Array>,
    /// Logits dtype `inv_temp` and `neg_inf` were built against.
    /// `None` until the first sample.
    bound_dtype: Option<Dtype>,
}

impl SamplerState {
    pub fn new(sampler: Sampler) -> Self {
        let top_p_threshold = sampler.top_p().map(Array::from_f32);
        Self {
            sampler,
            inv_temp: None,
            top_p_threshold,
            neg_inf: None,
            bound_dtype: None,
        }
    }

    /// Sample one token from the given logits, reusing cached scalar
    /// arrays.
    pub fn sample(&mut self, logits: &Array) -> Result<Array, Exception> {
        if matches!(self.sampler, Sampler::Greedy) {
            return argmax_axis!(logits, -1);
        }
        let dtype = logits.dtype();
        self.bind(dtype)?;
        // SAFETY: bind() populated inv_temp.
        let inv_temp = self
            .inv_temp
            .as_ref()
            .expect("inv_temp populated by bind()");
        let scaled = multiply(logits, inv_temp)?;
        match self.sampler {
            Sampler::Greedy => unreachable!("greedy handled above"),
            Sampler::Temperature(_) => categorical!(&scaled),
            Sampler::TopP { .. } => self.top_p_sample(&scaled),
        }
    }

    /// Read access to the sampler the state was built from. MTP
    /// rejection-sampling needs the temperature + top-p values to
    /// drive its own masked log-prob computation.
    pub fn sampler(&self) -> Sampler {
        self.sampler
    }

    /// Apply temperature scaling and (optionally) a shared top-p
    /// keep mask, then return `log_softmax` of the result. Caches
    /// the `inv_temp` scalar against `logits.dtype()` so repeated
    /// calls in one decode loop don't re-allocate. Caller is
    /// responsible for building the keep mask via
    /// `crate::qwen3_5::text::sampling::top_p_keep_mask` (or its
    /// union across draft + verify) for MTP-style callers.
    ///
    /// Errors at `temperature == 0.0`: `1/temp` would be `inf` and
    /// silently propagate NaN through `log_softmax`. Greedy callers
    /// must branch separately and never reach this path.
    pub fn masked_log_probs(
        &mut self,
        logits: &Array,
        keep_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        if matches!(self.sampler, Sampler::Greedy) {
            return Err(Exception::custom(
                "masked_log_probs: Sampler::Greedy has no temperature; greedy callers go through the argmax path",
            ));
        }
        let dtype = logits.dtype();
        self.bind(dtype)?;
        let inv_temp = self
            .inv_temp
            .as_ref()
            .expect("inv_temp populated by bind()");
        masked_temp_log_probs(logits, keep_mask, inv_temp)
    }

    fn bind(&mut self, dtype: Dtype) -> Result<(), Exception> {
        if self.bound_dtype == Some(dtype) {
            return Ok(());
        }
        let t = self
            .sampler
            .temperature()
            .expect("bind() called on Sampler::Greedy");
        let inv_temp = Array::from_f32(1.0_f32 / t).as_dtype(dtype)?;
        let neg_inf = Array::from_f32(f32::NEG_INFINITY).as_dtype(dtype)?;
        self.inv_temp = Some(inv_temp);
        self.neg_inf = Some(neg_inf);
        self.bound_dtype = Some(dtype);
        Ok(())
    }

    fn top_p_sample(&self, logits: &Array) -> Result<Array, Exception> {
        let p = self
            .top_p_threshold
            .as_ref()
            .expect("top_p_sample called without top_p set");
        let neg_inf = self.neg_inf.as_ref().expect("neg_inf populated by bind()");
        let probs = softmax_axis(logits, -1, true)?;
        let neg = probs.negative()?;
        let order = argsort_axis(&neg, -1)?;
        let sorted_probs = take_along_axis(&probs, &order, -1)?;
        let csum = cumsum(&sorted_probs, -1, false, false)?;
        let prev = csum.subtract(&sorted_probs)?;
        let keep = prev.lt(p)?;
        let sorted_logits = take_along_axis(logits, &order, -1)?;
        let masked = r#where(&keep, &sorted_logits, neg_inf)?;
        let sorted_pick = categorical!(&masked)?;
        let pick = sorted_pick.index((Ellipsis, NewAxis));
        let token = take_along_axis(&order, &pick, -1)?;
        token.squeeze_axes(&[-1])
    }
}

/// Nucleus sampling: keep the smallest descending-prob set whose
/// cumulative mass covers `p`, mask the rest with `-inf` in logit
/// space, sample. Returns original token ids.
pub fn top_p_sample(logits: &Array, p: f32) -> Result<Array, Exception> {
    let probs = softmax_axis(logits, -1, true)?;
    // argsort is ascending; negate to get descending order of probs.
    let neg = probs.negative()?;
    let order = argsort_axis(&neg, -1)?;
    let sorted_probs = take_along_axis(&probs, &order, -1)?;
    let csum = cumsum(&sorted_probs, -1, false, false)?;
    // Keep a sorted slot iff preceding cumulative mass is < p.
    let prev = csum.subtract(&sorted_probs)?;
    let keep = prev.lt(Array::from_f32(p))?;
    // Mask in logit space so categorical sees a well-formed distribution.
    let sorted_logits = take_along_axis(logits, &order, -1)?;
    let neg_inf = Array::from_f32(f32::NEG_INFINITY).as_dtype(sorted_logits.dtype())?;
    let masked = r#where(&keep, &sorted_logits, &neg_inf)?;
    let sorted_pick = categorical!(&masked)?;
    let pick = sorted_pick.index((Ellipsis, NewAxis));
    let token = take_along_axis(&order, &pick, -1)?;
    token.squeeze_axes(&[-1])
}

/// Log-probabilities over a single distribution after temperature
/// scaling and (optional) top-p masking. `[1, vocab]` in, same shape
/// out; ids masked out by top-p get `-inf`. Caller can `exp` the
/// result to recover the probability distribution, or sample from it
/// via `categorical!` on the same masked logits scaled by `1/temp`.
///
/// Used by the MTP rejection-sampling path: both the draft and the
/// verify side are passed through this helper with the *same*
/// `keep_mask` (the union of the two per-side top-p masks) so the
/// resulting log-probs are directly comparable in the accept ratio.
pub(crate) fn masked_temp_log_probs(
    logits: &Array,
    keep_mask: Option<&Array>,
    inv_temp: &Array,
) -> Result<Array, Exception> {
    let scaled = multiply(logits, inv_temp)?;
    let masked = if let Some(mask) = keep_mask {
        let neg_inf = Array::from_f32(f32::NEG_INFINITY).as_dtype(scaled.dtype())?;
        r#where(mask, &scaled, &neg_inf)?
    } else {
        scaled
    };
    log_softmax(&masked, -1)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    #![allow(clippy::print_stdout, reason = "test code")]
    #![allow(clippy::print_stderr, reason = "test code")]
    use super::*;

    #[test]
    fn top_p_returns_real_token_id() {
        // Vocab=5, logits put 99% mass on id=3. top_p=0.5 must keep
        // only id=3 → sampled token must be 3, not 0 (sorted-axis idx).
        let logits = Array::from_slice(&[-10.0_f32, -10.0, -10.0, 5.0, -10.0], &[1, 5]);
        for _ in 0..16 {
            let tok = top_p_sample(&logits, 0.5).unwrap();
            assert_eq!(tok.item::<u32>(), 3);
        }
    }

    #[test]
    fn top_p_keeps_only_top_token_when_p_small() {
        // Probs ~ [0.1, 0.6, 0.3] — top is id=1, p=0.01 keeps only it.
        let logits = Array::from_slice(&[0.0_f32, 1.8, 1.1], &[1, 3]);
        for _ in 0..16 {
            let tok = top_p_sample(&logits, 0.01).unwrap();
            assert_eq!(tok.item::<u32>(), 1);
        }
    }

    #[test]
    fn sampler_state_matches_sample_with_greedy() {
        let logits = Array::from_slice(&[0.1_f32, 0.5, 0.3, 0.05, 0.05], &[1, 5]);
        let mut state = SamplerState::new(Sampler::Greedy);
        let one = state.sample(&logits).unwrap();
        let two = sample_with(&logits, Sampler::Greedy).unwrap();
        assert_eq!(one.item::<u32>(), two.item::<u32>());
    }

    #[test]
    fn sampler_state_top_p_returns_top_token() {
        let logits = Array::from_slice(&[-10.0_f32, -10.0, -10.0, 5.0, -10.0], &[1, 5]);
        let mut state = SamplerState::new(Sampler::TopP {
            temperature: 1.0,
            p: 0.5,
        });
        for _ in 0..16 {
            let tok = state.sample(&logits).unwrap();
            assert_eq!(tok.item::<u32>(), 3);
        }
    }

    #[test]
    fn masked_temp_log_probs_matches_log_softmax_without_mask() {
        let logits = Array::from_slice(&[0.1_f32, 1.0, 0.3], &[1, 3]);
        let inv_temp = Array::from_f32(1.0_f32 / 0.7);
        let got = masked_temp_log_probs(&logits, None, &inv_temp).unwrap();
        let expected =
            log_softmax(logits.multiply(Array::from_f32(1.0_f32 / 0.7)).unwrap(), -1).unwrap();
        let g: &[f32] = got.as_slice();
        let e: &[f32] = expected.as_slice();
        for (a, b) in g.iter().zip(e.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn masked_temp_log_probs_zeroes_excluded_ids() {
        let logits = Array::from_slice(&[-10.0_f32, 5.0, -10.0], &[1, 3]);
        let inv_temp = Array::from_f32(1.0_f32);
        let mask = Array::from_slice(&[false, true, false], &[1, 3]);
        let lp = masked_temp_log_probs(&logits, Some(&mask), &inv_temp).unwrap();
        let v: &[f32] = lp.as_slice();
        assert!(v[0].is_infinite() && v[0] < 0.0);
        assert!((v[1] - 0.0).abs() < 1e-5);
        assert!(v[2].is_infinite() && v[2] < 0.0);
    }

    #[test]
    fn sampler_state_caches_across_calls() {
        // Single bind + repeated sample must produce valid token ids.
        let logits = Array::from_slice(&[0.1_f32, 0.9, 0.2], &[1, 3]);
        let mut state = SamplerState::new(Sampler::TopP {
            temperature: 0.7,
            p: 0.95,
        });
        for _ in 0..32 {
            let tok = state.sample(&logits).unwrap();
            let id = tok.item::<u32>();
            assert!(id < 3);
        }
    }
}
