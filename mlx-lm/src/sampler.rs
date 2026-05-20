use mlx_rs::{
    argmax_axis, array, categorical,
    error::Exception,
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

/// Sampling knobs. `temperature == 0.0` → argmax.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    /// Nucleus (top-p). Ignored when `temperature == 0`.
    pub top_p: Option<f32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: None,
        }
    }
}

/// Sample with full parameter support: argmax at temp=0, plain
/// categorical at temp>0, or nucleus (top-p) when `top_p` is set.
///
/// One-shot convenience: each call re-allocates the temperature and
/// top-p scalar arrays. Use [`SamplerState`] in a hot decode loop.
pub fn sample_with(logits: &Array, params: &SamplingParams) -> Result<Array, Exception> {
    if params.temperature == 0.0 {
        return argmax_axis!(logits, -1);
    }
    let scaled = multiply(logits, array!(1.0_f32 / params.temperature))?;
    match params.top_p {
        None => categorical!(scaled),
        Some(p) => top_p_sample(&scaled, p),
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
    params: SamplingParams,
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
    pub fn new(params: SamplingParams) -> Self {
        let top_p_threshold = params.top_p.map(Array::from_f32);
        Self {
            params,
            inv_temp: None,
            top_p_threshold,
            neg_inf: None,
            bound_dtype: None,
        }
    }

    /// Sample one token from the given logits, reusing cached scalar
    /// arrays. Argmax at `temperature == 0.0`.
    pub fn sample(&mut self, logits: &Array) -> Result<Array, Exception> {
        if self.params.temperature == 0.0 {
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
        match self.params.top_p {
            None => categorical!(&scaled),
            Some(_) => self.top_p_sample(&scaled),
        }
    }

    fn bind(&mut self, dtype: Dtype) -> Result<(), Exception> {
        if self.bound_dtype == Some(dtype) {
            return Ok(());
        }
        let inv_temp = Array::from_f32(1.0_f32 / self.params.temperature).as_dtype(dtype)?;
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
        let params = SamplingParams {
            temperature: 0.0,
            top_p: None,
        };
        let mut state = SamplerState::new(params.clone());
        let one = state.sample(&logits).unwrap();
        let two = sample_with(&logits, &params).unwrap();
        assert_eq!(one.item::<u32>(), two.item::<u32>());
    }

    #[test]
    fn sampler_state_top_p_returns_top_token() {
        let logits = Array::from_slice(&[-10.0_f32, -10.0, -10.0, 5.0, -10.0], &[1, 5]);
        let params = SamplingParams {
            temperature: 1.0,
            top_p: Some(0.5),
        };
        let mut state = SamplerState::new(params);
        for _ in 0..16 {
            let tok = state.sample(&logits).unwrap();
            assert_eq!(tok.item::<u32>(), 3);
        }
    }

    #[test]
    fn sampler_state_caches_across_calls() {
        // Single bind + repeated sample must produce valid token ids.
        let logits = Array::from_slice(&[0.1_f32, 0.9, 0.2], &[1, 3]);
        let params = SamplingParams {
            temperature: 0.7,
            top_p: Some(0.95),
        };
        let mut state = SamplerState::new(params);
        for _ in 0..32 {
            let tok = state.sample(&logits).unwrap();
            let id = tok.item::<u32>();
            assert!(id < 3);
        }
    }
}
