use mlx_rs::{
    argmax_axis, array, categorical,
    error::Exception,
    ops::{
        argsort_axis, cumsum,
        indexing::{take_along_axis, Ellipsis, IndexOp, NewAxis},
        multiply, r#where, softmax_axis,
    },
    Array,
};

/// Argmax at `temp == 0.0`, categorical sampling otherwise.
pub fn sample(logits: &Array, temp: f32) -> Result<Array, Exception> {
    match temp {
        0.0 => argmax_axis!(logits, -1),
        _ => {
            let logits = logits.multiply(array!(1.0 / temp))?;
            categorical!(logits)
        }
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
}
