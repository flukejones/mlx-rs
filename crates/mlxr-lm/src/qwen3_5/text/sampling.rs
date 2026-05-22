//! Qwen3.5-family-local sampling helpers.
//!
//! MTP rejection sampling builds a vocab-positional top-p keep mask
//! shared between the draft and verify distributions. The mask lives
//! here (rather than `crate::sampler`) because no other family
//! consumes it: keeping it co-located with the MoE adapter keeps the
//! family's sampling surface inside one feature gate.

use mlxr::{
    error::Exception,
    ops::{argsort_axis, cumsum, indexing::take_along_axis, softmax_axis},
    Array,
};

/// Build a vocab-positional keep mask (`[1, vocab]` bool) for top-p
/// at threshold `p` over the given logits. Slot `i` is `true` iff
/// token id `i` belongs to the smallest descending-probability set
/// whose preceding cumulative mass is below `p`. Same set
/// `crate::sampler::top_p_sample` keeps, indexed by vocab id rather
/// than by sort position. Used by the MTP rejection-sampling path
/// to apply a shared keep mask to both the draft and verify
/// distributions before computing the accept ratio.
pub(crate) fn top_p_keep_mask(logits: &Array, p: f32) -> Result<Array, Exception> {
    let probs = softmax_axis(logits, -1, true)?;
    let neg = probs.negative()?;
    let order = argsort_axis(&neg, -1)?;
    let sorted_probs = take_along_axis(&probs, &order, -1)?;
    let csum = cumsum(&sorted_probs, -1, false, false)?;
    let prev = csum.subtract(&sorted_probs)?;
    let keep_sorted = prev.lt(Array::from_f32(p))?;
    // argsort(order) is the inverse permutation: for each vocab id,
    // it gives the sort position it landed on. take_along_axis with
    // that pulls the keep flag for each vocab id back into vocab
    // order.
    let inverse = argsort_axis(&order, -1)?;
    take_along_axis(&keep_sorted, &inverse, -1)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    use super::*;

    #[test]
    fn top_p_keep_mask_keeps_only_top_token() {
        // Probs ~ [0.0, ~0.99, ~0.0]: only id=1 should be kept at p=0.5.
        let logits = Array::from_slice(&[-10.0_f32, 5.0, -10.0], &[1, 3]);
        let mask = top_p_keep_mask(&logits, 0.5).unwrap();
        let m: &[bool] = mask.as_slice();
        assert_eq!(m, &[false, true, false]);
    }

    #[test]
    fn top_p_keep_mask_keeps_full_distribution_at_p_one() {
        let logits = Array::from_slice(&[0.1_f32, 0.5, 0.3, 0.05, 0.05], &[1, 5]);
        let mask = top_p_keep_mask(&logits, 1.0).unwrap();
        let m: &[bool] = mask.as_slice();
        assert_eq!(m, &[true, true, true, true, true]);
    }
}
