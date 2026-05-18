//! [`RotatingKVCache`]: sliding-window KV cache. Backs Gemma 3/4
//! sliding layers.
//!
//! 2× ring buffer over the rotating region: steady-state decode is one
//! `try_index_mut` write + one contiguous `index` view return; the
//! previous implementation paid per-step `concatenate_axis` of 3 slices
//! × 2 (K and V) on every wrapped decode step.
//!
//! Layout: `[B, H, keep + 2 * window, D]` where
//! `window = max_size - keep`.
//!
//! - Slots `[0, keep)` are write-once head tokens (the `keep` prefix).
//! - Slots `[keep, keep + 2 * window)` are the rotating ring. Logical
//!   position `t` (post-keep) is at physical slot
//!   `keep + (t % (2 * window))`.
//! - When `write_head` (post-keep token count) reaches `2 * window`,
//!   compact: copy the last `window` rotating slots to the first
//!   `window` rotating slots, reset `write_head = window`. O(window)
//!   per compaction, amortised O(1) per token.
//!
//! At read time, the logical window is the last `min(write_head, window)`
//! rotating tokens. With the 2× layout, this is always a single
//! contiguous slice — no concat needed.

use std::collections::HashMap;

use mlx_rs::{
    error::Exception,
    ops::{
        concatenate_axis,
        indexing::{Ellipsis, IndexOp, TryIndexMutOp},
        zeros_dtype,
    },
    Array,
};

use super::trait_def::KeyValueCache;

/// Sliding-window KV cache.
///
/// Mirrors Python `mlx_lm.models.cache.RotatingKVCache` semantics
/// (oldest non-keep slot overwritten on append once the window is
/// full) but uses a 2× ring buffer internally so decode-step cost is
/// O(1) Array ops instead of O(1) concat.
#[derive(Debug, Clone)]
pub struct RotatingKVCache {
    keys: Option<Array>,
    values: Option<Array>,
    /// Real token count seen so far (monotonic; not bounded by max_size).
    offset: i32,
    /// Token count of writes into the rotating region (i.e. tokens past
    /// the `keep` prefix). Resets to `window` on each compaction.
    /// `0 <= write_head < 2 * window` between compactions.
    write_head: i32,
    /// Sliding-window capacity in tokens (`keep + window` ≤ `max_size`).
    max_size: i32,
    /// Number of head tokens to pin in the first `keep` slots.
    keep: i32,
}

impl RotatingKVCache {
    /// New empty cache. `max_size` is the sliding-window capacity in
    /// tokens; `keep` is the number of leading tokens that are never
    /// overwritten (must satisfy `0 <= keep < max_size`).
    pub fn new(max_size: i32, keep: i32) -> Self {
        assert!(max_size > 0, "max_size must be positive");
        assert!(
            keep >= 0 && keep < max_size,
            "keep must be in [0, max_size)"
        );
        Self {
            keys: None,
            values: None,
            offset: 0,
            write_head: 0,
            max_size,
            keep,
        }
    }

    #[inline]
    fn window(&self) -> i32 {
        self.max_size - self.keep
    }

    /// Physical capacity: keep prefix + 2× rotating window.
    #[inline]
    fn physical_capacity(&self) -> i32 {
        self.keep + 2 * self.window()
    }

    fn alloc_like(template: &Array, capacity: i32) -> Result<Array, Exception> {
        let shape = template.shape();
        let mut buf_shape = shape.to_vec();
        let t_axis = buf_shape.len() - 2;
        buf_shape[t_axis] = capacity;
        zeros_dtype(&buf_shape, template.dtype())
    }

    /// Compact: copy the last `window` rotating slots to the first
    /// `window` rotating slots, then reset `write_head = window`.
    /// Called when `write_head` would otherwise exceed `2 * window`.
    fn compact(&mut self) -> Result<(), Exception> {
        let window = self.window();
        let keep = self.keep;
        let buf_k = self.keys.as_mut().expect("compact: buffer exists");
        let buf_v = self.values.as_mut().expect("compact: buffer exists");
        // Source: [keep + window, keep + 2*window). Dest: [keep, keep + window).
        let src_start = keep + window;
        let src_end = keep + 2 * window;
        let dst_start = keep;
        let dst_end = keep + window;
        let src_k = buf_k.index((Ellipsis, src_start..src_end, ..));
        let src_v = buf_v.index((Ellipsis, src_start..src_end, ..));
        buf_k.try_index_mut((Ellipsis, dst_start..dst_end, ..), src_k)?;
        buf_v.try_index_mut((Ellipsis, dst_start..dst_end, ..), src_v)?;
        self.write_head = window;
        Ok(())
    }
}

impl KeyValueCache for RotatingKVCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        Some(self.max_size)
    }

    fn is_trimmable(&self) -> bool {
        // Trim is only well-defined while the rotating region hasn't filled.
        self.offset <= self.max_size
    }

    fn trim(&mut self, n: i32) -> i32 {
        if !self.is_trimmable() {
            return 0;
        }
        let trimmed = n.min(self.offset).max(0);
        self.offset -= trimmed;
        self.write_head = (self.offset - self.keep).max(0);
        trimmed
    }

    fn class_name(&self) -> &'static str {
        "RotatingKVCache"
    }

    fn state(&self) -> Vec<Array> {
        match (self.keys.as_ref(), self.values.as_ref()) {
            (Some(k), Some(v)) => vec![k.clone(), v.clone()],
            _ => Vec::new(),
        }
    }

    fn meta_state(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("offset".into(), self.offset.to_string());
        m.insert("write_head".into(), self.write_head.to_string());
        m.insert("max_size".into(), self.max_size.to_string());
        m.insert("keep".into(), self.keep.to_string());
        m
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        let key_shape = keys.shape();
        let t_axis = key_shape.len() - 2;
        let s = key_shape[t_axis];
        let keep = self.keep;
        let window = self.window();

        // Allocate the 2× ring buffer on first append.
        if self.keys.is_none() {
            let cap = self.physical_capacity();
            self.keys = Some(Self::alloc_like(&keys, cap)?);
            self.values = Some(Self::alloc_like(&values, cap)?);
        }

        // Per-token write loop. The keep prefix fills first, then the
        // rotating region. Compaction triggers when write_head reaches
        // 2 * window. S=1 decode bypasses any loop overhead for the
        // common case (single iteration, hot branch only).
        for i in 0..s {
            let token_k = keys.index((Ellipsis, i..i + 1, ..));
            let token_v = values.index((Ellipsis, i..i + 1, ..));

            if self.offset < keep {
                // Filling the keep prefix.
                let slot = self.offset;
                let buf_k = self.keys.as_mut().expect("alloc'd above");
                let buf_v = self.values.as_mut().expect("alloc'd above");
                buf_k.try_index_mut((Ellipsis, slot..slot + 1, ..), token_k)?;
                buf_v.try_index_mut((Ellipsis, slot..slot + 1, ..), token_v)?;
            } else {
                // Rotating region. Compact if at the 2× boundary so the
                // logical window stays contiguous.
                if self.write_head >= 2 * window {
                    self.compact()?;
                }
                let slot = keep + self.write_head;
                let buf_k = self.keys.as_mut().expect("alloc'd above");
                let buf_v = self.values.as_mut().expect("alloc'd above");
                buf_k.try_index_mut((Ellipsis, slot..slot + 1, ..), token_k)?;
                buf_v.try_index_mut((Ellipsis, slot..slot + 1, ..), token_v)?;
                self.write_head += 1;
            }
            self.offset += 1;
        }

        // Return the populated buffer in temporal order. Pre-fill (no
        // wrap): single contiguous slice. Post-fill: keep prefix +
        // single contiguous rotating slice — concat only when keep > 0.
        let buf_k = self.keys.as_ref().expect("alloc'd above");
        let buf_v = self.values.as_ref().expect("alloc'd above");

        // Effective rotating tokens visible: min(write_head, window).
        let visible_rot = self.write_head.min(window);
        let rot_start = keep + self.write_head - visible_rot;
        let rot_end = keep + self.write_head;

        if keep == 0 {
            // Common case (all current callers). No keep prefix, no concat.
            Ok((
                buf_k.index((Ellipsis, rot_start..rot_end, ..)),
                buf_v.index((Ellipsis, rot_start..rot_end, ..)),
            ))
        } else {
            // keep > 0: head prefix + rotating window. Two-slice concat
            // (vs the old 3-slice concat in the wrap case).
            let keep_filled = self.offset.min(keep);
            let head_k = buf_k.index((Ellipsis, 0..keep_filled, ..));
            let head_v = buf_v.index((Ellipsis, 0..keep_filled, ..));
            let rot_k = buf_k.index((Ellipsis, rot_start..rot_end, ..));
            let rot_v = buf_v.index((Ellipsis, rot_start..rot_end, ..));
            Ok((
                concatenate_axis(&[head_k, rot_k], -2)?,
                concatenate_axis(&[head_v, rot_v], -2)?,
            ))
        }
    }
}
