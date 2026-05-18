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
    fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask},
    ops::{
        concatenate_axis,
        indexing::{Ellipsis, IndexOp, TryIndexMutOp},
        zeros_dtype,
    },
    Array,
};

use super::kernels::{cached_steel_attention_kernel, STEEL_SUPPORTED_HEAD_DIMS};
use super::trait_def::KeyValueCache;
use crate::steel_attention::{steel_attention_dispatch, SteelAttentionInputs};

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
    /// Route prefill (`n_q > 1`) through the steel-attention tiled
    /// kernel when shape/mask permit. Opt-in via [`Self::with_steel_prefill`].
    use_steel_prefill: bool,
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
            use_steel_prefill: false,
        }
    }

    /// Opt in to the steel-attention tiled prefill kernel. Active only
    /// when `n_q > 1`, `head_dim ∈ {128, 256}`, and the caller passes
    /// no explicit mask. Falls back to `fast::SDPA` otherwise.
    pub fn with_steel_prefill(mut self) -> Self {
        self.use_steel_prefill = true;
        self
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

    /// Snapshot the current logical window (keep prefix + rotating
    /// region in temporal order) as a pair of fresh arrays. Used by the
    /// prefill path so attention can see `old_window ++ new` without
    /// having to write the new tokens back through the ring first.
    fn snapshot_window(&self) -> Result<(Array, Array), Exception> {
        let buf_k = self.keys.as_ref().expect("snapshot: buffer exists");
        let buf_v = self.values.as_ref().expect("snapshot: buffer exists");
        let keep = self.keep;
        let window = self.window();
        let visible_rot = self.write_head.min(window);
        let rot_start = keep + self.write_head - visible_rot;
        let rot_end = keep + self.write_head;
        let keep_filled = self.offset.min(keep);
        let rot_k = buf_k.index((Ellipsis, rot_start..rot_end, ..));
        let rot_v = buf_v.index((Ellipsis, rot_start..rot_end, ..));
        if keep_filled == 0 {
            Ok((rot_k, rot_v))
        } else {
            let head_k = buf_k.index((Ellipsis, 0..keep_filled, ..));
            let head_v = buf_v.index((Ellipsis, 0..keep_filled, ..));
            Ok((
                concatenate_axis(&[head_k, rot_k], -2)?,
                concatenate_axis(&[head_v, rot_v], -2)?,
            ))
        }
    }

    /// Write one token (S=1 slice) into the ring buffer. Extracted from
    /// the decode loop so the prefill path can reuse the same eviction
    /// semantics without duplicating logic.
    fn write_one(&mut self, token_k: Array, token_v: Array) -> Result<(), Exception> {
        let keep = self.keep;
        let window = self.window();
        if self.offset < keep {
            let slot = self.offset;
            let buf_k = self.keys.as_mut().expect("write_one: buffer exists");
            let buf_v = self.values.as_mut().expect("write_one: buffer exists");
            buf_k.try_index_mut((Ellipsis, slot..slot + 1, ..), token_k)?;
            buf_v.try_index_mut((Ellipsis, slot..slot + 1, ..), token_v)?;
        } else {
            if self.write_head >= 2 * window {
                self.compact()?;
            }
            let slot = keep + self.write_head;
            let buf_k = self.keys.as_mut().expect("write_one: buffer exists");
            let buf_v = self.values.as_mut().expect("write_one: buffer exists");
            buf_k.try_index_mut((Ellipsis, slot..slot + 1, ..), token_k)?;
            buf_v.try_index_mut((Ellipsis, slot..slot + 1, ..), token_v)?;
            self.write_head += 1;
        }
        self.offset += 1;
        Ok(())
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

        // Prefill (S > 1) after the cache already holds context: return
        // `pre_window ++ new` so every new token can attend to the full
        // sliding window of past context. Total cols =
        // `min(prev_offset, max_size) + S`, matching the mask built by
        // `create_attention_mask` (which clamps `offset` to `max_size`).
        if s > 1 && self.offset > 0 {
            let (old_k, old_v) = self.snapshot_window()?;
            for i in 0..s {
                let token_k = keys.index((Ellipsis, i..i + 1, ..));
                let token_v = values.index((Ellipsis, i..i + 1, ..));
                self.write_one(token_k, token_v)?;
            }
            return Ok((
                concatenate_axis(&[old_k, keys], -2)?,
                concatenate_axis(&[old_v, values], -2)?,
            ));
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

    fn attention(
        &mut self,
        queries: &Array,
        keys: Array,
        values: Array,
        scale: f32,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        // Pre-update offset = steel `ql_off` (causal diagonal shift).
        let ql_off = self.offset;
        let (k_full, v_full) = self.update_and_fetch(keys, values)?;

        let q_shape = queries.shape();
        let n_q = q_shape[q_shape.len() - 2];
        let head_dim = q_shape[q_shape.len() - 1];
        let h_q = q_shape[1];
        let h_kv = k_full.shape()[1];

        let steel_ok = self.use_steel_prefill
            && n_q > 1
            && STEEL_SUPPORTED_HEAD_DIMS.contains(&head_dim)
            && h_q % h_kv == 0;

        if steel_ok {
            return steel_attention_dispatch(
                cached_steel_attention_kernel(),
                SteelAttentionInputs {
                    q: queries,
                    k: &k_full,
                    v: &v_full,
                    mask: None,
                    causal: true,
                    ql_off,
                    scale,
                    head_dim,
                    h_q,
                    h_kv,
                },
            );
        }

        scaled_dot_product_attention(
            queries.clone(),
            k_full,
            v_full,
            scale,
            mask.map(ScaledDotProductAttentionMask::Array),
            None,
        )
    }
}
