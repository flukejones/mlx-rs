//! Flat scalar packing of upstream `AttnParams`. mlx-rs `metal_kernel`
//! exposes named array inputs, not C structs — every field of mlx's
//! `mlx::steel::AttnParams` becomes its own scalar / 3-element `Array`.

use mlx_rs::Array;

/// Mirrors mlx-swift `AttnParams` field-for-field, split into Arrays.
/// Holds owned arrays so the caller can `.clone()` cheaply into the
/// kernel input list.
pub struct FlatAttnParams {
    b: Array,
    h: Array,
    d: Array,
    q_len: Array,
    k_len: Array,
    gqa_factor: Array,
    scale: Array,
    nq: Array,
    nk: Array,
    nq_aligned: Array,
    nk_aligned: Array,
    ql_rem: Array,
    kl_rem: Array,
    ql_off: Array,
    q_strides: Array,
    k_strides: Array,
    v_strides: Array,
    o_strides: Array,
}

impl FlatAttnParams {
    /// Build flat params from `[B, H_q, qL, D]` / `[B, H_kv, kL, D]`
    /// shapes. Block sizes (`bq`, `bk`) determine the `NQ` / `NK`
    /// tile counts and remainders.
    #[allow(clippy::too_many_arguments)]
    pub fn from_shapes(
        b: i32,
        h_q: i32,
        h_kv: i32,
        q_len: i32,
        k_len: i32,
        d: i32,
        scale: f32,
        bq: i32,
        bk: i32,
        ql_off: i32,
    ) -> Self {
        let gqa_factor = h_q / h_kv;
        let nq = (q_len + bq - 1) / bq;
        let nk = (k_len + bk - 1) / bk;
        let nq_aligned = q_len / bq;
        let nk_aligned = k_len / bk;
        let ql_rem = q_len - nq_aligned * bq;
        let kl_rem = k_len - nk_aligned * bk;
        // Steel's AttnParams stores 3 strides (B, H, L) per tensor —
        // the D axis is assumed contiguous (stride = 1), so its stride
        // doesn't appear in the struct. Compute these from the full 4-D
        // shape, then drop the trailing D entry.
        let q_strides = batch_head_seq_strides(b, h_q, q_len, d);
        let k_strides = batch_head_seq_strides(b, h_kv, k_len, d);
        let v_strides = batch_head_seq_strides(b, h_kv, k_len, d);
        let o_strides = batch_head_seq_strides(b, h_q, q_len, d);
        Self {
            b: Array::from_int(b),
            h: Array::from_int(h_q),
            d: Array::from_int(d),
            q_len: Array::from_int(q_len),
            k_len: Array::from_int(k_len),
            gqa_factor: Array::from_int(gqa_factor),
            scale: Array::from_f32(scale),
            nq: Array::from_int(nq),
            nk: Array::from_int(nk),
            nq_aligned: Array::from_int(nq_aligned),
            nk_aligned: Array::from_int(nk_aligned),
            ql_rem: Array::from_int(ql_rem),
            kl_rem: Array::from_int(kl_rem),
            ql_off: Array::from_int(ql_off),
            q_strides: Array::from_slice(&q_strides, &[3]),
            k_strides: Array::from_slice(&k_strides, &[3]),
            v_strides: Array::from_slice(&v_strides, &[3]),
            o_strides: Array::from_slice(&o_strides, &[3]),
        }
    }

    pub fn b_param(&self) -> Array { self.b.clone() }
    pub fn h_param(&self) -> Array { self.h.clone() }
    pub fn d_param(&self) -> Array { self.d.clone() }
    pub fn q_len_param(&self) -> Array { self.q_len.clone() }
    pub fn k_len_param(&self) -> Array { self.k_len.clone() }
    pub fn gqa_factor_param(&self) -> Array { self.gqa_factor.clone() }
    pub fn scale_param(&self) -> Array { self.scale.clone() }
    pub fn nq_param(&self) -> Array { self.nq.clone() }
    pub fn nk_param(&self) -> Array { self.nk.clone() }
    pub fn nq_aligned_param(&self) -> Array { self.nq_aligned.clone() }
    pub fn nk_aligned_param(&self) -> Array { self.nk_aligned.clone() }
    pub fn ql_rem_param(&self) -> Array { self.ql_rem.clone() }
    pub fn kl_rem_param(&self) -> Array { self.kl_rem.clone() }
    pub fn ql_off_param(&self) -> Array { self.ql_off.clone() }
    pub fn q_strides_arr(&self) -> Array { self.q_strides.clone() }
    pub fn k_strides_arr(&self) -> Array { self.k_strides.clone() }
    pub fn v_strides_arr(&self) -> Array { self.v_strides.clone() }
    pub fn o_strides_arr(&self) -> Array { self.o_strides.clone() }
}

/// Returns (batch_stride, head_stride, seq_stride) for a row-major
/// `[B, H, L, D]` array — the three strides upstream's `AttnParams`
/// expects (D axis is implicit-contiguous, stride = 1).
fn batch_head_seq_strides(_b: i32, h: i32, l: i32, d: i32) -> [i64; 3] {
    let d = d as i64;
    let l = l as i64;
    let h = h as i64;
    [h * l * d, l * d, d]
}
