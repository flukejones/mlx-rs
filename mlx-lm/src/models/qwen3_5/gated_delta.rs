//! Gated DeltaNet (Mamba2-style) recurrent operator used by every
//! `linear_attention` layer of Qwen3.5. This module ships the pure-ops
//! scan (`gated_delta_update_ops`) that mirrors `mlx_lm.models.gated_delta`.
//! A Metal kernel fast path lands separately.
//!
//! Shapes (matching the Python reference):
//! - `q`, `k`: `[B, T, Hk, Dk]`
//! - `v`: `[B, T, Hv, Dv]`
//! - `a`, `b`, `dt_bias`, `A_log`: `[B, T, Hv]` (the projector outputs) and
//!   `[Hv]` (the learned per-head params).
//! - returned `y`: `[B, T, Hv, Dv]`
//! - returned `state`: `[B, Hv, Dv, Dk]`

use mlx_rs::{
    error::Exception,
    nn,
    ops::{
        exp as exp_op, expand_dims, indexing::take_axis, r#where, repeat_axis, reshape, sigmoid,
        stack_axis, zeros, zeros_dtype,
    },
    Array, Dtype,
};

/// Compute the per-step decay `g = exp(-exp(A_log) * softplus(a + dt_bias))`.
///
/// Mirrors Python's `compute_g`. The 5-op chain is run inline here; a
/// `transforms::compile`-fused fast path lands separately.
pub fn compute_g(a_log: &Array, a: &Array, dt_bias: &Array) -> Result<Array, Exception> {
    let a_log_f32 = a_log.as_dtype(Dtype::Float32)?;
    let inner = a.add(dt_bias)?;
    let s = nn::softplus(&inner)?;
    exp_op(a_log_f32)?.negative()?.multiply(&s)?.exp()
}

/// Run one recurrent step of the gated delta SSM.
///
/// Inputs:
/// - `q_t`, `k_t`: `[B, Hv, Dk]` (with key heads already broadcast to Hv).
/// - `v_t`: `[B, Hv, Dv]`.
/// - `g_t`: `[B, Hv]` (scalar decay) or `[B, Hv, Dk]` (vectorised decay).
/// - `beta_t`: `[B, Hv]`.
/// - `state`: `[B, Hv, Dv, Dk]`.
/// - `mask_t`: optional `[B]` bool — when false at element `bi`, the new
///   state for that batch element is replaced with the previous state and `y`
///   is zeroed.
///
/// Returns `(y, new_state)` with `y` shaped `[B, Hv, Dv]`.
pub fn step_ops(
    q_t: &Array,
    k_t: &Array,
    v_t: &Array,
    g_t: &Array,
    beta_t: &Array,
    state: &Array,
    mask_t: Option<&Array>,
) -> Result<(Array, Array), Exception> {
    let old_state = state.clone();

    let decay = match g_t.ndim() {
        2 => {
            // [B, Hv] -> [B, Hv, 1, 1]
            let e = expand_dims(g_t, 2)?;
            expand_dims(&e, 3)?
        }
        3 => {
            // [B, Hv, Dk] -> [B, Hv, 1, Dk]
            expand_dims(g_t, 2)?
        }
        n => {
            return Err(Exception::custom(format!(
                "step_ops: unsupported g ndim {n}"
            )))
        }
    };
    let state = state.multiply(&decay)?;
    // k[..., None, :] -> [B, Hv, 1, Dk]
    let k_b = expand_dims(k_t, 2)?;
    let kv_mem = state.multiply(&k_b)?.sum_axes(&[-1], false)?; // [B, Hv, Dv]
                                                                // beta[..., None] -> [B, Hv, 1]
    let beta_b = expand_dims(beta_t, 2)?;
    let delta = v_t.subtract(&kv_mem)?.multiply(&beta_b)?; // [B, Hv, Dv]
                                                           // delta[..., None] -> [B, Hv, Dv, 1]
    let delta_b = expand_dims(&delta, 3)?;
    let state = state.add(&k_b.multiply(&delta_b)?)?;

    // y = (state * q[..., None, :]).sum(-1) -> [B, Hv, Dv]
    let q_b = expand_dims(q_t, 2)?;
    let y = state.multiply(&q_b)?.sum_axes(&[-1], false)?;

    let (state, y) = if let Some(mask_t) = mask_t {
        // mask_t: [B] -> for state [B, 1, 1, 1], for y [B, 1, 1]
        let m_state = expand_dims(mask_t, 1)?;
        let m_state = expand_dims(&m_state, 2)?;
        let m_state = expand_dims(&m_state, 3)?;
        let m_y = expand_dims(mask_t, 1)?;
        let m_y = expand_dims(&m_y, 2)?;
        let state_dtype = state.dtype();
        let new_state = r#where(&m_state, &state, &old_state)?;
        let zero = zeros_dtype(y.shape(), y.dtype())?;
        let _ = state_dtype;
        let new_y = r#where(&m_y, &y, &zero)?;
        (new_state, new_y)
    } else {
        (state, y)
    };
    let y = y.as_dtype(q_t.dtype())?;
    Ok((y, state))
}

/// Run the full sequential scan over `T` steps using the ops-only kernel.
///
/// Inputs match `gated_delta_ops` in the Python reference. `state` is
/// optional — `None` initialises a zero state of dtype float32. Returns
/// `(y, final_state)`.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_update_ops(
    q: &Array,
    k: &Array,
    v: &Array,
    a: &Array,
    b: &Array,
    a_log: &Array,
    dt_bias: &Array,
    state: Option<&Array>,
    mask: Option<&Array>,
) -> Result<(Array, Array), Exception> {
    let q_shape = q.shape();
    let v_shape = v.shape();
    if q_shape.len() != 4 || v_shape.len() != 4 {
        return Err(Exception::custom(
            "gated_delta_update_ops: q/v must be 4-D [B, T, H, D]",
        ));
    }
    let batch = q_shape[0];
    let time = q_shape[1];
    let hk = q_shape[2];
    let dk = q_shape[3];
    let hv = v_shape[2];
    let dv = v_shape[3];

    let beta = sigmoid(b)?;
    let g = compute_g(a_log, a, dt_bias)?;

    let owned_state;
    let state = match state {
        Some(s) => s.clone(),
        None => {
            owned_state = zeros::<f32>(&[batch, hv, dv, dk])?;
            owned_state
        }
    };

    if hv % hk != 0 {
        return Err(Exception::custom(format!(
            "gated_delta_update_ops: Hv ({hv}) must be divisible by Hk ({hk})"
        )));
    }
    let (q_eff, k_eff) = if hv == hk {
        (q.clone(), k.clone())
    } else {
        let rep = hv / hk;
        let q_r = repeat_axis::<f32>(q.clone(), rep, -2)?;
        let k_r = repeat_axis::<f32>(k.clone(), rep, -2)?;
        (q_r, k_r)
    };

    let mut state = state;
    let mut ys = Vec::with_capacity(time as usize);
    for t in 0..time {
        let q_t = slice_t(&q_eff, t)?;
        let k_t = slice_t(&k_eff, t)?;
        let v_t = slice_t(v, t)?;
        let g_t = slice_t(&g, t)?;
        let beta_t = slice_t(&beta, t)?;
        let mask_t = match mask {
            Some(m) => Some(slice_t(m, t)?),
            None => None,
        };
        let (y_t, new_state) = step_ops(&q_t, &k_t, &v_t, &g_t, &beta_t, &state, mask_t.as_ref())?;
        state = new_state;
        ys.push(y_t);
    }
    let y = stack_axis(&ys, 1)?;
    Ok((y, state))
}

/// Index the time axis at `t` and squeeze it out. `x[:, t]` in numpy / mlx.
fn slice_t(x: &Array, t: i32) -> Result<Array, Exception> {
    // Use a length-1 index so take_axis preserves the axis at size 1, then
    // squeeze that axis away. A 0-D scalar index would drop the axis directly
    // in some builds but not others, so the explicit shape keeps behaviour
    // consistent.
    let idx = Array::from_slice(&[t], &[1]);
    let y = take_axis(x, &idx, 1)?;
    let shape = y.shape();
    if shape[1] != 1 {
        return Err(Exception::custom(format!(
            "slice_t: expected axis 1 to be size 1 after take_axis, got shape {:?}",
            shape
        )));
    }
    let new_shape: Vec<i32> = shape
        .iter()
        .enumerate()
        .filter_map(|(i, s)| if i == 1 { None } else { Some(*s) })
        .collect();
    reshape(&y, &new_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::{random::uniform, transforms::eval};

    fn rand(shape: &[i32]) -> Array {
        uniform::<_, f32>(0.0, 1.0, shape, None).unwrap()
    }

    fn flatten_f32(arr: &Array) -> Vec<f32> {
        let total: i32 = arr.shape().iter().product();
        let flat = reshape(arr, &[total]).unwrap();
        let zero = Array::from_f32(0.0);
        let evald = flat.add(&zero).unwrap();
        eval([&evald]).unwrap();
        evald.as_slice::<f32>().to_vec()
    }

    #[test]
    fn compute_g_shape_and_range() {
        let hv = 4;
        let a_log = rand(&[hv]);
        let dt_bias = rand(&[hv]);
        let a = rand(&[2, 3, hv]);
        let g = compute_g(&a_log, &a, &dt_bias).unwrap();
        assert_eq!(g.shape(), &[2, 3, hv]);
        // g = exp(-exp(A_log_pos) * softplus(...)) <= 1.0 strictly.
        let max = g.max(None).unwrap().item::<f32>();
        assert!(max <= 1.0 + 1e-5, "g max {max} should be <= 1");
        let min = g.min(None).unwrap().item::<f32>();
        assert!(min >= 0.0, "g min {min} should be >= 0");
    }

    #[test]
    fn zero_state_input_and_random_inputs_have_expected_shapes() {
        let (b, t, hk, dk) = (1, 3, 2, 4);
        let (hv, dv) = (4, 4);
        let q = rand(&[b, t, hk, dk]);
        let k = rand(&[b, t, hk, dk]);
        let v = rand(&[b, t, hv, dv]);
        let a = rand(&[b, t, hv]);
        let bb = rand(&[b, t, hv]);
        let a_log = rand(&[hv]);
        let dt_bias = rand(&[hv]);

        let (y, state) =
            gated_delta_update_ops(&q, &k, &v, &a, &bb, &a_log, &dt_bias, None, None).unwrap();
        assert_eq!(y.shape(), &[b, t, hv, dv]);
        assert_eq!(state.shape(), &[b, hv, dv, dk]);
    }

    #[test]
    fn single_step_with_zero_state_and_zero_decay() {
        // When state starts at zero and `decay = g` is irrelevant for the
        // first step, the produced y at t=0 reduces to:
        //   y = (state' * q[None, :]).sum(-1)
        // with state' = k[..., None, :] * delta[..., None] and
        // delta = v * beta (since kv_mem = 0).
        let b = 1;
        let t = 1;
        let hk = 1;
        let dk = 2;
        let hv = 1;
        let dv = 2;
        let q = Array::from_slice(&[1.0_f32, 1.0], &[b, t, hk, dk]);
        let k = Array::from_slice(&[1.0_f32, 0.0], &[b, t, hk, dk]);
        let v = Array::from_slice(&[0.5_f32, 0.25], &[b, t, hv, dv]);
        let a = Array::from_slice(&[0.0_f32], &[b, t, hv]); // softplus(0) = ln2
        let bb = Array::from_slice(&[0.0_f32], &[b, t, hv]); // sigmoid(0) = 0.5
        let a_log = Array::from_slice(&[-30.0_f32], &[hv]); // exp(-30) ≈ 0 -> g ≈ 1
        let dt_bias = Array::from_slice(&[0.0_f32], &[hv]);

        let (y, _state) =
            gated_delta_update_ops(&q, &k, &v, &a, &bb, &a_log, &dt_bias, None, None).unwrap();
        let y_flat = flatten_f32(&y);
        // Manual derivation:
        //   g ≈ 1, beta = 0.5, state starts zero -> kv_mem = 0
        //   delta = (v - 0) * 0.5 = [0.25, 0.125]
        //   state += k[..., None, :] * delta[..., None]
        //     k = [1, 0], delta = [0.25, 0.125]
        //     state[Dv=0, Dk=0] = 1 * 0.25 = 0.25
        //     state[Dv=0, Dk=1] = 0 * 0.25 = 0
        //     state[Dv=1, Dk=0] = 1 * 0.125 = 0.125
        //     state[Dv=1, Dk=1] = 0 * 0.125 = 0
        //   y[Dv=0] = sum_k state[Dv=0, :] * q = 0.25*1 + 0*1 = 0.25
        //   y[Dv=1] = 0.125*1 + 0*1 = 0.125
        assert!((y_flat[0] - 0.25).abs() < 1e-5, "y[0] = {}", y_flat[0]);
        assert!((y_flat[1] - 0.125).abs() < 1e-5, "y[1] = {}", y_flat[1]);
    }

    #[test]
    fn mask_zeros_y_and_freezes_state() {
        let b = 1;
        let t = 2;
        let hk = 1;
        let dk = 2;
        let hv = 1;
        let dv = 2;
        let q = rand(&[b, t, hk, dk]);
        let k = rand(&[b, t, hk, dk]);
        let v = rand(&[b, t, hv, dv]);
        let a = rand(&[b, t, hv]);
        let bb = rand(&[b, t, hv]);
        let a_log = rand(&[hv]);
        let dt_bias = rand(&[hv]);

        let mask = Array::from_slice(&[true, false], &[b, t]);
        let (y, _) =
            gated_delta_update_ops(&q, &k, &v, &a, &bb, &a_log, &dt_bias, None, Some(&mask))
                .unwrap();
        assert_eq!(y.shape(), &[b, t, hv, dv]);
        let y_flat = flatten_f32(&y);
        // y[:, t=1] must be zero.
        let off = (hv * dv) as usize;
        for i in 0..(hv * dv) as usize {
            let v = y_flat[off + i];
            assert!(
                v.abs() < 1e-6,
                "y[t=1][{i}] = {v} should be zero under mask"
            );
        }
    }
}
