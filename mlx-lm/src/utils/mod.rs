use mlx_rs::{
    arange,
    error::Exception,
    ops::{
        expand_dims,
        indexing::{IndexOp, NewAxis},
        quantized_matmul, reshape, softmax_axis,
    },
    Array, Dtype,
};

use crate::cache::KeyValueCache;

pub mod rope;
pub mod tokenizer;

/// Try-and-propagate macro for `Iterator::next` style returns.
/// On `Err`, returns `Some(Err(e.into()))` from the enclosing function.
#[macro_export]
macro_rules! tri {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => return Some(Err(e.into())),
        }
    };
}

// def quantized_scaled_dot_product_attention(
//     queries: mx.array,
//     q_keys: tuple[mx.array, mx.array, mx.array],
//     q_values: tuple[mx.array, mx.array, mx.array],
//     scale: float,
//     mask: Optional[mx.array],
//     group_size: int = 64,
//     bits: int = 8,
// ) -> mx.array:
//     B, n_q_heads, L, D = queries.shape
//     n_kv_heads = q_keys[0].shape[-3]
//     n_repeats = n_q_heads // n_kv_heads

//     queries *= scale

//     if n_repeats > 1:
//         queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
//         q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
//         q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

//     scores = mx.quantized_matmul(
//         queries, *q_keys, transpose=True, group_size=group_size, bits=bits
//     )
//     if mask is not None:
//         if isinstance(mask, str):
//             qL, kL = scores.shape[-2:]
//             q_indices = mx.arange(kL - qL, kL)
//             k_indices = mx.arange(kL)
//             mask = q_indices[:, None] >= k_indices[None]
//         if mask.dtype == mx.bool_:
//             scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
//         else:
//             scores += mask
//     scores = mx.softmax(scores, axis=-1, precise=True)
//     out = mx.quantized_matmul(
//         scores, *q_values, transpose=False, group_size=group_size, bits=bits
//     )

//     if n_repeats > 1:
//         out = mx.reshape(out, (B, n_q_heads, L, D))

//     return out

fn index_out_of_bound_exception() -> Exception {
    Exception::custom("index out of bound")
}

#[allow(
    non_snake_case,
    reason = "local bindings mirror attention tensor names (Q, K, V)"
)]
pub(crate) fn quantized_scaled_dot_product_attention(
    queries: Array,
    mut q_keys: QuantizedKeys,
    mut q_values: QuantizedValues,
    scale: f32,
    mask: Option<&Array>,
    group_size: i32,
    bits: i32,
) -> Result<Array, Exception> {
    let q_shape = queries.shape();
    let B = *q_shape.first().ok_or_else(index_out_of_bound_exception)?;
    let n_q_heads = *q_shape.get(1).ok_or_else(index_out_of_bound_exception)?;
    let L = *q_shape.get(2).ok_or_else(index_out_of_bound_exception)?;
    let D = *q_shape.get(3).ok_or_else(index_out_of_bound_exception)?;

    let q_keys_shape = q_keys.keys.shape();
    let n_kv_heads = q_keys_shape[q_keys_shape.len() - 3];
    let n_repeats = n_q_heads / n_kv_heads;

    // `queries * f32_scale` would promote bf16/f16 inputs to f32 and
    // poison every downstream op (gemm, softmax, the layer's residual
    // path) for the rest of the forward. Stage scale into the input
    // dtype so the multiply stays in-place.
    let q_dtype = queries.dtype();
    let scale_arr = Array::from_f32(scale).as_dtype(q_dtype)?;
    let mut queries = queries.multiply(&scale_arr)?;

    if n_repeats > 1 {
        queries = reshape(&queries, &[B, n_kv_heads, n_repeats, L, D])?;

        q_keys.keys = expand_dims(q_keys.keys, -3)?;
        q_keys.scales = expand_dims(q_keys.scales, -3)?;
        q_keys.biases = expand_dims(q_keys.biases, -3)?;

        q_values.values = expand_dims(q_values.values, -3)?;
        q_values.scales = expand_dims(q_values.scales, -3)?;
        q_values.biases = expand_dims(q_values.biases, -3)?;
    }

    let mut scores = quantized_matmul(
        &queries,
        q_keys.keys,
        q_keys.scales,
        &q_keys.biases,
        true,
        group_size,
        bits,
    )?;

    if let Some(mask) = mask {
        // TODO: handle str type mask

        if mask.dtype() == Dtype::Bool {
            // f64 array from `from_f64` lands on the Metal stream and is
            // rejected under mlx v0.31. Build as f32 then cast to scores'
            // dtype.
            let finfo_min = scores.dtype().finfo_min()? as f32;
            let sentinel = Array::from_f32(finfo_min).as_dtype(scores.dtype())?;
            scores = mlx_rs::ops::r#where(mask, scores, sentinel)?;
        } else {
            scores += mask;
        }
    }
    scores = softmax_axis(scores, -1, true)?;
    let mut out = quantized_matmul(
        scores,
        q_values.values,
        q_values.scales,
        &q_values.biases,
        false,
        group_size,
        bits,
    )?;

    if n_repeats > 1 {
        out = reshape(out, &[B, n_q_heads, L, D])?;
    }

    Ok(out)
}

pub struct QuantizedKeys {
    pub keys: Array,
    pub scales: Array,
    pub biases: Array,
}

pub struct QuantizedValues {
    pub values: Array,
    pub scales: Array,
    pub biases: Array,
}

pub enum MaybeQuantizedKeys {
    Original(Array),
    Quantized(QuantizedKeys),
}

impl From<Array> for MaybeQuantizedKeys {
    fn from(value: Array) -> Self {
        Self::Original(value)
    }
}

impl From<QuantizedKeys> for MaybeQuantizedKeys {
    fn from(value: QuantizedKeys) -> Self {
        Self::Quantized(value)
    }
}

pub enum MaybeQuantizedValues {
    Original(Array),
    Quantized(QuantizedValues),
}

impl From<Array> for MaybeQuantizedValues {
    fn from(value: Array) -> Self {
        Self::Original(value)
    }
}

impl From<QuantizedValues> for MaybeQuantizedValues {
    fn from(value: QuantizedValues) -> Self {
        Self::Quantized(value)
    }
}

#[allow(
    non_snake_case,
    reason = "local bindings mirror attention tensor names (N, T, S)"
)]
pub(crate) fn create_causal_mask(
    N: i32,
    offset: Option<i32>,
    window_size: Option<i32>,
    lengths: Option<Array>,
) -> Result<Array, Exception> {
    let offset = offset.unwrap_or(0);

    let rinds = arange!(stop = offset + N)?;
    let linds = arange!(start = offset, stop = offset + N)?;
    let linds = linds.index((.., NewAxis));
    let rinds = rinds.index(NewAxis);

    let mut mask = linds.ge(&rinds)?;
    if let Some(window_size) = window_size {
        mask = mask.logical_and(&linds.le(&(rinds + window_size))?)?;
    }

    if let Some(lengths) = lengths {
        let lengths = lengths.index((.., NewAxis, NewAxis, NewAxis));
        mask = mask.logical_and(&linds.lt(&lengths)?)?;
    }

    Ok(mask)
}

#[allow(
    non_snake_case,
    reason = "local bindings mirror attention tensor names (N, T, S)"
)]
pub(crate) fn create_attention_mask<C>(
    h: &Array,
    cache: &[Option<C>],
) -> Result<Option<Array>, Exception>
where
    C: KeyValueCache,
{
    let T = h.shape()[1];
    if T > 1 {
        let mut offset = 0;
        let mut window_size = None;
        if let Some(c) = cache.first().and_then(|c| c.as_ref()) {
            offset = c.offset();
            if let Some(window_size_) = c.max_size() {
                window_size = Some(window_size_);
                offset = offset.min(window_size_);
            }
        }

        create_causal_mask(T, Some(offset), window_size, None).map(Some)
    } else {
        Ok(None)
    }
}
