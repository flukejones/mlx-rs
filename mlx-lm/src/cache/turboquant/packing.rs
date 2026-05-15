//! Bit-packing for TurboQuant quantised indices and 1-bit QJL signs.
//!
//! Mirrors 0xSero `_pack_indices` / `_pack_qjl_signs` semantics:
//!
//! - `bits = 1`: 8 values per byte (`shifts = 1, 2, 4, ..., 128`).
//! - `bits = 2`: 4 values per byte (`shifts = 0, 2, 4, 6`).
//! - `bits = 3`: **rounds up** to 4-bit packing (2 values per byte, the
//!   high nibble of each byte is wasted). This is a deliberate paper
//!   choice — keeps the unpack path branch-free and matches the
//!   reference implementation. A future 3-bit-exact packing path could
//!   reclaim ~12.5% memory but is out of scope.
//! - `bits = 4`: 2 values per byte (`shifts = 0, 4`).
//! - `bits >= 5` (up to 8): no packing; stored as raw `uint8` per value.
//!
//! For QJL signs (always 1 bit) the packing is identical to `bits = 1`
//! and exposed as a separate helper that takes a `float`/`bf16` tensor
//! plus a sign convention.
//!
//! All ops here run on whatever stream the inputs live on (no explicit
//! CPU/GPU choice). Packing is per-token at quantize time and per-token
//! at unpack time during attention — these are the cache hot paths, so
//! we use mlx-rs ops rather than CPU loops.

use mlx_rs::error::Exception;
use mlx_rs::ops::{floor_divide, pad, remainder};
use mlx_rs::{Array, Dtype};

/// Effective on-disk bits used to pack a logical bit-width. 1/2/4/5/6/7/8 are
/// stored at their natural width; 3 rounds up to 4 (matches 0xSero).
pub fn effective_bits(bits: i32) -> i32 {
    match bits {
        1 => 1,
        2 => 2,
        3 | 4 => 4,
        n if (5..=8).contains(&n) => 8,
        n => panic!("unsupported bits={n}; expected 1..=8"),
    }
}

/// How many values fit per byte at `bits` (after the 3→4 round-up).
fn vals_per_byte(bits: i32) -> i32 {
    8 / effective_bits(bits)
}

/// Pack integer indices in range `[0, 2^bits)` into `uint8` bytes.
///
/// Input shape: `[..., d]`. Output shape: `[..., ceil(d / vpb)]` where
/// `vpb = 8 / effective_bits(bits)`.
///
/// For `bits >= 5` the input is just cast to `uint8` and returned (no
/// packing).
pub fn pack_indices(indices: &Array, bits: i32) -> Result<Array, Exception> {
    let eff = effective_bits(bits);
    if eff == 8 {
        return indices.as_dtype(Dtype::Uint8);
    }
    let vpb = vals_per_byte(bits);
    let d = indices.shape()[indices.ndim() - 1];

    // Pad the last axis up to a multiple of vpb (zero-pad in the index space
    // — the unpack path doesn't care what those slots decode to since they
    // sit past `d`).
    let pad_amt = (vpb - (d % vpb)) % vpb;
    let padded = if pad_amt > 0 {
        let widths: Vec<(i32, i32)> = (0..indices.ndim())
            .map(|i| if i == indices.ndim() - 1 { (0, pad_amt) } else { (0, 0) })
            .collect();
        pad(indices, &widths[..], None, None)?
    } else {
        indices.clone()
    };

    let padded_d = d + pad_amt;
    let d_packed = padded_d / vpb;

    // Reshape `[..., padded_d]` → `[..., d_packed, vpb]`.
    let mut new_shape: Vec<i32> = padded.shape().to_vec();
    let last = new_shape.len() - 1;
    new_shape[last] = d_packed;
    new_shape.push(vpb);
    let grouped = padded.reshape(&new_shape)?;

    // Build the shift weights `[2^(0*eff), 2^(1*eff), ..., 2^((vpb-1)*eff)]`
    // as a `[vpb]` `uint8` constant. The multiplication broadcasts over the
    // batch axes.
    let shifts: Vec<u8> = (0..vpb).map(|k| 1u8 << (k as u32 * eff as u32)).collect();
    let weights = Array::from_slice(&shifts, &[vpb]);

    // Cast group to uint8 (it might come in as int32 from searchsorted).
    let grouped_u8 = grouped.as_dtype(Dtype::Uint8)?;
    let scaled = grouped_u8.multiply(&weights)?;

    // Sum along the inner vpb axis → packed bytes `[..., d_packed]`.
    // `sum_axis` widens uint8 to uint32 to avoid overflow; the values fit
    // in uint8 by construction (each byte sums at most 8 bit-shifted
    // values that all together fit in 8 bits), so cast back.
    scaled.sum_axis(-1, false)?.as_dtype(Dtype::Uint8)
}

/// Inverse of [`pack_indices`]. Returns indices of shape `[..., d]`
/// truncated from the packed `[..., d_packed]` representation.
pub fn unpack_indices(packed: &Array, bits: i32, d: i32) -> Result<Array, Exception> {
    let eff = effective_bits(bits);
    if eff == 8 {
        // No-op: stored as raw uint8 in `[..., d]`.
        return packed.as_dtype(Dtype::Uint8);
    }
    let vpb = vals_per_byte(bits);
    let mask = (1u8 << eff as u32) - 1;

    // For each output byte: `(byte >> (k * eff)) & mask` for k in 0..vpb.
    // We do this with broadcasting: expand `packed` to `[..., d_packed, 1]`
    // and multiply/divide against a `[vpb]` shift table.
    let last_idx = packed.ndim() - 1;
    let packed_expanded = packed.expand_dims((last_idx + 1) as i32)?;

    let divisors: Vec<u8> = (0..vpb).map(|k| 1u8 << (k as u32 * eff as u32)).collect();
    let div_arr = Array::from_slice(&divisors, &[vpb]);
    let mask_arr = Array::from_slice(&[mask], &[1]);

    let divided = floor_divide(&packed_expanded, &div_arr)?;
    let unpacked_grouped = remainder(&divided, &mask_arr.add(Array::from_int(1))?)?;

    // Reshape `[..., d_packed, vpb]` → `[..., d_packed * vpb]` then truncate
    // to the original `d`.
    let mut new_shape: Vec<i32> = unpacked_grouped.shape().to_vec();
    let last = new_shape.len() - 2;
    new_shape[last] *= vpb;
    new_shape.pop();
    let flat = unpacked_grouped.reshape(&new_shape)?;

    if flat.shape()[flat.ndim() - 1] == d {
        Ok(flat)
    } else {
        // Slice to `[..., :d]`.
        use mlx_rs::ops::indexing::{Ellipsis, IndexOp};
        Ok(flat.index((Ellipsis, 0..d)))
    }
}

/// Pack 1-bit QJL signs into `uint8` bytes (8 signs per byte).
///
/// Input: float tensor of shape `[..., d]`. A coord is encoded as `1` when
/// it's strictly positive, `0` otherwise. Equivalent to `bits=1` packing
/// of the boolean `(x > 0)` indicator.
pub fn pack_signs(signed_values: &Array) -> Result<Array, Exception> {
    let positive = signed_values.gt(Array::from_int(0))?;
    let as_bits = positive.as_dtype(Dtype::Uint8)?;
    pack_indices(&as_bits, 1)
}

/// Unpack 1-bit QJL signs into a `±1` float tensor.
///
/// Returns `+1.0` where the bit is set, `-1.0` where unset. Output dtype
/// matches the type parameter of `Dtype::Float32` (caller can cast).
pub fn unpack_signs(packed: &Array, d: i32) -> Result<Array, Exception> {
    let bits = unpack_indices(packed, 1, d)?;
    // (bit * 2.0) - 1.0 → -1 for bit=0, +1 for bit=1.
    let as_f32 = bits.as_dtype(Dtype::Float32)?;
    as_f32.multiply(Array::from_f32(2.0))?.subtract(Array::from_f32(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::arange;

    fn array_eq(a: &Array, b: &Array) -> bool {
        let diff_max = a
            .subtract(b)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<u32>();
        diff_max == 0
    }

    /// For each bit width, generate every representable index and check
    /// pack ∘ unpack is the identity.
    #[test]
    fn pack_unpack_round_trip_all_bits() {
        for bits in [1, 2, 3, 4, 5, 6, 7, 8] {
            let max_idx = ((1u32 << effective_bits(bits) as u32) - 1) as i32;
            let d = 16;
            // Make a [d] array with values 0..max+1 cycled.
            let raw: Vec<u8> = (0..d).map(|i| (i as u32 % (max_idx as u32 + 1)) as u8).collect();
            let indices = Array::from_slice(&raw, &[d]);
            let packed = pack_indices(&indices, bits).unwrap();
            let back = unpack_indices(&packed, bits, d).unwrap();
            assert!(
                array_eq(&indices, &back),
                "round-trip failed for bits={bits}: indices={raw:?}"
            );
        }
    }

    /// Round-trip with `d` not a multiple of `vpb` exercises the pad path.
    #[test]
    fn pack_unpack_handles_non_multiple_d() {
        let raw: Vec<u8> = (0..13).map(|i| (i % 4) as u8).collect();
        let indices = Array::from_slice(&raw, &[13]);
        let packed = pack_indices(&indices, 2).unwrap();
        let back = unpack_indices(&packed, 2, 13).unwrap();
        assert!(array_eq(&indices, &back));
    }

    /// Round-trip across a batched shape `[B, H, d]`.
    #[test]
    fn pack_unpack_handles_batched_shape() {
        let b = 2;
        let h = 3;
        let d = 32;
        let raw: Vec<u8> = (0..b * h * d).map(|i| (i % 16) as u8).collect();
        let indices = Array::from_slice(&raw, &[b, h, d]);
        let packed = pack_indices(&indices, 4).unwrap();
        assert_eq!(
            packed.shape(),
            &[b, h, d / 2],
            "expected [B, H, d/2] packed shape"
        );
        let back = unpack_indices(&packed, 4, d).unwrap();
        assert!(array_eq(&indices, &back));
    }

    /// Signs round-trip: alternating ±1 input → ±1 output.
    #[test]
    fn signs_round_trip() {
        let d = 32;
        let raw: Vec<f32> = (0..d).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let input = Array::from_slice(&raw, &[d]);
        let packed = pack_signs(&input).unwrap();
        assert_eq!(packed.shape(), &[d / 8]);
        let unpacked = unpack_signs(&packed, d).unwrap();
        // input == unpacked element-wise.
        let diff = input
            .subtract(&unpacked)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap();
        assert!(diff.item::<f32>() < 1e-6);
    }

    /// Zero input → all signs are `-1` (since `(x > 0)` is strictly positive).
    #[test]
    fn signs_zero_encodes_as_negative() {
        let input = Array::zeros::<f32>(&[8]).unwrap();
        let packed = pack_signs(&input).unwrap();
        let unpacked = unpack_signs(&packed, 8).unwrap();
        let expected = Array::from_slice(&[-1.0f32; 8], &[8]);
        let diff = unpacked.subtract(&expected).unwrap().abs().unwrap().max(None).unwrap();
        assert!(diff.item::<f32>() < 1e-6);
    }

    /// arange round-trip: `[0, 1, ..., d-1]` mod 2^bits.
    #[test]
    fn arange_round_trip_at_4_bit() {
        let d = 128;
        let raw = arange::<_, i32>(0, d, None).unwrap().as_dtype(Dtype::Uint8).unwrap();
        // Take mod 16 to keep values in 4-bit range.
        let modded = remainder(
            &raw,
            Array::from_int(16).as_dtype(Dtype::Uint8).unwrap(),
        )
        .unwrap();
        let packed = pack_indices(&modded, 4).unwrap();
        assert_eq!(packed.shape(), &[d / 2]);
        let back = unpack_indices(&packed, 4, d).unwrap();
        assert!(array_eq(&modded, &back));
    }
}
