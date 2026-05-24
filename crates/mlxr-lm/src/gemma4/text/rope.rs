//! Proportional RoPE for Gemma 4: rotate only the first
//! `partial_rotary_factor × D` dimensions; remaining freqs are `inf`
//! so `mx.fast.rope` leaves those positions untouched.

use mlxr::layers::RopeInput;
use mlxr::module::{Module, ModuleParamMut, ModuleParamRef, ModuleParameters};
use mlxr::ops::{arange, concatenate_axis, full};
use mlxr::{fast, Array};

use crate::error::Error;

const F32_INF: f32 = f32::INFINITY;

#[derive(Debug, Clone)]
pub struct ProportionalRope {
    pub dims: i32,
    pub rotated_dims: i32,
    pub traditional: bool,
    /// Precomputed freqs vector of length `dims/2`. First `rotated_dims/2`
    /// are `factor * base^exponent`; remaining are `+inf`. Not a learnable
    /// parameter — excluded from `ModuleParameters`.
    pub freqs: Array,
}

impl ModuleParameters for ProportionalRope {
    fn num_parameters(&self) -> usize {
        0
    }
    fn freeze_parameters(&mut self, _r: bool) {}
    fn unfreeze_parameters(&mut self, _r: bool) {}
    fn parameters(&self) -> ModuleParamRef<'_> {
        ModuleParamRef::default()
    }
    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        ModuleParamMut::default()
    }
    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        ModuleParamRef::default()
    }
    fn all_frozen(&self) -> Option<bool> {
        None
    }
    fn any_frozen(&self) -> Option<bool> {
        None
    }
}

impl ProportionalRope {
    pub fn new(
        dims: i32,
        rotated_dims: i32,
        traditional: bool,
        base: f32,
        factor: f32,
    ) -> Result<Self, Error> {
        assert!(rotated_dims <= dims, "rotated_dims must be ≤ dims");
        assert!(rotated_dims % 2 == 0, "rotated_dims must be even");

        let exp = arange::<_, f32>(0.0f32, rotated_dims as f32, 2.0)?
            .divide(Array::from_f32(dims as f32))?;
        let rotated_freqs = exp
            .multiply(Array::from_f32(base.ln()))?
            .exp()?
            .multiply(Array::from_f32(factor))?;
        let inf_count = (dims - rotated_dims) / 2;
        let freqs = if inf_count > 0 {
            let infs = full::<f32>(&[inf_count], Array::from_f32(F32_INF))?;
            concatenate_axis(&[rotated_freqs, infs], 0)?
        } else {
            rotated_freqs
        };
        Ok(Self {
            dims,
            rotated_dims,
            traditional,
            freqs,
        })
    }
}

impl<'a> Module<RopeInput<'a>> for ProportionalRope {
    type Output = Array;
    type Error = Error;

    fn forward(&mut self, input: RopeInput<'a>) -> Result<Array, Self::Error> {
        let RopeInput { x, offset } = input;
        Ok(fast::rope(
            x,
            self.dims,
            self.traditional,
            None,
            1.0,
            offset,
            Some(&self.freqs),
        )?)
    }

    fn training_mode(&mut self, _mode: bool) {}
}
