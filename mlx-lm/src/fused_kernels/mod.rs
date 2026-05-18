//! Hand-written Metal kernels for ops MLX doesn't fuse via `compile`
//! (chains crossing a `fast::*` boundary).

pub mod gather_qmm_combine;

pub use gather_qmm_combine::{
    cached_gather_qmm_combine_kernel, gather_qmm_combine, make_gather_qmm_combine_kernel,
    GatherQmmCombineInputs,
};
