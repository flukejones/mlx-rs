//! KV-cache implementations for decoder-only models.
//!
//! Layout:
//!
//! - [`trait_def`] — the [`KeyValueCache`] trait + blanket `&mut T` impl
//! - [`kvcache`] — [`KVCache`] (default pre-allocated step-grown cache)
//!   + [`DEFAULT_KV_CACHE_STEP`]
//! - [`quantized_kvcache`] — [`QuantizedKVCache`] (affine-quant + Π
//!   rotation + packed-matmul + fused/steel kernel paths)
//! - [`rotating_kvcache`] — [`RotatingKVCache`] (sliding-window with
//!   rotation; Gemma 3/4 sliding layers)
//! - [`kernels`] — `OnceLock<MetalKernel>` accessors + steel head-dim set
//! - [`io`] — `make_prompt_cache`, `save_prompt_cache`,
//!   `load_prompt_cache`, trim helpers, `LoadedCache`
//! - [`fused_quantized_sdpa`] — fused Metal kernel for n_q=1 q-decode
//! - [`rotation`] — random orthogonal Π matrix generator for KV q-cache
//!
//! Tests for these modules live in `crates/mlxr-lm/tests/cache_basics.rs`.

#[macro_use]
mod delegate;

pub mod full_attn;
pub mod fused_quantized_sdpa;
pub mod io;
pub mod kernels;
pub mod kvcache;
pub mod options;
pub mod quantized_kvcache;
pub mod rotating_kvcache;
pub mod rotation;
pub mod trait_def;

pub use full_attn::FullAttnCache;
pub(crate) use full_attn::build_rotation;
pub use io::{
    can_trim_prompt_cache, load_prompt_cache, make_prompt_cache, save_prompt_cache,
    trim_prompt_cache, LoadedCache,
};
pub use kvcache::KVCache;
pub(crate) use kvcache::DEFAULT_KV_CACHE_STEP;
pub use options::{
    effective_prefill_chunk, effective_prefill_chunk_opt, CacheKind, CacheOptions,
    DEFAULT_PREFILL_CHUNK,
};
pub use quantized_kvcache::QuantizedKVCache;
pub use rotating_kvcache::RotatingKVCache;
pub use trait_def::KeyValueCache;
