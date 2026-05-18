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
//! Re-exports from `mod.rs` preserve the historical import path
//! `use mlx_lm::cache::{KVCache, KeyValueCache, ...}` so downstream
//! code keeps compiling after the split.

pub mod fused_quantized_sdpa;
pub mod io;
pub mod kernels;
pub mod kvcache;
pub mod quantized_kvcache;
pub mod rotating_kvcache;
pub mod rotation;
pub mod trait_def;

pub use io::{
    can_trim_prompt_cache, load_prompt_cache, make_prompt_cache, save_prompt_cache, trim_prompt_cache,
    LoadedCache,
};
// `kernels::*` is accessible via `crate::cache::kernels::...` directly.
// Removed the pub(crate) re-export — only one consumer (qwen3_5/text.rs)
// and the direct path is no less readable.
pub use kvcache::{KVCache, DEFAULT_KV_CACHE_STEP};
pub use quantized_kvcache::QuantizedKVCache;
pub use rotating_kvcache::RotatingKVCache;
pub use trait_def::KeyValueCache;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use mlx_rs::{
        ops::{concatenate_axis, indexing::{Ellipsis, IndexOp}},
        transforms::eval,
        Array, Dtype,
    };

    use super::*;

    /// Make a fresh `[B=1, H=2, S, D=4]` float32 array filled with sequential
    /// values per token-row, distinct from any other call (so we can spot
    /// out-of-order writes).
    fn token_block(s: i32, base: f32) -> Array {
        let b = 1;
        let h = 2;
        let d = 4;
        let mut data = Vec::with_capacity((b * h * s * d) as usize);
        for t in 0..s {
            for hi in 0..h {
                for di in 0..d {
                    data.push(base + (t * 1000 + hi * 100 + di) as f32);
                }
            }
        }
        // Layout above is `[t, h, d]`; reshape to `[B, H, S, D]` by writing
        // axis order then transposing in.
        let raw = Array::from_slice(&data, &[s, h, d]);
        let with_batch = raw.expand_dims(0).unwrap();
        // Swap axes 1 and 2 to get `[B, H, S, D]`.
        with_batch.swap_axes(1, 2).unwrap()
    }

    #[test]
    fn kvcache_first_update_returns_input_rows() {
        let mut cache = KVCache::new();
        let k = token_block(3, 0.0);
        let v = token_block(3, 100.0);
        let (out_k, out_v) = cache.update_and_fetch(k.clone(), v.clone()).unwrap();
        eval([&out_k, &out_v]).unwrap();
        assert_eq!(out_k.shape(), &[1, 2, 3, 4]);
        assert_eq!(out_v.shape(), &[1, 2, 3, 4]);
        assert_eq!(cache.offset(), 3);
        assert_eq!(cache.capacity(), 256);
        let diff = out_k
            .subtract(&k)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap();
        assert!(diff.item::<f32>() < 1e-6);
    }

    #[test]
    fn kvcache_appends_across_updates_in_token_order() {
        let mut cache = KVCache::new();
        let k1 = token_block(2, 0.0);
        let v1 = token_block(2, 1000.0);
        cache.update_and_fetch(k1.clone(), v1.clone()).unwrap();

        let k2 = token_block(3, 2.0);
        let v2 = token_block(3, 1002.0);
        let (out_k, out_v) = cache.update_and_fetch(k2.clone(), v2.clone()).unwrap();
        eval([&out_k, &out_v]).unwrap();

        assert_eq!(out_k.shape(), &[1, 2, 5, 4]);
        assert_eq!(cache.offset(), 5);

        let expected_k = concatenate_axis(&[k1, k2], -2).unwrap();
        let expected_v = concatenate_axis(&[v1, v2], -2).unwrap();
        let dk = out_k
            .subtract(&expected_k)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap();
        let dv = out_v
            .subtract(&expected_v)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap();
        assert!(dk.item::<f32>() < 1e-6, "K mismatch: {}", dk.item::<f32>());
        assert!(dv.item::<f32>() < 1e-6, "V mismatch: {}", dv.item::<f32>());
    }

    #[test]
    fn kvcache_grows_buffer_past_initial_step() {
        let mut cache = KVCache::with_step(4);
        cache
            .update_and_fetch(token_block(3, 0.0), token_block(3, 0.0))
            .unwrap();
        assert_eq!(cache.capacity(), 4);
        cache
            .update_and_fetch(token_block(5, 100.0), token_block(5, 100.0))
            .unwrap();
        assert_eq!(cache.offset(), 8);
        assert_eq!(cache.capacity(), 8);
    }

    #[test]
    fn kvcache_trim_drops_trailing_tokens() {
        let mut cache = KVCache::new();
        cache
            .update_and_fetch(token_block(10, 0.0), token_block(10, 0.0))
            .unwrap();
        assert!(cache.is_trimmable());
        assert_eq!(cache.trim(3), 3);
        assert_eq!(cache.offset(), 7);
        assert_eq!(cache.trim(100), 7);
        assert_eq!(cache.offset(), 0);
        assert_eq!(cache.trim(5), 0);
    }

    #[test]
    fn kvcache_dtype_matches_inputs() {
        let mut cache = KVCache::new();
        let k = token_block(2, 0.0).as_dtype(Dtype::Bfloat16).unwrap();
        let v = token_block(2, 0.0).as_dtype(Dtype::Bfloat16).unwrap();
        let (out_k, out_v) = cache.update_and_fetch(k, v).unwrap();
        assert_eq!(out_k.dtype(), Dtype::Bfloat16);
        assert_eq!(out_v.dtype(), Dtype::Bfloat16);
    }

    /// Build `[B, H, T, D]` fp16 random tensor for a prefill test.
    fn random_4d_fp16(b: i32, h: i32, t: i32, d: i32, seed: u64) -> Array {
        use mlx_rs::random::{key, normal};
        let kctx = key(seed).unwrap();
        normal::<f32>(&[b, h, t, d], None, None, &kctx)
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap()
    }

    /// Build a `[1, 1, T, T]` lower-triangular bool mask the way the
    /// standard transformer decoder does for offset=0 prefill.
    fn causal_bool_mask(t: i32) -> Array {
        let mut buf = Vec::with_capacity((t * t) as usize);
        for i in 0..t {
            for j in 0..t {
                buf.push(j <= i);
            }
        }
        Array::from_slice(&buf, &[t, t])
            .expand_dims_axes(&[0, 1])
            .unwrap()
    }

    #[test]
    fn kvcache_steel_prefill_matches_default() {
        let (b, h, t, d) = (1, 8, 64, 128);
        let q = random_4d_fp16(b, h, t, d, 1);
        let k = random_4d_fp16(b, h, t, d, 2);
        let v = random_4d_fp16(b, h, t, d, 3);
        let scale = 1.0 / (d as f32).sqrt();
        let mask = causal_bool_mask(t);

        let mut base = KVCache::new();
        let baseline = base
            .attention(&q, k.clone(), v.clone(), scale, Some(&mask))
            .unwrap();

        let mut steel = KVCache::new().with_steel_prefill();
        let routed = steel.attention(&q, k, v, scale, Some(&mask)).unwrap();

        eval([&baseline, &routed]).unwrap();
        let diff = baseline
            .subtract(&routed)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(
            diff < 5e-3,
            "steel-prefill vs fast::SDPA diverged: max_abs={diff}"
        );
    }

    #[test]
    fn kvcache_steel_prefill_falls_back_on_unsupported_head_dim() {
        let (b, h, t, d) = (1, 2, 8, 64);
        let q = random_4d_fp16(b, h, t, d, 4);
        let k = random_4d_fp16(b, h, t, d, 5);
        let v = random_4d_fp16(b, h, t, d, 6);
        let scale = 1.0 / (d as f32).sqrt();

        let mut steel = KVCache::new().with_steel_prefill();
        let out = steel.attention(&q, k, v, scale, None).unwrap();
        eval([&out]).unwrap();
        assert_eq!(out.shape(), &[b, h, t, d]);
    }

    #[test]
    fn kvcache_steel_prefill_multiturn_matches_default() {
        let (b, h, d) = (1, 8, 128);
        let scale = 1.0 / (d as f32).sqrt();
        let sys_q = random_4d_fp16(b, h, 8, d, 51);
        let sys_k = random_4d_fp16(b, h, 8, d, 52);
        let sys_v = random_4d_fp16(b, h, 8, d, 53);
        let usr_q = random_4d_fp16(b, h, 8, d, 54);
        let usr_k = random_4d_fp16(b, h, 8, d, 55);
        let usr_v = random_4d_fp16(b, h, 8, d, 56);

        let sys_mask = causal_bool_mask(8);
        let mut usr_mask_buf = Vec::with_capacity(8 * 16);
        for i in 0..8 {
            for j in 0..16 {
                usr_mask_buf.push(j <= i + 8);
            }
        }
        let usr_mask = Array::from_slice(&usr_mask_buf, &[8, 16])
            .expand_dims_axes(&[0, 1])
            .unwrap();

        let mut base = KVCache::new();
        base.attention(&sys_q, sys_k.clone(), sys_v.clone(), scale, Some(&sys_mask))
            .unwrap();
        let baseline = base
            .attention(&usr_q, usr_k.clone(), usr_v.clone(), scale, Some(&usr_mask))
            .unwrap();

        let mut steel = KVCache::new().with_steel_prefill();
        steel
            .attention(&sys_q, sys_k, sys_v, scale, Some(&sys_mask))
            .unwrap();
        let routed = steel
            .attention(&usr_q, usr_k, usr_v, scale, Some(&usr_mask))
            .unwrap();

        eval([&baseline, &routed]).unwrap();
        let diff = baseline
            .subtract(&routed)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(
            diff < 5e-3,
            "multi-turn dense steel-prefill vs fast::SDPA diverge: max_abs={diff}"
        );
    }

    #[test]
    fn quantized_kvcache_steel_prefill_matches_qmm_path() {
        let (b, h, t, d) = (1, 8, 16, 128);
        let q = random_4d_fp16(b, h, t, d, 21);
        let k = random_4d_fp16(b, h, t, d, 22);
        let v = random_4d_fp16(b, h, t, d, 23);
        let scale = 1.0 / (d as f32).sqrt();
        let mask = causal_bool_mask(t);

        let mut base = QuantizedKVCache::with_config(256, 64, 8);
        let baseline = base
            .attention(&q, k.clone(), v.clone(), scale, Some(&mask))
            .unwrap();

        let mut steel = QuantizedKVCache::with_config(256, 64, 8).with_steel_prefill();
        let routed = steel.attention(&q, k, v, scale, Some(&mask)).unwrap();

        eval([&baseline, &routed]).unwrap();
        let diff = baseline
            .subtract(&routed)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(
            diff < 5e-3,
            "quant steel-prefill vs qmm-composed diverge: max_abs={diff}"
        );
    }

    #[test]
    fn quantized_kvcache_steel_prefill_with_rotation_matches_qmm_path() {
        let (b, h, t, d) = (1, 8, 16, 128);
        let q = random_4d_fp16(b, h, t, d, 31);
        let k = random_4d_fp16(b, h, t, d, 32);
        let v = random_4d_fp16(b, h, t, d, 33);
        let scale = 1.0 / (d as f32).sqrt();
        let mask = causal_bool_mask(t);

        let mut base = QuantizedKVCache::with_config(256, 64, 4)
            .with_rotation(d, 42)
            .unwrap();
        let baseline = base
            .attention(&q, k.clone(), v.clone(), scale, Some(&mask))
            .unwrap();

        let mut steel = QuantizedKVCache::with_config(256, 64, 4)
            .with_rotation(d, 42)
            .unwrap()
            .with_steel_prefill();
        let routed = steel.attention(&q, k, v, scale, Some(&mask)).unwrap();

        eval([&baseline, &routed]).unwrap();
        let diff = baseline
            .subtract(&routed)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(
            diff < 1e-2,
            "quant steel-prefill + rotation vs qmm-composed diverge: max_abs={diff}"
        );
    }

    #[test]
    fn quantized_kvcache_steel_prefill_multiturn_matches_qmm() {
        let (b, h, d) = (1, 8, 128);
        let scale = 1.0 / (d as f32).sqrt();
        let sys_q = random_4d_fp16(b, h, 8, d, 41);
        let sys_k = random_4d_fp16(b, h, 8, d, 42);
        let sys_v = random_4d_fp16(b, h, 8, d, 43);
        let usr_q = random_4d_fp16(b, h, 8, d, 44);
        let usr_k = random_4d_fp16(b, h, 8, d, 45);
        let usr_v = random_4d_fp16(b, h, 8, d, 46);

        let sys_mask = causal_bool_mask(8);
        let mut usr_mask_buf = Vec::with_capacity(8 * 16);
        for i in 0..8 {
            for j in 0..16 {
                usr_mask_buf.push(j <= i + 8);
            }
        }
        let usr_mask = Array::from_slice(&usr_mask_buf, &[8, 16])
            .expand_dims_axes(&[0, 1])
            .unwrap();

        let mut base = QuantizedKVCache::with_config(256, 64, 8);
        base.attention(&sys_q, sys_k.clone(), sys_v.clone(), scale, Some(&sys_mask))
            .unwrap();
        let baseline = base
            .attention(&usr_q, usr_k.clone(), usr_v.clone(), scale, Some(&usr_mask))
            .unwrap();

        let mut steel = QuantizedKVCache::with_config(256, 64, 8).with_steel_prefill();
        steel
            .attention(&sys_q, sys_k, sys_v, scale, Some(&sys_mask))
            .unwrap();
        let routed = steel
            .attention(&usr_q, usr_k, usr_v, scale, Some(&usr_mask))
            .unwrap();

        eval([&baseline, &routed]).unwrap();
        let diff = baseline
            .subtract(&routed)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(
            diff < 5e-3,
            "multi-turn quant steel-prefill vs qmm-composed diverge: max_abs={diff}"
        );
    }

    /// Like `token_block` but with head_dim that's a multiple of the
    /// default quantisation group size (64).
    fn quant_token_block(s: i32, base: f32) -> Array {
        let b = 1;
        let h = 2;
        let d = 64;
        let mut data = Vec::with_capacity((b * h * s * d) as usize);
        for t in 0..s {
            for hi in 0..h {
                for di in 0..d {
                    data.push(base + (t * 100 + hi * 10) as f32 + (di as f32) * 0.01);
                }
            }
        }
        let raw = Array::from_slice(&data, &[s, h, d]);
        raw.expand_dims(0).unwrap().swap_axes(1, 2).unwrap()
    }

    #[test]
    fn quantized_kvcache_q8_round_trip_is_near_lossless() {
        let mut cache = QuantizedKVCache::with_config(256, 64, 8);
        let k = quant_token_block(3, 0.0);
        let v = quant_token_block(3, 100.0);
        let (out_k, out_v) = cache.update_and_fetch(k.clone(), v.clone()).unwrap();
        eval([&out_k, &out_v]).unwrap();
        assert_eq!(out_k.shape(), &[1, 2, 3, 64]);
        assert!(!cache.is_quantized());
        assert_eq!(cache.group_size(), Some(64));
        assert_eq!(cache.bits(), Some(8));
        let dk = out_k
            .subtract(&k)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap();
        let dv = out_v
            .subtract(&v)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap();
        assert!(dk.item::<f32>() < 1.0, "K diff {}", dk.item::<f32>());
        assert!(dv.item::<f32>() < 1.0, "V diff {}", dv.item::<f32>());
    }

    #[test]
    fn quantized_kvcache_appends_in_token_order() {
        let mut cache = QuantizedKVCache::with_config(8, 64, 8);
        let k1 = quant_token_block(2, 0.0);
        let v1 = quant_token_block(2, 1000.0);
        cache.update_and_fetch(k1.clone(), v1.clone()).unwrap();
        let k2 = quant_token_block(3, 2.0);
        let v2 = quant_token_block(3, 1002.0);
        let (out_k, _out_v) = cache.update_and_fetch(k2.clone(), v2.clone()).unwrap();
        eval([&out_k]).unwrap();
        assert_eq!(out_k.shape(), &[1, 2, 5, 64]);
        assert_eq!(cache.offset(), 5);

        let head = out_k.index((Ellipsis, 0..2, ..));
        let dk = head
            .subtract(&k1)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap();
        assert!(
            dk.item::<f32>() < 1.0,
            "first-2 mismatch: {}",
            dk.item::<f32>()
        );
    }

    #[test]
    fn quantized_kvcache_q4_round_trip_loses_some_precision_but_works() {
        let mut cache = QuantizedKVCache::with_config(256, 64, 4);
        let k = quant_token_block(3, 0.0);
        let v = quant_token_block(3, 100.0);
        let (out_k, out_v) = cache.update_and_fetch(k, v).unwrap();
        eval([&out_k, &out_v]).unwrap();
        assert_eq!(out_k.shape(), &[1, 2, 3, 64]);
        let mean = out_k.mean(None).unwrap();
        assert!(mean.item::<f32>().is_finite());
    }

    #[test]
    fn quantized_kvcache_trim_drops_tokens() {
        let mut cache = QuantizedKVCache::new();
        cache
            .update_and_fetch(quant_token_block(5, 0.0), quant_token_block(5, 0.0))
            .unwrap();
        assert!(cache.is_trimmable());
        assert_eq!(cache.trim(2), 2);
        assert_eq!(cache.offset(), 3);
    }

    #[test]
    fn make_prompt_cache_returns_per_layer() {
        let caches = make_prompt_cache(4, None);
        assert_eq!(caches.len(), 4);
        for c in &caches {
            assert_eq!(c.offset(), 0);
        }
    }

    #[test]
    fn trim_and_can_trim_helpers() {
        let mut caches = make_prompt_cache(3, None);
        assert!(!can_trim_prompt_cache(&caches[..0]));
        assert!(can_trim_prompt_cache(&caches));
        for c in caches.iter_mut() {
            c.update_and_fetch(token_block(5, 0.0), token_block(5, 0.0))
                .unwrap();
        }
        let trimmed = trim_prompt_cache(&mut caches, 2);
        assert_eq!(trimmed, 2);
        for c in &caches {
            assert_eq!(c.offset(), 3);
        }
    }

    #[test]
    fn prompt_cache_kvcache_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache.safetensors");

        let mut caches = make_prompt_cache(2, None);
        caches[0]
            .update_and_fetch(token_block(3, 0.0), token_block(3, 1000.0))
            .unwrap();
        caches[1]
            .update_and_fetch(token_block(2, 5.0), token_block(2, 2000.0))
            .unwrap();

        let mut extra = HashMap::new();
        extra.insert("prompt_hash".into(), "deadbeef".into());
        save_prompt_cache(&path, &caches, Some(&extra)).unwrap();

        let (loaded, meta) = load_prompt_cache(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(meta.get("prompt_hash"), Some(&"deadbeef".into()));

        match &loaded[0] {
            LoadedCache::Plain(c) => {
                assert_eq!(c.offset(), 3);
                assert_eq!(c.class_name(), "KVCache");
                let state = c.state();
                let diff = state[0]
                    .subtract(token_block(3, 0.0))
                    .unwrap()
                    .abs()
                    .unwrap()
                    .max(None)
                    .unwrap();
                assert!(diff.item::<f32>() < 1e-6);
            }
            _ => panic!("expected plain KVCache"),
        }
    }

    #[test]
    fn rotating_kvcache_grows_until_full_then_rotates() {
        let mut cache = RotatingKVCache::new(4, 1);
        for i in 0..4 {
            let k = token_block(1, i as f32);
            let v = token_block(1, 100.0 + i as f32);
            let (out_k, _) = cache.update_and_fetch(k, v).unwrap();
            eval([&out_k]).unwrap();
        }
        assert_eq!(cache.offset(), 4);
        assert!(cache.is_trimmable());

        let k5 = token_block(1, 99.0);
        let v5 = token_block(1, 199.0);
        let (out_k, _) = cache.update_and_fetch(k5, v5).unwrap();
        eval([&out_k]).unwrap();
        assert_eq!(cache.offset(), 5);
        assert!(!cache.is_trimmable(), "trim disabled once wrapped");
        assert_eq!(out_k.shape()[out_k.shape().len() - 2], 4);
    }

    #[test]
    fn rotating_kvcache_trim_before_wrap() {
        let mut cache = RotatingKVCache::new(8, 0);
        cache
            .update_and_fetch(token_block(5, 0.0), token_block(5, 0.0))
            .unwrap();
        assert_eq!(cache.trim(2), 2);
        assert_eq!(cache.offset(), 3);
    }

    #[test]
    fn prompt_cache_quantized_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("qcache.safetensors");

        let mut caches: Vec<QuantizedKVCache> = (0..2)
            .map(|_| QuantizedKVCache::with_config(64, 64, 8))
            .collect();
        caches[0]
            .update_and_fetch(quant_token_block(3, 0.0), quant_token_block(3, 100.0))
            .unwrap();
        caches[1]
            .update_and_fetch(quant_token_block(4, 1.0), quant_token_block(4, 200.0))
            .unwrap();

        save_prompt_cache(&path, &caches, None).unwrap();

        let (loaded, _) = load_prompt_cache(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        match &loaded[0] {
            LoadedCache::Quantized(c) => {
                assert_eq!(c.offset(), 3);
                assert_eq!(c.bits(), Some(8));
                assert_eq!(c.group_size(), Some(64));
            }
            _ => panic!("expected quantized cache"),
        }
    }
}
