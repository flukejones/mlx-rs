//! Generate iterator for Gemma 4. Mirrors `gemma3::Generate`.
//!
//! Pipelined async-eval: after producing token N's lazy graph the
//! iterator immediately builds token N+1's graph and submits it via
//! `mlx_rs::transforms::async_eval`. The previous token is returned
//! to the caller; its `.item()` (or any downstream `eval`) overlaps
//! with token N+1's GPU compute, mirroring the
//! `mx.async_eval` / `mx.eval` interleave in Python `mlx_lm.generate`.
//! Runtime-agnostic — no executor, no threads.

use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use mlx_rs::transforms::async_eval;
use mlx_rs::Array;

use crate::cache::KeyValueCache;
use crate::models::gemma4::text::{Model, ModelInput};
use crate::tri;

pub use crate::sampler::sample;

pub struct Generate<'a, C> {
    model: &'a mut Model,
    cache: &'a mut Vec<Option<C>>,
    temp: f32,
    state: GenerateState<'a>,
}

pub enum GenerateState<'a> {
    /// Haven't run the prompt yet.
    Prefill { prompt_token: &'a Array },
    /// `pending` is the next token to hand out; its predecessor has
    /// already been returned to the caller. We hold `pending` here so
    /// the next `.next()` can build `pending`'s successor (submitted
    /// to the GPU) before yielding `pending`.
    Decode { pending: Array },
}

impl<'a, C> Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    pub fn new(
        model: &'a mut Model,
        cache: &'a mut Vec<Option<C>>,
        temp: f32,
        prompt_token: &'a Array,
    ) -> Self {
        Self {
            model,
            cache,
            temp,
            state: GenerateState::Prefill { prompt_token },
        }
    }

    /// Step one token: forward + sample, slice logits at last position.
    /// Caller-owned cache is mutated in-place (offset advances eagerly,
    /// graph nodes captured lazily). Returns the lazy sample tensor.
    fn step(&mut self, inputs: &Array) -> Result<Array, Exception> {
        let input = ModelInput {
            inputs,
            mask: None,
            cache: self.cache,
        };
        let logits = self.model.forward(input)?;
        sample(&logits.index((.., -1, ..)), self.temp)
    }
}

impl<C> Iterator for Generate<'_, C>
where
    C: KeyValueCache + Default,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        match std::mem::replace(
            &mut self.state,
            GenerateState::Decode {
                pending: Array::from_int(0),
            },
        ) {
            GenerateState::Prefill { prompt_token } => {
                // Build & submit prefill (y0), then immediately build &
                // submit the first decode token (y1). Yield y0 to the
                // caller while y1's compute runs on the GPU.
                let y0 = tri!(self.step(prompt_token));
                tri!(async_eval([&y0]));
                let inputs = y0.index((.., NewAxis));
                let y1 = tri!(self.step(&inputs));
                tri!(async_eval([&y1]));
                self.state = GenerateState::Decode { pending: y1 };
                Some(Ok(y0))
            }
            GenerateState::Decode { pending } => {
                // The next-after-`pending` graph is built and submitted
                // before we hand `pending` back. By the time the caller
                // resolves `pending` (via `.item()` / `eval`) the GPU is
                // already chewing on its successor.
                let inputs = pending.index((.., NewAxis));
                let next_y = tri!(self.step(&inputs));
                tri!(async_eval([&next_y]));
                self.state = GenerateState::Decode { pending: next_y };
                Some(Ok(pending))
            }
        }
    }
}
