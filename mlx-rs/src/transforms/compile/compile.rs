//! Compilation of functions.

use std::marker::PhantomData;

use crate::{error::Exception, Array};

use super::{next_compile_id, Closure, Compiled, CompiledState, Guarded, VectorArray};

/// Boxed adapter from the per-arity user closure to the slice-based one MLX
/// invokes internally (infallible path).
///
/// `+ Send` is required so a `Compiled<F, G>` (which holds one of these in
/// `CompiledState.f`) can move across thread boundaries — necessary for
/// downstream consumers that pass models into `tokio::task::spawn_blocking`
/// (e.g. examples/chandra).
pub type BoxedSliceFn = Box<dyn FnMut(&[Array]) -> Vec<Array> + Send + 'static>;

/// Boxed adapter from the per-arity user closure to the slice-based one MLX
/// invokes internally (fallible path).
pub type BoxedSliceTryFn =
    Box<dyn FnMut(&[Array]) -> Result<Vec<Array>, Exception> + Send + 'static>;

/// Returns a compiled function that produces the same output as `f`.
///
/// Please refer to the [swift binding
/// documentation](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/compilation)
/// for more information.
///
/// The returned closure holds the [`Compiled`] state for its lifetime so
/// the underlying `mlx_closure` from `mlx_detail_compile` is reused
/// across invocations — Python's `mx.compile` decorator semantics.
pub fn compile<F, A, O, E>(
    f: F,
    shapeless: impl Into<Option<bool>>,
) -> impl for<'a> FnMut(<F::Output as CallMut<O, E>>::Args<'a>) -> Result<O, Exception>
where
    F: Compile<A, O, E> + 'static + Copy,
    F::Output: CallMut<O, E>,
{
    let shapeless = shapeless.into().unwrap_or(false);
    let mut compiled = f.compile(shapeless);
    move |args| compiled.call_mut(args)
}

/// A trait for functions that can be compiled.
///
/// # Generic parameters
///
/// - `A`: marker for the argument-arity (e.g. `&Array`, `(&Array, &Array)`).
/// - `O`: output type.
/// - `E`: error marker (`()` for infallible, [`Exception`] otherwise).
pub trait Compile<A, O, E>: Sized {
    /// Concrete [`Compiled`] type produced by [`Self::compile`].
    type Output: CallMut<O, E>;

    /// Compiles the function. The returned value can be invoked many
    /// times via [`CallMut::call_mut`]; the underlying compiled graph
    /// is built on the first call and reused afterwards.
    fn compile(self, shapeless: bool) -> Self::Output;

    /// Like [`Self::compile`] but pins the compile-cache id to `id`
    /// instead of allocating a fresh one. Mirrors Python's
    /// `@mx.compile` decorator: one cache entry per logical function
    /// shared across every caller. Use this when the same activation
    /// runs from many module instances and you want MLX's compiler
    /// cache to reuse one compiled metal kernel.
    fn compile_with_id(self, id: usize, shapeless: bool) -> Self::Output;
}

/// A trait for a compiled function that can be called.
///
/// The argument-borrow lifetime is carried via a GAT on the method, so
/// a single long-lived [`Compiled`] (e.g. one held inside the `FnMut`
/// closure returned by [`compile`]) accepts borrows of any short
/// lifetime.
pub trait CallMut<O, E> {
    /// The input-argument type, parameterised by the borrow lifetime.
    type Args<'a>;

    /// Invokes the compiled function on `args`.
    fn call_mut<'a>(&mut self, args: Self::Args<'a>) -> Result<O, Exception>;
}

/// Shape markers used to discriminate the per-arity [`CallMut`] impls on
/// [`Compiled`]. They are zero-sized and never need to be named by callers.
pub mod shape {
    /// Marker for `&[Array]` inputs.
    #[derive(Debug, Clone, Copy)]
    pub struct ArraySlice;
    /// Marker for `&Array` inputs.
    #[derive(Debug, Clone, Copy)]
    pub struct OneArg;
    /// Marker for `(&Array, &Array)` inputs.
    #[derive(Debug, Clone, Copy)]
    pub struct TwoArgs;
    /// Marker for `(&Array, &Array, &Array)` inputs.
    #[derive(Debug, Clone, Copy)]
    pub struct ThreeArgs;
}

// ---------------------------------------------------------------------------
// Compile impls
// ---------------------------------------------------------------------------

impl<F> Compile<&[Array], Vec<Array>, ()> for F
where
    F: FnMut(&[Array]) -> Vec<Array> + 'static,
{
    type Output = Compiled<F, F, shape::ArraySlice>;

    fn compile(self, shapeless: bool) -> Self::Output {
        self.compile_with_id(next_compile_id(), shapeless)
    }

    fn compile_with_id(self, id: usize, shapeless: bool) -> Self::Output {
        Compiled {
            shape: PhantomData,
            f_marker: PhantomData,
            state: CompiledState {
                f: self,
                shapeless,
                id,
                cached_compiled: None,
            },
        }
    }
}

impl<F> Compile<&Array, Array, ()> for F
where
    F: FnMut(&Array) -> Array + Send + 'static,
{
    type Output = Compiled<F, BoxedSliceFn, shape::OneArg>;

    fn compile(self, shapeless: bool) -> Self::Output {
        self.compile_with_id(next_compile_id(), shapeless)
    }

    fn compile_with_id(mut self, id: usize, shapeless: bool) -> Self::Output {
        let f: BoxedSliceFn = Box::new(move |args: &[Array]| vec![(self)(&args[0])]);
        Compiled {
            shape: PhantomData,
            f_marker: PhantomData,
            state: CompiledState {
                f,
                shapeless,
                id,
                cached_compiled: None,
            },
        }
    }
}

impl<F> Compile<(&Array, &Array), Array, ()> for F
where
    F: FnMut((&Array, &Array)) -> Array + Send + 'static,
{
    type Output = Compiled<F, BoxedSliceFn, shape::TwoArgs>;

    fn compile(self, shapeless: bool) -> Self::Output {
        self.compile_with_id(next_compile_id(), shapeless)
    }

    fn compile_with_id(mut self, id: usize, shapeless: bool) -> Self::Output {
        let f: BoxedSliceFn = Box::new(move |args: &[Array]| vec![(self)((&args[0], &args[1]))]);
        Compiled {
            shape: PhantomData,
            f_marker: PhantomData,
            state: CompiledState {
                f,
                shapeless,
                id,
                cached_compiled: None,
            },
        }
    }
}

impl<F> Compile<(&Array, &Array, &Array), Array, ()> for F
where
    F: FnMut((&Array, &Array, &Array)) -> Array + Send + 'static,
{
    type Output = Compiled<F, BoxedSliceFn, shape::ThreeArgs>;

    fn compile(self, shapeless: bool) -> Self::Output {
        self.compile_with_id(next_compile_id(), shapeless)
    }

    fn compile_with_id(mut self, id: usize, shapeless: bool) -> Self::Output {
        let f: BoxedSliceFn =
            Box::new(move |args: &[Array]| vec![(self)((&args[0], &args[1], &args[2]))]);
        Compiled {
            shape: PhantomData,
            f_marker: PhantomData,
            state: CompiledState {
                f,
                shapeless,
                id,
                cached_compiled: None,
            },
        }
    }
}

impl<F> Compile<&[Array], Vec<Array>, Exception> for F
where
    F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + Send + 'static,
{
    type Output = Compiled<F, F, shape::ArraySlice>;

    fn compile(self, shapeless: bool) -> Self::Output {
        self.compile_with_id(next_compile_id(), shapeless)
    }

    fn compile_with_id(self, id: usize, shapeless: bool) -> Self::Output {
        Compiled {
            shape: PhantomData,
            f_marker: PhantomData,
            state: CompiledState {
                f: self,
                shapeless,
                id,
                cached_compiled: None,
            },
        }
    }
}

impl<F> Compile<&Array, Array, Exception> for F
where
    F: FnMut(&Array) -> Result<Array, Exception> + Send + 'static,
{
    type Output = Compiled<F, BoxedSliceTryFn, shape::OneArg>;

    fn compile(self, shapeless: bool) -> Self::Output {
        self.compile_with_id(next_compile_id(), shapeless)
    }

    fn compile_with_id(mut self, id: usize, shapeless: bool) -> Self::Output {
        let f: BoxedSliceTryFn = Box::new(move |args: &[Array]| Ok(vec![(self)(&args[0])?]));
        Compiled {
            shape: PhantomData,
            f_marker: PhantomData,
            state: CompiledState {
                f,
                shapeless,
                id,
                cached_compiled: None,
            },
        }
    }
}

impl<F> Compile<(&Array, &Array), Array, Exception> for F
where
    F: FnMut((&Array, &Array)) -> Result<Array, Exception> + Send + 'static,
{
    type Output = Compiled<F, BoxedSliceTryFn, shape::TwoArgs>;

    fn compile(self, shapeless: bool) -> Self::Output {
        self.compile_with_id(next_compile_id(), shapeless)
    }

    fn compile_with_id(mut self, id: usize, shapeless: bool) -> Self::Output {
        let f: BoxedSliceTryFn =
            Box::new(move |args: &[Array]| Ok(vec![(self)((&args[0], &args[1]))?]));
        Compiled {
            shape: PhantomData,
            f_marker: PhantomData,
            state: CompiledState {
                f,
                shapeless,
                id,
                cached_compiled: None,
            },
        }
    }
}

impl<F> Compile<(&Array, &Array, &Array), Array, Exception> for F
where
    F: FnMut((&Array, &Array, &Array)) -> Result<Array, Exception> + Send + 'static,
{
    type Output = Compiled<F, BoxedSliceTryFn, shape::ThreeArgs>;

    fn compile(self, shapeless: bool) -> Self::Output {
        self.compile_with_id(next_compile_id(), shapeless)
    }

    fn compile_with_id(mut self, id: usize, shapeless: bool) -> Self::Output {
        let f: BoxedSliceTryFn =
            Box::new(move |args: &[Array]| Ok(vec![(self)((&args[0], &args[1], &args[2]))?]));
        Compiled {
            shape: PhantomData,
            f_marker: PhantomData,
            state: CompiledState {
                f,
                shapeless,
                id,
                cached_compiled: None,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// CallMut impls: 4 arity shapes × 2 error modes
// ---------------------------------------------------------------------------

impl<F, G> CallMut<Vec<Array>, ()> for Compiled<F, G, shape::ArraySlice>
where
    G: FnMut(&[Array]) -> Vec<Array> + 'static,
{
    type Args<'a> = &'a [Array];

    fn call_mut<'a>(&mut self, args: Self::Args<'a>) -> Result<Vec<Array>, Exception> {
        self.state.call_mut_with(args)
    }
}

impl<F, G> CallMut<Array, ()> for Compiled<F, G, shape::OneArg>
where
    G: FnMut(&[Array]) -> Vec<Array> + 'static,
{
    type Args<'a> = &'a Array;

    fn call_mut<'a>(&mut self, args: Self::Args<'a>) -> Result<Array, Exception> {
        self.state.call_mut_with_one(std::slice::from_ref(args))
    }
}

impl<F, G> CallMut<Array, ()> for Compiled<F, G, shape::TwoArgs>
where
    G: FnMut(&[Array]) -> Vec<Array> + 'static,
{
    type Args<'a> = (&'a Array, &'a Array);

    fn call_mut<'a>(&mut self, args: Self::Args<'a>) -> Result<Array, Exception> {
        self.state.call_mut_with_one(&[args.0, args.1])
    }
}

impl<F, G> CallMut<Array, ()> for Compiled<F, G, shape::ThreeArgs>
where
    G: FnMut(&[Array]) -> Vec<Array> + 'static,
{
    type Args<'a> = (&'a Array, &'a Array, &'a Array);

    fn call_mut<'a>(&mut self, args: Self::Args<'a>) -> Result<Array, Exception> {
        self.state.call_mut_with_one(&[args.0, args.1, args.2])
    }
}

impl<F, G> CallMut<Vec<Array>, Exception> for Compiled<F, G, shape::ArraySlice>
where
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'static,
{
    type Args<'a> = &'a [Array];

    fn call_mut<'a>(&mut self, args: Self::Args<'a>) -> Result<Vec<Array>, Exception> {
        self.state.fallible_call_mut_with(args)
    }
}

impl<F, G> CallMut<Array, Exception> for Compiled<F, G, shape::OneArg>
where
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'static,
{
    type Args<'a> = &'a Array;

    fn call_mut<'a>(&mut self, args: Self::Args<'a>) -> Result<Array, Exception> {
        self.state
            .fallible_call_mut_with_one(std::slice::from_ref(args))
    }
}

impl<F, G> CallMut<Array, Exception> for Compiled<F, G, shape::TwoArgs>
where
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'static,
{
    type Args<'a> = (&'a Array, &'a Array);

    fn call_mut<'a>(&mut self, args: Self::Args<'a>) -> Result<Array, Exception> {
        self.state.fallible_call_mut_with_one(&[args.0, args.1])
    }
}

impl<F, G> CallMut<Array, Exception> for Compiled<F, G, shape::ThreeArgs>
where
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'static,
{
    type Args<'a> = (&'a Array, &'a Array, &'a Array);

    fn call_mut<'a>(&mut self, args: Self::Args<'a>) -> Result<Array, Exception> {
        self.state
            .fallible_call_mut_with_one(&[args.0, args.1, args.2])
    }
}

// ---------------------------------------------------------------------------
// CompiledState: caches the compiled mlx_closure across invocations
// ---------------------------------------------------------------------------

#[inline]
fn apply_compiled(
    compiled: &Closure<'_>,
    args: &[impl AsRef<Array>],
) -> Result<Vec<Array>, Exception> {
    let inner_inputs_vector = VectorArray::try_from_iter(args.iter())?;
    let result_vector = VectorArray::try_from_op(|res| unsafe {
        mlx_sys::mlx_closure_apply(res, compiled.as_ptr(), inner_inputs_vector.as_ptr())
    })?;
    result_vector.try_into_values()
}

/// Single-output variant of [`apply_compiled`]. Reads the C-vector
/// directly, no `Vec` allocation on the hot path.
#[inline]
fn apply_compiled_one(
    compiled: &Closure<'_>,
    args: &[impl AsRef<Array>],
) -> Result<Array, Exception> {
    let inner_inputs_vector = VectorArray::try_from_iter(args.iter())?;
    let result_vector = VectorArray::try_from_op(|res| unsafe {
        mlx_sys::mlx_closure_apply(res, compiled.as_ptr(), inner_inputs_vector.as_ptr())
    })?;
    result_vector.try_into_one()
}

#[inline]
fn build_compiled(
    inner_closure: Closure<'_>,
    fun_id: usize,
    shapeless: bool,
) -> Result<Closure<'static>, Exception> {
    Closure::try_from_op(|res| unsafe {
        let constants: &[u64] = &[];
        mlx_sys::mlx_detail_compile(
            res,
            inner_closure.as_ptr(),
            fun_id,
            shapeless,
            constants.as_ptr(),
            0,
        )
    })
}

impl<F> CompiledState<F> {
    pub(super) fn call_mut_with(
        &mut self,
        args: &[impl AsRef<Array>],
    ) -> Result<Vec<Array>, Exception>
    where
        F: FnMut(&[Array]) -> Vec<Array> + 'static,
    {
        if let Some(compiled) = self.cached_compiled.as_ref() {
            return apply_compiled(compiled, args);
        }
        let inner_closure = Closure::new(&mut self.f);
        let compiled = build_compiled(inner_closure, self.id, self.shapeless)?;
        let result = apply_compiled(&compiled, args);
        self.cached_compiled = Some(compiled);
        result
    }

    pub(super) fn call_mut_with_one(
        &mut self,
        args: &[impl AsRef<Array>],
    ) -> Result<Array, Exception>
    where
        F: FnMut(&[Array]) -> Vec<Array> + 'static,
    {
        if let Some(compiled) = self.cached_compiled.as_ref() {
            return apply_compiled_one(compiled, args);
        }
        let inner_closure = Closure::new(&mut self.f);
        let compiled = build_compiled(inner_closure, self.id, self.shapeless)?;
        let result = apply_compiled_one(&compiled, args);
        self.cached_compiled = Some(compiled);
        result
    }

    pub(super) fn fallible_call_mut_with(
        &mut self,
        args: &[impl AsRef<Array>],
    ) -> Result<Vec<Array>, Exception>
    where
        F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'static,
    {
        if let Some(compiled) = self.cached_compiled.as_ref() {
            return apply_compiled(compiled, args);
        }
        let inner_closure = Closure::new_fallible(&mut self.f);
        let compiled = build_compiled(inner_closure, self.id, self.shapeless)?;
        let result = apply_compiled(&compiled, args);
        self.cached_compiled = Some(compiled);
        result
    }

    pub(super) fn fallible_call_mut_with_one(
        &mut self,
        args: &[impl AsRef<Array>],
    ) -> Result<Array, Exception>
    where
        F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'static,
    {
        if let Some(compiled) = self.cached_compiled.as_ref() {
            return apply_compiled_one(compiled, args);
        }
        let inner_closure = Closure::new_fallible(&mut self.f);
        let compiled = build_compiled(inner_closure, self.id, self.shapeless)?;
        let result = apply_compiled_one(&compiled, args);
        self.cached_compiled = Some(compiled);
        result
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    #![allow(clippy::print_stdout, reason = "test code")]
    #![allow(clippy::print_stderr, reason = "test code")]
    use crate::{
        array,
        error::Exception,
        ops::{multiply, ones},
        Array,
    };

    use super::{compile, Compile};

    #[test]
    fn distinct_fn_pointers_get_distinct_compile_ids() {
        // Regression: prior `type_id_to_usize<T>()` derived the cache id from
        // `TypeId::of::<T>()`. Two `fn` pointers cast to the same concrete
        // signature share a TypeId — so the second compile reused the first
        // compiled graph (chandra-ocr-2 produced `sigmoid(output) * gate`
        // instead of `sigmoid(gate) * output` after swiglu warmed the slot).
        // The current `next_compile_id()` counter must hand out distinct ids
        // for every compile call regardless of source type.
        fn f0(x: &Array) -> Array {
            x.clone()
        }
        fn f1(x: &Array) -> Array {
            x.clone()
        }
        let c0 = (f0 as fn(&Array) -> Array).compile(false);
        let c1 = (f1 as fn(&Array) -> Array).compile(false);
        assert_ne!(c0.state.id, c1.state.id);
    }

    #[test]
    fn test_compile() {
        let f = |inputs: &[Array]| -> Vec<Array> { vec![&inputs[0] * &inputs[1]] };
        let mut compiled = compile(f, None);

        let i1 = ones::<f32>(&[20, 20]).unwrap();
        let i2 = ones::<f32>(&[20, 20]).unwrap();

        let args = [i1, i2];

        let r1 = f(&args).drain(0..1).next().unwrap();
        let r2 = compiled(&args).unwrap().drain(0..1).next().unwrap();

        assert_eq!(&r1, &r2);

        let r3 = compiled(&args).unwrap().drain(0..1).next().unwrap();
        assert_eq!(&r1, &r3);
    }

    #[test]
    fn test_compile_with_error() {
        let f = |inputs: &[Array]| -> Result<Vec<Array>, Exception> {
            multiply(&inputs[0], &inputs[1]).map(|x| vec![x])
        };

        let i1 = ones::<f32>(&[20, 20]).unwrap();
        let i2 = ones::<f32>(&[20, 20]).unwrap();
        let args = [i1, i2];

        let r1 = f(&args).unwrap().drain(0..1).next().unwrap();

        let mut compiled = compile(f, None);
        let r2 = compiled(&args).unwrap().drain(0..1).next().unwrap();

        assert_eq!(&r1, &r2);

        let r3 = compiled(&args).unwrap().drain(0..1).next().unwrap();
        assert_eq!(&r1, &r3);

        let a = array!([1.0, 2.0, 3.0]);
        let b = array!([4.0, 5.0]);
        let args = [a, b];

        let c = array!([4.0, 5.0, 6.0]);
        let d = array!([7.0, 8.0]);
        let another_args = [c, d];

        let result = f(&args);
        assert!(result.is_err());

        let mut compiled = compile(f, None);
        let result = compiled(&args);
        assert!(result.is_err());

        let result = compiled(&args);
        assert!(result.is_err());

        let result = compiled(&another_args);
        assert!(result.is_err());
    }

    #[test]
    fn test_compile_with_one_arg() {
        let f = |x: &Array| x * x;

        let i = ones::<f32>(&[20, 20]).unwrap();

        let r1 = f(&i);

        let mut compiled = compile(f, None);
        let r2 = compiled(&i).unwrap();

        assert_eq!(&r1, &r2);

        let r3 = compiled(&i).unwrap();
        assert_eq!(&r1, &r3);
    }

    #[test]
    fn test_compile_with_two_args() {
        let f = |(x, y): (&Array, &Array)| x * y;

        let i1 = ones::<f32>(&[20, 20]).unwrap();
        let i2 = ones::<f32>(&[20, 20]).unwrap();

        let r1 = f((&i1, &i2));

        let mut compiled = compile(f, None);
        let r2 = compiled((&i1, &i2)).unwrap();

        assert_eq!(&r1, &r2);

        let r3 = compiled((&i1, &i2)).unwrap();
        assert_eq!(&r1, &r3);
    }

    #[test]
    fn test_compile_with_three_args() {
        let f = |(x, y, z): (&Array, &Array, &Array)| x * y * z;
        let mut compiled = compile(f, None);

        let i1 = ones::<f32>(&[20, 20]).unwrap();
        let i2 = ones::<f32>(&[20, 20]).unwrap();
        let i3 = ones::<f32>(&[20, 20]).unwrap();

        let r1 = f((&i1, &i2, &i3));

        let r2 = compiled((&i1, &i2, &i3)).unwrap();

        assert_eq!(&r1, &r2);

        let r3 = compiled((&i1, &i2, &i3)).unwrap();
        assert_eq!(&r1, &r3);
    }
}
