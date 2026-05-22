#![allow(non_upper_case_globals, reason = "bindgen-generated C bindings")]
#![allow(non_camel_case_types, reason = "bindgen-generated C bindings")]
#![allow(non_snake_case, reason = "bindgen-generated C bindings")]
#![allow(clippy::all, reason = "bindgen-generated C bindings")]
#![allow(
    clippy::undocumented_unsafe_blocks,
    reason = "bindgen-generated C bindings"
)]
#![allow(clippy::pedantic, reason = "bindgen-generated C bindings")]
#![allow(unsafe_op_in_unsafe_fn, reason = "bindgen-generated C bindings")]
#![allow(unused_qualifications, reason = "bindgen-generated C bindings")]
#![allow(trivial_casts, reason = "bindgen-generated C bindings")]
#![allow(clippy::use_self, reason = "bindgen-generated C bindings")]
#![allow(
    clippy::derive_partial_eq_without_eq,
    reason = "bindgen-generated C bindings"
)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
