//! `delegate_kv!` — collapses `impl KeyValueCache for $Enum` forwarding match arms.
//!
//! `#[macro_export]` is required for `crate::delegate_kv!` paths in
//! sibling modules outside `cache/` (e.g. `gemma4/text/loader.rs`).
//! `#[doc(hidden)]` keeps it off the rendered public API.

#[doc(hidden)]
#[macro_export]
macro_rules! delegate_kv {
    ($Enum:ident { $($Variant:ident),+ $(,)? }) => {
        fn is_quantized(&self) -> bool {
            match self { $($Enum::$Variant(c) => c.is_quantized(),)+ }
        }
        fn group_size(&self) -> Option<i32> {
            match self { $($Enum::$Variant(c) => c.group_size(),)+ }
        }
        fn bits(&self) -> Option<i32> {
            match self { $($Enum::$Variant(c) => c.bits(),)+ }
        }
        fn offset(&self) -> i32 {
            match self { $($Enum::$Variant(c) => c.offset(),)+ }
        }
        fn max_size(&self) -> Option<i32> {
            match self { $($Enum::$Variant(c) => c.max_size(),)+ }
        }
        fn update_and_fetch(
            &mut self,
            keys: ::mlxr::Array,
            values: ::mlxr::Array,
        ) -> ::std::result::Result<(::mlxr::Array, ::mlxr::Array), $crate::error::Error> {
            match self { $($Enum::$Variant(c) => c.update_and_fetch(keys, values),)+ }
        }
        fn is_trimmable(&self) -> bool {
            match self { $($Enum::$Variant(c) => c.is_trimmable(),)+ }
        }
        fn trim(&mut self, n: i32) -> i32 {
            match self { $($Enum::$Variant(c) => c.trim(n),)+ }
        }
        fn class_name(&self) -> &'static str {
            match self { $($Enum::$Variant(c) => c.class_name(),)+ }
        }
        fn state(&self) -> ::std::vec::Vec<::mlxr::Array> {
            match self { $($Enum::$Variant(c) => c.state(),)+ }
        }
        fn meta_state(&self) -> ::std::collections::HashMap<::std::string::String, ::std::string::String> {
            match self { $($Enum::$Variant(c) => c.meta_state(),)+ }
        }
        fn attention(
            &mut self,
            queries: &::mlxr::Array,
            keys: ::mlxr::Array,
            values: ::mlxr::Array,
            scale: f32,
            mask: ::std::option::Option<&::mlxr::Array>,
        ) -> ::std::result::Result<::mlxr::Array, $crate::error::Error> {
            match self {
                $($Enum::$Variant(c) => c.attention(queries, keys, values, scale, mask),)+
            }
        }
    };
}
