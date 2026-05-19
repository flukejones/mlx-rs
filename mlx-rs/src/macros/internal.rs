/// See `assertEqual` in the swift binding tests
#[allow(unused_macros, reason = "test helper; only used under #[cfg(test)] in some modules")]
macro_rules! assert_array_all_close {
    ($a:tt, $b:tt) => {
        let _b: Array = $b.into();
        let assert = $a.all_close(&_b, None, None, None).unwrap();
        assert!(assert.item::<bool>());
    };
}

#[allow(unused_macros, reason = "only invoked when the safetensors feature is enabled")]
macro_rules! cfg_safetensors {
    ($($item:item)*) => {
        $(
            #[cfg(feature = "safetensors")]
            $item
        )*
    };
}
