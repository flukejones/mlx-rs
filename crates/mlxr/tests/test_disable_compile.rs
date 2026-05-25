#![allow(clippy::unwrap_used, reason = "test code")]
#![allow(clippy::missing_assert_message, reason = "test code")]
#![allow(clippy::print_stdout, reason = "test code")]
#![allow(clippy::print_stderr, reason = "test code")]

use mlxr::{
    array,
    error::Exception,
    exp, negative,
    transforms::compile::{compile, disable_compile, enable_compile},
    Array,
};

#[test]
fn test_disable_compile() {
    disable_compile();

    let f = |x: &Array| -> Result<Array, Exception> {
        let z = negative!(x)?;

        // this will crash is compile is enabled
        println!("{z:?}");

        exp!(z)
    };

    let x = array!(10.0);
    let mut compiled = compile(f, None);

    // This will panic if compilation is enabled
    let _result = compiled(&x).unwrap();

    // Re-enable compilation for other tests
    enable_compile();
}
