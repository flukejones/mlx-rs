#![allow(clippy::missing_assert_message, reason = "test code")]
#![allow(clippy::print_stdout, reason = "test code")]
#![allow(clippy::print_stderr, reason = "test code")]

use mlxr::{
    error::Exception,
    layers::Linear,
    macros::{ModuleParameters, Quantizable},
    module::Module,
    quantization::MaybeQuantized,
    Array,
};

#[derive(Debug, ModuleParameters, Quantizable)]
#[allow(
    dead_code,
    reason = "derive-only compile test; struct is never constructed"
)]
struct QuantizableExample {
    #[quantizable]
    pub ql: MaybeQuantized<Linear>,
}

impl Module<&Array> for QuantizableExample {
    type Output = Array;

    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        self.ql.forward(x)
    }

    fn training_mode(&mut self, mode: bool) {
        self.ql.training_mode(mode);
    }
}
