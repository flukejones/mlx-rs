#![allow(clippy::unwrap_used, reason = "test code")]
#![allow(clippy::missing_assert_message, reason = "test code")]
#![allow(clippy::print_stdout, reason = "test code")]
#![allow(clippy::print_stderr, reason = "test code")]

use mlxr::{error::Exception, layers::Linear, macros::ModuleParameters, module::Module, Array};

#[derive(Debug, ModuleParameters)]
struct M {
    #[param]
    linear: Linear,
}

impl M {
    pub fn new() -> Self {
        Self {
            linear: Linear::new(5, 5).unwrap(),
        }
    }
}

impl Module<&Array> for M {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        self.linear.forward(x)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

#[test]
fn test_nested_module() {
    let mut m = M::new();
    let x = mlxr::random::uniform::<_, f32>(1.0, 2.0, &[1, 5], None).unwrap();
    let y = m.forward(&x).unwrap();
    assert_ne!(y.sum(None).unwrap(), mlxr::array!(0.0));
}
