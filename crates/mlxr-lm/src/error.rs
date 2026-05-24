use mlxr::error::Exception;

/// Crate-internal `Result` shorthand: every fallible mlxr-lm fn
/// returns this. `Exception` (mlxr ops), `io::Error`,
/// `serde_json::Error`, and `mlxr::error::IoError` all auto-convert
/// via `?` thanks to the `#[from]` arms on [`Error`].
///
/// `pub(crate)` deliberately — consumers should be explicit with the
/// error type (`Result<_, mlxr_lm::Error>`) or use `anyhow`, not
/// import a `Result` alias that would collide across crates.
#[allow(
    dead_code,
    reason = "alias awaits the workspace-wide sweep that adopts it"
)]
pub(crate) type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Exception(#[from] Exception),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Deserialize(#[from] serde_json::Error),

    #[error(transparent)]
    LoadWeights(#[from] mlxr::error::IoError),

    /// The user input carries a modality (image, audio, video) that
    /// the loaded model's [`crate::UserInputProcessor`] does not
    /// support. Includes the family that rejected the input and the
    /// modality name.
    #[error("{family}: {modality} input not supported by this model")]
    ModalityUnsupported {
        family: &'static str,
        modality: &'static str,
    },

    /// Fell out of a checked pre-condition on a tensor shape — e.g.
    /// the bypass path of `UserInput::Image::Pixels` was handed
    /// pixels whose grid disagrees with what the vision processor
    /// would have produced.
    #[error("shape mismatch: {0}")]
    Shape(String),

    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}
