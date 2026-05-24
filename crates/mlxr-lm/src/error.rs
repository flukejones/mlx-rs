use mlxr::error::Exception;

/// Collapse an mlxr-lm [`Error`] back into the upstream
/// [`Exception`] type. Lossy by necessity — `Exception` is a flat
/// string-bearing FFI error, and the mlxr-lm variants that do not
/// originate from one (`Io`, `Deserialize`, `LoadWeights`,
/// `ModalityUnsupported`, `Shape`, `Other`) are formatted into
/// `Exception::custom`.
///
/// The boundary that requires this conversion is the mlxr
/// `Module` trait: `type Error = Exception` is fixed there, so any
/// helper that returns `Result<_, Error>` and is `?`-bubbled from
/// inside a `Module::forward` impl needs `From<Error> for
/// Exception`.
impl From<Error> for Exception {
    fn from(e: Error) -> Self {
        match e {
            Error::Exception(ex) => ex,
            other => Self::custom(other.to_string()),
        }
    }
}

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

    /// A tensor shape, rank, or axis assertion failed. Covers ndim
    /// checks, head-count mismatches, GQA divisibility, axis-length
    /// disagreements between tensors that should match.
    #[error("shape mismatch: {0}")]
    Shape(String),

    /// A required `config.json` key was missing, held the wrong
    /// type, or held a value that fails a downstream invariant
    /// (e.g. `mrope_section` length != 3). Use [`Self::config`] /
    /// [`Self::config_missing`] to build instances.
    #[error("config: {reason}")]
    Config { reason: String },

    /// A required forward-pass input was absent (e.g. neither
    /// `inputs` nor `inputs_embeds` provided).
    #[error("missing input: {0}")]
    MissingInput(&'static str),

    /// An index, axis, or token id was outside the addressable
    /// range of the tensor / vocab it was being applied to.
    #[error("index out of bounds: {0}")]
    OutOfBounds(String),

    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

impl Error {
    /// Build a [`Self::Config`] with a freeform reason.
    pub fn config(reason: impl Into<String>) -> Self {
        Self::Config {
            reason: reason.into(),
        }
    }

    /// Build a [`Self::Config`] for a missing required key.
    pub fn config_missing(key: &str) -> Self {
        Self::Config {
            reason: format!("required key {key:?} not found"),
        }
    }

    /// Build a [`Self::Shape`].
    pub fn shape(reason: impl Into<String>) -> Self {
        Self::Shape(reason.into())
    }

    /// Build a [`Self::OutOfBounds`].
    pub fn out_of_bounds(reason: impl Into<String>) -> Self {
        Self::OutOfBounds(reason.into())
    }
}
