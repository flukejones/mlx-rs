use mlx_rs::error::Exception;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Exception(#[from] Exception),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Deserialize(#[from] serde_json::Error),

    #[error(transparent)]
    LoadWeights(#[from] mlx_rs::error::IoError),

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
