//! Local soft-binding detectors. Each detector decodes one registered algorithm.

use image::DynamicImage;
use serde::Serialize;

pub mod trustmark;
pub mod videoseal;

/// What a detector recovered from an image.
#[derive(Debug, Clone, Serialize)]
pub struct Payload {
    /// The decoded payload as a bitstring.
    pub bits: String,
    /// The payload's error-correction schema, where the algorithm has one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<String>,
    /// The payload as hex, where it is a whole number of bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hex: Option<String>,
}

/// The outcome of looking for one algorithm's watermark.
#[derive(Debug, Default)]
pub struct Detection {
    /// The payload, when a watermark was found.
    pub payload: Option<Payload>,
    /// A detector-specific confidence score, reported whether or not a
    /// watermark was found.
    pub score: Option<f32>,
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct DetectError(pub String);

pub trait Detector: Send + Sync {
    /// The algorithm's `alg` value in the soft-binding algorithm list.
    fn alg(&self) -> &str;

    fn detect(&self, img: &DynamicImage) -> Result<Detection, DetectError>;

    /// Whether `payload` matches a `c2pa.soft-binding` block value for this algorithm.
    fn matches_value(&self, block_value: &str, payload: &Payload) -> bool;

    /// Caveats about how this detector relates to the registered algorithm.
    fn note(&self) -> Option<&str> {
        None
    }
}
