//! Microsoft's InvisMark (<https://github.com/microsoft/InvisMark>), MIT
//! licensed, run from an ONNX export of the decoder in the paper checkpoint
//! (`scripts/export_invismark_onnx.py`).
//!
//! Nothing ties the paper checkpoint to what Microsoft registered as
//! `com.microsoft.invismark.1`, so Microsoft's own marks may not decode.

use std::{path::Path, sync::Mutex};

use image::{DynamicImage, imageops::FilterType};
use ndarray::Array4;
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
};

use super::{DetectError, Detection, Detector, Payload, bits};

pub const ALG: &str = "com.microsoft.invismark.1";
pub const MODEL_FILENAME: &str = "invismark_decoder.onnx";

const INPUT_SIZE: u32 = 256;
const NBITS: usize = 100;

/// Mean |2p - 1| over the 100 bit probabilities at or above which a watermark
/// is reported.
///
/// On 216 transformed Kodak images (crops, rotations, JPEG 30, downscales,
/// grayscale) the clean maximum was 0.66. Watermarked photos scored 0.90 or
/// more as PNG, after JPEG 80, after 2x and 4x downscales and after a
/// brightness change, and 0.83 on the synthetic test fixture. JPEG 50 and
/// below, and crops, bring watermarked images into the clean range, so those
/// are missed. See `scripts/calibrate_invismark.py`.
pub const THRESHOLD: f32 = 0.8;

/// Minimum share of matching bits for a payload to match a manifest value.
const MATCH_AGREEMENT: f32 = 0.9;

const NOTE: &str = "Microsoft's InvisMark paper checkpoint (100 bits). Nothing ties it to the \
                    model Microsoft registered, so Microsoft's marks may not decode.";

pub struct InvisMarkDetector {
    session: Mutex<Session>,
}

impl InvisMarkDetector {
    pub fn load(models: &Path) -> Result<Self, ort::Error> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(models.join(MODEL_FILENAME))?;
        Ok(Self {
            session: Mutex::new(session),
        })
    }

    /// The 100 bit probabilities for `img`.
    fn probabilities(&self, img: &DynamicImage) -> Result<Vec<f32>, DetectError> {
        // The paper code resizes with torchvision's default antialiased
        // bilinear filter and scales pixels to [-1, 1].
        let rgb = img
            .resize_exact(INPUT_SIZE, INPUT_SIZE, FilterType::Triangle)
            .to_rgb8();
        let size = INPUT_SIZE as usize;
        let input = Array4::from_shape_fn((1, 3, size, size), |(_, c, y, x)| {
            f32::from(rgb.get_pixel(x as u32, y as u32)[c]) / 127.5 - 1.0
        });
        let input = Value::from_array(input).map_err(|e| DetectError(e.to_string()))?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| DetectError("ONNX session lock is poisoned".to_owned()))?;
        let outputs = session
            .run(ort::inputs!["image" => input])
            .map_err(|e| DetectError(e.to_string()))?;
        let probs: Vec<f32> = outputs["bits"]
            .try_extract_array::<f32>()
            .map_err(|e| DetectError(e.to_string()))?
            .iter()
            .copied()
            .collect();
        if probs.len() != NBITS {
            return Err(DetectError(format!(
                "expected {NBITS} bit probabilities, got {}",
                probs.len()
            )));
        }
        Ok(probs)
    }
}

impl Detector for InvisMarkDetector {
    fn alg(&self) -> &str {
        ALG
    }

    /// The checkpoint has no detection output and no error correction, so
    /// presence is judged by how far the bit probabilities are from 0.5.
    fn detect(&self, img: &DynamicImage) -> Result<Detection, DetectError> {
        let probs = self.probabilities(img)?;
        let score = probs.iter().map(|p| (2.0 * p - 1.0).abs()).sum::<f32>() / NBITS as f32;
        let payload = (score >= THRESHOLD).then(|| Payload {
            bits: probs
                .iter()
                .map(|&p| if p > 0.5 { '1' } else { '0' })
                .collect(),
            schema: None,
            hex: None,
        });
        Ok(Detection {
            payload,
            score: Some(score),
        })
    }

    /// The paper code embeds the first 100 bits of a UUID4, so values are
    /// compared on their first 100 bits; a UUID, hex, base64 or bitstring is
    /// accepted.
    fn matches_value(&self, block_value: &str, payload: &Payload) -> bool {
        bits::value_bits(block_value, NBITS)
            .is_some_and(|expected| bits::agreement(&expected, &payload.bits) >= MATCH_AGREEMENT)
    }

    fn note(&self) -> Option<&str> {
        Some(NOTE)
    }
}
