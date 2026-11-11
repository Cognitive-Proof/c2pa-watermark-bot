//! Meta's VideoSeal and PixelSeal (<https://github.com/facebookresearch/videoseal>),
//! MIT licensed, run from ONNX exports of the released detectors
//! (`scripts/export_videoseal_onnx.py`).
//!
//! Two registered algorithms say they use these models, but neither vendor
//! says which checkpoint they deploy or whether they changed it:
//! `me.reconize.videoseal.1` (VideoSeal) and `com.aiwatermark.pixelseal.1`
//! (PixelSeal).

use std::{path::Path, sync::Mutex};

use base64::prelude::{BASE64_STANDARD, Engine as _};
use image::{DynamicImage, imageops::FilterType};
use ndarray::Array4;
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
};

use super::{DetectError, Detection, Detector, Payload};

const INPUT_SIZE: u32 = 256;
const NBITS: usize = 256;

/// Mean absolute message logit at or above which a watermark is reported.
///
/// On 240 transformed Kodak images (crops, rotations, JPEG 30, downscales,
/// grayscale) the clean maximum was 1.4 for VideoSeal and 2.6 for PixelSeal.
/// Watermarked images scored 6.6 or more after JPEG 80, and down to 3.8 after
/// JPEG 30 and 2.8 after a 4x downscale, so heavily degraded marks are missed
/// in favour of no false positives. See `scripts/calibrate_videoseal.py`.
pub const THRESHOLD: f32 = 4.0;

/// Minimum share of matching bits for a payload to match a manifest value.
/// There is no error correction, so a few bits may be wrong; unrelated
/// 256-bit payloads agree on about half.
const MATCH_AGREEMENT: f32 = 0.9;

#[derive(Clone, Copy, Debug)]
pub enum Model {
    VideoSeal,
    PixelSeal,
}

impl Model {
    pub const ALL: [Model; 2] = [Model::VideoSeal, Model::PixelSeal];

    pub fn alg(self) -> &'static str {
        match self {
            Model::VideoSeal => "me.reconize.videoseal.1",
            Model::PixelSeal => "com.aiwatermark.pixelseal.1",
        }
    }

    pub fn filename(self) -> &'static str {
        match self {
            Model::VideoSeal => "videoseal_detector.onnx",
            Model::PixelSeal => "pixelseal_detector.onnx",
        }
    }

    fn note(self) -> &'static str {
        match self {
            Model::VideoSeal => {
                "Meta's public VideoSeal 1.0 checkpoint. Reconize says it uses VideoSeal but not \
                 which checkpoint, so its marks may not decode."
            }
            Model::PixelSeal => {
                "Meta's public PixelSeal checkpoint. AIWatermark says it uses PixelSeal but not \
                 whether it changed the model, so its marks may not decode."
            }
        }
    }
}

pub struct VideoSealDetector {
    model: Model,
    session: Mutex<Session>,
}

impl VideoSealDetector {
    pub fn load(models: &Path, model: Model) -> Result<Self, ort::Error> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(models.join(model.filename()))?;
        Ok(Self {
            model,
            session: Mutex::new(session),
        })
    }

    /// The 256 message logits for `img`.
    fn logits(&self, img: &DynamicImage) -> Result<Vec<f32>, DetectError> {
        // `Videoseal.detect` resizes with antialiased bilinear interpolation;
        // the triangle filter is the same kernel, scaled when downsampling.
        let rgb = img
            .resize_exact(INPUT_SIZE, INPUT_SIZE, FilterType::Triangle)
            .to_rgb8();
        let size = INPUT_SIZE as usize;
        let input = Array4::from_shape_fn((1, 3, size, size), |(_, c, y, x)| {
            f32::from(rgb.get_pixel(x as u32, y as u32)[c]) / 255.0
        });
        let input = Value::from_array(input).map_err(|e| DetectError(e.to_string()))?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| DetectError("ONNX session lock is poisoned".to_owned()))?;
        let outputs = session
            .run(ort::inputs!["image" => input])
            .map_err(|e| DetectError(e.to_string()))?;
        let preds = outputs["preds"]
            .try_extract_array::<f32>()
            .map_err(|e| DetectError(e.to_string()))?;
        // preds is [1, 1 + NBITS]: an untrained detection channel, then the message.
        let logits: Vec<f32> = preds.iter().skip(1).copied().collect();
        if logits.len() != NBITS {
            return Err(DetectError(format!(
                "expected {NBITS} message logits, got {}",
                logits.len()
            )));
        }
        Ok(logits)
    }
}

impl Detector for VideoSealDetector {
    fn alg(&self) -> &str {
        self.model.alg()
    }

    /// The released checkpoints have no trained detection channel and no
    /// error correction, so presence is judged by how confident the message
    /// logits are: their mean absolute value, against [`THRESHOLD`].
    fn detect(&self, img: &DynamicImage) -> Result<Detection, DetectError> {
        let logits = self.logits(img)?;
        let score = logits.iter().map(|l| l.abs()).sum::<f32>() / logits.len() as f32;
        let payload = (score >= THRESHOLD).then(|| {
            let bits: String = logits
                .iter()
                .map(|&l| if l > 0.0 { '1' } else { '0' })
                .collect();
            Payload {
                hex: Some(hex::encode(bits_to_bytes(&bits))),
                bits,
                schema: None,
            }
        });
        Ok(Detection {
            payload,
            score: Some(score),
        })
    }

    /// Accepts a value as base64 (Reconize writes the 32 payload bytes as a
    /// byte string), hex, or a bitstring, and compares bits MSB first.
    fn matches_value(&self, block_value: &str, payload: &Payload) -> bool {
        let Some(expected) = value_bits(block_value) else {
            return false;
        };
        if expected.len() != payload.bits.len() {
            return false;
        }
        let agree = expected
            .chars()
            .zip(payload.bits.chars())
            .filter(|(a, b)| a == b)
            .count();
        agree as f32 / expected.len() as f32 >= MATCH_AGREEMENT
    }

    fn note(&self) -> Option<&str> {
        Some(self.model.note())
    }
}

fn bits_to_bytes(bits: &str) -> Vec<u8> {
    bits.as_bytes()
        .chunks(8)
        .map(|chunk| chunk.iter().fold(0u8, |acc, &b| (acc << 1) | (b - b'0')))
        .collect()
}

fn bytes_to_bits(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:08b}")).collect()
}

/// A block value's payload as a 256-bit string, or `None` if it isn't one.
fn value_bits(value: &str) -> Option<String> {
    let value = value.trim();
    if value.len() == NBITS && value.chars().all(|c| c == '0' || c == '1') {
        return Some(value.to_owned());
    }
    if value.len() == NBITS / 4
        && let Ok(bytes) = hex::decode(value)
    {
        return Some(bytes_to_bits(&bytes));
    }
    match BASE64_STANDARD.decode(value) {
        Ok(bytes) if bytes.len() == NBITS / 8 => Some(bytes_to_bits(&bytes)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_and_bytes_round_trip() {
        let bytes: Vec<u8> = (0..32).map(|i| i * 7).collect();
        assert_eq!(bits_to_bytes(&bytes_to_bits(&bytes)), bytes);
    }

    #[test]
    fn parses_value_encodings() {
        let bytes = [0xA5u8; 32];
        let bits = bytes_to_bits(&bytes);
        assert_eq!(
            value_bits(&BASE64_STANDARD.encode(bytes)).as_deref(),
            Some(&*bits)
        );
        assert_eq!(value_bits(&hex::encode(bytes)).as_deref(), Some(&*bits));
        assert_eq!(value_bits(&bits).as_deref(), Some(&*bits));
        assert_eq!(value_bits("1*0101"), None);
    }
}
