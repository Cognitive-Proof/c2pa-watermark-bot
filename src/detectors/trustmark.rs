//! Adobe TrustMark (<https://github.com/adobe/trustmark>), MIT licensed.
//!
//! The Rust crate supports variants B, C, P and Q. Only C, P and Q are in the
//! algorithm list; `com.adobe.trustmark.F` has no public model.

use std::path::Path;

use image::{DynamicImage, imageops::FilterType};
use trustmark::{Trustmark, Variant, Version};

use super::{DetectError, Detection, Detector, Payload};

/// The registered TrustMark variants with public models.
pub const VARIANTS: [Variant; 3] = [Variant::Q, Variant::C, Variant::P];

pub struct TrustmarkDetector {
    alg: String,
    model: Trustmark,
}

impl TrustmarkDetector {
    pub fn alg_for(variant: Variant) -> String {
        format!("com.adobe.trustmark.{variant}")
    }

    /// Whether the encoder and decoder files for `variant` are in `models`.
    /// `Trustmark::new` loads both, even though only the decoder is used.
    pub fn models_present(models: &Path, variant: Variant) -> bool {
        ["encoder", "decoder"]
            .iter()
            .all(|kind| models.join(format!("{kind}_{variant}.onnx")).is_file())
    }

    pub fn load(models: &Path, variant: Variant) -> Result<Self, trustmark::Error> {
        // The decoder reads the error-correction version from the payload, so the
        // version given here only affects encoding.
        let model = Trustmark::new(models, variant, Version::Bch5)?;
        Ok(Self {
            alg: Self::alg_for(variant),
            model,
        })
    }

    pub fn encode(
        &self,
        bits: String,
        img: DynamicImage,
    ) -> Result<DynamicImage, trustmark::Error> {
        self.model.encode(bits, img, 0.95)
    }
}

impl TrustmarkDetector {
    fn decode(&self, img: DynamicImage) -> Result<Option<String>, DetectError> {
        match self.model.decode(img) {
            Ok(bits) => Ok(Some(bits)),
            Err(trustmark::Error::CorruptWatermark) => Ok(None),
            Err(err) => Err(DetectError(err.to_string())),
        }
    }
}

/// The BCH schema, which the decoder picks from the payload and which sets the
/// number of data bits it returns.
fn schema_for(bits: &str) -> Option<&'static str> {
    [
        Version::BchSuper,
        Version::Bch5,
        Version::Bch4,
        Version::Bch3,
    ]
    .into_iter()
    .find(|v| usize::from(v.data_bits()) == bits.len())
    .map(|v| match v {
        Version::BchSuper => "BCH_SUPER",
        Version::Bch5 => "BCH_5",
        Version::Bch4 => "BCH_4",
        Version::Bch3 => "BCH_3",
    })
}

impl Detector for TrustmarkDetector {
    fn alg(&self) -> &str {
        &self.alg
    }

    /// Decodes the image, then a copy downscaled to 90%, and reports a
    /// watermark only when both give the same payload.
    ///
    /// The decoder tries every BCH schema, and BCH_3 corrects up to 3 bit flips,
    /// so unmarked images sometimes decode to a valid-looking payload. On the
    /// Kodak test set that happened in 2 of 72 decodes, and neither repeated on
    /// the resized copy; all 72 real watermarks did. See `examples/fp_probe.rs`.
    fn detect(&self, img: &DynamicImage) -> Result<Detection, DetectError> {
        let Some(bits) = self.decode(img.clone())? else {
            return Ok(Detection::default());
        };
        let resized = img.resize(
            (img.width() * 9 / 10).max(1),
            (img.height() * 9 / 10).max(1),
            FilterType::Lanczos3,
        );
        if self.decode(resized)?.as_deref() != Some(bits.as_str()) {
            tracing::debug!(alg = %self.alg, %bits, "decode not confirmed on resized copy");
            return Ok(Detection::default());
        }
        Ok(Detection {
            payload: Some(Payload {
                schema: schema_for(&bits).map(str::to_owned),
                bits,
                hex: None,
            }),
            score: None,
        })
    }

    fn matches_value(&self, block_value: &str, payload: &Payload) -> bool {
        value_matches(block_value, &payload.bits)
    }
}

/// TrustMark's C2PA example writes block values as `<schema>*<bits>`, so the
/// part after `*` is compared. The decoder returns the full data field, which
/// may be longer than the bits that were embedded.
fn value_matches(block_value: &str, decoded_bits: &str) -> bool {
    let bits = block_value.rsplit('*').next().unwrap_or(block_value);
    !bits.is_empty() && decoded_bits.starts_with(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_schema_prefixed_values() {
        assert!(value_matches("1*0101", "010100000"));
        assert!(value_matches("0101", "0101"));
        assert!(!value_matches("1*0110", "010100000"));
        assert!(!value_matches("1*", "0101"));
    }

    #[test]
    fn schema_from_payload_length() {
        assert_eq!(schema_for(&"0".repeat(68)), Some("BCH_4"));
        assert_eq!(schema_for(&"0".repeat(56)), None);
    }
}
