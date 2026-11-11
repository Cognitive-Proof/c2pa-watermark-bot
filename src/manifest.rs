//! Reads `c2pa.soft-binding` assertions from an embedded C2PA manifest store.

use std::io::Cursor;

use base64::prelude::{BASE64_STANDARD, Engine as _};
use c2pa::{Context, Reader};
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Serialize)]
pub struct ManifestReport {
    pub present: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_manifest: Option<String>,
    pub soft_bindings: Vec<SoftBinding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SoftBinding {
    pub manifest: String,
    pub alg: Option<String>,
    pub values: Vec<String>,
}

impl ManifestReport {
    fn absent() -> Self {
        Self {
            present: false,
            validation_state: None,
            active_manifest: None,
            soft_bindings: Vec::new(),
            error: None,
        }
    }
}

pub fn read(media_type: &str, bytes: &[u8]) -> ManifestReport {
    let reader =
        match Reader::from_context(Context::new()).with_stream(media_type, Cursor::new(bytes)) {
            Ok(reader) => reader,
            Err(c2pa::Error::JumbfNotFound | c2pa::Error::ProvenanceMissing) => {
                return ManifestReport::absent();
            }
            Err(err) => {
                return ManifestReport {
                    error: Some(err.to_string()),
                    ..ManifestReport::absent()
                };
            }
        };

    let mut soft_bindings = Vec::new();
    for (label, manifest) in reader.manifests() {
        for assertion in manifest.assertions() {
            if assertion.label() != c2pa::assertions::labels::SOFT_BINDING {
                continue;
            }
            if let Ok(value) = assertion.value() {
                soft_bindings.push(parse_soft_binding(label, value));
            }
        }
    }
    soft_bindings.sort_by(|a, b| a.manifest.cmp(&b.manifest));

    ManifestReport {
        present: true,
        validation_state: Some(format!("{:?}", reader.validation_state())),
        active_manifest: reader
            .active_manifest()
            .and_then(|m| m.label())
            .map(str::to_owned),
        soft_bindings,
        error: None,
    }
}

fn parse_soft_binding(manifest: &str, value: &Value) -> SoftBinding {
    let values = value["blocks"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|block| block_value(&block["value"]))
        .collect();
    SoftBinding {
        manifest: manifest.to_owned(),
        alg: value["alg"].as_str().map(str::to_owned),
        values,
    }
}

/// A block value as a string. The spec allows a byte string, which shows up
/// in the JSON as an array of numbers; that is returned as base64.
fn block_value(value: &Value) -> Option<String> {
    if let Some(s) = value.as_str() {
        return Some(s.to_owned());
    }
    let bytes = value
        .as_array()?
        .iter()
        .map(|b| b.as_u64().and_then(|b| u8::try_from(b).ok()))
        .collect::<Option<Vec<u8>>>()?;
    Some(BASE64_STANDARD.encode(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_soft_binding_blocks() {
        let value = serde_json::json!({
            "alg": "com.adobe.trustmark.Q",
            "blocks": [{ "scope": {}, "value": "1*0101" }],
        });
        let binding = parse_soft_binding("urn:c2pa:test", &value);
        assert_eq!(binding.alg.as_deref(), Some("com.adobe.trustmark.Q"));
        assert_eq!(binding.values, ["1*0101"]);
    }

    #[test]
    fn byte_string_values_become_base64() {
        let value = serde_json::json!({
            "alg": "me.reconize.videoseal.1",
            "blocks": [{ "scope": {}, "value": [1, 2, 255] }],
        });
        let binding = parse_soft_binding("urn:c2pa:test", &value);
        assert_eq!(binding.values, ["AQL/"]);
    }

    #[test]
    fn image_without_manifest_is_absent() {
        let mut png = Vec::new();
        image::DynamicImage::new_rgb8(8, 8)
            .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let report = read("image/png", &png);
        assert!(!report.present, "{report:?}");
        assert!(report.error.is_none(), "{report:?}");
    }
}
