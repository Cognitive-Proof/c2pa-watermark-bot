//! The C2PA soft-binding algorithm list, vendored from
//! <https://github.com/c2pa-org/softbinding-algorithm-list>.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

const ALGORITHM_LIST: &str = include_str!("../data/softbinding-algorithm-list.json");

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Algorithm {
    pub identifier: u32,
    pub alg: String,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub decoded_media_types: Vec<String>,
    #[serde(default)]
    pub encoded_media_types: Vec<String>,
    pub entry_metadata: EntryMetadata,
    #[serde(default)]
    pub soft_binding_resolution_apis: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EntryMetadata {
    pub description: String,
    pub date_entered: String,
    pub contact: String,
    pub informational_url: String,
}

impl Algorithm {
    /// Whether the algorithm declares that it applies to still images.
    pub fn applies_to_images(&self) -> bool {
        self.decoded_media_types.iter().any(|t| t == "image")
            || self
                .encoded_media_types
                .iter()
                .any(|t| t.starts_with("image/"))
    }
}

pub struct Registry {
    algorithms: Vec<Algorithm>,
    by_alg: HashMap<String, usize>,
}

impl Registry {
    pub fn load() -> Result<Self, serde_json::Error> {
        let algorithms: Vec<Algorithm> = serde_json::from_str(ALGORITHM_LIST)?;
        let by_alg = algorithms
            .iter()
            .enumerate()
            .map(|(i, a)| (a.alg.clone(), i))
            .collect();
        Ok(Self { algorithms, by_alg })
    }

    pub fn all(&self) -> &[Algorithm] {
        &self.algorithms
    }

    pub fn get(&self, alg: &str) -> Option<&Algorithm> {
        self.by_alg.get(alg).map(|&i| &self.algorithms[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendored_list_parses() {
        let registry = Registry::load().unwrap();
        assert!(!registry.all().is_empty());
        let q = registry.get("com.adobe.trustmark.Q").unwrap();
        assert_eq!(q.kind, "watermark");
        assert!(q.applies_to_images());
    }
}
